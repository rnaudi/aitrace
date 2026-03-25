// cost.go — Model pricing lookup and per-call cost calculation.
//
// Design: Prices are embedded from prices.json at compile time. The lookup
// uses longest-prefix matching so dated model variants (e.g.
// "gpt-4o-2024-05-13") resolve to the base model entry ("gpt-4o").
//
// Why: Prefix matching (longest first) prevents "gpt-4o-mini" from matching
// "gpt-4o". Sorting model keys by length descending ensures the most specific
// entry wins.
//
// Why fallback: GitHub Copilot routes requests to both OpenAI and Anthropic
// models. The provider is "github-copilot" for all of them, so we try the
// provider's own table first, then fall back to all other tables. The cost
// formula (OpenAI vs Anthropic cache semantics) follows whichever table
// matched, not the original provider string.
package cost

import (
	_ "embed"
	"encoding/json"
	"fmt"
	"log"
	"sort"
	"strings"
)

//go:embed prices.json
var pricesJSON []byte

// modelPrice holds per-1M-token prices for a single model.
type modelPrice struct {
	Input       float64 `json:"input"`
	CachedInput float64 `json:"cached_input,omitempty"` // OpenAI: 50% of input
	CacheRead   float64 `json:"cache_read,omitempty"`   // Anthropic: 10% of input
	CacheWrite  float64 `json:"cache_write,omitempty"`  // Anthropic: 125% of input
	Output      float64 `json:"output"`
}

// providerPrices maps model name prefixes to their pricing for one provider.
type providerPrices struct {
	models map[string]modelPrice
	// sorted is model keys sorted by length descending for longest-prefix matching.
	sorted []string
}

// priceTable holds all provider pricing, loaded once from embedded JSON.
var priceTable = mustLoadPrices()

// mustLoadPrices parses the embedded prices.json into a lookup table.
// Panics on invalid JSON — this is a programmer error (broken embedded data).
func mustLoadPrices() map[string]*providerPrices {
	var raw map[string]map[string]modelPrice
	if err := json.Unmarshal(pricesJSON, &raw); err != nil {
		log.Panicf("parse embedded prices.json: %v", err)
	}

	table := make(map[string]*providerPrices, len(raw))
	for provider, models := range raw {
		pp := &providerPrices{models: models}
		pp.sorted = make([]string, 0, len(models))
		for k := range models {
			pp.sorted = append(pp.sorted, k)
		}
		// Sort by length descending so longest prefix matches first.
		sort.Slice(pp.sorted, func(i, j int) bool {
			return len(pp.sorted[i]) > len(pp.sorted[j])
		})
		table[provider] = pp
	}
	return table
}

// Calculate returns the estimated cost in USD for a single LLM call.
// Returns 0 if the provider or model is not in the pricing table.
//
// Token counts follow the CapturedCall semantics:
//   - input: total input tokens (includes cached tokens for OpenAI)
//   - output: output tokens
//   - cacheRead: tokens served from cache (OpenAI cached_tokens, Anthropic cache_read_input_tokens)
//   - cacheWrite: tokens written to cache (Anthropic cache_creation_input_tokens only)
//
// Cost formula:
//
//	OpenAI:    (input - cacheRead) * inputPrice + cacheRead * cachedInputPrice + output * outputPrice
//	Anthropic: input * inputPrice + cacheRead * cacheReadPrice + cacheWrite * cacheWritePrice + output * outputPrice
func Calculate(provider, model string, input, output, cacheRead, cacheWrite int64) float64 {
	matchedProvider, price, ok := lookupModel(provider, model)
	if !ok {
		return 0
	}

	const perMillion = 1_000_000.0

	// The cost formula follows the pricing table that matched, not the
	// caller's provider. A Claude model routed through GitHub Copilot
	// still uses Anthropic's cache semantics.
	if matchedProvider == "anthropic" {
		return anthropicCost(price, input, output, cacheRead, cacheWrite, perMillion)
	}
	return openaiCost(price, input, output, cacheRead, perMillion)
}

func openaiCost(p modelPrice, input, output, cacheRead int64, perM float64) float64 {
	// OpenAI: input tokens include cached tokens. The cached portion gets
	// a discount (typically 50%), so we subtract it from the full-price input.
	regularInput := input - cacheRead
	if regularInput < 0 {
		regularInput = 0
	}

	cost := float64(regularInput) * p.Input / perM
	cost += float64(cacheRead) * p.CachedInput / perM
	cost += float64(output) * p.Output / perM
	return cost
}

func anthropicCost(p modelPrice, input, output, cacheRead, cacheWrite int64, perM float64) float64 {
	// Anthropic: input, cache_read, and cache_write are separate token counts.
	// Cache read tokens get a 90% discount, cache write tokens cost 125% of input.
	cost := float64(input) * p.Input / perM
	cost += float64(cacheRead) * p.CacheRead / perM
	cost += float64(cacheWrite) * p.CacheWrite / perM
	cost += float64(output) * p.Output / perM
	return cost
}

// lookupModel finds the pricing entry for a model, returning which provider
// table it was found in. For multi-model routers like GitHub Copilot, it tries
// the primary table first ("openai"), then falls back to all other tables.
func lookupModel(provider, model string) (string, modelPrice, bool) {
	if provider == "" || model == "" {
		return "", modelPrice{}, false
	}

	// Resolve the primary table for known aliases.
	primary := provider
	if primary == "github-copilot" {
		primary = "openai"
	}

	// Try the primary table first.
	if pp := priceTable[primary]; pp != nil {
		if price, ok := pp.lookup(model); ok {
			return primary, price, true
		}
	}

	// Fallback: try every other table. This handles cases like Claude
	// models routed through GitHub Copilot.
	for name, pp := range priceTable {
		if name == primary {
			continue
		}
		if price, ok := pp.lookup(model); ok {
			return name, price, true
		}
	}

	return "", modelPrice{}, false
}

// lookup finds the best-matching model price using longest-prefix matching.
// Model names from API responses often include date suffixes (e.g.
// "gpt-4o-2024-05-13") that aren't in the pricing table.
func (pp *providerPrices) lookup(model string) (modelPrice, bool) {
	// Exact match first (fast path).
	if p, ok := pp.models[model]; ok {
		return p, true
	}

	// Longest-prefix match. The sorted slice is ordered by key length
	// descending, so the first match is the most specific.
	for _, key := range pp.sorted {
		if strings.HasPrefix(model, key) {
			return pp.models[key], true
		}
	}

	return modelPrice{}, false
}

// FormatUSD formats a cost value as "$X.XXX" for terminal display.
// Returns "" for zero cost (unknown model) or costs too small to display
// at 3 decimal places (e.g. $0.0004 rounds to "$0.000").
func FormatUSD(cost float64) string {
	if cost == 0 {
		return ""
	}
	s := fmt.Sprintf("$%.3f", cost)
	if s == "$0.000" {
		return ""
	}
	return s
}
