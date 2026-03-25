package cost

import (
	"testing"

	"github.com/stretchr/testify/assert"
	"pgregory.net/rapid"
)

// checkCalculate is the single assertion point for Calculate tests.
// When Calculate's signature changes, only this function needs updating.
func checkCalculate(t *testing.T, provider, model string, input, output, cacheRead, cacheWrite int64, wantCost float64) {
	t.Helper()
	got := Calculate(provider, model, input, output, cacheRead, cacheWrite)
	assert.InDelta(t, wantCost, got, 0.0001, "Calculate(%q, %q, %d, %d, %d, %d)", provider, model, input, output, cacheRead, cacheWrite)
}

// checkFormatUSD is the single assertion point for FormatUSD tests.
func checkFormatUSD(t *testing.T, cost float64, want string) {
	t.Helper()
	got := FormatUSD(cost)
	assert.Equal(t, want, got)
}

// --- OpenAI ---

func TestCalculateOpenAIExactModel(t *testing.T) {
	t.Parallel()
	// gpt-4o: $2.50/1M input, $10.00/1M output
	// 1000 input * 2.50/1M + 500 output * 10.00/1M = 0.0025 + 0.005 = 0.0075
	checkCalculate(t, "openai", "gpt-4o", 1000, 500, 0, 0, 0.0075)
}

func TestCalculateOpenAIPrefixMatch(t *testing.T) {
	t.Parallel()
	// "gpt-4o-2024-05-13" should match "gpt-4o"
	checkCalculate(t, "openai", "gpt-4o-2024-05-13", 1000, 500, 0, 0, 0.0075)
}

func TestCalculateOpenAIMiniNotMatchBase(t *testing.T) {
	t.Parallel()
	// "gpt-4o-mini" must NOT match "gpt-4o" — it should match "gpt-4o-mini"
	// gpt-4o-mini: $0.15/1M input, $0.60/1M output
	// 1000 * 0.15/1M + 500 * 0.60/1M = 0.00015 + 0.0003 = 0.00045
	checkCalculate(t, "openai", "gpt-4o-mini", 1000, 500, 0, 0, 0.00045)
}

func TestCalculateOpenAICachedTokens(t *testing.T) {
	t.Parallel()
	// gpt-4o: $2.50/1M input, $1.25/1M cached, $10.00/1M output
	// 1000 input total, 400 cached: (600 * 2.50 + 400 * 1.25 + 500 * 10.00) / 1M
	// = (1500 + 500 + 5000) / 1M = 0.007
	checkCalculate(t, "openai", "gpt-4o", 1000, 500, 400, 0, 0.007)
}

func TestCalculateOpenAIAllCached(t *testing.T) {
	t.Parallel()
	// All input tokens are cached.
	// gpt-4o: 0 regular + 1000 * $1.25/1M + 500 * $10.00/1M = 0.00125 + 0.005 = 0.00625
	checkCalculate(t, "openai", "gpt-4o", 1000, 500, 1000, 0, 0.00625)
}

// --- Anthropic ---

func TestCalculateAnthropicExactModel(t *testing.T) {
	t.Parallel()
	// claude-sonnet-4: $3.00/1M input, $15.00/1M output
	// 2000 * 3.00/1M + 1000 * 15.00/1M = 0.006 + 0.015 = 0.021
	checkCalculate(t, "anthropic", "claude-sonnet-4", 2000, 1000, 0, 0, 0.021)
}

func TestCalculateAnthropicPrefixMatch(t *testing.T) {
	t.Parallel()
	// "claude-sonnet-4-20250514" matches "claude-sonnet-4"
	checkCalculate(t, "anthropic", "claude-sonnet-4-20250514", 2000, 1000, 0, 0, 0.021)
}

func TestCalculateAnthropicCacheRead(t *testing.T) {
	t.Parallel()
	// claude-sonnet-4: $3.00 input, $0.30 cache_read, $15.00 output
	// 2000 * 3.00/1M + 500 * 0.30/1M + 1000 * 15.00/1M
	// = 0.006 + 0.00015 + 0.015 = 0.02115
	checkCalculate(t, "anthropic", "claude-sonnet-4", 2000, 1000, 500, 0, 0.02115)
}

func TestCalculateAnthropicCacheWrite(t *testing.T) {
	t.Parallel()
	// claude-sonnet-4: $3.00 input, $3.75 cache_write, $15.00 output
	// 2000 * 3.00/1M + 300 * 3.75/1M + 1000 * 15.00/1M
	// = 0.006 + 0.001125 + 0.015 = 0.022125
	checkCalculate(t, "anthropic", "claude-sonnet-4", 2000, 1000, 0, 300, 0.022125)
}

func TestCalculateAnthropicCacheReadAndWrite(t *testing.T) {
	t.Parallel()
	// Both cache read and write tokens present.
	// claude-sonnet-4: $3.00 input, $0.30 cache_read, $3.75 cache_write, $15.00 output
	// 2000 * 3.00/1M + 500 * 0.30/1M + 300 * 3.75/1M + 1000 * 15.00/1M
	// = 0.006 + 0.00015 + 0.001125 + 0.015 = 0.022275
	checkCalculate(t, "anthropic", "claude-sonnet-4", 2000, 1000, 500, 300, 0.022275)
}

// --- Anthropic dotted model names ---
// Anthropic API responses return dots (claude-opus-4.6) but docs use hyphens (claude-opus-4-6).
// Both forms must resolve to the same pricing.

func TestCalculateAnthropicDottedOpus(t *testing.T) {
	t.Parallel()
	// "claude-opus-4.6" (dotted, from API response) must match.
	// $5.00/1M input, $25.00/1M output
	// 2000 * 5.00/1M + 1000 * 25.00/1M = 0.010 + 0.025 = 0.035
	checkCalculate(t, "anthropic", "claude-opus-4.6", 2000, 1000, 0, 0, 0.035)
}

func TestCalculateAnthropicDottedSonnet(t *testing.T) {
	t.Parallel()
	// "claude-sonnet-4.6" matches dotted entry.
	checkCalculate(t, "anthropic", "claude-sonnet-4.6", 2000, 1000, 0, 0, 0.021)
}

func TestCalculateAnthropicDottedHaiku(t *testing.T) {
	t.Parallel()
	// "claude-haiku-4.5" matches dotted entry.
	// $1.00/1M input, $5.00/1M output
	// 2000 * 1.00/1M + 1000 * 5.00/1M = 0.002 + 0.005 = 0.007
	checkCalculate(t, "anthropic", "claude-haiku-4.5", 2000, 1000, 0, 0, 0.007)
}

func TestCalculateAnthropicDottedMatchesHyphenated(t *testing.T) {
	t.Parallel()
	// Dotted and hyphenated forms must produce identical costs.
	dotted := Calculate("anthropic", "claude-opus-4.6", 5000, 2000, 300, 100)
	hyphen := Calculate("anthropic", "claude-opus-4-6", 5000, 2000, 300, 100)
	assert.InDelta(t, hyphen, dotted, 0.0001, "dotted and hyphenated costs must match")
}

func TestCalculateAnthropicDottedWithDateSuffix(t *testing.T) {
	t.Parallel()
	// "claude-opus-4.6-20250601" should prefix-match "claude-opus-4.6".
	checkCalculate(t, "anthropic", "claude-opus-4.6-20250601", 2000, 1000, 0, 0, 0.035)
}

// --- GitHub Copilot → OpenAI ---

func TestCalculateGitHubCopilotMapsToOpenAI(t *testing.T) {
	t.Parallel()
	// GitHub Copilot uses OpenAI models, same pricing.
	checkCalculate(t, "github-copilot", "gpt-4o", 1000, 500, 0, 0, 0.0075)
}

func TestCalculateGitHubCopilotPrefixMatch(t *testing.T) {
	t.Parallel()
	checkCalculate(t, "github-copilot", "gpt-4o-2024-05-13", 1000, 500, 0, 0, 0.0075)
}

func TestCalculateGitHubCopilotGPT5Mini(t *testing.T) {
	t.Parallel()
	// gpt-5-mini is a Copilot-specific alias. Same prices as gpt-5.4-mini.
	// $0.75/1M input, $4.50/1M output
	// 1000 * 0.75/1M + 500 * 4.50/1M = 0.00075 + 0.00225 = 0.003
	checkCalculate(t, "github-copilot", "gpt-5-mini", 1000, 500, 0, 0, 0.003)
}

func TestCalculateGitHubCopilotGPT5(t *testing.T) {
	t.Parallel()
	// gpt-5 alias. Same prices as gpt-5.4.
	// $2.50/1M input, $15.00/1M output
	// 1000 * 2.50/1M + 500 * 15.00/1M = 0.0025 + 0.0075 = 0.01
	checkCalculate(t, "github-copilot", "gpt-5", 1000, 500, 0, 0, 0.01)
}

// --- GitHub Copilot → Anthropic (fallback) ---
// Copilot routes Claude models but provider is "github-copilot". The cost
// lookup must fall back to the Anthropic pricing table.

func TestCalculateGitHubCopilotClaudeModel(t *testing.T) {
	t.Parallel()
	// "github-copilot" + "claude-opus-4.6" → falls back to Anthropic table.
	// claude-opus-4.6: $5.00/1M input, $25.00/1M output
	// 2000 * 5.00/1M + 1000 * 25.00/1M = 0.010 + 0.025 = 0.035
	checkCalculate(t, "github-copilot", "claude-opus-4.6", 2000, 1000, 0, 0, 0.035)
}

func TestCalculateGitHubCopilotClaudeWithCache(t *testing.T) {
	t.Parallel()
	// Copilot + Claude with cache tokens must use Anthropic cache formula.
	// claude-sonnet-4: $3.00 input, $0.30 cache_read, $3.75 cache_write, $15.00 output
	// 2000 * 3.00/1M + 500 * 0.30/1M + 300 * 3.75/1M + 1000 * 15.00/1M
	// = 0.006 + 0.00015 + 0.001125 + 0.015 = 0.022275
	checkCalculate(t, "github-copilot", "claude-sonnet-4", 2000, 1000, 500, 300, 0.022275)
}

func TestCalculateGitHubCopilotClaudeSonnetDated(t *testing.T) {
	t.Parallel()
	// Prefix match through the fallback path.
	// "claude-sonnet-4-20250514" should match "claude-sonnet-4" in the Anthropic table.
	checkCalculate(t, "github-copilot", "claude-sonnet-4-20250514", 2000, 1000, 0, 0, 0.021)
}

// --- Edge cases ---

func TestCalculateUnknownProvider(t *testing.T) {
	t.Parallel()
	checkCalculate(t, "google", "gemini-pro", 1000, 500, 0, 0, 0)
}

func TestCalculateUnknownModel(t *testing.T) {
	t.Parallel()
	checkCalculate(t, "openai", "gpt-99-turbo", 1000, 500, 0, 0, 0)
}

func TestCalculateEmptyProvider(t *testing.T) {
	t.Parallel()
	checkCalculate(t, "", "gpt-4o", 1000, 500, 0, 0, 0)
}

func TestCalculateEmptyModel(t *testing.T) {
	t.Parallel()
	checkCalculate(t, "openai", "", 1000, 500, 0, 0, 0)
}

func TestCalculateZeroTokens(t *testing.T) {
	t.Parallel()
	checkCalculate(t, "openai", "gpt-4o", 0, 0, 0, 0, 0)
}

// --- FormatUSD ---

func TestFormatUSDNormal(t *testing.T) {
	t.Parallel()
	checkFormatUSD(t, 0.0075, "$0.007")
}

func TestFormatUSDZero(t *testing.T) {
	t.Parallel()
	checkFormatUSD(t, 0, "")
}

func TestFormatUSDLargeCost(t *testing.T) {
	t.Parallel()
	checkFormatUSD(t, 1.23456, "$1.235")
}

func TestFormatUSDSmallCost(t *testing.T) {
	t.Parallel()
	checkFormatUSD(t, 0.001, "$0.001")
}

func TestFormatUSDSubMillSuppressed(t *testing.T) {
	t.Parallel()
	// Costs that round to "$0.000" at 3 decimal places should be suppressed.
	checkFormatUSD(t, 0.0004, "")
}

func TestFormatUSDJustAboveThreshold(t *testing.T) {
	t.Parallel()
	// $0.0005 rounds to "$0.001" — should be displayed.
	checkFormatUSD(t, 0.0005, "$0.001")
}

// --- Property tests ---

func TestCalculateNonNegativeProperty(t *testing.T) {
	t.Parallel()
	providers := []string{"openai", "anthropic", "github-copilot"}
	rapid.Check(t, func(t *rapid.T) {
		provider := rapid.SampledFrom(providers).Draw(t, "provider")
		model := rapid.SampledFrom([]string{
			"gpt-4o", "gpt-4o-mini", "gpt-5.4", "gpt-5", "gpt-5-mini",
			"claude-sonnet-4", "claude-opus-4-6", "claude-opus-4.6",
			"claude-haiku-4.5", "claude-3-haiku",
		}).Draw(t, "model")
		input := rapid.Int64Range(0, 1_000_000).Draw(t, "input")
		output := rapid.Int64Range(0, 1_000_000).Draw(t, "output")
		cacheRead := rapid.Int64Range(0, input).Draw(t, "cacheRead")
		cacheWrite := rapid.Int64Range(0, 100_000).Draw(t, "cacheWrite")

		cost := Calculate(provider, model, input, output, cacheRead, cacheWrite)
		if cost < 0 {
			t.Fatalf("negative cost: %f for %s/%s", cost, provider, model)
		}
	})
}

func TestCalculateUnknownModelZeroProperty(t *testing.T) {
	t.Parallel()
	rapid.Check(t, func(t *rapid.T) {
		// Random gibberish model names should never match pricing entries.
		model := rapid.StringMatching(`zzz-[a-z]{5}-[0-9]{4}`).Draw(t, "model")
		input := rapid.Int64Range(0, 1_000_000).Draw(t, "input")
		output := rapid.Int64Range(0, 1_000_000).Draw(t, "output")

		cost := Calculate("openai", model, input, output, 0, 0)
		if cost != 0 {
			t.Fatalf("expected 0 for unknown model %q, got %f", model, cost)
		}
	})
}

func TestCalculateMonotonicInTokensProperty(t *testing.T) {
	t.Parallel()
	rapid.Check(t, func(t *rapid.T) {
		input1 := rapid.Int64Range(0, 500_000).Draw(t, "input1")
		input2 := rapid.Int64Range(input1, 1_000_000).Draw(t, "input2")
		output := rapid.Int64Range(0, 100_000).Draw(t, "output")

		cost1 := Calculate("openai", "gpt-4o", input1, output, 0, 0)
		cost2 := Calculate("openai", "gpt-4o", input2, output, 0, 0)
		if cost2 < cost1 {
			t.Fatalf("cost not monotonic: input %d→%d, cost %f→%f", input1, input2, cost1, cost2)
		}
	})
}
