package capture

import "time"

// CallKind distinguishes LLM API calls from plain HTTP calls.
type CallKind string

const (
	KindLLM  CallKind = "llm"
	KindHTTP CallKind = "http"
)

// Call represents a single intercepted HTTP call.
type Call struct {
	Kind       CallKind
	Method     string
	Host       string
	Path       string
	StatusCode int
	Duration   time.Duration
	StartTime  time.Time // actual request start from the proxy flow
	EndTime    time.Time // actual response end from the proxy flow
	Sequence   int       // 1-based call number within the session

	// Parsed from request/response bodies. Only populated when Kind == KindLLM.
	Provider     string // "openai", "github-copilot", "anthropic", or ""
	RequestModel string // model from the request body
	Model        string // model from the response body (authoritative)
	ResponseID   string // e.g. "chatcmpl-abc123"
	InputTokens  int64
	OutputTokens int64
	FinishReason string
	ToolCalls    []string // tool function names the model invoked
	ToolCallArgs []string // JSON argument strings, parallel to ToolCalls
	ErrorMessage string   // error message from 4xx/5xx responses

	// Cache token fields (provider-normalized).
	// OpenAI: cached_tokens → CacheReadTokens (50% input discount, no write cost).
	// Anthropic: cache_read_input_tokens → CacheReadTokens (90% discount),
	//   cache_creation_input_tokens → CacheWriteTokens (25% surcharge).
	CacheReadTokens  int64
	CacheWriteTokens int64
}

// EffectiveModel returns the response model, falling back to the request
// model (e.g. for error responses that don't echo the model back).
func (c Call) EffectiveModel() string {
	if c.Model != "" {
		return c.Model
	}
	return c.RequestModel
}
