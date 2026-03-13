package capture

import "time"

// CapturedCall represents a single intercepted HTTP call.
type CapturedCall struct {
	Method     string
	Host       string
	Path       string
	StatusCode int
	Duration   time.Duration
	StartTime  time.Time // actual request start from the proxy flow
	EndTime    time.Time // actual response end from the proxy flow
	Sequence   int       // 1-based call number within the session
	IsLLM      bool      // true for known LLM API hosts

	// Parsed from request/response bodies. Only populated for LLM calls.
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
}

// EffectiveModel returns the response model, falling back to the request
// model (e.g. for error responses that don't echo the model back).
func (c CapturedCall) EffectiveModel() string {
	if c.Model != "" {
		return c.Model
	}
	return c.RequestModel
}
