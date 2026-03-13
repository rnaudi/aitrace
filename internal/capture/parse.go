// parse.go — Request/response parsing for LLM APIs (OpenAI and Anthropic).
//
// Design: Separate parser functions per provider rather than a unified parser
// with conditionals. The proxy dispatches to the right parser based on
// InferProvider(). GitHub Copilot uses the OpenAI wire format.
//
// ParseRequestModel is shared — both providers use {"model":"..."}.
package capture

import (
	"encoding/json"
	"strings"
)

// Adding a provider requires updating InferProvider(), DefaultLLMHosts,
// DefaultLLMWildcards (if subdomains) in proxy.go, and proxy_test.go
// host allowlist tests.
const (
	ProviderOpenAI        = "openai"
	ProviderGitHubCopilot = "github-copilot"
	ProviderAnthropic     = "anthropic"
)

// InferProvider returns the provider name based on the request host.
func InferProvider(host string) string {
	switch {
	case host == "api.openai.com":
		return ProviderOpenAI
	case host == "api.anthropic.com":
		return ProviderAnthropic
	case host == "api.githubcopilot.com",
		host == "copilot-proxy.githubusercontent.com",
		strings.HasSuffix(host, ".githubcopilot.com"):
		return ProviderGitHubCopilot
	default:
		return ""
	}
}

type openaiRequest struct {
	Model string `json:"model"`
}

// ParseRequestModel extracts the model from an LLM API request body.
// Both OpenAI and Anthropic use {"model":"..."} so a single parser works.
func ParseRequestModel(body []byte) string {
	var req openaiRequest
	if err := json.Unmarshal(body, &req); err != nil {
		return ""
	}
	return req.Model
}

type openaiResponse struct {
	ID      string         `json:"id"`
	Model   string         `json:"model"`
	Usage   *openaiUsage   `json:"usage"`
	Choices []openaiChoice `json:"choices"`
}

type openaiUsage struct {
	PromptTokens     int64 `json:"prompt_tokens"`
	CompletionTokens int64 `json:"completion_tokens"`
}

type openaiChoice struct {
	FinishReason string `json:"finish_reason"`
	Message      struct {
		ToolCalls []openaiToolCall `json:"tool_calls"`
	} `json:"message"`
}

type openaiToolCall struct {
	Index    int `json:"index"`
	Function struct {
		Name      string `json:"name"`
		Arguments string `json:"arguments"`
	} `json:"function"`
}

// ParsedResponse holds fields extracted from an LLM API response body.
type ParsedResponse struct {
	ID           string
	Model        string
	InputTokens  int64
	OutputTokens int64
	FinishReason string
	ToolCalls    []string // tool function names the model invoked
	ToolCallArgs []string // JSON argument strings, parallel to ToolCalls

	// rawToolCalls carries per-chunk tool call data including the index
	// field. Only populated by ParseOpenAISSEChunk for use in MergeSSEChunks
	// which needs the index to accumulate argument fragments across chunks.
	rawToolCalls []openaiToolCall
}

// ParseOpenAIResponse extracts fields from an OpenAI-format response body.
func ParseOpenAIResponse(body []byte) ParsedResponse {
	var resp openaiResponse
	if err := json.Unmarshal(body, &resp); err != nil {
		return ParsedResponse{}
	}

	var pr ParsedResponse
	pr.ID = resp.ID
	pr.Model = resp.Model

	if resp.Usage != nil {
		pr.InputTokens = resp.Usage.PromptTokens
		pr.OutputTokens = resp.Usage.CompletionTokens
	}

	if len(resp.Choices) > 0 {
		pr.FinishReason = resp.Choices[0].FinishReason
		pr.ToolCalls, pr.ToolCallArgs = extractToolCalls(resp.Choices[0].Message.ToolCalls)
	}

	return pr
}

func extractToolCalls(calls []openaiToolCall) (names []string, args []string) {
	if len(calls) == 0 {
		return nil, nil
	}
	for _, tc := range calls {
		if tc.Function.Name != "" {
			names = append(names, tc.Function.Name)
			args = append(args, tc.Function.Arguments)
		}
	}
	if len(names) == 0 {
		return nil, nil
	}
	return names, args
}

// openaiSSEChunk is the subset of an OpenAI streaming chunk we parse.
type openaiSSEChunk struct {
	ID      string            `json:"id"`
	Model   string            `json:"model"`
	Usage   *openaiUsage      `json:"usage"`
	Choices []openaiSSEChoice `json:"choices"`
}

// openaiSSEChoice uses *string for FinishReason to distinguish
// null (not finished) from "" (unknown finish reason).
type openaiSSEChoice struct {
	FinishReason *string `json:"finish_reason"`
	Delta        struct {
		ToolCalls []openaiToolCall `json:"tool_calls"`
	} `json:"delta"`
}

// ParseOpenAISSEChunk parses a single SSE data payload from an OpenAI
// streaming response. Returns a zero ParsedResponse for [DONE] or invalid JSON.
func ParseOpenAISSEChunk(data string) ParsedResponse {
	data = strings.TrimSpace(data)
	if data == "[DONE]" || data == "" {
		return ParsedResponse{}
	}

	var chunk openaiSSEChunk
	if err := json.Unmarshal([]byte(data), &chunk); err != nil {
		return ParsedResponse{}
	}

	var pr ParsedResponse
	pr.ID = chunk.ID
	pr.Model = chunk.Model

	if chunk.Usage != nil {
		pr.InputTokens = chunk.Usage.PromptTokens
		pr.OutputTokens = chunk.Usage.CompletionTokens
	}

	if len(chunk.Choices) > 0 && chunk.Choices[0].FinishReason != nil {
		pr.FinishReason = *chunk.Choices[0].FinishReason
	}

	if len(chunk.Choices) > 0 {
		tcs := chunk.Choices[0].Delta.ToolCalls
		pr.ToolCalls, pr.ToolCallArgs = extractToolCalls(tcs)
		// Preserve raw tool calls with index for MergeSSEChunks to
		// accumulate argument fragments across streaming chunks.
		if len(tcs) > 0 {
			pr.rawToolCalls = tcs
		}
	}

	return pr
}

// MergeSSEChunks combines parsed chunks from an SSE stream into a single
// ParsedResponse. Takes the last non-empty value for scalar fields and the
// last non-zero token counts (usage typically appears only on the final chunk).
//
// Tool calls are accumulated by index because SSE streams deliver names and
// argument fragments incrementally across multiple chunks.
func MergeSSEChunks(chunks []ParsedResponse) ParsedResponse {
	var merged ParsedResponse

	// Accumulate tool calls by index.
	type toolCallAcc struct {
		name string
		args string
	}
	var toolCalls map[int]*toolCallAcc
	var maxIndex int

	for _, c := range chunks {
		if c.ID != "" {
			merged.ID = c.ID
		}
		if c.Model != "" {
			merged.Model = c.Model
		}
		if c.FinishReason != "" {
			merged.FinishReason = c.FinishReason
		}
		if c.InputTokens > 0 {
			merged.InputTokens = c.InputTokens
		}
		if c.OutputTokens > 0 {
			merged.OutputTokens = c.OutputTokens
		}
		for _, tc := range c.rawToolCalls {
			if toolCalls == nil {
				toolCalls = make(map[int]*toolCallAcc)
			}
			acc, ok := toolCalls[tc.Index]
			if !ok {
				acc = &toolCallAcc{}
				toolCalls[tc.Index] = acc
			}
			if tc.Function.Name != "" {
				acc.name = tc.Function.Name
			}
			acc.args += tc.Function.Arguments
			if tc.Index > maxIndex {
				maxIndex = tc.Index
			}
		}
	}

	// Flatten accumulated tool calls in index order.
	if len(toolCalls) > 0 {
		for i := 0; i <= maxIndex; i++ {
			acc, ok := toolCalls[i]
			if !ok || acc.name == "" {
				continue
			}
			merged.ToolCalls = append(merged.ToolCalls, acc.name)
			merged.ToolCallArgs = append(merged.ToolCallArgs, acc.args)
		}
	}

	return merged
}

// --- Anthropic Messages API parsers ---

// anthropicResponse is the subset of an Anthropic Messages API response we parse.
// Anthropic uses input_tokens/output_tokens (not prompt_tokens/completion_tokens),
// stop_reason (not finish_reason in choices), and content blocks (not choices).
type anthropicResponse struct {
	ID         string             `json:"id"`
	Model      string             `json:"model"`
	StopReason *string            `json:"stop_reason"` // pointer to distinguish null from ""
	Usage      *anthropicUsage    `json:"usage"`
	Content    []anthropicContent `json:"content"`
	Type       string             `json:"type"` // "message" for responses, "error" for errors
}

type anthropicUsage struct {
	InputTokens  int64 `json:"input_tokens"`
	OutputTokens int64 `json:"output_tokens"`
}

// anthropicContent represents one block in the response content array.
// Text blocks have type "text", tool use blocks have type "tool_use".
type anthropicContent struct {
	Type  string          `json:"type"`
	Name  string          `json:"name"`            // tool_use only
	Input json.RawMessage `json:"input,omitempty"` // tool_use only
}

// ParseAnthropicResponse extracts fields from an Anthropic Messages API response body.
func ParseAnthropicResponse(body []byte) ParsedResponse {
	var resp anthropicResponse
	if err := json.Unmarshal(body, &resp); err != nil {
		return ParsedResponse{}
	}

	var pr ParsedResponse
	pr.ID = resp.ID
	pr.Model = resp.Model

	if resp.StopReason != nil {
		pr.FinishReason = *resp.StopReason
	}

	if resp.Usage != nil {
		pr.InputTokens = resp.Usage.InputTokens
		pr.OutputTokens = resp.Usage.OutputTokens
	}

	for _, block := range resp.Content {
		if block.Type == "tool_use" && block.Name != "" {
			pr.ToolCalls = append(pr.ToolCalls, block.Name)
			pr.ToolCallArgs = append(pr.ToolCallArgs, string(block.Input))
		}
	}

	return pr
}

// anthropicSSEEvent is the subset of an Anthropic SSE event we parse.
// Anthropic streaming uses distinct event types (message_start, content_block_start,
// content_block_delta, message_delta, message_stop) rather than a single data format.
type anthropicSSEEvent struct {
	Type    string          `json:"type"`
	Message json.RawMessage `json:"message,omitempty"` // message_start
	Index   int             `json:"index"`             // content_block_start, content_block_delta
	Delta   json.RawMessage `json:"delta,omitempty"`   // content_block_delta, message_delta
	Usage   *anthropicUsage `json:"usage,omitempty"`   // message_start (input), message_delta (output)

	// content_block_start carries the full content block.
	ContentBlock *anthropicContent `json:"content_block,omitempty"`
}

// anthropicDelta is the delta payload inside content_block_delta and message_delta events.
type anthropicDelta struct {
	Type        string  `json:"type"`                   // "text_delta", "input_json_delta", etc.
	Text        string  `json:"text,omitempty"`         // text_delta
	PartialJSON string  `json:"partial_json,omitempty"` // input_json_delta
	StopReason  *string `json:"stop_reason,omitempty"`  // message_delta
}

// ParseAnthropicSSEChunk parses a single SSE event from an Anthropic streaming response.
// eventType is the SSE event name (e.g. "message_start", "content_block_delta").
// data is the JSON payload. Returns a zero ParsedResponse for unrecognized events.
//
// Anthropic SSE flow:
//  1. message_start  → id, model, input_tokens
//  2. content_block_start → tool name (for tool_use blocks)
//  3. content_block_delta → input_json_delta (tool args) or text_delta
//  4. message_delta  → stop_reason, output_tokens
//  5. message_stop   → ignored
func ParseAnthropicSSEChunk(eventType, data string) ParsedResponse {
	data = strings.TrimSpace(data)
	if data == "" {
		return ParsedResponse{}
	}

	var event anthropicSSEEvent
	if err := json.Unmarshal([]byte(data), &event); err != nil {
		return ParsedResponse{}
	}

	switch eventType {
	case "message_start":
		return parseAnthropicMessageStart(event)
	case "content_block_start":
		return parseAnthropicContentBlockStart(event)
	case "content_block_delta":
		return parseAnthropicContentBlockDelta(event)
	case "message_delta":
		return parseAnthropicMessageDelta(event)
	default:
		return ParsedResponse{}
	}
}

func parseAnthropicMessageStart(event anthropicSSEEvent) ParsedResponse {
	// message_start carries the full message envelope (minus content).
	// We need to re-parse event.Message to get id, model, and usage.
	var msg anthropicResponse
	if event.Message != nil {
		_ = json.Unmarshal(event.Message, &msg)
	}

	var pr ParsedResponse
	pr.ID = msg.ID
	pr.Model = msg.Model
	if msg.Usage != nil {
		pr.InputTokens = msg.Usage.InputTokens
	}
	return pr
}

func parseAnthropicContentBlockStart(event anthropicSSEEvent) ParsedResponse {
	if event.ContentBlock == nil {
		return ParsedResponse{}
	}
	// Tool use blocks carry the tool name at content_block_start.
	if event.ContentBlock.Type == "tool_use" && event.ContentBlock.Name != "" {
		return ParsedResponse{
			ToolCalls:    []string{event.ContentBlock.Name},
			ToolCallArgs: []string{""},
			rawToolCalls: []openaiToolCall{{
				Index: event.Index,
				Function: struct {
					Name      string `json:"name"`
					Arguments string `json:"arguments"`
				}{Name: event.ContentBlock.Name},
			}},
		}
	}
	return ParsedResponse{}
}

func parseAnthropicContentBlockDelta(event anthropicSSEEvent) ParsedResponse {
	if event.Delta == nil {
		return ParsedResponse{}
	}

	var delta anthropicDelta
	if err := json.Unmarshal(event.Delta, &delta); err != nil {
		return ParsedResponse{}
	}

	// input_json_delta carries tool call argument fragments.
	if delta.Type == "input_json_delta" && delta.PartialJSON != "" {
		return ParsedResponse{
			rawToolCalls: []openaiToolCall{{
				Index: event.Index,
				Function: struct {
					Name      string `json:"name"`
					Arguments string `json:"arguments"`
				}{Arguments: delta.PartialJSON},
			}},
		}
	}

	// text_delta and other types don't carry fields we track.
	return ParsedResponse{}
}

func parseAnthropicMessageDelta(event anthropicSSEEvent) ParsedResponse {
	var pr ParsedResponse

	if event.Delta != nil {
		var delta anthropicDelta
		if err := json.Unmarshal(event.Delta, &delta); err == nil {
			if delta.StopReason != nil {
				pr.FinishReason = *delta.StopReason
			}
		}
	}

	// message_delta carries cumulative output token count.
	if event.Usage != nil {
		pr.OutputTokens = event.Usage.OutputTokens
	}

	return pr
}

// anthropicError is the subset of an Anthropic error response we parse.
// Format: {"type":"error","error":{"type":"overloaded_error","message":"Overloaded"}}
type anthropicError struct {
	Type  string `json:"type"` // "error"
	Error struct {
		Type    string `json:"type"`
		Message string `json:"message"`
	} `json:"error"`
}

// ParseAnthropicError extracts a short error description from an Anthropic
// error response. Returns the error type (e.g. "overloaded_error") when
// available, falling back to message.
func ParseAnthropicError(body []byte) string {
	var resp anthropicError
	if err := json.Unmarshal(body, &resp); err != nil {
		return ""
	}
	if resp.Error.Type != "" {
		return resp.Error.Type
	}
	return resp.Error.Message
}

// openaiError is the subset of an OpenAI error response we parse.
// Both OpenAI and Anthropic use {"error": {"message": "...", "type": "..."}}.
type openaiError struct {
	Error struct {
		Message string          `json:"message"`
		Type    string          `json:"type"`
		Code    json.RawMessage `json:"code"` // string, number, or null
	} `json:"error"`
}

// ParseOpenAIError extracts a short error description from an OpenAI-format
// error response. Returns the error type (e.g. "rate_limit_exceeded") when
// available, falling back to code, then message.
func ParseOpenAIError(body []byte) string {
	var resp openaiError
	if err := json.Unmarshal(body, &resp); err != nil {
		return ""
	}
	// Prefer type because it's a stable identifier (e.g. "insufficient_quota",
	// "rate_limit_exceeded"). Code can be a string, number, or null.
	if resp.Error.Type != "" {
		return resp.Error.Type
	}
	if len(resp.Error.Code) > 0 {
		var code string
		if json.Unmarshal(resp.Error.Code, &code) == nil && code != "" {
			return code
		}
	}
	return resp.Error.Message
}
