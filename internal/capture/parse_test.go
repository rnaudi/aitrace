package capture

import (
	"encoding/json"
	"testing"

	"github.com/stretchr/testify/assert"
	"pgregory.net/rapid"
)

func checkInferProvider(t *testing.T, host string, want string) {
	t.Helper()
	assert.Equal(t, want, InferProvider(host))
}

func TestInferProviderOpenAI(t *testing.T) {
	t.Parallel()
	checkInferProvider(t, "api.openai.com", ProviderOpenAI)
}

func TestInferProviderAnthropic(t *testing.T) {
	t.Parallel()
	checkInferProvider(t, "api.anthropic.com", ProviderAnthropic)
}

func TestInferProviderGitHubCopilot(t *testing.T) {
	t.Parallel()
	checkInferProvider(t, "api.githubcopilot.com", ProviderGitHubCopilot)
	checkInferProvider(t, "copilot-proxy.githubusercontent.com", ProviderGitHubCopilot)
}

func TestInferProviderGitHubCopilotWildcard(t *testing.T) {
	t.Parallel()
	checkInferProvider(t, "models.individual.githubcopilot.com", ProviderGitHubCopilot)
	checkInferProvider(t, "anything.githubcopilot.com", ProviderGitHubCopilot)
}

func TestInferProviderUnknown(t *testing.T) {
	t.Parallel()
	checkInferProvider(t, "example.com", "")
	checkInferProvider(t, "", "")
}

func checkParseRequestModel(t *testing.T, body []byte, want string) {
	t.Helper()
	assert.Equal(t, want, ParseRequestModel(body))
}

func TestParseRequestModelOpenAI(t *testing.T) {
	t.Parallel()
	checkParseRequestModel(t,
		[]byte(`{"model":"gpt-4o","messages":[{"role":"user","content":"hi"}]}`),
		"gpt-4o")
}

func TestParseRequestModelAnthropic(t *testing.T) {
	t.Parallel()
	checkParseRequestModel(t,
		[]byte(`{"model":"claude-3-opus-20240229","max_tokens":1024,"messages":[{"role":"user","content":"hi"}]}`),
		"claude-3-opus-20240229")
}

func TestParseRequestModelNoModel(t *testing.T) {
	t.Parallel()
	checkParseRequestModel(t,
		[]byte(`{"messages":[{"role":"user","content":"hi"}]}`),
		"")
}

func TestParseRequestModelInvalidJSON(t *testing.T) {
	t.Parallel()
	checkParseRequestModel(t, []byte(`not json`), "")
	checkParseRequestModel(t, nil, "")
	checkParseRequestModel(t, []byte{}, "")
}

func TestParseOpenAIResponse(t *testing.T) {
	t.Parallel()
	body := []byte(`{
		"id": "chatcmpl-abc123",
		"model": "gpt-4o-2024-05-13",
		"usage": {
			"prompt_tokens": 25,
			"completion_tokens": 8
		},
		"choices": [
			{
				"finish_reason": "stop",
				"message": {"role": "assistant", "content": "Hello!"}
			}
		]
	}`)

	assert.Equal(t, ParsedResponse{
		ID:           "chatcmpl-abc123",
		Model:        "gpt-4o-2024-05-13",
		InputTokens:  25,
		OutputTokens: 8,
		FinishReason: "stop",
	}, ParseOpenAIResponse(body))
}

func TestParseOpenAIResponseMissingUsage(t *testing.T) {
	t.Parallel()
	body := []byte(`{
		"id": "chatcmpl-xyz",
		"model": "gpt-4o",
		"choices": [{"finish_reason": "length"}]
	}`)

	assert.Equal(t, ParsedResponse{
		ID:           "chatcmpl-xyz",
		Model:        "gpt-4o",
		FinishReason: "length",
	}, ParseOpenAIResponse(body))
}

func TestParseOpenAIResponseNoChoices(t *testing.T) {
	t.Parallel()
	body := []byte(`{"id": "chatcmpl-xyz", "model": "gpt-4o", "usage": {"prompt_tokens": 10, "completion_tokens": 5}}`)

	assert.Equal(t, ParsedResponse{
		ID:           "chatcmpl-xyz",
		Model:        "gpt-4o",
		InputTokens:  10,
		OutputTokens: 5,
	}, ParseOpenAIResponse(body))
}

func TestParseOpenAIResponseInvalidJSON(t *testing.T) {
	t.Parallel()
	pr := ParseOpenAIResponse([]byte(`not json`))
	assert.Equal(t, ParsedResponse{}, pr)

	pr2 := ParseOpenAIResponse(nil)
	assert.Equal(t, ParsedResponse{}, pr2)
}

func TestParseUnknownFormat(t *testing.T) {
	t.Parallel()
	// Anthropic format — should return zero values since we only parse OpenAI.
	body := []byte(`{
		"id": "msg_123",
		"type": "message",
		"content": [{"type": "text", "text": "Hi!"}],
		"model": "claude-3-opus-20240229",
		"usage": {"input_tokens": 10, "output_tokens": 5}
	}`)

	// Model and ID still parse (they share the same JSON keys).
	// Tokens don't — Anthropic uses input_tokens, not prompt_tokens.
	assert.Equal(t, ParsedResponse{
		ID:    "msg_123",
		Model: "claude-3-opus-20240229",
	}, ParseOpenAIResponse(body))
}

func TestParseRequestModelProperties(t *testing.T) {
	t.Parallel()
	rapid.Check(t, func(t *rapid.T) {
		model := rapid.StringMatching(`[a-z0-9\-]{1,30}`).Draw(t, "model")
		body, _ := json.Marshal(map[string]string{"model": model})
		got := ParseRequestModel(body)
		assert.Equal(t, model, got)
	})
}

func TestParseOpenAIResponseProperties(t *testing.T) {
	t.Parallel()
	rapid.Check(t, func(t *rapid.T) {
		id := rapid.StringMatching(`chatcmpl-[a-z0-9]{5,15}`).Draw(t, "id")
		model := rapid.StringMatching(`[a-z0-9\-]{1,30}`).Draw(t, "model")
		promptTokens := rapid.Int64Range(0, 1_000_000).Draw(t, "promptTokens")
		completionTokens := rapid.Int64Range(0, 1_000_000).Draw(t, "completionTokens")
		finishReason := rapid.SampledFrom([]string{"stop", "length", "content_filter"}).Draw(t, "finishReason")

		resp := map[string]any{
			"id":    id,
			"model": model,
			"usage": map[string]int64{
				"prompt_tokens":     promptTokens,
				"completion_tokens": completionTokens,
			},
			"choices": []map[string]string{
				{"finish_reason": finishReason},
			},
		}
		body, _ := json.Marshal(resp)

		pr := ParseOpenAIResponse(body)
		assert.Equal(t, ParsedResponse{
			ID:           id,
			Model:        model,
			InputTokens:  promptTokens,
			OutputTokens: completionTokens,
			FinishReason: finishReason,
		}, pr)
	})
}

func TestParseOpenAISSEChunkNormal(t *testing.T) {
	t.Parallel()
	data := `{"id":"chatcmpl-abc","model":"gpt-4o","choices":[{"delta":{"content":"Hello"},"finish_reason":null}]}`
	assert.Equal(t, ParsedResponse{
		ID:    "chatcmpl-abc",
		Model: "gpt-4o",
	}, ParseOpenAISSEChunk(data))
}

func TestParseOpenAISSEChunkWithFinishReason(t *testing.T) {
	t.Parallel()
	data := `{"id":"chatcmpl-abc","model":"gpt-4o","choices":[{"delta":{},"finish_reason":"stop"}],"usage":{"prompt_tokens":25,"completion_tokens":8}}`
	assert.Equal(t, ParsedResponse{
		ID:           "chatcmpl-abc",
		Model:        "gpt-4o",
		FinishReason: "stop",
		InputTokens:  25,
		OutputTokens: 8,
	}, ParseOpenAISSEChunk(data))
}

func TestParseOpenAISSEChunkDone(t *testing.T) {
	t.Parallel()
	pr := ParseOpenAISSEChunk("[DONE]")
	assert.Equal(t, ParsedResponse{}, pr)
}

func TestParseOpenAISSEChunkEmpty(t *testing.T) {
	t.Parallel()
	assert.Equal(t, ParsedResponse{}, ParseOpenAISSEChunk(""))
	assert.Equal(t, ParsedResponse{}, ParseOpenAISSEChunk("   "))
}

func TestParseOpenAISSEChunkInvalidJSON(t *testing.T) {
	t.Parallel()
	assert.Equal(t, ParsedResponse{}, ParseOpenAISSEChunk("not json"))
}

func TestParseOpenAISSEChunkNullFinishReason(t *testing.T) {
	t.Parallel()
	// finish_reason: null should be treated as absent.
	data := `{"id":"chatcmpl-abc","model":"gpt-4o","choices":[{"delta":{"content":"Hi"},"finish_reason":null}]}`
	assert.Equal(t, ParsedResponse{
		ID:    "chatcmpl-abc",
		Model: "gpt-4o",
	}, ParseOpenAISSEChunk(data))
}

func TestMergeSSEChunks(t *testing.T) {
	t.Parallel()
	chunks := []ParsedResponse{
		{ID: "chatcmpl-abc", Model: "gpt-4o"},
		{ID: "chatcmpl-abc", Model: "gpt-4o"},
		{ID: "chatcmpl-abc", Model: "gpt-4o", FinishReason: "stop", InputTokens: 25, OutputTokens: 8},
	}
	assert.Equal(t, ParsedResponse{
		ID:           "chatcmpl-abc",
		Model:        "gpt-4o",
		FinishReason: "stop",
		InputTokens:  25,
		OutputTokens: 8,
	}, MergeSSEChunks(chunks))
}

func TestMergeSSEChunksEmpty(t *testing.T) {
	t.Parallel()
	assert.Equal(t, ParsedResponse{}, MergeSSEChunks(nil))
	assert.Equal(t, ParsedResponse{}, MergeSSEChunks([]ParsedResponse{}))
}

func TestMergeSSEChunksSingle(t *testing.T) {
	t.Parallel()
	chunks := []ParsedResponse{
		{ID: "chatcmpl-abc", Model: "gpt-4o", FinishReason: "stop", InputTokens: 10, OutputTokens: 5},
	}
	merged := MergeSSEChunks(chunks)
	assert.Equal(t, chunks[0], merged)
}

func TestMergeSSEChunksProperties(t *testing.T) {
	t.Parallel()
	rapid.Check(t, func(t *rapid.T) {
		n := rapid.IntRange(1, 20).Draw(t, "numChunks")
		id := rapid.StringMatching(`chatcmpl-[a-z0-9]{5,10}`).Draw(t, "id")
		model := rapid.StringMatching(`[a-z0-9\-]{1,20}`).Draw(t, "model")
		finishReason := rapid.SampledFrom([]string{"stop", "length"}).Draw(t, "finishReason")
		inputTokens := rapid.Int64Range(1, 100000).Draw(t, "inputTokens")
		outputTokens := rapid.Int64Range(1, 100000).Draw(t, "outputTokens")

		chunks := make([]ParsedResponse, n)
		// All chunks have id and model (like real OpenAI streams).
		for i := range chunks {
			chunks[i] = ParsedResponse{ID: id, Model: model}
		}
		// Last chunk gets finish_reason and tokens.
		chunks[n-1].FinishReason = finishReason
		chunks[n-1].InputTokens = inputTokens
		chunks[n-1].OutputTokens = outputTokens

		merged := MergeSSEChunks(chunks)
		assert.Equal(t, ParsedResponse{
			ID:           id,
			Model:        model,
			FinishReason: finishReason,
			InputTokens:  inputTokens,
			OutputTokens: outputTokens,
		}, merged)
	})
}

func TestParseOpenAIResponseWithToolCalls(t *testing.T) {
	t.Parallel()
	body := []byte(`{
		"id": "chatcmpl-tc1",
		"model": "gpt-4o-2024-05-13",
		"usage": {"prompt_tokens": 100, "completion_tokens": 50},
		"choices": [{
			"finish_reason": "tool_calls",
			"message": {
				"role": "assistant",
				"tool_calls": [
					{"id": "call_1", "type": "function", "function": {"name": "read_file", "arguments": "{\"path\":\"main.go\"}"}},
					{"id": "call_2", "type": "function", "function": {"name": "grep", "arguments": "{\"pattern\":\"TODO\"}"}}
				]
			}
		}]
	}`)

	assert.Equal(t, ParsedResponse{
		ID:           "chatcmpl-tc1",
		Model:        "gpt-4o-2024-05-13",
		InputTokens:  100,
		OutputTokens: 50,
		FinishReason: "tool_calls",
		ToolCalls:    []string{"read_file", "grep"},
		ToolCallArgs: []string{`{"path":"main.go"}`, `{"pattern":"TODO"}`},
	}, ParseOpenAIResponse(body))
}

func TestParseOpenAIResponseNoToolCalls(t *testing.T) {
	t.Parallel()
	body := []byte(`{
		"id": "chatcmpl-tc2",
		"model": "gpt-4o",
		"choices": [{"finish_reason": "stop", "message": {"role": "assistant", "content": "Done."}}]
	}`)

	pr := ParseOpenAIResponse(body)
	assert.Nil(t, pr.ToolCalls)
	assert.Equal(t, "stop", pr.FinishReason)
}

func TestParseOpenAISSEChunkWithToolCall(t *testing.T) {
	t.Parallel()
	data := `{"id":"chatcmpl-sse-tc","model":"gpt-4o","choices":[{"delta":{"tool_calls":[{"index":0,"function":{"name":"edit_file"}}]},"finish_reason":null}]}`

	pr := ParseOpenAISSEChunk(data)
	assert.Equal(t, "chatcmpl-sse-tc", pr.ID)
	assert.Equal(t, "gpt-4o", pr.Model)
	assert.Equal(t, []string{"edit_file"}, pr.ToolCalls)
	assert.Equal(t, []string{""}, pr.ToolCallArgs)
}

func TestMergeSSEChunksWithToolCalls(t *testing.T) {
	t.Parallel()
	// Tool call names and arguments arrive across multiple chunks.
	tc := func(index int, name string, args string) openaiToolCall {
		return openaiToolCall{Index: index, Function: struct {
			Name      string `json:"name"`
			Arguments string `json:"arguments"`
		}{Name: name, Arguments: args}}
	}

	chunks := []ParsedResponse{
		{ID: "chatcmpl-m1", Model: "gpt-4o", rawToolCalls: []openaiToolCall{tc(0, "read_file", "")}},
		{ID: "chatcmpl-m1", Model: "gpt-4o", rawToolCalls: []openaiToolCall{tc(0, "", `{"path":"`)}},
		{ID: "chatcmpl-m1", Model: "gpt-4o", rawToolCalls: []openaiToolCall{tc(0, "", `main.go"}`)}},
		{ID: "chatcmpl-m1", Model: "gpt-4o", rawToolCalls: []openaiToolCall{tc(1, "grep", `{"pattern":"TODO"}`)}},
		{ID: "chatcmpl-m1", Model: "gpt-4o", FinishReason: "tool_calls", InputTokens: 100, OutputTokens: 50},
	}

	assert.Equal(t, ParsedResponse{
		ID:           "chatcmpl-m1",
		Model:        "gpt-4o",
		FinishReason: "tool_calls",
		InputTokens:  100,
		OutputTokens: 50,
		ToolCalls:    []string{"read_file", "grep"},
		ToolCallArgs: []string{`{"path":"main.go"}`, `{"pattern":"TODO"}`},
	}, MergeSSEChunks(chunks))
}

func TestMergeSSEChunksNoToolCalls(t *testing.T) {
	t.Parallel()
	chunks := []ParsedResponse{
		{ID: "chatcmpl-nt", Model: "gpt-4o"},
		{ID: "chatcmpl-nt", Model: "gpt-4o", FinishReason: "stop"},
	}

	merged := MergeSSEChunks(chunks)
	assert.Nil(t, merged.ToolCalls)
}

func TestParseOpenAIErrorWithType(t *testing.T) {
	t.Parallel()
	body := []byte(`{"error":{"message":"Rate limit exceeded","type":"rate_limit_exceeded","code":"rate_limit"}}`)
	assert.Equal(t, "rate_limit_exceeded", ParseOpenAIError(body))
}

func TestParseOpenAIErrorFallbackToCode(t *testing.T) {
	t.Parallel()
	body := []byte(`{"error":{"message":"Something went wrong","type":"","code":"server_error"}}`)
	assert.Equal(t, "server_error", ParseOpenAIError(body))
}

func TestParseOpenAIErrorFallbackToMessage(t *testing.T) {
	t.Parallel()
	body := []byte(`{"error":{"message":"Unexpected failure","type":"","code":null}}`)
	assert.Equal(t, "Unexpected failure", ParseOpenAIError(body))
}

func TestParseOpenAIErrorInvalidJSON(t *testing.T) {
	t.Parallel()
	assert.Equal(t, "", ParseOpenAIError([]byte(`not json`)))
	assert.Equal(t, "", ParseOpenAIError(nil))
}

func TestParseOpenAIErrorEmptyError(t *testing.T) {
	t.Parallel()
	assert.Equal(t, "", ParseOpenAIError([]byte(`{"error":{}}`)))
}

func TestParseOpenAIErrorNonErrorBody(t *testing.T) {
	t.Parallel()
	assert.Equal(t, "", ParseOpenAIError([]byte(`{"id":"chatcmpl-abc","model":"gpt-4o"}`)))
}

// --- Anthropic parser tests ---

func checkParseAnthropicResponse(t *testing.T, body string, want ParsedResponse) {
	t.Helper()
	got := ParseAnthropicResponse([]byte(body))
	assert.Equal(t, want, got)
}

func TestParseAnthropicResponse(t *testing.T) {
	t.Parallel()
	checkParseAnthropicResponse(t, `{
		"id": "msg_01XFDUDYJgAACzvnptvVoYEL",
		"type": "message",
		"model": "claude-3-opus-20240229",
		"stop_reason": "end_turn",
		"usage": {"input_tokens": 25, "output_tokens": 8},
		"content": [{"type": "text", "text": "Hello!"}]
	}`, ParsedResponse{
		ID:           "msg_01XFDUDYJgAACzvnptvVoYEL",
		Model:        "claude-3-opus-20240229",
		InputTokens:  25,
		OutputTokens: 8,
		FinishReason: "end_turn",
	})
}

func TestParseAnthropicResponseWithToolUse(t *testing.T) {
	t.Parallel()
	checkParseAnthropicResponse(t, `{
		"id": "msg_tools1",
		"type": "message",
		"model": "claude-3-5-sonnet-20241022",
		"stop_reason": "tool_use",
		"usage": {"input_tokens": 100, "output_tokens": 50},
		"content": [
			{"type": "text", "text": "Let me check the weather."},
			{"type": "tool_use", "id": "toolu_01", "name": "get_weather", "input": {"location": "SF"}},
			{"type": "tool_use", "id": "toolu_02", "name": "get_time", "input": {"timezone": "PST"}}
		]
	}`, ParsedResponse{
		ID:           "msg_tools1",
		Model:        "claude-3-5-sonnet-20241022",
		InputTokens:  100,
		OutputTokens: 50,
		FinishReason: "tool_use",
		ToolCalls:    []string{"get_weather", "get_time"},
		ToolCallArgs: []string{`{"location": "SF"}`, `{"timezone": "PST"}`},
	})
}

func TestParseAnthropicResponseMissingUsage(t *testing.T) {
	t.Parallel()
	checkParseAnthropicResponse(t, `{
		"id": "msg_no_usage",
		"type": "message",
		"model": "claude-3-haiku-20240307",
		"stop_reason": "max_tokens",
		"content": [{"type": "text", "text": "..."}]
	}`, ParsedResponse{
		ID:           "msg_no_usage",
		Model:        "claude-3-haiku-20240307",
		FinishReason: "max_tokens",
	})
}

func TestParseAnthropicResponseNullStopReason(t *testing.T) {
	t.Parallel()
	// stop_reason: null should be treated as absent.
	checkParseAnthropicResponse(t, `{
		"id": "msg_null_stop",
		"type": "message",
		"model": "claude-3-opus-20240229",
		"stop_reason": null,
		"usage": {"input_tokens": 10, "output_tokens": 5},
		"content": []
	}`, ParsedResponse{
		ID:           "msg_null_stop",
		Model:        "claude-3-opus-20240229",
		InputTokens:  10,
		OutputTokens: 5,
	})
}

func TestParseAnthropicResponseInvalidJSON(t *testing.T) {
	t.Parallel()
	assert.Equal(t, ParsedResponse{}, ParseAnthropicResponse([]byte(`not json`)))
	assert.Equal(t, ParsedResponse{}, ParseAnthropicResponse(nil))
}

func TestParseAnthropicResponseProperties(t *testing.T) {
	t.Parallel()
	rapid.Check(t, func(t *rapid.T) {
		id := rapid.StringMatching(`msg_[a-zA-Z0-9]{10,20}`).Draw(t, "id")
		model := rapid.StringMatching(`claude-[a-z0-9\-]{5,30}`).Draw(t, "model")
		inputTokens := rapid.Int64Range(0, 1_000_000).Draw(t, "inputTokens")
		outputTokens := rapid.Int64Range(0, 1_000_000).Draw(t, "outputTokens")
		stopReason := rapid.SampledFrom([]string{"end_turn", "max_tokens", "stop_sequence"}).Draw(t, "stopReason")

		resp := map[string]any{
			"id":          id,
			"type":        "message",
			"model":       model,
			"stop_reason": stopReason,
			"usage": map[string]int64{
				"input_tokens":  inputTokens,
				"output_tokens": outputTokens,
			},
			"content": []any{},
		}
		body, _ := json.Marshal(resp)

		pr := ParseAnthropicResponse(body)
		assert.Equal(t, ParsedResponse{
			ID:           id,
			Model:        model,
			InputTokens:  inputTokens,
			OutputTokens: outputTokens,
			FinishReason: stopReason,
		}, pr)
	})
}

func TestParseAnthropicSSEChunkMessageStart(t *testing.T) {
	t.Parallel()
	data := `{"type":"message_start","message":{"id":"msg_sse1","type":"message","role":"assistant","content":[],"model":"claude-3-opus-20240229","stop_reason":null,"usage":{"input_tokens":25,"output_tokens":1}}}`
	assert.Equal(t, ParsedResponse{
		ID:          "msg_sse1",
		Model:       "claude-3-opus-20240229",
		InputTokens: 25,
	}, ParseAnthropicSSEChunk("message_start", data))
}

func TestParseAnthropicSSEChunkContentBlockStartText(t *testing.T) {
	t.Parallel()
	data := `{"type":"content_block_start","index":0,"content_block":{"type":"text","text":""}}`
	// Text blocks don't carry useful data at start.
	assert.Equal(t, ParsedResponse{}, ParseAnthropicSSEChunk("content_block_start", data))
}

func TestParseAnthropicSSEChunkContentBlockStartToolUse(t *testing.T) {
	t.Parallel()
	data := `{"type":"content_block_start","index":1,"content_block":{"type":"tool_use","id":"toolu_01","name":"get_weather","input":{}}}`
	pr := ParseAnthropicSSEChunk("content_block_start", data)
	assert.Equal(t, []string{"get_weather"}, pr.ToolCalls)
	assert.Equal(t, []string{""}, pr.ToolCallArgs)
}

func TestParseAnthropicSSEChunkContentBlockDeltaTextDelta(t *testing.T) {
	t.Parallel()
	data := `{"type":"content_block_delta","index":0,"delta":{"type":"text_delta","text":"Hello"}}`
	// We don't track text content, so this should return zero.
	assert.Equal(t, ParsedResponse{}, ParseAnthropicSSEChunk("content_block_delta", data))
}

func TestParseAnthropicSSEChunkContentBlockDeltaInputJSON(t *testing.T) {
	t.Parallel()
	data := `{"type":"content_block_delta","index":1,"delta":{"type":"input_json_delta","partial_json":"{\"location\": \"San Fra"}}`
	pr := ParseAnthropicSSEChunk("content_block_delta", data)
	// Tool arg fragment stored in rawToolCalls for MergeSSEChunks.
	assert.Equal(t, 1, len(pr.rawToolCalls))
	assert.Equal(t, 1, pr.rawToolCalls[0].Index)
	assert.Equal(t, `{"location": "San Fra`, pr.rawToolCalls[0].Function.Arguments)
}

func TestParseAnthropicSSEChunkMessageDelta(t *testing.T) {
	t.Parallel()
	data := `{"type":"message_delta","delta":{"stop_reason":"end_turn","stop_sequence":null},"usage":{"output_tokens":15}}`
	assert.Equal(t, ParsedResponse{
		FinishReason: "end_turn",
		OutputTokens: 15,
	}, ParseAnthropicSSEChunk("message_delta", data))
}

func TestParseAnthropicSSEChunkMessageStop(t *testing.T) {
	t.Parallel()
	assert.Equal(t, ParsedResponse{}, ParseAnthropicSSEChunk("message_stop", `{"type":"message_stop"}`))
}

func TestParseAnthropicSSEChunkPing(t *testing.T) {
	t.Parallel()
	assert.Equal(t, ParsedResponse{}, ParseAnthropicSSEChunk("ping", `{"type":"ping"}`))
}

func TestParseAnthropicSSEChunkEmpty(t *testing.T) {
	t.Parallel()
	assert.Equal(t, ParsedResponse{}, ParseAnthropicSSEChunk("message_start", ""))
	assert.Equal(t, ParsedResponse{}, ParseAnthropicSSEChunk("message_start", "   "))
}

func TestParseAnthropicSSEChunkInvalidJSON(t *testing.T) {
	t.Parallel()
	assert.Equal(t, ParsedResponse{}, ParseAnthropicSSEChunk("message_start", "not json"))
}

func TestMergeAnthropicSSEChunks(t *testing.T) {
	t.Parallel()
	// Simulates a full Anthropic SSE stream: message_start → text deltas → message_delta.
	chunks := []ParsedResponse{
		{ID: "msg_sse1", Model: "claude-3-opus-20240229", InputTokens: 25},
		// text_delta chunks return zero ParsedResponse (no fields we track)
		{FinishReason: "end_turn", OutputTokens: 15},
	}
	assert.Equal(t, ParsedResponse{
		ID:           "msg_sse1",
		Model:        "claude-3-opus-20240229",
		InputTokens:  25,
		OutputTokens: 15,
		FinishReason: "end_turn",
	}, MergeSSEChunks(chunks))
}

func TestMergeAnthropicSSEChunksWithToolCalls(t *testing.T) {
	t.Parallel()
	// Simulates Anthropic SSE with tool use:
	// 1. message_start (id, model, input_tokens)
	// 2. content_block_start tool_use (name at index 1)
	// 3. content_block_delta input_json_delta fragments
	// 4. message_delta (stop_reason, output_tokens)
	tc := func(index int, name string, args string) openaiToolCall {
		return openaiToolCall{Index: index, Function: struct {
			Name      string `json:"name"`
			Arguments string `json:"arguments"`
		}{Name: name, Arguments: args}}
	}

	chunks := []ParsedResponse{
		{ID: "msg_tools_sse", Model: "claude-3-5-sonnet-20241022", InputTokens: 100},
		{ToolCalls: []string{"get_weather"}, ToolCallArgs: []string{""}, rawToolCalls: []openaiToolCall{tc(1, "get_weather", "")}},
		{rawToolCalls: []openaiToolCall{tc(1, "", `{"location"`)}},
		{rawToolCalls: []openaiToolCall{tc(1, "", `: "SF"}`)}},
		{FinishReason: "tool_use", OutputTokens: 50},
	}

	assert.Equal(t, ParsedResponse{
		ID:           "msg_tools_sse",
		Model:        "claude-3-5-sonnet-20241022",
		InputTokens:  100,
		OutputTokens: 50,
		FinishReason: "tool_use",
		ToolCalls:    []string{"get_weather"},
		ToolCallArgs: []string{`{"location": "SF"}`},
	}, MergeSSEChunks(chunks))
}

func TestParseAnthropicErrorWithType(t *testing.T) {
	t.Parallel()
	body := []byte(`{"type":"error","error":{"type":"overloaded_error","message":"Overloaded"}}`)
	assert.Equal(t, "overloaded_error", ParseAnthropicError(body))
}

func TestParseAnthropicErrorFallbackToMessage(t *testing.T) {
	t.Parallel()
	body := []byte(`{"type":"error","error":{"type":"","message":"Something went wrong"}}`)
	assert.Equal(t, "Something went wrong", ParseAnthropicError(body))
}

func TestParseAnthropicErrorRateLimitExceeded(t *testing.T) {
	t.Parallel()
	body := []byte(`{"type":"error","error":{"type":"rate_limit_error","message":"Rate limit exceeded"}}`)
	assert.Equal(t, "rate_limit_error", ParseAnthropicError(body))
}

func TestParseAnthropicErrorInvalidJSON(t *testing.T) {
	t.Parallel()
	assert.Equal(t, "", ParseAnthropicError([]byte(`not json`)))
	assert.Equal(t, "", ParseAnthropicError(nil))
}

func TestParseAnthropicErrorEmptyError(t *testing.T) {
	t.Parallel()
	assert.Equal(t, "", ParseAnthropicError([]byte(`{"type":"error","error":{}}`)))
}
