package trace

import (
	"testing"
	"time"

	"github.com/rnaudi/aitrace/internal/capture"

	"github.com/stretchr/testify/assert"
	"go.opentelemetry.io/otel/attribute"
	"go.opentelemetry.io/otel/codes"
	sdktrace "go.opentelemetry.io/otel/sdk/trace"
	"go.opentelemetry.io/otel/sdk/trace/tracetest"
)

func TestEmitSpanAttributes(t *testing.T) {
	t.Parallel()

	tp, exporter := newTestTracerProvider(t)

	call := capture.CapturedCall{
		Method:       "POST",
		Host:         "api.openai.com",
		Path:         "/v1/chat/completions",
		StatusCode:   200,
		Duration:     142 * time.Millisecond,
		Sequence:     1,
		IsLLM:        true,
		Provider:     capture.ProviderOpenAI,
		RequestModel: "gpt-4o",
		Model:        "gpt-4o-2024-05-13",
		ResponseID:   "chatcmpl-abc123",
		InputTokens:  25,
		OutputTokens: 8,
		FinishReason: "stop",
	}

	EmitSpan(t.Context(), tp, call)
	tp.ForceFlush(t.Context())

	spans := exporter.GetSpans()
	assert.Equal(t, 1, len(spans))

	span := spans[0]
	assert.Equal(t, "stop", span.Name)

	attrMap := spanAttrMap(span)
	assert.Equal(t, map[string]attribute.Value{
		"gen_ai.operation.name":          attribute.StringValue("chat"),
		"gen_ai.system":                  attribute.StringValue("openai"),
		"gen_ai.request.model":           attribute.StringValue("gpt-4o"),
		"gen_ai.response.model":          attribute.StringValue("gpt-4o-2024-05-13"),
		"gen_ai.response.id":             attribute.StringValue("chatcmpl-abc123"),
		"gen_ai.usage.input_tokens":      attribute.Int64Value(25),
		"gen_ai.usage.output_tokens":     attribute.Int64Value(8),
		"gen_ai.response.finish_reasons": attribute.StringSliceValue([]string{"stop"}),
		"aitrace.call.kind":              attribute.StringValue("llm"),
		"aitrace.call.outcome":           attribute.StringValue("stop"),
		"aitrace.call.sequence":          attribute.Int64Value(1),
		"aitrace.cost":                   attribute.Float64Value(0.0001425),
		"server.address":                 attribute.StringValue("api.openai.com"),
		"http.request.method":            attribute.StringValue("POST"),
		"http.response.status_code":      attribute.Int64Value(200),
	}, attrMap)
}

func TestEmitSpanNameFallsBackToChat(t *testing.T) {
	t.Parallel()

	tp, exporter := newTestTracerProvider(t)

	call := capture.CapturedCall{
		Method:       "POST",
		Host:         "api.openai.com",
		Path:         "/v1/chat/completions",
		StatusCode:   200,
		Duration:     100 * time.Millisecond,
		IsLLM:        true,
		RequestModel: "gpt-4o",
		// No response model, no finish reason, no tool calls — falls back to "chat".
	}

	EmitSpan(t.Context(), tp, call)
	tp.ForceFlush(t.Context())

	spans := exporter.GetSpans()
	assert.Equal(t, 1, len(spans))
	assert.Equal(t, "chat", spans[0].Name)
}

func TestEmitSpanNoModel(t *testing.T) {
	t.Parallel()

	tp, exporter := newTestTracerProvider(t)

	call := capture.CapturedCall{
		Method:     "POST",
		Host:       "api.openai.com",
		Path:       "/v1/chat/completions",
		StatusCode: 401,
		Duration:   50 * time.Millisecond,
		IsLLM:      true,
		// No model at all, error status code.
	}

	EmitSpan(t.Context(), tp, call)
	tp.ForceFlush(t.Context())

	spans := exporter.GetSpans()
	assert.Equal(t, 1, len(spans))
	assert.Equal(t, "chat", spans[0].Name)
}

func TestEmitSpanOmitsZeroTokens(t *testing.T) {
	t.Parallel()

	tp, exporter := newTestTracerProvider(t)

	call := capture.CapturedCall{
		Method:     "POST",
		Host:       "api.openai.com",
		Path:       "/v1/chat/completions",
		StatusCode: 200,
		Duration:   100 * time.Millisecond,
		IsLLM:      true,
		Model:      "gpt-4o",
		// Tokens are 0 — should not appear as attributes.
	}

	EmitSpan(t.Context(), tp, call)
	tp.ForceFlush(t.Context())

	spans := exporter.GetSpans()
	assert.Equal(t, 1, len(spans))

	attrMap := spanAttrMap(spans[0])
	_, hasInput := attrMap["gen_ai.usage.input_tokens"]
	_, hasOutput := attrMap["gen_ai.usage.output_tokens"]
	_, hasCacheRead := attrMap["gen_ai.usage.cache_read.input_tokens"]
	_, hasCacheWrite := attrMap["gen_ai.usage.cache_creation.input_tokens"]
	assert.False(t, hasInput, "should not have input_tokens when 0")
	assert.False(t, hasOutput, "should not have output_tokens when 0")
	assert.False(t, hasCacheRead, "should not have cache_read tokens when 0")
	assert.False(t, hasCacheWrite, "should not have cache_creation tokens when 0")
}

func TestEmitSpanCacheTokenAttributes(t *testing.T) {
	t.Parallel()

	tp, exporter := newTestTracerProvider(t)

	call := capture.CapturedCall{
		Method:           "POST",
		Host:             "api.anthropic.com",
		Path:             "/v1/messages",
		StatusCode:       200,
		Duration:         1200 * time.Millisecond,
		Sequence:         1,
		IsLLM:            true,
		Provider:         capture.ProviderAnthropic,
		Model:            "claude-3-5-sonnet-20241022",
		InputTokens:      2000,
		OutputTokens:     500,
		CacheReadTokens:  800,
		CacheWriteTokens: 1500,
		FinishReason:     "end_turn",
	}

	EmitSpan(t.Context(), tp, call)
	tp.ForceFlush(t.Context())

	spans := exporter.GetSpans()
	assert.Equal(t, 1, len(spans))

	attrMap := spanAttrMap(spans[0])
	assert.Equal(t, attribute.Int64Value(2000), attrMap["gen_ai.usage.input_tokens"])
	assert.Equal(t, attribute.Int64Value(500), attrMap["gen_ai.usage.output_tokens"])
	assert.Equal(t, attribute.Int64Value(800), attrMap["gen_ai.usage.cache_read.input_tokens"])
	assert.Equal(t, attribute.Int64Value(1500), attrMap["gen_ai.usage.cache_creation.input_tokens"])
}

func TestEmitSpanCacheReadOnlyAttribute(t *testing.T) {
	t.Parallel()

	tp, exporter := newTestTracerProvider(t)

	call := capture.CapturedCall{
		Method:          "POST",
		Host:            "api.openai.com",
		Path:            "/v1/chat/completions",
		StatusCode:      200,
		Duration:        1 * time.Second,
		Sequence:        1,
		IsLLM:           true,
		Provider:        capture.ProviderOpenAI,
		Model:           "gpt-4o",
		InputTokens:     2000,
		OutputTokens:    500,
		CacheReadTokens: 1500,
		FinishReason:    "stop",
	}

	EmitSpan(t.Context(), tp, call)
	tp.ForceFlush(t.Context())

	spans := exporter.GetSpans()
	assert.Equal(t, 1, len(spans))

	attrMap := spanAttrMap(spans[0])
	assert.Equal(t, attribute.Int64Value(1500), attrMap["gen_ai.usage.cache_read.input_tokens"])
	_, hasCacheWrite := attrMap["gen_ai.usage.cache_creation.input_tokens"]
	assert.False(t, hasCacheWrite, "OpenAI has no cache write cost — attribute should be absent")
}

func TestEmitSpanClientKind(t *testing.T) {
	t.Parallel()

	tp, exporter := newTestTracerProvider(t)

	call := capture.CapturedCall{
		Method:     "POST",
		Host:       "api.openai.com",
		Path:       "/v1/chat/completions",
		StatusCode: 200,
		Duration:   100 * time.Millisecond,
		IsLLM:      true,
	}

	EmitSpan(t.Context(), tp, call)
	tp.ForceFlush(t.Context())

	spans := exporter.GetSpans()
	assert.Equal(t, 1, len(spans))
	assert.Equal(t, "client", spans[0].SpanKind.String())
}

func TestSessionSpanGroupsCallsIntoOneTrace(t *testing.T) {
	t.Parallel()

	tp, exporter := newTestTracerProvider(t)

	ctx, sessionSpan := StartSessionSpan(t.Context(), tp, "opencode")

	call1 := capture.CapturedCall{
		Method:       "POST",
		Host:         "api.openai.com",
		Path:         "/v1/chat/completions",
		StatusCode:   200,
		Duration:     100 * time.Millisecond,
		Sequence:     1,
		IsLLM:        true,
		Model:        "gpt-4o",
		FinishReason: "stop",
	}
	call2 := capture.CapturedCall{
		Method:       "POST",
		Host:         "api.openai.com",
		Path:         "/v1/chat/completions",
		StatusCode:   200,
		Duration:     200 * time.Millisecond,
		Sequence:     2,
		IsLLM:        true,
		Model:        "gpt-4o",
		FinishReason: "stop",
	}

	EmitSpan(ctx, tp, call1)
	EmitSpan(ctx, tp, call2)
	sessionSpan.End()
	tp.ForceFlush(t.Context())

	spans := exporter.GetSpans()
	assert.Equal(t, 3, len(spans))

	// All three spans (session + 2 calls) share the same trace ID.
	traceID := spans[0].SpanContext.TraceID()
	for _, s := range spans {
		assert.Equal(t, traceID, s.SpanContext.TraceID(), "span %q has different trace ID", s.Name)
	}

	// The two call spans are children of the session span.
	sessionSpanID := spans[2].SpanContext.SpanID()
	assert.Equal(t, "opencode", spans[2].Name)
	assert.Equal(t, sessionSpanID, spans[0].Parent.SpanID(), "call1 parent")
	assert.Equal(t, sessionSpanID, spans[1].Parent.SpanID(), "call2 parent")
}

func TestEmitSpanToolCallsAttribute(t *testing.T) {
	t.Parallel()

	tp, exporter := newTestTracerProvider(t)

	call := capture.CapturedCall{
		Method:       "POST",
		Host:         "api.openai.com",
		Path:         "/v1/chat/completions",
		StatusCode:   200,
		Duration:     100 * time.Millisecond,
		Sequence:     1,
		IsLLM:        true,
		Model:        "gpt-4o",
		FinishReason: "tool_calls",
		ToolCalls:    []string{"read_file", "grep"},
		ToolCallArgs: []string{`{"path":"main.go"}`, `{"pattern":"TODO"}`},
	}

	EmitSpan(t.Context(), tp, call)
	tp.ForceFlush(t.Context())

	spans := exporter.GetSpans()
	assert.Equal(t, 1, len(spans))

	attrMap := spanAttrMap(spans[0])
	assert.Equal(t, attribute.StringSliceValue([]string{"read_file", "grep"}), attrMap["aitrace.tool_calls"])
	assert.Equal(t, attribute.StringSliceValue([]string{`{"path":"main.go"}`, `{"pattern":"TODO"}`}), attrMap["aitrace.tool_call_arguments"])
	assert.Equal(t, attribute.StringValue("read_file, grep"), attrMap["aitrace.call.outcome"])
}

func TestEmitSpanNoToolCallsOmitted(t *testing.T) {
	t.Parallel()

	tp, exporter := newTestTracerProvider(t)

	call := capture.CapturedCall{
		Method:       "POST",
		Host:         "api.openai.com",
		Path:         "/v1/chat/completions",
		StatusCode:   200,
		Duration:     100 * time.Millisecond,
		IsLLM:        true,
		Model:        "gpt-4o",
		FinishReason: "stop",
		// No ToolCalls — attribute should not appear.
	}

	EmitSpan(t.Context(), tp, call)
	tp.ForceFlush(t.Context())

	spans := exporter.GetSpans()
	assert.Equal(t, 1, len(spans))

	attrMap := spanAttrMap(spans[0])
	_, hasToolCalls := attrMap["aitrace.tool_calls"]
	assert.False(t, hasToolCalls, "should not have aitrace.tool_calls when empty")
}

func TestEmitSpanErrorStatus(t *testing.T) {
	t.Parallel()

	tp, exporter := newTestTracerProvider(t)

	call := capture.CapturedCall{
		Method:       "POST",
		Host:         "api.openai.com",
		Path:         "/v1/chat/completions",
		StatusCode:   429,
		Duration:     200 * time.Millisecond,
		Sequence:     1,
		IsLLM:        true,
		RequestModel: "gpt-4o",
		ErrorMessage: "rate_limit_exceeded",
	}

	EmitSpan(t.Context(), tp, call)
	tp.ForceFlush(t.Context())

	spans := exporter.GetSpans()
	assert.Equal(t, 1, len(spans))

	span := spans[0]

	// Span status should be Error with description.
	assert.Equal(t, codes.Error, span.Status.Code)
	assert.Equal(t, "429 rate_limit_exceeded", span.Status.Description)

	// Attributes should include error message and outcome.
	attrMap := spanAttrMap(span)
	assert.Equal(t, attribute.StringValue("rate_limit_exceeded"), attrMap["aitrace.error.message"])
	assert.Equal(t, attribute.Int64Value(429), attrMap["http.response.status_code"])
	assert.Equal(t, attribute.StringValue("429 rate_limit_exceeded"), attrMap["aitrace.call.outcome"])
}

func TestEmitSpanErrorStatusNoMessage(t *testing.T) {
	t.Parallel()

	tp, exporter := newTestTracerProvider(t)

	call := capture.CapturedCall{
		Method:     "POST",
		Host:       "api.openai.com",
		Path:       "/v1/chat/completions",
		StatusCode: 500,
		Duration:   50 * time.Millisecond,
		IsLLM:      true,
		// ErrorMessage is empty — status description should be just the code.
	}

	EmitSpan(t.Context(), tp, call)
	tp.ForceFlush(t.Context())

	spans := exporter.GetSpans()
	assert.Equal(t, 1, len(spans))

	span := spans[0]
	assert.Equal(t, codes.Error, span.Status.Code)
	assert.Equal(t, "500", span.Status.Description)

	attrMap := spanAttrMap(span)
	_, hasErrorMsg := attrMap["aitrace.error.message"]
	assert.False(t, hasErrorMsg, "should not have aitrace.error.message when empty")
}

func TestEmitSpanSuccessNoErrorStatus(t *testing.T) {
	t.Parallel()

	tp, exporter := newTestTracerProvider(t)

	call := capture.CapturedCall{
		Method:     "POST",
		Host:       "api.openai.com",
		Path:       "/v1/chat/completions",
		StatusCode: 200,
		Duration:   100 * time.Millisecond,
		IsLLM:      true,
		Model:      "gpt-4o",
	}

	EmitSpan(t.Context(), tp, call)
	tp.ForceFlush(t.Context())

	spans := exporter.GetSpans()
	assert.Equal(t, 1, len(spans))

	// Status should be Unset for successful calls.
	assert.Equal(t, codes.Unset, spans[0].Status.Code)
}

func TestCallSpanNameLLMWithToolCalls(t *testing.T) {
	t.Parallel()
	name := callSpanName(capture.CapturedCall{
		Sequence:     1,
		IsLLM:        true,
		Model:        "gpt-4o",
		StatusCode:   200,
		FinishReason: "tool_calls",
		ToolCalls:    []string{"read_file", "grep"},
	})
	// Span name shows tool call names, not model.
	assert.Equal(t, "read_file, grep", name)
}

func TestCallSpanNameLLMWithError(t *testing.T) {
	t.Parallel()
	name := callSpanName(capture.CapturedCall{
		Sequence:     3,
		IsLLM:        true,
		RequestModel: "gpt-4o",
		StatusCode:   429,
		ErrorMessage: "rate_limit_exceeded",
	})
	// No tool calls, no finish reason — falls back to "chat".
	assert.Equal(t, "chat", name)
}

func TestCallSpanNameLLMWithStop(t *testing.T) {
	t.Parallel()
	name := callSpanName(capture.CapturedCall{
		Sequence:     2,
		IsLLM:        true,
		Model:        "claude-opus-4.6",
		StatusCode:   200,
		FinishReason: "stop",
	})
	// No tool calls — falls back to finish reason.
	assert.Equal(t, "stop", name)
}

func TestCallSpanNameLLMNoOutcome(t *testing.T) {
	t.Parallel()
	// No finish reason, no error, no tool calls — falls back to "chat".
	name := callSpanName(capture.CapturedCall{
		Sequence:   1,
		IsLLM:      true,
		Model:      "gpt-4o",
		StatusCode: 200,
	})
	assert.Equal(t, "chat", name)
}

func TestCallSpanNameLLMErrorNoModel(t *testing.T) {
	t.Parallel()
	// Error without any model or finish reason — falls back to "chat".
	name := callSpanName(capture.CapturedCall{
		IsLLM:      true,
		StatusCode: 500,
	})
	assert.Equal(t, "chat", name)
}

func TestCallSpanNameLLMFallback(t *testing.T) {
	t.Parallel()
	// LLM call with no model — should return "chat".
	name := callSpanName(capture.CapturedCall{IsLLM: true})
	assert.Equal(t, "chat", name)
}

func TestCallSpanNameHTTP(t *testing.T) {
	t.Parallel()
	name := callSpanName(capture.CapturedCall{
		Method:     "GET",
		Host:       "github.com",
		Path:       "/foo/bar/issues/123",
		StatusCode: 200,
		Sequence:   5,
	})
	// Low cardinality: method + host only.
	assert.Equal(t, "GET github.com", name)
}

func TestCallSpanNameHTTPPost(t *testing.T) {
	t.Parallel()
	name := callSpanName(capture.CapturedCall{
		Method:     "POST",
		Host:       "api.stripe.com",
		Path:       "/v1/charges",
		StatusCode: 201,
	})
	assert.Equal(t, "POST api.stripe.com", name)
}

func TestCallSpanNameNonLLMEmptyCall(t *testing.T) {
	t.Parallel()
	// Non-LLM call with no fields — method + host (both empty).
	name := callSpanName(capture.CapturedCall{})
	assert.Equal(t, " ", name)
}

func TestEmitSpanNonLLMAttributes(t *testing.T) {
	t.Parallel()

	tp, exporter := newTestTracerProvider(t)

	call := capture.CapturedCall{
		Method:     "GET",
		Host:       "github.com",
		Path:       "/foo/bar/issues/123",
		StatusCode: 200,
		Duration:   800 * time.Millisecond,
		Sequence:   5,
		IsLLM:      false,
	}

	EmitSpan(t.Context(), tp, call)
	tp.ForceFlush(t.Context())

	spans := exporter.GetSpans()
	assert.Equal(t, 1, len(spans))

	span := spans[0]
	assert.Equal(t, "GET github.com", span.Name)
	assert.Equal(t, "client", span.SpanKind.String())

	attrMap := spanAttrMap(span)
	assert.Equal(t, map[string]attribute.Value{
		"aitrace.call.kind":         attribute.StringValue("http"),
		"aitrace.call.outcome":      attribute.StringValue("200"),
		"aitrace.call.sequence":     attribute.Int64Value(5),
		"server.address":            attribute.StringValue("github.com"),
		"http.request.method":       attribute.StringValue("GET"),
		"http.response.status_code": attribute.Int64Value(200),
		"url.path":                  attribute.StringValue("/foo/bar/issues/123"),
	}, attrMap)

	// No gen_ai.* attributes on non-LLM spans.
	_, hasGenAI := attrMap["gen_ai.operation.name"]
	assert.False(t, hasGenAI, "non-LLM span should not have gen_ai attributes")
}

func TestEmitSpanNonLLMErrorStatus(t *testing.T) {
	t.Parallel()

	tp, exporter := newTestTracerProvider(t)

	call := capture.CapturedCall{
		Method:     "POST",
		Host:       "api.example.com",
		Path:       "/webhook",
		StatusCode: 500,
		Duration:   100 * time.Millisecond,
		Sequence:   3,
		IsLLM:      false,
	}

	EmitSpan(t.Context(), tp, call)
	tp.ForceFlush(t.Context())

	spans := exporter.GetSpans()
	assert.Equal(t, 1, len(spans))

	span := spans[0]
	assert.Equal(t, "POST api.example.com", span.Name)
	assert.Equal(t, codes.Error, span.Status.Code)
	assert.Equal(t, "500", span.Status.Description)
}

func TestCallOutcomeLLMToolCalls(t *testing.T) {
	t.Parallel()
	outcome := CallOutcome(capture.CapturedCall{
		IsLLM:      true,
		StatusCode: 200,
		ToolCalls:  []string{"read_file", "grep"},
	})
	assert.Equal(t, "read_file, grep", outcome)
}

func TestCallOutcomeLLMError(t *testing.T) {
	t.Parallel()
	outcome := CallOutcome(capture.CapturedCall{
		IsLLM:        true,
		StatusCode:   429,
		ErrorMessage: "rate_limit_exceeded",
	})
	assert.Equal(t, "429 rate_limit_exceeded", outcome)
}

func TestCallOutcomeLLMFinishReason(t *testing.T) {
	t.Parallel()
	outcome := CallOutcome(capture.CapturedCall{
		IsLLM:        true,
		StatusCode:   200,
		FinishReason: "stop",
	})
	assert.Equal(t, "stop", outcome)
}

func TestCallOutcomeHTTP(t *testing.T) {
	t.Parallel()
	outcome := CallOutcome(capture.CapturedCall{
		IsLLM:      false,
		StatusCode: 200,
	})
	assert.Equal(t, "200", outcome)
}

func TestCallOutcomeHTTPNoStatus(t *testing.T) {
	t.Parallel()
	outcome := CallOutcome(capture.CapturedCall{IsLLM: false})
	assert.Equal(t, "", outcome)
}

func TestSessionSpanNameShowsCommand(t *testing.T) {
	t.Parallel()

	tp, exporter := newTestTracerProvider(t)

	_, span := StartSessionSpan(t.Context(), tp, "opencode")
	span.End()
	tp.ForceFlush(t.Context())

	spans := exporter.GetSpans()
	assert.Equal(t, 1, len(spans))
	assert.Equal(t, "opencode", spans[0].Name)
}

func TestSessionSpanNameEmptyFallback(t *testing.T) {
	t.Parallel()

	tp, exporter := newTestTracerProvider(t)

	_, span := StartSessionSpan(t.Context(), tp, "")
	span.End()
	tp.ForceFlush(t.Context())

	spans := exporter.GetSpans()
	assert.Equal(t, 1, len(spans))
	assert.Equal(t, "aitrace session", spans[0].Name)
}

func TestSessionSpanSetNameWithPID(t *testing.T) {
	t.Parallel()

	tp, exporter := newTestTracerProvider(t)

	_, span := StartSessionSpan(t.Context(), tp, "opencode")
	// Simulate what main.go does after RunChild returns the PID.
	span.SetName("opencode (pid 42)")
	span.End()
	tp.ForceFlush(t.Context())

	spans := exporter.GetSpans()
	assert.Equal(t, 1, len(spans))
	assert.Equal(t, "opencode (pid 42)", spans[0].Name)
}

func newTestTracerProvider(t *testing.T) (*sdktrace.TracerProvider, *tracetest.InMemoryExporter) {
	t.Helper()
	exporter := tracetest.NewInMemoryExporter()
	tp := sdktrace.NewTracerProvider(sdktrace.WithSyncer(exporter))
	t.Cleanup(func() { tp.Shutdown(t.Context()) })
	return tp, exporter
}

func spanAttrMap(span tracetest.SpanStub) map[string]attribute.Value {
	m := make(map[string]attribute.Value, len(span.Attributes))
	for _, a := range span.Attributes {
		m[string(a.Key)] = a.Value
	}
	return m
}
