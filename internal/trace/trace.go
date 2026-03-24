// trace.go — OTel TracerProvider setup and span emission.
//
// Span attributes follow the gen_ai.* semantic conventions for LLM
// observability. Non-LLM HTTP calls use standard HTTP semconv.
package trace

import (
	"context"
	"fmt"
	"strconv"
	"strings"
	"time"

	"github.com/rnaudi/aitrace/internal/capture"

	"go.opentelemetry.io/otel/attribute"
	"go.opentelemetry.io/otel/codes"
	"go.opentelemetry.io/otel/exporters/otlp/otlptrace/otlptracegrpc"
	"go.opentelemetry.io/otel/sdk/resource"
	sdktrace "go.opentelemetry.io/otel/sdk/trace"
	semconv "go.opentelemetry.io/otel/semconv/v1.26.0"
	oteltrace "go.opentelemetry.io/otel/trace"
)

const defaultOTLPEndpoint = "localhost:4317"

// TracerOptions configures the OTel tracer provider.
type TracerOptions struct {
	// Endpoint is the OTLP/gRPC collector address. Defaults to localhost:4317.
	Endpoint string
	// ServiceName sets the OTel service.name resource attribute.
	// Defaults to "aitrace".
	ServiceName string
}

// NewTracerProvider creates an OTel TracerProvider that exports spans
// via OTLP/gRPC. The caller must call Shutdown on the returned provider.
func NewTracerProvider(ctx context.Context, opts TracerOptions) (*sdktrace.TracerProvider, error) {
	endpoint := opts.Endpoint
	if endpoint == "" {
		endpoint = defaultOTLPEndpoint
	}

	serviceName := opts.ServiceName
	if serviceName == "" {
		serviceName = "aitrace"
	}

	exporter, err := otlptracegrpc.New(ctx,
		otlptracegrpc.WithEndpoint(endpoint),
		otlptracegrpc.WithInsecure(),
	)
	if err != nil {
		return nil, fmt.Errorf("create otlp exporter: %w", err)
	}

	res, err := resource.New(ctx,
		resource.WithAttributes(
			semconv.ServiceNameKey.String(serviceName),
		),
	)
	if err != nil {
		return nil, fmt.Errorf("create resource: %w", err)
	}

	tp := sdktrace.NewTracerProvider(
		sdktrace.WithBatcher(exporter),
		sdktrace.WithResource(res),
	)

	return tp, nil
}

// StartSessionSpan creates a root span that groups all calls from a single
// aitrace run into one trace. The caller must End() the span when done.
func StartSessionSpan(ctx context.Context, tp oteltrace.TracerProvider, command string) (context.Context, oteltrace.Span) {
	tracer := tp.Tracer("aitrace")

	spanName := command
	if spanName == "" {
		spanName = "aitrace session"
	}

	ctx, span := tracer.Start(ctx, spanName,
		oteltrace.WithSpanKind(oteltrace.SpanKindInternal),
	)
	return ctx, span
}

// EmitSpan creates an OTel span from a CapturedCall.
func EmitSpan(ctx context.Context, tp oteltrace.TracerProvider, call capture.CapturedCall) {
	tracer := tp.Tracer("aitrace")

	spanName := callSpanName(call)

	// Use actual flow timestamps when available. Fall back to synthetic
	// timestamps (now - duration) for tests that don't set StartTime/EndTime.
	startTime := call.StartTime
	endTime := call.EndTime
	if startTime.IsZero() || endTime.IsZero() {
		endTime = time.Now()
		startTime = endTime.Add(-call.Duration)
	}

	_, span := tracer.Start(ctx, spanName,
		oteltrace.WithSpanKind(oteltrace.SpanKindClient),
		oteltrace.WithTimestamp(startTime),
	)

	attrs := []attribute.KeyValue{
		attribute.String("server.address", call.Host),
		attribute.String("http.request.method", call.Method),
		attribute.Int("http.response.status_code", call.StatusCode),
	}

	if call.IsLLM {
		attrs = append(attrs, attribute.String("aitrace.call.kind", "llm"))
	} else {
		attrs = append(attrs, attribute.String("aitrace.call.kind", "http"))
		if call.Path != "" {
			attrs = append(attrs, attribute.String("url.path", call.Path))
		}
	}

	// LLM-specific attributes (gen_ai.* semantic conventions).
	if call.IsLLM {
		attrs = append(attrs, attribute.String("gen_ai.operation.name", "chat"))
		if call.Provider != "" {
			attrs = append(attrs, attribute.String("gen_ai.system", call.Provider))
		}
		if call.RequestModel != "" {
			attrs = append(attrs, attribute.String("gen_ai.request.model", call.RequestModel))
		}
		if call.Model != "" {
			attrs = append(attrs, attribute.String("gen_ai.response.model", call.Model))
		}
		if call.ResponseID != "" {
			attrs = append(attrs, attribute.String("gen_ai.response.id", call.ResponseID))
		}
		if call.InputTokens > 0 {
			attrs = append(attrs, attribute.Int64("gen_ai.usage.input_tokens", call.InputTokens))
		}
		if call.OutputTokens > 0 {
			attrs = append(attrs, attribute.Int64("gen_ai.usage.output_tokens", call.OutputTokens))
		}
		if call.CacheReadTokens > 0 {
			attrs = append(attrs, attribute.Int64("gen_ai.usage.cache_read.input_tokens", call.CacheReadTokens))
		}
		if call.CacheWriteTokens > 0 {
			attrs = append(attrs, attribute.Int64("gen_ai.usage.cache_creation.input_tokens", call.CacheWriteTokens))
		}
		if call.FinishReason != "" {
			// Semantic convention defines finish_reasons as a string array to
			// support multi-choice responses. We always emit a single element.
			attrs = append(attrs, attribute.StringSlice("gen_ai.response.finish_reasons", []string{call.FinishReason}))
		}
		if len(call.ToolCalls) > 0 {
			attrs = append(attrs, attribute.StringSlice("aitrace.tool_calls", call.ToolCalls))
		}
		if len(call.ToolCallArgs) > 0 {
			attrs = append(attrs, attribute.StringSlice("aitrace.tool_call_arguments", call.ToolCallArgs))
		}
		if call.ErrorMessage != "" {
			attrs = append(attrs, attribute.String("aitrace.error.message", call.ErrorMessage))
		}
	}

	// Outcome captures the variable part that's excluded from span names
	// for cardinality: tool call names, error detail, or finish reason.
	if outcome := CallOutcome(call); outcome != "" {
		attrs = append(attrs, attribute.String("aitrace.call.outcome", outcome))
	}

	if call.Sequence > 0 {
		attrs = append(attrs, attribute.Int("aitrace.call.sequence", call.Sequence))
	}

	span.SetAttributes(attrs...)

	if call.StatusCode >= 400 {
		statusStr := strconv.Itoa(call.StatusCode)
		desc := statusStr
		if call.ErrorMessage != "" {
			desc = statusStr + " " + call.ErrorMessage
		}
		span.SetStatus(codes.Error, desc)
	}

	span.End(oteltrace.WithTimestamp(endTime))
}

// callSpanName builds the span name for Jaeger.
// LLM: "read_file, grep", "stop", "chat". HTTP: "GET github.com".
func callSpanName(call capture.CapturedCall) string {
	if !call.IsLLM {
		return httpCallSpanName(call)
	}
	return llmCallSpanName(call)
}

func llmCallSpanName(call capture.CapturedCall) string {
	if len(call.ToolCalls) > 0 {
		return strings.Join(call.ToolCalls, ", ")
	}
	if call.FinishReason != "" {
		return call.FinishReason
	}
	return "chat"
}

func httpCallSpanName(call capture.CapturedCall) string {
	return call.Method + " " + call.Host
}

// CallOutcome returns the aitrace.call.outcome attribute value.
func CallOutcome(call capture.CapturedCall) string {
	if call.IsLLM {
		switch {
		case call.StatusCode >= 400:
			outcome := strconv.Itoa(call.StatusCode)
			if call.ErrorMessage != "" {
				outcome += " " + call.ErrorMessage
			}
			return outcome
		case len(call.ToolCalls) > 0:
			return strings.Join(call.ToolCalls, ", ")
		case call.FinishReason != "":
			return call.FinishReason
		}
		return ""
	}
	if call.StatusCode > 0 {
		return strconv.Itoa(call.StatusCode)
	}
	return ""
}
