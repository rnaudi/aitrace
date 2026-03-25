// integration_test.go — End-to-end tests wiring proxy + parse + OTel.
//
// Why capture_test: This file imports both capture and trace. Since trace
// imports capture, using package capture would create an import cycle.
package capture_test

import (
	"context"
	"crypto/tls"
	"fmt"
	"io"
	"net/http"
	"net/http/httptest"
	"net/url"
	"strings"
	"sync"
	"testing"
	"time"

	"github.com/rnaudi/aitrace/internal/capture"
	"github.com/rnaudi/aitrace/internal/trace"

	"github.com/stretchr/testify/assert"
	"go.opentelemetry.io/otel/attribute"
	sdktrace "go.opentelemetry.io/otel/sdk/trace"
	"go.opentelemetry.io/otel/sdk/trace/tracetest"
)

// Duplicated from proxy_test.go (external test package).
func startTestProxyWithOpts(t *testing.T, opts capture.ProxyOptions) *capture.Proxy {
	t.Helper()

	opts.SslInsecure = true // test servers use self-signed certs
	p, err := capture.NewProxy(opts)
	if err != nil {
		t.Fatalf("NewProxy: %v", err)
	}

	var wg sync.WaitGroup
	wg.Go(func() {
		if err := p.Start(); err != nil && !strings.Contains(err.Error(), "closed") {
			// Start returns error on shutdown; ignore it.
		}
	})

	ctx, cancel := context.WithTimeout(t.Context(), 2*time.Second)
	defer cancel()
	if err := capture.WaitForProxy(ctx, p.Addr()); err != nil {
		t.Fatalf("proxy not ready: %v", err)
	}

	t.Cleanup(func() {
		p.Stop()
		wg.Wait()
	})

	return p
}

// Duplicated from proxy_test.go (external test package).
// InsecureSkipVerify because go-mitmproxy's leaf certs lack IP SANs for 127.0.0.1.
func proxyClient(t *testing.T, p *capture.Proxy) *http.Client {
	t.Helper()

	proxyURL, err := url.Parse("http://" + p.Addr())
	if err != nil {
		t.Fatalf("parse proxy URL: %v", err)
	}

	return &http.Client{
		Timeout: 10 * time.Second,
		Transport: &http.Transport{
			Proxy: http.ProxyURL(proxyURL),
			TLSClientConfig: &tls.Config{
				InsecureSkipVerify: true,
			},
		},
	}
}

func TestIntegrationProxyParseOTel(t *testing.T) {
	t.Parallel()

	exporter := tracetest.NewInMemoryExporter()
	tp := sdktrace.NewTracerProvider(sdktrace.WithSyncer(exporter))
	defer tp.Shutdown(t.Context())

	openaiResp := `{
		"id": "chatcmpl-integ1",
		"model": "gpt-4o-2024-05-13",
		"usage": {"prompt_tokens": 42, "completion_tokens": 17},
		"choices": [{"finish_reason": "stop", "message": {"role": "assistant", "content": "Integration test!"}}]
	}`
	fakeServer := httptest.NewTLSServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		w.Header().Set("Content-Type", "application/json")
		w.WriteHeader(http.StatusOK)
		w.Write([]byte(openaiResp))
	}))
	defer fakeServer.Close()

	serverURL, _ := url.Parse(fakeServer.URL)

	var mu sync.Mutex
	var capturedCalls []capture.CapturedCall

	p := startTestProxyWithOpts(t, capture.ProxyOptions{
		Hosts: []string{serverURL.Hostname()},
		OnCall: func(c capture.CapturedCall) {
			mu.Lock()
			capturedCalls = append(capturedCalls, c)
			mu.Unlock()
			trace.EmitSpan(t.Context(), tp, c)
		},
	})

	client := proxyClient(t, p)
	reqBody := `{"model":"gpt-4o","messages":[{"role":"user","content":"test"}]}`
	resp, err := client.Post(
		fakeServer.URL+"/v1/chat/completions",
		"application/json",
		strings.NewReader(reqBody),
	)
	if err != nil {
		t.Fatalf("POST: %v", err)
	}
	defer resp.Body.Close()
	io.ReadAll(resp.Body)

	mu.Lock()
	defer mu.Unlock()
	if len(capturedCalls) != 1 {
		t.Fatalf("expected 1 captured call, got %d", len(capturedCalls))
	}
	call := capturedCalls[0]

	assert.Greater(t, call.Duration, time.Duration(0))
	assert.False(t, call.StartTime.IsZero(), "StartTime should be set")
	assert.False(t, call.EndTime.IsZero(), "EndTime should be set")
	assert.Equal(t, capture.CapturedCall{
		Method:       "POST",
		Host:         serverURL.Hostname(),
		Path:         "/v1/chat/completions",
		StatusCode:   200,
		Duration:     call.Duration,
		StartTime:    call.StartTime,
		EndTime:      call.EndTime,
		Sequence:     1,
		IsLLM:        true,
		RequestModel: "gpt-4o",
		Model:        "gpt-4o-2024-05-13",
		ResponseID:   "chatcmpl-integ1",
		InputTokens:  42,
		OutputTokens: 17,
		FinishReason: "stop",
	}, call)

	tp.ForceFlush(t.Context())
	spans := exporter.GetSpans()
	if len(spans) != 1 {
		t.Fatalf("expected 1 span, got %d", len(spans))
	}

	span := spans[0]
	assert.Equal(t, "stop", span.Name)
	assert.Equal(t, "client", span.SpanKind.String())

	assert.Equal(t, map[string]attribute.Value{
		"gen_ai.operation.name":          attribute.StringValue("chat"),
		"gen_ai.request.model":           attribute.StringValue("gpt-4o"),
		"gen_ai.response.model":          attribute.StringValue("gpt-4o-2024-05-13"),
		"gen_ai.response.id":             attribute.StringValue("chatcmpl-integ1"),
		"gen_ai.usage.input_tokens":      attribute.Int64Value(42),
		"gen_ai.usage.output_tokens":     attribute.Int64Value(17),
		"gen_ai.response.finish_reasons": attribute.StringSliceValue([]string{"stop"}),
		"aitrace.call.kind":              attribute.StringValue("llm"),
		"aitrace.call.outcome":           attribute.StringValue("stop"),
		"aitrace.call.sequence":          attribute.Int64Value(1),
		"server.address":                 attribute.StringValue(serverURL.Hostname()),
		"http.request.method":            attribute.StringValue("POST"),
		"http.response.status_code":      attribute.Int64Value(200),
	}, spanAttrMap(span))
}

func TestIntegrationProxyParseOTelSSE(t *testing.T) {
	t.Parallel()

	exporter := tracetest.NewInMemoryExporter()
	tp := sdktrace.NewTracerProvider(sdktrace.WithSyncer(exporter))
	defer tp.Shutdown(t.Context())

	sseChunks := []string{
		`{"id":"chatcmpl-sseinteg","model":"gpt-4o","choices":[{"delta":{"role":"assistant"},"finish_reason":null}]}`,
		`{"id":"chatcmpl-sseinteg","model":"gpt-4o","choices":[{"delta":{"content":"Hello"},"finish_reason":null}]}`,
		`{"id":"chatcmpl-sseinteg","model":"gpt-4o","choices":[{"delta":{"content":" world"},"finish_reason":null}]}`,
		`{"id":"chatcmpl-sseinteg","model":"gpt-4o","choices":[{"delta":{},"finish_reason":"stop"}],"usage":{"prompt_tokens":30,"completion_tokens":12}}`,
	}
	fakeServer := httptest.NewTLSServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		w.Header().Set("Content-Type", "text/event-stream")
		w.Header().Set("Cache-Control", "no-cache")
		w.WriteHeader(http.StatusOK)

		flusher, ok := w.(http.Flusher)
		if !ok {
			t.Error("ResponseWriter does not implement Flusher")
			return
		}
		for _, chunk := range sseChunks {
			fmt.Fprintf(w, "data: %s\n\n", chunk)
			flusher.Flush()
		}
		fmt.Fprintf(w, "data: [DONE]\n\n")
		flusher.Flush()
	}))
	defer fakeServer.Close()

	serverURL, _ := url.Parse(fakeServer.URL)

	var mu sync.Mutex
	var capturedCalls []capture.CapturedCall

	p := startTestProxyWithOpts(t, capture.ProxyOptions{
		Hosts: []string{serverURL.Hostname()},
		OnCall: func(c capture.CapturedCall) {
			mu.Lock()
			capturedCalls = append(capturedCalls, c)
			mu.Unlock()
			trace.EmitSpan(t.Context(), tp, c)
		},
	})

	client := proxyClient(t, p)
	reqBody := `{"model":"gpt-4o","messages":[{"role":"user","content":"hi"}],"stream":true}`
	resp, err := client.Post(
		fakeServer.URL+"/v1/chat/completions",
		"application/json",
		strings.NewReader(reqBody),
	)
	if err != nil {
		t.Fatalf("POST: %v", err)
	}
	defer resp.Body.Close()

	// Reading the full body drives SSEMessage/SSEEnd hooks synchronously.
	io.ReadAll(resp.Body)

	mu.Lock()
	defer mu.Unlock()
	if len(capturedCalls) != 1 {
		t.Fatalf("expected 1 captured call from SSE stream, got %d", len(capturedCalls))
	}
	call := capturedCalls[0]

	assert.Greater(t, call.Duration, time.Duration(0))
	assert.False(t, call.StartTime.IsZero(), "StartTime should be set")
	assert.False(t, call.EndTime.IsZero(), "EndTime should be set")
	assert.Equal(t, capture.CapturedCall{
		Method:       "POST",
		Host:         serverURL.Hostname(),
		Path:         "/v1/chat/completions",
		StatusCode:   200,
		Duration:     call.Duration,
		StartTime:    call.StartTime,
		EndTime:      call.EndTime,
		Sequence:     1,
		IsLLM:        true,
		RequestModel: "gpt-4o",
		Model:        "gpt-4o",
		ResponseID:   "chatcmpl-sseinteg",
		InputTokens:  30,
		OutputTokens: 12,
		FinishReason: "stop",
	}, call)

	tp.ForceFlush(t.Context())
	spans := exporter.GetSpans()
	if len(spans) != 1 {
		t.Fatalf("expected 1 span, got %d", len(spans))
	}

	span := spans[0]
	assert.Equal(t, "stop", span.Name)
	assert.Equal(t, "client", span.SpanKind.String())

	assert.Equal(t, map[string]attribute.Value{
		"gen_ai.operation.name":          attribute.StringValue("chat"),
		"gen_ai.request.model":           attribute.StringValue("gpt-4o"),
		"gen_ai.response.model":          attribute.StringValue("gpt-4o"),
		"gen_ai.response.id":             attribute.StringValue("chatcmpl-sseinteg"),
		"gen_ai.usage.input_tokens":      attribute.Int64Value(30),
		"gen_ai.usage.output_tokens":     attribute.Int64Value(12),
		"gen_ai.response.finish_reasons": attribute.StringSliceValue([]string{"stop"}),
		"aitrace.call.kind":              attribute.StringValue("llm"),
		"aitrace.call.outcome":           attribute.StringValue("stop"),
		"aitrace.call.sequence":          attribute.Int64Value(1),
		"server.address":                 attribute.StringValue(serverURL.Hostname()),
		"http.request.method":            attribute.StringValue("POST"),
		"http.response.status_code":      attribute.Int64Value(200),
	}, spanAttrMap(span))
}

// Duplicated from trace_test.go (external test package).
func spanAttrMap(span tracetest.SpanStub) map[string]attribute.Value {
	m := make(map[string]attribute.Value, len(span.Attributes))
	for _, a := range span.Attributes {
		m[string(a.Key)] = a.Value
	}
	return m
}
