package capture

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

	"github.com/lqqyt2423/go-mitmproxy/proxy"
	"github.com/stretchr/testify/assert"
)

func startTestProxyWithOpts(t *testing.T, opts ProxyOptions) *Proxy {
	t.Helper()

	p, err := NewProxy(opts)
	if err != nil {
		t.Fatalf("NewProxy: %v", err)
	}

	var wg sync.WaitGroup
	wg.Add(1)
	go func() {
		defer wg.Done()
		if err := p.Start(); err != nil && !strings.Contains(err.Error(), "closed") {
			// Start returns error on shutdown; ignore it.
		}
	}()

	ctx, cancel := context.WithTimeout(t.Context(), 2*time.Second)
	defer cancel()
	if err := WaitForProxy(ctx, p.Addr()); err != nil {
		t.Fatalf("proxy not ready: %v", err)
	}

	t.Cleanup(func() {
		p.Stop()
		wg.Wait()
	})

	return p
}

func startTestProxy(t *testing.T, hosts []string) *Proxy {
	t.Helper()
	return startTestProxyWithOpts(t, ProxyOptions{
		Hosts: hosts,
	})
}

// InsecureSkipVerify because go-mitmproxy's leaf certs lack IP SANs for 127.0.0.1.
func proxyClient(t *testing.T, p *Proxy) *http.Client {
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

func TestProxyInterceptsHTTPS(t *testing.T) {
	t.Parallel()

	fakeServer := httptest.NewTLSServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		w.Header().Set("Content-Type", "application/json")
		w.WriteHeader(http.StatusOK)
		w.Write([]byte(`{"model":"gpt-4o","choices":[{"message":{"content":"hello"}}]}`))
	}))
	defer fakeServer.Close()

	serverURL, _ := url.Parse(fakeServer.URL)
	serverHost := serverURL.Hostname()

	p := startTestProxy(t, []string{serverHost})
	client := proxyClient(t, p)

	resp, err := client.Get(fakeServer.URL + "/v1/chat/completions")
	if err != nil {
		t.Fatalf("GET through proxy: %v", err)
	}
	defer resp.Body.Close()

	body, err := io.ReadAll(resp.Body)
	if err != nil {
		t.Fatalf("read body: %v", err)
	}

	assert.Equal(t, http.StatusOK, resp.StatusCode)
	assert.Contains(t, string(body), `"model":"gpt-4o"`)

	// Response hook fires synchronously, so p.Calls() is populated.
	calls := p.Calls()
	if len(calls) != 1 {
		t.Fatalf("expected 1 captured call, got %d", len(calls))
	}
	call := calls[0]
	assert.Greater(t, call.Duration, time.Duration(0))
	assert.False(t, call.StartTime.IsZero(), "StartTime should be set")
	assert.False(t, call.EndTime.IsZero(), "EndTime should be set")
	assert.Equal(t, CapturedCall{
		Method:     "GET",
		Host:       serverHost,
		Path:       "/v1/chat/completions",
		StatusCode: 200,
		Duration:   call.Duration,
		StartTime:  call.StartTime,
		EndTime:    call.EndTime,
		Sequence:   1,
		IsLLM:      true,
		Model:      "gpt-4o",
	}, call)
}

func TestProxyNonLLMHostCapturesMetadataOnly(t *testing.T) {
	t.Parallel()

	fakeServer := httptest.NewTLSServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		w.Header().Set("Content-Type", "application/json")
		w.WriteHeader(http.StatusOK)
		// Body looks like OpenAI but should NOT be parsed — host is not in the LLM allowlist.
		w.Write([]byte(`{"model":"gpt-4o","choices":[{"message":{"content":"sneaky"}}]}`))
	}))
	defer fakeServer.Close()

	serverURL, _ := url.Parse(fakeServer.URL)
	serverHost := serverURL.Hostname()

	// Test server host is NOT in the LLM allowlist.
	p := startTestProxy(t, []string{"api.openai.com"})
	client := proxyClient(t, p)

	resp, err := client.Get(fakeServer.URL + "/some/api/path")
	if err != nil {
		t.Fatalf("GET through proxy: %v", err)
	}
	defer resp.Body.Close()

	body, _ := io.ReadAll(resp.Body)
	assert.Equal(t, http.StatusOK, resp.StatusCode)
	assert.Contains(t, string(body), "sneaky")

	calls := p.Calls()
	if len(calls) != 1 {
		t.Fatalf("expected 1 captured call, got %d", len(calls))
	}

	call := calls[0]
	assert.Greater(t, call.Duration, time.Duration(0))
	assert.False(t, call.StartTime.IsZero(), "StartTime should be set")
	assert.False(t, call.EndTime.IsZero(), "EndTime should be set")
	assert.Equal(t, CapturedCall{
		Method:     "GET",
		Host:       serverHost,
		Path:       "/some/api/path",
		StatusCode: 200,
		Duration:   call.Duration,
		StartTime:  call.StartTime,
		EndTime:    call.EndTime,
		Sequence:   1,
		IsLLM:      false,
	}, call)
}

func TestProxyOnCallCallback(t *testing.T) {
	t.Parallel()

	fakeServer := httptest.NewTLSServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		w.WriteHeader(http.StatusOK)
		w.Write([]byte("ok"))
	}))
	defer fakeServer.Close()

	serverURL, _ := url.Parse(fakeServer.URL)
	serverHost := serverURL.Hostname()

	var mu sync.Mutex
	var callbackCalls []CapturedCall

	p := startTestProxyWithOpts(t, ProxyOptions{
		Hosts: []string{serverHost},
		OnCall: func(c CapturedCall) {
			mu.Lock()
			callbackCalls = append(callbackCalls, c)
			mu.Unlock()
		},
	})

	client := proxyClient(t, p)

	for i := 0; i < 3; i++ {
		resp, err := client.Get(fakeServer.URL + "/v1/chat/completions")
		if err != nil {
			t.Fatalf("request %d: %v", i, err)
		}
		resp.Body.Close()
	}

	mu.Lock()
	defer mu.Unlock()
	assert.Equal(t, 3, len(callbackCalls))
}

func TestProxyMultipleHosts(t *testing.T) {
	t.Parallel()

	server1 := httptest.NewTLSServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		w.WriteHeader(http.StatusOK)
		w.Write([]byte("server1"))
	}))
	defer server1.Close()

	server2 := httptest.NewTLSServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		w.WriteHeader(http.StatusCreated)
		w.Write([]byte("server2"))
	}))
	defer server2.Close()

	url1, _ := url.Parse(server1.URL)
	url2, _ := url.Parse(server2.URL)

	p := startTestProxy(t, []string{url1.Hostname(), url2.Hostname()})
	client := proxyClient(t, p)

	resp1, err := client.Get(server1.URL + "/api/v1")
	if err != nil {
		t.Fatalf("GET server1: %v", err)
	}
	resp1.Body.Close()

	resp2, err := client.Get(server2.URL + "/api/v2")
	if err != nil {
		t.Fatalf("GET server2: %v", err)
	}
	resp2.Body.Close()

	calls := p.Calls()
	if len(calls) != 2 {
		t.Fatalf("expected 2 captured calls, got %d", len(calls))
	}

	assert.Equal(t, CapturedCall{
		Method:     "GET",
		Host:       url1.Hostname(),
		Path:       "/api/v1",
		StatusCode: 200,
		Duration:   calls[0].Duration,
		StartTime:  calls[0].StartTime,
		EndTime:    calls[0].EndTime,
		Sequence:   calls[0].Sequence,
		IsLLM:      true,
	}, calls[0])
	assert.Equal(t, CapturedCall{
		Method:     "GET",
		Host:       url2.Hostname(),
		Path:       "/api/v2",
		StatusCode: 201,
		Duration:   calls[1].Duration,
		StartTime:  calls[1].StartTime,
		EndTime:    calls[1].EndTime,
		Sequence:   calls[1].Sequence,
		IsLLM:      true,
	}, calls[1])
}

func TestProxyPOSTRequest(t *testing.T) {
	t.Parallel()

	var receivedBody string

	openaiResp := `{
		"id": "chatcmpl-abc123",
		"model": "gpt-4o-2024-05-13",
		"usage": {"prompt_tokens": 25, "completion_tokens": 8},
		"choices": [{"finish_reason": "stop", "message": {"role": "assistant", "content": "Hello!"}}]
	}`

	fakeServer := httptest.NewTLSServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		b, _ := io.ReadAll(r.Body)
		receivedBody = string(b)
		w.Header().Set("Content-Type", "application/json")
		w.WriteHeader(http.StatusOK)
		w.Write([]byte(openaiResp))
	}))
	defer fakeServer.Close()

	serverURL, _ := url.Parse(fakeServer.URL)
	p := startTestProxy(t, []string{serverURL.Hostname()})
	client := proxyClient(t, p)

	reqBody := `{"model":"gpt-4o","messages":[{"role":"user","content":"hello"}]}`
	resp, err := client.Post(
		fakeServer.URL+"/v1/chat/completions",
		"application/json",
		strings.NewReader(reqBody),
	)
	if err != nil {
		t.Fatalf("POST: %v", err)
	}
	defer resp.Body.Close()

	body, _ := io.ReadAll(resp.Body)

	assert.Contains(t, string(body), "chatcmpl-abc123")
	assert.Equal(t, reqBody, receivedBody)

	calls := p.Calls()
	if len(calls) != 1 {
		t.Fatalf("expected 1 call, got %d", len(calls))
	}
	call := calls[0]
	assert.Greater(t, call.Duration, time.Duration(0))
	assert.False(t, call.StartTime.IsZero(), "StartTime should be set")
	assert.False(t, call.EndTime.IsZero(), "EndTime should be set")
	assert.Equal(t, CapturedCall{
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
		ResponseID:   "chatcmpl-abc123",
		InputTokens:  25,
		OutputTokens: 8,
		FinishReason: "stop",
	}, call)
}

func TestProxySSEStreamingResponse(t *testing.T) {
	t.Parallel()

	sseChunks := []string{
		`{"id":"chatcmpl-sse1","model":"gpt-4o","choices":[{"delta":{"role":"assistant"},"finish_reason":null}]}`,
		`{"id":"chatcmpl-sse1","model":"gpt-4o","choices":[{"delta":{"content":"Hello"},"finish_reason":null}]}`,
		`{"id":"chatcmpl-sse1","model":"gpt-4o","choices":[{"delta":{"content":"!"},"finish_reason":null}]}`,
		`{"id":"chatcmpl-sse1","model":"gpt-4o","choices":[{"delta":{},"finish_reason":"stop"}],"usage":{"prompt_tokens":15,"completion_tokens":3}}`,
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
	p := startTestProxy(t, []string{serverURL.Hostname()})
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

	// Reading the full body drives go-mitmproxy's sseReader which fires
	// SSEMessage/SSEEnd hooks synchronously.
	body, err := io.ReadAll(resp.Body)
	if err != nil {
		t.Fatalf("read SSE body: %v", err)
	}

	// Verify we got SSE data back.
	assert.Contains(t, string(body), "data: ")

	// SSEEnd fires synchronously on EOF, so p.Calls() is populated.
	calls := p.Calls()
	if len(calls) != 1 {
		t.Fatalf("expected 1 captured call from SSE stream, got %d", len(calls))
	}

	call := calls[0]
	assert.Greater(t, call.Duration, time.Duration(0))
	assert.False(t, call.StartTime.IsZero(), "StartTime should be set")
	assert.False(t, call.EndTime.IsZero(), "EndTime should be set")
	assert.Equal(t, CapturedCall{
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
		ResponseID:   "chatcmpl-sse1",
		InputTokens:  15,
		OutputTokens: 3,
		FinishReason: "stop",
	}, call)
}

func TestProxySSEStreamingCallbackFires(t *testing.T) {
	t.Parallel()

	fakeServer := httptest.NewTLSServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		w.Header().Set("Content-Type", "text/event-stream")
		w.WriteHeader(http.StatusOK)

		flusher := w.(http.Flusher)
		fmt.Fprintf(w, "data: %s\n\n", `{"id":"chatcmpl-cb1","model":"gpt-4o","choices":[{"delta":{"content":"Hi"},"finish_reason":null}]}`)
		flusher.Flush()
		fmt.Fprintf(w, "data: %s\n\n", `{"id":"chatcmpl-cb1","model":"gpt-4o","choices":[{"delta":{},"finish_reason":"stop"}]}`)
		flusher.Flush()
		fmt.Fprintf(w, "data: [DONE]\n\n")
		flusher.Flush()
	}))
	defer fakeServer.Close()

	serverURL, _ := url.Parse(fakeServer.URL)

	var mu sync.Mutex
	var callbackCalls []CapturedCall

	p := startTestProxyWithOpts(t, ProxyOptions{
		Hosts: []string{serverURL.Hostname()},
		OnCall: func(c CapturedCall) {
			mu.Lock()
			callbackCalls = append(callbackCalls, c)
			mu.Unlock()
		},
	})
	client := proxyClient(t, p)

	resp, err := client.Post(
		fakeServer.URL+"/v1/chat/completions",
		"application/json",
		strings.NewReader(`{"model":"gpt-4o","stream":true}`),
	)
	if err != nil {
		t.Fatalf("POST: %v", err)
	}
	io.ReadAll(resp.Body)
	resp.Body.Close()

	mu.Lock()
	defer mu.Unlock()
	if len(callbackCalls) != 1 {
		t.Fatalf("expected 1 callback call from SSE stream, got %d", len(callbackCalls))
	}
	assert.Equal(t, "chatcmpl-cb1", callbackCalls[0].ResponseID)
}

func TestProxyToolCallsNonStreaming(t *testing.T) {
	t.Parallel()

	openaiResp := `{
		"id": "chatcmpl-tc1",
		"model": "gpt-4o-2024-05-13",
		"usage": {"prompt_tokens": 100, "completion_tokens": 50},
		"choices": [{
			"finish_reason": "tool_calls",
			"message": {
				"role": "assistant",
				"tool_calls": [
					{"id": "call_1", "type": "function", "function": {"name": "read_file", "arguments": "{}"}},
					{"id": "call_2", "type": "function", "function": {"name": "grep", "arguments": "{}"}}
				]
			}
		}]
	}`

	fakeServer := httptest.NewTLSServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		w.Header().Set("Content-Type", "application/json")
		w.WriteHeader(http.StatusOK)
		w.Write([]byte(openaiResp))
	}))
	defer fakeServer.Close()

	serverURL, _ := url.Parse(fakeServer.URL)
	p := startTestProxy(t, []string{serverURL.Hostname()})
	client := proxyClient(t, p)

	reqBody := `{"model":"gpt-4o","messages":[{"role":"user","content":"search for TODOs"}]}`
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

	calls := p.Calls()
	if len(calls) != 1 {
		t.Fatalf("expected 1 call, got %d", len(calls))
	}

	call := calls[0]
	assert.Greater(t, call.Duration, time.Duration(0))
	assert.False(t, call.StartTime.IsZero(), "StartTime should be set")
	assert.False(t, call.EndTime.IsZero(), "EndTime should be set")
	assert.Equal(t, CapturedCall{
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
		ResponseID:   "chatcmpl-tc1",
		InputTokens:  100,
		OutputTokens: 50,
		FinishReason: "tool_calls",
		ToolCalls:    []string{"read_file", "grep"},
		ToolCallArgs: []string{"{}", "{}"},
	}, call)
}

func TestProxySSEStreamingToolCalls(t *testing.T) {
	t.Parallel()

	sseChunks := []string{
		`{"id":"chatcmpl-sse-tc","model":"gpt-4o","choices":[{"delta":{"role":"assistant"},"finish_reason":null}]}`,
		`{"id":"chatcmpl-sse-tc","model":"gpt-4o","choices":[{"delta":{"tool_calls":[{"index":0,"function":{"name":"read_file"}}]},"finish_reason":null}]}`,
		`{"id":"chatcmpl-sse-tc","model":"gpt-4o","choices":[{"delta":{"tool_calls":[{"index":0,"function":{"arguments":"{}"}}]},"finish_reason":null}]}`,
		`{"id":"chatcmpl-sse-tc","model":"gpt-4o","choices":[{"delta":{"tool_calls":[{"index":1,"function":{"name":"grep"}}]},"finish_reason":null}]}`,
		`{"id":"chatcmpl-sse-tc","model":"gpt-4o","choices":[{"delta":{},"finish_reason":"tool_calls"}],"usage":{"prompt_tokens":80,"completion_tokens":30}}`,
	}

	fakeServer := httptest.NewTLSServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		w.Header().Set("Content-Type", "text/event-stream")
		w.Header().Set("Cache-Control", "no-cache")
		w.WriteHeader(http.StatusOK)

		flusher := w.(http.Flusher)
		for _, chunk := range sseChunks {
			fmt.Fprintf(w, "data: %s\n\n", chunk)
			flusher.Flush()
		}
		fmt.Fprintf(w, "data: [DONE]\n\n")
		flusher.Flush()
	}))
	defer fakeServer.Close()

	serverURL, _ := url.Parse(fakeServer.URL)
	p := startTestProxy(t, []string{serverURL.Hostname()})
	client := proxyClient(t, p)

	resp, err := client.Post(
		fakeServer.URL+"/v1/chat/completions",
		"application/json",
		strings.NewReader(`{"model":"gpt-4o","messages":[{"role":"user","content":"find TODOs"}],"stream":true}`),
	)
	if err != nil {
		t.Fatalf("POST: %v", err)
	}
	defer resp.Body.Close()
	io.ReadAll(resp.Body)

	calls := p.Calls()
	if len(calls) != 1 {
		t.Fatalf("expected 1 call, got %d", len(calls))
	}

	call := calls[0]
	assert.Greater(t, call.Duration, time.Duration(0))
	assert.False(t, call.StartTime.IsZero(), "StartTime should be set")
	assert.False(t, call.EndTime.IsZero(), "EndTime should be set")
	assert.Equal(t, CapturedCall{
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
		ResponseID:   "chatcmpl-sse-tc",
		InputTokens:  80,
		OutputTokens: 30,
		FinishReason: "tool_calls",
		ToolCalls:    []string{"read_file", "grep"},
		ToolCallArgs: []string{"{}", ""},
	}, call)
}

func TestProxyErrorResponse(t *testing.T) {
	t.Parallel()

	errorResp := `{"error":{"message":"Rate limit exceeded","type":"rate_limit_exceeded","code":"rate_limit"}}`

	fakeServer := httptest.NewTLSServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		w.Header().Set("Content-Type", "application/json")
		w.WriteHeader(http.StatusTooManyRequests)
		w.Write([]byte(errorResp))
	}))
	defer fakeServer.Close()

	serverURL, _ := url.Parse(fakeServer.URL)
	p := startTestProxy(t, []string{serverURL.Hostname()})
	client := proxyClient(t, p)

	reqBody := `{"model":"gpt-4o","messages":[{"role":"user","content":"hello"}]}`
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

	calls := p.Calls()
	if len(calls) != 1 {
		t.Fatalf("expected 1 call, got %d", len(calls))
	}

	call := calls[0]
	assert.Greater(t, call.Duration, time.Duration(0))
	assert.False(t, call.StartTime.IsZero(), "StartTime should be set")
	assert.False(t, call.EndTime.IsZero(), "EndTime should be set")
	assert.Equal(t, CapturedCall{
		Method:       "POST",
		Host:         serverURL.Hostname(),
		Path:         "/v1/chat/completions",
		StatusCode:   429,
		Duration:     call.Duration,
		StartTime:    call.StartTime,
		EndTime:      call.EndTime,
		Sequence:     1,
		IsLLM:        true,
		RequestModel: "gpt-4o",
		ErrorMessage: "rate_limit_exceeded",
	}, call)
}

func TestIsLLMHostExactMatch(t *testing.T) {
	t.Parallel()

	p, err := NewProxy(ProxyOptions{
		Hosts: []string{"api.openai.com", "api.anthropic.com"},
	})
	if err != nil {
		t.Fatalf("NewProxy: %v", err)
	}

	assert.True(t, p.isLLMHost("api.openai.com"))
	assert.True(t, p.isLLMHost("api.anthropic.com"))
	assert.False(t, p.isLLMHost("github.com"))
	assert.False(t, p.isLLMHost(""))
}

func TestIsLLMHostWildcardSuffix(t *testing.T) {
	t.Parallel()

	p, err := NewProxy(ProxyOptions{
		Hosts:     []string{"api.openai.com"},
		Wildcards: []string{".githubcopilot.com"},
	})
	if err != nil {
		t.Fatalf("NewProxy: %v", err)
	}

	assert.True(t, p.isLLMHost("api.openai.com"))
	assert.True(t, p.isLLMHost("models.individual.githubcopilot.com"))
	assert.True(t, p.isLLMHost("anything.githubcopilot.com"))
	assert.False(t, p.isLLMHost("githubcopilot.com")) // no leading dot
	assert.False(t, p.isLLMHost("example.com"))
}

// --- Anthropic dispatch tests ---
//
// Full proxy E2E tests (fake HTTPS server → proxy → client) exercise the OpenAI
// default path because test servers bind to 127.0.0.1 and InferProvider("127.0.0.1")
// returns "". To test Anthropic dispatch without DNS tricks, we call the unexported
// parseNonStreamingResponse and parseSSEChunk helpers directly.

func TestParseNonStreamingResponseAnthropicDispatch(t *testing.T) {
	t.Parallel()

	anthropicResp := `{
		"id": "msg_dispatch1",
		"type": "message",
		"model": "claude-3-5-sonnet-20241022",
		"stop_reason": "end_turn",
		"usage": {"input_tokens": 50, "output_tokens": 20},
		"content": [{"type": "text", "text": "Hello from Claude!"}]
	}`

	call := CapturedCall{
		Provider:   ProviderAnthropic,
		StatusCode: 200,
	}
	parseNonStreamingResponse(&call, []byte(anthropicResp))

	assert.Equal(t, "msg_dispatch1", call.ResponseID)
	assert.Equal(t, "claude-3-5-sonnet-20241022", call.Model)
	assert.Equal(t, int64(50), call.InputTokens)
	assert.Equal(t, int64(20), call.OutputTokens)
	assert.Equal(t, "end_turn", call.FinishReason)
	assert.Equal(t, "", call.ErrorMessage)
}

func TestParseNonStreamingResponseAnthropicToolUseDispatch(t *testing.T) {
	t.Parallel()

	anthropicResp := `{
		"id": "msg_dispatch_tools",
		"type": "message",
		"model": "claude-3-5-sonnet-20241022",
		"stop_reason": "tool_use",
		"usage": {"input_tokens": 100, "output_tokens": 60},
		"content": [
			{"type": "text", "text": "Let me look that up."},
			{"type": "tool_use", "id": "toolu_01", "name": "search", "input": {"query": "weather"}}
		]
	}`

	call := CapturedCall{
		Provider:   ProviderAnthropic,
		StatusCode: 200,
	}
	parseNonStreamingResponse(&call, []byte(anthropicResp))

	assert.Equal(t, "msg_dispatch_tools", call.ResponseID)
	assert.Equal(t, "claude-3-5-sonnet-20241022", call.Model)
	assert.Equal(t, int64(100), call.InputTokens)
	assert.Equal(t, int64(60), call.OutputTokens)
	assert.Equal(t, "tool_use", call.FinishReason)
	assert.Equal(t, []string{"search"}, call.ToolCalls)
	assert.Equal(t, []string{`{"query": "weather"}`}, call.ToolCallArgs)
}

func TestParseNonStreamingResponseAnthropicErrorDispatch(t *testing.T) {
	t.Parallel()

	errorResp := `{"type": "error", "error": {"type": "overloaded_error", "message": "Overloaded"}}`

	call := CapturedCall{
		Provider:   ProviderAnthropic,
		StatusCode: 529,
	}
	parseNonStreamingResponse(&call, []byte(errorResp))

	assert.Equal(t, "overloaded_error", call.ErrorMessage)
	assert.Equal(t, "", call.Model)
	assert.Equal(t, "", call.ResponseID)
}

func TestParseSSEChunkAnthropicDispatch(t *testing.T) {
	t.Parallel()

	chunk := parseSSEChunk(ProviderAnthropic, &proxy.SSEEvent{
		Event: "message_start",
		Data:  `{"type": "message_start", "message": {"id": "msg_sse_dispatch", "model": "claude-3-5-sonnet-20241022", "usage": {"input_tokens": 30}}}`,
	})

	assert.Equal(t, "msg_sse_dispatch", chunk.ID)
	assert.Equal(t, "claude-3-5-sonnet-20241022", chunk.Model)
	assert.Equal(t, int64(30), chunk.InputTokens)
}

func TestParseSSEChunkAnthropicMessageDeltaDispatch(t *testing.T) {
	t.Parallel()

	chunk := parseSSEChunk(ProviderAnthropic, &proxy.SSEEvent{
		Event: "message_delta",
		Data:  `{"type": "message_delta", "delta": {"stop_reason": "end_turn"}, "usage": {"output_tokens": 15}}`,
	})

	assert.Equal(t, "end_turn", chunk.FinishReason)
	assert.Equal(t, int64(15), chunk.OutputTokens)
}

func TestParseSSEChunkAnthropicToolUseDispatch(t *testing.T) {
	t.Parallel()

	// content_block_start with tool_use
	startChunk := parseSSEChunk(ProviderAnthropic, &proxy.SSEEvent{
		Event: "content_block_start",
		Data:  `{"type": "content_block_start", "index": 1, "content_block": {"type": "tool_use", "id": "toolu_01", "name": "get_weather", "input": {}}}`,
	})
	assert.Equal(t, []string{"get_weather"}, startChunk.ToolCalls)

	// content_block_delta with input_json_delta
	deltaChunk := parseSSEChunk(ProviderAnthropic, &proxy.SSEEvent{
		Event: "content_block_delta",
		Data:  `{"type": "content_block_delta", "index": 1, "delta": {"type": "input_json_delta", "partial_json": "{\"city\":"}}`,
	})
	assert.Len(t, deltaChunk.rawToolCalls, 1)
	assert.Equal(t, 1, deltaChunk.rawToolCalls[0].Index)
}

// TestProxyAnthropicSSEFullStream tests the full SSE merge path for Anthropic
// by calling parseSSEChunk for a realistic stream of events, then MergeSSEChunks.
func TestProxyAnthropicSSEFullStream(t *testing.T) {
	t.Parallel()

	events := []struct {
		eventType string
		data      string
	}{
		{"message_start", `{"type":"message_start","message":{"id":"msg_full","model":"claude-3-5-sonnet-20241022","usage":{"input_tokens":40}}}`},
		{"content_block_start", `{"type":"content_block_start","index":0,"content_block":{"type":"text","text":""}}`},
		{"content_block_delta", `{"type":"content_block_delta","index":0,"delta":{"type":"text_delta","text":"Hello"}}`},
		{"content_block_delta", `{"type":"content_block_delta","index":0,"delta":{"type":"text_delta","text":" world"}}`},
		{"content_block_start", `{"type":"content_block_start","index":1,"content_block":{"type":"tool_use","id":"toolu_01","name":"read_file","input":{}}}`},
		{"content_block_delta", `{"type":"content_block_delta","index":1,"delta":{"type":"input_json_delta","partial_json":"{\"path\":"}}`},
		{"content_block_delta", `{"type":"content_block_delta","index":1,"delta":{"type":"input_json_delta","partial_json":"\"foo.txt\"}"}}`},
		{"message_delta", `{"type":"message_delta","delta":{"stop_reason":"tool_use"},"usage":{"output_tokens":25}}`},
		{"message_stop", `{"type":"message_stop"}`},
	}

	var chunks []ParsedResponse
	for _, e := range events {
		chunk := parseSSEChunk(ProviderAnthropic, &proxy.SSEEvent{
			Event: e.eventType,
			Data:  e.data,
		})
		// Same accumulation logic as the proxy: skip empty chunks.
		if chunk.ID != "" || chunk.Model != "" || chunk.FinishReason != "" ||
			chunk.InputTokens > 0 || chunk.OutputTokens > 0 ||
			len(chunk.ToolCalls) > 0 || len(chunk.rawToolCalls) > 0 {
			chunks = append(chunks, chunk)
		}
	}

	merged := MergeSSEChunks(chunks)
	assert.Equal(t, ParsedResponse{
		ID:           "msg_full",
		Model:        "claude-3-5-sonnet-20241022",
		InputTokens:  40,
		OutputTokens: 25,
		FinishReason: "tool_use",
		ToolCalls:    []string{"read_file"},
		ToolCallArgs: []string{`{"path":"foo.txt"}`},
	}, merged)
}
