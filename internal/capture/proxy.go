// proxy.go — MITM proxy with host allowlist and SSE flow tracking.
//
// Design: Two separate mutexes guard captured calls (mu) and in-flight
// SSE streams (sseMu) so non-streaming responses never block SSE
// accumulation. The sseFlows map is keyed by go-mitmproxy's flow UUID.
//
// Why: go-mitmproxy's Response() hook never fires for SSE streams
// (Content-Type: text/event-stream). We use SSEStart/SSEMessage/SSEEnd
// instead and merge chunks into a single Call at stream end.
//
// Design: All addon hooks that can panic (SSE hooks, Response) have
// defer/recover so a parsing bug cannot crash the proxy goroutine.
// SSEStart uses a committed flag to call hookWg.Done() only when the
// flow was not stored in sseFlows (otherwise SSEEnd handles it).
// Wait() has a hard timeout as a safety net against unbalanced hookWg.
package capture

import (
	"context"
	"fmt"
	"io"
	"net"
	"net/http"
	"os"
	"strings"
	"sync"
	"sync/atomic"
	"time"

	"github.com/lqqyt2423/go-mitmproxy/cert"
	"github.com/lqqyt2423/go-mitmproxy/proxy"
	uuid "github.com/satori/go.uuid"
)

// DefaultLLMHosts returns the default LLM API host allowlist.
func DefaultLLMHosts() []string {
	return []string{
		"api.openai.com",
		"api.anthropic.com",
		"api.githubcopilot.com",
		"copilot-proxy.githubusercontent.com",
	}
}

// DefaultLLMWildcards returns suffix patterns matched against the host (e.g. ".githubcopilot.com").
func DefaultLLMWildcards() []string {
	return []string{
		".githubcopilot.com",
	}
}

// ProxyOptions configures the MITM proxy.
type ProxyOptions struct {
	// Hosts to intercept (exact match). If nil, DefaultLLMHosts() is used.
	Hosts []string
	// Wildcard suffixes to intercept. If nil, DefaultLLMWildcards() is used.
	Wildcards []string
	// OnCall is invoked for each intercepted call. May be nil.
	OnCall func(Call)
	// SslInsecure disables upstream TLS certificate verification.
	// Default (false) verifies upstream certs against the system CA bundle.
	SslInsecure bool
}

// sseFlow tracks an in-flight SSE stream for a single proxy flow.
// Chunks accumulate here until SSEEnd fires, then we merge and emit
// a single Call. The flow is keyed by go-mitmproxy's UUID
// and protected by Proxy.sseMu.
type sseFlow struct {
	startTime    time.Time
	method       string
	host         string
	path         string
	statusCode   int
	kind         CallKind
	provider     string
	requestModel string
	chunks       []ParsedResponse
}

// Proxy wraps go-mitmproxy and captures HTTP calls to known LLM API hosts.
type Proxy struct {
	opts      ProxyOptions
	inner     *proxy.Proxy
	addr      string // actual "host:port" after Start
	hostSet   map[string]bool
	wildcards []string

	callSeq atomic.Uint64

	mu    sync.Mutex
	calls []Call

	sseMu    sync.Mutex
	sseFlows map[uuid.UUID]*sseFlow

	// hookWg tracks in-flight addon hook chains (Response and SSE flows).
	// go-mitmproxy's Shutdown doesn't wait for hijacked HTTPS connections'
	// hooks to complete. Call Wait() after Stop() to drain pending callbacks.
	hookWg sync.WaitGroup
}

// NewProxy creates a new MITM proxy. It does not start listening.
func NewProxy(opts ProxyOptions) (*Proxy, error) {
	if opts.Hosts == nil {
		opts.Hosts = DefaultLLMHosts()
	}
	if opts.Wildcards == nil {
		opts.Wildcards = DefaultLLMWildcards()
	}

	// Pre-bind to discover a free port, then release it.
	// go-mitmproxy's Start() doesn't expose the bound address.
	ln, err := net.Listen("tcp", "127.0.0.1:0")
	if err != nil {
		return nil, fmt.Errorf("pre-bind: %w", err)
	}
	addr := ln.Addr().String()
	_ = ln.Close()

	proxyOpts := &proxy.Options{
		Addr: addr,
		// Default (SslInsecure=false) verifies upstream TLS certs against the
		// system CA bundle. Tests pass SslInsecure=true because httptest servers
		// use self-signed certs not in the system trust store.
		SslInsecure: opts.SslInsecure,
		NewCaFunc:   cert.NewSelfSignCAMemory,
	}

	inner, err := proxy.NewProxy(proxyOpts)
	if err != nil {
		return nil, fmt.Errorf("create proxy: %w", err)
	}

	hostSet := make(map[string]bool, len(opts.Hosts))
	for _, h := range opts.Hosts {
		hostSet[h] = true
	}

	p := &Proxy{
		opts:      opts,
		inner:     inner,
		addr:      addr,
		hostSet:   hostSet,
		sseFlows:  make(map[uuid.UUID]*sseFlow),
		wildcards: opts.Wildcards,
	}

	// MITM all HTTPS traffic so we capture metadata for non-LLM calls too.
	inner.SetShouldInterceptRule(func(req *http.Request) bool {
		return true
	})

	inner.AddAddon(p)

	return p, nil
}

// Addr returns the proxy's listen address (e.g. "127.0.0.1:52341").
func (p *Proxy) Addr() string { return p.addr }

// Inner returns the underlying go-mitmproxy Proxy (for cert access etc.).
func (p *Proxy) Inner() *proxy.Proxy { return p.inner }

func (p *Proxy) isLLMHost(host string) bool {
	if p.hostSet[host] {
		return true
	}
	for _, suffix := range p.wildcards {
		if strings.HasSuffix(host, suffix) {
			return true
		}
	}
	return false
}

// Start begins listening. Blocks until Stop is called.
func (p *Proxy) Start() error {
	return p.inner.Start()
}

// Stop gracefully shuts down the proxy with a 2-second timeout.
// Call Wait() after Stop() to ensure all in-flight hooks have completed.
func (p *Proxy) Stop() error {
	ctx, cancel := context.WithTimeout(context.Background(), 2*time.Second)
	defer cancel()
	return p.inner.Shutdown(ctx)
}

// Wait blocks until all in-flight addon hooks (Response, SSE flows)
// have completed their OnCall callbacks. Call after Stop() to ensure
// no callbacks are missed before reading stats or shutting down.
//
// Why: A hard 5-second timeout prevents hanging forever if a hook panic
// causes an unbalanced hookWg (Add without Done). This shouldn't happen
// with the recovery defers, but the timeout is a safety net.
func (p *Proxy) Wait() {
	done := make(chan struct{})
	go func() {
		p.hookWg.Wait()
		close(done)
	}()
	select {
	case <-done:
	case <-time.After(5 * time.Second):
		fmt.Fprintf(os.Stderr, "[aitrace] warning: timed out waiting for in-flight hooks\n")
	}
}

// recoverHook logs a recovered panic from a proxy addon hook.
// Use as: defer recoverHook("SSEMessage")
func recoverHook(hook string) {
	if r := recover(); r != nil {
		fmt.Fprintf(os.Stderr, "[aitrace] recovered panic in %s: %v\n", hook, r)
	}
}

// WaitForProxy blocks until a TCP connection to addr succeeds or ctx is
// cancelled. Used to detect when go-mitmproxy's Start() has bound the port,
// since the library exposes no ready signal.
func WaitForProxy(ctx context.Context, addr string) error {
	for {
		conn, err := net.DialTimeout("tcp", addr, 20*time.Millisecond)
		if err == nil {
			_ = conn.Close()
			return nil
		}
		select {
		case <-ctx.Done():
			return fmt.Errorf("wait for proxy: %w", ctx.Err())
		case <-time.After(5 * time.Millisecond):
		}
	}
}

func stripPort(host string) string {
	if h, _, err := net.SplitHostPort(host); err == nil {
		return h
	}
	return host
}

// callKind returns KindLLM for known LLM API hosts, KindHTTP otherwise.
func (p *Proxy) callKind(host string) CallKind {
	if p.isLLMHost(host) {
		return KindLLM
	}
	return KindHTTP
}

func (p *Proxy) recordCall(call Call) {
	if call.Kind == "" {
		panic("call recorded without Kind")
	}
	p.mu.Lock()
	p.calls = append(p.calls, call)
	p.mu.Unlock()

	if p.opts.OnCall != nil {
		p.opts.OnCall(call)
	}
}

// Calls returns a copy of all captured calls.
func (p *Proxy) Calls() []Call {
	p.mu.Lock()
	defer p.mu.Unlock()
	out := make([]Call, len(p.calls))
	copy(out, p.calls)
	return out
}

// Ensure Proxy implements proxy.Addon at compile time.
var _ proxy.Addon = (*Proxy)(nil)

func (p *Proxy) ClientConnected(*proxy.ClientConn)                            {}
func (p *Proxy) ClientDisconnected(*proxy.ClientConn)                         {}
func (p *Proxy) ServerConnected(*proxy.ConnContext)                           {}
func (p *Proxy) ServerDisconnected(*proxy.ConnContext)                        {}
func (p *Proxy) TlsEstablishedServer(*proxy.ConnContext)                      {}
func (p *Proxy) Requestheaders(*proxy.Flow)                                   {}
func (p *Proxy) Request(*proxy.Flow)                                          {}
func (p *Proxy) Responseheaders(*proxy.Flow)                                  {}
func (p *Proxy) StreamRequestModifier(f *proxy.Flow, r io.Reader) io.Reader   { return r }
func (p *Proxy) StreamResponseModifier(f *proxy.Flow, r io.Reader) io.Reader  { return r }
func (p *Proxy) AccessProxyServer(req *http.Request, res http.ResponseWriter) {}
func (p *Proxy) WebSocketStart(*proxy.Flow)                                   {}
func (p *Proxy) WebSocketMessage(*proxy.Flow)                                 {}
func (p *Proxy) WebSocketEnd(*proxy.Flow)                                     {}
func (p *Proxy) SSEStart(f *proxy.Flow) {
	// Track this SSE flow so Wait() blocks until SSEEnd fires and OnCall completes.
	p.hookWg.Add(1)

	// Why: committed tracks whether the flow was stored in sseFlows. If a panic
	// occurs before that, SSEEnd will never fire for this flow, so the defer
	// must call hookWg.Done() to avoid hanging Wait(). Once stored, SSEEnd
	// handles the Done().
	committed := false
	defer func() {
		if r := recover(); r != nil {
			fmt.Fprintf(os.Stderr, "[aitrace] recovered panic in SSEStart: %v\n", r)
			if !committed {
				p.hookWg.Done()
			}
		}
	}()

	host := stripPort(f.Request.URL.Host)
	kind := p.callKind(host)

	sf := &sseFlow{
		startTime:  f.StartTime,
		method:     f.Request.Method,
		host:       host,
		path:       f.Request.URL.Path,
		statusCode: f.Response.StatusCode,
		kind:       kind,
	}

	if kind == KindLLM {
		sf.provider = InferProvider(host)
		if f.Request.Body != nil {
			sf.requestModel = ParseRequestModel(f.Request.Body)
		}
	}

	p.sseMu.Lock()
	p.sseFlows[f.Id] = sf
	p.sseMu.Unlock()
	committed = true
}

func (p *Proxy) SSEMessage(f *proxy.Flow) {
	defer recoverHook("SSEMessage")

	p.sseMu.Lock()
	sf, ok := p.sseFlows[f.Id]
	p.sseMu.Unlock()
	if !ok {
		return
	}

	// Only parse SSE chunks for LLM hosts.
	if sf.kind != KindLLM {
		return
	}

	events := f.SSE.Events
	if len(events) == 0 {
		return
	}
	latest := events[len(events)-1]
	chunk := parseSSEChunk(sf.provider, latest)
	// Only accumulate chunks that have useful data.
	if chunk.ID != "" || chunk.Model != "" || chunk.FinishReason != "" ||
		chunk.InputTokens > 0 || chunk.OutputTokens > 0 ||
		len(chunk.ToolCalls) > 0 || len(chunk.rawToolCalls) > 0 {
		p.sseMu.Lock()
		sf.chunks = append(sf.chunks, chunk)
		p.sseMu.Unlock()
	}
}

func (p *Proxy) SSEEnd(f *proxy.Flow) {
	defer p.hookWg.Done()
	defer recoverHook("SSEEnd")

	p.sseMu.Lock()
	sf, ok := p.sseFlows[f.Id]
	if ok {
		delete(p.sseFlows, f.Id)
	}
	p.sseMu.Unlock()
	if !ok {
		return
	}

	// Safe to read sf without lock — delete(sseFlows) above ensures no other
	// hook can access this flow.
	endTime := time.Now()
	call := Call{
		Kind:       sf.kind,
		Method:     sf.method,
		Host:       sf.host,
		Path:       sf.path,
		StatusCode: sf.statusCode,
		Duration:   endTime.Sub(sf.startTime),
		StartTime:  sf.startTime,
		EndTime:    endTime,
		Sequence:   int(p.callSeq.Add(1)),
	}

	if sf.kind == KindLLM {
		merged := MergeSSEChunks(sf.chunks)
		call.Provider = sf.provider
		call.RequestModel = sf.requestModel
		call.Model = merged.Model
		call.ResponseID = merged.ID
		call.InputTokens = merged.InputTokens
		call.OutputTokens = merged.OutputTokens
		call.FinishReason = merged.FinishReason
		call.ToolCalls = merged.ToolCalls
		call.ToolCallArgs = merged.ToolCallArgs
		call.CacheReadTokens = merged.CacheReadTokens
		call.CacheWriteTokens = merged.CacheWriteTokens
	}

	p.recordCall(call)
}

// RequestError fires when an HTTP request fails after the CONNECT tunnel
// is established (dial error, TLS failure, response body read error, etc.).
// We emit a Call so the failure appears in the trace instead of silently
// disappearing.
func (p *Proxy) RequestError(f *proxy.Flow, err error) {
	p.hookWg.Add(1)
	defer p.hookWg.Done()
	defer recoverHook("RequestError")

	host := stripPort(f.Request.URL.Host)
	kind := p.callKind(host)

	endTime := time.Now()
	call := Call{
		Kind:         kind,
		Method:       f.Request.Method,
		Host:         host,
		Path:         f.Request.URL.Path,
		Duration:     endTime.Sub(f.StartTime),
		StartTime:    f.StartTime,
		EndTime:      endTime,
		Sequence:     int(p.callSeq.Add(1)),
		ErrorMessage: shortError(err),
	}

	// f.Response is non-nil when the response headers arrived but the body
	// read failed. Use the status code when available.
	if f.Response != nil {
		call.StatusCode = f.Response.StatusCode
	}

	if kind == KindLLM {
		call.Provider = InferProvider(host)
		if f.Request.Body != nil {
			call.RequestModel = ParseRequestModel(f.Request.Body)
		}
	}

	p.recordCall(call)
}

// HTTPConnectError fires when the HTTPS CONNECT tunnel setup fails
// (DNS resolution, TLS handshake, connection refused, etc.). We emit a
// Call so the failure appears in the trace.
func (p *Proxy) HTTPConnectError(f *proxy.Flow, err error) {
	p.hookWg.Add(1)
	defer p.hookWg.Done()
	defer recoverHook("HTTPConnectError")

	// CONNECT requests use "host:port" as the URL path. The host may
	// already include a port, which stripPort handles.
	host := stripPort(f.Request.URL.Host)
	if host == "" {
		// Fallback: some CONNECT flows put the host in the URL path.
		host = stripPort(f.Request.URL.Path)
	}
	kind := p.callKind(host)

	endTime := time.Now()
	call := Call{
		Kind:         kind,
		Method:       f.Request.Method,
		Host:         host,
		Path:         "",
		Duration:     endTime.Sub(f.StartTime),
		StartTime:    f.StartTime,
		EndTime:      endTime,
		Sequence:     int(p.callSeq.Add(1)),
		ErrorMessage: shortError(err),
	}

	if kind == KindLLM {
		call.Provider = InferProvider(host)
	}

	p.recordCall(call)
}

// shortError extracts a concise description from an error, stripping
// verbose Go wrappers (e.g. "dial tcp: lookup foo.com: no such host"
// → "no such host").
func shortError(err error) string {
	if err == nil {
		return ""
	}
	msg := err.Error()

	// Go net errors often wrap the root cause after the last colon.
	// "dial tcp 1.2.3.4:443: connection refused" → "connection refused"
	// "dial tcp: lookup foo.com: no such host" → "no such host"
	if i := strings.LastIndex(msg, ": "); i >= 0 {
		return strings.TrimSpace(msg[i+2:])
	}
	return msg
}

func (p *Proxy) Response(f *proxy.Flow) {
	p.hookWg.Add(1)
	defer p.hookWg.Done()
	defer recoverHook("Response")

	host := stripPort(f.Request.URL.Host)
	kind := p.callKind(host)

	endTime := time.Now()
	call := Call{
		Kind:       kind,
		Method:     f.Request.Method,
		Host:       host,
		Path:       f.Request.URL.Path,
		StatusCode: f.Response.StatusCode,
		Duration:   endTime.Sub(f.StartTime),
		StartTime:  f.StartTime,
		EndTime:    endTime,
		Sequence:   int(p.callSeq.Add(1)),
	}

	// Only parse request/response bodies for LLM API hosts.
	if kind == KindLLM {
		call.Provider = InferProvider(host)

		if f.Request.Body != nil {
			call.RequestModel = ParseRequestModel(f.Request.Body)
		}

		// TODO: log decode errors at debug level once we have structured logging.
		if f.Response.Body != nil {
			respBody, err := f.Response.DecodedBody()
			if err == nil {
				parseNonStreamingResponse(&call, respBody)
			}
		}
	}

	p.recordCall(call)
}

// parseNonStreamingResponse dispatches to the provider-specific response parser
// and populates the Call fields. Separate code paths per provider rather
// than conditionals inside a single parser.
func parseNonStreamingResponse(call *Call, body []byte) {
	if call.StatusCode >= 400 {
		switch call.Provider {
		case ProviderAnthropic:
			call.ErrorMessage = ParseAnthropicError(body)
		default:
			call.ErrorMessage = ParseOpenAIError(body)
		}
		return
	}

	var pr ParsedResponse
	switch call.Provider {
	case ProviderAnthropic:
		pr = ParseAnthropicResponse(body)
	default:
		pr = ParseOpenAIResponse(body)
	}

	call.Model = pr.Model
	call.ResponseID = pr.ID
	call.InputTokens = pr.InputTokens
	call.OutputTokens = pr.OutputTokens
	call.FinishReason = pr.FinishReason
	call.ToolCalls = pr.ToolCalls
	call.ToolCallArgs = pr.ToolCallArgs
	call.CacheReadTokens = pr.CacheReadTokens
	call.CacheWriteTokens = pr.CacheWriteTokens
}

// parseSSEChunk dispatches to the provider-specific SSE chunk parser.
func parseSSEChunk(provider string, event *proxy.SSEEvent) ParsedResponse {
	switch provider {
	case ProviderAnthropic:
		return ParseAnthropicSSEChunk(event.Event, event.Data)
	default:
		return ParseOpenAISSEChunk(event.Data)
	}
}
