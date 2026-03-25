// main.go — CLI entry point and flag dispatch.
//
// aitrace run [flags] -- <command> [args...]
//
// Two execution paths: runTerminal (default) prints call summaries to
// stderr. runOTel (--otel) additionally emits OpenTelemetry spans to
// an OTLP collector. Both paths accept --json to replace human-readable
// output with JSONL (one object per call + a summary object at the end).
package main

import (
	"cmp"
	"context"
	"crypto/tls"
	"crypto/x509"
	"encoding/json"
	"errors"
	"flag"
	"fmt"
	"io"
	"net/http"
	"os"
	"path/filepath"
	"runtime/debug"
	"slices"
	"strings"
	"sync"
	"time"

	"github.com/rnaudi/aitrace/internal/capture"
	"github.com/rnaudi/aitrace/internal/cert"
	"github.com/rnaudi/aitrace/internal/cost"
	"github.com/rnaudi/aitrace/internal/envtags"
	"github.com/rnaudi/aitrace/internal/run"
	"github.com/rnaudi/aitrace/internal/trace"

	"go.opentelemetry.io/otel/attribute"
	semconv "go.opentelemetry.io/otel/semconv/v1.26.0"
)

const usage = `Usage:
    aitrace run [flags] -- <command> [args...]
    aitrace doctor
    aitrace --version
    aitrace --help

Flags:
    --json      Output JSONL to stderr (one object per call + summary)
    --otel      Also emit OpenTelemetry spans (requires OTLP collector)

Examples:
    aitrace run -- cursor .
    aitrace run -- aider "fix the login bug"
    aitrace run -- python my_agent.py
    aitrace run --json -- python my_agent.py 2>calls.jsonl
    aitrace run --otel -- aider "explain auth"
    aitrace doctor

aitrace intercepts HTTPS calls to LLM providers (OpenAI, Anthropic,
GitHub Copilot) and shows you every request: model, tokens, latency.
No code changes needed.`

// Version can be set at link time to override debug.ReadBuildInfo when
// building with goreleaser. It should look like "v0.1.0".
var Version string

func main() {
	if len(os.Args) < 2 {
		fmt.Fprintf(os.Stderr, "%s\n", usage)
		os.Exit(1)
	}

	switch os.Args[1] {
	case "run":
		runCmd(os.Args[2:])
	case "doctor":
		doctorCmd()
	case "--version", "-version":
		fmt.Println(version())
	case "--help", "-help", "-h":
		fmt.Fprintf(os.Stderr, "%s\n", usage)
	default:
		fmt.Fprintf(os.Stderr, "%s\n", usage)
		os.Exit(1)
	}
}

func runCmd(args []string) {
	flags := flag.NewFlagSet("run", flag.ContinueOnError)
	flags.Usage = func() { fmt.Fprintf(os.Stderr, "%s\n", usage) }

	jsonFlag := flags.Bool("json", false, "output JSONL to stderr")
	otelFlag := flags.Bool("otel", false, "emit OpenTelemetry spans")

	if err := flags.Parse(args); err != nil {
		os.Exit(1)
	}

	childArgs := flags.Args()
	if len(childArgs) == 0 {
		flags.Usage()
		os.Exit(1)
	}

	envs := envtags.Detect()

	if *otelFlag {
		runOTel(childArgs[0], childArgs[1:], *jsonFlag, envs)
	} else {
		runTerminal(childArgs[0], childArgs[1:], *jsonFlag, envs)
	}
}

// runTerminal runs the child process and prints call summaries to stderr.
// No OpenTelemetry, no external dependencies — standalone terminal output.
// When jsonOutput is true, JSONL replaces human-readable output.
func runTerminal(cmd string, args []string, jsonOutput bool, envs []envtags.Env) {
	sessionStart := time.Now()
	var stats sessionStats
	p, err := capture.NewProxy(capture.ProxyOptions{
		OnCall: func(c capture.CapturedCall) {
			stats.record(c)
			if jsonOutput {
				fmt.Fprintln(os.Stderr, formatCallJSON(c))
			} else if c.IsLLM {
				fmt.Fprintln(os.Stderr, formatCallLine(c))
			}
		},
	})
	if err != nil {
		fmt.Fprintf(os.Stderr, "[aitrace] create proxy: %v\n", err)
		os.Exit(1)
	}

	proxyErrCh := make(chan error, 1)
	go func() {
		proxyErrCh <- p.Start()
	}()

	waitCtx, waitCancel := context.WithTimeout(context.Background(), 2*time.Second)
	defer waitCancel()
	if err := capture.WaitForProxy(waitCtx, p.Addr()); err != nil {
		fmt.Fprintf(os.Stderr, "[aitrace] proxy not ready: %v\n", err)
		os.Exit(1)
	}

	select {
	case err := <-proxyErrCh:
		fmt.Fprintf(os.Stderr, "[aitrace] start proxy: %v\n", err)
		os.Exit(1)
	default:
	}

	fmt.Fprintf(os.Stderr, "[aitrace] listening on %s\n", p.Addr())

	if line := envSummary(envs); line != "" {
		fmt.Fprintf(os.Stderr, "[aitrace] environment: %s\n", line)
	}

	if os.Getenv("SSH_CONNECTION") != "" {
		fmt.Fprintf(os.Stderr, "[aitrace] SSH session detected\n")
	}

	caCert := p.Inner().GetCertificate()
	tmpDir, err := os.MkdirTemp("", "aitrace-*")
	if err != nil {
		fmt.Fprintf(os.Stderr, "[aitrace] create temp dir: %v\n", err)
		os.Exit(1)
	}
	defer os.RemoveAll(tmpDir)

	pemPath, err := cert.WriteCombinedPEM(&caCert, tmpDir)
	if err != nil {
		fmt.Fprintf(os.Stderr, "[aitrace] write combined PEM: %v\n", err)
		os.Exit(1)
	}

	exitCode, _, err := run.Child(run.Options{
		ProxyAddr:       p.Addr(),
		CombinedPEMPath: pemPath,
		Command:         cmd,
		Args:            args,
	})
	if err != nil {
		fmt.Fprintf(os.Stderr, "[aitrace] %v\n", err)
	}

	if err := p.Stop(); err != nil {
		fmt.Fprintf(os.Stderr, "[aitrace] proxy shutdown: %v\n", err)
	}
	p.Wait()

	if jsonOutput {
		stats.printJSON(os.Stderr, sessionStart, envs)
	} else {
		stats.print(os.Stderr, sessionStart, "")
	}
	os.Exit(exitCode)
}

// runOTel runs the child process, prints call summaries to stderr, and
// emits OpenTelemetry spans to an OTLP collector (default localhost:4317).
// When jsonOutput is true, JSONL replaces human-readable output.
func runOTel(cmd string, args []string, jsonOutput bool, envs []envtags.Env) {
	ctx := context.Background()
	tp, err := trace.NewTracerProvider(ctx, trace.TracerOptions{
		ServiceName:   filepath.Base(cmd),
		ResourceAttrs: envResourceAttrs(envs),
	})
	if err != nil {
		fmt.Fprintf(os.Stderr, "[aitrace] create tracer provider: %v\n", err)
		os.Exit(1)
	}
	defer func() {
		shutdownCtx, cancel := context.WithTimeout(context.Background(), 5*time.Second)
		defer cancel()
		if err := tp.Shutdown(shutdownCtx); err != nil {
			fmt.Fprintf(os.Stderr, "[aitrace] tracer shutdown: %v\n", err)
		}
	}()

	ctx, sessionSpan := trace.StartSessionSpan(ctx, tp, filepath.Base(cmd))

	sessionStart := time.Now()
	var stats sessionStats
	p, err := capture.NewProxy(capture.ProxyOptions{
		OnCall: func(c capture.CapturedCall) {
			stats.record(c)
			if jsonOutput {
				fmt.Fprintln(os.Stderr, formatCallJSON(c))
			} else if c.IsLLM {
				fmt.Fprintln(os.Stderr, formatCallLine(c))
			}
			trace.EmitSpan(ctx, tp, c)
		},
	})
	if err != nil {
		fmt.Fprintf(os.Stderr, "[aitrace] create proxy: %v\n", err)
		os.Exit(1)
	}

	proxyErrCh := make(chan error, 1)
	go func() {
		proxyErrCh <- p.Start()
	}()

	waitCtx, waitCancel := context.WithTimeout(ctx, 2*time.Second)
	defer waitCancel()
	if err := capture.WaitForProxy(waitCtx, p.Addr()); err != nil {
		fmt.Fprintf(os.Stderr, "[aitrace] proxy not ready: %v\n", err)
		os.Exit(1)
	}

	select {
	case err := <-proxyErrCh:
		fmt.Fprintf(os.Stderr, "[aitrace] start proxy: %v\n", err)
		os.Exit(1)
	default:
	}

	fmt.Fprintf(os.Stderr, "[aitrace] listening on %s\n", p.Addr())

	if line := envSummary(envs); line != "" {
		fmt.Fprintf(os.Stderr, "[aitrace] environment: %s\n", line)
	}

	if os.Getenv("SSH_CONNECTION") != "" {
		fmt.Fprintf(os.Stderr, "[aitrace] SSH session detected\n")
	}

	caCert := p.Inner().GetCertificate()
	tmpDir, err := os.MkdirTemp("", "aitrace-*")
	if err != nil {
		fmt.Fprintf(os.Stderr, "[aitrace] create temp dir: %v\n", err)
		os.Exit(1)
	}
	defer os.RemoveAll(tmpDir)

	pemPath, err := cert.WriteCombinedPEM(&caCert, tmpDir)
	if err != nil {
		fmt.Fprintf(os.Stderr, "[aitrace] write combined PEM: %v\n", err)
		os.Exit(1)
	}

	exitCode, childPID, err := run.Child(run.Options{
		ProxyAddr:       p.Addr(),
		CombinedPEMPath: pemPath,
		Command:         cmd,
		Args:            args,
	})
	if err != nil {
		fmt.Fprintf(os.Stderr, "[aitrace] %v\n", err)
	}

	sessionSpan.SetName(fmt.Sprintf("%s (pid %d)", filepath.Base(cmd), childPID))

	sessionCmd := cmd
	if len(args) > 0 {
		sessionCmd += " " + strings.Join(args, " ")
	}
	// Jaeger truncates attribute values at ~1 KB.
	const maxCmdAttrLen = 500
	if len(sessionCmd) > maxCmdAttrLen {
		sessionCmd = sessionCmd[:maxCmdAttrLen-3] + "..."
	}
	sessionSpan.SetAttributes(
		attribute.String("aitrace.session.command", sessionCmd),
		attribute.Int("aitrace.session.pid", childPID),
	)

	if err := p.Stop(); err != nil {
		fmt.Fprintf(os.Stderr, "[aitrace] proxy shutdown: %v\n", err)
	}
	p.Wait()

	sessionSpan.End()
	tp.ForceFlush(ctx)

	traceID := sessionSpan.SpanContext().TraceID()
	if jsonOutput {
		stats.printJSON(os.Stderr, sessionStart, envs)
	} else {
		stats.print(os.Stderr, sessionStart, "http://localhost:16686/trace/"+traceID.String())
	}
	os.Exit(exitCode)
}

// sessionStats accumulates metrics across calls for the exit summary.
type sessionStats struct {
	mu               sync.Mutex
	llmCalls         int
	httpCalls        int
	inputTokens      int64
	outputTokens     int64
	cacheReadTokens  int64
	cacheWriteTokens int64
	totalCost        float64 // accumulated USD cost across all LLM calls
	llmDuration      time.Duration
	models           map[string]int // model name -> call count
	hosts            map[string]int // non-LLM host -> call count
}

func (s *sessionStats) record(c capture.CapturedCall) {
	s.mu.Lock()
	defer s.mu.Unlock()

	if c.IsLLM {
		s.llmCalls++
		s.inputTokens += c.InputTokens
		s.outputTokens += c.OutputTokens
		s.cacheReadTokens += c.CacheReadTokens
		s.cacheWriteTokens += c.CacheWriteTokens
		s.totalCost += cost.Calculate(
			c.Provider, c.EffectiveModel(),
			c.InputTokens, c.OutputTokens,
			c.CacheReadTokens, c.CacheWriteTokens,
		)
		s.llmDuration += c.Duration

		if model := c.EffectiveModel(); model != "" {
			if s.models == nil {
				s.models = make(map[string]int)
			}
			s.models[model]++
		}
	} else {
		s.httpCalls++
		if c.Host != "" {
			if s.hosts == nil {
				s.hosts = make(map[string]int)
			}
			s.hosts[c.Host]++
		}
	}
}

func (s *sessionStats) print(w io.Writer, sessionStart time.Time, traceURL string) {
	s.mu.Lock()
	defer s.mu.Unlock()

	totalCalls := s.llmCalls + s.httpCalls
	if totalCalls == 0 {
		fmt.Fprintf(w, "[aitrace] no calls captured\n")
		return
	}

	elapsed := time.Since(sessionStart).Round(time.Millisecond)

	fmt.Fprintf(w, "[aitrace] ---\n")

	line := fmt.Sprintf("[aitrace] %d calls", s.llmCalls)
	if s.inputTokens > 0 || s.outputTokens > 0 {
		line += fmt.Sprintf(" %d/%d tok", s.inputTokens, s.outputTokens)
	}
	if costStr := cost.FormatUSD(s.totalCost); costStr != "" {
		line += " " + costStr
	}
	line += " " + elapsed.String() + " elapsed"
	fmt.Fprintln(w, line)

	if len(s.models) > 0 {
		fmt.Fprintf(w, "[aitrace] models: %s\n", formatCountsDesc(s.models))
	}

	if traceURL != "" {
		fmt.Fprintf(w, "[aitrace] trace:  %s\n", traceURL)
	}
}

// formatCountsDesc formats a key->count map as "foo (10) bar (2)",
// sorted by count descending.
func formatCountsDesc(m map[string]int) string {
	type kc struct {
		key   string
		count int
	}
	sorted := make([]kc, 0, len(m))
	for k, c := range m {
		sorted = append(sorted, kc{k, c})
	}
	slices.SortFunc(sorted, func(a, b kc) int {
		return cmp.Compare(b.count, a.count)
	})

	var b strings.Builder
	for i, item := range sorted {
		if i > 0 {
			b.WriteByte(' ')
		}
		fmt.Fprintf(&b, "%s (%d)", item.key, item.count)
	}
	return b.String()
}

// Flag thresholds for the human-readable call line.
const (
	largeTokThreshold     = 10_000
	longDurationThreshold = 10 * time.Second
)

// formatCallLine builds the one-line stderr summary for an LLM call.
//
// Normal: [aitrace] #1 gpt-4o | tok: 500 in / 100 out | 1.2s | $0.007
// Tools:  [aitrace] #2 claude-opus-4.6 | tok: 19,659 in / 106 out | 3.8s | $0.100 | tools: skill,glob !large
// Cache:  [aitrace] #3 claude-3-5-sonnet | tok: 2,000 in / 500 out | 1.2s | $0.009 !cache
// Error:  [aitrace] #4 gpt-4o | 429 rate_limit_exceeded | 0.2s !error
// Flags:  !cache (prompt caching active), !large (total tokens > 10k),
//
//	!long (duration > 10s), !error (status >= 400)
func formatCallLine(c capture.CapturedCall) string {
	model := c.EffectiveModel()
	if model == "" {
		model = "(unknown)"
	}

	line := fmt.Sprintf("[aitrace] #%d %s", c.Sequence, model)

	if c.StatusCode >= 400 {
		// Error path: show status + message instead of tokens.
		errPart := fmt.Sprintf("%d", c.StatusCode)
		if c.ErrorMessage != "" {
			errPart += " " + c.ErrorMessage
		}
		line += " | " + errPart
	} else if c.InputTokens > 0 || c.OutputTokens > 0 {
		line += fmt.Sprintf(" | tok: %s in / %s out",
			formatTokenCount(c.InputTokens), formatTokenCount(c.OutputTokens))
	}

	line += " | " + c.Duration.Round(time.Millisecond).String()

	// Cost — shown after duration when the model is in the pricing table.
	callCost := cost.Calculate(
		c.Provider, c.EffectiveModel(),
		c.InputTokens, c.OutputTokens,
		c.CacheReadTokens, c.CacheWriteTokens,
	)
	if costStr := cost.FormatUSD(callCost); costStr != "" {
		line += " | " + costStr
	}

	if len(c.ToolCalls) > 0 {
		line += " | tools: " + strings.Join(c.ToolCalls, ",")
	}

	// Flags — appended at end.
	if c.CacheReadTokens > 0 || c.CacheWriteTokens > 0 {
		line += " !cache"
	}
	if c.InputTokens+c.OutputTokens > largeTokThreshold {
		line += " !large"
	}
	if c.Duration > longDurationThreshold {
		line += " !long"
	}
	if c.StatusCode >= 400 {
		line += " !error"
	}

	return line
}

// formatTokenCount formats an integer with comma separators (e.g. 19659 → "19,659").
func formatTokenCount(n int64) string {
	if n < 0 {
		return "-" + formatTokenCount(-n)
	}
	s := fmt.Sprintf("%d", n)
	if len(s) <= 3 {
		return s
	}

	var b strings.Builder
	remainder := len(s) % 3
	if remainder > 0 {
		b.WriteString(s[:remainder])
	}
	for i := remainder; i < len(s); i += 3 {
		if b.Len() > 0 {
			b.WriteByte(',')
		}
		b.WriteString(s[i : i+3])
	}
	return b.String()
}

// --- JSONL output ---
//
// --json replaces the human-readable terminal output with one JSON object
// per line to stderr. Each captured call emits a {"type":"call",...} line.
// After the child exits, a {"type":"summary",...} line is emitted with
// session totals. All calls (LLM + non-LLM) are included.

// jsonCall is the per-call JSONL object. Separate from CapturedCall to
// control the exact wire format (snake_case, duration as milliseconds,
// timestamps as RFC3339) without coupling to the internal type.
type jsonCall struct {
	Type         string   `json:"type"`
	Sequence     int      `json:"sequence"`
	Method       string   `json:"method"`
	Host         string   `json:"host"`
	Path         string   `json:"path"`
	StatusCode   int      `json:"status_code"`
	DurationMs   int64    `json:"duration_ms"`
	StartTime    string   `json:"start_time"`
	EndTime      string   `json:"end_time"`
	IsLLM        bool     `json:"is_llm"`
	Provider     string   `json:"provider,omitempty"`
	RequestModel string   `json:"request_model,omitempty"`
	Model        string   `json:"model,omitempty"`
	ResponseID   string   `json:"response_id,omitempty"`
	InputTokens  int64    `json:"input_tokens,omitempty"`
	OutputTokens int64    `json:"output_tokens,omitempty"`
	FinishReason string   `json:"finish_reason,omitempty"`
	ToolCalls    []string `json:"tool_calls,omitempty"`
	ToolCallArgs []string `json:"tool_call_args,omitempty"`
	ErrorMessage string   `json:"error_message,omitempty"`

	CacheReadTokens  int64   `json:"cache_read_tokens,omitempty"`
	CacheWriteTokens int64   `json:"cache_write_tokens,omitempty"`
	Cost             float64 `json:"cost,omitempty"`
}

// formatCallJSON converts a CapturedCall to a single JSON line.
func formatCallJSON(c capture.CapturedCall) string {
	jc := jsonCall{
		Type:         "call",
		Sequence:     c.Sequence,
		Method:       c.Method,
		Host:         c.Host,
		Path:         c.Path,
		StatusCode:   c.StatusCode,
		DurationMs:   c.Duration.Milliseconds(),
		StartTime:    c.StartTime.UTC().Format(time.RFC3339Nano),
		EndTime:      c.EndTime.UTC().Format(time.RFC3339Nano),
		IsLLM:        c.IsLLM,
		Provider:     c.Provider,
		RequestModel: c.RequestModel,
		Model:        c.Model,
		ResponseID:   c.ResponseID,
		InputTokens:  c.InputTokens,
		OutputTokens: c.OutputTokens,
		FinishReason: c.FinishReason,
		ToolCalls:    c.ToolCalls,
		ToolCallArgs: c.ToolCallArgs,
		ErrorMessage: c.ErrorMessage,

		CacheReadTokens:  c.CacheReadTokens,
		CacheWriteTokens: c.CacheWriteTokens,
		Cost: cost.Calculate(
			c.Provider, c.EffectiveModel(),
			c.InputTokens, c.OutputTokens,
			c.CacheReadTokens, c.CacheWriteTokens,
		),
	}
	b, _ := json.Marshal(jc)
	return string(b)
}

// jsonSummary is the session summary JSONL object, emitted once after
// the child exits.
type jsonSummary struct {
	Type              string            `json:"type"`
	LLMCalls          int               `json:"llm_calls"`
	HTTPCalls         int               `json:"http_calls"`
	InputTokens       int64             `json:"input_tokens"`
	OutputTokens      int64             `json:"output_tokens"`
	CacheReadTokens   int64             `json:"cache_read_tokens,omitempty"`
	CacheWriteTokens  int64             `json:"cache_write_tokens,omitempty"`
	TotalCost         float64           `json:"total_cost,omitempty"`
	LLMDurationMs     int64             `json:"llm_duration_ms"`
	SessionDurationMs int64             `json:"session_duration_ms"`
	Models            map[string]int    `json:"models,omitempty"`
	Environment       map[string]string `json:"environment,omitempty"`
}

func (s *sessionStats) printJSON(w io.Writer, sessionStart time.Time, envs []envtags.Env) {
	s.mu.Lock()
	defer s.mu.Unlock()

	js := jsonSummary{
		Type:              "summary",
		LLMCalls:          s.llmCalls,
		HTTPCalls:         s.httpCalls,
		InputTokens:       s.inputTokens,
		OutputTokens:      s.outputTokens,
		CacheReadTokens:   s.cacheReadTokens,
		CacheWriteTokens:  s.cacheWriteTokens,
		TotalCost:         s.totalCost,
		LLMDurationMs:     s.llmDuration.Milliseconds(),
		SessionDurationMs: time.Since(sessionStart).Milliseconds(),
		Models:            s.models,
		Environment:       envJSONTags(envs),
	}
	b, _ := json.Marshal(js)
	fmt.Fprintln(w, string(b))
}

// envSummary returns a short comma-separated label string for terminal
// output, or "" if no environments were detected.
func envSummary(envs []envtags.Env) string {
	if len(envs) == 0 {
		return ""
	}
	labels := make([]string, len(envs))
	for i, e := range envs {
		labels[i] = e.Kind.String()
	}
	return strings.Join(labels, ", ")
}

// envJSONTags flattens detected environments into a single map for JSONL
// output. Returns nil when nothing was detected.
func envJSONTags(envs []envtags.Env) map[string]string {
	if len(envs) == 0 {
		return nil
	}
	m := make(map[string]string)
	for _, e := range envs {
		m[e.Kind.String()] = ""
		for k, v := range e.Tags {
			m[k] = v
		}
	}
	return m
}

// envResourceAttrs converts detected environments to OTel resource
// attributes using semantic conventions where available.
func envResourceAttrs(envs []envtags.Env) []attribute.KeyValue {
	var attrs []attribute.KeyValue
	for _, e := range envs {
		switch e.Kind {
		case envtags.GithubActions, envtags.GitLabCI, envtags.CircleCI,
			envtags.Jenkins, envtags.Buildkite, envtags.TravisCI:
			attrs = append(attrs, attribute.String("aitrace.ci.system", e.Kind.String()))
			attrs = append(attrs, semconv.DeploymentEnvironment("ci"))
		case envtags.Kubernetes:
			attrs = append(attrs, attribute.Bool("aitrace.kubernetes", true))
		default:
			// Cloud providers: AWS, GCP, Azure, Fly, Railway.
			attrs = append(attrs, semconv.CloudProviderKey.String(e.Kind.String()))
		}
		for k, v := range e.Tags {
			attrs = append(attrs, attribute.String("aitrace."+k, v))
		}
	}
	return attrs
}

func version() string {
	if Version != "" {
		return Version
	}
	if buildInfo, ok := debug.ReadBuildInfo(); ok {
		return buildInfo.Main.Version
	}
	return "(unknown)"
}

// --- doctor subcommand ---
//
// aitrace doctor probes each known LLM API host with a TLS handshake to
// verify reachability and certificate trust from the current machine.
// No proxy is started — this checks the system CA bundle directly.

// probeResult holds the outcome of a TLS probe to a single host.
type probeResult struct {
	Host    string
	OK      bool
	Elapsed time.Duration
	Err     error
}

const doctorTimeout = 10 * time.Second

// probeHost performs an HTTPS HEAD request to the host using the given
// HTTP client and returns the result. The client should be configured
// with the desired TLS settings (system roots, custom CA pool, etc.).
func probeHost(client *http.Client, host string) probeResult {
	start := time.Now()
	resp, err := client.Head("https://" + host + "/")
	elapsed := time.Since(start).Round(time.Millisecond)
	if err != nil {
		return probeResult{Host: host, OK: false, Elapsed: elapsed, Err: err}
	}
	resp.Body.Close()
	return probeResult{Host: host, OK: true, Elapsed: elapsed}
}

// classifyProbeError returns a short human-readable diagnosis and a
// suggested action for the given probe error.
func classifyProbeError(err error) (diagnosis, hint string) {
	if certErr, ok := errors.AsType[*tls.CertificateVerificationError](err); ok {
		return certErr.Err.Error(), "your network may use a corporate proxy with custom CAs"
	}
	if _, ok := errors.AsType[x509.UnknownAuthorityError](err); ok {
		return "x509: certificate signed by unknown authority",
			"your network may use a corporate proxy with custom CAs"
	}
	return err.Error(), "check DNS, firewall, or network connectivity"
}

func doctorCmd() {
	bundlePath, err := cert.FindSystemCertBundle()
	if err != nil {
		fmt.Fprintf(os.Stderr, "[aitrace] %v\n", err)
		fmt.Fprintf(os.Stderr, "[aitrace] set SSL_CERT_FILE to the path of your CA bundle\n")
		os.Exit(1)
	}
	fmt.Fprintf(os.Stderr, "[aitrace] system CA bundle: %s\n", bundlePath)

	// Use system roots (RootCAs: nil means Go's default system trust store).
	client := &http.Client{
		Timeout: doctorTimeout,
		Transport: &http.Transport{
			TLSClientConfig: &tls.Config{
				MinVersion: tls.VersionTLS12,
			},
		},
	}

	hosts := capture.DefaultLLMHosts()
	results := make([]probeResult, len(hosts))
	for i, host := range hosts {
		results[i] = probeHost(client, host)
	}

	// Find longest host name for alignment.
	maxLen := 0
	for _, h := range hosts {
		if len(h) > maxLen {
			maxLen = len(h)
		}
	}

	failed := 0
	for _, r := range results {
		pad := strings.Repeat(" ", maxLen-len(r.Host)+2)
		if r.OK {
			fmt.Fprintf(os.Stderr, "[aitrace] %s%sok (%s)\n", r.Host, pad, r.Elapsed)
		} else {
			failed++
			diagnosis, hint := classifyProbeError(r.Err)
			fmt.Fprintf(os.Stderr, "[aitrace] %s%sFAIL (%s)\n", r.Host, pad, r.Elapsed)
			fmt.Fprintf(os.Stderr, "[aitrace]   %s\n", diagnosis)
			fmt.Fprintf(os.Stderr, "[aitrace]   %s\n", hint)
		}
	}

	if failed == 0 {
		fmt.Fprintf(os.Stderr, "[aitrace] all hosts reachable\n")
	} else {
		fmt.Fprintf(os.Stderr, "[aitrace] %d host(s) failed — check network and certificate configuration\n", failed)
		os.Exit(1)
	}
}
