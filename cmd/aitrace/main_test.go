package main

import (
	"bytes"
	"crypto/tls"
	"encoding/json"
	"net/http"
	"net/http/httptest"
	"strings"
	"testing"
	"time"

	"github.com/rnaudi/aitrace/internal/capture"
	"github.com/rnaudi/aitrace/internal/envtags"

	"github.com/stretchr/testify/assert"
	"go.opentelemetry.io/otel/attribute"
	semconv "go.opentelemetry.io/otel/semconv/v1.26.0"
)

func TestSessionStatsRecordLLM(t *testing.T) {
	t.Parallel()

	var stats sessionStats
	stats.record(capture.CapturedCall{
		IsLLM:        true,
		Model:        "gpt-4o",
		InputTokens:  100,
		OutputTokens: 50,
		Duration:     2 * time.Second,
	})

	stats.mu.Lock()
	defer stats.mu.Unlock()
	assert.Equal(t, 1, stats.llmCalls)
	assert.Equal(t, 0, stats.httpCalls)
	assert.Equal(t, int64(100), stats.inputTokens)
	assert.Equal(t, int64(50), stats.outputTokens)
	assert.Equal(t, int64(0), stats.cacheReadTokens)
	assert.Equal(t, int64(0), stats.cacheWriteTokens)
	assert.Equal(t, 2*time.Second, stats.llmDuration)
	assert.Equal(t, map[string]int{"gpt-4o": 1}, stats.models)
}

func TestSessionStatsRecordLLMWithCacheTokens(t *testing.T) {
	t.Parallel()

	var stats sessionStats
	stats.record(capture.CapturedCall{
		IsLLM:            true,
		Model:            "claude-3-5-sonnet-20241022",
		InputTokens:      2000,
		OutputTokens:     500,
		CacheReadTokens:  800,
		CacheWriteTokens: 1500,
		Duration:         1 * time.Second,
	})
	stats.record(capture.CapturedCall{
		IsLLM:           true,
		Model:           "claude-3-5-sonnet-20241022",
		InputTokens:     2000,
		OutputTokens:    300,
		CacheReadTokens: 1800,
		Duration:        1 * time.Second,
	})

	stats.mu.Lock()
	defer stats.mu.Unlock()
	assert.Equal(t, 2, stats.llmCalls)
	assert.Equal(t, int64(4000), stats.inputTokens)
	assert.Equal(t, int64(800), stats.outputTokens)
	assert.Equal(t, int64(2600), stats.cacheReadTokens)
	assert.Equal(t, int64(1500), stats.cacheWriteTokens)
}

func TestSessionStatsRecordHTTP(t *testing.T) {
	t.Parallel()

	var stats sessionStats
	stats.record(capture.CapturedCall{
		IsLLM: false,
		Host:  "github.com",
	})

	stats.mu.Lock()
	defer stats.mu.Unlock()
	assert.Equal(t, 0, stats.llmCalls)
	assert.Equal(t, 1, stats.httpCalls)
	assert.Equal(t, map[string]int{"github.com": 1}, stats.hosts)
}

func TestSessionStatsRecordMixed(t *testing.T) {
	t.Parallel()

	var stats sessionStats
	stats.record(capture.CapturedCall{
		IsLLM:        true,
		Model:        "gpt-4o",
		InputTokens:  100,
		OutputTokens: 50,
		Duration:     1 * time.Second,
	})
	stats.record(capture.CapturedCall{
		IsLLM:        true,
		Model:        "gpt-4o",
		InputTokens:  200,
		OutputTokens: 100,
		Duration:     2 * time.Second,
	})
	stats.record(capture.CapturedCall{
		IsLLM:        true,
		Model:        "claude-opus-4.6",
		InputTokens:  500,
		OutputTokens: 200,
		Duration:     3 * time.Second,
	})
	stats.record(capture.CapturedCall{IsLLM: false, Host: "github.com"})
	stats.record(capture.CapturedCall{IsLLM: false, Host: "github.com"})
	stats.record(capture.CapturedCall{IsLLM: false, Host: "models.dev"})

	stats.mu.Lock()
	defer stats.mu.Unlock()
	assert.Equal(t, 3, stats.llmCalls)
	assert.Equal(t, 3, stats.httpCalls)
	assert.Equal(t, int64(800), stats.inputTokens)
	assert.Equal(t, int64(350), stats.outputTokens)
	assert.Equal(t, 6*time.Second, stats.llmDuration)
	assert.Equal(t, map[string]int{"gpt-4o": 2, "claude-opus-4.6": 1}, stats.models)
	assert.Equal(t, map[string]int{"github.com": 2, "models.dev": 1}, stats.hosts)
}

func TestSessionStatsRecordLLMFallsBackToRequestModel(t *testing.T) {
	t.Parallel()

	var stats sessionStats
	stats.record(capture.CapturedCall{
		IsLLM:        true,
		RequestModel: "gpt-4o",
	})

	stats.mu.Lock()
	defer stats.mu.Unlock()
	assert.Equal(t, map[string]int{"gpt-4o": 1}, stats.models)
}

func TestSessionStatsRecordLLMNoModel(t *testing.T) {
	t.Parallel()

	var stats sessionStats
	stats.record(capture.CapturedCall{IsLLM: true})

	stats.mu.Lock()
	defer stats.mu.Unlock()
	assert.Equal(t, 1, stats.llmCalls)
	assert.Nil(t, stats.models)
}

func TestSessionStatsRecordHTTPEmptyHost(t *testing.T) {
	t.Parallel()

	var stats sessionStats
	stats.record(capture.CapturedCall{IsLLM: false, Host: ""})

	stats.mu.Lock()
	defer stats.mu.Unlock()
	assert.Equal(t, 1, stats.httpCalls)
	assert.Nil(t, stats.hosts)
}

func TestSessionStatsPrintNoCalls(t *testing.T) {
	t.Parallel()

	var stats sessionStats

	var buf bytes.Buffer
	stats.print(&buf, time.Now(), "")
	assert.Equal(t, "[aitrace] no calls captured\n", buf.String())
}

func TestFormatCountsDesc(t *testing.T) {
	t.Parallel()
	result := formatCountsDesc(map[string]int{
		"gpt-4o":          5,
		"claude-opus-4.6": 10,
		"gpt-3.5":         2,
	})
	assert.Equal(t, "claude-opus-4.6 (10) gpt-4o (5) gpt-3.5 (2)", result)
}

func TestFormatCountsDescSingleEntry(t *testing.T) {
	t.Parallel()
	result := formatCountsDesc(map[string]int{"gpt-4o": 3})
	assert.Equal(t, "gpt-4o (3)", result)
}

func TestFormatCountsDescEmpty(t *testing.T) {
	t.Parallel()
	assert.Equal(t, "", formatCountsDesc(map[string]int{}))
	assert.Equal(t, "", formatCountsDesc(nil))
}

func checkFormatCallLine(t *testing.T, c capture.CapturedCall, want string) {
	t.Helper()
	got := formatCallLine(c)
	assert.Equal(t, want, got)
}

func TestFormatCallLineLLMWithToolCalls(t *testing.T) {
	t.Parallel()
	checkFormatCallLine(t, capture.CapturedCall{
		Sequence:     1,
		IsLLM:        true,
		Model:        "claude-opus-4.6",
		StatusCode:   200,
		FinishReason: "tool_calls",
		ToolCalls:    []string{"read_file", "grep"},
		InputTokens:  10568,
		OutputTokens: 268,
		Duration:     4500 * time.Millisecond,
	}, "[aitrace] #1 claude-opus-4.6 | tok: 10,568 in / 268 out | 4.5s | tools: read_file,grep !large")
}

func TestFormatCallLineLLMWithError(t *testing.T) {
	t.Parallel()
	checkFormatCallLine(t, capture.CapturedCall{
		Sequence:     4,
		IsLLM:        true,
		Model:        "claude-opus-4.6",
		StatusCode:   429,
		ErrorMessage: "rate_limit_exceeded",
		Duration:     200 * time.Millisecond,
	}, "[aitrace] #4 claude-opus-4.6 | 429 rate_limit_exceeded | 200ms !error")
}

func TestFormatCallLineLLMWithStop(t *testing.T) {
	t.Parallel()
	checkFormatCallLine(t, capture.CapturedCall{
		Sequence:     2,
		IsLLM:        true,
		Model:        "gpt-4o",
		StatusCode:   200,
		FinishReason: "stop",
		InputTokens:  500,
		OutputTokens: 100,
		Duration:     1200 * time.Millisecond,
	}, "[aitrace] #2 gpt-4o | tok: 500 in / 100 out | 1.2s")
}

func TestFormatCallLineLLMNoTokens(t *testing.T) {
	t.Parallel()
	checkFormatCallLine(t, capture.CapturedCall{
		Sequence:   3,
		IsLLM:      true,
		Model:      "gpt-4o",
		StatusCode: 200,
		Duration:   800 * time.Millisecond,
	}, "[aitrace] #3 gpt-4o | 800ms")
}

func TestFormatCallLineLLMNoModel(t *testing.T) {
	t.Parallel()
	checkFormatCallLine(t, capture.CapturedCall{
		Sequence:     1,
		IsLLM:        true,
		StatusCode:   200,
		FinishReason: "stop",
		InputTokens:  500,
		OutputTokens: 100,
		Duration:     500 * time.Millisecond,
	}, "[aitrace] #1 (unknown) | tok: 500 in / 100 out | 500ms")
}

func TestFormatCallLineLargeFlag(t *testing.T) {
	t.Parallel()
	checkFormatCallLine(t, capture.CapturedCall{
		Sequence:     1,
		IsLLM:        true,
		Model:        "claude-opus-4.6",
		StatusCode:   200,
		InputTokens:  9000,
		OutputTokens: 1500,
		Duration:     3 * time.Second,
	}, "[aitrace] #1 claude-opus-4.6 | tok: 9,000 in / 1,500 out | 3s !large")
}

func TestFormatCallLineLongFlag(t *testing.T) {
	t.Parallel()
	checkFormatCallLine(t, capture.CapturedCall{
		Sequence:     1,
		IsLLM:        true,
		Model:        "gpt-4o",
		StatusCode:   200,
		InputTokens:  500,
		OutputTokens: 100,
		Duration:     15 * time.Second,
	}, "[aitrace] #1 gpt-4o | tok: 500 in / 100 out | 15s !long")
}

func TestFormatCallLineLargeAndLongFlags(t *testing.T) {
	t.Parallel()
	checkFormatCallLine(t, capture.CapturedCall{
		Sequence:     2,
		IsLLM:        true,
		Model:        "claude-opus-4.6",
		StatusCode:   200,
		InputTokens:  45000,
		OutputTokens: 2500,
		Duration:     25 * time.Second,
		ToolCalls:    []string{"read_file", "grep"},
	}, "[aitrace] #2 claude-opus-4.6 | tok: 45,000 in / 2,500 out | 25s | tools: read_file,grep !large !long")
}

func TestFormatCallLineNoFlags(t *testing.T) {
	t.Parallel()
	checkFormatCallLine(t, capture.CapturedCall{
		Sequence:     1,
		IsLLM:        true,
		Model:        "gpt-4o",
		StatusCode:   200,
		InputTokens:  500,
		OutputTokens: 100,
		Duration:     1 * time.Second,
	}, "[aitrace] #1 gpt-4o | tok: 500 in / 100 out | 1s")
}

func TestFormatCallLineCacheFlag(t *testing.T) {
	t.Parallel()
	checkFormatCallLine(t, capture.CapturedCall{
		Sequence:        1,
		IsLLM:           true,
		Model:           "claude-3-5-sonnet-20241022",
		StatusCode:      200,
		InputTokens:     2000,
		OutputTokens:    500,
		CacheReadTokens: 1500,
		Duration:        1200 * time.Millisecond,
	}, "[aitrace] #1 claude-3-5-sonnet-20241022 | tok: 2,000 in / 500 out | 1.2s !cache")
}

func TestFormatCallLineCacheWriteFlag(t *testing.T) {
	t.Parallel()
	// CacheWriteTokens alone should also trigger !cache.
	checkFormatCallLine(t, capture.CapturedCall{
		Sequence:         1,
		IsLLM:            true,
		Model:            "claude-3-5-sonnet-20241022",
		StatusCode:       200,
		InputTokens:      2000,
		OutputTokens:     500,
		CacheWriteTokens: 1500,
		Duration:         1200 * time.Millisecond,
	}, "[aitrace] #1 claude-3-5-sonnet-20241022 | tok: 2,000 in / 500 out | 1.2s !cache")
}

func TestFormatCallLineNoCacheFlag(t *testing.T) {
	t.Parallel()
	// No cache tokens — no !cache flag.
	checkFormatCallLine(t, capture.CapturedCall{
		Sequence:     1,
		IsLLM:        true,
		Model:        "gpt-4o",
		StatusCode:   200,
		InputTokens:  500,
		OutputTokens: 100,
		Duration:     1 * time.Second,
	}, "[aitrace] #1 gpt-4o | tok: 500 in / 100 out | 1s")
}

func TestFormatCallLineCacheAndLargeFlags(t *testing.T) {
	t.Parallel()
	// Both !cache and !large should appear.
	checkFormatCallLine(t, capture.CapturedCall{
		Sequence:        1,
		IsLLM:           true,
		Model:           "claude-3-5-sonnet-20241022",
		StatusCode:      200,
		InputTokens:     12000,
		OutputTokens:    1000,
		CacheReadTokens: 10000,
		Duration:        3 * time.Second,
	}, "[aitrace] #1 claude-3-5-sonnet-20241022 | tok: 12,000 in / 1,000 out | 3s !cache !large")
}

func TestFormatCallLineErrorNoMessage(t *testing.T) {
	t.Parallel()
	checkFormatCallLine(t, capture.CapturedCall{
		Sequence:   1,
		IsLLM:      true,
		Model:      "gpt-4o",
		StatusCode: 500,
		Duration:   100 * time.Millisecond,
	}, "[aitrace] #1 gpt-4o | 500 | 100ms !error")
}

func TestFormatTokenCount(t *testing.T) {
	t.Parallel()
	assert.Equal(t, "0", formatTokenCount(0))
	assert.Equal(t, "1", formatTokenCount(1))
	assert.Equal(t, "100", formatTokenCount(100))
	assert.Equal(t, "999", formatTokenCount(999))
	assert.Equal(t, "1,000", formatTokenCount(1000))
	assert.Equal(t, "9,999", formatTokenCount(9999))
	assert.Equal(t, "10,000", formatTokenCount(10000))
	assert.Equal(t, "19,659", formatTokenCount(19659))
	assert.Equal(t, "100,000", formatTokenCount(100000))
	assert.Equal(t, "1,000,000", formatTokenCount(1000000))
}

// --- JSON output tests ---

// checkFormatCallJSON verifies that formatCallJSON produces valid JSON that
// round-trips through jsonCall and matches the expected struct.
func checkFormatCallJSON(t *testing.T, c capture.CapturedCall, want jsonCall) {
	t.Helper()
	line := formatCallJSON(c)

	var got jsonCall
	if err := json.Unmarshal([]byte(line), &got); err != nil {
		t.Fatalf("formatCallJSON produced invalid JSON: %v\nline: %s", err, line)
	}
	assert.Equal(t, want, got)
}

func TestFormatCallJSONLLMCall(t *testing.T) {
	t.Parallel()
	start := time.Date(2026, 3, 24, 12, 0, 0, 0, time.UTC)
	end := start.Add(1432 * time.Millisecond)

	checkFormatCallJSON(t, capture.CapturedCall{
		Sequence:     1,
		Method:       "POST",
		Host:         "api.openai.com",
		Path:         "/v1/chat/completions",
		StatusCode:   200,
		Duration:     1432 * time.Millisecond,
		StartTime:    start,
		EndTime:      end,
		IsLLM:        true,
		Provider:     "openai",
		RequestModel: "gpt-4o",
		Model:        "gpt-4o-2024-05-13",
		ResponseID:   "chatcmpl-abc123",
		InputTokens:  847,
		OutputTokens: 512,
		FinishReason: "stop",
	}, jsonCall{
		Type:         "call",
		Sequence:     1,
		Method:       "POST",
		Host:         "api.openai.com",
		Path:         "/v1/chat/completions",
		StatusCode:   200,
		DurationMs:   1432,
		StartTime:    "2026-03-24T12:00:00Z",
		EndTime:      "2026-03-24T12:00:01.432Z",
		IsLLM:        true,
		Provider:     "openai",
		RequestModel: "gpt-4o",
		Model:        "gpt-4o-2024-05-13",
		ResponseID:   "chatcmpl-abc123",
		InputTokens:  847,
		OutputTokens: 512,
		FinishReason: "stop",
	})
}

func TestFormatCallJSONNonLLMCall(t *testing.T) {
	t.Parallel()
	start := time.Date(2026, 3, 24, 12, 0, 2, 0, time.UTC)
	end := start.Add(800 * time.Millisecond)

	checkFormatCallJSON(t, capture.CapturedCall{
		Sequence:   3,
		Method:     "GET",
		Host:       "github.com",
		Path:       "/foo/bar",
		StatusCode: 200,
		Duration:   800 * time.Millisecond,
		StartTime:  start,
		EndTime:    end,
		IsLLM:      false,
	}, jsonCall{
		Type:       "call",
		Sequence:   3,
		Method:     "GET",
		Host:       "github.com",
		Path:       "/foo/bar",
		StatusCode: 200,
		DurationMs: 800,
		StartTime:  "2026-03-24T12:00:02Z",
		EndTime:    "2026-03-24T12:00:02.8Z",
	})
}

func TestFormatCallJSONErrorCall(t *testing.T) {
	t.Parallel()
	start := time.Date(2026, 3, 24, 12, 0, 5, 0, time.UTC)
	end := start.Add(200 * time.Millisecond)

	checkFormatCallJSON(t, capture.CapturedCall{
		Sequence:     4,
		Method:       "POST",
		Host:         "api.openai.com",
		Path:         "/v1/chat/completions",
		StatusCode:   429,
		Duration:     200 * time.Millisecond,
		StartTime:    start,
		EndTime:      end,
		IsLLM:        true,
		Provider:     "openai",
		RequestModel: "gpt-4o",
		ErrorMessage: "rate_limit_exceeded",
	}, jsonCall{
		Type:         "call",
		Sequence:     4,
		Method:       "POST",
		Host:         "api.openai.com",
		Path:         "/v1/chat/completions",
		StatusCode:   429,
		DurationMs:   200,
		StartTime:    "2026-03-24T12:00:05Z",
		EndTime:      "2026-03-24T12:00:05.2Z",
		IsLLM:        true,
		Provider:     "openai",
		RequestModel: "gpt-4o",
		ErrorMessage: "rate_limit_exceeded",
	})
}

func TestFormatCallJSONToolCalls(t *testing.T) {
	t.Parallel()
	start := time.Date(2026, 3, 24, 12, 0, 1, 0, time.UTC)
	end := start.Add(4500 * time.Millisecond)

	checkFormatCallJSON(t, capture.CapturedCall{
		Sequence:     2,
		Method:       "POST",
		Host:         "api.anthropic.com",
		Path:         "/v1/messages",
		StatusCode:   200,
		Duration:     4500 * time.Millisecond,
		StartTime:    start,
		EndTime:      end,
		IsLLM:        true,
		Provider:     "anthropic",
		RequestModel: "claude-3-5-sonnet-20241022",
		Model:        "claude-3-5-sonnet-20241022",
		ResponseID:   "msg_01abc",
		InputTokens:  10568,
		OutputTokens: 268,
		FinishReason: "tool_use",
		ToolCalls:    []string{"read_file", "grep"},
		ToolCallArgs: []string{`{"path":"foo.go"}`, `{"pattern":"TODO"}`},
	}, jsonCall{
		Type:         "call",
		Sequence:     2,
		Method:       "POST",
		Host:         "api.anthropic.com",
		Path:         "/v1/messages",
		StatusCode:   200,
		DurationMs:   4500,
		StartTime:    "2026-03-24T12:00:01Z",
		EndTime:      "2026-03-24T12:00:05.5Z",
		IsLLM:        true,
		Provider:     "anthropic",
		RequestModel: "claude-3-5-sonnet-20241022",
		Model:        "claude-3-5-sonnet-20241022",
		ResponseID:   "msg_01abc",
		InputTokens:  10568,
		OutputTokens: 268,
		FinishReason: "tool_use",
		ToolCalls:    []string{"read_file", "grep"},
		ToolCallArgs: []string{`{"path":"foo.go"}`, `{"pattern":"TODO"}`},
	})
}

func TestFormatCallJSONWithCacheTokens(t *testing.T) {
	t.Parallel()
	start := time.Date(2026, 3, 24, 12, 0, 3, 0, time.UTC)
	end := start.Add(1200 * time.Millisecond)

	checkFormatCallJSON(t, capture.CapturedCall{
		Sequence:         5,
		Method:           "POST",
		Host:             "api.anthropic.com",
		Path:             "/v1/messages",
		StatusCode:       200,
		Duration:         1200 * time.Millisecond,
		StartTime:        start,
		EndTime:          end,
		IsLLM:            true,
		Provider:         "anthropic",
		RequestModel:     "claude-3-5-sonnet-20241022",
		Model:            "claude-3-5-sonnet-20241022",
		ResponseID:       "msg_cache1",
		InputTokens:      2000,
		OutputTokens:     500,
		CacheReadTokens:  800,
		CacheWriteTokens: 1500,
		FinishReason:     "end_turn",
	}, jsonCall{
		Type:             "call",
		Sequence:         5,
		Method:           "POST",
		Host:             "api.anthropic.com",
		Path:             "/v1/messages",
		StatusCode:       200,
		DurationMs:       1200,
		StartTime:        "2026-03-24T12:00:03Z",
		EndTime:          "2026-03-24T12:00:04.2Z",
		IsLLM:            true,
		Provider:         "anthropic",
		RequestModel:     "claude-3-5-sonnet-20241022",
		Model:            "claude-3-5-sonnet-20241022",
		ResponseID:       "msg_cache1",
		InputTokens:      2000,
		OutputTokens:     500,
		CacheReadTokens:  800,
		CacheWriteTokens: 1500,
		FinishReason:     "end_turn",
	})
}

func TestFormatCallJSONIsValidJSON(t *testing.T) {
	t.Parallel()

	line := formatCallJSON(capture.CapturedCall{
		Sequence:   1,
		Method:     "GET",
		Host:       "example.com",
		Path:       "/",
		StatusCode: 200,
		Duration:   100 * time.Millisecond,
		StartTime:  time.Date(2026, 1, 1, 0, 0, 0, 0, time.UTC),
		EndTime:    time.Date(2026, 1, 1, 0, 0, 0, int(100*time.Millisecond), time.UTC),
	})

	assert.True(t, json.Valid([]byte(line)), "formatCallJSON should produce valid JSON")
}

// checkPrintJSON calls printJSON on stats and asserts the result matches want.
// SessionDurationMs is non-deterministic, so it's verified as >= 0 then zeroed
// before the whole-value comparison.
func checkPrintJSON(t *testing.T, stats *sessionStats, envs []envtags.Env, want jsonSummary) {
	t.Helper()
	var buf bytes.Buffer
	stats.printJSON(&buf, time.Now(), envs)

	var got jsonSummary
	if err := json.Unmarshal(buf.Bytes(), &got); err != nil {
		t.Fatalf("printJSON produced invalid JSON: %v\nline: %s", err, buf.String())
	}

	assert.GreaterOrEqual(t, got.SessionDurationMs, int64(0))
	got.SessionDurationMs = 0
	assert.Equal(t, want, got)
}

func TestPrintJSONSummary(t *testing.T) {
	t.Parallel()

	var stats sessionStats
	stats.record(capture.CapturedCall{
		IsLLM:        true,
		Model:        "gpt-4o",
		InputTokens:  100,
		OutputTokens: 50,
		Duration:     2 * time.Second,
	})
	stats.record(capture.CapturedCall{
		IsLLM:        true,
		Model:        "gpt-4o",
		InputTokens:  200,
		OutputTokens: 100,
		Duration:     3 * time.Second,
	})
	stats.record(capture.CapturedCall{
		IsLLM:        true,
		Model:        "claude-3-5-sonnet-20241022",
		InputTokens:  500,
		OutputTokens: 200,
		Duration:     4 * time.Second,
	})
	stats.record(capture.CapturedCall{IsLLM: false, Host: "github.com"})

	checkPrintJSON(t, &stats, nil, jsonSummary{
		Type:          "summary",
		LLMCalls:      3,
		HTTPCalls:     1,
		InputTokens:   800,
		OutputTokens:  350,
		LLMDurationMs: 9000,
		Models:        map[string]int{"gpt-4o": 2, "claude-3-5-sonnet-20241022": 1},
	})
}

func TestPrintJSONSummaryNoCalls(t *testing.T) {
	t.Parallel()

	var stats sessionStats
	checkPrintJSON(t, &stats, nil, jsonSummary{
		Type: "summary",
	})
}

func TestPrintJSONSummaryWithCacheTokens(t *testing.T) {
	t.Parallel()

	var stats sessionStats
	stats.record(capture.CapturedCall{
		IsLLM:            true,
		Model:            "claude-3-5-sonnet-20241022",
		InputTokens:      2000,
		OutputTokens:     500,
		CacheReadTokens:  800,
		CacheWriteTokens: 1500,
		Duration:         1 * time.Second,
	})

	checkPrintJSON(t, &stats, nil, jsonSummary{
		Type:             "summary",
		LLMCalls:         1,
		InputTokens:      2000,
		OutputTokens:     500,
		CacheReadTokens:  800,
		CacheWriteTokens: 1500,
		LLMDurationMs:    1000,
		Models:           map[string]int{"claude-3-5-sonnet-20241022": 1},
	})
}

func TestPrintJSONSummaryWithEnvironmentTags(t *testing.T) {
	t.Parallel()

	var stats sessionStats
	stats.record(capture.CapturedCall{
		IsLLM:    true,
		Model:    "gpt-4o",
		Duration: 1 * time.Second,
	})

	envs := []envtags.Env{
		{
			Kind: envtags.GithubActions,
			Tags: map[string]string{
				"ci.repository": "rnaudi/aitrace",
			},
		},
		{Kind: envtags.GCP},
	}

	checkPrintJSON(t, &stats, envs, jsonSummary{
		Type:          "summary",
		LLMCalls:      1,
		LLMDurationMs: 1000,
		Models:        map[string]int{"gpt-4o": 1},
		Environment: map[string]string{
			"github-actions": "",
			"ci.repository":  "rnaudi/aitrace",
			"gcp":            "",
		},
	})
}

// --- envSummary tests ---

func checkEnvSummary(t *testing.T, envs []envtags.Env, want string) {
	t.Helper()
	got := envSummary(envs)
	assert.Equal(t, want, got)
}

func TestEnvSummaryEmpty(t *testing.T) {
	t.Parallel()
	checkEnvSummary(t, nil, "")
}

func TestEnvSummarySingleCI(t *testing.T) {
	t.Parallel()
	checkEnvSummary(t, []envtags.Env{
		{Kind: envtags.GithubActions},
	}, "github-actions")
}

func TestEnvSummaryMultiple(t *testing.T) {
	t.Parallel()
	checkEnvSummary(t, []envtags.Env{
		{Kind: envtags.GithubActions, Tags: map[string]string{"ci.repository": "foo/bar"}},
		{Kind: envtags.AWS},
		{Kind: envtags.Kubernetes},
	}, "github-actions, aws, k8s")
}

// --- envJSONTags tests ---

func checkEnvJSONTags(t *testing.T, envs []envtags.Env, want map[string]string) {
	t.Helper()
	got := envJSONTags(envs)
	assert.Equal(t, want, got)
}

func TestEnvJSONTagsEmpty(t *testing.T) {
	t.Parallel()
	checkEnvJSONTags(t, nil, nil)
}

func TestEnvJSONTagsSingleNoTags(t *testing.T) {
	t.Parallel()
	checkEnvJSONTags(t, []envtags.Env{
		{Kind: envtags.GCP},
	}, map[string]string{
		"gcp": "",
	})
}

func TestEnvJSONTagsMultipleWithTags(t *testing.T) {
	t.Parallel()
	checkEnvJSONTags(t, []envtags.Env{
		{Kind: envtags.GithubActions, Tags: map[string]string{
			"ci.repository": "rnaudi/aitrace",
			"ci.run_id":     "12345",
		}},
		{Kind: envtags.AWS, Tags: map[string]string{
			"cloud.region": "us-east-1",
		}},
	}, map[string]string{
		"github-actions": "",
		"ci.repository":  "rnaudi/aitrace",
		"ci.run_id":      "12345",
		"aws":            "",
		"cloud.region":   "us-east-1",
	})
}

// --- envResourceAttrs tests ---

func checkEnvResourceAttrs(t *testing.T, envs []envtags.Env, want []attribute.KeyValue) {
	t.Helper()
	got := envResourceAttrs(envs)
	assert.Equal(t, want, got)
}

func TestEnvResourceAttrsEmpty(t *testing.T) {
	t.Parallel()
	checkEnvResourceAttrs(t, nil, nil)
}

func TestEnvResourceAttrsCI(t *testing.T) {
	t.Parallel()
	checkEnvResourceAttrs(t, []envtags.Env{
		{Kind: envtags.GithubActions, Tags: map[string]string{
			"ci.repository": "rnaudi/aitrace",
		}},
	}, []attribute.KeyValue{
		attribute.String("aitrace.ci.system", "github-actions"),
		semconv.DeploymentEnvironment("ci"),
		attribute.String("aitrace.ci.repository", "rnaudi/aitrace"),
	})
}

func TestEnvResourceAttrsKubernetes(t *testing.T) {
	t.Parallel()
	checkEnvResourceAttrs(t, []envtags.Env{
		{Kind: envtags.Kubernetes},
	}, []attribute.KeyValue{
		attribute.Bool("aitrace.kubernetes", true),
	})
}

func TestEnvResourceAttrsCloud(t *testing.T) {
	t.Parallel()
	checkEnvResourceAttrs(t, []envtags.Env{
		{Kind: envtags.AWS, Tags: map[string]string{
			"cloud.region": "us-east-1",
		}},
	}, []attribute.KeyValue{
		semconv.CloudProviderKey.String("aws"),
		attribute.String("aitrace.cloud.region", "us-east-1"),
	})
}

func TestEnvResourceAttrsMixed(t *testing.T) {
	t.Parallel()
	checkEnvResourceAttrs(t, []envtags.Env{
		{Kind: envtags.GitLabCI},
		{Kind: envtags.GCP},
		{Kind: envtags.Kubernetes},
	}, []attribute.KeyValue{
		attribute.String("aitrace.ci.system", "gitlab-ci"),
		semconv.DeploymentEnvironment("ci"),
		semconv.CloudProviderKey.String("gcp"),
		attribute.Bool("aitrace.kubernetes", true),
	})
}

// --- doctor / probeHost tests ---

func TestProbeHostSuccess(t *testing.T) {
	t.Parallel()
	server := httptest.NewTLSServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		w.WriteHeader(http.StatusOK)
	}))
	defer server.Close()

	// Use the test server's CA so TLS verification passes.
	client := server.Client()
	client.Timeout = 5 * time.Second

	// server.URL is "https://127.0.0.1:<port>", strip the scheme.
	host := strings.TrimPrefix(server.URL, "https://")
	result := probeHost(client, host)

	assert.True(t, result.OK)
	assert.NoError(t, result.Err)
	assert.Equal(t, host, result.Host)
	assert.Greater(t, result.Elapsed, time.Duration(0))
}

func TestProbeHostTLSError(t *testing.T) {
	t.Parallel()
	server := httptest.NewTLSServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		w.WriteHeader(http.StatusOK)
	}))
	defer server.Close()

	// Client with empty CA pool — will not trust the test server's cert.
	client := &http.Client{
		Timeout: 5 * time.Second,
		Transport: &http.Transport{
			TLSClientConfig: &tls.Config{
				MinVersion: tls.VersionTLS12,
			},
		},
	}

	host := strings.TrimPrefix(server.URL, "https://")
	result := probeHost(client, host)

	assert.False(t, result.OK)
	assert.Error(t, result.Err)
	assert.Contains(t, result.Err.Error(), "certificate")
}

func TestProbeHostNetworkError(t *testing.T) {
	t.Parallel()

	client := &http.Client{Timeout: 2 * time.Second}
	// Port 1 is almost certainly not listening.
	result := probeHost(client, "127.0.0.1:1")

	assert.False(t, result.OK)
	assert.Error(t, result.Err)
}

func TestClassifyProbeErrorTLS(t *testing.T) {
	t.Parallel()
	server := httptest.NewTLSServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {}))
	defer server.Close()

	client := &http.Client{
		Timeout: 5 * time.Second,
		Transport: &http.Transport{
			TLSClientConfig: &tls.Config{
				MinVersion: tls.VersionTLS12,
			},
		},
	}

	host := strings.TrimPrefix(server.URL, "https://")
	result := probeHost(client, host)
	if result.Err == nil {
		t.Fatal("expected TLS error, got nil")
	}

	diagnosis, hint := classifyProbeError(result.Err)
	assert.Contains(t, diagnosis, "certificate")
	assert.Contains(t, hint, "corporate proxy")
}

func TestClassifyProbeErrorNetwork(t *testing.T) {
	t.Parallel()

	client := &http.Client{Timeout: 2 * time.Second}
	result := probeHost(client, "127.0.0.1:1")
	if result.Err == nil {
		t.Fatal("expected network error, got nil")
	}

	_, hint := classifyProbeError(result.Err)
	assert.Contains(t, hint, "network")
}
