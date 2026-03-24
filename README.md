# aitrace

zero-config agent tracer

## Install

```sh
brew install rnaudi/tap/aitrace
```

Or download a binary from [releases](https://github.com/rnaudi/aitrace/releases).

```sh
aitrace run -- opencode "refactor the auth module"
[aitrace] #1 gpt-4o | tok: 537 in / 19 out | 857ms
[aitrace] #2 claude-opus-4.6 | tok: 19,620 in / 69 out | 3.8s | tools: glob !large
[aitrace] #3 claude-opus-4.6 | tok: 8,441 in / 1,205 out | 12.4s | tools: read, edit, bash
[aitrace] #4 claude-opus-4.6 | tok: 9,102 in / 87 out | 1.9s
[aitrace] ---
[aitrace] 4 calls | 37,700 in / 1,380 out | 18.9s
```

Works with OpenAI, Anthropic, and GitHub Copilot.

```sh
aitrace run -- cursor .                          # any AI tool
aitrace run -- aider "fix the login bug"         # any command
aitrace run -- python my_agent.py                # any language
aitrace run --json -- opencode 2>calls.jsonl     # JSONL export
aitrace run --otel -- opencode                   # OpenTelemetry spans via OTLP/gRPC
aitrace doctor                                   # check TLS connectivity
```

## Comparison

|                          | aitrace | Langfuse | Helicone | OpenLLMetry | Subtrace |
|--------------------------|---------|----------|----------|-------------|----------|
| Zero-config              | yes     | no       | no       | no          | yes      |
| Code changes needed      | no      | SDK      | base URL | SDK         | no       |
| LLM-aware                | yes     | yes      | yes      | yes         | no       |
| Works on 3rd-party tools | yes     | no       | no       | no          | yes      |
| OTel native              | yes     | yes      | no       | yes         | no       |
| Data stays local         | yes     | self-host| no       | depends     | no       |
| Language-agnostic        | yes     | no       | no       | no          | yes      |
