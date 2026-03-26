.PHONY: build test test-slow test-snap test-cover lint jaeger jaeger-stop

build:
	go build -o aitrace ./cmd/aitrace

test:
	go test -count=1 -race -short ./...

test-slow:
	go test -count=1 -race -v ./...

test-snap:
	UPDATE_SNAPS=true go test -count=1 -race -short ./...

test-cover:
	go test -count=1 -race -short -coverprofile=coverage.out ./...

lint:
	go run honnef.co/go/tools/cmd/staticcheck@latest ./...
	go run golang.org/x/vuln/cmd/govulncheck@latest ./...

jaeger:
	docker compose -f examples/jaeger.yml up -d

jaeger-stop:
	docker compose -f examples/jaeger.yml down
