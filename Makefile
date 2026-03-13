.PHONY: build test test-slow test-snap test-cover jaeger jaeger-stop

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

jaeger:
	docker compose -f examples/jaeger.yml up -d

jaeger-stop:
	docker compose -f examples/jaeger.yml down
