// run.go — Child process spawning with proxy environment variables.
//
// Inherits the parent's env, overlays HTTP_PROXY/HTTPS_PROXY and CA trust
// env vars. Forwards SIGINT/SIGTERM to the child and returns its exit code.
package run

import (
	"fmt"
	"os"
	"os/exec"
	"os/signal"
	"syscall"
)

// RunOptions configures the child process.
type RunOptions struct {
	ProxyAddr       string // "host:port"
	CombinedPEMPath string
	Command         string
	Args            []string
}

// RunChild spawns a child process with proxy env vars, pipes its stdio,
// and returns (exitCode, pid, error).
func RunChild(opts RunOptions) (int, int, error) {
	cmd := exec.Command(opts.Command, opts.Args...)
	cmd.Stdin = os.Stdin
	cmd.Stdout = os.Stdout
	cmd.Stderr = os.Stderr

	proxyURL := "http://" + opts.ProxyAddr

	// Appending to os.Environ() means our proxy vars override any existing
	// HTTP_PROXY/HTTPS_PROXY values (last occurrence wins in exec.Cmd.Env).
	cmd.Env = append(os.Environ(),
		"HTTP_PROXY="+proxyURL,
		"HTTPS_PROXY="+proxyURL,
		"http_proxy="+proxyURL,
		"https_proxy="+proxyURL,
		// Each runtime checks a different env var for CA trust:
		//   SSL_CERT_FILE      — OpenSSL, Python ssl module
		//   NODE_EXTRA_CA_CERTS — Node.js
		//   REQUESTS_CA_BUNDLE  — Python requests library
		//   DENO_CERT           — Deno
		"SSL_CERT_FILE="+opts.CombinedPEMPath,
		"NODE_EXTRA_CA_CERTS="+opts.CombinedPEMPath,
		"REQUESTS_CA_BUNDLE="+opts.CombinedPEMPath,
		"DENO_CERT="+opts.CombinedPEMPath,
	)

	// Without forwarding, Ctrl+C kills aitrace but leaves the child orphaned.
	sigCh := make(chan os.Signal, 1)
	signal.Notify(sigCh, syscall.SIGINT, syscall.SIGTERM)
	defer signal.Stop(sigCh)

	if err := cmd.Start(); err != nil {
		return 1, 0, fmt.Errorf("start child: %w", err)
	}

	childPID := cmd.Process.Pid

	done := make(chan struct{})
	go func() {
		for {
			select {
			case sig := <-sigCh:
				// Signal may fail if the child has already exited; this is expected.
				_ = cmd.Process.Signal(sig)
			case <-done:
				return
			}
		}
	}()

	waitErr := cmd.Wait()
	close(done)
	if waitErr == nil { // child exited cleanly with code 0
		return 0, childPID, nil
	}

	if exitErr, ok := waitErr.(*exec.ExitError); ok {
		return exitErr.ExitCode(), childPID, nil
	}

	return 1, childPID, fmt.Errorf("wait child: %w", waitErr)
}
