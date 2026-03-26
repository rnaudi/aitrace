// cert.go — CA bundle generation for MITM TLS interception.
//
// Why: Child processes need to trust the proxy's ephemeral CA.
// We append its PEM to the system CA bundle and point env vars at the
// combined file.
package cert

import (
	"crypto/x509"
	"encoding/pem"
	"fmt"
	"os"
	"path/filepath"
	"runtime"
)

func systemCertPaths() []string {
	switch runtime.GOOS {
	case "linux":
		return []string{
			"/etc/ssl/certs/ca-certificates.crt",                // Debian/Ubuntu
			"/etc/pki/tls/certs/ca-bundle.crt",                  // Fedora/RHEL
			"/etc/ssl/ca-bundle.pem",                            // OpenSUSE
			"/etc/pki/ca-trust/extracted/pem/tls-ca-bundle.pem", // CentOS
			"/etc/ssl/cert.pem",                                 // Alpine
		}
	case "darwin":
		return []string{
			"/etc/ssl/cert.pem",
		}
	default:
		return nil
	}
}

// FindSystemCertBundle returns the path to the system CA bundle.
// It checks SSL_CERT_FILE first, then probes well-known paths for the
// current OS.
func FindSystemCertBundle() (string, error) {
	// Check SSL_CERT_FILE first — respect existing config.
	if p := os.Getenv("SSL_CERT_FILE"); p != "" {
		if _, err := os.Stat(p); err == nil {
			return p, nil
		}
	}
	for _, p := range systemCertPaths() {
		if _, err := os.Stat(p); err == nil {
			return p, nil
		}
	}
	return "", fmt.Errorf("no system CA bundle found")
}

// WriteCombinedPEM writes a PEM file that contains the system CAs plus
// the proxy's ephemeral CA. Returns the path to the combined file.
// The caller is responsible for cleaning up dir when done.
func WriteCombinedPEM(dir string, caCert *x509.Certificate) (string, error) {
	sysBundlePath, err := FindSystemCertBundle()
	if err != nil {
		return "", fmt.Errorf("find system certs: %w", err)
	}

	sysBundle, err := os.ReadFile(sysBundlePath)
	if err != nil {
		return "", fmt.Errorf("read system certs: %w", err)
	}

	caPEM := pem.EncodeToMemory(&pem.Block{
		Type:  "CERTIFICATE",
		Bytes: caCert.Raw,
	})

	combined := append(sysBundle, '\n')
	combined = append(combined, caPEM...)

	outPath := filepath.Join(dir, "combined-ca.pem")
	if err := os.WriteFile(outPath, combined, 0600); err != nil {
		return "", fmt.Errorf("write combined PEM: %w", err)
	}

	return outPath, nil
}
