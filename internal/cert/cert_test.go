package cert

import (
	"crypto/ecdsa"
	"crypto/elliptic"
	"crypto/rand"
	"crypto/x509"
	"crypto/x509/pkix"
	"encoding/pem"
	"math/big"
	"os"
	"testing"
	"time"

	"github.com/stretchr/testify/assert"
)

func generateTestCA(t *testing.T) *x509.Certificate {
	t.Helper()

	key, err := ecdsa.GenerateKey(elliptic.P256(), rand.Reader)
	if err != nil {
		t.Fatalf("generate key: %v", err)
	}

	template := &x509.Certificate{
		SerialNumber: big.NewInt(1),
		Subject:      pkix.Name{CommonName: "aitrace test CA"},
		NotBefore:    time.Now().Add(-time.Hour),
		NotAfter:     time.Now().Add(time.Hour),
		IsCA:         true,
		KeyUsage:     x509.KeyUsageCertSign,
	}

	certDER, err := x509.CreateCertificate(rand.Reader, template, template, &key.PublicKey, key)
	if err != nil {
		t.Fatalf("create certificate: %v", err)
	}

	cert, err := x509.ParseCertificate(certDER)
	if err != nil {
		t.Fatalf("parse certificate: %v", err)
	}
	return cert
}

func TestWriteCombinedPEMContainsEphemeralCA(t *testing.T) {
	t.Parallel()

	ca := generateTestCA(t)
	dir := t.TempDir()

	pemPath, err := WriteCombinedPEM(ca, dir)
	if err != nil {
		t.Fatalf("WriteCombinedPEM: %v", err)
	}

	data, err := os.ReadFile(pemPath)
	if err != nil {
		t.Fatalf("read combined PEM: %v", err)
	}

	// Walk all PEM blocks and check the last one matches our ephemeral CA.
	var lastBlock *pem.Block
	rest := data
	for {
		var block *pem.Block
		block, rest = pem.Decode(rest)
		if block == nil {
			break
		}
		lastBlock = block
	}

	if lastBlock == nil {
		t.Fatal("no PEM blocks found in combined file")
	}

	assert.Equal(t, "CERTIFICATE", lastBlock.Type)
	assert.Equal(t, ca.Raw, lastBlock.Bytes, "last PEM block should be the ephemeral CA")
}

func TestWriteCombinedPEMContainsSystemCAs(t *testing.T) {
	t.Parallel()

	ca := generateTestCA(t)
	dir := t.TempDir()

	pemPath, err := WriteCombinedPEM(ca, dir)
	if err != nil {
		t.Fatalf("WriteCombinedPEM: %v", err)
	}

	info, err := os.Stat(pemPath)
	if err != nil {
		t.Fatalf("stat combined PEM: %v", err)
	}

	// The ephemeral CA PEM alone is ~300-400 bytes. The system CA bundle is
	// typically 200KB+. If the file is larger than 1KB, the system CAs are present.
	assert.Greater(t, info.Size(), int64(1024),
		"combined PEM should be much larger than just the ephemeral CA")
}

func TestWriteCombinedPEMMultiplePEMBlocks(t *testing.T) {
	t.Parallel()

	ca := generateTestCA(t)
	dir := t.TempDir()

	pemPath, err := WriteCombinedPEM(ca, dir)
	if err != nil {
		t.Fatalf("WriteCombinedPEM: %v", err)
	}

	data, err := os.ReadFile(pemPath)
	if err != nil {
		t.Fatalf("read combined PEM: %v", err)
	}

	blockCount := 0
	rest := data
	for {
		var block *pem.Block
		block, rest = pem.Decode(rest)
		if block == nil {
			break
		}
		blockCount++
	}

	// System CA bundles contain dozens of certificates. Our ephemeral CA adds one more.
	assert.Greater(t, blockCount, 1, "should have system CAs + ephemeral CA")
}
