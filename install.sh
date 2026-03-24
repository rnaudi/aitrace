#!/bin/sh
# install.sh — download and install aitrace from GitHub Releases.
#
# Usage:
#   curl -sSL https://raw.githubusercontent.com/rnaudi/aitrace/main/install.sh | sh

set -eu

readonly REPO="rnaudi/aitrace"
readonly INSTALL_DIR="/usr/local/bin"

info() {
    printf '  %s\n' "$*"
}

error() {
    printf '  error: %s\n' "$*" >&2
    exit 1
}

detect_os() {
    local os
    os="$(uname -s | tr '[:upper:]' '[:lower:]')"
    case "$os" in
        darwin) ;;
        linux)  ;;
        *)      error "unsupported OS: $os" ;;
    esac
    printf '%s' "$os"
}

detect_arch() {
    local arch
    arch="$(uname -m)"
    case "$arch" in
        x86_64)          arch="amd64" ;;
        aarch64|arm64)   arch="arm64" ;;
        *)               error "unsupported architecture: $arch" ;;
    esac
    printf '%s' "$arch"
}

latest_version() {
    if ! command -v curl >/dev/null 2>&1; then
        error "curl is required but not found"
    fi
    local version
    version="$(curl -sSf "https://api.github.com/repos/${REPO}/releases/latest" |
        grep '"tag_name"' | cut -d'"' -f4)"
    if [ -z "$version" ]; then
        error "could not determine latest version"
    fi
    printf '%s' "$version"
}

main() {
    local os arch version url tmpdir

    os="$(detect_os)"
    arch="$(detect_arch)"
    version="$(latest_version)"

    readonly url="https://github.com/${REPO}/releases/download/${version}/aitrace-${os}-${arch}.tar.gz"

    info "installing aitrace ${version} (${os}/${arch})"

    tmpdir="$(mktemp -d)"
    trap 'rm -rf "$tmpdir"' EXIT

    curl -sSfL "$url" -o "${tmpdir}/aitrace.tar.gz" ||
        error "download failed: ${url}"

    tar -xzf "${tmpdir}/aitrace.tar.gz" -C "$tmpdir"

    if [ -w "$INSTALL_DIR" ]; then
        mv "${tmpdir}/aitrace" "${INSTALL_DIR}/aitrace"
    else
        info "writing to ${INSTALL_DIR} requires elevated permissions"
        sudo mv "${tmpdir}/aitrace" "${INSTALL_DIR}/aitrace"
    fi

    chmod +x "${INSTALL_DIR}/aitrace"

    info "installed aitrace to ${INSTALL_DIR}/aitrace"
    info "$(${INSTALL_DIR}/aitrace --version)"
}

main
