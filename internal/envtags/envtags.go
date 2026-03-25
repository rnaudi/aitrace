// envtags.go — detect cloud, CI, and container environment at startup.
//
// Detection strategy (ordered by cost):
//  1. Environment variables — instant, covers CI systems + Fly.io/Railway.
//  2. DMI sysfs files — Linux only, covers GCP/AWS/Azure bare metal and VMs.
//  3. IMDS endpoint (169.254.169.254) — 1s timeout, covers cloud VMs that
//     lack DMI or env var signals.
//  4. Kubernetes service account — filesystem read, no network.
//
// Why no async detection: aitrace is a CLI tool that runs once. A 1s IMDS
// timeout at startup is acceptable. Async detection would complicate the
// code for marginal benefit.
//
// Adding a provider requires adding a Kind constant, a String() case,
// a detect function, and tests in envtags_test.go.
package envtags

import (
	"io"
	"net/http"
	"os"
	"strings"
	"time"
)

// Kind identifies a detected runtime environment.
type Kind int

const (
	GithubActions Kind = iota + 1
	GitLabCI
	CircleCI
	Jenkins
	Buildkite
	TravisCI
	AWS
	GCP
	Azure
	FlyIO
	Railway
	Kubernetes
)

// String returns the short human-readable label for this kind.
func (k Kind) String() string {
	switch k {
	case GithubActions:
		return "github-actions"
	case GitLabCI:
		return "gitlab-ci"
	case CircleCI:
		return "circleci"
	case Jenkins:
		return "jenkins"
	case Buildkite:
		return "buildkite"
	case TravisCI:
		return "travis-ci"
	case AWS:
		return "aws"
	case GCP:
		return "gcp"
	case Azure:
		return "azure"
	case FlyIO:
		return "fly"
	case Railway:
		return "railway"
	case Kubernetes:
		return "k8s"
	default:
		return "unknown"
	}
}

// Env represents a single detected runtime environment with its metadata.
type Env struct {
	Kind Kind
	Tags map[string]string
}

// Detect probes the local environment for cloud, CI, and container metadata.
// It reads env vars, sysfs files, and the IMDS endpoint (with a 1-second
// timeout). Safe to call from any platform.
func Detect() []Env {
	return detect(detectCloudDMI, detectCloudIMDS)
}

// detect is the internal implementation of Detect. The probe parameters
// allow tests to skip the sysfs reads and 1-second IMDS timeout by
// injecting no-ops.
func detect(dmiProbe, imdsProbe func() (Env, bool)) []Env {
	var envs []Env

	if e, ok := detectGithubActions(); ok {
		envs = append(envs, e)
	} else if e, ok := detectGitLabCI(); ok {
		envs = append(envs, e)
	} else if e, ok := detectCircleCI(); ok {
		envs = append(envs, e)
	} else if e, ok := detectJenkins(); ok {
		envs = append(envs, e)
	} else if e, ok := detectBuildkite(); ok {
		envs = append(envs, e)
	} else if e, ok := detectTravisCI(); ok {
		envs = append(envs, e)
	}

	if e, ok := detectFlyIO(); ok {
		envs = append(envs, e)
	} else if e, ok := detectRailway(); ok {
		envs = append(envs, e)
	} else if e, ok := dmiProbe(); ok {
		envs = append(envs, e)
	} else if e, ok := imdsProbe(); ok {
		envs = append(envs, e)
	}

	if e, ok := detectKubernetes(); ok {
		envs = append(envs, e)
	}

	return envs
}

func detectGithubActions() (Env, bool) {
	if os.Getenv("GITHUB_ACTIONS") != "true" {
		return Env{}, false
	}
	tags := make(map[string]string)
	if v := os.Getenv("GITHUB_REPOSITORY"); v != "" {
		tags["ci.repository"] = v
	}
	if v := os.Getenv("GITHUB_RUN_ID"); v != "" {
		tags["ci.run_id"] = v
	}
	if v := os.Getenv("GITHUB_WORKFLOW"); v != "" {
		tags["ci.workflow"] = v
	}
	return Env{Kind: GithubActions, Tags: tags}, true
}

func detectGitLabCI() (Env, bool) {
	if os.Getenv("GITLAB_CI") != "true" {
		return Env{}, false
	}
	tags := make(map[string]string)
	if v := os.Getenv("CI_PROJECT_PATH"); v != "" {
		tags["ci.repository"] = v
	}
	if v := os.Getenv("CI_PIPELINE_ID"); v != "" {
		tags["ci.pipeline_id"] = v
	}
	return Env{Kind: GitLabCI, Tags: tags}, true
}

func detectCircleCI() (Env, bool) {
	if os.Getenv("CIRCLECI") != "true" {
		return Env{}, false
	}
	tags := make(map[string]string)
	if v := os.Getenv("CIRCLE_PROJECT_REPONAME"); v != "" {
		tags["ci.repository"] = v
	}
	if v := os.Getenv("CIRCLE_BUILD_NUM"); v != "" {
		tags["ci.build_num"] = v
	}
	return Env{Kind: CircleCI, Tags: tags}, true
}

func detectJenkins() (Env, bool) {
	if os.Getenv("JENKINS_URL") == "" {
		return Env{}, false
	}
	tags := make(map[string]string)
	if v := os.Getenv("BUILD_NUMBER"); v != "" {
		tags["ci.build_num"] = v
	}
	return Env{Kind: Jenkins, Tags: tags}, true
}

func detectBuildkite() (Env, bool) {
	if os.Getenv("BUILDKITE") != "true" {
		return Env{}, false
	}
	tags := make(map[string]string)
	if v := os.Getenv("BUILDKITE_PIPELINE_SLUG"); v != "" {
		tags["ci.pipeline"] = v
	}
	if v := os.Getenv("BUILDKITE_BUILD_NUMBER"); v != "" {
		tags["ci.build_num"] = v
	}
	return Env{Kind: Buildkite, Tags: tags}, true
}

func detectTravisCI() (Env, bool) {
	if os.Getenv("TRAVIS") != "true" {
		return Env{}, false
	}
	tags := make(map[string]string)
	if v := os.Getenv("TRAVIS_REPO_SLUG"); v != "" {
		tags["ci.repository"] = v
	}
	if v := os.Getenv("TRAVIS_BUILD_NUMBER"); v != "" {
		tags["ci.build_num"] = v
	}
	return Env{Kind: TravisCI, Tags: tags}, true
}

func detectFlyIO() (Env, bool) {
	machineID := os.Getenv("FLY_MACHINE_ID")
	region := os.Getenv("FLY_REGION")
	if machineID == "" || region == "" {
		return Env{}, false
	}
	return Env{
		Kind: FlyIO,
		Tags: map[string]string{
			"cloud.machine_id": machineID,
			"cloud.region":     region,
		},
	}, true
}

func detectRailway() (Env, bool) {
	railwayEnv := os.Getenv("RAILWAY_ENVIRONMENT")
	if railwayEnv == "" {
		return Env{}, false
	}
	tags := map[string]string{
		"cloud.environment": railwayEnv,
	}
	if v := os.Getenv("RAILWAY_SERVICE_NAME"); v != "" {
		tags["cloud.service"] = v
	}
	return Env{Kind: Railway, Tags: tags}, true
}

// readDMI reads a DMI identity file from sysfs. Returns "" on any error
// or on non-Linux platforms (the paths simply won't exist).
func readDMI(name string) string {
	for _, dir := range []string{"/sys/class/dmi/id/", "/sys/devices/virtual/dmi/id/"} {
		if b, err := os.ReadFile(dir + name); err == nil {
			return strings.TrimSpace(string(b))
		}
	}
	return ""
}

func detectCloudDMI() (Env, bool) {
	switch val := readDMI("product_name"); {
	case val == "Google" || val == "Google Compute Engine":
		return Env{Kind: GCP}, true
	case val == "Microsoft Corporation":
		return Env{Kind: Azure}, true
	}

	switch val := readDMI("bios_version"); {
	case val == "Google":
		return Env{Kind: GCP}, true
	case strings.HasSuffix(val, "amazon"):
		return Env{Kind: AWS}, true
	}

	switch val := readDMI("sys_vendor"); {
	case val == "Google":
		return Env{Kind: GCP}, true
	case val == "Microsoft Corporation":
		return Env{Kind: Azure}, true
	}

	return Env{}, false
}

const imdsTimeout = 1 * time.Second

// detectCloudIMDS probes the Instance Metadata Service endpoint.
// Returns the cloud provider or false if unreachable / unrecognized.
func detectCloudIMDS() (Env, bool) {
	client := &http.Client{Timeout: imdsTimeout}
	resp, err := client.Get("http://169.254.169.254")
	if err != nil {
		return Env{}, false
	}
	defer resp.Body.Close()
	// Drain body to allow connection reuse.
	io.Copy(io.Discard, io.LimitReader(resp.Body, 1<<10))

	if resp.Header.Get("server") == "EC2ws" {
		return Env{Kind: AWS}, true
	}
	if resp.Header.Get("metadata-flavor") == "Google" {
		return Env{Kind: GCP}, true
	}
	return Env{}, false
}

func detectKubernetes() (Env, bool) {
	if os.Getenv("KUBERNETES_SERVICE_HOST") == "" {
		return Env{}, false
	}
	e := Env{Kind: Kubernetes}

	// Read namespace from the mounted service account secret.
	if b, err := os.ReadFile("/var/run/secrets/kubernetes.io/serviceaccount/namespace"); err == nil {
		ns := strings.TrimSpace(string(b))
		if ns != "" {
			e.Tags = map[string]string{"k8s.namespace": ns}
		}
	}
	return e, true
}
