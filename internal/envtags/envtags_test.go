package envtags

import (
	"testing"

	"github.com/stretchr/testify/assert"
	"pgregory.net/rapid"
)

// noIMDS is a no-op IMDS probe that skips the 1-second network timeout.
func noIMDS() (Env, bool) { return Env{}, false }

// checkDetect sets the given env vars, runs detect() with no IMDS probe,
// and asserts the result matches want.
//
// Why not t.Parallel: t.Setenv is incompatible with parallel subtests
// because it mutates process-global state.
func checkDetect(t *testing.T, envVars map[string]string, want []Env) {
	t.Helper()
	for k, v := range envVars {
		t.Setenv(k, v)
	}
	got := detect(noIMDS)
	assert.Equal(t, want, got)
}

func TestDetectGitHubActions(t *testing.T) {
	checkDetect(t, map[string]string{
		"GITHUB_ACTIONS":    "true",
		"GITHUB_REPOSITORY": "rnaudi/aitrace",
		"GITHUB_RUN_ID":     "12345",
		"GITHUB_WORKFLOW":   "test",
	}, []Env{{
		Kind: GithubActions,
		Tags: map[string]string{
			"ci.repository": "rnaudi/aitrace",
			"ci.run_id":     "12345",
			"ci.workflow":   "test",
		},
	}})
}

func TestDetectGitHubActionsMinimal(t *testing.T) {
	checkDetect(t, map[string]string{
		"GITHUB_ACTIONS": "true",
	}, []Env{{
		Kind: GithubActions,
		Tags: map[string]string{},
	}})
}

func TestDetectGitLabCI(t *testing.T) {
	checkDetect(t, map[string]string{
		"GITLAB_CI":       "true",
		"CI_PROJECT_PATH": "group/project",
		"CI_PIPELINE_ID":  "999",
	}, []Env{{
		Kind: GitLabCI,
		Tags: map[string]string{
			"ci.repository":  "group/project",
			"ci.pipeline_id": "999",
		},
	}})
}

func TestDetectCircleCI(t *testing.T) {
	checkDetect(t, map[string]string{
		"CIRCLECI":                "true",
		"CIRCLE_PROJECT_REPONAME": "aitrace",
		"CIRCLE_BUILD_NUM":        "42",
	}, []Env{{
		Kind: CircleCI,
		Tags: map[string]string{
			"ci.repository": "aitrace",
			"ci.build_num":  "42",
		},
	}})
}

func TestDetectJenkins(t *testing.T) {
	checkDetect(t, map[string]string{
		"JENKINS_URL":  "http://ci.example.com/",
		"BUILD_NUMBER": "100",
	}, []Env{{
		Kind: Jenkins,
		Tags: map[string]string{
			"ci.build_num": "100",
		},
	}})
}

func TestDetectBuildkite(t *testing.T) {
	checkDetect(t, map[string]string{
		"BUILDKITE":               "true",
		"BUILDKITE_PIPELINE_SLUG": "my-pipeline",
		"BUILDKITE_BUILD_NUMBER":  "7",
	}, []Env{{
		Kind: Buildkite,
		Tags: map[string]string{
			"ci.pipeline":  "my-pipeline",
			"ci.build_num": "7",
		},
	}})
}

func TestDetectTravisCI(t *testing.T) {
	checkDetect(t, map[string]string{
		"TRAVIS":              "true",
		"TRAVIS_REPO_SLUG":    "user/repo",
		"TRAVIS_BUILD_NUMBER": "55",
	}, []Env{{
		Kind: TravisCI,
		Tags: map[string]string{
			"ci.repository": "user/repo",
			"ci.build_num":  "55",
		},
	}})
}

func TestDetectNone(t *testing.T) {
	checkDetect(t, nil, nil)
}

func TestDetectFlyIO(t *testing.T) {
	checkDetect(t, map[string]string{
		"FLY_MACHINE_ID": "abc123",
		"FLY_REGION":     "iad",
	}, []Env{{
		Kind: FlyIO,
		Tags: map[string]string{
			"cloud.machine_id": "abc123",
			"cloud.region":     "iad",
		},
	}})
}

func TestDetectRailway(t *testing.T) {
	checkDetect(t, map[string]string{
		"RAILWAY_ENVIRONMENT":  "production",
		"RAILWAY_SERVICE_NAME": "api",
	}, []Env{{
		Kind: Railway,
		Tags: map[string]string{
			"cloud.service":     "api",
			"cloud.environment": "production",
		},
	}})
}

func TestDetectKubernetes(t *testing.T) {
	checkDetect(t, map[string]string{
		"KUBERNETES_SERVICE_HOST": "10.0.0.1",
	}, []Env{{Kind: Kubernetes}})
}

func TestDetectGitLabOnKubernetes(t *testing.T) {
	checkDetect(t, map[string]string{
		"GITLAB_CI":               "true",
		"CI_PROJECT_PATH":         "team/service",
		"KUBERNETES_SERVICE_HOST": "10.0.0.1",
	}, []Env{
		{
			Kind: GitLabCI,
			Tags: map[string]string{
				"ci.repository": "team/service",
			},
		},
		{Kind: Kubernetes},
	})
}

func TestKindString(t *testing.T) {
	t.Parallel()

	assert.Equal(t, "github-actions", GithubActions.String())
	assert.Equal(t, "gitlab-ci", GitLabCI.String())
	assert.Equal(t, "circleci", CircleCI.String())
	assert.Equal(t, "jenkins", Jenkins.String())
	assert.Equal(t, "buildkite", Buildkite.String())
	assert.Equal(t, "travis-ci", TravisCI.String())
	assert.Equal(t, "aws", AWS.String())
	assert.Equal(t, "gcp", GCP.String())
	assert.Equal(t, "azure", Azure.String())
	assert.Equal(t, "fly", FlyIO.String())
	assert.Equal(t, "railway", Railway.String())
	assert.Equal(t, "k8s", Kubernetes.String())
	assert.Equal(t, "unknown", Kind(0).String())
}

func TestReadDMINonexistentPath(t *testing.T) {
	t.Parallel()
	assert.Equal(t, "", readDMI("nonexistent_file_that_does_not_exist"))
}

func TestDetectCIPropertyGitHubActions(t *testing.T) {
	rapid.Check(t, func(rt *rapid.T) {
		repo := rapid.StringMatching(`[a-z]+/[a-z]+`).Draw(rt, "repo")
		runID := rapid.StringMatching(`[0-9]{1,10}`).Draw(rt, "runID")
		t.Setenv("GITHUB_ACTIONS", "true")
		t.Setenv("GITHUB_REPOSITORY", repo)
		t.Setenv("GITHUB_RUN_ID", runID)

		envs := detect(noIMDS)

		if assert.Len(rt, envs, 1) {
			assert.Equal(rt, GithubActions, envs[0].Kind)
			assert.Equal(rt, repo, envs[0].Tags["ci.repository"])
			assert.Equal(rt, runID, envs[0].Tags["ci.run_id"])
		}
	})
}

func TestDetectPropertyNonEmpty(t *testing.T) {
	t.Parallel()
	rapid.Check(t, func(rt *rapid.T) {
		kind := rapid.SampledFrom([]Kind{
			GithubActions, GitLabCI, CircleCI, AWS, GCP, Azure, FlyIO, Kubernetes,
		}).Draw(rt, "kind")
		e := Env{Kind: kind}
		assert.NotEmpty(rt, e.Kind.String())
		assert.NotEqual(rt, "unknown", e.Kind.String())
	})
}
