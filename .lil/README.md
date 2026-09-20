# Integration container changelog fragments

A runtime-affecting change that will be pushed or imported into an
`integration/*` branch must include one versioned JSON fragment in
`.lil/changes/`. The community container publisher compares these fragments
between exact component revisions and creates the container release notes.

The complete schema, examples, correction policy, and publication flow are in
the [integration container changelog runbook](https://github.com/local-inference-lab/blackwell-llm-docker/blob/main/docs/integration-release-changelog.md).

Use a component-prefixed identifier that matches the filename. For example,
`.lil/changes/b12x-402.json` contains:

```json
{
  "schema": "local-inference-release-change/v1",
  "id": "b12x-402",
  "category": "fix",
  "summary": "Merge QSA selection across DCP ranks",
  "models": ["Qwen3.8-Flash-Next"],
  "compatibility": "No user action required.",
  "details": [
    "DCP1, DCP2, and DCP4 use the same distributed selection contract."
  ],
  "pull_requests": [402],
  "requires": []
}
```

The fragment belongs in the implementation PR or in the commit series that
imports the reviewed change. Preserve contributor attribution. Once a container
release has published a fragment, do not edit or delete it; add a corrective
fragment instead.
