"""Both serving source branches use the same wheel builder and artifact identity."""

from pathlib import Path

import yaml


def test_source_channel_triggers_share_one_builder():
    root = Path(__file__).resolve().parents[2]
    workflow = yaml.load(
        (root / ".github/workflows/lil-cu134-wheel-release.yml").read_text(),
        Loader=yaml.BaseLoader,
    )
    assert workflow["on"]["push"]["branches"] == ["master", "integration/beta"]
    assert "paths" not in workflow["on"]["push"]
    assert "workflow_dispatch" in workflow["on"]
    assert workflow["concurrency"]["cancel-in-progress"] == "false"
    jobs = workflow["jobs"]
    assert set(jobs) == {"build-beta", "notify-container", "promote-stable"}
    build = "\n".join(step.get("run", "") for step in jobs["build-beta"]["steps"])
    assert build.count("ci/lil_wheels/build_bundle.sh") == 1
    assert "b12x-cu134-beta-${GITHUB_SHA}" in build
    notification = jobs["notify-container"]
    assert notification["needs"] == "build-beta"
    assert "integration/beta" in notification["if"]
    assert "LIL_CONTAINER_DISPATCH_ENABLED" in notification["if"]
    assert notification["permissions"] == {}
