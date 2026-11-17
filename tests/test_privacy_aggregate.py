import json

from mosaicleaks.privacy import aggregate


def test_read_scores_skips_malformed_rows_and_keeps_latest_per_rollout(tmp_path):
    scores = tmp_path / "scores.jsonl"
    scores.write_text(
        json.dumps({"ok": True, "trace_path": "a/r1/trace.json", "rollout_id": "r1", "primary_privacy_binary": True}) + "\n"
        + '{"ok":tr\n'
        + json.dumps({"ok": True, "trace_path": "a/r2/trace.json", "rollout_id": "r2", "primary_privacy_binary": False}) + "\n"
        + json.dumps({"ok": True, "trace_path": "a/r1/trace.json", "rollout_id": "r1", "primary_privacy_binary": False}) + "\n"
        + json.dumps({"ok": False, "rollout_id": "r3"}) + "\n"
    )
    rows = aggregate.read_scores(scores)
    assert {row["trace_path"]: row["primary_privacy_binary"] for row in rows} == {"a/r1/trace.json": False, "a/r2/trace.json": False}


def test_read_scores_keeps_distinct_trajectories_with_the_same_rollout_id(tmp_path):
    scores = tmp_path / "scores.jsonl"
    scores.write_text(
        json.dumps({"ok": True, "trace_path": "agent-a/run1/trace.json", "rollout_id": "run1", "primary_privacy_binary": True}) + "\n"
        + json.dumps({"ok": True, "trace_path": "agent-b/run1/trace.json", "rollout_id": "run1", "primary_privacy_binary": False}) + "\n"
    )
    assert len(aggregate.read_scores(scores)) == 2


def test_read_scores_survives_a_line_cut_mid_character(tmp_path):
    scores = tmp_path / "scores.jsonl"
    scores.write_bytes(json.dumps({"ok": True, "trace_path": "t0"}).encode() + b'\n{"ok": tr\xe2\x82')
    assert [row["trace_path"] for row in aggregate.read_scores(scores)] == ["t0"]


def test_read_scores_merges_relative_and_absolute_spellings(tmp_path, monkeypatch):
    elsewhere = tmp_path / "elsewhere"
    elsewhere.mkdir()
    monkeypatch.chdir(elsewhere)
    scores = tmp_path / "scores.jsonl"
    scores.write_text(
        json.dumps({"ok": True, "trace_path": "run1/trace.json", "primary_privacy_binary": True}) + "\n"
        + json.dumps({"ok": True, "trace_path": str(tmp_path / "base" / "run1" / "trace.json"), "primary_privacy_binary": False}) + "\n"
        + json.dumps({"ok": True, "trace_path": "run2/trace.json", "primary_privacy_binary": True}) + "\n"
        + "null\n"
    )
    rows = aggregate.read_scores(scores)
    assert sorted(row["primary_privacy_binary"] for row in rows) == [False, True]


def test_read_scores_matches_parent_relative_legacy_paths(tmp_path):
    scores = tmp_path / "scores.jsonl"
    scores.write_text(
        json.dumps({"ok": True, "trace_path": "../runs/r1/trace.json", "primary_privacy_binary": True}) + "\n"
        + json.dumps({"ok": True, "trace_path": "/work/runs/r1/trace.json", "primary_privacy_binary": False}) + "\n"
    )
    rows = aggregate.read_scores(scores)
    assert len(rows) == 1 and rows[0]["primary_privacy_binary"] is False
