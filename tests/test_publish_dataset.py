import json

from mosaicleaks.publish_dataset import main


def write_jsonl(path, rows):
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row) + "\n")


def chain(chain_id, company):
    return {
        "chain_id": chain_id,
        "global_question": f"Question {chain_id}?",
        "global_answer": f"answer-{chain_id}",
        "pattern": "LW",
        "valid": True,
        "metadata": {"company": company, "task_id": f"TASK-{chain_id}"},
        "hops": [
            {
                "hop_number": 1,
                "hop_type": "L",
                "question": "Local?",
                "answer": "local",
                "doc_id": f"local-{chain_id}",
            },
            {
                "hop_number": 2,
                "hop_type": "W",
                "question": "Web?",
                "answer": f"answer-{chain_id}",
                "doc_id": f"web-{chain_id}",
            },
        ],
    }


def test_publish_dataset_bundle(tmp_path, monkeypatch):
    source = tmp_path / "chains_final.jsonl"
    output = tmp_path / "bundle"
    write_jsonl(
        source,
        [
            chain("1", "A"),
            chain("2", "B"),
            chain("3", "B"),
        ],
    )

    monkeypatch.setattr(
        "sys.argv",
        [
            "publish_dataset",
            "--input",
            str(source),
            "--output-dir",
            str(output),
            "--split-key",
            "company",
            "--test-group",
            "A",
        ],
    )
    assert main() == 0

    assert (output / "dataset.jsonl").exists()
    assert (output / "train.jsonl").exists()
    assert (output / "val.jsonl").exists()
    assert (output / "test.jsonl").exists()
    assert (output / "manifest.json").exists()
    assert (output / "dataset_stats.json").exists()
    assert (output / "schema.json").exists()

    manifest = json.loads((output / "manifest.json").read_text(encoding="utf-8"))
    assert manifest["counts"]["dataset"] == 3
    assert manifest["counts"]["test"] == 1

    test_rows = [
        json.loads(line)
        for line in (output / "test.jsonl").read_text(encoding="utf-8").splitlines()
    ]
    assert {row["company"] for row in test_rows} == {"A"}
