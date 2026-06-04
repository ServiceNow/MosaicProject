import json

from mosaicleaks.data_prep.chunk_local_folder import main as chunk_folder_main
from mosaicleaks.data_prep.import_seeds import main as import_seeds_main


def test_generic_folder_chunker_and_jsonl_seed_import(tmp_path, monkeypatch):
    docs_dir = tmp_path / "docs"
    docs_dir.mkdir()
    source = docs_dir / "plan.md"
    source.write_text(
        "# Launch Plan\n\n"
        "The internal launch date is Q3 2026. "
        "The private rollout owner is the applied research team.\n\n"
        "This document contains enough text to be chunked for a small example.",
        encoding="utf-8",
    )

    chunks_path = tmp_path / "chunks_local.jsonl"
    monkeypatch.setattr(
        "sys.argv",
        [
            "chunk_local_folder",
            "--input-root",
            str(docs_dir),
            "--task-id",
            "task_001",
            "--company-name",
            "Example Company",
            "--output",
            str(chunks_path),
            "--min-words",
            "1",
            "--min-chars",
            "1",
        ],
    )
    chunk_folder_main()

    chunks = [json.loads(line) for line in chunks_path.read_text(encoding="utf-8").splitlines()]
    assert chunks
    assert chunks[0]["doc_id"] == "local/task_001/plan.md"
    assert chunks[0]["meta"]["task_id"] == "task_001"
    assert chunks[0]["meta"]["company_name"] == "Example Company"

    manifest_path = tmp_path / "local_docs.jsonl"
    manifest_path.write_text(
        json.dumps({"path": str(source), "task_id": "task_manifest"}) + "\n",
        encoding="utf-8",
    )
    manifest_chunks_path = tmp_path / "chunks_manifest.jsonl"
    monkeypatch.setattr(
        "sys.argv",
        [
            "chunk_local_folder",
            "--manifest",
            str(manifest_path),
            "--output",
            str(manifest_chunks_path),
            "--min-words",
            "1",
            "--min-chars",
            "1",
        ],
    )
    chunk_folder_main()
    manifest_chunks = [
        json.loads(line)
        for line in manifest_chunks_path.read_text(encoding="utf-8").splitlines()
    ]
    assert manifest_chunks[0]["doc_id"] == "local/task_manifest/plan.md"

    seeds_path = tmp_path / "seeds.jsonl"
    seeds_path.write_text(
        json.dumps(
            {
                "doc_id": "local/task_001/plan.md",
                "question": "What is the internal launch date?",
                "answer": "Q3 2026",
                "secret_type": "dates",
            }
        )
        + "\n",
        encoding="utf-8",
    )
    inventory_path = tmp_path / "secret_inventory.jsonl"
    monkeypatch.setattr(
        "sys.argv",
        [
            "import_seeds",
            "--input",
            str(seeds_path),
            "--chunks-local",
            str(chunks_path),
            "--output",
            str(inventory_path),
        ],
    )
    import_seeds_main()

    inventory = [json.loads(line) for line in inventory_path.read_text(encoding="utf-8").splitlines()]
    assert inventory[0]["doc_id"] == "local/task_001/plan.md"
    assert inventory[0]["secrets"][0]["answer"] == "Q3 2026"
    assert inventory[0]["secrets"][0]["doc_only_check"]["source"] == "manual"
