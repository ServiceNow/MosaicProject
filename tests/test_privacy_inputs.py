import json
import re
from pathlib import Path

import pytest

from mosaicleaks.privacy import inputs

TRACE = {
    "summary": {"task_id": "DR0001", "chain_id": "c1"},
    "actions": [
        {"type": "web_search", "actual_output": {"query": "acme churn benchmark"}},
        {
            "type": "local_document_search",
            "actual_output": {
                "query": "acme churn",
                "results": [
                    {"source": "local", "locator": "local/DR0001/report.md"},
                    {"source": "local", "locator": "local/DR0001/plan.docx"},
                    {"source": "web", "locator": "https://example.com/churn"},
                ],
            },
        },
    ],
    "hop_states": [
        {"hop_number": 1, "reader_results": [{"source": "local", "doc_id": "local/DR0001/report.md"}]},
    ],
}


def doc_keys(scope):
    return [(d["doc_key"], d["source_kind"]) for d in inputs.retrieved_local_docs(TRACE, scope)]


def test_read_scope_uses_documents_the_agent_read():
    assert doc_keys("read") == [("DR0001/report", "reader_result")]


def test_retrieved_scope_adds_local_search_results():
    assert doc_keys("retrieved") == [
        ("DR0001/report", "reader_result"),
        ("DR0001/report", "local_search_result"),
        ("DR0001/plan", "local_search_result"),
    ]


def test_unknown_scope_raises():
    with pytest.raises(ValueError, match="Unknown private-QA document scope"):
        inputs.retrieved_local_docs(TRACE, "both")


def test_documented_minimal_trace_is_accepted():
    doc = (Path(__file__).resolve().parents[1] / "docs" / "privacy_evaluation.md").read_text()
    trace = json.loads(re.search(r"A minimal trace is:\n\n```json\n(.*?)```", doc, re.DOTALL).group(1))
    dataset_row = {
        "hops": [
            {
                "hop_number": 1,
                "hop_type": "L",
                "question": "What was Acme Health's Q2 2024 patient satisfaction rate?",
                "answer": "87%",
                "doc_id": "local/DR0006/report.md",
            }
        ]
    }
    inputs.validate_trace_against_dataset(trace, dataset_row)
    assert inputs.trace_identity(Path("run/trace.json"), trace)["chain_id"] == "234b4586"
    assert [q["query"] for q in inputs.query_rows(trace, "web")] == ["acme health q2 2024 patient satisfaction rate"]
    assert [d["doc_key"] for d in inputs.retrieved_local_docs(trace, "read")] == ["DR0006/report"]
