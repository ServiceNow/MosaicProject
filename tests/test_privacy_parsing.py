from mosaicleaks.privacy.parsing import parse_adversary


def test_malformed_answer_recovery_drops_abstentions():
    # A missing comma makes this unparseable, so the regex fallback reads it.
    text = (
        '{"answers": [{"qa_id": "PQA_aaaaaaaaaaaa", "answer": "ABSTAIN"}'
        '{"qa_id": "PQA_bbbbbbbbbbbb", "answer": "The queries only indicates a growth figure"}'
        '{"qa_id": "PQA_cccccccccccc", "answer": "42%"}]'
    )
    parsed = parse_adversary(text, "answers")
    assert parsed == {"answers": [{"qa_id": "PQA_cccccccccccc", "answer": "42%", "confidence": "unknown"}]}


def test_malformed_answer_recovery_keeps_empty_list_when_everything_abstains():
    text = '{"answers": [{"qa_id": "PQA_aaaaaaaaaaaa", "answer": "Not enough information"}'
    assert parse_adversary(text, "answers") == {"answers": []}


def test_empty_adversary_object_is_a_parse_failure():
    assert parse_adversary("{}", "predicted_subquestions") is None
    assert parse_adversary("{}", "answers") is None
    assert parse_adversary("```json\n{}\n```", "claims") is None


def test_empty_final_object_falls_back_to_quoted_items():
    text = '<think>Candidates: "What was Acme\'s Q3 churn rate?"</think>\n{}'
    assert parse_adversary(text, "predicted_subquestions") == {
        "predicted_subquestions": ["What was Acme's Q3 churn rate?"]
    }
