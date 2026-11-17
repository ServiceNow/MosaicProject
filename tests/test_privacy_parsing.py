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
