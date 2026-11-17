from mosaicleaks.privacy.normalize import normalize_answer_output


def test_answer_output_drops_abstentions():
    output = {
        "answers": [
            {"qa_id": "PQA_aaaaaaaaaaaa", "answer": "ABSTAIN", "confidence": "low"},
            {"qa_id": "PQA_bbbbbbbbbbbb", "answer": "The queries do not reveal the rate."},
            {"qa_id": "PQA_cccccccccccc", "answer": "Revenue grew 12% only in Q3", "confidence": "high"},
            {"qa_id": "PQA_dddddddddddd", "answer": "It only indicates that 12% is internal."},
            {"qa_id": "PQA_eeeeeeeeeeee", "answer": "  "},
        ]
    }
    normalized = normalize_answer_output(output)
    assert normalized["answers"] == [
        {"qa_id": "PQA_cccccccccccc", "answer": "Revenue grew 12% only in Q3", "confidence": "high"},
    ]


def test_answer_output_single_abstaining_answer_shape():
    assert normalize_answer_output({"qa_id": "PQA_aaaaaaaaaaaa", "answer": "abstain"})["answers"] == []
