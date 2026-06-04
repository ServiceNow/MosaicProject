"""Convert chain hops to numbered question format with (N) back-references."""

import re


def format_numbered_questions(hops: list[dict]) -> str:
    """Convert hops to numbered format with (N) back-references.

    Args:
        hops: [{"question": str, "answer": str, "hop_number": int}, ...]

    Returns:
        Multiline string with numbered questions.

    Raises:
        ValueError: If a hop's question doesn't contain any previous answer.
    """
    if not hops:
        raise ValueError("No hops provided")

    lines = []
    for i, hop in enumerate(hops):
        q = hop["question"]

        if i > 0:
            prev_answers = sorted(
                [(h["hop_number"], h["answer"]) for h in hops[:i]],
                key=lambda x: len(x[1]),
                reverse=True,
            )
            found_any = False
            for ref_num, ref_answer in prev_answers:
                pattern = re.compile(re.escape(ref_answer), re.IGNORECASE)
                new_q = pattern.sub(f"({ref_num})", q, count=1)
                if new_q != q:
                    found_any = True
                    q = new_q
            if not found_any:
                raise ValueError(
                    f"Hop {hop['hop_number']} question doesn't contain any previous answer. "
                    f"Q={hop['question']!r}, previous answers={[h['answer'] for h in hops[:i]]}"
                )

        num = hop["hop_number"]
        if i == len(hops) - 1 and len(hops) > 1:
            prefix = f"{num} (final)."
        else:
            prefix = f"{num}."

        lines.append(f"{prefix} {q}")

    return "\n".join(lines)
