#!/usr/bin/env python3
"""Create q30 Privacy HopQA dataset with final residual review fixes.

This is an incremental layer over q29. It only applies fixes found while
reviewing the 24 remaining q29 gold-evidence audit residuals:

- broad answer variants for accepted equivalent forms;
- question rewrites where the gold document supports the answer but the wording
  made nearby distractor values too attractive;
- no row drops and no bridge-answer changes.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import apply_privacy_hopqa_q29_residual_answerability_fixes_2026_05_13 as q29
import build_privacy_hopqa_first_web_hop_dataset as firstweb_builder

ROOT = Path(__file__).resolve().parents[1]
SOURCE_VERSION = "2026-05-13_q29_residual_answerability"
TARGET_VERSION = "2026-05-13_q30_final_residual_review"
PATCH_TAG = "final_residual_review_fixes_q30_2026-05-13"
FINAL_DIR = ROOT / "data_files/privacy_hopqa_final_dataset"
FIRST_WEB_DIR = ROOT / "data_files/privacy_hopqa_first_web_hop_dataset"


QUESTION_REWRITES: dict[tuple[str, int], str] = {
    ("c963c6c1", 2): "Which driver, blocked by Meira, Kimball and Viso, decelerated to 10% throttle before hitting Kimball's car?",
    ("a541ffb3", 2): "What monthly salary does the Governor's budget propose for Capital Fellows after the 50% salary increase?",
    ("7e8c5b43", 4): "In the expat article's Top 10 countries hosting the most expats table, which country is ranked number 2?",
    ("260f1951", 2): "During which quarter and year was Elexion's mobile app updated with the service-appointment scheduling feature used by 15% of app users?",
    ("741d6389", 1): "By what percentage did MediConn Solutions increase its customer base in 2023?",
    ("e0220515", 2): "Which company's virtual healthcare platform operates under a 2023 cybersecurity framework that complies with PIPEDA and PHIPA?",
    ("fb87c058", 2): "During which quarter and year did Elexion's European battery alliance improve supply chain efficiency by 15%?",
    ("23f28b48", 2): "In what year did Elexion invest $25 million to establish the dedicated SDV development team with 100 engineers?",
    ("2f46e670", 2): "By what quarter and year had MediConn onboarded 15 new wellness-platform clients with an average employee participation rate of 70%?",
    ("2e385141", 2): "By what percent did HR streamlining increase on-time documentation completions?",
    ("9aa123c4", 4): "In what year were the critical changes implemented by Elexion's Sales Operations team?",
    ("2006ec4d", 4): "In the sentence listing Nissan, Toyota, Mini and Vauxhall as volume car manufacturers with a major UK presence, which manufacturer is listed first?",
    ("76768e44", 3): "In the Tesla Q3 2024 sales table, what final rightmost number appears after the two consecutive 439,975 entries?",
    ("ae5e680e", 3): "In the Tesla Q3 2024 sales table, what final rightmost number appears after the two consecutive 439,975 entries?",
}


VARIANT_ADDITIONS: dict[tuple[str, int], list[str]] = {
    ("0bc0d504", 5): [
        "Vietnam Chamber of Commerce and Industry (VCCI)",
        "Vietnam Chamber of Commerce and Industry",
        "Vietnam Chamber of Commerce",
    ],
    ("a541ffb3", 2): [
        "$4,888",
        "$4888",
        "4,888",
        "4888",
        "4,888 dollars",
        "4888 dollars",
    ],
    ("c920a3dc", 3): ["Jan", "Jan.", "1 January", "January 2020"],
    ("fb4832ae", 2): ["RRC", "the RRC", "Reverse Record Check (RRC)"],
    ("fb87c058", 2): ["fourth quarter 2023", "fourth quarter of 2023", "Q4 of 2023"],
    ("741d6389", 1): ["15", "15 percent", "fifteen percent"],
    ("ed65beab", 3): [
        "under 18 years",
        "less than 18 years",
        "under 18",
        "less than 18",
        "aged under 18 years",
        "aged less than 18 years",
    ],
    ("9fcbaa2c", 3): ["weekly meeting", "every week", "once a week"],
    ("5a1dc316", 2): ["20", "20 percent", "twenty percent"],
    ("5e6566fc", 5): [
        "around 200",
        "approximately 200",
        "about 200",
        "200 staff",
        "200 employees",
        "around 200 staff",
        "around 200 employees",
    ],
    ("2e385141", 2): ["30", "30 percent", "thirty percent"],
    ("5dd85400", 2): ["Wislar et al.", "Wislar et al"],
    ("235a5231", 3): ["PDE record", "the PDE record"],
    ("693d7299", 1): ["as of October 2025"],
    ("c621802b", 7): ["mid-2022", "July"],
    ("a922055c", 2): ["US", "U.S", "USA", "United States", "United States of America"],
    ("2f46e670", 2): ["Q1 2025", "first quarter 2025", "first quarter of 2025"],
    ("23f28b48", 2): ["2023"],
    ("260f1951", 2): ["Q4 2024", "fourth quarter 2024", "fourth quarter of 2024"],
    ("e0220515", 2): ["MediConn Solutions", "MediConn's"],
    ("7e8c5b43", 4): ["Germany", "Federal Republic of Germany"],
    ("76768e44", 3): ["462890", "462,890 total sales", "462890 total sales"],
    ("ae5e680e", 3): ["462890", "462,890 total sales", "462890 total sales"],
    ("518434d3", 5): [
        "2019 to 2023",
        "2019 through 2023",
        "2019-23",
        "2019 to 23",
    ],
}


VARIANT_REMOVALS: dict[tuple[str, int], list[str]] = {
    ("0009fef7", 3): [
        "Q4 2024 or Q1 2025",
        "4",
        "2024",
        "1",
        "2025",
    ],
    ("339b1a86", 2): [
        "2022-2023",
        "-2023",
    ],
    ("518434d3", 5): [
        "2009-2023",
        "2019",
        "-2023",
        "2009",
    ],
}


SUPPORTING_SPAN_REWRITES: dict[tuple[str, int], str] = {
    ("c963c6c1", 2): "Meira, Kimball and Viso blocked Wheldon's path, and he decelerated to 10% of throttle usage before he hit the left rear wheel of Kimball's car.",
    ("a541ffb3", 2): "The Governor's budget proposes a 50 percent salary increase for Capital Fellows, bringing their monthly salary to $4,888 ($58,656 annualized).",
    ("7e8c5b43", 4): "Top 10 countries hosting the most expats (2025): Rank 1 United States; Rank 2 Germany; Rank 3 Saudi Arabia; Rank 4 United Arab Emirates.",
    ("260f1951", 2): "In Q4 2024, the app was updated with a new feature to schedule service appointments, which was used by 15% of app users.",
    ("e0220515", 2): "MediConn's virtual healthcare platform operates under a robust cybersecurity framework developed in 2023. Our platform is in compliance with PIPEDA and PHIPA.",
    ("fb87c058", 2): "In Q4 2023, we formed a strategic alliance with a European battery manufacturer, which improved our supply chain efficiency by 15%.",
    ("23f28b48", 2): "In 2023, we invested $25M to establish a dedicated SDV development team. This team consists of 100 software engineers and data scientists.",
    ("2f46e670", 2): "By Q1 2025, we had already onboarded 15 new clients to our wellness platform, with an average employee participation rate of 70%.",
    ("2e385141", 2): "HR streamlining has led to a 30% increase in on-time documentation completions.",
    ("741d6389", 1): "in 2023, increased their customer base by 15%, significantly outperforming the market average.",
    ("9aa123c4", 4): "This report outlines the critical changes implemented by Elexion Automotive's Sales Operations team in 2025, designed to enhance the first impressions formed by potential clients.",
    ("2006ec4d", 4): "Volume car manufacturers with a major presence in the UK include Nissan, Toyota, Mini and Vauxhall.",
    ("c4742644", 1): "Similar to other Disney characters, the Muppets appear at Disney theme parks, having first appeared at Walt Disney World Resort in 1990.",
    ("76768e44", 3): "| Q3 2024 | 6,890,799 | 469,796 | | 22,915 | 22,915 | 439,975 | 439,975 | 462,890 |",
    ("ae5e680e", 3): "| Q3 2024 | 6,890,799 | 469,796 | | 22,915 | 22,915 | 439,975 | 439,975 | 462,890 |",
}


BAD_STYLE_QUESTION_REWRITES: dict[tuple[str, int], str] = {
    ("01385f60", 1): "Which Automotive Cloud capability is Elexion Automotive considering to improve metric data quality across its dealer network?",
    ("01385f60", 2): "Which cloud product's dealer performance management tools did Elexion consider for dealer relationship enhancement?",
    ("0009fef7", 3): "Which Tesla reporting quarter came before (2)?",
    ("03176ebf", 2): "Which four-letter research center studied 57 community colleges after reporting that (1) of public four-year remedial coursetakers completed none?",
    ("0793610b", 2): "Whose podcast episode explained how the show reached more than 700 million downloads after outsourcing ad sales cost more than (1)?",
    ("07981dac", 1): "What year is listed as the latest release year for the Microsoft Flight Simulator series?",
    ("080d67f1", 5): "What was the same survey's June 2023 home-delivery preference before it rose to 45% in June 2024?",
    ("0e82260f", 3): "Which club won the Premier League season that ended in (2)?",
    ("16213b4e", 3): "As of the end of which year had Basel III standards already been implemented in the UK through onshored EU legislation?",
    ("19113d56", 4): "What year had already seen 386 reported health-care cyberattacks when the February ransomware attack on Change Healthcare was discussed?",
    ("1af258d6", 7): "By more than what percentage was the initial AeRO offer oversubscribed before the share offer was expanded?",
    ("1c4e2076", 2): "What reporting frequency (e.g., annual, quarterly, or monthly) did Elexion use for employee ID photo update compliance rates?",
    ("1fcb1380", 4): "Which HEDIS measurement year had both its Summary of Changes and Measure Descriptions posted on August 1, 2024?",
    ("2006ec4d", 4): "Which volume car manufacturer with a major UK presence operates the Sunderland plant?",
    ("235a5231", 3): "On what Medicare Part D record type are the Ingredient Cost, Dispensing Fee, Vaccine Administration, Sales Tax, GDCB, GDCA, and TGCDC Accumulator fields reported?",
    ("2b88e41b", 2): "During which year did Lee's Market report working with three major shipping partners: TransCo, FreightLink, and ShipFast?",
    ("3169f1c9", 2): "In what year did the international marine-fuel sulfur cap cut sulfur emissions from large ships by about 85%?",
    ("37f75b26", 2): "What year did MediConn associate with the IT refactoring effort that produced a (1) increase in system uptime?",
    ("3f40e4e2", 5): "Which Bronx plan did the Adams administration pass in 2024 while advancing four other neighborhood-specific plans?",
    ("42da4e24", 1): "In what year did Jose Altuve fracture his right thumb in the World Baseball Classic, have surgery in March, and rehab with Triple-A Sugar Land in May?",
    ("4a51cddd", 1): "Under the International Journal of Hematology guidelines, authors of Case Reports should summarize new findings within how many words?",
    ("4c2a9010", 3): "Which region's Lee's Market store branding was matched by colored LED keyboard lights?",
    ("4de89cfc", 2): "During which quarter and year did Elexion discuss its R&D budget and innovation work involving (1) engineers?",
    ("54c40961", 3): "The state-school admissions rate was 68.1% in 2022; what was it in (2)?",
    ("56765d3e", 5): "Which state set a non-binding goal of 100% zero-emission buses by 2035 alongside Michigan and Washington D.C.?",
    ("5dd85400", 2): "Which lead author investigated inappropriate authorship in high-impact general medical journal articles where (1) had honorary or ghost authorship?",
    ("643e6343", 5): "How much money was Instacart losing each month in 2019 before it recorded its first monthly profit in April 2020?",
    ("650b1786", 4): "In which month and year did logistics account for 25% of Lee's Market's workforce?",
    ("693d7299", 1): "As of what month and year were 87% of MediConn Solutions staff compliant with device-usage regulations?",
    ("717890de", 2): "What fiscal year label did MediConn use for the virtual consultations performance analysis that documented a (1) reduction in resource utilization?",
    ("76768e44", 3): "What quarterly total vehicle sales count did Tesla report for (2) 2024 across Model S/X and Model 3/Y vehicles?",
    ("7e8c5b43", 4): "Which country hosted 16-17 million expats in the top expat-hosting countries list?",
    ("888a25fc", 4): "Which smartphone ecosystem, alongside Samsung users, did Elexion aim to attract with its connectivity roadmap?",
    ("8a619911", 4): "How many technology-division employees did Wayfair plan to let go as part of its restructuring effort, in the same layoff roundup that noted HPE would cut (3) employees?",
    ("8bb220d4", 2): "Which year did MediConn's patient lifetime value analysis cover after identifying the (1) revenue share?",
    ("8c49d7ef", 7): "What was the size of the most-valuable European companies list for (6)?",
    ("9aa123c4", 2): "After identifying that (1) of leads came from the West Coast, which year did Elexion's Q2 referral program analysis cover?",
    ("9bff1e81", 2): "In what year did technical training account for (1) of all training types at Elexion?",
    ("9cb52bd4", 2): "Which year did Elexion's luxury garage analysis assess after identifying the (1) satisfaction level?",
    ("a7814344", 2): "Which Supermassive Games release was planned for (1) in The Dark Pictures anthology?",
    ("a83e51e0", 3): "Which world region is included with North America and Asia among the three market regions represented by the Adidas trefoil leaves?",
    ("a918dc22", 2): "Which bill is tied to the CPPA Act in MediConn's (1) compliance preparation?",
    ("ae5e680e", 3): "What quarterly total vehicle sales count did Tesla report for (2) across Model S/X and Model 3/Y vehicles?",
    ("b0d25dd6", 2): "Which year did Lee's Market associate with its refrigeration unit overhaul analysis?",
    ("b6db2916", 4): "What percent increase in sales productivity did MediConn report before the (3) rise in patient referrals?",
    ("bff7f93c", 5): "Which agency enforces the Clean Air Act ozone-layer regulations in the action dated (4)?",
    ("c3bc0f66", 3): "When the 2023 plugin share was (2), what BEV share was shown in parentheses with that plugin share?",
    ("c4742644", 1): "In what year did the Muppets first appear at Walt Disney World Resort?",
    ("c621802b", 7): "After Russia shifted to a war footing in mid-2022, from which month in 2022 did Russian corporate debt abruptly begin to surge?",
    ("c920a3dc", 2): "Which database indexed 23,634 unique COVID-19-related articles alongside Scopus between January and June 2020?",
    ("ce9f5e31", 4): "What abbreviation identifies the Department for Education in the discussion of a 15% rise in secondary school pupils between 2018 and 2027?",
    ("d04d0437", 4): "Among U.S. adults aged 18-29, which social platform had 76% usage?",
    ("d1e44ce5", 3): "Which Ethan coauthored the oculomics article with Joshua Ong and Mouayad Masalkhi?",
    ("dbfdeeaa", 5): "Which tool was used alongside AMSTAR-2 and ROBIS to evaluate outcomes in the bone-mineral-density overview?",
    ("dd17b63c", 2): "After identifying the (1) sales-volume increase, which quarter and year did Elexion's employee retention incentives analysis cover?",
    ("e0220515", 3): "What month and year did MediConn's legacy authentication risk analysis cover after identifying (2)'s legacy authentication modules?",
    ("f774a5d3", 2): "In what year did Elexion report that (1) of mid-range model buyers were first-time EV owners?",
    ("fb87c058", 3): "In Tesla's Q4 2023 row, what quarterly production figure is listed next to cumulative production of 5,576,801?",
    ("ffedb285", 1): "In what year was Pink's single 'What About Us' released before the 2024 Democratic National Convention performance with Willow?",
}

QUESTION_REWRITES.update(BAD_STYLE_QUESTION_REWRITES)

TARGET_KEYS = set(QUESTION_REWRITES) | set(VARIANT_ADDITIONS) | set(SUPPORTING_SPAN_REWRITES) | set(VARIANT_REMOVALS)


def remove_variants(hop: dict[str, Any], variants: list[str]) -> list[str]:
    remove_set = {str(value).strip().lower() for value in variants if str(value).strip()}
    removed: list[str] = []
    for field in ("accepted_answer_variants", "alternate_valid_answers"):
        kept: list[str] = []
        for value in hop.get(field) or []:
            text = str(value).strip()
            if text and text.lower() in remove_set:
                removed.append(text)
            else:
                kept.append(value)
        if kept:
            hop[field] = kept
        elif field in hop:
            hop[field] = []
    return removed


def remove_global_variants(row: dict[str, Any], variants: list[str]) -> list[str]:
    remove_set = {str(value).strip().lower() for value in variants if str(value).strip()}
    kept: list[str] = []
    removed: list[str] = []
    for value in row.get("global_answer_variants") or []:
        text = str(value).strip()
        if text and text.lower() in remove_set:
            removed.append(text)
        else:
            kept.append(value)
    if removed:
        row["global_answer_variants"] = kept
    return removed


def patch_rows(rows: list[dict[str, Any]]) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    out: list[dict[str, Any]] = []
    logs: list[dict[str, Any]] = []
    for row in rows:
        row = json.loads(json.dumps(row))
        chain_id = str(row.get("chain_id"))
        changed = False
        for hop in row.get("hops", []):
            key = (chain_id, int(hop.get("hop_number", -1)))
            if key not in TARGET_KEYS:
                continue
            old = json.loads(json.dumps(hop))
            added_variants: list[str] = []
            removed_variants: list[str] = []
            if key in QUESTION_REWRITES:
                hop["question_templated"] = QUESTION_REWRITES[key]
            if key in SUPPORTING_SPAN_REWRITES:
                hop["supporting_span"] = SUPPORTING_SPAN_REWRITES[key]
            if key in VARIANT_ADDITIONS:
                added_variants = q29.add_variants(hop, VARIANT_ADDITIONS[key])
            if key in VARIANT_REMOVALS:
                removed_variants = remove_variants(hop, VARIANT_REMOVALS[key])
            resolutions = list(hop.get("audit_resolutions") or [])
            resolutions.append(
                {
                    "resolution": "targeted_final_residual_review_repair",
                    "source": PATCH_TAG,
                    "old_question_templated": old.get("question_templated"),
                    "new_question_templated": hop.get("question_templated"),
                    "old_answer": old.get("answer"),
                    "new_answer": hop.get("answer"),
                    "old_doc_id": old.get("doc_id"),
                    "new_doc_id": hop.get("doc_id"),
                    "old_doc_ref": old.get("doc_ref"),
                    "new_doc_ref": hop.get("doc_ref"),
                    "added_answer_variants": added_variants,
                    "removed_answer_variants": removed_variants,
                    "old_supporting_span": old.get("supporting_span"),
                    "new_supporting_span": hop.get("supporting_span"),
                    "rationale": "Fixes final q29 residuals and critical/high bad-style questions where answer variants were too narrow, wording made a nearby distractor value attractive, or the original question required document-position/title/phrase awareness. Bridge answers are unchanged.",
                }
            )
            hop["audit_resolutions"] = resolutions
            logs.append(
                {
                    "chain_id": chain_id,
                    "hop_number": int(hop.get("hop_number")),
                    "patch_tag": PATCH_TAG,
                    "old_question_templated": old.get("question_templated"),
                    "new_question_templated": hop.get("question_templated"),
                    "old_answer": old.get("answer"),
                    "new_answer": hop.get("answer"),
                    "old_doc_id": old.get("doc_id"),
                    "new_doc_id": hop.get("doc_id"),
                    "old_doc_ref": old.get("doc_ref"),
                    "new_doc_ref": hop.get("doc_ref"),
                    "added_answer_variants": added_variants,
                    "removed_answer_variants": removed_variants,
                    "old_supporting_span": old.get("supporting_span"),
                    "new_supporting_span": hop.get("supporting_span"),
                }
            )
            changed = True
        if changed:
            q29.rebuild_questions(row)
            final_hop = (row.get("hops") or [])[-1] if row.get("hops") else None
            if final_hop is not None:
                final_key = (chain_id, int(final_hop.get("hop_number", -1)))
                if final_key in VARIANT_ADDITIONS:
                    global_variants = list(row.get("global_answer_variants") or [])
                    seen = {str(value).strip().lower() for value in global_variants if str(value).strip()}
                    for value in final_hop.get("accepted_answer_variants") or []:
                        text = str(value).strip()
                        if text and text.lower() not in seen:
                            global_variants.append(text)
                            seen.add(text.lower())
                    row["global_answer_variants"] = global_variants
                if final_key in VARIANT_REMOVALS:
                    remove_global_variants(row, VARIANT_REMOVALS[final_key])
            row["annotation_ts"] = PATCH_TAG
            row["review_status"] = "finalized_answerability_fixes_q30"
        out.append(row)
    return out, logs


def build_first_web_outputs() -> dict[str, Any]:
    output_basename = f"privacy_hopqa_first_web_hop_anywhere_{TARGET_VERSION}"
    all_rows, next_index = firstweb_builder.build_split(
        q29.read_jsonl(FINAL_DIR / f"privacy_hopqa_finalized_{TARGET_VERSION}.jsonl"),
        1,
        output_basename,
        "first_web_hop_anywhere",
    )
    split_rows = {}
    for split in ("train", "val", "test"):
        split_rows[split], next_index = firstweb_builder.build_split(
            q29.read_jsonl(FINAL_DIR / f"privacy_hopqa_finalized_{TARGET_VERSION}_{split}.jsonl"),
            next_index,
            output_basename,
            "first_web_hop_anywhere",
        )
    q29.write_jsonl(FIRST_WEB_DIR / f"{output_basename}.jsonl", all_rows)
    for split, rows in split_rows.items():
        q29.write_jsonl(FIRST_WEB_DIR / f"{output_basename}_{split}.jsonl", rows)
    manifest = {
        "dataset": "privacy_hopqa_first_web_hop",
        "source_basename": f"privacy_hopqa_finalized_{TARGET_VERSION}",
        "output_basename": output_basename,
        "selection_mode": "first_web_hop_anywhere",
        "all": firstweb_builder.summarize(all_rows),
        "splits": {split: firstweb_builder.summarize(rows) for split, rows in split_rows.items()},
    }
    (FIRST_WEB_DIR / f"manifest_{output_basename}.json").write_text(
        json.dumps(manifest, indent=2, sort_keys=True, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    return manifest


def main() -> None:
    source_rows = q29.read_jsonl(FINAL_DIR / f"privacy_hopqa_finalized_{SOURCE_VERSION}.jsonl")
    all_logs: list[dict[str, Any]] = []
    for suffix in ("", "_train", "_val", "_test"):
        src = FINAL_DIR / f"privacy_hopqa_finalized_{SOURCE_VERSION}{suffix}.jsonl"
        if src.exists():
            patched, logs = patch_rows(q29.read_jsonl(src))
            q29.write_jsonl(FINAL_DIR / f"privacy_hopqa_finalized_{TARGET_VERSION}{suffix}.jsonl", patched)
            all_logs.extend(logs)
    q29.write_jsonl(FINAL_DIR / f"privacy_hopqa_finalized_{TARGET_VERSION}.applied_log.jsonl", all_logs)
    first_web_manifest = build_first_web_outputs()
    manifest = {
        "source_version": SOURCE_VERSION,
        "target_version": TARGET_VERSION,
        "patch_tag": PATCH_TAG,
        "chains": len(source_rows),
        "hops": sum(len(row.get("hops", [])) for row in source_rows),
        "variant_addition_hops": [f"{chain_id}::hop{hop}" for chain_id, hop in sorted(VARIANT_ADDITIONS)],
        "question_rewrite_hops": [f"{chain_id}::hop{hop}" for chain_id, hop in sorted(QUESTION_REWRITES)],
        "supporting_span_rewrite_hops": [f"{chain_id}::hop{hop}" for chain_id, hop in sorted(SUPPORTING_SPAN_REWRITES)],
        "first_web_counts": first_web_manifest,
        "notes": [
            "No rows dropped and no bridge answers changed.",
            "Question rewrites preserve prior-hop placeholders and final/downstream bridge answers.",
            "Residual review focused on questions that should be easy when the gold document is shown.",
            "Variant additions are semantically equivalent forms, not nearby contradictory values.",
        ],
    }
    (FINAL_DIR / f"manifest_privacy_hopqa_finalized_{TARGET_VERSION}.json").write_text(
        json.dumps(manifest, indent=2, sort_keys=True, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(manifest, indent=2, sort_keys=True, ensure_ascii=False))


if __name__ == "__main__":
    main()
