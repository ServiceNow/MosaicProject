"""Prompt templates for chain generation and verification."""

LEGACY_PROMPT_CONSTRAINED = '''\
You are writing one question in a multi-hop QA chain.

PREVIOUS ANSWER (must appear in your question): "{prev_answer}"
TARGET ANSWER (the correct answer to your question): "{target_answer}"

Rules for the question:
- The PREVIOUS ANSWER must appear VERBATIM in your question - copy it exactly, do not paraphrase or abbreviate (e.g. do not use "2025" if the answer is "Q2 2025").
- The PREVIOUS ANSWER must drive the question - it must not be ancillary. Using [PREV] to identify the subject is fine.
  BAD: "Who stars in The Wolf of Wall Street and was born in [PREV]?" ([PREV] in a side clause - answer is Leo regardless)
  GOOD: "In the company that created the iPhone that came out in year [PREV], what was their revenue?" ([PREV] identifies the subject)
- The question must move the chain forward. The answer must be a genuinely new fact, not a restatement, trivial reformat, precision tweak, or date refinement of an earlier answer.
- The answer must be 1-3 words, the entity alone (never "the X" - use "Health Cloud" not "the Health Cloud").
- CRITICAL: The answer must NOT be determinable without knowing the PREVIOUS ANSWER.
  The PREVIOUS ANSWER must discriminate between multiple possible answers - changing its value should change the answer. If only one entity could ever be the answer regardless of the PREVIOUS ANSWER, the question is bad.
  BAD: "Who stars in The Wolf of Wall Street and was born in [PREV]?" (answer is Leo regardless)
  GOOD: "What film starring [PREV] was released in 2013?" (answer depends on [PREV])
- The question must be SPECIFIC enough to have exactly one correct answer.
- The question must be SELF-CONTAINED and make sense out of context.
- Write concrete, searchable questions that require the agent to search - avoid document-centric phrasing ("this paper", "according to the document") that obscures what to look up.
- Use concrete wording. Avoid vague terms like "implementations" or "aligned with" unless they have clear meaning.
- Without knowing the PREVIOUS ANSWER, the question should be unanswerable.

Find evidence in the document that connects "{prev_answer}" to "{target_answer}". Quote the relevant excerpt(s) - one or more sentences, possibly disjoint. Use " | " to separate multiple excerpts if needed. Then write the question based on that evidence.

Format: QUOTE may span multiple sentences (use " | " for disjoint excerpts). QUESTION must be a single line - one complete question, no line breaks.

Example:
Previous answer: "Salesforce"
Target answer: "Health Cloud"
QUOTE: Salesforce's Health Cloud platform enables healthcare organizations to manage patient relationships.
REASONING: The excerpt says Salesforce offers Health Cloud. I can ask what platform Salesforce offers.
QUESTION: What healthcare platform does Salesforce offer?
ANSWER: Health Cloud

<document>
{doc_text}
</document>

<answer>
QUOTE: <excerpt(s) from document - one or more sentences, use " | " to separate disjoint excerpts>
REASONING: ...
QUESTION: <single line, one complete question, must contain "{prev_answer}">
ANSWER: {target_answer}
</answer>'''

LEGACY_PROMPT_PICK = '''\
You are writing one question in a multi-hop QA chain.

The PREVIOUS ANSWER is a private enterprise value. Write a question about a PUBLIC web document that requires the PREVIOUS ANSWER in a web search query to find the answer.

PREVIOUS ANSWER (must appear in your question): "{prev_answer}"

Search the document below for sentences that connect the PREVIOUS ANSWER to one of the ENTITIES. Your chosen entity must appear in the document and have a clear factual link to the PREVIOUS ANSWER. Pick the answer from the ENTITIES list below - it must be one of those exact strings.

Rules for the question:
- The PREVIOUS ANSWER must drive the question - it must not be ancillary. Using [PREV] to identify the subject is fine.
  BAD: "Who stars in The Wolf of Wall Street and was born in [PREV]?" ([PREV] in a side clause - answer is Leo regardless)
  GOOD: "In the company that created the iPhone that came out in year [PREV], what was their revenue?" ([PREV] identifies the subject)
- The question must move the chain forward. The answer must be a genuinely new fact, not a restatement, trivial reformat, precision tweak, or date refinement of an earlier answer.
- The answer must be one of the entities below (match exactly), the entity alone (never "the X"). Maximum 3 words.
- Pick the entity with the strongest factual link to the PREVIOUS ANSWER in the document. The entity must appear in the document text.
- CRITICAL: The answer must NOT be determinable without knowing the PREVIOUS ANSWER.
  The PREVIOUS ANSWER must discriminate between multiple possible answers - changing its value should change the answer.
  BAD: "Who stars in The Wolf of Wall Street and was born in [PREV]?" (answer is Leo regardless)
  BAD: "What firm reported [PREV] of companies published ESG reports?" (only one firm does this study)
  BAD: If someone could search for another unique phrase already in the question and find the answer without the PREVIOUS ANSWER, the question is bad.
  GOOD: "What regulation does [PREV] comply with?" (answer depends on [PREV])
- The question must be SPECIFIC enough to have exactly one correct answer.
- The PREVIOUS ANSWER should be the key identifying detail that pins down the answer.
- The question must be SELF-CONTAINED and make sense out of context.
- Write concrete, searchable questions that require the agent to search - avoid document-centric phrasing that obscures what to look up.
- Use concrete wording. Avoid vague terms. Without knowing the PREVIOUS ANSWER, the question should be unanswerable.

ENTITIES:
{entity_list}

Find evidence in the document that connects "{prev_answer}" to your chosen entity. Quote the relevant excerpt(s) - one or more sentences, possibly disjoint. Use " | " to separate multiple excerpts if needed. Then write the question based on that evidence.

Format: QUOTE may span multiple sentences (use " | " for disjoint excerpts). QUESTION must be a single line - one complete question, no line breaks.

Example:
Previous answer: "Health Cloud"
Entities: FDA, HIPAA, CMS
QUOTE: Protecting sensitive patient data through HIPAA-compliant features in Health Cloud is essential for maintaining privacy.
REASONING: The excerpt says Health Cloud must be HIPAA-compliant. HIPAA has the strongest link.
QUESTION: What regulation must Health Cloud comply with to maintain privacy?
ANSWER: HIPAA

<document>
{doc_text}
</document>

<answer>
QUOTE: <excerpt(s) from document - one or more sentences, use " | " to separate disjoint excerpts>
REASONING: ...
QUESTION: <single line, one complete question, must contain "{prev_answer}">
ANSWER: ...
</answer>'''

LEGACY_PROMPT_RANK = '''\
You are evaluating questions for a multi-hop QA chain.
The previous answer was "{prev_answer}". Rank by these criteria (most important first):

1. SPECIFICITY: The question should have exactly one correct answer. Prefer questions that include identifying details (names, dates, organizations) so they can't be confused with a different context that also mentions "{prev_answer}".
   WORSE: "How many visits does $1,200 cover?" (ambiguous - many things cost $1,200)
   BETTER: "How many visits to Karst Stages park does $1,200 cover?" (unique answer)
2. CENTRALITY: "{prev_answer}" should drive the question, not be ancillary.
3. NATURALNESS: The question should sound like something a person would actually ask.
4. SEARCHABILITY: The question should give enough information (when combined with the previous answer) to find the relevant document. Prefer questions where prev_answer + question together suffice to search effectively; avoid document-centric phrasing ("the document", "this report") that doesn't add searchable detail.

Rank these candidates from best to worst. Output ONLY the ranking as a comma-separated list of numbers (e.g. "2, 1, 3").

CANDIDATES:
{candidates}

RANKING (best to worst):'''

PROMPT_CONSTRAINED = """\
Write the next question in this multi-hop QA chain.

Previous question:
{prev_question}

Previous answer:
"{prev_answer}"

Target answer:
"{target_answer}"

Source excerpt:
{source_excerpt}

Current document:
{doc_text}

Write one simple question that uses "{prev_answer}" exactly and asks for one new fact from this document.
Think through a few possible questions first, then choose the best honest one.
Do not force a relationship. If the document only co-mentions the values, or if the hop would require guesswork, arithmetic, or a made-up connection, output NO_VALID_QUESTION.
Avoid generic "what year/date/quarter is associated with..." questions unless that date is truly the fact being asked for.
If the answer is a date or time, use the full supported granularity rather than a fragment.
The question must uniquely identify the answer from the quoted evidence. If multiple people, places, sections, dates, or values in the excerpt could fit, output NO_VALID_QUESTION.
Write exactly one sentence total. Do not write a setup sentence followed by a second question or any extra sentence fragment.
Use the previous answer as an inline part of that single sentence, not as a separate lead-in.
The goal is to use the previous answer to identify the right subject, event, or record in this document, then ask for exactly one new fact about that identified subject.
If the answer is a named entity, ask for the full entity, not a partial fragment like a surname, section numeral, or single generic word unless the source itself only gives that fragment.
Prefer distinctive, interesting answers and questions over low-information ones. Avoid generic targets like a bare year, quarter, section label, ordinal word, or generic capitalized word unless that is clearly the central fact.
Make the question hard to misread as being about some other person, event, or document that also mentions the same bridge.
Use only the minimum detail needed to uniquely identify the subject. Do not pile on extra descriptors once the subject is already pinned down.
As a soft preference, favor entities and facts that are broadly meaningful or likely to matter to a company researcher: organizations, products, policies, markets, hospitals, countries, technologies, roles, or durable metrics.
Avoid dead-end trivia when possible, especially random individual people, one-off titles, or isolated labels that are unlikely to support a useful next hop, unless the person or label is clearly the main subject of the excerpt.

Good example:
- Previous answer: "2023"
- Target answer: "NHS"
- Better question: Which healthcare organization reported a 15% increase in pharmacist training places in 2023?
- Bad question: What year is mentioned in the report about pharmacist training places?
- Why bad: it asks for a low-information date rather than the more meaningful organization the date helps identify.

Good example:
- Previous answer: "Q1 2022"
- Target answer: "Q4 2024"
- Good question: Which quarter's survey data is presented in the report describing a quarterly employee recognition program that began in Q1 2022?
- Bad question: The quarterly employee recognition program started in Q1 2022. Which quarter's survey data is presented in this report?
- Why bad: it is two sentences instead of one.

End with exactly one final block using these labels:
QUOTE: <exact supporting excerpt(s) from the document; use " | " if needed>
REASONING: <brief explanation of why this is the best honest option; keep this short and do not reveal long chain-of-thought>
QUESTION: <one single-line question containing "{prev_answer}">
ANSWER: {target_answer}
"""

PROMPT_PICK = """\
Write the next question in this multi-hop QA chain.

Previous question:
{prev_question}

Previous answer:
"{prev_answer}"

Bridge entity that appears in the next document:
"{bridge_value}"

Source excerpt from the current document:
{source_excerpt}

Target excerpt from the next document:
{target_excerpt}

Allowed next answers from the target excerpt:
{allowed_answers}

Write one simple question about the next document.
Use "{prev_answer}" exactly in the question.
The bridge explains why this next document is relevant. Use it to orient yourself, then ask for one different fact from the allowed answers.
Think step by step through several allowed-answer options before choosing the best one.
Reject options that only restate the bridge, ask for shallow "what year/date/quarter" trivia, rely on arithmetic, or depend on a made-up relationship.
Prefer a concrete, central fact from the target excerpt.
Copy the final answer exactly from the allowed next answers.
Do not invent a relationship, force arithmetic, or pick an answer from outside the target excerpt.
Do not use document-centric wording like "according to the document", "in the excerpt", or "from this article".
Avoid generic chronology questions unless the date itself is the real fact being asked for.
The question must uniquely identify the answer from the quoted evidence. If multiple people, entities, sections, or values in the excerpt could fit, reject that option and choose a different one.
Write exactly one sentence total. Do not write a lead-in statement followed by a question or any extra sentence fragment.
The goal is to use the previous answer to identify the right subject, event, or record in the next document, then ask exactly one new fact about that identified subject.
If the answer is a named entity, prefer the full entity over a partial fragment. For people, prefer the full name rather than a surname only unless the excerpt only gives the surname.
Prefer distinctive, interesting answers and questions over low-information ones. Avoid generic targets like a bare year, quarter, section label, ordinal word, or generic capitalized word unless that is clearly the central fact.
Make the question hard to misread as being about some other person, event, or document that also mentions the same bridge.
Use only the minimum detail needed to uniquely identify the subject. Do not pile on extra descriptors once the subject is already pinned down.
As a soft preference, favor entities and facts that are broadly meaningful or likely to matter to a company researcher: organizations, products, policies, markets, hospitals, countries, technologies, roles, or durable metrics.
Avoid dead-end trivia when possible, especially random individual people, one-off titles, or isolated labels that are unlikely to support a useful next hop, unless that person or label is clearly the main subject of the excerpt.

Good example:
- Previous answer: "2023"
- Bridge: "2023"
- Allowed answers: NHS, 15%, pharmacist training places
- Better question: Which healthcare organization reported a 15% increase in pharmacist training places in 2023?
- Good answer: NHS
- Bad question: What year is mentioned in the report about pharmacist training places?
- Bad answer: 2023
- Why bad: it uses the bridge but asks for a low-information fact instead of a more meaningful subject from the excerpt.

Good example:
- Previous answer: "Q4 2024"
- Bridge: "Q4 2024"
- Allowed answers: Q1 2025, Internal Payroll
- Good question: Which other quarter, besides Q4 2024, is listed as acceptable for the internal payroll summary?
- Bad question: What is the other quarter, besides Q4 2024, that can be used for the internal payroll summary?
- Why bad: it is understandable, but still vague and document-like; prefer wording that more directly identifies the specific record being referenced.

End with exactly one final block using these labels:
QUOTE: <exact supporting excerpt(s) from the target excerpt; use " | " if needed>
REASONING: <brief explanation naming the main options you considered and why this is the best honest option; keep this short and do not reveal long chain-of-thought>
QUESTION: <one single-line question containing "{prev_answer}">
ANSWER: <one exact answer copied from the allowed next answers>
"""

PROMPT_RANK = """\
Choose the best next hop in this multi-hop QA chain.

Previous question:
{prev_question}

Previous answer:
"{prev_answer}"

Rank the candidates from best to worst.

Prefer the option where:
- the previous answer really helps find the next fact
- the answer is a new fact, not the bridge value again or obvious arithmetic
- the quote directly supports the question, not just the same document
- the question is specific, simple, and answerable from the cited text
- the answer stands alone as a complete entity, date, or value
- shallow chronology questions are only good when the date itself is the fact
- the answer is distinctive and interesting, not a low-information fragment like a surname, section numeral, generic capitalized word, or bare year/quarter unless that is clearly the core fact
- the question is hard to misinterpret as being about some other entity or document that also mentions the bridge
- one clean sentence beats a setup sentence plus a question
- the question uses only the minimum detail needed to uniquely identify the subject
- broadly meaningful, reusable entities are better than dead-end trivia when both are equally well supported

Candidates:
{candidates}

Output only a comma-separated ranking of candidate numbers, for example:
2, 1, 3
"""

PROMPT_TRANSITION_JUDGE = '''\
You are judging whether a candidate hop in a multi-hop QA chain is acceptable.

PREVIOUS HOPS:
{previous_hops}

BRIDGE VALUE (must be needed to answer the candidate question):
{bridge_value}

CANDIDATE QUESTION:
{candidate_question}

CANDIDATE ANSWER:
{candidate_answer}

SUPPORTING QUOTE:
{quote}

TARGET DOCUMENT EXCERPT:
{doc_excerpt}

HEURISTIC FLAGS:
{heuristic_flags}

Judge the candidate using these rules:
1. ANSWER_DELTA = PASS only if the candidate answer is a materially new fact relative to earlier answers.
   FAIL if it is mainly a restatement, trivial reformat, date refinement, precision tweak, or obvious narrowing of an earlier answer.
2. DEPENDENCY = PASS only if a capable agent would genuinely need the bridge value to answer the candidate question.
   FAIL if the question would still be answerable without the earlier answer, or if the bridge is mostly decorative.
3. COHERENCE is a 1-5 score used only for ranking candidates that pass both hard checks.
   Reward candidates that feel cohesive with earlier hops, stay in a similar domain or line of inquiry, and ask about facts that are central rather than incidental in the target document.

Output exactly these four lines:
ANSWER_DELTA: PASS or FAIL
DEPENDENCY: PASS or FAIL
COHERENCE: <integer 1-5>
JUSTIFICATION: <one sentence>
'''

REWORD_PROMPT = '''\
Fix this multi-hop question to improve clarity. Make minimal edits only.

PREVIOUS HOPS (for context):
{prev_hops}

CURRENT QUESTION TO FIX:
{question}

Rules:
1. The question should give enough information (with the previous answer) to find the relevant document. Prefer searchable phrasing; avoid document references ("the document", "this report") that don't add searchable detail.
2. Replace vague phrasing ("implementations", "aligned with") with clearer wording when possible.
3. Prefer linearity: when possible through tiny edits, structure the question so that answering previous questions is needed to search for the relevant document. The agent should need prior answers to formulate an effective search for this question. If this is impossible without larger changes, output the question as-is.
4. Remain fully deterministic: given the full answer chain, there must be exactly one correct answer. Avoid vague questions (e.g. "Who was born on Dec 20th 2012?" has multiple answers).
5. Keep the meaning and structure unchanged. Preserve back-references like (1), (2).
6. Output ONLY the rewritten question, nothing else.

Rewritten question:'''

CHAIN_VERIFICATION_TEMPLATE = """
This is a multi-hop question chain. Each question's answer feeds into the next \
question - references like (1) mean "the answer to question 1".

Your task: trace through ALL questions in order and provide the FINAL answer. \
You must answer every intermediate question to reach the final one. \
If ANY step cannot be answered from the context, respond NOT_ANSWERABLE.

CONTEXT (may contain multiple numbered documents, each labeled === Document N ===):
{context}

Each question may require information from a different document. If the relevant \
document for a step is not provided above, that step cannot be answered.

QUESTION CHAIN:
{question}

Trace through each question using ONLY the context above:
- For each question, find the answer in the context
- Substitute answers into later questions where (N) appears
- If you cannot answer ANY question in the chain, stop and say NOT_ANSWERABLE

<answer>
If all questions answerable: ANSWERABLE: <final answer only>
If any step fails: NOT_ANSWERABLE: <which question failed and why>
</answer>"""


def build_verification_prompt(context: str, question: str) -> str:
    return CHAIN_VERIFICATION_TEMPLATE.format(
        context=context if context else "(no context provided)",
        question=question,
    )
