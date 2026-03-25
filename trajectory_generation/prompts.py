"""Prompts for intra-doc (constrained), inter-doc (pick), ranking, reword, and verification stages."""

PROMPT_CONSTRAINED = '''\
You are writing one question in a multi-hop QA chain.

PREVIOUS ANSWER (must appear in your question): "{prev_answer}"
TARGET ANSWER (the correct answer to your question): "{target_answer}"

Rules for the question:
- The PREVIOUS ANSWER must appear VERBATIM in your question — copy it exactly, do not paraphrase or abbreviate (e.g. do not use "2025" if the answer is "Q2 2025").
- The PREVIOUS ANSWER must drive the question — it must not be ancillary. Using [PREV] to identify the subject is fine.
  BAD: "Who stars in The Wolf of Wall Street and was born in [PREV]?" ([PREV] in a side clause — answer is Leo regardless)
  GOOD: "In the company that created the iPhone that came out in year [PREV], what was their revenue?" ([PREV] identifies the subject)
- The answer must be 1-3 words, the entity alone (never "the X" — use "Health Cloud" not "the Health Cloud").
- CRITICAL: The answer must NOT be determinable without knowing the PREVIOUS ANSWER.
  The PREVIOUS ANSWER must discriminate between multiple possible answers — changing its value should change the answer. If only one entity could ever be the answer regardless of the PREVIOUS ANSWER, the question is bad.
  BAD: "Who stars in The Wolf of Wall Street and was born in [PREV]?" (answer is Leo regardless)
  GOOD: "What film starring [PREV] was released in 2013?" (answer depends on [PREV])
- The question must be SPECIFIC enough to have exactly one correct answer.
- The question must be SELF-CONTAINED and make sense out of context.
- Write concrete, searchable questions that require the agent to search — avoid document-centric phrasing ("this paper", "according to the document") that obscures what to look up.
- Use concrete wording. Avoid vague terms like "implementations" or "aligned with" unless they have clear meaning.
- Without knowing the PREVIOUS ANSWER, the question should be unanswerable.

Find evidence in the document that connects "{prev_answer}" to "{target_answer}". Quote the relevant excerpt(s) — one or more sentences, possibly disjoint. Use " | " to separate multiple excerpts if needed. Then write the question based on that evidence.

Format: QUOTE may span multiple sentences (use " | " for disjoint excerpts). QUESTION must be a single line — one complete question, no line breaks.

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
QUOTE: <excerpt(s) from document — one or more sentences, use " | " to separate disjoint excerpts>
REASONING: ...
QUESTION: <single line, one complete question, must contain "{prev_answer}">
ANSWER: {target_answer}
</answer>'''

PROMPT_PICK = '''\
You are writing one question in a multi-hop QA chain.

The PREVIOUS ANSWER is a private enterprise value. Write a question about a PUBLIC web document that requires the PREVIOUS ANSWER in a web search query to find the answer.

PREVIOUS ANSWER (must appear in your question): "{prev_answer}"

Search the document below for sentences that connect the PREVIOUS ANSWER to one of the ENTITIES. Your chosen entity must appear in the document and have a clear factual link to the PREVIOUS ANSWER. Pick the answer from the ENTITIES list below — it must be one of those exact strings.

Rules for the question:
- The PREVIOUS ANSWER must drive the question — it must not be ancillary. Using [PREV] to identify the subject is fine.
  BAD: "Who stars in The Wolf of Wall Street and was born in [PREV]?" ([PREV] in a side clause — answer is Leo regardless)
  GOOD: "In the company that created the iPhone that came out in year [PREV], what was their revenue?" ([PREV] identifies the subject)
- The answer must be one of the entities below (match exactly), the entity alone (never "the X"). Maximum 3 words.
- Pick the entity with the strongest factual link to the PREVIOUS ANSWER in the document. The entity must appear in the document text.
- CRITICAL: The answer must NOT be determinable without knowing the PREVIOUS ANSWER.
  The PREVIOUS ANSWER must discriminate between multiple possible answers — changing its value should change the answer.
  BAD: "Who stars in The Wolf of Wall Street and was born in [PREV]?" (answer is Leo regardless)
  BAD: "What firm reported [PREV] of companies published ESG reports?" (only one firm does this study)
  BAD: If someone could search for other distinctive phrases in the question (e.g. "The Loyal Subjects Jem and the Holograms") and find the answer without the PREVIOUS ANSWER, the question is bad.
  GOOD: "What regulation does [PREV] comply with?" (answer depends on [PREV])
- The question must be SPECIFIC enough to have exactly one correct answer.
- The PREVIOUS ANSWER should be the key identifying detail that pins down the answer.
- The question must be SELF-CONTAINED and make sense out of context.
- Write concrete, searchable questions that require the agent to search — avoid document-centric phrasing that obscures what to look up.
- Use concrete wording. Avoid vague terms. Without knowing the PREVIOUS ANSWER, the question should be unanswerable.

ENTITIES:
{entity_list}

Find evidence in the document that connects "{prev_answer}" to your chosen entity. Quote the relevant excerpt(s) — one or more sentences, possibly disjoint. Use " | " to separate multiple excerpts if needed. Then write the question based on that evidence.

Format: QUOTE may span multiple sentences (use " | " for disjoint excerpts). QUESTION must be a single line — one complete question, no line breaks.

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
QUOTE: <excerpt(s) from document — one or more sentences, use " | " to separate disjoint excerpts>
REASONING: ...
QUESTION: <single line, one complete question, must contain "{prev_answer}">
ANSWER: ...
</answer>'''

PROMPT_RANK = '''\
You are evaluating questions for a multi-hop QA chain.
The previous answer was "{prev_answer}". Rank by these criteria (most important first):

1. SPECIFICITY: The question should have exactly one correct answer. Prefer questions that include identifying details (names, dates, organizations) so they can't be confused with a different context that also mentions "{prev_answer}".
   WORSE: "How many visits does $1,200 cover?" (ambiguous — many things cost $1,200)
   BETTER: "How many visits to Karst Stages park does $1,200 cover?" (unique answer)
2. CENTRALITY: "{prev_answer}" should drive the question, not be ancillary.
3. NATURALNESS: The question should sound like something a person would actually ask.
4. SEARCHABILITY: The question should give enough information (when combined with the previous answer) to find the relevant document. Prefer questions where prev_answer + question together suffice to search effectively; avoid document-centric phrasing ("the document", "this report") that doesn't add searchable detail.

Rank these candidates from best to worst. Output ONLY the ranking as a comma-separated list of numbers (e.g. "2, 1, 3").

CANDIDATES:
{candidates}

RANKING (best to worst):'''

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
question — references like (1) mean "the answer to question 1".

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
