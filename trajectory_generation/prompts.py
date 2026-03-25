"""Prompts for intra-doc (constrained), inter-doc (pick), ranking, reword, and verification stages."""

PROMPT_CONSTRAINED = '''\
You are writing one question in a multi-hop QA chain.

PREVIOUS ANSWER (must appear VERBATIM in your question): "{prev_answer}"
TARGET ANSWER (the correct answer to your question): "{target_answer}"

Rules for the question:
- The PREVIOUS ANSWER text "{prev_answer}" must appear word-for-word in your question. Do not paraphrase, abbreviate, or use a placeholder. For example if the previous answer is "Q2 2025", write "Q2 2025" in the question, not just "2025".
- The PREVIOUS ANSWER must be central to the question — changing it should change the answer.
  BAD (previous answer "1984"): "Who stars in The Wolf of Wall Street and Good Will Hunting and was born in 1984?" (answer is Leo regardless of the year)
  GOOD (previous answer "1984"): "What Orwell novel is set in the year 1984?" (answer depends on the year)
- The answer must be 1-3 words, the entity alone (never "the X" — use "Health Cloud" not "the Health Cloud").
- The question must be SPECIFIC enough to have exactly one correct answer, but not so specific that it can be answerable without the previous answer.
- The question must be SELF-CONTAINED and sound natural — like something a curious person would actually ask. It's often helpful to imagine you are a forgetful person who remembers tangentially related information (e.g. the previous answer, and some other information), and is trying to answer a new question based on that information.
- Write concrete, searchable questions — avoid document-centric phrasing ("this paper", "according to the document"). It is okay to say "In the paper that found X, what was Y?" where X is the previous answer and Y is the target answer, but only because X guides the search for Y. A question like "In the document, X found Y" is not a good question because 'the document' is too vague to search for.
- Without knowing the PREVIOUS ANSWER, the question should be unanswerable AND a research agent should have a hard time knowing what to even search for to answer it.

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

The PREVIOUS ANSWER is a private enterprise value. Write a question about the PUBLIC document below where the PREVIOUS ANSWER plays a meaningful role — as a quantity, threshold, date, identifier, or parameter that naturally appears in the question.

PREVIOUS ANSWER (must appear VERBATIM in your question): "{prev_answer}"

Search the document for a context where "{prev_answer}" plays a natural, functional role — for example as a sample size, a percentage threshold, a year something happened, a dollar amount, etc. Then write a question where knowing "{prev_answer}" is essential to identifying the answer.

The answer MUST be one of the ENTITIES below — pick exactly one of these strings:
{entity_list}

Rules:
- The text "{prev_answer}" must appear word-for-word in your question. Do not paraphrase, abbreviate, or use a placeholder.
- The PREVIOUS ANSWER must be central to the question — changing it should change the answer.
  BAD (previous answer "12%"): "What organization publishes an annual obesity report that mentions 12%?" (answer is the same org regardless of the percentage)
  GOOD (previous answer "12%"): "In the study that found a 12% decrease in obesity rates in Tamale, what year was the data collected?" (changing the percentage points to a different study)
  GOOD (previous answer "3"): "In the clinical trial with 3 participants, what drug was tested for efficacy?" (the number identifies a specific trial)
- Your answer must EXACTLY match one of the ENTITIES listed above. Do not combine entities, modify their spelling, or invent new answers.
- The question should sound natural — like something a researcher or analyst would genuinely ask. Find a context in the document where the previous answer fits organically.
- The question must be SPECIFIC enough to have exactly one correct answer.
- The question must be SELF-CONTAINED — avoid document-centric phrasing ("this report", "the document").
- Without knowing the PREVIOUS ANSWER, the question should be unanswerable.

Find evidence in the document that connects "{prev_answer}" to your chosen entity. Quote the relevant excerpt(s). Then write the question.

Example:
Previous answer: "80%"
Entities: FDA, HIPAA, 112
QUOTE: In the study that achieved 80% sensitivity for predicting bladder cancer recurrence, 112 patient specimens were analyzed.
REASONING: "80%" identifies a specific study whose sample size is 112. The percentage is central — a different sensitivity level would point to a different study.
QUESTION: How many patient specimens were analyzed in the study that achieved 80% sensitivity for predicting bladder cancer recurrence?
ANSWER: 112

<document>
{doc_text}
</document>

<answer>
QUOTE: <excerpt(s) from document>
REASONING: ...
QUESTION: <single line, one complete question, must contain "{prev_answer}">
ANSWER: <must be one of the ENTITIES above>
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
