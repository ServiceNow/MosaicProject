# Current System

The current generator is a five-stage pipeline. Each stage writes an output that the next stage reads, so you can resume from the middle if needed.

1. `load`: read local chunks, web chunks, curated web metadata, and secrets into a saved manifest
2. `build`: generate chains with retrieval, candidate generation, filtering checks, reranking, and a first verification pass
3. `verify`: rerun answerability checks under several document-drop settings
4. `filter`: score chains with a judge model
5. `reword`: make small wording fixes without changing meaning

## Inputs

By default the package expects these files under `trajectory_generation/data/`:

- `chunks_local.jsonl`
- `chunks_web.jsonl`
- `secret_inventory.jsonl`
- optional `chunks_curated_web.jsonl`
- optional `docs_curated_web.jsonl`
- optional `tasks/` directory for curated URL refresh tools

## Stage Walkthrough

Below is a toy example of how one chain might move through the pipeline.

### 1. `load`

What it does:
- reads the local and web chunk files
- groups local chunks back into full documents
- loads the secret inventory
- writes a manifest such as `state_0_load.json`

Toy example:
- local chunk says a factory audit finished in `Q2 2024`
- web chunk says `Northwind Robotics` expanded a plant in `Q2 2024`
- the saved manifest records where those files came from and which tasks are available

### 2. `build`

What it does:
- picks a starting secret or seed question
- searches for a bridge into the next document
- drafts candidate next questions
- runs simple checks
- ranks the surviving candidates

Toy example:
- hop 1 question: `When did the factory audit finish?`
- hop 1 answer: `Q2 2024`
- hop 2 question: `Which company expanded a plant in Q2 2024?`
- hop 2 answer: `Northwind Robotics`

### 3. `verify`

What it does:
- tests whether the chain still works when different parts of the context are removed
- helps catch chains that only look good because too much context was handed to the verifier

Toy example:
- with both documents present, the final answer is `Northwind Robotics`
- with only the local document present, hop 2 should fail
- with only the web document present, hop 1 should fail

### 4. `filter`

What it does:
- asks a judge model whether the chain is specific, grounded, and worth keeping
- removes chains that are technically valid but weak, vague, or low-value

Toy example:
- keep the chain if `Q2 2024` is actually needed to find the `Northwind Robotics` fact
- drop it if the second question is too obvious even without the first answer

### 5. `reword`

What it does:
- makes small wording fixes after filtering
- keeps the answer and structure the same

Toy example:
- change `According to the document, which company...` to `Which company...`
- keep the answer `Northwind Robotics` unchanged

## Retrieval

The builder combines:

- local retrieval over chunk JSONL
- web retrieval over the fixed BrowseComp collection
- an optional second web source limited to documents linked to the current task
- if both web sources return good hits, it blends the two ranked lists instead of trusting only one source

## Verification Harness

Lightweight local smoke checks live outside the repo at `/home/toolkit/my_code/mosaicproject_smoke/`. That keeps the advisor-facing repo itself focused on the package and docs.
