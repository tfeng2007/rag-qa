# RAG Q&A over Chinese ML Textbook Notes

A retrieval-augmented generation (RAG) system that answers questions over a
Chinese machine-learning knowledge base (study notes on Zhou Zhihua's
*Machine Learning* / 西瓜书). Built from scratch without RAG frameworks to
understand every component, with a **two-layer evaluation harness** and a
documented, **eval-driven optimization** process.

> The headline isn't "I built a RAG" — plenty of people can. It's that the
> system is **measured**, its failures are **diagnosed**, every change is
> **justified by data**, and a silent bug that produced a *wrong* experimental
> conclusion was **caught and corrected**.

**Final results:** single-fact accuracy **100%**, multi-point coverage **100%**,
out-of-scope refusal (no hallucination) **100%**.

---

## What it does

Point it at a folder of Markdown documents; ask natural-language questions; get
answers grounded **only** in the retrieved source, with explicit refusal when
the answer isn't in the corpus (anti-hallucination).

```
"决策树为什么要剪枝？"       →  grounded answer from the notes
"决策树有哪几种划分度量？"   →  lists 信息增益/ID3, 增益率/C4.5, 基尼/CART (from source)
"解释 Transformer 架构"      →  "资料中未提及" (correctly refuses; not in corpus)
```

---

## Architecture

```
Offline (index once):
  Markdown docs ── header-aware chunking ── local embeddings ── Chroma vector store

Online (per query):
  question ── embed ── recall top-20 (Chroma) ── rerank (cross-encoder) ── top-5
           ── context assembly ── Claude ── answer
```

| Stage | Choice | Notes |
|---|---|---|
| Embeddings | `BAAI/bge-small-zh-v1.5` (local) | Free, no API key; Anthropic has no embedding model |
| Vector store | Chroma (persistent, cosine) | Local, survives restarts |
| Chunking | Markdown-header split (`#`/`##`), char fallback for long sections | See optimization story |
| **Retrieval** | **Two-stage: embedding recall (top-20) → cross-encoder rerank (`bge-reranker-base`) → top-5** | Rerank was the key fix (below) |
| Generation | Anthropic Claude | Strict "answer only from context" system prompt |

---

## Evaluation (the core of this project)

RAG failures happen in two different places that need different fixes, so the
system is evaluated in **two layers**:

**1. Retrieval layer** — were the relevant chunks retrieved? (keyword / key-point
coverage over retrieved chunks, Recall@k flavour)

**2. Answer layer** — is the final answer correct and faithful? **LLM-as-judge**:
Claude scores each answer against a reference, with type-specific rubrics.

Test set (`eval_set.json`, 15 questions, unified schema), three types:

| Type | Tests | Scoring |
|---|---|---|
| `single` | single-fact recall | 0 / 1 |
| `multi` | multi-point / cross-section synthesis | key-point coverage ratio |
| `out_of_scope` | **anti-hallucination** (answer not in corpus) | refuse = 1, fabricate = 0 |

The out-of-scope set includes a **hard negative**: a question about SVM "kernel
tricks", which the corpus *mentions by name* in a chapter preview but never
explains. The system correctly refuses instead of fabricating from the teaser.

### Final results (answer layer)

| Metric | Score |
|---|---|
| Single-fact accuracy | **100%** (5/5) |
| Multi-point coverage | **100%** (10 questions) |
| Out-of-scope refusal (no hallucination) | **100%** (3/3) |

---

## Optimization journey (eval-driven)

Every change was kept, reverted, or re-examined based on eval numbers, not
intuition.

| Step | Change | Multi-coverage | Outcome |
|---|---|---|---|
| Baseline | fixed-size character chunking | 90% | One question scored 0 — chunks cut mid-word / mid-formula (`"logis\|tic regression"`) |
| v1 | split on **all** Markdown headers | 92% | Fixed target case but **over-fragmented** two others |
| v2 | split on **top-level headers only** (`#`/`##`) | 95% | Balanced granularity; recovered regressions |
| v3 | raise top_k 3 → 5 | 95% | Recovered a case whose answer section was just outside the window |
| ✗ | filter out "summary/objective" meta-sections | reverted | **Backfired** — removed partial-credit content |
| ✗ | enlarge max chunk size | reverted | **Backfired** — diluted a definition, hurt a single-fact question (chunk-size/precision tradeoff) |
| ✓ | **two-stage retrieval with reranking** | **100%** | **The fix** for the last stubborn case (details below) |

---

## Key insights

**1. A keyword-based retrieval metric gave a false positive — the answer-layer
eval caught it.**
One multi-part question scored a perfect 1.0 on retrieval (all keywords present)
yet was only half-answered: the keywords were scattered across a keyword-dense
*summary* chunk, while the section actually containing the answer was never
retrieved. **Keyword-match retrieval metrics are lenient; an answer-level eval is
needed to catch this.**

**2. A silent duplicate-function bug produced a *false negative* on reranking —
and rigorous re-running caught it.**
My first reranking experiment appeared to *hurt* the system, and I nearly
discarded reranking. Investigating the eval-config revealed two `search()`
definitions in the file — Python silently uses the last one, so **reranking was
never actually running** during that experiment. After removing the duplicate and
re-running from a clean, single-config state, reranking's true effect appeared:
multi-coverage **95% → 100%**, fixing the one stubborn case (decision-tree
metrics → algorithms) that chunking, top_k, meta-filtering and chunk-size tuning
had all failed to solve. **A non-erroring code bug produced a wrong conclusion;
eval discipline — clean state, single variable, reproducible re-run — corrected
it.**

**3. Why reranking was the right fix.**
A bi-encoder (embedding) compresses query and chunk into separate vectors and
was consistently out-ranking the detailed answer section with a keyword-dense
summary. A cross-encoder reranker scores each (query, chunk) *pair* jointly, so
it recognised that the detailed section actually answers the question and
promoted it into the top-k. The answer section was always within the top-20
recall; reranking is what surfaced it.

**4. Faithful refusal over confident fabrication.**
On questions whose answer is absent or incomplete in the corpus, the system
flags the gap ("资料中未提及") rather than hallucinating — verified by the 100%
out-of-scope refusal rate.

---

## Known limitations & future work

- **Reranking adds latency** — one extra cross-encoder inference per query. The
  accuracy gain (95% → 100% multi-coverage, plus the hardest case fixed) was
  judged worth it here; at larger scale this is a tradeoff to monitor.
- **Larger, independently-labelled eval set** to reduce metric noise (15
  questions is small).
- **Deployment** as a small web UI with inline source citations.

---

## Tech stack

Python · Anthropic Claude (generation) · sentence-transformers / BGE embeddings ·
`bge-reranker-base` cross-encoder (reranking) · Chroma (vector store) · no RAG
framework (built from primitives)

---

## Project structure

```
rag-qa/
├── src/
│   ├── rag.py        # load_docs, chunk_by_header, build_index, search (recall+rerank), answer
│   └── eval.py       # retrieval + answer eval, LLM-as-judge, run comparison
├── data/             # Markdown knowledge base
├── eval_set.json     # 15 questions (single / multi / out_of_scope)
├── requirements.txt
├── .env.example      # ANTHROPIC_API_KEY=...
└── README.md
```

---

## Setup & run

```bash
python -m venv venv && source venv/bin/activate
pip install -r requirements.txt
cp .env.example .env            # add your ANTHROPIC_API_KEY

python src/rag.py                     # build the index (once)
python src/eval.py --mode retrieval   # retrieval-layer eval
python src/eval.py --mode answer      # answer-layer eval (LLM-as-judge)
```
