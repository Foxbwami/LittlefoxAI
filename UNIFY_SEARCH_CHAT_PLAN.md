# Unify Search + Chat — Upgrade Plan for LittlefoxAI

Date: 2026-03-22
Author: LittlefoxAI Engineering (draft)

## Goal
Create a single unified interface (Search + Chat) that behaves like Google‑AI: single prompt, immediate synthesized AI answer with inline citations, and the ability to follow up conversationally while preserving full search-style result cards and preview/open actions.

## Executive Summary / Audit (current state)
- Crawling: Present (`backend/crawler/worker.py`, `backend/crawler/scheduler.py`).
- Indexing: Partial (TF‑IDF and embeddings exist; `data/search_index` empty — index build not populated).
- Vectorization: Implemented (`backend/retrieval/embeddings.py` using SentenceTransformer). Embeddings fallback enabled by config; `EMBEDDINGS_DISABLED=True` by default.
- Vector Store: FAISS implemented (`backend/memory/vector_store.py`); serialized index path `data/memory/faiss.index` expected.
- Hybrid retrieval & rerank: Present (`backend/retrieval/search.py`, `backend/server/search_service.py`) using TF‑IDF + embeddings + BM25/reranker.
- Web search connectors: Present (`backend/tools/browser.py`) with Bing/Google/Wikipedia/DuckDuckGo fallbacks; needs API keys for best results.
- RAG & answer generation: Present — `/search` and `/chat` endpoints combine local + web results and generate answers (`backend/core/responder.py`, `backend/server/app.py`).
- UI: Separate `frontend/index.html` (chat) and `frontend/search.html` (search). Frontend has hero, chips, result cards, preview modal. Unified single-page UX not yet implemented.
- Missing / partial: Meilisearch (not present), dedicated neural reranker model (not present), full chunking pipeline / index builder (`build_index.py` missing), connectors for private drives, multimodal search (images/video) disabled in config.

## High-Level Requirements
1. Single prompt UI that supports both ad-hoc chat and search queries.
2. Return an AI‑synthesized answer at the top with clear inline citations (sources list below the answer).
3. Display ranked result cards (hybrid vector + lexical ranking) with action buttons: Open, Preview, Summarize, Explain, Cite, Save.
4. Maintain conversational context: follow-ups refer to prior query & chosen sources.
5. Provide quick mode toggles (Chat / Research / Code / Creative) and advanced options for academic citations.
6. Fast response path with caching for high-frequency queries.
7. Robust backend: index builder, embedding pipeline, reranker hook, and simple connectors.

## Architecture Overview (proposed)
- Data layer: crawler -> preprocessor -> chunker -> embed -> FAISS (vector store) + TF‑IDF + optional Meilisearch.
- Retrieval: hybrid_search(query) -> candidate set (vector+tfidf+bm25) -> reranker (bm25+emb+pagerank) -> top N.
- RAG: combine top N chunks -> pass to LLM with citation mapping -> synthesize answer + inline citations.
- Frontend: single page `search_chat.html` (merge of `index.html` + `search.html`) -> one query input -> results + assistant area.
- Interaction: user issues query -> backend returns { answer, results[], citations[], used_sources[] } -> UI renders answer + cards + insights panel.

## Implementation Plan — Phases & Tasks
Phase 0 — Prep (quick, 1–2 days)
- Enable embeddings in config (set `EMBEDDINGS_DISABLED=False`) and verify `EMBEDDING_MODEL` availability or install sentence-transformers in `requirements.txt`.
- Create a lightweight `build_index.py` that loads raw data / crawler output, chunks, computes embeddings, builds/serializes TF‑IDF and FAISS indices, and writes to `data/search_index/` and `data/memory/`.

Phase 1 — Backend (3–7 days)
- Task B1: Implement or wire `build_index.py` chunking strategy (semantic chunking with overlap), store `pages.pkl`, `tfidf.pkl`, `matrix.pkl`, `embeddings.pkl`, `pagerank.pkl` in `data/search_index/` (needed by `SearchIndex.load()`).
- Task B2: Add optional Meilisearch connector (adapter) as alternate lexical index (config toggle). Provide minimal integration so Meilisearch can be used instead of TF‑IDF.
- Task B3: Add a simple neural reranker hook (optional): create `backend/retrieval/reranker.py` that exposes `rerank(query, candidates)` and initially implements BM25+emb scoring (placeholder for future model).
- Task B4: Improve `/search` payload to accept `action` and `url` — already partially handled by `askAIOnResult`; ensure `generate_search_answer` returns structured citations array with stable ids.
- Task B5: Add caching layer (simple Redis or in‑memory LRU) for frequent queries and top‑N candidate sets.

Phase 2 — Frontend (2–5 days)
- Task F1: Create `frontend/search_chat.html` merging hero + chat input (single input bar at top). Reuse existing JS/CSS components.
- Task F2: Make UI render both synthesized answer card (with inline citations and copy/save actions) and below it the result cards (Open/Preview/Summarize/Explain). Keep preview iframe with proper fallback.
- Task F3: Add conversation area to the right or below showing follow-ups; wire conversational context: each follow-up sends `conversation_id` and `parent_query` to backend.
- Task F4: Keyboard UX: Enter to send, Shift+Enter new line, Ctrl+K to focus search, Ctrl+N new chat.

Phase 3 — RAG & Quality (2–5 days)
- Task R1: Improve chunk ranking: implement a second-pass reranker (uses `reranker.py`) to reorder top 50->top 10 candidates.
- Task R2: Ensure citations are mapped to unique source ids and inline bracket references are stable across follow-ups.
- Task R3: Add source transparency UI: show snippet, domain, and link; allow "show original" which opens preview.

Phase 4 — Optional advanced (2–6 days)
- Meilisearch full integration and switchable lexical backend.
- Add connectors for Google Drive/SharePoint (OAuth + simple file extraction pipeline).
- Add multimodal: image upload -> CLIP embeddings pipeline + indexing.
- Add streaming LLM responses (if using model that supports streaming).

## Minimal Changes to Get a Working MVP (priority)
1. Add `build_index.py` and run to populate `data/search_index` and `data/memory/faiss.index` (so local search returns non-empty results).
2. Set `EMBEDDINGS_DISABLED=False` and ensure dependencies for sentence-transformers are installed.
3. Create merged frontend page `frontend/search_chat.html` that calls `/search` for both chat-like and search-like queries. Reuse `runSearch()` and `askAIOnResult()`.
4. Add a small caching decorator around `search_index.search()` in `backend/server/search_service.py`.
5. Verify citations and result rendering work end‑to‑end.

## Files to Create / Edit (concrete)
- Create: `build_index.py` (root or `backend/`) — index builder + chunker.
- Edit: `backend/core/config.py` — set `EMBEDDINGS_DISABLED=False` (or expose environment toggle) for testing.
- Edit: `backend/server/app.py` — ensure `/search` returns well-formed `citations`, `used_sources` and supports `conversation_id`.
- Create: `backend/retrieval/reranker.py` (hook for scoring).
- Create: `frontend/search_chat.html` (merge UI) and update `frontend/static/app.js` to initialize unified page.
- Update: `frontend/static/style.css` — ensure layout supports merged page (two-column assistant + results).

## Dev / Run Steps (quick commands)
1. Install or update Python deps (in venv):

```bash
python -m pip install -r backend/requirements.txt
python -m pip install sentence-transformers faiss-cpu scikit-learn rank_bm25
```

2. Build index (after creating `build_index.py`):

```bash
python backend/build_index.py --input data/raw --out data/search_index --faiss data/memory/faiss.index
```

3. Run the server locally:

```bash
cd backend
python server/app.py
# then open http://localhost:5000/explore or new merged page
```

## Acceptance Criteria (MVP)
- Single page `search_chat.html` accepts queries and returns a synthesized answer + 3–5 result cards.
- Inline citations are shown for the synthesized answer and link to source cards.
- Preview modal opens pages when allowed, otherwise "Open in new tab" works.
- Follow-up question retains context and updates answer accordingly.
- Local index returns results (non-empty) after running index builder.

## Risks & Considerations
- Embedding model size and FAISS memory: indexing large corpora requires significant RAM/CPU/GPU.
- Web crawling must respect robots.txt and be rate-limited; legal/privacy considerations for private connectors.
- External sites may block iframe previews (X-Frame-Options) — fallback must open in new tab.
- Meilisearch adds infra complexity; keep it optional.

## Next Steps (suggested immediate tasks)
1. I can draft `build_index.py` (chunking, embedding, TF‑IDF, FAISS) next — confirm and I will implement.
2. Or I can create `frontend/search_chat.html` and wire the unified UI to existing `/search` endpoint so you can test UX while index is being built.

---

If you want this doc adjusted (shorter/longer, more technical, a one‑page roadmap, or split into milestones with sprint estimates), tell me which style and I will revise immediately.

## Citation Policy & UI Rules (concise)

Purpose: ensure provenance is shown only when appropriate, avoid noise for model-only answers (chat/code/creative), and give users clear controls to surface sources on demand.

Detection (backend):
- Mark `has_provenance = true` when retrieval returns >=1 reliable source with relevance >= `MIN_SOURCE_RELEVANCE`.
- Include `source_confidence` (0..1) and `used_sources[]` (objects with `id,title,url,snippet,score`).
- Set `explanation_type` to one of `synthesized|model-only|mixed`.

Response schema (suggested additions to `/search` and `/chat` responses):
- `has_provenance: boolean`
- `source_confidence: number`
- `used_sources: [{id,title,url,snippet,score}]`
- `provenance_map: { citation_id: chunk_id }` (optional)
- `explanation_type: "synthesized"|"model-only"|"mixed"`

Frontend rules:
- If `has_provenance` true: show a green "Sourced" badge and a collapsed "Sources" panel with inline citations (e.g., [1], [2]) in the answer; allow "Show citations" toggle.
- If `has_provenance` false: show a grey "Model" badge, hide the sources panel, and surface actions: "Find sources" (runs retrieval) and "Explain basis" (asks model to describe its reasoning).
- For code answers: default to model-only UI but provide "Search docs" action to annotate code with sources when available.
- Allow user preference in profile (`always_show_sources` / `hide_sources_by_default`).

UX details and fallbacks:
- Keep sources collapsed by default to reduce visual noise; expand when user clicks a source or the toggle.
- For mixed responses, list top-k sources and attach claim-level provenance where feasible.
- If preview is blocked by X-Frame-Options, show domain + "Open in new tab" fallback.
- When `source_confidence` is low, label sources as "Possible sources — verify".

Minimal implementation steps:
1. Backend: add the schema fields in `backend/server/app.py` responses and ensure `search_index.search()` returns scores used to compute `source_confidence`.
2. Frontend: update answer renderer in `frontend/static/app.js` to respect `has_provenance` and render badge, toggle, and actions.
3. Profile: add display preference in `frontend/profile.html` and save to `littlefoxProfile`.
4. Tests: verify flows for research query (sourced), code snippet (model-only with "Search docs"), and creative prompt (model-only).

Acceptance: UI shows correct badge and sources only for grounded answers; model-only answers offer on-demand source lookup without empty citation blocks.

---

If you want, I can now implement the backend schema changes or update the frontend renderer — tell me which to start with.