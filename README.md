# RAG From Scratch: Learning Notebooks

Jupyter notebooks that build up Retrieval Augmented Generation (RAG) step by step with LangChain: indexing, retrieval and generation first, then query transformations, routing, query construction, advanced indexing and re-ranking.

LLMs are trained on a large but fixed corpus, so they cannot reason about private or recent information on their own. RAG grounds the model's answer in documents retrieved from an external source at query time. These notebooks show each piece of that pipeline in small, runnable cells.

![RAG overview](https://github.com/langchain-ai/rag-from-scratch/assets/122662504/54a2d76c-b07e-49e7-b4ce-fc45667360a1)

## Origin and credits

This repository is a fork of [langchain-ai/rag-from-scratch](https://github.com/langchain-ai/rag-from-scratch) by Lance Martin (LangChain), which accompanies the [RAG From Scratch video playlist](https://www.youtube.com/playlist?list=PLfaIDFEXuae2LXbO1_PKyVJiQ23ZztA0x). The notebook content is the upstream material. This fork is maintained by Shibil as personal learning material and adds a pinned dependency set, safer key handling, and an automated notebook check (see [Changes in this fork](#changes-in-this-fork)).

The upstream repository does not include a license file, so no license is granted here beyond what the original authors allow.

## Notebooks

| Notebook | Parts | What it covers |
| --- | --- | --- |
| [`rag_from_scratch_1_to_4.ipynb`](rag_from_scratch_1_to_4.ipynb) | 1–4 | End-to-end RAG quickstart; indexing (token counting with `tiktoken`, OpenAI embeddings, cosine similarity, loading and splitting a blog post, Chroma vector store); retrieval; generation with a prompt + LLM chain |
| [`rag_from_scratch_5_to_9.ipynb`](rag_from_scratch_5_to_9.ipynb) | 5–9 | Query transformations: Multi-Query, RAG-Fusion (reciprocal rank fusion), Decomposition (recursive and individual answering), Step-Back prompting, HyDE |
| [`rag_from_scratch_10_and_11.ipynb`](rag_from_scratch_10_and_11.ipynb) | 10–11 | Logical routing with structured output, semantic routing with embeddings, query construction for metadata filters (YouTube video search schema) |
| [`rag_from_scratch_12_to_14.ipynb`](rag_from_scratch_12_to_14.ipynb) | 12–14 | Multi-representation indexing (summaries + `MultiVectorRetriever`), RAPTOR (links only), ColBERT via RAGatouille |
| [`rag_from_scratch_15_to_18.ipynb`](rag_from_scratch_15_to_18.ipynb) | 15–18 | Re-ranking with RAG-Fusion and Cohere Rerank; CRAG, Self-RAG and long-context discussion (links to videos and LangGraph notebooks, no code) |

## Pipeline

```
Documents (web pages, Wikipedia, YouTube metadata)
  → load (WebBaseLoader / YoutubeLoader / requests)
  → split (RecursiveCharacterTextSplitter)
  → embed (OpenAIEmbeddings) → store (Chroma, in memory)
Question
  → optional query transformation / routing / structuring (ChatOpenAI)
  → retrieve (vector store retriever, optional RRF or Cohere re-rank)
  → prompt with retrieved context → ChatOpenAI → answer
```

## Tech stack

- Python 3.10+ and Jupyter
- LangChain 0.3 (`langchain`, `langchain-core`, `langchain-community`, `langchain-openai`, `langchain-text-splitters`, `langchain-cohere`)
- OpenAI chat and embedding models, Chroma vector store
- Optional: Cohere Rerank (Part 15), RAGatouille / ColBERT (Part 14), LangSmith tracing

## Setup

```bash
git clone https://github.com/ShibilAhamed701212/retrieval-augmented-generation-learning.git
cd retrieval-augmented-generation-learning
python -m venv .venv
source .venv/bin/activate        # Windows: .venv\Scripts\activate
pip install -r requirements.txt
cp .env.example .env             # then fill in your keys
jupyter notebook
```

Each notebook's first cell runs `%pip install -q -r requirements.txt`, so it also works when started from a fresh kernel in the repository folder.

### Configuration

Keys are read from `.env` (git-ignored) by `python-dotenv`. If `OPENAI_API_KEY` (or `COHERE_API_KEY` in the Part 15 notebook) is not set, the notebook prompts for it with `getpass`, so it is never written into the notebook.

| Variable | Needed for | Required |
| --- | --- | --- |
| `OPENAI_API_KEY` | All notebooks (chat and embedding calls) | Yes |
| `COHERE_API_KEY` | Part 15, Cohere Rerank | Only for that section |
| `LANGCHAIN_API_KEY` | LangSmith tracing; tracing is switched on only when this is set | No |
| `USER_AGENT` | Identifies `WebBaseLoader` requests and silences its warning | No |

Running the notebooks makes paid OpenAI (and Cohere) API calls.

### Part 14 (ColBERT)

RAGatouille is not in `requirements.txt` because it pulls in PyTorch. The notebook installs it in its own cell (`pip install -U ragatouille`) and downloads the `colbert-ir/colbertv2.0` model on first use.

## Testing

There is no API-key-free way to execute the notebooks end to end, so CI runs static checks instead:

```bash
pip install -r requirements.txt ruff
python scripts/check_notebooks.py --imports   # every code cell parses; every import resolves
ruff check .                                   # notebooks: syntax errors and undefined names
ruff check --select E,F,W,I,B,UP scripts
ruff format --check .
```

The GitHub Actions workflow in [`.github/workflows/ci.yml`](.github/workflows/ci.yml) runs the same commands on every push to `main` and every pull request.

## Changes in this fork

- **Notebooks failed on their setup cells.** The key cells contained `os.environ['OPENAI_API_KEY'] = <your-api-key>`, which is a `SyntaxError`. They now load keys from `.env` and fall back to a `getpass` prompt; LangSmith tracing is only enabled when a LangSmith key exists.
- **Unpinned requirements installed LangChain 1.x**, which removed `langchain.load`, `langchain.retrievers`, `langchain.storage`, `langchain.text_splitter`, `langchain.prompts`, `langchain.utils` and `langchain_core.pydantic_v1`, so the notebooks failed at their imports. `requirements.txt` now pins the LangChain 0.3 line, adds the missing `langchain-cohere` and `pydantic`, and the per-notebook `pip install` cells use it.
- **`hub.pull("rlm/rag-prompt")` no longer works without a LangSmith account** (current `langsmith` refuses anonymous public prompt pulls and requires opting in to deserialize remote objects). The same prompt text is now defined inline.
- **`langchain_core.pydantic_v1` is deprecated**; the routing and query-construction schemas now use Pydantic v2 directly.
- Added `.env.example`, `scripts/check_notebooks.py`, `ruff.toml` and the CI workflow.

## Known limitations

- The notebooks have not been executed end to end in this fork (no API keys in CI). The saved cell outputs are from the upstream author's original runs.
- Several cells use `gpt-3.5-turbo` / `gpt-3.5-turbo-0125`. Whether those model names are still served depends on OpenAI's current model list; swap in a current model if the API rejects them.
- Part 11 loads YouTube metadata with `YoutubeLoader(add_video_info=True)`, which relies on `pytube`. `pytube` is unmaintained and returned `HTTP Error 400` when tried during this audit. The rest of Part 11 (LLM query structuring) does not depend on that cell. YouTube may also block transcript requests from cloud IPs.
- The notebooks use LangChain 0.3 APIs that are deprecated in later releases (`get_relevant_documents`, `langchain.load`, `langchain.retrievers`). Moving to LangChain 1.x would need `langchain-classic` or rewritten imports.
- RAPTOR, CRAG, Self-RAG and the long-context section link to external videos and notebooks; this repository has no code for them.
