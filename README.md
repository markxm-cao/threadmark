# Threadmark

Threadmark is an experimental repository-aware code retrieval and question-answering system for locating relevant source code from natural-language questions.

Instead of relying on a single retrieval method, Threadmark explores a combination of semantic retrieval, lexical search, source-code structure, and program behavior to identify code regions that are most likely to answer a developer's question.

> **Status:** Threadmark is under active development. The current implementation focuses on retrieval, structural analysis, and evaluation rather than a polished end-user interface.

## Features

- Repository cloning and source-file discovery with GitPython
- Fixed-window and AST-aware code chunking
- Semantic retrieval with sentence-transformer embeddings
- BM25 lexical retrieval
- Hybrid retrieval using reciprocal-rank fusion
- Cross-encoder reranking
- Python AST-based behavior extraction
- Control-flow and code-motif detection
- LLM-assisted query planning
- Grounded answer generation from retrieved repository context
- Retrieval evaluation against expected source-code regions

## How It Works

At a high level, Threadmark processes a repository in several stages:

1. **Repository ingestion**  
   Clone a Git repository and discover supported source files.

2. **Code chunking**  
   Split source code into retrievable regions. Threadmark currently supports both fixed-window chunking and AST-aware chunking for Python.

3. **Candidate retrieval**  
   Retrieve relevant code using:
   - semantic similarity
   - BM25 lexical search
   - hybrid reciprocal-rank fusion

4. **Optional reranking**  
   Rerank retrieved candidates with a cross-encoder model.

5. **Structural analysis**  
   Analyze Python ASTs to extract program behaviors, control-flow relationships, and selected implementation motifs.

6. **Query planning**  
   Convert natural-language questions into likely program behaviors, relationships, and concepts.

7. **Grounded generation**  
   Build a prompt from retrieved code regions so an LLM can answer using repository evidence rather than unsupported implementation guesses.

## Project Structure

```text
threadmark/
├── eval/
│   └── query_plans.json
├── src/
│   └── threadmark/
│       ├── behavior.py
│       ├── behavior_categories.py
│       ├── chunking.py
│       ├── control_flow.py
│       ├── evaluation.py
│       ├── freeze_query_plans.py
│       ├── generation.py
│       ├── query_planner.py
│       ├── repository.py
│       ├── retrieval.py
│       └── structural_retrieval.py
├── .gitignore
├── README.md
└── requirements.txt
```

## Setup

### 1. Clone the repository

```bash
git clone https://github.com/markxm-cao/threadmark.git
cd threadmark
```

### 2. Create a virtual environment

On macOS or Linux:

```bash
python3 -m venv .venv
source .venv/bin/activate
```

### 3. Install dependencies

```bash
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

### 4. Optional: configure OpenAI access

Some parts of Threadmark, including LLM-assisted query planning and grounded answer generation, use the OpenAI API.

Set an API key in your environment before running those components:

```bash
export OPENAI_API_KEY="your_api_key_here"
```

The basic retrieval evaluation does not require an OpenAI API call.

## Running the Retrieval Evaluation

Threadmark includes a small benchmark of natural-language repository questions paired with expected source-code regions.

Run it from the repository root with:

```bash
PYTHONPATH=src python -m threadmark.evaluation
```

The evaluation currently compares:

- semantic retrieval
- BM25 retrieval
- hybrid retrieval
- cross-encoder reranking

For each benchmark question, the script reports the rank of the expected source-code region.

Example output:

```text
What happens when the Riot API returns HTTP status code 429?
  Semantic: Rank 6
  BM25:     Rank 3
  Hybrid:   Rank 4
  Reranked: Rank 4
```

The goal of the benchmark is not to assume that one retrieval method is always best, but to make it possible to compare different retrieval strategies on the same questions.

## Retrieval Methods

### Semantic Retrieval

Threadmark embeds code chunks and natural-language queries using a sentence-transformer model, then ranks chunks by cosine similarity.

### BM25 Retrieval

Threadmark also supports lexical search using BM25. This is useful when identifiers, filenames, API names, or exact implementation terms are strong signals.

### Hybrid Retrieval

Semantic and BM25 rankings can be combined using reciprocal-rank fusion, allowing Threadmark to benefit from both semantic similarity and lexical matches.

### Cross-Encoder Reranking

A cross-encoder can rerank an initial candidate set by jointly scoring each query and code chunk.

## AST-Aware Chunking

For Python files, Threadmark can use the Python AST to chunk code around functions, asynchronous functions, and classes rather than relying only on fixed line windows.

Large symbols can still be divided into overlapping windows so retrieval remains practical.

For non-Python source files, Threadmark currently falls back to fixed-window chunking.

## Structural Retrieval

Threadmark also experiments with retrieval signals derived from program structure rather than raw text alone.

The current Python AST analysis can extract behavior facts such as:

- membership checks
- collection mutations
- control-flow exits
- function calls
- state assignments
- compound conditions
- periodic trigger patterns

These lower-level operations can be mapped into more abstract behavior categories and relationships.

Threadmark also contains experimental control-flow motif detection for patterns such as:

- skip-if-already-seen deduplication
- guarded insertion of newly discovered items

The purpose of this work is to test whether structural information can improve retrieval for questions that are difficult to answer using text similarity alone.

## Query Planning

Threadmark can use an LLM to convert a natural-language repository question into a structured query plan containing:

- an intent
- likely program behaviors
- likely relationships between behaviors
- related implementation concepts

These query plans can then be used by structural retrieval experiments.

Frozen query plans are stored in:

```text
eval/query_plans.json
```

This makes it possible to evaluate retrieval behavior without regenerating the same plans on every run.

## Grounded Answer Generation

After retrieving source-code regions, Threadmark can format them as repository evidence and provide them to an LLM.

The generation prompt instructs the model to:

- answer only from the supplied repository context
- avoid inventing unsupported implementation details
- cite the provided source ranges
- state when the retrieved evidence is insufficient

## Supported Source Files

Repository discovery currently recognizes:

```text
.py
.java
.c
.cpp
.h
.hpp
```

AST-aware structural analysis is currently Python-specific.

## Testing

Threadmark includes unit tests for:

- code-aware tokenization
- fixed-window and AST-aware chunking
- condition classification
- structural deduplication motif detection

Run the test suite with:

```bash
python -m pytest -v

## Current Limitations

Threadmark is still an experimental project. Current limitations include:

- structural analysis is focused on Python
- evaluation uses a small hand-labeled benchmark
- retrieval quality varies by query
- some modules still contain experimental analysis code
- there is not yet a polished command-line or graphical interface
- the project is not yet packaged for installation from PyPI

These limitations are part of the current development focus rather than hidden production assumptions.

## Current Development

Current work focuses on:

- comparing semantic, lexical, hybrid, and structural retrieval
- improving behavior and control-flow representations
- identifying which structural signals help specific query types
- expanding evaluation coverage
- separating reusable library code from experimental analysis scripts
- improving tests and project documentation

## Why Threadmark

Many code-search systems rely heavily on text or embedding similarity. Threadmark is an experiment in combining those signals with explicit information about program structure and behavior.

The central question is:

> Can structural information extracted from source code help retrieve the implementation a developer is actually asking about?

Threadmark is being built as a practical way to explore that question while developing a repository-aware code question-answering pipeline.
```
