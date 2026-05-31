# Mem0: Building Production‑Ready AI Agents with Scalable Long‑Term Memory

[![arXiv](https://img.shields.io/badge/arXiv-Paper-b31b1b.svg)](https://arxiv.org/abs/2504.19413)
[![Website](https://img.shields.io/badge/Website-Project-blue)](https://mem0.ai/research)

This repository contains the code and dataset for our paper: **Mem0: Building Production‑Ready AI Agents with Scalable Long‑Term Memory**.

## 📋 Overview

This project evaluates Mem0 and compares it with different memory and retrieval techniques for AI systems:

1. **Established LOCOMO Benchmarks**: We evaluate against five established approaches from the literature: LoCoMo, ReadAgent, MemoryBank, MemGPT, and A-Mem.
2. **Open-Source Memory Solutions**: We test promising open-source memory architectures including LangMem, which provides flexible memory management capabilities.
3. **RAG Systems**: We implement Retrieval-Augmented Generation with various configurations, testing different chunk sizes and retrieval counts to optimize performance.
4. **Full-Context Processing**: We examine the effectiveness of passing the entire conversation history within the context window of the LLM as a baseline approach.
5. **Proprietary Memory Systems**: We evaluate OpenAI's built-in memory feature available in their ChatGPT interface to compare against commercial solutions.
6. **Third-Party Memory Providers**: We incorporate Zep, a specialized memory management platform designed for AI agents, to assess the performance of dedicated memory infrastructure.
7. **TMem**: We support the local TMem implementation from the repository and evaluate it with the same results JSON format consumed by `evals.py`.

We test these techniques on the LOCOMO dataset, which contains conversational data with various question types to evaluate memory recall and understanding.

## 🔍 Dataset

The LOCOMO dataset used in our experiments can be downloaded from our Google Drive repository:

[Download LOCOMO Dataset](https://drive.google.com/drive/folders/1L-cTjTm0ohMsitsHg4dijSPJtqNflwX-?usp=drive_link)

The dataset contains conversational data specifically designed to test memory recall and understanding across various question types and complexity levels.

Place the dataset files in the `dataset/` directory:
- `locomo10.json`: Original dataset
- `locomo10_rag.json`: Dataset formatted for RAG experiments

## 📁 Project Structure

```
.
├── src/                  # Source code for different memory techniques
│   ├── mem0/             # Implementation of the Mem0 technique
│   ├── openai/           # Implementation of the OpenAI memory
│   ├── tmem.py           # Implementation of the local TMem technique
│   ├── zep/              # Implementation of the Zep memory
│   ├── rag.py            # Implementation of the RAG technique
│   └── langmem.py        # Implementation of the Language-based memory
├── metrics/              # Code for evaluation metrics
├── results/              # Results of experiments
├── dataset/              # Dataset files
├── evals.py              # Evaluation script
├── run_experiments.py    # Script to run experiments
├── generate_scores.py    # Script to generate scores from results
└── prompts.py            # Prompts used for the models
```

## 🚀 Getting Started

### Prerequisites

Create a `.env` file with your API keys and configurations. The following keys are required:

```
# OpenAI API key for GPT models and embeddings
OPENAI_API_KEY="your-openai-api-key"

# Mem0 API keys (for Mem0 and Mem0+ techniques)
MEM0_API_KEY="your-mem0-api-key"
MEM0_PROJECT_ID="your-mem0-project-id"
MEM0_ORGANIZATION_ID="your-mem0-organization-id"

# Model configuration
MODEL="gpt-4o-mini"  # or your preferred model
EMBEDDING_MODEL="text-embedding-3-small"  # or your preferred embedding model
ZEP_API_KEY="api-key-from-zep"
```

### Running Experiments

You can run experiments using the provided Makefile commands:

#### Memory Techniques

```bash
# Run Mem0 experiments
make run-mem0-add         # Add memories using Mem0
make run-mem0-search      # Search memories using Mem0

# Run Mem0+ experiments (with graph-based search)
make run-mem0-plus-add    # Add memories using Mem0+
make run-mem0-plus-search # Search memories using Mem0+

# Run RAG experiments
make run-rag              # Run RAG with chunk size 500
make run-full-context     # Run RAG with full context

# Run LangMem experiments
make run-langmem          # Run LangMem

# Run TMem experiments
make run-tmem             # Run local TMem on LOCOMO

# Run Zep experiments
make run-zep-add          # Add memories using Zep
make run-zep-search       # Search memories using Zep

# Run OpenAI experiments
make run-openai           # Run OpenAI experiments
```

Alternatively, you can run experiments directly with custom parameters:

```bash
python run_experiments.py --technique_type [mem0|rag|langmem|tmem] [additional parameters]
```

#### Command-line Parameters:

| Parameter | Description | Default |
|-----------|-------------|---------|
| `--technique_type` | Memory technique to use (mem0, rag, langmem, tmem) | mem0 |
| `--method` | Method to use (add, search) | add |
| `--chunk_size` | Chunk size for processing | 1000 |
| `--top_k` | Number of top memories to retrieve | 30 |
| `--filter_memories` | Whether to filter memories | False |
| `--is_graph` | Whether to use graph-based search | False |
| `--num_chunks` | Number of chunks to process for RAG | 1 |

### TMem Notes

TMem is integrated through [src/tmem.py](src/tmem.py) and follows the new evaluation workflow directly:

1. Load one LOCOMO conversation.
2. Build two in-process TMem instances, one per speaker perspective.
3. Answer all QA items for that conversation.
4. Save a results JSON under `results/`.

Run it with:

```bash
make run-tmem
```

or:

```bash
python run_experiments.py --technique_type tmem --output_folder results/ --top_k 30
```

If you want the **full TMem architecture** inside the same evaluation workflow, including Neo4j and Qdrant, run:

Before running on a server, copy [evaluation/.env.example](.env.example) to `.env` and fill in the model endpoints you want to use for:

- evaluation answer generation and judging (`OPENAI_BASE_URL`, `OPENAI_API_KEY`, `MODEL`, `JUDGE_MODEL`)
- TMem internal extraction, retrieval, and embedding (`LLM_*`, `EMBEDDING_*`, `OLLAMA_EMBEDDING_*`)

```bash
make run-tmem-full
```

or:

```bash
python run_experiments.py --technique_type tmem --tmem_full --output_folder results/ --top_k 30
```

When `--tmem_full` is enabled, the evaluation runner will automatically start Neo4j and Qdrant using the repository-level [docker-compose.yml](../docker-compose.yml) before the test begins.

If you need to skip automatic deployment and use already-running services instead, run:

```bash
python run_experiments.py --technique_type tmem --tmem_full --skip_tmem_deploy --output_folder results/ --top_k 30
```

Then evaluate the generated file with:

```bash
python evals.py --input_file results/tmem_results_top_30.json --output_file results/tmem_eval_metrics.json
```

For the full-architecture run, use:

```bash
python evals.py --input_file results/tmem_full_results_top_30.json --output_file results/tmem_full_eval_metrics.json
```

Important:

- This evaluation integration always builds one TMem instance per speaker perspective. The default `make run-tmem` path keeps those instances in-process only, while `make run-tmem-full` enables Neo4j and Qdrant.
- The default `make run-tmem` path keeps this simplified in-process mode for quick evaluation parity.
- The `make run-tmem-full` path uses namespaced Neo4j/Qdrant storage so each speaker perspective remains isolated while preserving the same output JSON format expected by `evals.py`.
- The root [docker-compose.yml](../docker-compose.yml) maps ports to TMem's default config values: Neo4j Bolt `17687`, Neo4j Browser `7474`, and Qdrant `16333`.
- Docker auto-deployment only starts Neo4j and Qdrant. Your LLM and embedding endpoints still need to be reachable through [evaluation/.env.example](.env.example) -> `.env`.
- TMem's own extraction and retrieval LLM backends still read the repository-level environment variables such as `LLM_MODEL`, `LLM_BASE_URL`, `LLM_API_KEY`, `EMBEDDING_BACKEND`, `OLLAMA_EMBEDDING_MODEL`, and `OLLAMA_EMBEDDING_URL`.
- Final answer generation in the evaluation framework still uses `MODEL`, `OPENAI_API_KEY`, and optionally `OPENAI_BASE_URL`.
- LLM judging in `evals.py` now uses `JUDGE_MODEL` when provided, otherwise it falls back to `MODEL`.

### 📊 Evaluation

To evaluate results, run:

```bash
python evals.py --input_file [path_to_results] --output_file [output_path]
```

This script:
1. Processes each question-answer pair
2. Calculates BLEU and F1 scores automatically
3. Uses an LLM judge to evaluate answer correctness
4. Saves the combined results to the output file

### 📈 Generating Scores

Generate final scores with:

```bash
python generate_scores.py
```

This script:
1. Loads the evaluation metrics data
2. Calculates mean scores for each category (BLEU, F1, LLM)
3. Reports the number of questions per category
4. Calculates overall mean scores across all categories

Example output:
```
Mean Scores Per Category:
         bleu_score  f1_score  llm_score  count
category                                       
1           0.xxxx    0.xxxx     0.xxxx     xx
2           0.xxxx    0.xxxx     0.xxxx     xx
3           0.xxxx    0.xxxx     0.xxxx     xx

Overall Mean Scores:
bleu_score    0.xxxx
f1_score      0.xxxx
llm_score     0.xxxx
```

## 📏 Evaluation Metrics

We use several metrics to evaluate the performance of different memory techniques:

1. **BLEU Score**: Measures the similarity between the model's response and the ground truth
2. **F1 Score**: Measures the harmonic mean of precision and recall
3. **LLM Score**: A binary score (0 or 1) determined by an LLM judge evaluating the correctness of responses
4. **Token Consumption**: Number of tokens required to generate final answer.
5. **Latency**: Time required during search and to generate response.

## 📚 Citation

If you use this code or dataset in your research, please cite our paper:

```bibtex
@article{mem0,
  title={Mem0: Building Production-Ready AI Agents with Scalable Long-Term Memory},
  author={Chhikara, Prateek and Khant, Dev and Aryan, Saket and Singh, Taranjeet and Yadav, Deshraj},
  journal={arXiv preprint arXiv:2504.19413},
  year={2025}
}
```

## 📄 License

[MIT License](LICENSE)

## 👥 Contributors

- [Prateek Chhikara](https://github.com/prateekchhikara)
- [Dev Khant](https://github.com/Dev-Khant)
- [Saket Aryan](https://github.com/whysosaket)
- [Taranjeet Singh](https://github.com/taranjeet)
- [Deshraj Yadav](https://github.com/deshraj)

