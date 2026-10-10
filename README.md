# MicroWorld: Empowering Multimodal Large Language Models to Bridge the Microscopic Domain Gap with Multimodal Attribute Graph

MicroWorld is a knowledge graph (KG) system designed to support visual question answering (VQA) in the biomedical microscopy domain. It retrieves relevant biological knowledge based on question text and/or image context, and injects it as a structured prompt prefix to improve the reasoning of multimodal large language models (MLLMs).

## Overview

The pipeline consists of seven stages (S0–S6) to build the KG, plus a retrieval module (`microworld.py`) for inference-time use.

```
Raw data (images + captions)
        │
        ▼
  S0: Data preparation & indexing
        │
        ▼
  S1: Entity extraction + LLM relation extraction
        │
        ▼
  S2: KG construction (nodes, edges, adjacency)
        │
        ▼
  S3: Entity description generation (NCBI + LLM fallback)
        │
        ▼
  S4: Visual embedding (Qwen3-VL-Embedding)
        │
        ▼
  S5: K-hop neighbor precomputation
        │
        ▼
  S6: Similarity ranking (Jaccard + Cosine)
        │
        ▼
  microworld.py: KG retrieval at inference time
```

## Dataset

Data is available at modelscope [MicroWorld](https://modelscope.cn/datasets/ieellee/MicroWorld)

## Requirements

```bash
# pip install openai scispacy torch
### Install scispaCy biomedical model:
# pip install https://s3-us-west-2.amazonaws.com/ai2-s3-scispacy/releases/v0.5.4/en_core_sci_lg-0.5.4.tar.gz
conda env create -f environment.yml
conda activate mw

```

For Stage 4 (visual embeddings), you also need the [Qwen3-VL-Embedding](https://huggingface.co/Qwen/Qwen3-VL-Embedding-8B) model.

## Building the Knowledge Graph

### Environment Variables

Before running S1 or S3, set your OpenAI-compatible API credentials:

```bash
export OPENAI_API_BASE="https://api.openai.com/v1"
export OPENAI_API_KEY="your-api-key-here"
export OPENAI_MODEL="gpt-5.4"
```

For S3 (NCBI queries), you can optionally set an NCBI API key to increase the rate limit from 3 to 10 requests/second:
```bash
# Get a free key at https://www.ncbi.nlm.nih.gov/account/
python stages/microWorld_s3.py --ncbi_api_key YOUR_NCBI_KEY
```

### Step-by-step

**S0: Data preparation**

```bash
# From MicroVQA CSV
python stages/microWorld_s0.py --input /path/to/microvqa.csv

# From OmniScience JSONL with bio-relevance filtering
python stages/microWorld_s0.py \
    --input omniscience_subset/omnisci_20k_lf.jsonl \
    --support_dir ./support \
    --filter_bio \
    --max_samples 20000
```

Output: `support/dataset_index.json`

**S1: Entity extraction + relation extraction**

```bash
python stages/microWorld_s1.py --support_dir ./support --resume --workers 8
```

Output: `support/raw_triplets.json`

**S2: KG construction**

```bash
python stages/microWorld_s2.py --support_dir ./support
```

Output: `support/KG/nodes.json`, `support/KG/edges.json`, `support/KG/graph.json`

**S3: Entity descriptions**

```bash
python stages/microWorld_s3.py --support_dir ./support --resume
```

Output: `support/entity_descriptions.json`

**S4: Visual embeddings** (requires GPU)

```bash
python stages/microWorld_s4.py --support_dir ./support --model 2B --batch 4
```

Output: `support/visual_embeddings/`

**S5: K-hop neighbor precomputation**

```bash
python stages/microWorld_s5.py --support_dir ./support
```

Output: `support/KG/results_close_entity.json`

**S6: Similarity ranking**

```bash
python stages/microWorld_s6.py --support_dir ./support
```

Output: `support/entity_similarity_sorted.json`, `support/image_similarity_sorted.json`

## Using MicroWorld for Retrieval

Once the KG is built (at minimum S0–S3 are required), use `microworld.py` for inference-time knowledge retrieval:

```python
from microworld import MicroWorld

# Initialize with default support/ directory
mw = MicroWorld(support_dir="./support")

# Build a knowledge-augmented prompt
result = mw.build_prompt(
    question="What structure is shown by cryo-ET in this neuron?",
    image_path="/path/to/image.png"
)

print(result["prompt"])          # Full prompt with knowledge context
print(result["matched_entities"])  # List of matched KG entities
print(result["knowledge_context"]) # Raw knowledge block
```

**Two-pass mode** (entity list provided by MLLM):

```python
entities = ["mitochondria", "cryo-electron tomography", "inner membrane"]
context = mw.build_context_from_entities(entities)
```

**CLI debug tool:**

```bash
python microworld.py "What is a ribosome?" --no_nlp
python microworld.py "cryo-EM mitochondria" --support_dir ./support
```

## Directory Structure

```
MicroWorld/
├── stages/
│   ├── microworld.py          # KG retrieval module (used at inference time)
│   ├── microWorld_s0.py       # S0: Data preparation
│   ├── microWorld_s1.py       # S1: Entity + relation extraction
│   ├── microWorld_s2.py       # S2: KG construction
│   ├── microWorld_s3.py       # S3: Entity description generation
│   ├── microWorld_s4.py       # S4: Visual embeddings
│   ├── microWorld_s5.py       # S5: K-hop neighbor precomputation
│   └── microWorld_s6.py       # S6: Similarity ranking
└── support/               # KG data directory (generated by the pipeline)
    ├── dataset_index.json
    ├── raw_triplets.json
    ├── entity_descriptions.json
    ├── KG/
    │   ├── nodes.json
    │   ├── edges.json
    │   └── graph.json
    └── visual_embeddings/
```

## Key Parameters

| Parameter | Default | Description |
|-----------|---------|-------------|
| `max_text_entities` | 6 | Max entities retrieved from question text |
| `max_visual_entities` | 3 | Max entities retrieved from image |
| `max_context_chars` | 6000 | Max characters in knowledge context |
| `freq_skip_ratio` | 0.08 | Skip entities appearing in >8% of images (too generic) |
| `no_nlp` | False | Skip scispaCy NER (use alias matching only) |
| `no_2hop` | False | Disable 2-hop neighbor expansion |
| `definition_only` | False | Only show entity definitions, skip relations |

## Citation


