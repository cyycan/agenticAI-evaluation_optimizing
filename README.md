# Evaluating & Optimizing AI Agents

A comprehensive guide to evaluating, optimizing, and deploying AI agents in production. Covers evaluation frameworks, prompt engineering, fine-tuning, and inference optimization techniques.

---

## Table of Contents

1. [AI Agent Evaluation](#1-ai-agent-evaluation)
   - [Task Success & Outcome Quality](#task-success--outcome-quality)
   - [Rule-Based Evaluation (BLEU & ROUGE)](#rule-based-evaluation-bleu--rouge)
   - [Learned Metrics (BERTScore, COMET)](#learned-metrics-bertscore-comet)
   - [Human Evaluation](#human-evaluation)
   - [LLM-as-a-Judge](#llm-as-a-judge)
   - [Interaction Quality & User Experience](#interaction-quality--user-experience)
   - [Performance, Reliability & Efficiency](#performance-reliability--efficiency)
2. [Agent Operational Readiness](#2-agent-operational-readiness)
   - [Observability & Logging](#observability--logging)
   - [Common Monitoring Tools](#common-monitoring-tools)
   - [Product Metrics](#product-metrics)
3. [Prompt Engineering vs Fine-Tuning](#3-prompt-engineering-vs-fine-tuning)
   - [Prompt Engineering Best Practices](#prompt-engineering-best-practices)
   - [Fine-Tuning](#fine-tuning)
   - [When to Use Each](#when-to-use-each)
4. [Optimizing Inference Speed & Model Costs](#4-optimizing-inference-speed--model-costs)
   - [GPU Landscape](#gpu-landscape)
   - [Quantization](#a-quantization)
   - [Model Pruning](#b-model-pruning)
   - [Distillation](#c-distillation)
   - [Inference Optimization Techniques](#inference-optimization-techniques)
   - [Deployment Strategies](#deployment-strategies)

---

## 1. AI Agent Evaluation

> **Why it matters:** Evaluation ensures agents do what they're supposed to.

There are three main evaluation dimensions:

1. **Task Success / Outcome Quality** — Did the agent achieve the intended goal?
2. **Interaction Quality** — How did the user feel about the experience?
3. **Performance, Reliability & Efficiency** — Can it scale and stay stable?

---

### Task Success & Outcome Quality

The most fundamental question: *Did the agent achieve the intended task?*

- **Simple tasks:** Yes/no outcomes — e.g., "Did it book the meeting?"
- **Complex tasks:** Require richer evaluation methods:

| Method | Description |
|---|---|
| Rule-based (BLEU / ROUGE) | Surface-level n-gram overlap |
| Learned models (BERTScore, COMET, BLEURT) | Semantic similarity via embeddings |
| Human evaluation | Score relevance, clarity, and helpfulness |
| LLM-as-a-Judge | Use another model to approximate human judgment at scale |
---

### Rule-Based Evaluation (BLEU & ROUGE)

Rule-based metrics rely on **n-gram overlap** — contiguous sequences of n words — rather than meaning or context.

**What is an n-gram?**

Given the sentence *"the cat sat on the mat"*:

| n | Type | Examples |
|---|---|---|
| 1 | Unigram | the, cat, sat, on, the, mat |
| 2 | Bigram | the cat, cat sat, sat on, on the, the mat |
| 3 | Trigram | the cat sat, cat sat on, sat on the, on the mat |

**BLEU vs ROUGE:**

| Aspect | BLEU | ROUGE |
|---|---|---|
| Measures | Precision of n-gram overlap | Recall (or F1) of n-gram or sequence overlap |
| Best for | Machine translation, structured output | Summarization, flexible generation |
| Bias | Rewards concise, exact matches | Rewards coverage and phrasing variety |

---

### Learned Metrics (BERTScore, COMET)

#### BERTScore

Evaluates **semantic similarity** using contextual embeddings and cosine similarity.

1. Pass both the reference and candidate sentences through a pre-trained model (e.g., BERT)
2. BERT outputs a contextual embedding vector for each token
3. Compute pairwise cosine similarity between token embeddings across reference and candidate
   - e.g., "cold" vs "freezing" → high similarity
   - e.g., "weather" vs "it" → lower similarity

#### COMET

Goal: Predict **human judgment** (fluency, adequacy) for a translation using regression over contextual embeddings.

1. **Inputs:** Translation, Reference, Source sentence
2. **Encoding:** Multilingual transformer (XLM-R) encodes each into contextual embeddings
3. **Feature Combination:** Combine raw vectors, element-wise differences and products
4. **Regression:** Feedforward network predicts a human score (trained with MSE loss)

**Why learned metrics over BLEU/ROUGE?**
- Use pre-trained LMs fine-tuned to match human ratings
- Better at capturing fluency, adequacy, and semantic meaning

---

### Human Evaluation

When machine metrics fall short, human annotators evaluate:
- Relevance and accuracy
- Clarity and helpfulness
- Overall quality

**Chatbot Arena** (LMSYS) is a notable crowdsourced benchmark where users compare LLM outputs head-to-head without knowing which model generated them, driving rankings based on real-world preferences.

---

### LLM-as-a-Judge

Instead of relying on human raters or hardcoded metrics, an LLM acts as the evaluator.

**Input Types:**
- **Point-wise:** One output evaluated independently — *"Rate this response from 1–5 stars."*
- **Pair-wise / List-wise:** Outputs compared against each other — *"Which of these two responses is better?"*

**Output Types:**
- **Score:** Numerical rating (e.g., 3 out of 5)
- **Ranking:** Orders outputs from best to worst
- **Selection:** Pass/fail against a quality threshold

**Validation:** GPT-4 as a judge agrees with individual human annotators ~85% of the time, compared to 81% inter-human agreement — suggesting LLMs can be more consistent than individual humans.

**Known Biases:**
- **Length bias:** Longer responses score higher
- **Position bias:** First response scores higher
- **Narcissism bias:** A model scores its own responses higher
- **Cost-quality trade-off:** Better judges cost more to run

---

### Interaction Quality & User Experience

How does the user *feel* about their interaction?

| Metric | Description |
|---|---|
| **Satisfaction** | CSAT surveys, NPS ratings |
| **Containment rate** | % of queries fully handled without human escalation |
| **Escalation rate** | High rate signals the agent isn't trusted or capable enough |
| **Conversation depth** | Number of turns until issue is resolved |

**Systematic feedback collection:**
- In-App signals (👍/👎, report buttons)
- Active solicitation after low-confidence responses
- Behavioral signals: edits, re-asks, time-to-next-query
- Tooling: Humanloop, TruLens, LangSmith

**Turning feedback into improvements:**
- Fine-tune models with high-quality labeled data
- Refine prompts based on common issues
- Augment training data to fill knowledge gaps

---

### Performance, Reliability & Efficiency

**Performance:**
- **Latency:** Goal is sub-500ms for good UX
- **Token usage:** More tokens = higher API costs; monitor average tokens/request
- **Compute usage:** Watch CPU, GPU, and memory for bottlenecks
- **Throughput:** Requests the system can handle per second under load

**Reliability:**
- **Error rate:** Track failures, timeouts, bad responses
- **OOD Handling:** Does it recognize and gracefully reject out-of-scope queries?
- **Graceful degradation:** If tools fail, the agent should offer fallback paths

---

## 2. Agent Operational Readiness

### Observability & Logging

Effective evaluation is impossible without comprehensive logging. Essential logs include:

- **Trace internal reasoning:** Prompts, tool calls, intermediate model outputs
- **Metadata:** Timestamps, token counts, user/session IDs for correlation
- **Feedback signals:** Thumbs up/down, corrections, rewrites, abandonment
- **Key events:** User queries, API responses, tool failures, model errors

**Alerts & Monitoring:**
- Set alerts for metric anomalies (error rate spikes, latency surges)
- Use centralized dashboards (Grafana, W&B) to unify logs and metrics
- Review trends regularly: rising latency, growing cost, performance degradation

---

### Common Monitoring Tools

| Tool | Focus | Best For | Watch Out For |
|---|---|---|---|
| **LangSmith** | LangChain debugging & eval | Teams building agents with LangChain needing step-level introspection | Not ideal for non-LangChain stacks |
| **Arize AI** | ML-first model observability | Monitoring LLM/ML performance, drift, and quality in production | Learning curve; can be pricey |
| **Datadog** | Full-stack observability | Unified monitoring across infra, APIs, and apps including AI | Complex setup; not ML-specific |
| **Helicone** | LLM API logging & caching | Quick logging, cost tracking, and caching for LLM API calls | Limited advanced features |

**Quick decision guide:**
- **LangSmith** → LangChain agent step tracing
- **Arize** → Model-level quality, tracing, and evaluation
- **Datadog** → Full-stack infra monitoring with AI add-ons
- **Helicone** → Simple API logging and cost control with zero infra overhead

---

### Product Metrics

**User Engagement:**
- **Active users:** Unique individuals interacting with the agent over time
- **Interaction volume:** Total number of exchanges or sessions
- **Session duration:** How long users remain engaged

---

## 3. Prompt Engineering vs Fine-Tuning

### Prompt Engineering Best Practices

> Prompt engineering is the process of designing high-quality prompts that guide LLMs to produce accurate outputs.

#### 1. Clear Instructions and Constraints

- Be specific — provide detailed instructions with context
- Specify output format (JSON, bullet points, tables)
- Define tone and constraints ("use a formal tone", "do not include sensitive topics")
- Use delimiters to separate sections:

```
### Instructions:
You are an AI assistant helping students learn history. Summarize the following text in two sentences.

### Context:
This is for a high school student preparing for a quiz.

### User Input:
The Treaty of Versailles was signed in 1919 after World War I...
```

Or with XML tags:
```xml
<instructions>Summarize the following text in two sentences.</instructions>
<context>High school student preparing for a quiz.</context>
<user_input>The Treaty of Versailles was signed in 1919...</user_input>
```

#### 2. Self-Ask and Step-by-Step Reasoning (Chain of Thought)

Break complex tasks into sequential steps:

```
Step 1: What sub-questions do I need to answer first?
Step 2: Answer each sub-question.
Step 3: Combine the answers to solve the original question.
```

Example prompt templates:

- **Research-style:** *"To answer the final question, first list sub-questions that would help. Then answer each, and finally give the main answer."*
- **Decision-making:** *"First ask yourself what factors are important to consider. Then find answers to each one. Finally, make a decision based on the answers."*
- **Fact-checking:** *"Start by asking what facts would confirm or refute this claim. Then answer those one by one. Finally, say whether the claim holds up."*
- **Tool-use (ReAct style):** *"Ask what info you need from a tool. List and answer each question using that tool, then synthesize a final answer."*

#### 3. Iterative Refinement and Few-Shot Learning

- Treat prompt writing as an experiment — change wording, structure, or examples
- Include **few-shot examples** (input-output pairs) to show the model what good looks like
- A/B test variants to measure response quality
- Add error-handling constraints: *"If you don't know, say you don't know."*
- Maintain a **prompt library** of proven patterns for different tasks

---

### Fine-Tuning

Goal: Teach a pre-trained model new patterns or behavior by updating its weights on task-specific data.

**Steps:**
1. **Collect Labeled Data** — (Input, Desired Output) pairs; questions & ideal answers, chat transcripts, etc.
2. **Prepare the Dataset** — Clean, tokenize, and ensure consistent formatting (e.g., JSONL)
3. **Choose a Base Model** — General-purpose model (GPT, LLaMA); consider size vs compute constraints
4. **Run Supervised Fine-Tuning** — Train with gradient descent; minimize difference between model output and target
5. **Validate and Test** — Evaluate on held-out data; adjust hyperparameters if needed
6. **Deploy & Monitor** — Watch for model drift, hallucinations, or task failures

**Parameter-Efficient Fine-Tuning (PEFT):**

Instead of updating all weights, PEFT modifies only a small subset — cheaper, faster, and easier to deploy.

| Technique | What It Does |
|---|---|
| **LoRA** (Low-Rank Adaptation) | Injects trainable rank-decomposed matrices into existing layers |
| **Prefix Tuning** | Learns task-specific "prefix" vectors prepended to input embeddings |
| **Adapter Layers** | Adds small trainable modules between frozen layers |
| **BitFit** | Only fine-tunes bias terms in the model |

---

### When to Use Each

| Consideration | Prompt Engineering | Fine-Tuning |
|---|---|---|
| **Task Complexity** | Simple, well-defined tasks | Complex, domain-specific tasks |
| **Data Availability** | Limited or no training data | Substantial high-quality labeled data available |
| **Cost & Effort** | Lower initial cost | Higher initial cost for data prep and training |
| **Performance** | Good for quick iterations | Higher accuracy, more nuanced responses |
| **Generalization** | Base model already performs reasonably | Base model struggles with domain nuances |

---

## 4. Optimizing Inference Speed & Model Costs

### GPU Landscape

| GPU | Year | VRAM | TFLOPs | Price | Cloud Cost |
|---|---|---|---|---|---|
| V100 | 2017 | 16/32 GB | 125 (FP16) | $1,500 | $2.50/hr |
| T4 | 2018 | 16 GB | 65 (INT8) | $450 | $0.40/hr |
| A10 | 2021 | 24 GB | 150 (INT8) | $1,000 | $1.00/hr |
| A100 | 2020 | 40/80 GB | 312 (FP16) | $9,000 | $4.50/hr |
| H100 | 2022 | 80 GB | 1,000–2,000 (FP8) | $32,000 | $11/hr |
| RTX 3090 | 2020 | 24 GB | 35–40 (FP16) | $1,000 | $1.25/hr |
| B100 | 2024 | 192 GB | 4,000 (FP8) | $43,000 | ~$20/hr |

**Memory rule of thumb:** Each parameter ≈ 2 bytes (FP16). Add ~30% overhead for activations and KV cache.

| Model Size | Example | FP16 VRAM | INT8 VRAM | Runs On |
|---|---|---|---|---|
| 1B | DistilGPT2 | 3 GB | 1.5 GB | Any GPU |
| 7B | LLaMA 2 7B | 20 GB | 10 GB | RTX 3090, A10, A100 |
| 70B | LLaMA 2 70B | 200 GB | 75 GB | B100, Multi-GPU |
| 175B | GPT-3 | 375 GB | 190 GB | Multi-GPU (8× A100+) |
| ~1T | GPT-4 | 400 GB | 200 GB | Multi-GPU (8× H100+) |

---

### A. Quantization

**Intuition:** Reduce the numerical precision of model weights to save memory and speed up inference.

- FP32 → FP16: 2× memory reduction
- FP32 → INT8: 4× memory reduction

**Why it works:**
- **Smaller memory footprint** — INT8 model is 4× smaller than FP32; loads faster
- **Faster data transfer** — Moving smaller data types from memory to compute is quicker
- **Faster computation** — Modern CPUs/GPUs perform integer operations faster than floating-point

**Tooling:** ONNX Runtime, TensorRT — act as a compiler + runtime for executing quantized graphs efficiently on GPU.

**Tradeoff:** May result in slight accuracy loss.

---

### B. Model Pruning

**Intuition:** Set some weights to zero to save storage and compute.

| Method | Description | Hardware |
|---|---|---|
| **Unstructured Pruning** | Removes individual weights → sparse models | Requires special hardware/software |
| **Structured Pruning** | Removes entire neurons or attention heads | Works on standard hardware (A100, H100) |

**When to use each:**
- **Unstructured** → Maximum compression with the right infrastructure
- **Structured** → Real-world latency/throughput gains on typical hardware

**Tradeoff:** Potential accuracy loss if done aggressively.

---

### C. Distillation

**Intuition:** Train a small "student" model to mimic a large "teacher" model.

**Process:**
1. Pass training data through both the **teacher** (large pretrained model) and **student** (smaller model)
2. Backpropagate two loss signals:
   - **Distillation Loss** — Student mimics teacher's output distribution
   - **Cross-Entropy Loss** — Student matches true labels directly

**Why not just train the student model directly?**
The teacher's output logits contain richer information than hard labels alone, enabling faster and better learning.

**Tradeoff:** More expensive to train than standard fine-tuning.

---

### Optimization Summary

| Optimization | Summary | Tradeoff |
|---|---|---|
| **Quantization** | Reduce precision to shrink model size and latency up to 4× | Potential accuracy loss |
| **Pruning** | Set weights to zero; requires sparse execution engine | Potential accuracy loss |
| **Distillation** | Train a smaller model to mimic a larger one | Expensive to train; can modify architecture |

**The Deployment Tradeoff Triangle** — You can't have all three at once:

```
         Accuracy
           /\
          /  \
         /    \
        /      \
Low Dev Cost — Low Inference Cost
```

- **Full Model** → Highest accuracy, minimal dev time, but expensive to deploy
- **Quantization** → Lower inference cost, simple dev, but can hurt accuracy
- **Distillation** → Balances accuracy and inference cost, but requires extra training

---

### Inference Optimization Techniques

#### Caching

**KV Cache (Model Level):**
- When generating token-by-token, transformers compute Key (K) and Value (V) matrices for all preceding tokens
- KV cache stores these matrices to avoid recomputation on each new token
- Non-negotiable for any conversational or autoregressive model

**Semantic Cache (Application Level):**
- Stores full queries and their final answers
- Uses embeddings for semantic matching — not just exact matches
- Example: "How can I change my password?" and "I forgot my login and need to reset it" resolve to the same cached answer

#### Batching

Process multiple inference requests together to maximize GPU utilization and reduce per-request cost.

#### Speculative Decoding

**Key intuitions:**
1. Reviewing results is cheaper than generating them from scratch
2. Not every token is equally hard to generate

A small "draft" model generates candidate tokens quickly; the large model verifies them in parallel. Easy tokens are accepted; hard ones are regenerated by the full model.

#### Software Enhancements

- `scaled_dot_product_attention` (PyTorch)
- **FlashAttention** — Memory-efficient attention implementation
- **Group-Query Attention** — Reduces KV cache size for large models

---

### Deployment Strategies

- **Right-size models per task** — Don't use a 70B model for simple classification
- **Hardware/model match** — Match quantization level and model size to available GPU VRAM
- **Pay for what you use** — Consider managed APIs vs self-hosting based on traffic patterns
- **Inference stack options:**
  - Managed services (OpenAI, Anthropic, Together AI)
  - **vLLM** — High-throughput inference server for self-hosted models
  - **NVIDIA Triton** — Production inference serving for multiple model types

---

## 5. Summary

| Topic | What | How | Tools |
|---|---|---|---|
| **AI Agent Evaluation** | Task success, interaction quality, performance | Rule-based (BLEU, ROUGE), learned models, human eval, LLM-as-a-Judge | BERTScore, COMET, Chatbot Arena |
| **Operational Readiness** | Observability, monitoring, feedback loops | Comprehensive logging, product metrics, user engagement | LangSmith, Arize AI, Datadog, Helicone |
| **Prompt Engineering / Fine-Tuning** | Get more from your model | Clear prompts with examples and CoT; fine-tune with labeled data | PEFT, LoRA, HuggingFace |
| **Inference Optimization** | Reduce cost and latency | Quantization, pruning, distillation, caching, speculative decoding | ONNX, TensorRT, vLLM, FlashAttention |
