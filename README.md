# GoEmotions-MA: Fine-Grained Emotion Analysis with Structured Prompting and Multi-Agent Reasoning

> **Enhancing Fine-Grained Emotion Classification using Structured Prompt Engineering and Multi-Agent LLM Architectures**

This repository presents a **research-grade emotion analysis framework** built on Large Language Models (LLMs), leveraging **structured prompt engineering** and **multi-agent reasoning** to improve fine-grained emotion classification performance on the **GoEmotions dataset (28 labels)**.

The project systematically explores how *prompt modularization* and *agent-based decomposition* can mitigate cognitive overload in single LLMs and yield more **accurate, consistent, and explainable emotion predictions**, particularly in nuanced and ambiguous emotional contexts :contentReference[oaicite:0]{index=0}.

---

## 📌 Key Contributions

- **Structured Prompt Design**  
  A modular prompt framework consisting of Persona, Emotion Definitions, Guidelines, Output Schema, and Few-shot Examples, enabling consistent and interpretable emotion reasoning.

- **Multi-Agent Emotion Reasoning Pipelines**  
  Three distinct multi-agent architectures are proposed and evaluated:
  - Linear (Hierarchical) Multi-Agent
  - Court-style (Debate-based) Multi-Agent
  - Probabilistic Multi-Agent

- **Explainable Emotion Outputs**  
  All agents are required to output **emotion labels with explicit textual evidence**, improving transparency and auditability.

- **Comprehensive Evaluation**  
  Performance is evaluated using **macro-F1**, **micro-F1**, precision, and recall, with comparisons against traditional BERT-based classifiers.

---

## 🧠 Problem Motivation

Conventional sentiment analysis systems are limited to coarse polarity (positive/negative), failing to capture **subtle emotional nuances** such as *disappointment vs. sadness* or *annoyance vs. anger*.  
Although LLMs demonstrate strong zero-shot capabilities, **unstructured prompts often lead to inconsistent predictions and hallucinations**, especially in fine-grained emotion tasks.

This project addresses the following research questions:

- Can **structured prompts alone**, without model fine-tuning, achieve competitive performance in fine-grained emotion classification?
- Does **multi-agent decomposition** improve reasoning stability and classification accuracy?
- How can we enforce **explainability** in LLM-based emotion analysis?

---

## 📂 Dataset

### GoEmotions
- **Source**: Google Research (2020)
- **Size**: ~58,000 Reddit comments
- **Labels**: 28 emotions (27 fine-grained + neutral)
- **Structure**: Multi-label, real-world conversational text

GoEmotions is particularly suitable for evaluating LLM reasoning capabilities due to its **semantic overlap between emotion classes** and **high label imbalance**, which makes macro-F1 a critical metric :contentReference[oaicite:1]{index=1}.

---

## 🏗️ System Architecture

### 1️⃣ Single-Agent (Baseline)
A single GPT-4o-mini instance performs emotion classification using a fully structured prompt:
- Persona role assignment
- Explicit emotion definitions
- Step-by-step reasoning guidelines
- Strict JSON output schema
- Few-shot demonstrations (grid-searched)

---

### 2️⃣ Linear Multi-Agent (Hierarchical)
A **top-down reasoning pipeline**:
