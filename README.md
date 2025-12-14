# GoEmotions 
## Fine-Grained Emotion Analysis with Structured Prompting and Multi-Agent Reasoning

> **Enhancing Fine-Grained Emotion Classification using Structured Prompt Engineering and Multi-Agent LLM Architectures**

This repository presents a **research-grade emotion analysis framework** based on Large Language Models (LLMs).
The project demonstrates how **structured prompt engineering** and **multi-agent reasoning architectures**
can significantly improve the **accuracy, consistency, and explainability** of fine-grained emotion classification
on the **GoEmotions dataset (28 emotion labels)**.

Rather than relying on model fine-tuning, this work shows that
**reasoning structure and task decomposition alone** can unlock substantial performance gains
in LLM-based emotion analysis systems :contentReference[oaicite:0]{index=0}.

---

## 📌 Key Contributions

- **Structured Prompt Engineering Framework**
  - Modular prompt design consisting of:
    - Persona role assignment
    - Explicit emotion definitions
    - Clear reasoning guidelines
    - Strict JSON output schema
    - Few-shot demonstrations
  - Enforces consistent reasoning and reduces hallucination.

- **Multi-Agent Emotion Reasoning Architectures**
  - Design and evaluation of three multi-agent systems:
    - Linear (Hierarchical) Multi-Agent
    - Court-Style (Debate-Based) Multi-Agent
    - Probabilistic Multi-Agent
  - Each agent specializes in a well-defined reasoning scope.

- **Explainable Emotion Classification**
  - All predictions include **explicit textual rationales**
    grounded in the input text.

- **Comprehensive Empirical Evaluation**
  - Evaluated using Precision, Recall, micro-F1, and macro-F1.
  - Compared against traditional BERT-based classifiers.

---

## 🧠 Motivation

Conventional sentiment analysis systems typically focus on **coarse polarity**
(e.g., positive vs. negative), which is insufficient for capturing
the **subtle emotional nuances** present in real-world text.

Examples of challenging distinctions include:
- *annoyance vs. anger*
- *disappointment vs. sadness*
- *curiosity vs. surprise*

While modern LLMs exhibit strong zero-shot reasoning abilities,
**unstructured prompts often lead to inconsistent predictions and over-inference**.
This project addresses the following research question:

> *Can structured prompting and agent-based reasoning compensate for the lack of fine-tuning in fine-grained emotion analysis?*

---

## 📂 Dataset

### GoEmotions

- **Source**: Google Research (2020)
- **Size**: Approximately 58,000 Reddit comments
- **Labels**: 28 emotion categories (27 fine-grained emotions + neutral)
- **Structure**: Multi-label, real-world conversational data

GoEmotions is particularly challenging due to:
- Semantic overlap between emotion classes
- Severe label imbalance
- Frequent presence of implicit or ambiguous emotions

For these reasons, **macro-F1** is used as the primary evaluation metric :contentReference[oaicite:1]{index=1}.

---

## 🏗️ System Architectures

This project explores multiple reasoning architectures for LLM-based emotion classification.
All systems are designed to enforce **structured reasoning**, **polarity consistency**, and **explainable outputs**.

---

### 1️⃣ Single-Agent Baseline

The single-agent baseline uses a single GPT-4o-mini model guided by a fully structured prompt.

The prompt is modularized into:
- **Persona Role**: Assigns the model the role of an expert emotion analyst
- **Emotion Definitions**: Explicit semantic definitions for all emotion labels
- **Guidelines**: Step-by-step reasoning constraints and labeling rules
- **Output Schema**: Strict JSON format enforcing structured outputs
- **Few-shot Examples**: Curated GoEmotions samples for inference stabilization

This configuration serves as the baseline for evaluating the impact of multi-agent decomposition.

---

### 2️⃣ Linear Multi-Agent (Hierarchical)

The **Linear Multi-Agent architecture** decomposes emotion classification
into a hierarchical pipeline, where each agent focuses on a reduced reasoning space.

**Hierarchical Linear Multi-Agent Pipeline**

- **Tier 1 — Polarity Agent**  
  Classifies the input text into one of four polarity categories:  
  *(Positive, Negative, Neutral, Ambiguous)*

- **Tier 2 — Ekman Emotion Agent**  
  Determines the emotion within Ekman’s seven basic emotions,
  while strictly preserving the polarity constraint:  
  *(Joy, Sadness, Anger, Fear, Disgust, Surprise, Neutral)*

- **Tier 3 — GoEmotions Agent**  
  Performs fine-grained classification into the full set of  
  **28 GoEmotions labels** and outputs a **textual rationale**
  explaining why each emotion was selected

By progressively narrowing the classification space,
this architecture reduces cognitive overload and improves reasoning consistency.

---

### 3️⃣ Court-Style Multi-Agent (Debate-Based)

The **Court-Style Multi-Agent architecture** is inspired by legal reasoning processes
and is designed to resolve ambiguous cases through structured debate.

This system consists of:

- **Neutral Filter Agent**  
  Identifies purely factual or weakly emotional inputs

- **Candidate Emotion Selector**  
  Determines whether the emotion is clearly settled or requires deliberation

- **Advocate Agents**  
  Each advocate defends a candidate emotion using explicit textual evidence

- **Judge Agent**  
  Evaluates all arguments and applies strict decision rules to determine the final label(s)

This architecture improves robustness in borderline cases
and enhances interpretability by exposing conflicting reasoning paths.

---

### 4️⃣ Probabilistic Multi-Agent

The **Probabilistic Multi-Agent architecture** estimates emotion likelihoods
using a hierarchical probability-based approach.

The pipeline includes:
- Polarity probability estimation
- Ekman-level emotion probability estimation
- Fine-grained GoEmotions probability estimation
- Threshold-based emotion selection

While this approach emphasizes probabilistic transparency,
it exhibited lower overall performance compared to reasoning-based hierarchical agents.

---

## 📊 Experimental Results

### Overall Performance (Macro-F1)

| Method                     | Precision | Recall | Macro-F1 |
|---------------------------|-----------|--------|----------|
| BERT-based Classifier     | 0.73      | 0.46   | 0.55     |
| Single-Agent LLM          | 0.46      | 0.52   | 0.46     |
| Linear Multi-Agent        | 0.50      | 0.51   | 0.50     |
| Court-Style Multi-Agent   | 0.49      | 0.47   | 0.47     |
| Probabilistic Multi-Agent | 0.37      | 0.44   | 0.37     |

**Key Insight**  
Although supervised BERT models still outperform LLM-based systems overall,
**multi-agent LLM architectures consistently outperform single-agent prompting**,
highlighting the importance of reasoning decomposition :contentReference[oaicite:2]{index=2}.

---

## 🔍 Why Structured Prompting Works

- Reduces over-inference and hallucination
- Clarifies emotion boundaries
- Improves prediction consistency
- Enables reproducibility
- Produces explainable and auditable outputs

This transforms LLMs from **black-box classifiers**
into **transparent emotion reasoning systems**.

---

## 🛠️ Tech Stack

- **Model**: GPT-4o-mini (OpenAI API)
- **Language**: Python
- **Prompting**: Structured, modular prompt engineering
- **Evaluation**: Precision, Recall, micro-F1, macro-F1
- **Dataset**: GoEmotions

---

## 🚀 Applications

- Customer Experience (CX) analytics
- Early risk and crisis detection
- Social media monitoring
- Emotion-aware conversational agents
- Business intelligence and decision support systems

---

## ⚠️ Limitations

- API-dependent inference environment
- No model fine-tuning performed
- Performance gap with supervised deep learning models
- Sensitivity to rare emotion classes (e.g., *disgust*)

---

## 🔮 Future Work

- Cross-model validation (GPT-4.1, Claude, Gemini)
- Dynamic agent routing strategies
- Emotion trajectory tracking in dialogue
- Domain adaptation and personalization
- Data augmentation for underrepresented emotions

---

## 📜 Reference

If you use this work, please cite:

> **Enhancing Fine-Grained Emotion Analysis with Structured Prompting and Multi-Agent Reasoning in LLMs**  
> Hanyang University, 2025 :contentReference[oaicite:3]{index=3}

---

## ⭐ Final Note

This repository demonstrates that **LLM performance is not solely determined by model scale**,
but by **how reasoning is structured and distributed**.
Well-designed prompts and collaborative agents can unlock
**reliable, explainable, and production-ready emotion analysis systems**.
