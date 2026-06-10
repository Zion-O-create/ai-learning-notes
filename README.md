# ai-learning-notes
# AI Learning Notes 🚀

Welcome to my public learning tracker! I am committing my notes here every day as public evidence of my learning journey.

* **Course:** Generative AI with Large Language Models (DeepLearning.AI + AWS)
* **Start Date:** May 17, 2026
* **Timeline:** 18-Day Sprint
* **Daily Target:** 30 minutes / day ⏱️

---

## 📚 Course Progress & Syllabus

### Week 1: Generative AI Use Cases & Project Lifecycle
- [x] Course Intro & Generative AI Overview (~21% Progress Point reached!)
- [x] LLM Use Cases, Tasks, and Transformers Architecture
- [x] Prompt Engineering & Generative Configuration
- [x] Pre-training Large Language Models & Computational Challenges
- [ ] Week 1 Quiz & Lab 1: Summarize Dialogue

### Week 2: Fine-Tuning and Evaluating LLMs
- [ ] Instruction Fine-Tuning & Multi-task Training
- [ ] Model Evaluation & Benchmarks (ROUGE, BLEU)
- [ ] Parameter Efficient Fine-Tuning (PEFT) & LoRA
- [ ] Week 2 Quiz & Lab 2: Fine-tune a Generative AI Model

### Week 3: Reinforcement Learning & LLM-Powered Applications
- [ ] Aligning Models with Human Values (RLHF)
- [ ] Proximal Policy Optimization & Reward Hacking
- [ ] Model Optimizations for Deployment
- [ ] LLM Application Architectures (Reasoning, Chain-of-Thought, ReAct)
- [ ] Week 3 Quiz & Lab 3: Fine-tune FLAN-T5 with RLHF

---
* May 17, 2026*

*JUNE 10, 2026*
# DeepLearning.AI — Generative AI with LLMs (Week 1 Notes)
> *A comprehensive compilation of my hand-written sanity checks, mathematical breakdowns, and survival guides for the upcoming AI revolution.*

---

## 1. Introduction to Generative AI & The LLM Lifecycle

### Core Concepts
* **Generative AI (GenAI):** A specialized subset of traditional Machine Learning (ML). It refers specifically to the models that underlie and power applications capable of creating fresh content instead of just predicting if an email is spam or not.
* **Foundation Models:** The heavy lifters. These are massive models trained on colossal volumes of data (e.g., GPT family, LLaMA, BERT, PaLM, BLOOM, FLAN-T5). They possess broad capabilities and are ready to be fine-tuned for your specific niche tasks so you don't have to spend $10 million pretraining your own.
* **Modalities:** GenAI isn't a one-trick pony. It natively handles multiple data streams including text, images, video, speech, and code.

### The Glossary of Terms You Can Use to Sound Smart
* **Context Window:** The cognitive "working memory" available to the model. Think of it as the maximum amount of text the model can juggle in its hands at one single time during a prompt session.
* **Prompt:** The text, question, or desperate plea passed into an LLM.
* **Completion:** The output text generated back by the model. 
* **Inference:** The act of actually using a model to generate text (i.e., sending a prompt and waiting for a completion).

### LLM Use Cases & Common Tasks
* **Next-Word Prediction:** The basic engine driving everything else (essentially glorified, hyper-intelligent autocomplete).
* **Text Processing:** Essay writing, summarization, and extracting entities out of unstructured text.
* **Translation:** Converting text between natural human languages, or mapping natural human instructions directly into executable programming code.
* **Augmentation:** Connecting LLMs to external databases, vector stores, and APIs to prevent them from confidently making things up.

> 💡 **The Golden Rule of Parameter Scaling:** As model parameters increase, the model's raw capacity to internalize nuance, grammar, and complex linguistic structures increases dramatically.

---

## 2. Text Generation Before Transformers (The Dark Ages)

### Recurrent Neural Networks (RNNs)
Before 2017, we relied heavily on RNNs for sequential data. Spoiler alert: **They weren't great at text generation.**
* **The Scale Problem:** They require massive, inefficient compute resources as data scales up.
* **The Growth Problem:** Computational resource requirements grow exponentially with text length.
* **The Memory Problem:** They process word-by-word sequentially. By the time an RNN reaches the end of a long sentence, it has already forgotten how the sentence started. It completely fails at long-range prediction tasks.

---

## 3. The Transformer Architecture
Introduced in the legendary 2017 paper *"Attention Is All You Need"* by Google and the University of Toronto. It threw out sequential processing entirely and changed the world.

### Key Advantages over Legacy Networks
* **Scales Efficiently:** Purpose-built to maximize modern multi-core GPU clusters.
* **Parallel Processing:** Processes an entire input sequence simultaneously rather than word-by-word. This means we can train on internet-scale datasets in humanly acceptable timeframes.
* **Attention to Input Meaning:** Learns the relevance and context of *all* words in a sentence relative to *all other* words at the exact same time.

### Weights & Maps
* **Attention Weights:** Dynamically calculated values that dictate exactly how much "attention" a word should pay to every other word in the sequence. 
* **Attention Map:** A visual matrix used to map and debug these attention weights, letting engineers see exactly what the model is looking at when it processes language.

---

## 4. Inside the Transformer: A Step-by-Step Data Journey

When a prompt enters a Transformer, it goes through a highly engineered processing pipeline:

```
[Raw Text Input] 
       │
       ▼
 [Tokenization]        --> Turns words into numeric IDs
       │
       ▼
  [Embedding]         --> Maps tokens to a high-dimensional concept space
       │
       ▼
[Positional Encoding]  --> Restores word order context to the parallel data
       │
       ▼
 [Self-Attention]     --> Evaluates relationships between all words
       │
       ▼
 [Feed-Forward Net]   --> Processes attention outputs into raw token scores (Logits)
       │
       ▼
   [Softmax]          --> Converts raw scores into clean percentage probabilities
       │
       ▼
 [Predicted Token]
```

### 1. Tokenize (Input Processing)
Computers don't read words; they read numbers. Tokenization splits raw text strings into chunked numbers (tokens) based on a pre-defined dictionary.

### 2. Embedding Layer
* A trainable, high-dimensional vector space.
* Every token is assigned a unique vector that occupies a coordinate in this massive space.
* **The Magic:** Words with similar meanings or contextual usage are naturally adjusted over training to sit close to each other geometrically (e.g., "king" and "queen" or "apple" and "banana").

### 3. Positional Encoding
Because Transformers process all words simultaneously in parallel, they lose track of word order. Without help, a Transformer wouldn't know the difference between *"The dog ate my homework"* and *"My homework ate the dog"*. **Positional Encoding** adds a clever mathematical wave signal to the embeddings to preserve word order context.

### 4. Self-Attention & Multi-Head Attention
* **Self-Attention:** Analyzes how different parts of an input sequence depend on and relate to one another.
* **Multi-Headed Self-Attention:** Instead of calculating attention once, the model boots up multiple attention "heads" (often between 12 and 100+) to run in parallel independently. 
* **The Catch:** Each head learns a completely different structural feature of the language. One head might focus on pronouns, another on verb tenses, and another on poetic relationships.

### 5. Feed-Forward Network (FFN)
Once attention is calculated, the data passes into a classic feed-forward neural network block. This processes the attention vectors and outputs a vector of **logits**—raw, unnormalized scores for every single possible word inside the model's entire dictionary.

### 6. Softmax Layer
The Softmax layer takes those chaotic raw logits and normalizes them into clean probability scores that add up to exactly 100%. The token with the highest percentage score is our winner and becomes the most likely candidate for the next word.

---

## 5. Sequence-to-Sequence (Seq2Seq) Walkthrough

Seq2Seq tasks (like translating French to English) represent the classic, full architecture layout originally intended by the Transformer designers.

1. **The Encoder Phase:** The source text (e.g., a French sentence) is tokenized, embedded, augmented with positional encodings, and pushed through the **Encoder** multi-head attention blocks. The output is a deep, mathematical representation of the sentence's structure and core meaning.
2. **The Handshake:** This deep representation is inserted directly into the middle of the **Decoder** block to heavily guide its own internal attention mechanism.
3. **The Decoder Loop:**
    * A special **Start of Sequence (<SOS>)** token is fed into the Decoder to jumpstart the generation.
    * The Decoder looks at its history and the Encoder's handshake data to compute its first attention map.
    * The output flows through an FFN and a Softmax layer to predict the very first English word token.
    * **The Cycle:** That newly predicted English word is fed *back* into the input of the Decoder along with the previous tokens. The loop runs over and over until the model confidently predicts an **End of Sequence (<EOS>)** token, signaling that its job is complete.

---

## 6. Transformer Variants: Choosing Your Weapon

You don't always need the full machine. Depending on your goals, you can chop a Transformer in half:

### 1. Encoder-Decoder Models (The Full Transformer)
* Uses both halves of the machine. Excellent for tasks where the input length and output length can be completely different.
* **Primary Use Cases:** Translation, complex summarization.
* **Famous Examples:** BART, FLAN-T5.

### 2. Encoder-Only Models
* Only reads and analyzes inputs; does not generate long open-ended text. It maps tokens to bidirectional representations.
* **Primary Use Cases:** Tasks where you need to understand text context globally—Sentence Classification, Sentiment Analysis, and Named Entity Recognition (NER).
* **Famous Examples:** BERT, RoBERTa.

### 3. Decoder-Only Models
* The reigning champions of the modern era. They excel at predicting the next token and generating open-ended text.
* Thanks to brute-force parameter scaling, modern decoder-only models can handle almost any task via zero- or few-shot inference.
* **Primary Use Cases:** Text generation, conversational AI, code synthesis.
* **Famous Examples:** GPT family, LLaMA, BLOOM, Claude.

---

## 7. Prompt Engineering & In-Context Learning (ICL)

Before rushing to retrain a model with millions of custom data points, you should always try to guide it using the available space inside its context window. This approach is called **In-Context Learning (ICL)**.

* **Zero-Shot Inference:** You give the model an instruction with absolutely zero completed examples. You just pray it's smart enough to understand.
    * *Example:* `"Classify this review: I love this meal! Sentiment:"`
* **One-Shot Inference:** You provide exactly one clear, completed example of how the task should be performed right before your actual prompt text.
    * *Example:* ```text
      Classify this review: I love this meal! Sentiment: Positive
      Classify this review: I hate this desk! Sentiment:
      ```
* **Few-Shot Inference:** You provide multiple distinct examples (usually 3 to 5) to firmly establish a clear pattern for the model to replicate.
    * *Example:*
      ```text
      Classify this review: I love this meal! Sentiment: Positive
      Classify this review: I hate this desk! Sentiment: Negative
      Classify this review: The color is okay. Sentiment: Neutral
      Classify this review: This engine exploded! Sentiment:
      ```

> ⚠️ **Developer Warning:** If an LLM is still failing to follow instructions after 5 to 6 distinct few-shot examples, stop tinkering with the prompt. It's time to transition to actual **Fine-Tuning**.

---

## 8. Generative Configuration (Inference Parameters)

Inference parameters are knobs you can turn *during runtime* to change how an LLM selects tokens without modifying the underlying weights learned during training.

### Training vs. Inference Parameters
* **Training Parameters:** Baked permanently into the neural network's architecture weights during the training phase.
* **Inference Parameters:** Swappable controls that guide how the Softmax output is sampled during generation.

### The Knobs and Dials

#### 1. Max New Tokens
A hard ceiling parameter. It forces the model to stop generating text after reaching a specific number of tokens, regardless of whether it has finished its sentence or emitted an `<EOS>` token. Use this to save on compute costs.

#### 2. Decoding Strategies: Greedy vs. Random
* **Greedy Decoding:** The default setting. The model looks at the Softmax probability output and unconditionally picks the single token with the absolute highest score. 
    * *The Catch:* Great for short, factual text, but highly prone to falling into recursive loops where it repeats phrases or full sentences over and over, making the writing look completely robotic.
* **Random Sampling:** Introduces organic variance. The model treats the Softmax output as a true lottery wheel where words with higher percentages have a bigger slice of the wheel, but lower-probability words still have a puncher's chance of winning. This prevents repetitive loops but can cause the model to drift off into hallucinations if left unconstrained.

#### 3. Top-K vs. Top-P (Keeping Randomness Sane)
To keep random sampling from picking completely bizarre words, we use truncating filters:
* **Top-K:** Tells the model to sort all dictionary tokens by probability, isolate the top $K$ options, completely discard everything else, and then run a weighted random selection among just those chosen few.
* **Top-P:** Uses a **cumulative probability threshold** ($P$). The model ranks tokens and starts adding up their percentage scores from top to bottom. The moment the total sum hits or exceeds your threshold $P$ (e.g., 0.90 or 90%), it locks down that group and throws out the rest before sampling.

#### 4. Temperature (The Mood Slider)
Temperature acts as a direct mathematical scaling multiplier applied directly inside the final Softmax layer. It alters the sharpness of the probability distribution curve:
* **Low Temperature (< 1.0):** Sharpens the curve. High-probability words become massive, and low-probability words drop near zero. This forces the model to be predictable, conservative, and precise.
* **High Temperature (> 1.0):** Flattens the curve out completely. The gap between smart choices and random words shrinks, leading to wild creativity, poetic leaps, and absolute chaos.

---

## 9. The Generative AI Project Lifecycle
*A structured blueprint for building systems that actually make it out of a sandbox and into production.*

1. **Scope:** Define the use case precisely. Do you need an expensive jack-of-all-trades generalist model, or a lean, hyper-optimized specialist model trained for a single purpose? Accurately narrowing your scope here saves months of wasted engineering later.
2. **Select:** Evaluate whether you should adopt an open-source pretrained Foundation Model off a hub (like Hugging Face) or build a custom architecture from scratch. Always read the **Model Cards** to understand training biases and license constraints.
3. **Adapt & Align:** Put the chosen model through its paces. Run prompt engineering, configure inference parameters, execute fine-tuning on proprietary data, and align the outputs with human preferences. Evaluate metrics rigorously.
4. **Application Integration:** Compress, optimize, and deploy your model weights to infrastructure. Augment your live application runtime with external components (RAG systems, agents, and external guardrail APIs) to deliver a robust software experience.

```
       [1. SCOPE]              [2. SELECT]               [3. ADAPT & ALIGN]           [4. INTEGRATE]
┌──────────────────────┐ ┌──────────────────────┐ ┌──────────────────────────────┐ ┌──────────────────┐
│ Define Use Case      │ │ Choose Pretrained    │ │ Prompt Eng. -> Fine-Tuning   │ │ Optimize Weights │
│ Narrow down scale    │ │ Evaluate Model Cards │ │ Human Alignment -> Evaluate  │ │ Augment with RAG │
└──────────────────────┘ └──────────────────────┘ └──────────────────────────────┘ └──────────────────┘
```

---

## 10. Deep Dive: Pretraining Objectives & Architectures

| Architecture Type | Primary Pretraining Objective | Context Direction | Native Strengths | Famous Archetypes |
| :--- | :--- | :--- | :--- | :--- |
| **Autoencoding** (Encoder-Only) | **Masked Language Modeling (MLM):** Hides tokens at random; forces model to reconstruct sentences by using context from both sides. | **Bidirectional** (Reads left-to-right AND right-to-left simultaneously) | Deep sentence structure comprehension, classifications, sentiment extraction, NER. | BERT, RoBERTa |
| **Autoregressive** (Decoder-Only) | **Causal Language Modeling (CLM):** Forces the network to strictly predict the next upcoming token based on preceding context history. | **Unidirectional** (Reads strictly left-to-right) | Flawless text generation, multi-task generalizability, conversational agents. | GPT family, LLaMA |
| **Sequence-to-Sequence** (Encoder-Decoder) | **Span Corruption:** Replaces entire variable-length token strings with dedicated custom **Sentinel Tokens** for the decoder to unpack. | **Mixed** (Bidirectional input reading -> Unidirectional generation) | Translating languages, structural text summarization, answering hard context questions. | T5, BART |

---
