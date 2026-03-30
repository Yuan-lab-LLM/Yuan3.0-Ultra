<div align="center">
<h1>
  Yuan3.0 Ultra Multimodal Foundation Large Language Model
</h1>
</div>

<hr>
<div align="center" style="line-height: 1;">
  <a href="https://huggingface.co/YuanLabAI"><img alt="Hugging Face"
    src="https://img.shields.io/badge/%F0%9F%A4%97%20Hugging%20Face-Yuan3.0-ffc107?color=ffc107&logoColor=white"/></a>
  <a href="https://www.modelscope.cn/profile/YuanLabAI"><img alt="ModelScope"
    src="https://img.shields.io/badge/💾%20ModelScope-Yuan3.0-6b4fbb?color=6b4fbb&logoColor=white"/></a>
  <a href="https://x.com/YuanAI_Lab"><img alt="Twitter Follow"
    src="https://img.shields.io/badge/Twitter-Yuanlabai-white?logo=x&logoColor=white"/></a>

</div>



<h4 align="center">
    <p>
        <a href="./README_ZH.md">简体中文</a> |
        <b>English</b>
    <p>
</h4>


-----



## Recent Updates 🎉🎉

* **[2026-03-04]** **Yuan3.0 Ultra Multimodal Foundation Large Model is released — a trillion-parameter flagship model designed for enterprise-grade scenarios.**

## 1. Introduction

Yuan3.0 Ultra employs a unified multimodal model architecture, integrating a vision encoder, a language backbone, and a multimodal alignment module to enabling synergistic modeling of visual and linguistic information. The language backbone is built on a Mixture-of-Experts (MoE) architecture, featuring 103 Transformer layers. The model was pre-trained from scratch original with 1515B parameters. Through the innovative Layer-Adaptive Expert Pruning (LAEP) algorithm, the parameter count was reduced to 1010B during pre-training, improving pre-training efficiency by 49%. The activated parameter count for Yuan3.0 Ultra is 68.8B. Furthermore, the model incorporates a Localized Filtering-based Attention (LFA) mechanism, which effectively enhances the modeling of semantic relationships and achieves higher accuracy compared to classical attention architectures.

Yuan3.0 Ultra enhances the Reflection Inhibition Reward Mechanism (RIRM) proposed in <a href="https://github.com/Yuan-lab-LLM/Yuan3.0" target="_blank">**Yuan3.0 Flash**</a>. By incorporating reward constraints based on the number of reflection steps, the model actively reduces ineffective reflections after arriving at the "first correct answer," while retaining the necessary reasoning depth for complex problems. This approach effectively mitigates the "overthinking" phenomenon in fast-thinking reinforcement learning. Training results demonstrate that under this controlled fast-thinking strategy, the model’s accuracy improves significantly, while the number of tokens generated during reasoning continually decreases—achieving simultaneous gains in both accuracy and computational efficiency.

Additionally, the <a href="./Docs/Yuan3.0_Ultra Paper.pdf">**technical report**</a> for Yuan3.0 Ultra has been released, which provides more detailed technical specifications and evaluation results.


<div align=center> <img src=https://github.com/Yuan-lab-LLM/Yuan3.0-Ultra/blob/main/Docs/Yuan3.0%20Ultra-architecture.png width=80% />

Fig.1: Yuan3.0 Ultra Multimodal Architecture

</div>

## Core Features

🚀 **Layer-Adaptive Expert Pruning (LAEP)**: The innovative Layer-Adaptive Expert Pruning (LAEP) algorithm is a novel method developed specifically for pre-training Mixture-of-Experts (MoE) Large Language Models. It improves pre-training efficiency by 49% and reduces the total parameter count by 33% (from 1515B to 1010B).


⚡ **Fast-Thinking Reinforcement Learning（RAPO）**: Enhanced the Reflection Inhibition Reward Mechanism (RIRM) introduced in Yuan3.0 Flash and integrated it into a fast-thinking reinforcement learning paradigm for post-training. The improved RIRM yields a 16.33% gain in training accuracy and reduces output token length by 14.38%.


🎯 **Deep Optimization for Enterprise Scenarios**: Yuan3.0 Ultra is designed from the ground up with its capabilities optimized for enterprise scenarios. It is not merely a powerful multimodal language model, but also serves as a robust "core engine" for driving complex intelligent agents. The model shows consistent advantages across a wide range of enterprise scenarios benchmarks:
* Retrieval-Augmented Generation (RAG): Yuan3.0 Ultra achieves leading accuracy on benchmarks like ChatRAG and DocMatix, capable of precisely locating and utilizing enterprise private domain knowledge to provide reliable evidence for responses.
* Complex Table and Document Understanding: On multi-task benchmarks such as MMTab, Yuan3.0 Ultra demonstrates deep analytical capabilities for structured data and can readily handle complex documents like financial reports and approval forms.
* High-Quality Summarization: On SummEval, Yuan3.0 Ultra can generate summaries that are both faithful to the source text and highly concise, ensuring accurate and efficient information delivery.
* Agent Tool Invocation: Yuan3.0 Ultra demonstrates proficiency in multi-step tool calling and collaboration on the BFCL benchmark, establishing a solid foundation for automating complex workflows.
* Database Query Statement Generation (Text-to-SQL): Yuan3.0 Ultra demonstrates excellent performance on the Spider 1.0 and BIRD benchmarks, delivering robust support for precise querying in enterprise scenarios that involve structured data.


## 2. Performance

Yuan3.0 Ultra delivers outstanding performance on retrieval-augmented generation, multimodal document understanding, tabular data analysis, content summarization, and tool invocation tasks, providing core capability support for enterprises building document-driven and data-driven Agent applications. Detailed evaluation results are presented in Section 5.

<div align=center> <img src=https://github.com/Yuan-lab-LLM/Yuan3.0-Ultra/blob/main/Docs/Yuan3.0-Ultra-benchmarks.png width=80% />

Fig.2: Benchmark Evaluation Results of Yuan3.0 Ultra



</div>

## 3. Core Technologies

### Layer-Adaptive Expert Pruning (LAEP)

The evolution of expert load during large model pre-training can be divided into two phases:
* **Phase 1 — Initial Transition Phase**: Occurring at the early stage of model pre-training, where expert loads exhibit substantial volatility inherited from random initialization, with the number of tokens routed to the same expert potentially varying by orders of magnitude;
* **Phase 2 — Stable Phase**: Expert token loads across experts become temporally stable, with per-expert token counts exhibiting only relatively minor fluctuations;

In the stable phase of training, expert token loads are highly imbalanced: a small number of experts carry a large share of computation while some experts remain persistently underutilized, leading to wasted computational resources. The disparity between the highest- and lowest-load experts in the stable phase can reach nearly 500×.

LAEP adaptively prunes low-load experts layer by layer according to the token distribution in each layer during the stable phase, and proposes an expert rearrangement algorithm that greedily rearranges the remaining experts across computing devices to achieve balanced load. Yuan3.0 Ultra begins pre-training with 1515B parameters and applies LAEP during the stable phase, achieving 33.3% parameter reduction and a 49% improvement in pre-training efficiency.

### Revised Reflection Inhibition Reward Mechanism (RIRM)

During the Fast-thinking RL phase, models tend to produce excessive reflection steps on mathematical and scientific reasoning tasks. Yuan3.0 Ultra refines the training mechanism on top of the RAPO framework from Yuan3.0 Flash: correct samples with fewer reflection steps receive higher rewards, while incorrect samples with more reflection steps incur heavier penalties. The improved mechanism yields a **16.33%** improvement in training accuracy and a **14.38%** reduction in output token length.


## 4. Model Download

**We provide download links for multiple model formats:**

|    Model     |   Parameters  |  Precision  |   Sequence Length  |   Format   |         Download         |
| :----------: | :------: | :------: | :------: | :-------: |:---------------------------: |
| Yuan3.0 Ultra |    1.01T   |  16bit     |    64K    |    HuggingFace    | [ModelScope]( https://modelscope.cn/models/YuanLabAI/Yuan3.0-Ultra ) \| [HuggingFace]( https://huggingface.co/YuanLabAI/Yuan3.0-Ultra ) \|  [WiseModel]( https://www.wisemodel.cn/models/YuanLabAI/Yuan3.0-Ultra )
| Yuan3.0 Ultra int4 |    1.01T   |  4bit     |    64K    |    HuggingFace    | [ModelScope]( https://modelscope.cn/models/YuanLabAI/Yuan3.0-Ultra-int4 ) \| [HuggingFace]( https://huggingface.co/YuanLabAI/Yuan3.0-Ultra-int4 ) \|  [WiseModel]( https://www.wisemodel.cn/models/YuanLabAI/Yuan3.0-Ultra-int4 )



## 5. Evaluation Results

Yuan3.0 Ultra achieves leading performance across multiple enterprise-level core benchmarks.

### 5.1 Multimodal RAG Evaluation: Docmatix 🏆

Docmatix evaluates a model's comprehensive ability to retrieve, associate, and accurately answer questions across multiple modalities (text, tables, images) within multi-page, complex documents.

| Model | Accuracy (%) |
|---|:---:|
| GPT-4o | 56.8 |
| o3 | 45.6 |
| GPT-5.1 | 48.5 |
| GPT-5.2 | 48.4 |
| Gemini 3.1 Pro | 35.3 |
| Claude Opus 4.6 | 46.2 |
| Kimi K2.5 | 36.9 |
| **Yuan3.0 Ultra** | **67.4** |

---

### 5.2 Text RAG Evaluation: ChatRAG 🏆

ChatRAG comprises 10 tasks, covering long-context retrieval (D2D, QuAC, QReCC), short-context and structured retrieval (CoQA, DoQA, CFQA, SQA, HDial), and Wikipedia-based retrieval (TCQA, INSCIT). Yuan3.0 Ultra achieves an average accuracy of **68.2%**, ranking first on 9 out of 10 tasks.

| Model | Avg. | D2D | QuAC | QReCC | CoQA | DoQA | CFQA | SQA | TCQA | HDial | INSCIT |
|---|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|
| DeepSeek-V3 | 50.5 | 31.6 | 28.9 | 49.3 | 77.0 | 26.1 | 83.5 | 82.1 | 46.7 | 47.4 | 32.1 |
| GPT-4o | 50.5 | 32.8 | 26.6 | 49.3 | 76.1 | 28.8 | 81.9 | 81.1 | 49.8 | 41.3 | 26.7 |
| o3 | 44.1 | 23.1 | 20.8 | 40.4 | 69.4 | 18.6 | 67.8 | 86.7 | 45.9 | 41.3 | 26.7 |
| DeepSeek-R1 | 39.4 | 21.5 | 22.2 | 42.4 | 62.5 | 24.7 | 81.5 | 82.1 | 30.7 | 38.0 | 28.7 |
| GPT-5.1 | 46.1 | 28.2 | 23.2 | 45.4 | 68.8 | 20.9 | 73.1 | 81.3 | 44.7 | 45.4 | 30.0 |
| GPT-5.2 | 45.6 | 30.2 | 23.1 | 47.0 | 64.8 | 25.3 | 72.3 | 79.1 | 38.3 | 45.3 | 30.9 |
| Gemini 3.1 Pro | 49.7 | 33.1 | 27.3 | 47.0 | 73.5 | 34.2 | 75.7 | 85.5 | 42.4 | 48.2 | 30.3 |
| Claude Opus 4.6 | 52.9 | 35.3 | 26.6 | 49.4 | 76.4 | 37.3 | **86.5** | 85.5 | 50.2 | 48.9 | 33.2 |
| Kimi K2.5 | 53.6 | 34.6 | 30.9 | 49.9 | 82.5 | 35.8 | 82.3 | 83.6 | 50.8 | 51.1 | 34.4 |
| **Yuan3.0 Ultra** | **68.2** | **55.8** | **54.5** | **57.3** | **94.6** | **63.4** | 79.8 | **91.0** | **72.4** | **72.9** | **40.0** |

---

### 5.3 Multimodal Complex Table Understanding Evaluation: MMTab

MMTab spans 15 evaluation sets, covering task types including table question answering, fact checking, and long-context table processing. Yuan3.0 Ultra surpasses Claude Opus 4.6 and Gemini 3.1 Pro with an average accuracy of **62.3%**, demonstrating comprehensive and well-balanced multimodal table processing capability.

| Model | Avg. | TABMWP | WTQ | HiTab | TAT-QA | FeTaQA | TabFact | InfoTabs |
|---|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|
| GPT-5.1 | 55.2 | 65.0 | 60.8 | **77.8** | 61.4 | 8.7 | 52.8 | 64.3 |
| GPT-5.2 | 37.3 | 67.2 | 69.8 | 15.8 | 28.0 | 6.2 | 63.5 | 69.3 |
| Gemini 3.1 Pro | 45.1 | 80.1 | **79.6** | 48.3 | 50.5 | 9.6 | 71.1 | 74.4 |
| Claude Opus 4.6 | 39.8 | 67.6 | 76.0 | 44.1 | 44.5 | 12.0 | 30.7 | 59.6 |
| Kimi K2.5 | **66.2** | **95.9** | 79.3 | 63.9 | 62.4 | 7.4 | **90.6** | 81.8 |
| **Yuan3.0 Ultra** | 62.3 | 91.8 | 77.9 | 67.6 | **74.9** | **39.2** | 90.4 | **89.7** |


*Full results across all 15 tasks are available in the technical report.*

---

### 5.4 Text Summarization Evaluation: SummEval 🏆

SummEval comprehensively evaluates summarization quality from three dimensions: lexical overlap (ROUGE-1/2), semantic similarity (BERTScore), and factual consistency (SummaC), serving as an important reference for historical information compression capability in Agent applications. Yuan3.0 Ultra achieves an average accuracy of **62.8%**.

| Model | Avg. | ROUGE-1 | ROUGE-2 | BERTScore | SummaC |
|---|:---:|:---:|:---:|:---:|:---:|
| DeepSeek-V3 | 59.3 | 25.5 | 9.2 | 86.3 | **68.2** |
| DeepSeek-V3.2 | 51.4 | 33.3 | 11.9 | 85.6 | 41.8 |
| GPT-4o | 46.5 | 25.0 | 8.9 | 85.9 | 32.5 |
| GPT-5.1 | 49.4 | 27.5 | 10.2 | 84.6 | 40.5 |
| GPT-5.2 | 48.6 | 30.3 | 10.7 | 84.9 | 36.4 |
| Gemini 3.1 Pro | 48.5 | 32.4 | 11.4 | 85.4 | 34.3 |
| Claude Opus 4.6 | 49.9 | 33.1 | 11.0 | 85.9 | 37.8 |
| Kimi K2.5 | 49.8 | 32.3 | 11.3 | 85.4 | 38.2 |
| **Yuan3.0 Ultra** | **62.8** | **59.1** | **41.0** | **91.1** | 45.4 |

---

### 5.5 Tool Invocation Evaluation: BFCL V3

BFCL V3 evaluates real-world tool invocation capability across dimensions including static function selection (Non-Live AST), dynamic real-time execution (Live AST), multi-turn context maintenance (Multi-turn), relevance detection (Relevance), and irrelevant call rejection (Irrelevance Detection). Yuan3.0 Ultra delivers balanced performance across all categories, achieving an average score of **67.8%**, with particular strength in Irrelevance Detection (86.0%).

| Model | Avg. | Non-Live AST | Live AST | Multi-turn | Relevance | Irrelevance |
|---|:---:|:---:|:---:|:---:|:---:|:---:|
| Qwen3-235B-A22B | 68.0 | 87.9 | 77.0 | 40.1 | **83.3** | 76.3 |
| Claude-3.7-Sonnet | 58.6 | 41.3 | 78.4 | 48.4 | 72.2 | 81.4 |
| GPT-5.2 | 60.6 | 80.9 | 76.2 | 24.6 | 72.2 | 79.7 |
| Gemini 3.1 Pro | **78.8** | **91.5** | **84.9** | **60.3** | 61.1 | **88.2**  |
| Claude Opus 4.6 | 74.9 | 88.2 | 78.9 | 59.8 | 61.1 | 78.0 |
| Kimi K2.5 | 70.6 | 86.4 | 78.6 | 48.6 | 61.1 | 77.0 |
| **Yuan3.0 Ultra** | 67.8 | 81.7 | 74.5 | 45.3 | 66.7 | 86.0 |




---

### 5.6 Text-to-SQL Evaluation: Spider 1.0 & BIRD

Spider 1.0 and BIRD are two major benchmarks in the Text-to-SQL domain. Yuan3.0 Ultra demonstrates strong performance on both evaluations.

| Model | Spider 1.0 | BIRD  |
|---|:---:|:---:|
| Qwen3.5-397B-A17B | 82.4 | 39.6 |
| DeepSeek-V3.2 | 80.7 | 38.9 |
| Kimi K2.5 | 82.7 | **43.5** |
| **Yuan3.0 Ultra** | **83.9** | 39.2 |




## 6. Quick Start

### 6.1 Yuan3.0 Ultra Inference

Yuan3.0 Ultra supports both bfloat16 and int4 quantized models. For usage details, please refer to [QuickStart](vllm/README_Yuan.md).


### 6.2 Yuan3.0 Ultra Training

We provide supervised fine-tuning scripts and reinforcement learning scripts for Yuan3.0 Ultra. Please refer to the fine-tuning training [documentation](rlhf/docs/instruct_tuning.md) and reinforcement learning [documentation](rlhf/docs/RL_training.md).



## 7. License
Use of Yuan 3.0 code and models must comply with the [Yuan 3.0 Model License Agreement](https://github.com/Yuan-lab-LLM/Yuan3.0?tab=License-1-ov-file). Yuan 3.0 models support commercial use and do not require an application for authorization. Please familiarize yourself with and adhere to the agreement. Do not use the open-source models, code, or any derivatives produced from this open-source project for any purposes that may cause harm to the nation or society, or for any services that have not undergone safety assessment and registration.

Although measures have been taken during training to ensure data compliance and accuracy to the best of our ability, given the enormous scale of model parameters and the influence of probabilistic randomness, we cannot guarantee the accuracy of generated outputs, and models are susceptible to being misled by input instructions. This project assumes no responsibility for data security risks, public opinion risks, or any risks and liabilities arising from the model being misled, misused, disseminated, or improperly exploited due to the use of open-source models and code. You shall bear full and sole responsibility for all risks and consequences arising from your use, copying, distribution, and modification of this open-source project.
