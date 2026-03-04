<div align="center">
<h1>
  源3.0多模态基础大模型
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
  <a href="https://arxiv.org/abs/2601.01718"><img alt="arXiv"
    src="https://img.shields.io/badge/arXiv-Yuan3.0%20Paper-b31b1b?logo=arxiv&logoColor=white"/></a>


  
</div>



<h4 align="center">
    <p>
        <b>简体中文</b> |
        <a href="./README.md">English</a>
    <p>
</h4>


-----



##  最近更新 🎉🎉

* **[2026-03-04]** **发布源3.0 多模态大模型，面向企业级场景的万亿参数旗舰模型：Yuan3.0 Ultra**





## 1. 简介

Yuan3.0 Ultra 是由 **YuanLab.ai 团队** 研发的万亿参数多模态基础大模型。模型采用统一多模态架构，语言主干基于 103 层 MoE Transformer 构建，初始参数规模 1515B。通过提出LAEP算法 （Layer-Adaptive Expert Pruning），在预训练阶段完成结构优化，将参数规模压缩至 1010B，单次推理激活 68.8B 参数，显著提升算力利用效率。

Yuan3.0 Ultra 基于 DAPO 强化学习框架，结合改进的 RIRM（反思抑制奖励机制），形成以 Fast-thinking 为核心的训练策略，在大规模强化学习过程中实现精度提升与推理 token 持续下降，使模型能力扩展与计算效率同步优化。同时，我们发布了Yuan3.0 Ultra模型的<a href="https://arxiv.org/abs/2601.01718" target="_blank">**技术报告**</a>，可以通过论文查看更详细的技术细节与测评结果。



<div align=center> <img src=https://github.com/Yuan-lab-LLM/Yuan3.0-Ultra/blob/main/Docs/Yuan3.0-Ultra-architecture.png width=80% />

Fig.1: Yuan3.0 Ultra多模态大模型架构图

</div>


## 核心特性

🚀 **训练与推理双重效率突破** ：基于 LAEP 算法实现 33.3% 参数压缩与 49% 训练效率提升；结合快思考强化学习机制，显著降低推理 token 消耗，实现更高性价比的智能输出。

🧠 **万亿参数 MoE 架构**  ：总参数 1010B，激活参数 68.8B。采用分层自适应专家剪枝（LAEP）与专家重排机制，在保持模型表达能力的同时，大幅提升算力利用率。

🎯 **企业场景深度优化**  ：在 RAG、多模态文档理解、复杂表格推理、内容摘要与工具调用等任务中表现领先，专为企业级复杂业务场景设计。

⚡ **快思考强化学习（RIRM + DAPO）**  ：通过反思抑制奖励机制，在大规模强化学习过程中实现“精度显著提升 + 推理长度持续下降”的同步优化，使模型更加高效、克制且稳定。

🤖 **Agent 友好型设计**  ：在多轮检索、工具调用与复杂流程决策任务中展现稳定且高质量的推理能力，是构建文档驱动与数据驱动型智能 Agent 的核心基础模型。



## 2. 性能表现

Yuan3.0 Ultra 基于统一多模态架构，能够对企业复杂文档与表格数据进行高质量理解，在检索增强生成（RAG）、多模态文档理解、表格数据分析、内容摘要与工具调用等任务中表现突出，为企业构建文档驱动与数据驱动的智能体应用（Agent）提供核心能力支撑。在 Docmatix（67.4%）、ChatRAG（68.2%）、MMTab（62.3%）、SummEval（62.8%）和 BFCL V3（67.8%）五项核心基准上均取得领先，详细评测结果见第 5 节。

<div align=center> <img src=https://github.com/Yuan-lab-LLM/Yuan3.0-Ultra/blob/main/Docs/Yuan3.0-Ultra-benchmarks.png width=80% />

Fig.2: Yuan3.0 Ultra模型评测结果








</div>

## 3. 核心技术

### Layer-Adaptive Expert Pruning（LAEP）

MoE 预训练中专家负载长期不均衡：进入稳定阶段后，少数专家持续承担大量 token，低频专家长期欠利用，层内最高/最低负载比可达数百倍。LAEP 在负载稳定后依据各层 token 分布，自适应裁剪低利用率专家，并配合**专家重排**将剩余专家贪心地重新分配至各计算设备以均衡负载。应用于 1515B 基础模型：总参数压缩至 **1010B**（↓33.3%），训练效率从 62.14 提升至 **92.60 TFLOPS/GPU**（↑49%），测试损失低于基础模型，优于辅助负载均衡损失方法。

### 改进的反思抑制奖励机制（RIRM）

在 Fast-thinking RL 阶段，模型易对数学、科学推理任务产生过多反思步骤（overthinking）。Yuan3.0 Ultra 在 Yuan3.0 Flash 的 DAPO 框架基础上改进 RIRM：正确样本反思步骤越少奖励越高，错误样本反思步骤越多惩罚越重，改进后训练准确率提升 **16.33%**，输出 token 长度降低 **14.38%**，AIME 2024 / MATH-500 分别达到 **47.8%** / **93.1%**。




## 4. 模型下载

**我们提供多种模型格式的下载链接：**

|    模型     |   参数量  |  精度  |   序列长度  |   模型格式   |         下载链接         |
| :----------: | :------: | :------: | :------: | :-------: |:---------------------------: |
| Yuan3.0 Ultra |    1.01万亿   |  16bit     |    32K    |    HuggingFace    | [ModelScope]( https://modelscope.cn/models/YuanLabAI/Yuan3.0-Ultra ) \| [HuggingFace]( https://huggingface.co/YuanLabAI/Yuan3.0-Ultra ) \|  [始智AI]( https://www.wisemodel.cn/models/YuanLabAI/Yuan3.0-Ultra )
| Yuan3.0 Ultra int4 |    1.01万亿   |  4bit     |    32K    |    HuggingFace    | [ModelScope]( https://modelscope.cn/models/YuanLabAI/Yuan3.0-Ultra-int4 ) \| [HuggingFace]( https://huggingface.co/YuanLabAI/Yuan3.0-Ultra-int4 ) \|  [始智AI]( https://www.wisemodel.cn/models/YuanLabAI/Yuan3.0-Ultra-int4 )





## 5. 测评结果

Yuan3.0 Ultra 在多项企业级核心基准测试中取得领先表现，持续超越 GPT-5.2、Gemini-3.1-Pro、Claude-Opus-4.6 及 Kimi-K2.5 等前沿模型。

### 5.1 多模态 RAG 评测：Docmatix 🏆

Docmatix 评测模型在多页复杂文档中跨文本、表格、图像等多种模态进行信息检索、关联与精准问答的综合能力，Yuan3.0 Ultra 以 **67.4%** 的准确率位列所有参评模型首位。

| 模型 | 准确率 (%) |
|---|:---:|
| Gemini 3.1 Pro | 35.3 |
| Kimi K2.5 | 36.9 |
| InternVL3-78B | 43.0 |
| Claude 3.5-Sonnet | 42.6 |
| OpenAI o3 | 45.6 |
| Claude Opus 4.6 | 46.2 |
| GPT-5.2 | 48.4 |
| GPT-5.1 | 48.5 |
| GPT-4o | 56.8 |
| **Yuan3.0 Ultra** | **67.4** |

---

### 5.2 文本 RAG 评测：ChatRAG 🏆

ChatRAG 包含 10 项任务，涵盖长文本检索（D2D、QuAC、QReCC）、短文本与结构化检索（CoQA、DoQA、CFQA、SQA、HDial）及维基百科检索（TCQA、INSCIT）。Yuan3.0 Ultra 平均准确率达 **68.2%**，在 10 项任务中的 9 项位居首位。

| 模型 | 平均 | D2D | QuAC | QReCC | CoQA | DoQA | CFQA | SQA | TCQA | HDial | INSCIT |
|---|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|
| DeepSeek-R1 | 39.4 | 21.5 | 22.2 | 42.4 | 62.5 | 24.7 | 81.5 | 82.1 | 30.7 | 38.0 | 28.7 |
| OpenAI o3 | 44.1 | 23.1 | 20.8 | 40.4 | 69.4 | 18.6 | 67.8 | 86.7 | 45.9 | 41.3 | 26.7 |
| GPT-5.2 | 45.6 | 30.2 | 23.1 | 47.0 | 64.8 | 25.3 | 72.3 | 79.1 | 38.3 | 45.3 | 30.9 |
| GPT-5.1 | 46.1 | 28.2 | 23.2 | 45.4 | 68.8 | 20.9 | 73.1 | 81.3 | 44.7 | 45.4 | 30.0 |
| Gemini 3.1 Pro | 49.7 | 33.1 | 27.3 | 47.0 | 73.5 | 34.2 | 75.7 | 85.5 | 42.4 | 48.2 | 30.3 |
| DeepSeek-V3 | 50.5 | 31.6 | 28.9 | 49.3 | 77.0 | 26.1 | 83.5 | 82.1 | 46.7 | 47.4 | 32.1 |
| GPT-4o | 50.5 | 32.8 | 26.6 | 49.3 | 76.1 | 28.8 | 81.9 | 81.1 | 49.8 | 41.3 | 26.7 |
| Claude Opus 4.6 | 52.9 | 35.3 | 26.6 | 49.4 | 76.4 | 37.3 | 86.5 | 85.5 | 50.2 | 48.9 | 33.2 |
| Kimi K2.5 | 53.6 | 34.6 | 30.9 | 49.9 | 82.5 | 35.8 | 82.3 | 83.6 | 50.8 | 51.1 | 34.4 |
| **Yuan3.0 Ultra** | **68.2** | **55.8** | **54.5** | **57.3** | **94.6** | **63.4** | 79.8 | **91.0** | **72.4** | **72.9** | **40.0** |

---

### 5.3 多模态复杂表格理解评测：MMTab 🏆

MMTab 涵盖 15 项权威基准，覆盖表格问答、事实核查、长文本表格处理等多个任务类型，是企业办公自动化场景的核心评测集。Yuan3.0 Ultra 以 **62.3%** 的平均准确率超越 Claude Opus 4.6 和 Gemini 3.1 Pro，展现出全面均衡的表格推理与生成能力。

| 模型 | 平均 | TABMWP | WTQ | HiTab | TAT-QA | FeTaQA | TabFact | InfoTabs |
|---|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|
| GPT-4V | 29.9 | 60.5 | 48.0 | 27.5 | 32.5 | 11.0 | 45.5 | 65.6 |
| GPT-5.2 | 37.3 | 67.2 | 69.8 | 15.8 | 28.0 | 6.2 | 63.5 | 69.3 |
| Claude Opus 4.6 | 39.8 | 67.6 | 76.0 | 44.1 | 44.5 | 12.0 | 30.7 | 59.6 |
| Gemini 3.1 Pro | 45.1 | 80.1 | 79.6 | 48.3 | 50.5 | 9.6 | 71.1 | 74.4 |
| GLM-4.5V | 52.0 | 88.2 | 77.4 | 51.5 | 62.7 | 5.3 | 89.4 | 79.5 |
| GPT-5.1 | 55.2 | 64.9 | 60.8 | 77.8 | 61.4 | 8.7 | 52.8 | 64.3 |
| **Yuan3.0 Ultra** | **62.3** | **91.8** | **77.9** | **67.6** | **74.9** | **39.2** | **90.4** | **89.7** |
| Kimi K2.5 | 66.2 | 95.9 | 79.3 | 63.9 | 62.4 | 7.4 | 90.6 | 81.8 |

*完整 15 项任务详细结果请参见技术报告。*

---

### 5.4 文本摘要生成评测：SummEval 🏆

SummEval 从词汇重叠（ROUGE-1/2）、语义相似度（BERTScore）与事实一致性（SummaC）三个维度综合评估摘要质量，是智能体应用中历史信息压缩能力的重要参考。Yuan3.0 Ultra 平均得分 **62.8%**，大幅领先所有参评模型。

| 模型 | 平均 | ROUGE-1 | ROUGE-2 | BERTScore | SummaC |
|---|:---:|:---:|:---:|:---:|:---:|
| Gemini-2.0-Flash | 45.4 | 24.8 | 8.7 | 85.7 | 29.5 |
| Claude-3.5-Sonnet | 45.4 | 24.1 | 8.3 | 85.2 | 30.7 |
| GPT-4o | 46.5 | 25.0 | 8.9 | 85.9 | 32.5 |
| GPT-5.2 | 48.6 | 30.3 | 10.7 | 84.9 | 36.4 |
| Gemini 3.1 Pro | 48.5 | 32.4 | 11.4 | 85.4 | 34.3 |
| GPT-5.1 | 49.4 | 27.5 | 10.2 | 84.6 | 40.5 |
| Kimi K2.5 | 49.8 | 32.3 | 11.3 | 85.4 | 38.2 |
| Claude Opus 4.6 | 49.9 | 33.1 | 11.0 | 85.9 | 37.8 |
| DeepSeek-V3 | 59.3 | 25.5 | 9.2 | 86.3 | 68.2 |
| **Yuan3.0 Ultra** | **62.8** | **59.1** | **41.0** | **91.1** | **45.4** |

---

### 5.5 工具调用评测：BFCL V3

BFCL V3 从静态函数选择（Non-Live AST）、动态实时执行（Live AST）、多轮上下文维护（Multi-turn）、相关性检测与无关调用拒绝等维度评估真实工具调用能力。Yuan3.0 Ultra 各项表现均衡，平均得分 **67.8%**，在无关调用拒绝（Irrelevance Detection，86.0%）上尤为突出，体现出成熟可靠的工具增强推理能力。

| 模型 | 平均 | Non-Live AST | Live AST | Multi-turn | Relevance | Irrelevance |
|---|:---:|:---:|:---:|:---:|:---:|:---:|
| OpenAI o3-mini | 51.3 | 42.1 | 77.3 | 26.1 | 77.8 | 80.7 |
| GPT-4.1-nano | 53.0 | 76.7 | 64.3 | 19.9 | 94.4 | 59.4 |
| Claude-3.7-Sonnet | 58.6 | 41.3 | 78.4 | 48.4 | 72.2 | 81.4 |
| GPT-5.2 | 60.6 | 80.9 | 76.2 | 24.6 | 72.2 | 79.7 |
| Qwen3-235B-A22B | 67.9 | 87.9 | 77.0 | 40.1 | 83.3 | 76.3 |
| **Yuan3.0 Ultra** | **67.8** | **81.7** | **74.5** | **45.3** | 66.7 | **86.0** |
| Kimi K2.5 | 70.6 | 86.4 | 78.6 | 48.6 | 61.1 | 77.0 |
| Claude Opus 4.6 | 74.9 | 88.2 | 78.9 | 59.8 | 61.1 | 78.0 |
| Gemini 3.1 Pro | 78.8 | 91.5 | 84.9 | 60.3 | 61.1 | 88.2  |

---

## 5.6 Text-to-SQL 评测：Spider 1.0 & BIRD

Spider 1.0 和 BIRD 是 Text-to-SQL 领域的两项权威基准。Yuan3.0 Ultra 在 Spider 1.0 上以 **83.90%** 实现精度领先，在 BIRD 评测上表现同样出色。

| 模型 | Spider 1.0 (Acc. %) | BIRD (Acc. %) |
|---|:---:|:---:|
| DeepSeek-V3.2 | 80.7 | 38.9 |
| Qwen3.5-397B-A17B | 82.4 | 39.6 |
| Kimi K2.5 | 82.70 | **43.5** |
| **Yuan3.0 Ultra** | **83.9** | 39.2 |





## 6. 快速开始

**6.1 Yuan3.0 Ultra推理**

Yuan3.0 Ultra 支持 bfloat16 和 int4 量化模型，具体使用方法，详见[QuickStart](vllm/README_Yuan_zh.md)


**6.2 数据处理**

我们提供了数据处理的脚本，参考数据处理[说明文档](rlhf/docs/data_process_zh.md)。

**6.3 模型训练**

我们提供了Yuan3.0 Ultra模型的监督微调脚本与强化学习流程，详见微调训练[文档](rlhf/docs/instruct_tuning_zh.md)与强化学习说明[文档](rlhf/docs/RL_training_zh.md)。



##  5. 协议声明
使用源3.0代码及模型需遵循[《源3.0模型许可协议》](https://github.com/Yuan-lab-LLM/Yuan3.0?tab=License-1-ov-file)，源3.0模型支持商用，不需要申请授权，请您了解并遵循，勿将开源模型和代码及基于开源项目产生的衍生物用于任何可能给国家和社会带来危害的用途以及用于任何未经过安全评估和备案的服务。

尽管模型在训练时我们已采取措施尽力确保数据的合规性和准确性，但模型参数量巨大且受概率随机性因素影响，我们无法保证输出内容的准确性，且模型易被输入指令所误导，本项目不承担开源模型和代码导致的数据安全、舆情风险或发生任何模型被误导、滥用、传播、不当利用而产生的风险和责任。您将对通过使用、复制、分发和修改模型等方式利用该开源项目所产生的风险与后果，独自承担全部责任。


