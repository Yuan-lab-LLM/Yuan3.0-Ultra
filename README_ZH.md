<div align="center">
<h1>
  源3.0 Ultra多模态基础大模型
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
        <b>简体中文</b> |
        <a href="./README.md">English</a>
    <p>
</h4>


-----



##  最近更新 🎉🎉

* **[2026-03-04]** **发布源3.0 Ultra多模态基础大模型，面向企业级场景的万亿参数旗舰模型。**




## 1. 简介

Yuan3.0 Ultra 采用统一多模态模型架构，由视觉编码器、语言主干网络与多模态对齐模块组成，实现视觉与语言信息的协同建模。其中，语言主干网络基于混合专家（MoE）架构构建，包含 103 层 Transformer，训练初始阶段参数规模 1515B，通过 LAEP 方法创新，团队在预训练过程中将模型参数优化至 1010B，预训练算力效率提升 49%。Yuan 3.0 Ultra模型的激活参数为 68.8B。此外，模型还引入了 Localized Filtering Attention（LFA）机制，有效强化对语义关系的建模能力，相比经典 Attention 结构可获得更高的模型精度表现。


Yuan3.0 Ultra 对 <a href="https://github.com/Yuan-lab-LLM/Yuan3.0" target="_blank">**Yuan3.0 Flash**</a> 中提出的 RIRM（反思抑制奖励机制）进行了改进，通过对反思次数引入奖励约束，使模型在获得可靠答案后主动减少无效反思，同时在复杂问题中保留必要的推理深度。这一机制有效缓解了快思考模式下的“过度思考”（overthinking）现象。训练结果表明，在这一受控快思考策略下，模型精度显著提升，同时推理过程中生成的 token 数量持续下降，实现了准确性与计算效率的同步优化。

同时，我们发布了 Yuan3.0 Ultra 模型的<a href="./Docs/Yuan3.0_Ultra Paper.pdf">**技术报告**</a>，可以通过报告查看更详细的技术细节与测评结果。


<div align=center> <img src=https://github.com/Yuan-lab-LLM/Yuan3.0-Ultra/blob/main/Docs/Yuan3.0%20Ultra-architecture.png width=80% />

Fig.1: Yuan3.0 Ultra 多模态大模型架构图

</div>


## 核心特性

🚀 **Layer-Adaptive Expert Pruning（LEAP）专家裁剪方法** ：创新层自适应专家裁剪（LAEP）算法是一种专为混合专家大语言模型（MoE LLM）预训练设计的新算法，实现万亿参数大模型预训练效率提升49%，参数量降低33%（从 1515B 降低至 1010B）。


⚡ **快思考强化学习（RAPO）**  ：改进了 Yuan3.0 Flash 中提出的反思抑制奖励机制（RIRM），并将其集成到一种快速思考强化学习范式中用于后训练。改进后的 RIRM 使训练准确率提升 16.33%，输出令牌长度减少 14.38%。


🎯 **企业场景深度优化**  ：Yuan3.0 Ultra 在设计阶段即面向企业真实应用场景进行能力构建。它不只是一个强大的多模态模型，更是一个能够驱动复杂智能体（Agent）的“核心引擎”。在多项企业级任务评测中，模型展现出稳定优势：
* 检索增强生成（RAG）：在 ChatRAG、DocMatix 等评测中取得领先成绩，能精准定位并利用企业私域知识，为回答提供可靠依据；
* 复杂表格与文档理解：在 MMTab 等多任务基准中，展现了对结构化数据的深度解析能力，轻松应对财报、审批表等复杂文档；
* 高质量总结生成：在 SummEval 上，能生成既忠于原文又高度概括的摘要，确保信息传递的准确与高效；
* 智能体工具调用：在 BFCL 评测中，精通多步骤工具调用与协作，为自动化执行复杂任务打下坚实基础；
* 数据库查询语句生成（Text2SQL）：在 Spider1.0 与 BIRD 基准测试中表现出色，有力支撑面向企业结构化数据场景的精准查询。


## 2. 性能表现

Yuan3.0 Ultra 在检索增强生成、多模态文档理解、表格数据分析、内容摘要与工具调用等任务中表现突出，为企业构建文档驱动与数据驱动的智能体应用（Agent）提供核心能力支撑，详细评测结果见第 5 节。

<div align=center> <img src=https://github.com/Yuan-lab-LLM/Yuan3.0-Ultra/blob/main/Docs/Yuan3.0-Ultra-benchmarks.png width=80% />

Fig.2: Yuan3.0 Ultra 模型评测结果



</div>

## 3. 核心技术

### Layer-Adaptive Expert Pruning（LAEP）

大模型预训练过程的专家负载演化可分为两个阶段：
* 第一阶段：初始过渡阶段，发生在模型预训练早期，此时专家负载波动剧烈，受随机初始化影响明显，同一专家所接收的 token 数量可能在数量级上存在显著差异；
* 第二阶段：稳定阶段，此时各专家之间的 token 负载趋于稳定，每个专家接收的 token 数量仅呈现相对较小的波动；

在训练稳定阶段，专家的token负载极不均衡，少数专家承担大量计算，而部分专家长期处于低负载状态，导致算力资源浪费，训练稳定阶段最高专家与最低专家负载差异近500倍。

LAEP 在训练稳定阶段依据各层 token 分布，自适应裁剪低负载专家，并提出专家重排算法将剩余专家贪心地重新分配至各计算设备以实现均衡负载。Yuan3.0 Ultra 模型从 1515B 参数开始预训练，在训练稳定阶段应用 LAEP 方法，实现参数压缩 33.3%，预训练效率提升 49%。

### 改进的反思抑制奖励机制（RIRM）

在 Fast-thinking RL 阶段，模型易对数学、科学推理任务产生过多反思步骤。Yuan3.0 Ultra 在 Yuan3.0 Flash 的 RAPO 框架基础上改进训练机制：正确样本反思步骤越少奖励越高，错误样本反思步骤越多惩罚越重，改进后训练准确率提升 **16.33%**，输出 token 长度降低 **14.38%**。


## 4. 模型下载

**我们提供多种模型格式的下载链接：**

|    模型     |   参数量  |  精度  |   序列长度  |   模型格式   |         下载链接         |
| :----------: | :------: | :------: | :------: | :-------: |:---------------------------: |
| Yuan3.0 Ultra |    1.01万亿   |  16bit     |    64K    |    HuggingFace    | [ModelScope]( https://modelscope.cn/models/YuanLabAI/Yuan3.0-Ultra ) \| [HuggingFace]( https://huggingface.co/YuanLabAI/Yuan3.0-Ultra ) \|  [始智AI]( https://www.wisemodel.cn/models/YuanLabAI/Yuan3.0-Ultra )
| Yuan3.0 Ultra int4 |    1.01万亿   |  4bit     |    64K    |    HuggingFace    | [ModelScope]( https://modelscope.cn/models/YuanLabAI/Yuan3.0-Ultra-int4 ) \| [HuggingFace]( https://huggingface.co/YuanLabAI/Yuan3.0-Ultra-int4 ) \|  [始智AI]( https://www.wisemodel.cn/models/YuanLabAI/Yuan3.0-Ultra-int4 )



## 5. 测评结果

Yuan3.0 Ultra 在多项企业级核心基准测试中取得领先表现。

### 5.1 多模态 RAG 评测：Docmatix 🏆

Docmatix 评测模型在多页复杂文档中跨文本、表格、图像等多种模态进行信息检索、关联与精准问答的综合能力。

| 模型 | 准确率 (%) |
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

### 5.2 文本 RAG 评测：ChatRAG 🏆

ChatRAG 包含 10 项任务，涵盖长文本检索（D2D、QuAC、QReCC）、短文本与结构化检索（CoQA、DoQA、CFQA、SQA、HDial）及维基百科检索（TCQA、INSCIT）。Yuan3.0 Ultra 平均准确率达 **68.2%**，在 10 项任务中的 9 项位居首位。

| 模型 | 平均 | D2D | QuAC | QReCC | CoQA | DoQA | CFQA | SQA | TCQA | HDial | INSCIT |
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

### 5.3 多模态复杂表格理解评测：MMTab

MMTab 涵盖 15 项评测集，覆盖表格问答、事实核查、长文本表格处理等多个任务类型。Yuan3.0 Ultra 以 **62.3%** 的平均准确率超越 Claude Opus 4.6 和 Gemini 3.1 Pro，展现出全面均衡的多模态表格处理能力。

| 模型 | 平均 | TABMWP | WTQ | HiTab | TAT-QA | FeTaQA | TabFact | InfoTabs |
|---|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|
| GPT-5.1 | 55.2 | 65.0 | 60.8 | **77.8** | 61.4 | 8.7 | 52.8 | 64.3 |
| GPT-5.2 | 37.3 | 67.2 | 69.8 | 15.8 | 28.0 | 6.2 | 63.5 | 69.3 |
| Gemini 3.1 Pro | 45.1 | 80.1 | **79.6** | 48.3 | 50.5 | 9.6 | 71.1 | 74.4 |
| Claude Opus 4.6 | 39.8 | 67.6 | 76.0 | 44.1 | 44.5 | 12.0 | 30.7 | 59.6 |
| Kimi K2.5 | **66.2** | **95.9** | 79.3 | 63.9 | 62.4 | 7.4 | **90.6** | 81.8 |
| **Yuan3.0 Ultra** | 62.3 | 91.8 | 77.9 | 67.6 | **74.9** | **39.2** | 90.4 | **89.7** |


*完整 15 项任务详细结果请参见技术报告。*

---

### 5.4 文本摘要生成评测：SummEval 🏆

SummEval 从词汇重叠（ROUGE-1/2）、语义相似度（BERTScore）与事实一致性（SummaC）三个维度综合评估摘要质量，是智能体应用中历史信息压缩能力的重要参考。Yuan3.0 Ultra 平均精度 **62.8%**。

| 模型 | 平均 | ROUGE-1 | ROUGE-2 | BERTScore | SummaC |
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

### 5.5 工具调用评测：BFCL V3

BFCL V3 从静态函数选择（Non-Live AST）、动态实时执行（Live AST）、多轮上下文维护（Multi-turn）、相关性检测（Relevance）与无关调用拒绝（Irrelevance）等维度评估真实工具调用能力。Yuan3.0 Ultra 各项表现均衡，平均得分 **67.8%**，在无关调用拒绝（Irrelevance Detection，86.0%）上尤为突出。

| 模型 | 平均 | Non-Live AST | Live AST | Multi-turn | Relevance | Irrelevance |
|---|:---:|:---:|:---:|:---:|:---:|:---:|
| Qwen3-235B-A22B | 68.0 | 87.9 | 77.0 | 40.1 | **83.3** | 76.3 |
| Claude-3.7-Sonnet | 58.6 | 41.3 | 78.4 | 48.4 | 72.2 | 81.4 |
| GPT-5.2 | 60.6 | 80.9 | 76.2 | 24.6 | 72.2 | 79.7 |
| Gemini 3.1 Pro | **78.8** | **91.5** | **84.9** | **60.3** | 61.1 | **88.2**  |
| Claude Opus 4.6 | 74.9 | 88.2 | 78.9 | 59.8 | 61.1 | 78.0 |
| Kimi K2.5 | 70.6 | 86.4 | 78.6 | 48.6 | 61.1 | 77.0 |
| **Yuan3.0 Ultra** | 67.8 | 81.7 | 74.5 | 45.3 | 66.7 | 86.0 |




---

### 5.6 Text-to-SQL 评测：Spider 1.0 & BIRD

Spider 1.0 和 BIRD 是 Text-to-SQL 领域的两项基准评测，Yuan3.0 Ultra 在 Spider 1.0 及 BIRD 评测上表现出色。

| 模型 | Spider 1.0| BIRD  |
|---|:---:|:---:|
| Qwen3.5-397B-A17B | 82.4 | 39.6 |
| DeepSeek-V3.2 | 80.7 | 38.9 |
| Kimi K2.5 | 82.7 | **43.5** |
| **Yuan3.0 Ultra** | **83.9** | 39.2 |




## 6. 快速开始

### 6.1 Yuan3.0 Ultra 推理

Yuan3.0 Ultra 支持 bfloat16 和 int4 量化模型，具体使用方法，详见[QuickStart](vllm/README_Yuan_zh.md)


### 6.2 Yuan3.0 Ultra 训练

我们提供了 Yuan3.0 Ultra 模型的监督微调脚本与强化学习脚本，详见微调训练[文档](rlhf/docs/instruct_tuning_zh.md)与强化学习[文档](rlhf/docs/RL_training_zh.md)。



## 7. 协议声明
使用源 3.0 代码及模型需遵循[《源 3.0 模型许可协议》](https://github.com/Yuan-lab-LLM/Yuan3.0?tab=License-1-ov-file)，源 3.0 模型支持商用，不需要申请授权，请您了解并遵循，勿将开源模型和代码及基于开源项目产生的衍生物用于任何可能给国家和社会带来危害的用途以及用于任何未经过安全评估和备案的服务。

尽管模型在训练时我们已采取措施尽力确保数据的合规性和准确性，但模型参数量巨大且受概率随机性因素影响，我们无法保证输出内容的准确性，且模型易被输入指令所误导，本项目不承担开源模型和代码导致的数据安全、舆情风险或发生任何模型被误导、滥用、传播、不当利用而产生的风险和责任。您将对通过使用、复制、分发和修改模型等方式利用该开源项目所产生的风险与后果，独自承担全部责任。


