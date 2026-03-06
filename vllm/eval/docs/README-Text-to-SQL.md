Bird/Spider Evaluation Pipeline
This repository contains the data processing and evaluation pipeline for Bird and Spider text-to-SQL benchmarks.

Prerequisites
Python 3.8+

Bash shell environment

Pipeline Overview
# 1. Data Processing
Process the raw datasets into the format required for evaluation:

```bash
cd eval/scripts/Text-to-SQL
bash run.sh
```
This script handles:

Data cleaning and formatting

Schema alignment

Train/validation/test split preparation

## 1.1 Bird Evaluation
Benchmark References
Bird: https://github.com/AlibabaResearch/DAMO-ConvAI/tree/main/bird

Run evaluation on the Bird benchmark:

```bash
cd eval/scripts/Text-to-SQL/
git clone https://github.com/AlibabaResearch/DAMO-ConvAI/tree/main/bird
cd DAMO-ConvAI-main/bird/llm
bash ./run/run_evaluation.sh
```
This executes:

Model inference on Bird test set

Execution accuracy calculation

Metric reporting

## 1.2 Spider Evaluation
Run evaluation on the Spider benchmark:

```bash
cd eval/scripts/Text-to-SQL/
git clone https://github.com/taoyds/spider
cd spider-master
python evaluation.py --gold spider_gold.txt --pred spider_pred.txt --etype all --db spider_data/database/ --table spider_data/tables.json
```
```bash
--gold spider_gold.txt : From: https://yale-lily.github.io/spider
--pred spider_pred.txt : Your model output
--db spider_data/database/ : From: https://yale-lily.github.io/spider
--table spider_data/tables.json : From: https://yale-lily.github.io/spider
```
This performs:

Spider test set evaluation

Exact set matching accuracy

Component-wise metric computation

Notes
Ensure all paths are correctly set in the respective configuration files

The pipeline assumes proper dataset placement in the expected directories

Results are saved in the respective output folders after each evaluation step