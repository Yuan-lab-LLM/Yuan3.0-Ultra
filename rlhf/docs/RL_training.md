# Yuan3.0 Ultra Reinforcement Learning

## 1. Introduction

This document provides instructions for Reflection-aware Adaptive Policy Optimization (RAPO) reinforcement learning for the Yuan3.0 Ultra model.

## 2. Usage

### Step 1: Start the Ray Cluster
```bash
# Navigate to the verl module directory
# start head node
RAY_USE_IP_ADDRESS=True ray start --head --num-cpus=64 --num-gpus=8 --port=6400 --memory=873741824000 --dashboard-host 0.0.0.0  --node-ip-address=${your_head_node_ip}
# start worker node
RAY_USE_IP_ADDRESS=True ray start --num-cpus=64 --num-gpus=8 --memory=873741824000 --dashboard-host 0.0.0.0 --address ${your_head_node_ip}:6400 --node-ip-address=${your_worker_node_ip}
```

### Step 2: Start RAPO Training
```bash
# Execute the RAPO training script for the Yuan3.0 Ultra model
cd Yuan3.0-Ultra/rlhf/verl
bash recipe/dapo/run_dapo_yuanvl_megatron_1020B.sh
```

## 3. Parameter Configuration

### 3.1 Variable Configuration
Before training, configure the following variables according to the actual paths to specify the data, model, and checkpoint storage locations:

| Variable | Type | Description |
|----------------------|------|-------------|
| `MODEL_PATH` | String | Path to the directory containing the base model files. |
| `TRAIN_16K_FILE` | String | File path for the 16K-length training dataset. |
| `TRAIN_4K_FILE` | String | File path for the 4K-length training dataset. |
| `CKPTS_DIR` | String | Directory for saving checkpoint files during training. |

### 3.2 Training Parameters
The following parameters control input/output length and long-text handling strategies. Adjust them in the training script as needed:

| Parameter | Type | Description |
|-----------|------|-------------|
| `max_prompt_length` | Integer | Maximum length of input prompt text that the model can accept. |
| `max_response_length` | Integer | Maximum text length limit for the model's generated responses. |
| `enable_overlong_buffer` | Boolean | Whether to enable the overlong buffer mechanism to handle inputs exceeding `max_prompt_length`. |
| `overlong_buffer_len` | Integer | Buffer size for the overlong buffer mechanism. |
| `overlong_penalty_factor` | Float | Penalty factor for overlong inputs (used for loss function weight adjustment). |
| `enable_overlong_prompts_filter` | Boolean | Whether to automatically filter training samples exceeding `max_prompt_length`. |


## 4. Convert Model to Hugging Face Format
### 4.1 File Copy
Copy all files from the `Yuan3.0-Ultra/rlhf/verl/tests/convert/` directory to the trained model path: `actor/huggingface`.

### 4.2 Script Modification and Execution
Modify the following three parameters in the `Yuan3.0-Ultra/rlhf/verl/tools/merge_1020B.sh` script and execute it:
```bash
--local_dir   # Path to the trained model
--target_dir  # Output path
--vit_dir     # Path to the ViT model
```

```bash
cd Yuan3.0-Ultra/rlhf/verl
bash tools/merge_1020B.sh
```

