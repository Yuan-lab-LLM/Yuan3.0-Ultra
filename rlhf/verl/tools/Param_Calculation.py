import os
import sys
import torch

# ================= 配置区域 =================
# 添加Yuan3.0模型完整路径,将如下路径替换成你的路径
MODEL_PATH = "/path/to/Yuan3.0-Model"

# 将模型目录设为 Python 搜索路径的第一优先级
if MODEL_PATH not in sys.path:
    sys.path.insert(0, MODEL_PATH)

# 设置环境变量，强制离线模式，禁止 HF 联网或访问远程缓存校验
os.environ["TRANSFORMERS_OFFLINE"] = "1"
os.environ["HF_DATASETS_OFFLINE"] = "1"
os.environ["HF_EVALUATE_OFFLINE"] = "1"

from transformers import AutoModel, AutoTokenizer, AutoConfig

print(f"🚀 开始从本地加载模型：{MODEL_PATH}")

# 加载模型
model = AutoModel.from_pretrained(
    MODEL_PATH,
    torch_dtype=torch.bfloat16,
    low_cpu_mem_usage=True,
    use_flash_attn=False,
    device_map="cpu",
    local_files_only=True,
    trust_remote_code=True,
)

print("\n" + "="*30)
print("--Yuan3.0 Model Parameter--")
print("="*30)

# 统计参数
vit_params = 0
yuan_params = 0
total_params = model.num_parameters()
for n, p in model.named_parameters():
    if 'vision_model' in n:
        vit_params += p.numel()
    else:
        yuan_params += p.numel()

print(f"Vit Model Parameters:     {vit_params:,}")
print(f"Yuan Model Parameters:     {yuan_params:,}")
print(f"Total Parameters:     {total_params:,}")
print("="*30)

