##  1.镜像方式（建议）

我们强烈建议使用 Yuan3.0 Ultra 的最新版本 docker 镜像。
```bash
docker pull yuanlabai/vllm:v0.11.0
```

##  2.源码编译（可选）

从源码安装vllm
```
git clone https://github.com/Yuan-lab-LLM/Yuan3.0-Ultra.git
cd Yuan3.0-Ultra/vllm
pip install -e .
```

##  3. 快速开始

**3.1  环境配置**

您可以使用以下 Docker 命令启动 Yuan3.0 Ultra 容器实例：
```bash
docker run --gpus all -itd --network=host --privileged --cap-add=IPC_LOCK --ulimit stack=68719476736 --shm-size=1000G -v /path/to/dataset:/workspace/dataset -v /path/to/checkpoints:/workspace/checkpoints --name your_name yuanlabai/vllm:v0.11.0
docker exec -it your_name bash
```

**3.2  部署服务**

Yuan3.0 Ultra Model 仅支持 vLLm V1架构。  
我们建议使用2个节点，张量和流水并行组合的方式部署。  
多节点ray服务启动命令请参考[多节点服务](./examples/online_serving/multi-node-serving.sh)教程。
```bash
python -m vllm.entrypoints.openai.api_server --model=/path/Yuan3.0-Ultra-int4 --port 8100 --gpu-memory-utilization 0.9 \
 --tensor-parallel-size 4 --pipeline-parallel-size 4 --trust-remote-code --allowed-local-media-path "/path/images"
```
> **Note 1**:如果您有复杂的网络配置，可能需要配置[网络设置](./docs/usage/troubleshooting.md:#L10)。   
> **Note 2**:对于int4模型，我们使用2个节点(16\*A800)进行部署服务，并行方式设置4路张量并行和4路流水线并行。    
> **Note 3**:对于bfloat16模型，我们使用6个节点(48\*A800)进行部署服务，并行方式设置4路张量并行和12路流水线并行。   


**3.3  请求调用**

```python
from openai import OpenAI

openai_api_key = "EMPTY"
openai_api_base = "http://localhost:8100/v1"

client = OpenAI(
    api_key=openai_api_key,
    base_url=openai_api_base,
)

prompt = '请描述这张图片的内容'
image_path = 'Yuan3.0-Ultra/vllm/docs/images/image.jpg'
image_url = f"file:{image_path}"

response = client.chat.completions.create(
    model="/path/Yuan3.0-Ultra-int4",
    messages=[{
        "role": "user",
        "content": [
            {"type": "text", "text": f"{prompt}"},
            {"type": "image_url", "image_url": {"url": image_url}},
        ],
    }
    ],
    max_tokens=256,
    temperature=1e-6,
)
print("Chat completion output:", response.choices[0].message.content)
```
