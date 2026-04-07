##  1. Docker image


We strongly recommend using the latest release of docker images of Yuan3.0 Ultra. You can launch an instance of the Yuan 3.0 Ultra container with the following Docker commands:

```bash
docker pull yuanlabai/vllm:v0.17.0
```


##  2. Install (optional)

Install vLLM for Yuan3.0 Ultra model from source:
```
git clone https://github.com/Yuan-lab-LLM/Yuan3.0-Ultra.git
cd Yuan3.0-Ultra/vllm
pip install -e .
```


##  3. Quick Start


**3.1  Environment Config**

You can launch an instance of the Yuan 3.0 Ultra container with the following Docker commands:

```bash
docker run --gpus all -itd --network=host --privileged --cap-add=IPC_LOCK --ulimit stack=68719476736 --shm-size=1000G -v /path/to/dataset:/workspace/dataset -v /path/to/checkpoints:/workspace/checkpoints --name your_name yuanlabai/vllm:v0.17.0
docker exec -it your_name bash
```

**3.2  Deployment service**

Yuan3.0 Ultra Model just support vLLm V1. <br>
Please refer to the tutorial [multi-node-serving](./examples/online_serving/multi-node-serving.sh) for starting the ray service. <br>
There are [bugs](https://github.com/vllm-project/vllm/issues/35848) in multi-node deployments using ray as the backend. <br>
Please use RAY_V2 as the backend. Set 'export VLLM_USE_RAY_V2_EXECUTOR_BACKEND=1' <br>
```bash
python -m vllm.entrypoints.openai.api_server --model=/path/Yuan3.0-Ultra-int4 --port 8100 --gpu-memory-utilization 0.9 \
 --tensor-parallel-size 4 --pipeline-parallel-size 4 --distributed-executor-backend ray \
 --trust-remote-code --allowed-local-media-path "/path/images"
```
> **Note 1**: You might also need to setup [network setup](./docs/usage/troubleshooting.md#L76).   
> **Note 2**: For the int4 model, the parallel configuration of tensor-parallel-size=4 and pipeline-parallel-size=4 is suggested.   
> **Note 3**: For the bfloat16 model,  the parallel configuration of tensor-parallel-size=4 and pipeline-parallel-size=12 is suggested. 

**3.3  Client request**

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
