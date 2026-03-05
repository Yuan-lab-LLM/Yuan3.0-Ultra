
## Table of Contents

- [Yuan3.0 Ultra](#yuan30-ultra)
    - [Environment Config](#environment-config)
    - [Data Processing](#data-preprocess)
    - [Instruct Tuning](#model-instruct-tuning)
    - [Reinforcment Learning](#reinforcment-learning)
    - [Inference Deployment](#inference-deployment)
    - [Contact Us](#contact-us)

# Yuan3.0 Ultra

The use of the source code in this repository requires compliance with the open source license agreement **Apache 2.0**.
The Yuan3.0 Ultra model supports commercial use and does not require authorization. Please understand and comply with the [《Yuan3.0 Ultra Model License Agreement》](./LICENSE-Yuan). Do not use the open source model and code, as well as derivatives generated from open source projects, for any purposes that may cause harm to the country and society, or for any services that have not undergone security assessment and filing.
Although we have taken measures to ensure the compliance and accuracy of the data during training, the model has a huge number of parameters and is affected by probability and randomness factors. We cannot guarantee the accuracy of the output content, and the model is easily misled by input instructions. This project does not assume any data security, public opinion risks, or any model misleading, abusing, spreading caused by open-source models and code Risks and responsibilities arising from improper utilization  **You will be solely responsible for the risks and consequences arising from the use, copying, distribution, and modification of the model in this open source project**

### Environment Config

We recommend using the latest pre-built docker image provided by us and starting the container with the following command:
```bash
docker pull yuanlabai/rlhf_yuan:v1.0

docker run --gpus all -itd --network=host -v /path/to/yuan_3.0:/workspace/yuan_3.0 -v /path/to/dataset:/workspace/dataset -v /path/to/checkpoints:/workspace/checkpoints --cap-add=IPC_LOCK --device=/dev/infiniband --privileged --name your_name --ulimit core=0 --ulimit memlock=-1 --ulimit stack=68719476736 --shm-size=1000G yuanlabai/rlhf_yuan:v1.0

docker exec -it your_name bash
```

### Data Processing

We have provided the data processing script. See documentation [here](./docs/data_process.md).

### Instruct Tuning

We have provided the supervised fine-tuning script. See documentation [here](./docs/instruct_tuning.md).

### Reinforcment Learning

We have provided the reinforcement learning script. See documentation [here](./docs/RL_training.md).


### Inference Deployment

We have provided the vllm deployment script. See documentation [here](../vllm/README_Yuan.md)

### Contact Us

Contact us: service@yuanlab.ai
