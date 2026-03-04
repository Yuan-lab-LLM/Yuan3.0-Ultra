
<div align="center">

![seed logo](https://avatars.githubusercontent.com/u/148021798?s=200&v=4)


<h1>Yuan3.0 Ultra: A Multimodal Large Language Model</h1>

</div>


## 目录

- [Yuan3.0 Ultra](#yuan30-ultra)
    - [环境配置](#环境配置)
    - [数据处理](#数据处理)
    - [模型微调](#模型微调)
    - [强化学习](#强化学习)
    - [推理服务](#推理服务)
    - [联系我们](#联系我们)

# Yuan3.0 Ultra

对本仓库源码的使用遵循开源许可协议 **Apache 2.0**。

Yuan3.0 Ultra模型支持商用，不需要申请授权，请您了解并遵循[《Yuan3.0 Ultra模型许可协议》](./LICENSE-Yuan)，勿将开源模型和代码及基于开源项目产生的衍生物用于任何可能给国家和社会带来危害的用途以及用于任何未经过安全评估和备案的服务。

尽管模型在训练时我们已采取措施尽力确保数据的合规性和准确性，但模型参数量巨大且受概率随机性因素影响，我们无法保证输出内容的准确性，且模型易被输入指令所误导，本项目不承担开源模型和代码导致的数据安全、舆情风险或发生任何模型被误导、滥用、传播、不当利用而产生的风险和责任。**您将对通过使用、复制、分发和修改模型等方式利用该开源项目所产生的风险与后果，独自承担全部责任。**

## 快速启动

### 环境配置

建议使用项目提供的image: yuanlabai/rlhf_yuan:v1.0，并可以通过下面命令启动容器：

```bash
docker pull yuanlabai/rlhf_yuan:v1.0

docker run --gpus all -itd --network=host -v /path/to/yuan_3.0:/workspace/yuan_3.0 -v /path/to/dataset:/workspace/dataset -v /path/to/checkpoints:/workspace/checkpoints --cap-add=IPC_LOCK --device=/dev/infiniband --privileged --name your_name --ulimit core=0 --ulimit memlock=-1 --ulimit stack=68719476736 --shm-size=1000G yuanlabai/rlhf_yuan:v1.0

docker exec -it your_name bash
```


### 数据处理

项目提供了数据处理的脚本，参考[源3.0数据处理说明文档](./docs/data_process_zh.md).

### 模型微调

微调流程请参考[源3.0指令微调说明文档](./docs/instruct_tuning_zh.md)。

### 强化学习
强化学习流程请参考[源3.0强化学习训练说明文档](./docs/RL_training_zh.md)。


### 推理服务
推理部署请参考[源3.0推理服务部署说明文档](../vllm/README_Yuan_zh.md)

### 联系我们

联系方式：service@yuanlab.ai

