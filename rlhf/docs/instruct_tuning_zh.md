# Yuan3.0 Ultra 监督微调

## 介绍

本文档提供了对Yuan3.0 Ultra进行监督微调（SFT）的指导说明。

## 使用方法

运行Yuan3.0 Ultra 1020B模型SFT的示例脚本如下：

```shell
cd Yuan3.0-Ultra/rlhf/megatron-lm
bash examples/pretrain_yuan3.0_1020B_sft.sh
```

### 参数设置

运行脚本前，应正确设置相关参数，包括设置环境变量 `CASE_CHECKPOINT_PATH`、`DATA_PATH`、`TOKENIZER_MODEL`、`CLIP_DOWNLOAD_PATH` 和 `CHECKPOINT_PATH_LOAD`。


如果数据集路径是：

```
/path/dataset.bin
```

`DATA_PATH` 可设置为：

```shell
DATA_PATH='<token_number> /path/dataset'
```


更多命令行参数的描述请见[README.md](https://github.com/NVIDIA/Megatron-LM/blob/main/README.md)。
