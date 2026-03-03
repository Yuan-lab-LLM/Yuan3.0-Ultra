
CUDA_VISIBLE_DEVICES=0,1 vllm serve $CKPTPATH --tensor-parallel-size 2 --trust-remote-code --gpu-memory-utilization 0.86 --enforce-eager --port=7230 --dtype bfloat16 --max-model-len 16384 &
CUDA_VISIBLE_DEVICES=2,3 vllm serve $CKPTPATH --tensor-parallel-size 2 --trust-remote-code --gpu-memory-utilization 0.86 --enforce-eager --port=7231 --dtype bfloat16 --max-model-len 16384 &
CUDA_VISIBLE_DEVICES=4,5 vllm serve $CKPTPATH --tensor-parallel-size 2 --trust-remote-code --gpu-memory-utilization 0.86 --enforce-eager --port=7232 --dtype bfloat16 --max-model-len 16384 &
CUDA_VISIBLE_DEVICES=6,7 vllm serve $CKPTPATH --tensor-parallel-size 2 --trust-remote-code --gpu-memory-utilization 0.86 --enforce-eager --port=7233 --dtype bfloat16 --max-model-len 16384

