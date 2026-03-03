NCCL_DEBUG=INFO torchrun --nproc_per_node 8  -m verl.model_merger merge \
    --backend megatron \
    --trust-remote-code \
    --local_dir <path-to-todo-convert-ckpt>/actor \
    --target_dir <path-to-converted-ckpt> \
    --vit_dir <path-to-vit-model> \
    --load_mode 'split' \
    --use_cpu_initialization
