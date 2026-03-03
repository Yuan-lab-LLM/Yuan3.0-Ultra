set -x

# Example runnable on H20 * 8

export CUDA_DEVICE_MAX_CONNECTIONS=1 # For megatron communication/computation overlapping

gsm8k_train_path=$HOME/data/gsm8k/train.parquet
gsm8k_test_path=$HOME/data/gsm8k/test.parquet
math_train_path=$HOME/data/math/train.parquet
math_test_path=$HOME/data/math/test.parquet

train_files="['$math_train_path', '$gsm8k_train_path']"
test_files="['$gsm8k_test_path']"

export VERL_USE_MODELSCOPE=True

ray job submit --address="http://localhost:8265" \
--working-dir="./" \
--runtime-env=verl/trainer/runtime_env.yaml \
-- \
python3 -m recipe.dapo.main_dapo --config-path=./config --config-name='ppo_megatron_trainer'\
    algorithm.adv_estimator=gae \
    data.train_files="$train_files" \
    data.val_files="$test_files" \
    data.train_batch_size=128 \
    data.max_prompt_length=1024 \
    data.max_response_length=128 \
    data.filter_overlong_prompts=False \
    data.truncation='error' \
    actor_rollout_ref.model.path=$MODEL_PATH \
    actor_rollout_ref.actor.optim.lr=1e-6 \
    actor_rollout_ref.actor.ppo_mini_batch_size=128 \
    actor_rollout_ref.actor.ppo_micro_batch_size_per_gpu=4 \
    actor_rollout_ref.actor.megatron.pipeline_model_parallel_size=2 \
    actor_rollout_ref.actor.megatron.tensor_model_parallel_size=2 \
    actor_rollout_ref.actor.use_kl_loss=False \
    actor_rollout_ref.rollout.log_prob_micro_batch_size_per_gpu=4 \
    actor_rollout_ref.rollout.tensor_model_parallel_size=4 \
    actor_rollout_ref.rollout.name=vllm \
    actor_rollout_ref.rollout.gpu_memory_utilization=0.4 \
    actor_rollout_ref.ref.megatron.pipeline_model_parallel_size=2 \
    actor_rollout_ref.ref.megatron.tensor_model_parallel_size=2 \
    actor_rollout_ref.actor.megatron.use_dist_checkpointing=False \
    actor_rollout_ref.ref.megatron.use_dist_checkpointing=False \
    critic.optim.lr=1e-5 \
    critic.model.path=$MODEL_PATH \
    critic.model.trust_remote_code=True \
    critic.ppo_micro_batch_size_per_gpu=4 \
    algorithm.use_kl_in_reward=False \
    trainer.critic_warmup=0 \
    trainer.logger='["console"]' \
    trainer.project_name='verl_ppo_gsm8k_math_examples' \
    trainer.experiment_name='deepseek_llm_7b_megatron' \
    trainer.n_gpus_per_node=8 \
    trainer.nnodes=1 \
    trainer.save_freq=20 \
    trainer.test_freq=5 \
    trainer.total_epochs=100 $@
