#!/bin/bash

DATE=`date '+%Y-%m-%d'`

# CHECKPOINT_PATH=/workspace/workspace/yanchao/yanc/bloom-1b1-optimizer-states
# use in docker, need -v when launch docker
CHECKPOINT_PATH=/workspace/bloomz-mt-176
SAVE_CHECKPOINT_PATH=/workspace/output_dir/$DATE/bloomz-mt-176-pretrain
DATA_PATH=data/alpaca_belle_cot_suffle/alpaca_belle_cot_suffle_text_document
TENSORBOARD_PATH=/workspace/output_dir/tensorboard/$DATE-bloomz-mt-176-pretrain

KILL_SWITCH_PATH=/workspace/output_dir/kill-switch-tr11-176B-pretrain-exp1

mkdir -p $SAVE_CHECKPOINT_PATH
mkdir -p $TENSORBOARD_PATH

N_GPUS=8
# global-batch-size should divisible by micro-batch-size times data-parallel-size

MICRO_BATCH_SIZE=4
GLOBAL_BATCH_SIZE=256
# NGPU = TP_SIZE*PP_SIZE*DP_SIZE
TP_SIZE=8
PP_SIZE=8

NLAYERS=70
NHIDDEN=14336
NHEADS=112
#SEQ_LEN=512
SEQ_LEN=1024

# llama config in huggingface
# "pad_token_id": -1, "rms_norm_eps": 1e-06,
# "bos_token_id": 0, "eos_token_id": 1, "hidden_act": "silu"
# "vocab_size": 32000


# 1 epoch need iter: 5287827/(2048*16)=161.37 iter
SAVE_INTERVAL=2200

# alpaca tokens: 5287827, 10 epoch samples = 5287827/2048*10=25,819
TRAIN_ITERS=13_200

LR_DECAY_ITERS=12_000
LR_WARMUP_ITERS=400

OPTIMIZER_ARGS=" \
    --optimizer adam \
    --adam-beta1 0.9 \
    --adam-beta2 0.99 \
    --adam-eps 1e-8 \
    --lr 2e-5 \
    --min-lr 1e-6 \
    --clip-grad 1.0 \
    --lr-decay-style cosine \
    --lr-decay-iters $LR_DECAY_ITERS \
    --weight-decay 0. \
    "

# no rampup-batch-size
GPT_ARGS=" \
    --pp-partition-method type:transformer|embedding \
    --num-layers $NLAYERS \
    --hidden-size $NHIDDEN \
    --num-attention-heads $NHEADS \
    --seq-length $SEQ_LEN \
    --max-position-embeddings $SEQ_LEN \
    --micro-batch-size $MICRO_BATCH_SIZE \
    --global-batch-size $GLOBAL_BATCH_SIZE \
    --tokenizer-type PretrainedFromHF \
    --tokenizer-name-or-path /workspace/yanc/tokenizer \
    --init-method-std 0.0048 \
    --embed-layernorm \
    --pad-id 3 \
    --bf16 \
    --loss-scale 12 \
    --clip-grad 1.0 \
    --sync-tp-duplicated-parameters \
    --seed 42 \
    --position-embedding-type alibi \
    --checkpoint-activations \
    --pad-vocab-size-to 250880 \
    --abort-on-unmet-fused-kernel-constraints \
    $OPTIMIZER_ARGS \
    "

OUTPUT_ARGS=" \
    --exit-interval 30000 \
    --log-interval 1 \
    --save-interval $SAVE_INTERVAL \
    --eval-interval 600 \
    --eval-iters 1 \
    --train-iters=$TRAIN_ITERS
    --lr-warmup-iters=$LR_WARMUP_ITERS
    --tensorboard-dir $TENSORBOARD_PATH \
    --tensorboard-queue-size 5 \
    --log-timers-to-tensorboard \
    --log-batch-size-to-tensorboard \
    --log-validation-ppl-to-tensorboard \
    "

ZERO_STAGE=0

config_json="./ds_config.json"

# Deepspeed figures out GAS dynamically from dynamic GBS via set_train_batch_size()
cat <<EOT > $config_json
{
  "train_micro_batch_size_per_gpu": $MICRO_BATCH_SIZE,
  "train_batch_size": $GLOBAL_BATCH_SIZE,
  "gradient_clipping": 1.0,
  "zero_optimization": {
    "stage": $ZERO_STAGE
  },
  "bf16": {
    "enabled": true
  },
  "steps_per_print": 2000,
  "wall_clock_breakdown": false
}
EOT

DEEPSPEED_ARGS=" \
    --deepspeed \
    --deepspeed_config ${config_json} \
    --zero-stage ${ZERO_STAGE} \
    --deepspeed-activation-checkpointing \
    "

CUSTOM_ARGS=" \
    --no-load-optim \
    --finetune \
    --override-lr-scheduler \
    "

export CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7

<<!
echo "Pass......................"

export LAUNCHER="python -u -m torch.distributed.launch \
    --nproc_per_node $N_GPUS \
    --nnodes 1 \
    --master_port 60400 \
    "
!

export LAUNCHER="deepspeed --hostfile=hostfile --no_ssh_check"
export CMD=" \
    $LAUNCHER pretrain_gpt.py \
    --tensor-model-parallel-size $TP_SIZE \
    --pipeline-model-parallel-size $PP_SIZE \
    $GPT_ARGS \
    $OUTPUT_ARGS \
    $CUSTOM_ARGS \
    --save $SAVE_CHECKPOINT_PATH \
    --load $CHECKPOINT_PATH \
    --data-path $DATA_PATH \
    --data-impl mmap \
    --split 949,50,1 \
    --kill-switch-path /tmp/kill-switch \
    --distributed-backend nccl \
    $DEEPSPEED_ARGS \
    "

# do not remove or the training will hang and nodes will be lost w/o this workaround
export CUDA_LAUNCH_BLOCKING=1

# force crashing on nccl issues like hanging broadcast
export NCCL_ASYNC_ERROR_HANDLING=1


# hide duplicated errors using this hack - will be properly fixed in pt-1.12
export TORCHELASTIC_ERROR_FILE=/tmp/torch-elastic-error.json

export NCCL_DEBUG=INFO

# yanc not work
#export NCCL_P2P_LEVEL=NVL
#export NCCL_P2P_DISABLE=1
#export NCCL_IB_DISABLE=1

#export NCCL_SOCKET_IFNAME=eth1,eth2,eth3,eth4,eth5,eth6,eth7,eth8
export NCCL_IB_HCA=mlx5_1,mlx5_2,mlx5_3,mlx5_4,mlx5_5,mlx5_6,mlx5_7,mlx5_8
export NCCL_IB_TIMEOUT=23
export NCCL_IB_GID_INDEX=3   # 走roce v2 协议
#export NCCL_IB_TC=128


echo $CMD

HOUR=`date '+%H-%M-%S'`

$CMD  2>&1 |tee pretrain_gpt.bloomz-176b.$DATE.$HOUR.log
