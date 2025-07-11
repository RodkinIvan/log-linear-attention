export WANDB_PROJECT="hattention"
export TORCH_NCCL_HEARTBEAT_TIMEOUT_SEC=1800
export TORCH_NCCL_ENABLE_MONITORING=1

# Add additional NCCL environment variables to fix timeout issues
# export NCCL_DEBUG=INFO
# export NCCL_DEBUG_SUBSYS=ALL
export NCCL_TIMEOUT=1800
export NCCL_IB_DISABLE=1
export NCCL_P2P_DISABLE=1
export NCCL_SOCKET_IFNAME=lo
export NCCL_BLOCKING_WAIT=1
export NCCL_ASYNC_ERROR_HANDLING=1

# DATASET=pg19
# DATASET=iohadrubin/wikitext-103-raw-v1
DATASET=HuggingFaceFW/fineweb-edu
AC_MODE="none"
TBS=32
BS=8
NP=2
GRAD_ACC_STEPS=$((TBS / BS / NP))
LENGTH=2048

# Parse command line arguments
while [[ $# -gt 0 ]]; do
  case $1 in
    --name)
      NAME="$2"
      shift 2
      ;;
    --config)
      CONFIG="$2"
      shift 2
      ;;
    --seed)
      SEED_CKPT=1
      shift
      ;;
    --ac)
      AC_MODE="selective"
      shift
      ;;
    *)
      echo "Unknown argument: $1"
      exit 1
      ;;
  esac
done

echo "Arguments:"
echo "NAME: ${NAME}"
echo "CONFIG: ${CONFIG}"
echo "SEED_CKPT: ${SEED_CKPT}"
echo "AC_MODE: ${AC_MODE}"

if [[ -z "${NAME}" ]] || [[ -z "${CONFIG}" ]]; then
  echo "Usage: $0 --name NAME --config CONFIG [--seed] [--ac]"
  exit 1
fi

if [[ -n "${SEED_CKPT}" ]]; then

echo "Creating seed checkpoint..."

NNODE=1 NGPU=1 LOG_RANK=0 bash train.sh \
  --job.config_file flame/models/fla.toml \
  --job.dump_folder exp/${NAME} \
  --model.config /mnt/data/users/ivan.rodkin/lab/log-linear-attention/configs/flame/${CONFIG}.json \
  --model.tokenizer_path fla-hub/transformer-1.3B-100B \
  --training.batch_size $BS \
  --training.seq_len $LENGTH \
  --training.context_len $LENGTH \
  --training.gradient_accumulation_steps $GRAD_ACC_STEPS \
  --training.steps 95368 \
  --training.dataset $DATASET \
  --training.dataset_split train \
  --training.streaming \
  --activation_checkpoint.mode ${AC_MODE} \
  --activation_checkpoint.selective_ac_option 4 \
  --checkpoint.load_step -1 \
  --checkpoint.keep_latest_k 25 \
  --checkpoint.create_seed_checkpoint

fi

echo "Training..."

NNODE=1 NGPU=$NP LOG_RANK=0 bash train.sh \
  --job.config_file flame/models/fla.toml \
  --job.dump_folder exp/${NAME} \
  --model.config /mnt/data/users/ivan.rodkin/lab/log-linear-attention/configs/flame/${CONFIG}.json \
  --model.tokenizer_path fla-hub/transformer-1.3B-100B \
  --training.batch_size $BS \
  --training.seq_len $LENGTH \
  --training.context_len $LENGTH \
  --training.gradient_accumulation_steps $GRAD_ACC_STEPS \
  --training.steps 95368 \
  --training.dataset $DATASET \
  --training.dataset_split train \
  --training.streaming \
  --activation_checkpoint.mode ${AC_MODE} \
  --activation_checkpoint.selective_ac_option 4 \
  --checkpoint.load_step -1 \
  --checkpoint.keep_latest_k 25
