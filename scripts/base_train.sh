
#!/bin/bash

# Training script template
# Set the following environment variables as needed:
# - WANDB_API_KEY: Your WandB API key (optional, for experiment tracking)
# - HF_HOME: HuggingFace cache directory (default: ~/.cache/huggingface)
# - HF_ENDPOINT: HuggingFace endpoint (e.g., https://hf-mirror.com for China)

FILE=$1
CONFIG_FILE=$2
TAG=$3
shift 3  # remove $1~$3 for $@

# Optional: Configure WandB if using experiment tracking
# export WANDB_API_KEY=your_wandb_api_key_here
# export WANDB_ENTITY=your_entity

# Optional: HuggingFace cache and mirror settings
# export HF_HOME=~/.cache/huggingface
# export TORCH_HOME=~/.cache/torch
# export HF_ENDPOINT=https://hf-mirror.com

export NCCL_DEBUG=VERSION
export HDF5_USE_FILE_LOCKING=FALSE
export ACCELERATE_LOG_LEVEL=info
export HYDRA_FULL_ERROR=1

python3 \
    ${FILE} \
        --config_file ${CONFIG_FILE} \
        --tag ${TAG} \
        --pin_memory \
        --allow_tf32 \
$@
