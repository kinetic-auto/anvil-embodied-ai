#!/usr/bin/env bash
# Bring a Lambda on-demand box to a runnable anvil-train container.
# Does not start anvil-trainer. Does not reconvert. Does not touch the
# laptop's running train container.
#
# From the laptop, after the instance is up:
#   ./scripts/setup_lambda_train.sh --ip <INSTANCE_IP> --ssh-key ~/.ssh/<lambda-key>
#
# Options:
#   --ip IP            Lambda public IP (required unless --on-lambda)
#   --ssh-key PATH     SSH private key (optional if the agent already has it)
#   --user NAME        SSH user (default: ubuntu)
#   --skip-dataset     Converted LeRobot set is already at the remote path
#   --skip-recordings  Raw MCAPs are already at the remote path
#   --on-lambda        Internal: run the on-box half (clone, docker, compose)
#   -h, --help
#
# Environment:
#   GIT_URL        default git@github.com:saichand44/anvil-embodied-ai.git
#   GIT_REF        default main
#   SESSION_NAME   dataset folder under datasets/ (default: car-door-opening-20260910)
#
# Hardcoded Lambda paths match docker-compose.train.yml.

set -euo pipefail

REMOTE_REPO="/home/sai/github_repos/anvil-embodied-ai"
HOST_DATA_ROOT="/var/tmp/anvil-robot/data"
SESSION_NAME="${SESSION_NAME:-car-door-opening-20260910}"
DATASET_PATH="${HOST_DATA_ROOT}/datasets/${SESSION_NAME}"
RECORDINGS_PATH="${HOST_DATA_ROOT}/recordings/${SESSION_NAME}"
GIT_URL="${GIT_URL:-git@github.com:saichand44/anvil-embodied-ai.git}"
GIT_REF="${GIT_REF:-main}"
SSH_USER="ubuntu"
INSTANCE_IP=""
SSH_KEY=""
SKIP_DATASET=false
SKIP_RECORDINGS=false
ON_LAMBDA=false

usage() {
    sed -n '2,/^set -/{ /^set -/d; s/^# \?//; p }' "$0"
}

die() {
    echo "error: $*" >&2
    exit 1
}

log() {
    echo "[setup_lambda_train] $*"
}

while [[ $# -gt 0 ]]; do
    case "$1" in
        --ip)
            INSTANCE_IP="${2:-}"
            shift 2
            ;;
        --ssh-key)
            SSH_KEY="${2:-}"
            shift 2
            ;;
        --user)
            SSH_USER="${2:-}"
            shift 2
            ;;
        --skip-dataset)
            SKIP_DATASET=true
            shift
            ;;
        --skip-recordings)
            SKIP_RECORDINGS=true
            shift
            ;;
        --on-lambda)
            ON_LAMBDA=true
            shift
            ;;
        -h|--help)
            usage
            exit 0
            ;;
        *)
            die "unknown argument: $1"
            ;;
    esac
done

ssh_cmd() {
    local args=(-A -o ServerAliveInterval=30 -o ServerAliveCountMax=10)
    [[ -n "$SSH_KEY" ]] && args+=(-i "$SSH_KEY")
    ssh "${args[@]}" "${SSH_USER}@${INSTANCE_IP}" "$@"
}

rsync_cmd() {
    local args=(-aH --info=progress2)
    local ssh_args=(-A -o ServerAliveInterval=30)
    [[ -n "$SSH_KEY" ]] && ssh_args+=(-i "$SSH_KEY")
    rsync "${args[@]}" -e "ssh ${ssh_args[*]}" "$@"
}

print_train_hint() {
    cat <<'EOF'

Container is up. Training is NOT started.

  ssh -A -i <key> ubuntu@<INSTANCE_IP>
  tmux attach -t train
  docker exec -it anvil-train bash

Then, inside the container:

  cd /workspace
  anvil-trainer \
    --dataset.root=data/datasets/car-door-opening-20260910 \
    --policy.type=act \
    --job_name=act-21dim-chunk100-aug-lambda \
    --dataset.image_transforms.enable=true \
    --dataset.image_transforms.max_num_transforms=3 \
    --policy.chunk_size=100 \
    --policy.n_action_steps=100 \
    --policy.normalization_mapping='{"ACTION":"MEAN_STD","STATE":"MEAN_STD","VISUAL":"IDENTITY"}' \
    --policy.kl_weight=10.0 \
    --backbone=resnet18 \
    --action-type=absolute \
    --batch_size=16 \
    --steps=100000 \
    --save_freq=10000 \
    --log_freq=200 \
    --split-ratio=8,1,1 \
    --wandb.enable=false

Use --batch_size=32 on A6000 48GB or A100 40GB. Absolute actions only.
EOF
}

on_lambda() {
    local uid gid
    uid="$(id -u)"
    gid="$(id -g)"

    log "nvidia-smi"
    nvidia-smi

    log "host dirs"
    sudo mkdir -p /home/sai/github_repos "${HOST_DATA_ROOT}/datasets" "${HOST_DATA_ROOT}/recordings"
    sudo chown -R "$(id -un):$(id -gn)" /home/sai /var/tmp/anvil-robot
    mkdir -p "${DATASET_PATH}" "${RECORDINGS_PATH}"

    if ! groups | grep -qw docker; then
        log "adding $(id -un) to docker group"
        sudo adduser "$(id -un)" docker
    fi
    if ! command -v docker >/dev/null; then
        die "docker is not installed (use the default Lambda Stack image)"
    fi
    if ! sg docker -c 'docker compose version' >/dev/null 2>&1; then
        log "installing docker compose plugin"
        sudo apt-get update -qq
        sudo apt-get install -y docker-compose-plugin
    fi

    if [[ ! -f "${DATASET_PATH}/meta/info.json" ]]; then
        die "dataset missing at ${DATASET_PATH} (rsync it from the laptop first)"
    fi

    mkdir -p ~/.ssh
    chmod 700 ~/.ssh
    if ! grep -q 'github.com' ~/.ssh/known_hosts 2>/dev/null; then
        ssh-keyscan -t ed25519,rsa github.com >> ~/.ssh/known_hosts 2>/dev/null || true
    fi

    if [[ -f "${REMOTE_REPO}/docker-compose.train.yml" ]]; then
        log "repo present at ${REMOTE_REPO}"
        if [[ -d "${REMOTE_REPO}/.git" ]]; then
            git -C "${REMOTE_REPO}" fetch origin && \
                git -C "${REMOTE_REPO}" checkout "${GIT_REF}" && \
                git -C "${REMOTE_REPO}" pull --ff-only origin "${GIT_REF}" || \
                log "git fetch/pull skipped (using files on disk)"
        fi
    else
        log "cloning ${GIT_URL} (${GIT_REF}) -> ${REMOTE_REPO}"
        mkdir -p "$(dirname "${REMOTE_REPO}")"
        git clone --branch "${GIT_REF}" "${GIT_URL}" "${REMOTE_REPO}"
    fi

    log "HEAD $(git -C "${REMOTE_REPO}" rev-parse --short HEAD 2>/dev/null || echo unknown)"
    cd "${REMOTE_REPO}"
    mkdir -p model_zoo eval_results .train_home .hf_cache
    # Container runs as HOST_UID:HOST_GID; these dirs must be writable there.
    # A prior root-owned docker run leaves .hf_cache root:root and dataset-valid fails.
    if ! [[ -O .hf_cache && -O .train_home && -O model_zoo && -O eval_results ]]; then
        log "fixing ownership on writable repo dirs (uid=${uid} gid=${gid})"
        sudo chown -R "${uid}:${gid}" model_zoo eval_results .train_home .hf_cache
    fi

    tmux has-session -t train 2>/dev/null || tmux new-session -d -s train

    log "building anvil-robot:latest (no LEROBOT_EXTRAS)"
    sg docker -c 'docker build -f docker/inference/Dockerfile -t anvil-robot:latest .'

    log "compose up anvil-train"
    sg docker -c "export HOST_UID=${uid} HOST_GID=${gid}; docker compose -f docker-compose.train.yml up -d --build"

    log "dataset-valid"
    sg docker -c "docker exec anvil-train dataset-valid --root data/datasets/${SESSION_NAME}"

    log "cuda in container"
    sg docker -c 'docker exec anvil-train python3 -c "import torch; print(\"cuda\", torch.cuda.is_available(), torch.cuda.get_device_name(0) if torch.cuda.is_available() else \"\")"'

    print_train_hint
}

orchestrate_from_laptop() {
    local script_path
    script_path="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)/$(basename "${BASH_SOURCE[0]}")"
    local local_repo
    local_repo="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"

    [[ -n "$INSTANCE_IP" ]] || die "--ip is required"
    [[ -f "$script_path" ]] || die "cannot find $script_path"
    if [[ -n "$SSH_KEY" && ! -f "$SSH_KEY" ]]; then
        die "ssh key not found: $SSH_KEY"
    fi
    if [[ "$SKIP_DATASET" == false && ! -f "${DATASET_PATH}/meta/info.json" ]]; then
        die "local dataset missing: ${DATASET_PATH}/meta/info.json"
    fi
    if [[ "$SKIP_RECORDINGS" == false && ! -d "${RECORDINGS_PATH}" ]]; then
        die "local recordings missing: ${RECORDINGS_PATH}"
    fi

    log "ssh ${SSH_USER}@${INSTANCE_IP}"
    ssh_cmd 'echo connected; nvidia-smi -L'

    log "creating remote dirs"
    ssh_cmd "sudo mkdir -p /home/sai/github_repos ${DATASET_PATH} ${RECORDINGS_PATH} && sudo chown -R ${SSH_USER}:${SSH_USER} /home/sai /var/tmp/anvil-robot"

    if [[ "$SKIP_DATASET" == false ]]; then
        log "rsync dataset -> ${DATASET_PATH}"
        rsync_cmd \
            "${DATASET_PATH}/" \
            "${SSH_USER}@${INSTANCE_IP}:${DATASET_PATH}/"
    else
        log "skipping dataset rsync"
    fi

    if [[ "$SKIP_RECORDINGS" == false ]]; then
        log "rsync recordings -> ${RECORDINGS_PATH}"
        rsync_cmd \
            "${RECORDINGS_PATH}/" \
            "${SSH_USER}@${INSTANCE_IP}:${RECORDINGS_PATH}/"
    else
        log "skipping recordings rsync"
    fi

    log "on-box setup (clone, docker, compose)"
    local remote_ok=0
    ssh_cmd "GIT_URL=$(printf %q "$GIT_URL") GIT_REF=$(printf %q "$GIT_REF") SESSION_NAME=$(printf %q "$SESSION_NAME") bash -s -- --on-lambda" \
        < "$script_path" || remote_ok=$?

    if [[ "$remote_ok" -ne 0 ]]; then
        log "git/docker setup failed (exit ${remote_ok}); rsyncing repo from laptop as fallback"
        rsync_cmd \
            --exclude 'model_zoo/' \
            --exclude 'eval_results/' \
            --exclude '.hf_cache/' \
            --exclude '.train_home/' \
            --exclude 'monitor_output/' \
            "${local_repo}/" \
            "${SSH_USER}@${INSTANCE_IP}:${REMOTE_REPO}/"
        ssh_cmd "GIT_URL=$(printf %q "$GIT_URL") GIT_REF=$(printf %q "$GIT_REF") SESSION_NAME=$(printf %q "$SESSION_NAME") bash -s -- --on-lambda" \
            < "$script_path"
    fi
}

if [[ "$ON_LAMBDA" == true ]]; then
    on_lambda
else
    orchestrate_from_laptop
fi
