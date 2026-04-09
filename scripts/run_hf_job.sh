#!/usr/bin/env bash
set -euo pipefail

if [[ -z "${HF_MODEL_REPO:-}" ]]; then
  echo "HF_MODEL_REPO is required (set as a GitHub Actions secret)." >&2
  exit 1
fi

HF_BASE_MODEL="${HF_BASE_MODEL:-google/flan-t5-base}"
HF_JOB_FLAVOR="${HF_JOB_FLAVOR:-a10g-large}"
HF_JOB_TIMEOUT="${HF_JOB_TIMEOUT:-4h}"
HF_JOB_IMAGE="${HF_JOB_IMAGE:-python:3.12}"
TRAIN_SCRIPT="${HF_TRAIN_SCRIPT:-backend/training/hf_finetune.py}"
REQS_PATH="${HF_REQS_PATH:-backend/requirements.txt}"

if [[ -z "${GITHUB_REPOSITORY:-}" || -z "${GITHUB_SHA:-}" ]]; then
  echo "GITHUB_REPOSITORY and GITHUB_SHA must be set by GitHub Actions." >&2
  exit 1
fi

REPO_URL="https://github.com/${GITHUB_REPOSITORY}.git"

hf jobs run \
  --flavor "${HF_JOB_FLAVOR}" \
  --timeout "${HF_JOB_TIMEOUT}" \
  --secrets HF_TOKEN \
  "${HF_JOB_IMAGE}" \
  bash -lc "apt-get update && apt-get install -y git && \
    git clone ${REPO_URL} repo && \
    cd repo && git checkout ${GITHUB_SHA} && \
    pip install -r ${REQS_PATH} && \
    python ${TRAIN_SCRIPT} \
      --model-name ${HF_BASE_MODEL} \
      --push-to-hub \
      --hub-repo ${HF_MODEL_REPO}"
