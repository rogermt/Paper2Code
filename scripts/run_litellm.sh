#!/bin/bash
set -e  # Fail on error

# -------- CONFIG --------
WORKING_DIR="/kaggle/working"
PAPER_NAME="NSGF"
GPT_VERSION="DeepSeek-R1-0528-Qwen3-8B-GGUF-Q8_K_XL_local"

WANDB_PROJECT_NAME = "nsgf-paper2code-playground"
WEAVE_PROJECT_NAME = "nsgf-paper2code-playground"

PDF_PATH="${WORKING_DIR}/${PAPER_NAME}.pdf"
PDF_JSON_PATH="${WORKING_DIR}/output/${PAPER_NAME}.json" 
PDF_JSON_CLEANED_PATH="${WORKING_DIR}/output/${PAPER_NAME}_cleaned.json"
OUTPUT_DIR="${WORKING_DIR}/output/${PAPER_NAME}"
OUTPUT_REPO_DIR="${WORKING_DIR}/output/${PAPER_NAME}_repo"

KAGGLE_INPUT_DIR="/kaggle/input" 
KAGGLE_RESUME_DATASET="paper2code-nsgf-paper-output"
KAGGLE_RESUME_DATASET_OUTPUT_DIR="${KAGGLE_INPUT_DIR}/${KAGGLE_RESUME_DATASET}/output/${PAPER_NAME}"

USE_KAGGLE_RESUMER="False"
WANDB_RUN_ID=None

RESUME_STAGE="ALL"  # Options: PLANNING, CONFIG, ANALYZING, CODING, ALL
PLANNING_STAGE_RESUME_INDEX="0"
ANALYZING_STAGE_RESUME_INDEX="0"
CODING_STAGE_RESUME_INDEX="0"


# -------- SETUP --------
mkdir -p "$OUTPUT_DIR"
mkdir -p "$OUTPUT_REPO_DIR"

echo "📄 PAPER: $PAPER_NAME"
echo "📁 OUTPUT DIR: $OUTPUT_DIR"
echo "📦 W&B RUN ID: $WANDB_RUN_ID"
echo "🔁 RESUME STAGE: $RESUME_STAGE"


# -------- PREPROCESS --------
echo "------- Preprocess -------"
python ../codes/0_pdf_process.py \
    --input_json_path "$PDF_JSON_PATH" \
    --output_json_path "$PDF_JSON_CLEANED_PATH" \
    --input_json_type standard


# -------- RESUME LOGIC --------
if [ "$USE_KAGGLE_RESUMER" == "True" ]; then
  RESUME_DIR="$KAGGLE_RESUME_DATASET_OUTPUT_DIR"
  use_kaggle_arg="--use_kaggle_resumer"
  kaggle_resume_dataset_arg="--kaggle_dataset_path $RESUME_DIR"
else
  RESUME_DIR="$OUTPUT_DIR"
  use_kaggle_arg=""
  kaggle_resume_dataset_arg=""
fi

if [ -n "$WANDB_RUN_ID" ]; then
  wandb_arg="--wandb_run_id $WANDB_RUN_ID"
else
  wandb_arg=""
fi

# -------- PLANNING --------
if [[ "$RESUME_STAGE" == "PLANNING" || "$RESUME_STAGE" == "ALL" ]]; then
  echo "------- Planning Stage -------"
  python ../codes/1_planning_litellm.py \
    --paper_name "$PAPER_NAME" \
    --gpt_version "$GPT_VERSION" \
    --pdf_json_path "$PDF_JSON_CLEANED_PATH" \
    --output_dir "$OUTPUT_DIR" \
    --resume_stage_index "$PLANNING_STAGE_RESUME_INDEX" \
    --max_context 100000 \
    $kaggle_resume_dataset_arg \
    $wandb_arg
fi

# -------- CONFIG EXTRACTION --------
if [[ "$RESUME_STAGE" == "CONFIG" || "$RESUME_STAGE" == "ALL" ]]; then
  echo "------- Config Extraction -------"
  python ../codes/1.1_extract_config.py \
    --paper_name "$PAPER_NAME" \
    --output_dir "$OUTPUT_DIR" 
 

  cp -rp "${OUTPUT_DIR}/planning_config.yaml" "${OUTPUT_REPO_DIR}/config.yaml"
  echo "✅ Copied config to repo: ${OUTPUT_REPO_DIR}/config.yaml"
fi

# -------- ANALYZING --------
if [[ "$RESUME_STAGE" == "ANALYZING" || "$RESUME_STAGE" == "ALL" ]]; then
  echo "------- Analyzing Stage -------"
  python ../codes/2_analyzing_litellm.py \
    --paper_name "$PAPER_NAME" \
    --gpt_version "$GPT_VERSION" \
    --pdf_json_path "$PDF_JSON_CLEANED_PATH" \
    --output_dir "$OUTPUT_DIR" \
    --resume_stage_index "$ANALYZING_STAGE_RESUME_INDEX" \
    --max_context 100000 \
    $kaggle_resume_dataset_arg \
    $wandb_arg 
fi

# -------- CODING --------
if [[ "$RESUME_STAGE" == "CODING" || "$RESUME_STAGE" == "ALL" ]]; then
  echo "------- Coding Stage -------"
  python ../codes/3_coding_litellm.py \
    --paper_name "$PAPER_NAME" \
    --gpt_version "$GPT_VERSION" \
    --pdf_json_path "$PDF_JSON_CLEANED_PATH" \
    --output_dir "$OUTPUT_DIR" \
    --output_repo_dir "$OUTPUT_REPO_DIR" \
    --resume_stage_index "$ANALYZING_STAGE_RESUME_INDEX" \
    --max_context 100000 \
    $kaggle_resume_dataset_arg \
    $wandb_arg 
fi

#ls -ltR "$OUTPUT_DIR"
