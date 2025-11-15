import os
import sys
import json
import re
import copy
import shutil
import glob
import tempfile
import logging
import argparse
from typing import List, Dict, Any, Tuple
from tqdm import tqdm
import wandb

from openai import OpenAI
from openai_client import get_client
from gguf_utils import format_for_backend
from safe_api_call import safe_api_call, safe_api_call_prompt
from tracking import ExperimentTracker, WandbWeaveTracker, AttrDict
from utils import (
    extract_planning, 
    content_to_json, 
    extract_code_from_content, 
    print_response,
    save_artifacts,
    load_from_kaggle
)

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


parser = argparse.ArgumentParser()
parser.add_argument("--paper_name",        required=True)
parser.add_argument("--gpt_version",       required=True, default="o3-mini")
parser.add_argument("--paper_format",      default="JSON", choices=["JSON", "LaTeX"])
parser.add_argument("--pdf_json_path")
parser.add_argument("--pdf_latex_path")
parser.add_argument("--output_dir",        default="")
parser.add_argument("--output_repo_dir",   default="")
parser.add_argument("--max_context",       type=int, default=32768)
parser.add_argument("--max_output_tokens", type=int, default=4096)
parser.add_argument("--reasoning_tokens",  type=int, default=1500)
parser.add_argument("--safety_margin",     type=float, default=0.10)
parser.add_argument("--resume_stage_index",type=int, default=0)

# resume sources
parser.add_argument("--kaggle_dataset_path", default=None,
                    help="Path like /kaggle/input/<dataset>/output/<paper>")
parser.add_argument("--wandb_artifact",     default=None,
                    help="e.g. 'rogermt23/nsgf-paper2code-playground/coding:latest'")

# tracking
parser.add_argument("--wandb_run_id",      default=None)

args = parser.parse_args()

# --- build config ---
wandb_run_id = None if args.wandb_run_id in (None, "None", "") else args.wandb_run_id
config = AttrDict(wandb_run_id=wandb_run_id)

# --- instantiate tracker ---
# If no run is active, create/resume via WandbWeaveTracker
# Otherwise fall back to ExperimentTracker
tracker = WandbWeaveTracker(config) if wandb.run is None else ExperimentTracker
logger.info(f"[TRACKER] Active run id → {tracker.run_id}")


client      = get_client()
output_dir  = args.output_dir or f"./output/{args.paper_name}"
output_repo_dir = args.output_repo_dir or f"{output_dir}/code_output"
os.makedirs(output_dir, exist_ok=True)
os.makedirs(output_repo_dir, exist_ok=True)

logger.info(f"Using outputdir: {output_dir}")
logger.info(f"Using output_repo_dir: {output_repo_dir}")


def restore_context(
    resume_stage_index: int,
    output_dir: str,
    kaggle_path: str | None,
    wandb_artifact_name: str | None,
) -> Tuple[List[Any], List[Any], Dict[str, str], List[str]]:
    """
    Restore (trajectories, responses, done_file_dict, done_file_lst) in order:
        1. local output_dir
        2. Kaggle dataset
        3. WandB artifact
    Returns ([], [], {}, []) if nothing was found.
    """
    if resume_stage_index == 0:
        return [], [], {}, []

    # -- local ---------------------------------------------------------------
    traj_file = os.path.join(output_dir, "coding_trajectories.json")
    resp_file = os.path.join(output_dir, "coding_response.json")
    done_dict_file = os.path.join(output_dir, "done_file_dict.json")
    done_list_file = os.path.join(output_dir, "done_file_lst.json")
    
    if all(os.path.exists(f) for f in [traj_file, resp_file, done_dict_file, done_list_file]):
        with open(traj_file) as f:
            traj = json.load(f)
        with open(resp_file) as f:
            resp = json.load(f)
        with open(done_dict_file) as f:
            done_dict = json.load(f)
        with open(done_list_file) as f:
            done_lst = json.load(f)
        logger.info("[RESUME] Restored from local directory.")
        return traj, resp, done_dict, done_lst

    # -- kaggle --------------------------------------------------------------
    if kaggle_path and os.path.exists(kaggle_path):
        traj, resp = load_from_kaggle(kaggle_path, output_dir)
        if traj and resp:
            # For coding stage, we need to reconstruct done files
            logger.info("[RESUME] Restored from Kaggle dataset.")
            return traj, resp, {}, []

    # -- wandb ---------------------------------------------------------------
    if wandb_artifact_name:
        try:
            import wandb as wandb_api
            api  = wandb_api.Api()
            art  = api.artifact(wandb_artifact_name, type="dataset")
            tmpd = tempfile.mkdtemp()
            art_dir = art.download(root=tmpd)
            t_file  = os.path.join(art_dir, "coding_trajectories.json")
            r_file  = os.path.join(art_dir, "coding_response.json")
            d_dict_file = os.path.join(art_dir, "done_file_dict.json")
            d_list_file = os.path.join(art_dir, "done_file_lst.json")
            if all(os.path.exists(f) for f in [t_file, r_file, d_dict_file, d_list_file]):
                shutil.copy(t_file, output_dir)
                shutil.copy(r_file, output_dir)
                shutil.copy(d_dict_file, output_dir)
                shutil.copy(d_list_file, output_dir)
                with open(t_file) as f:
                    traj = json.load(f)
                with open(r_file) as f:
                    resp = json.load(f)
                with open(d_dict_file) as f:
                    done_dict = json.load(f)
                with open(d_list_file) as f:
                    done_lst = json.load(f)
                logger.info("[RESUME] Restored from WandB artifact.")
                return traj, resp, done_dict, done_lst
        except Exception as e:
            logger.warning(f"[RESUME] WandB artifact restore failed: {e}")

    return [], [], {}, []

# ---------------------------------------------------------------------------
# 5. Load Paper Content
# ---------------------------------------------------------------------------
if args.paper_format == "JSON":
    with open(args.pdf_json_path, "r", encoding="utf-8") as f:
        paper_content = json.load(f)
else:  # LaTeX
    with open(args.pdf_latex_path, "r", encoding="utf-8") as f:
        paper_content = f.read()

# ---------------------------------------------------------------------------
# 6. Load Artifacts from Planning Stage
# ---------------------------------------------------------------------------
config_yaml_file = os.path.join(output_dir, "config.yaml")
plan_traj_json = os.path.join(output_dir, "planning_trajectories.json")

   
with open(os.path.join(output_dir, "planning_config.yaml")) as f: 
    config_yaml = f.read()

context_lst = extract_planning(plan_traj_json)
task_list_json = content_to_json(context_lst[2])

def get_key_from_dict(d, keys):
    for key in keys:
        if key in d:
            return d[key]
    return None

todo_file_lst = get_key_from_dict(task_list_json, ['Task list', 'task_list', 'task list'])
if not todo_file_lst:
    logger.error("[ERROR] Could not find 'Task list' in the planning output. Please re-generate.")
    print("Exiting coding stage.")
    sys.exit(0)


code_msg = [
    {
        "role": "system", 
        "content": f"""You are an expert researcher and software engineer with a deep understanding of experimental design and reproducibility in scientific research.
You will receive a research paper in {args.paper_format} format, an overview of the plan, a Design in JSON format consisting of "Implementation approach", "File list", "Data structures and interfaces", and "Program call flow", followed by a Task in JSON format that includes "Required packages", "Required other language third-party packages", "Logic Analysis", and "Task list", along with a configuration file named "config.yaml".
Your task is to write code to reproduce the experiments and methodologies described in the paper.

The code you write must be elegant, modular, and maintainable, adhering to Google-style guidelines.
The code must strictly align with the paper's methodology, experimental setup, and evaluation metrics.
Write code with triple quoto."""
    }
]

def get_write_msg(todo_file_name, detailed_logic_analysis, done_file_lst, done_file_dict):
    code_files = ""
    for done_file in done_file_lst:
        if done_file.endswith(".yaml"): 
            continue
        code_files += f"""
```python
{done_file_dict[done_file]}
```

"""

    return [
        {
            'role': 'user', 
            "content": f"""# Context
## Paper
{paper_content}

-----

## Overview of the plan
{context_lst[0]}

-----

## Design
{context_lst[1]}

-----

## Task
{context_lst[2]}

-----

## Configuration file
```yaml
{config_yaml}
```
-----

## Code Files
{code_files}

-----

# Format example
## Code: {todo_file_name}
```python
## {todo_file_name}
...
```

-----

# Instruction
Based on the paper, plan, design, task and configuration file(config.yaml) specified previously, follow "Format example", write the code.

We have {done_file_lst}.
Next, you must write only the "{todo_file_name}".
1. Only One file: do your best to implement THIS ONLY ONE FILE.
2. COMPLETE CODE: Your code will be part of the entire project, so please implement complete, reliable, reusable code snippets.
3. Set default value: If there is any setting, ALWAYS SET A DEFAULT VALUE, ALWAYS USE STRONG TYPE AND EXPLICIT VARIABLE. AVOID circular import.
4. Follow design: YOU MUST FOLLOW "Data structures and interfaces". DONT CHANGE ANY DESIGN. Do not use public member functions that do not exist in your design.
5. CAREFULLY CHECK THAT YOU DONT MISS ANY NECESSARY CLASS/FUNCTION IN THIS FILE.
6. Before using a external variable/module, make sure you import it first.
7. Write out EVERY CODE DETAIL, DON'T LEAVE TODO.
8. REFER TO CONFIGURATION: you must use configuration from "config.yaml". DO NOT FABRICATE any configuration values.
#### NSGF PAPER SPECIFICS
9. FULL IMPLEMENTATION REQUIRED: You must fully implement the Sinkhorn divergence computation using the GeomLoss library. Do not use placeholders or mock returns. Use the correct SamplesLoss interface and ensure all tensor shapes are compatible.
10. VELOCITY FIELD MUST BE TRAINABLE: The velocity field network must be implemented as a trainable PyTorch module. You must include forward pass, loss computation, and optimizer setup. Do not omit training logic.
11. NO PLACEHOLDERS: You must not use `return torch.tensor(0.0)` or similar stubs. Every function must be complete and executable.
12. USE CONFIGURATION FOR ALL PARAMETERS: All hyperparameters (e.g. epsilon, learning rate, batch size) must be read from config.yaml. Do not hardcode any values.
13. INCLUDE TRAINING LOOP: If the file includes model logic, you must also include a training loop that demonstrates how the model is trained using the provided data structures.
14. INCLUDE INFERENCE LOGIC: If applicable, include inference functions that show how the trained model is used to generate outputs.

{detailed_logic_analysis}

## Code: {todo_file_name}"""
        }
    ]


@tracker.trace
def process_coding_stage(
    client: OpenAI,
    trajectories: list,
    instruction_msg: list,
    idx: int,
    todo_file_name: str,
    output_dir: str,
    output_repo_dir: str,
    artifact_output_dir: str,
    gpt_version: str,
    done_file_lst: List[str],
    done_file_dict: Dict[str, str],
    tracker: ExperimentTracker
) -> Tuple[List[Any], Dict, List[str], Dict[str, str]]:
    """
    Process a single coding stage using safe_api_call for chunking, save the completion JSON, and update trajectories.
    """
    current_stage = f"[CODING] {todo_file_name}"
    logger.info(current_stage)
    
    if todo_file_name == "config.yaml":
        return trajectories, {}, done_file_lst, done_file_dict

    # Extend conversation history with the specific instruction for this file
    trajectories.extend(instruction_msg)

    # GGUF-aware API call
    if "gguf" in gpt_version.lower():
        formatted_input = format_for_backend(trajectories, gpt_version)
        if isinstance(formatted_input, str):
            completion = safe_api_call_prompt(
                client, formatted_input, gpt_version,
                max_context=args.max_context,
                max_output_tokens=args.max_output_tokens,
                reasoning_tokens=args.reasoning_tokens,
                safety_margin=args.safety_margin
            )
        else:
            completion = safe_api_call(
                client, formatted_input, gpt_version,
                max_context=args.max_context,
                max_output_tokens=args.max_output_tokens,
                reasoning_tokens=args.reasoning_tokens,
                safety_margin=args.safety_margin
            )
    else:
        completion = safe_api_call(
            client, trajectories, gpt_version,
            max_context=args.max_context,
            max_output_tokens=args.max_output_tokens,
            reasoning_tokens=args.reasoning_tokens,
            safety_margin=args.safety_margin
        )

    # --- Artifact Saving ---
    completion_json = json.loads(completion.model_dump_json())

    # Log usage statistics
    if hasattr(completion, 'usage'):
        usage = completion.usage
        logger.info(f"Usage Stats - Prompt Tokens: {usage.prompt_tokens}, Completion Tokens: {usage.completion_tokens}, Total Tokens: {usage.total_tokens}")
        tracker.log_metrics({
            f"coding_{idx}_tokens": usage.total_tokens,
            f"coding_{idx}_prompt_tokens": usage.prompt_tokens,
            f"coding_{idx}_completion_tokens": usage.completion_tokens
        })

    # Save coding artifact
    save_todo_file_name = todo_file_name.replace("/", "_")
    artifact_file = os.path.join(artifact_output_dir, f"{save_todo_file_name}_coding.txt")
    with open(artifact_file, 'w', encoding='utf-8') as f:
        f.write(completion_json['choices'][0]['message']['content'])
    logger.info(f"Saved coding to {artifact_file}")
    tracker.save_artifact(output_dir=output_dir, stage="coding", phase="api_call", file_pattern=artifact_file)

    # Update trajectories
    message = completion.choices[0].message
    trajectories.append({'role': message.role, 'content': message.content})

    # Extract and save code
    code = extract_code_from_content(message.content)
    if len(code) == 0:
        code = message.content
    done_file_dict[todo_file_name] = code

    # Create directory structure and save code file
    todo_file_dir = '/'.join(todo_file_name.split("/")[:-1])
    if todo_file_dir:
        os.makedirs(os.path.join(output_repo_dir, todo_file_dir), exist_ok=True)
    
    code_file_path = os.path.join(output_repo_dir, todo_file_name)
    with open(code_file_path, 'w', encoding='utf-8') as f:
        f.write(code)
    logger.info(f"Saved code to {code_file_path}")
    tracker.save_artifact(output_dir=output_dir, stage="coding", phase="api_call", file_pattern=code_file_path)

    done_file_lst.append(todo_file_name)

    # Print response
    print_response(completion_json)

    return trajectories, completion_json, done_file_lst, done_file_dict

def main():
    logger.info(f"Resuming from file index: {args.resume_stage_index}")
    
    # Restore context
    trajectories, responses, done_file_dict, done_file_lst = restore_context(
        args.resume_stage_index,
        output_dir,
        args.kaggle_dataset_path,
        args.wandb_artifact,
    )

    # Initialize if not restored
    if not done_file_lst:
        done_file_lst = ['config.yaml']
    if not done_file_dict:
        done_file_dict = {}

    # Log initialization
    tracker.save_artifact(output_dir=output_dir, stage="coding", phase="initialization", file_pattern="config.yaml")
    tracker.log_metrics({"stage": "coding", "phase": "initialization"})

    # Create artifact directory
    artifact_output_dir = os.path.join(output_dir, "coding_artifacts")
    os.makedirs(artifact_output_dir, exist_ok=True)

    # Preprocess: Load simple analysis responses
    detailed_logic_analysis_dict = {}
    for todo_file_name in todo_file_lst:
        save_todo_file_name = todo_file_name.replace("/", "_")
        if todo_file_name == "config.yaml":
            continue
        try:
            analysis_file = os.path.join(output_dir, f"{save_todo_file_name}_simple_analysis_response.json")
            with open(analysis_file, 'r', encoding='utf-8') as f:
                detailed_logic_analysis_response = json.load(f)
            detailed_logic_analysis_dict[todo_file_name] = detailed_logic_analysis_response[0]['choices'][0]['message']['content']
        except FileNotFoundError:
            logger.warning(f"File {save_todo_file_name}_simple_analysis_response.json not found, skipping.")
            detailed_logic_analysis_dict[todo_file_name] = ""

    # Process coding stages
    for todo_idx, todo_file_name in enumerate(tqdm(todo_file_lst)):
        # Skip already processed files
        if todo_idx < args.resume_stage_index:
            continue

        # Skip config.yaml
        if todo_file_name == "config.yaml":
            continue

        # Create fresh trajectory for each file
        current_trajectories = copy.deepcopy(code_msg)

        # Get file description and create instruction
        file_analysis = detailed_logic_analysis_dict.get(todo_file_name, "")
        instruction_msg = get_write_msg(todo_file_name, file_analysis, done_file_lst, done_file_dict)

        # Process coding stage
        final_trajectories, response_json, done_file_lst, done_file_dict = process_coding_stage(
            client=client,
            trajectories=current_trajectories,
            instruction_msg=instruction_msg,
            idx=todo_idx,
            todo_file_name=todo_file_name,
            output_dir=output_dir,
            output_repo_dir=output_repo_dir,
            artifact_output_dir=artifact_output_dir,
            gpt_version=args.gpt_version,
            done_file_lst=done_file_lst,
            done_file_dict=done_file_dict,
            tracker=tracker
        )

        # Collect responses
        responses.append(response_json)

        # Save progress after each file
        traj_file = os.path.join(output_dir, "coding_trajectories.json")
        resp_file = os.path.join(output_dir, "coding_response.json")
        done_dict_file = os.path.join(output_dir, "done_file_dict.json")
        done_list_file = os.path.join(output_dir, "done_file_lst.json")
        
        with open(traj_file, 'w') as f:
            json.dump(final_trajectories, f, indent=2)
        with open(resp_file, 'w') as f:
            json.dump(responses, f, indent=2)
        with open(done_dict_file, 'w') as f:
            json.dump(done_file_dict, f, indent=2)
        with open(done_list_file, 'w') as f:
            json.dump(done_file_lst, f, indent=2)

        tracker.save_artifact(output_dir=output_dir, stage="coding", phase="progress_save", file_pattern=traj_file)
        tracker.save_artifact(output_dir=output_dir, stage="coding", phase="progress_save", file_pattern=resp_file)
        tracker.save_artifact(output_dir=output_dir, stage="coding", phase="progress_save", file_pattern=done_dict_file)
        tracker.save_artifact(output_dir=output_dir, stage="coding", phase="progress_save", file_pattern=done_list_file)

        # Log progress
        tracker.log_metrics({
            "stage": "coding",
            "phase": "api_call",
            "file": todo_file_name,
            "step": todo_idx + 1
        })

    # --- Finalization ---
    # Save final artifacts
    final_traj_file = os.path.join(output_dir, "coding_trajectories_final.json")
    final_resp_file = os.path.join(output_dir, "coding_response_final.json")
    
    with open(final_traj_file, 'w') as f:
        json.dump(trajectories, f, indent=2)
    with open(final_resp_file, 'w') as f:
        json.dump(responses, f, indent=2)
    
    logger.info(f"Saved final trajectories to {final_traj_file}")
    logger.info(f"Saved final responses to {final_resp_file}")
    
    tracker.save_artifact(output_dir=output_dir, stage="coding", phase="final_save", file_pattern=final_traj_file)
    tracker.save_artifact(output_dir=output_dir, stage="coding", phase="final_save", file_pattern=final_resp_file)
    tracker.log_metrics({"stage": "coding", "phase": "final_save", "step": len(todo_file_lst) + 1})
    

    logger.info("✅ Coding complete")

if __name__ == "__main__":
    main()