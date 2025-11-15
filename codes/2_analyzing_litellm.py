import os
import sys
import json
import logging
import argparse
import copy
import shutil
import wandb
from typing import List, Dict, Any, Tuple

from openai import OpenAI
from openai_client import get_client
from gguf_utils import format_for_backend
from safe_api_call import safe_api_call, safe_api_call_prompt
from utils import extract_planning, content_to_json, print_response, save_artifacts
from tracking import ExperimentTracker, WandbWeaveTracker, AttrDict

logging.basicConfig(level=logging.INFO)
log = logging.getLogger(__name__)

parser = argparse.ArgumentParser()
parser.add_argument("--paper_name",        required=True)
parser.add_argument("--gpt_version",       required=True)
parser.add_argument("--paper_format",      default="JSON", choices=["JSON", "LaTeX"])
parser.add_argument("--pdf_json_path")
parser.add_argument("--pdf_latex_path")
parser.add_argument("--output_dir",        default="")
parser.add_argument("--max_context",       type=int, default=32768)
parser.add_argument("--max_output_tokens", type=int, default=4096)
parser.add_argument("--reasoning_tokens",  type=int, default=1500)
parser.add_argument("--safety_margin",     type=float, default=0.10)
parser.add_argument("--resume_stage_index",type=int, default=0)

# resume sources
parser.add_argument("--kaggle_dataset_path", default=None,
                    help="Path like /kaggle/input/<dataset>/output/<paper>")
parser.add_argument("--wandb_artifact",     default=None,
                    help="e.g. 'rogermt23/nsgf-paper2code-playground/planning:latest'")

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
log.info(f"[TRACKER] Active run id → {tracker.run_id}")


client      = get_client()
output_dir  = args.output_dir or f"./output/{args.paper_name}"
os.makedirs(output_dir, exist_ok=True)

print(f"Using outputdir: {output_dir}")
tracker.save_artifact(output_dir=output_dir, stage="analyzing", phase="initialization", file_pattern="*.json")

def restore_context(
    resume_stage_index: int,
    output_dir: str,
    kaggle_path: str | None,
    wandb_artifact_name: str | None,
) -> Tuple[List[Any], List[Any]]:
    """
    Restore (trajectories, responses) in order:
        1. local output_dir
        2. Kaggle dataset
        3. WandB artifact
    Returns ([], []) if nothing was found.
    """
    if resume_stage_index == 0:
        return [], []

    # -- local ---------------------------------------------------------------
    traj_file = os.path.join(output_dir, "analyzing_trajectories.json")
    resp_file = os.path.join(output_dir, "analyzing_response.json")
    
    if os.path.exists(traj_file) and os.path.exists(resp_file):
        with open(traj_file) as f:
            traj = json.load(f)
        with open(resp_file) as f:
            resp = json.load(f)
        log.info("[RESUME] Restored from local directory.")
        return traj, resp

    # -- kaggle --------------------------------------------------------------
    if kaggle_path and os.path.exists(kaggle_path):
        from utils import load_from_kaggle
        traj, resp = load_from_kaggle(kaggle_path, output_dir)
        if traj and resp:
            log.info("[RESUME] Restored from Kaggle dataset.")
            return traj, resp

    # -- wandb ---------------------------------------------------------------
    if wandb_artifact_name:
        try:
            api  = wandb.Api()
            art  = api.artifact(wandb_artifact_name, type="dataset")
            tmpd = tempfile.mkdtemp()
            art_dir = art.download(root=tmpd)
            t_file  = os.path.join(art_dir, "analyzing_trajectories.json")
            r_file  = os.path.join(art_dir, "analyzing_response.json")
            if os.path.exists(t_file) and os.path.exists(r_file):
                shutil.copy(t_file, output_dir)
                shutil.copy(r_file, output_dir)
                with open(t_file) as f:
                    traj = json.load(f)
                with open(r_file) as f:
                    resp = json.load(f)
                log.info("[RESUME] Restored from WandB artifact.")
                return traj, resp
        except Exception as e:
            log.warning(f"[RESUME] WandB artifact restore failed: {e}")

    return [], []


if args.paper_format == "JSON":
    with open(args.pdf_json_path, "r", encoding="utf-8") as f:
        paper_content = json.load(f)
else:  # LaTeX
    with open(args.pdf_latex_path, "r", encoding="utf-8") as f:
        paper_content = f.read()


config_yaml_file = os.path.join(output_dir, "config.yaml")
if args.kaggle_dataset_path:
    shutil.copytree(args.kaggle_dataset_path, output_dir, dirs_exist_ok=True)
   
with open(os.path.join(output_dir, "planning_config.yaml")) as f: 
    config_yaml = f.read()

context_lst = extract_planning(os.path.join(output_dir, "planning_trajectories.json"))

task_list_json = content_to_json(context_lst[2])

def get_key_from_dict(d, keys):
    for key in keys:
        if key in d:
            return d[key]
    return None

todo_file_lst = get_key_from_dict(task_list_json, ['Task list', 'task_list', 'task list'])
if not todo_file_lst:
    log.error("[ERROR] Could not find 'Task list' in the planning output. Please re-generate.")
    sys.exit(0)

logic_analysis_list = get_key_from_dict(task_list_json, ['Logic Analysis', 'logic_analysis', 'logic analysis'])
if not logic_analysis_list:
    log.error("[ERROR] Could not find 'Logic Analysis' in the planning output. Please re-generate.")
    sys.exit(0)

logic_analysis_dict = {desc[0]: desc[1] for desc in logic_analysis_list}

analysis_msg = [
    {
        "role": "system",
        "content": (
            "You are an expert researcher, strategic analyzer and software engineer with a deep understanding of "
            "experimental design and reproducibility in scientific research.\n"
            f"You will receive a research paper in {args.paper_format} format, an overview of the plan, a design in JSON format "
            "consisting of 'Implementation approach', 'File list', 'Data structures and interfaces', and 'Program call flow', "
            "followed by a task in JSON format that includes 'Required packages', 'Required other language third-party packages', "
            "'Logic Analysis', and 'Task list', along with a configuration file named 'config.yaml'.\n\n"
            "Your task is to conduct a comprehensive logic analysis to accurately reproduce the experiments and methodologies "
            "described in the research paper.\n"
            "This analysis must align precisely with the paper’s methodology, experimental setup, and evaluation criteria.\n\n"
            "1. Align with the Paper: Your analysis must strictly follow the methods, datasets, model configurations, "
            "hyperparameters, and experimental setups described in the paper.\n"
            "2. Be Clear and Structured: Present your analysis in a logical, well-organized, and actionable format that is easy to follow and implement.\n"
            "3. Prioritize Efficiency: Optimize the analysis for clarity and practical implementation while ensuring fidelity to the original experiments.\n"
            "4. Follow design: YOU MUST FOLLOW 'Data structures and interfaces'. DONT CHANGE ANY DESIGN. Do not use public member functions that do not exist in your design.\n"
            "5. REFER TO CONFIGURATION: Always reference settings from the config.yaml file. Do not invent or assume any values—"
            "only use configurations explicitly provided."
        )
    }
]

def get_analysis_instruction_msg(file_name, file_description):
    draft_desc = f"Write the logic analysis in '{file_name}', which is intended for '{file_description}'."
    if not file_description.strip():
        draft_desc = f"Write the logic analysis in '{file_name}'."

    return [{
        'role': 'user', 
        "content": f"""## Paper
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

## Instruction
Conduct a Logic Analysis to assist in writing the code, based on the paper, the plan, the design, the task and the previously specified configuration file (config.yaml). 
You DON'T need to provide the actual code yet; focus on a thorough, clear analysis.

{draft_desc}

-----

## Logic Analysis: {file_name}"""
    }]


@tracker.trace
def process_analysis_stage(
    client: OpenAI,
    trajectories: list,
    instruction_msg: list,
    idx: int,
    file_name: str,
    output_dir: str,
    gpt_version: str,
    tracker: ExperimentTracker
) -> Tuple[List[Any], Dict]:
    """
    Process the analysis for a single file, save artifacts, and update trajectories.
    """
    current_stage_name = f"[Analyzing] Stage: {idx} File: {file_name}"
    log.info(current_stage_name)
    log.info(f"Using tracker run id: {tracker.run_id}")

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

    # Sanitize filename for saving
    sanitized_filename = file_name.replace("/", "_")
    artifact_dir = os.path.join(output_dir, "analyzing_artifacts")
    os.makedirs(artifact_dir, exist_ok=True)

    # Save the full JSON response
    json_filename = os.path.join(artifact_dir, f"{sanitized_filename}_response.json")
    with open(json_filename, 'w', encoding='utf-8') as f:
        json.dump(completion_json, f, indent=4)
    log.info(f"Saved JSON to {json_filename}")
    tracker.save_artifact(output_dir=output_dir, stage="analyzing", phase="api_call", file_pattern=json_filename)
   
    # Save analysis content
    analysis_content = completion_json['choices'][0]['message']['content']
    analysis_filename = os.path.join(artifact_dir, f"{sanitized_filename}_analysis.txt")
    with open(analysis_filename, 'w', encoding='utf-8') as f:
        f.write(analysis_content)
    log.info(f"Saved analysis text to {analysis_filename}")
    tracker.save_artifact(output_dir=output_dir, stage="analyzing", phase="api_call", file_pattern=analysis_filename)

    # Update trajectories
    message = completion.choices[0].message
    trajectories.append({'role': message.role, 'content': message.content})

    # Log metrics
    if completion.usage:
        tracker.log_metrics({
            f"analyzing_{idx}_tokens": completion.usage.total_tokens,
            f"analyzing_{idx}_cost": completion.usage.cost if hasattr(completion.usage, 'cost') else 0
        })

    return trajectories, completion_json


def main():
    log.info(f"Resuming from file index: {args.resume_stage_index}")
    trajectories, responses = restore_context(
        args.resume_stage_index,
        output_dir,
        args.kaggle_dataset_path,
        args.wandb_artifact,
    )

    # Log initialization
    tracker.save_artifact(output_dir=output_dir, stage="analyzing", phase="initialization", file_pattern="config.yaml")
    tracker.log_metrics({"stage": "analyzing", "phase": "initialization"})

    # --- Analysis Loop for Each File ---
    for idx, file_name in enumerate(todo_file_lst):
        # Skip already processed files
        if idx < args.resume_stage_index:
            continue

        # Skip config.yaml
        if file_name == "config.yaml":
            continue

        # Create fresh trajectory for each file
        current_trajectories = copy.deepcopy(analysis_msg)

        # Get file description
        file_description = logic_analysis_dict.get(file_name, "")
        instruction_msg = get_analysis_instruction_msg(file_name, file_description)

        # Process analysis
        final_trajectories, response_json = process_analysis_stage(
            client=client,
            trajectories=current_trajectories,
            instruction_msg=instruction_msg,
            idx=idx,
            file_name=file_name,
            output_dir=output_dir,
            gpt_version=args.gpt_version,
            tracker=tracker
        )

        # Collect responses
        responses.append(response_json)

        # Log progress
        tracker.log_metrics({
            "stage": "analyzing",
            "phase": "api_call",
            "file": file_name,
            "step": idx + 1
        })

        # Save artifacts after each file
        save_artifacts(output_dir, final_trajectories, responses, todo_file_name=file_name, stage="analyzing")
    
    tracker.save_artifact(output_dir=output_dir, stage="analyzing", phase="final_save", file_pattern="*.json")

    log.info("✅ Analyzing complete")

if __name__ == "__main__":
    main()