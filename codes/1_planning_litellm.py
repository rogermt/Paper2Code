"""
Planning stage for Paper2Code.

Features
--------
• Reads a research paper (JSON or LaTeX).
• Produces a multi-step plan (overall plan → architecture → logic → config).
• Uses safe_api_call / safe_api_call_prompt (chunk-aware, max_context).
• Generic tracking abstraction (console or WandB + Weave).
• Resume support from:
      1. local output directory,
      2. Kaggle dataset (path via --kaggle_dataset_path),
      3. WandB artifact (name via --wandb_artifact).
"""
import os
import sys
import json
import tempfile
import shutil
import logging
import argparse
from typing import List, Tuple, Any

from openai import OpenAI
import wandb

from openai_client import get_client
from gguf_utils import format_for_backend
from safe_api_call import safe_api_call, safe_api_call_prompt
from tracking import ExperimentTracker, WandbWeaveTracker, AttrDict
from utils import (
    print_response,
    print_log_cost,
    load_accumulated_cost,
    save_accumulated_cost,
    save_input_variable,
    save_artifacts,
    restore_artifacts_planning,
    load_from_kaggle,                    
)

logging.basicConfig(level=logging.INFO)
log = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# 2. CLI arguments
# ---------------------------------------------------------------------------
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
print(f"[TRACKER] Active run id → {tracker.run_id}")


client      = get_client()
output_dir  = args.output_dir or f"./output/{args.paper_name}"
os.makedirs(output_dir, exist_ok=True)

print(f"Using outputdir: {output_dir}")
save_input_variable(output_dir, "plan_msg", {"stage": "Planning", "content": "Overall plan"})
tracker.save_artifact(output_dir=output_dir, stage="planning", phase="initialization", file_pattern="*.json",)

if args.paper_format == "JSON":
    with open(args.pdf_json_path, "r", encoding="utf-8") as f:
        paper_content = json.load(f)
else:  # LaTeX
    with open(args.pdf_latex_path, "r", encoding="utf-8") as f:
        paper_content = f.read()

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
    traj, resp = restore_artifacts_planning(output_dir, resume_stage_index)
    if traj and resp:
        log.info("[RESUME] Restored from local directory.")
        return traj, resp

    # -- kaggle --------------------------------------------------------------
    if kaggle_path and os.path.exists(kaggle_path):
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
            t_file  = os.path.join(art_dir, "planning_trajectories.json")
            r_file  = os.path.join(art_dir, "planning_response.json")
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


# -------- plan_msg ----------------------------------------------------------
plan_msg: list[dict] = [
    {
        "role": "system",
        "content": (
            "You are an expert researcher and strategic planner with a deep understanding "
            "of experimental design and reproducibility in scientific research.\n"
            f"You will receive a research paper in {args.paper_format} format.\n"
            "Your task is to create a detailed and efficient plan to reproduce the "
            "experiments and methodologies described in the paper.\n"
            "This plan should align precisely with the paper's methodology, experimental "
            "setup, and evaluation metrics.\n\n"
            "Instructions:\n"
            "1. Align with the Paper: Your plan must strictly follow the methods, datasets, "
            "model configurations, hyperparameters, and experimental setups described in the paper.\n"
            "2. Be Clear and Structured: Present the plan in a well-organised and easy-to-follow "
            "format, breaking it down into actionable steps.\n"
            "3. Prioritise Efficiency: Optimise the plan for clarity and practical implementation "
            "while ensuring fidelity to the original experiments."
        ),
    },
    {
        "role": "user",
        "content": (
            f"## Paper\n{paper_content}\n\n"
            "## Task\n"
            "1. We want to reproduce the method described in the attached paper.\n"
            "2. The authors did not release any official code, so we have to plan our own implementation.\n"
            "3. Before writing any Python code, please outline a comprehensive plan that covers:\n"
            "   - Key details from the paper's **Methodology**.\n"
            "   - Important aspects of **Experiments**, including dataset requirements, experimental settings, "
            "hyperparameters, or evaluation metrics.\n"
            "4. The plan should be as **detailed and informative** as possible to help us write the final code later.\n\n"
            "## Requirements\n"
            "• You don't need to provide the actual code yet; focus on a **thorough, clear strategy**.\n"
            "• If something is unclear from the paper, mention it explicitly.\n\n"
            "## Instruction\n"
            "The response should give us a strong roadmap, making it easier to write the code later."
        ),
    },
]


# -------- file_list_msg -----------------------------------------------------
file_list_msg: list[dict] = [
    {
        "role": "user",
        "content": (
            "Your goal is to create a concise, usable, and complete software system design for reproducing the paper's method. "
            "Use appropriate open-source libraries and keep the overall architecture simple.\n\n"
            "Based on the plan for reproducing the paper’s main method, please design a concise, usable, and complete software system.\n"
            "Keep the architecture simple and make effective use of open-source libraries.\n\n"
            "-----\n\n"
            "## Format Example\n"
            "[CONTENT]\n"
            "{\n"
            '    "Implementation approach": "We will ...,",\n'
            '    "File list": [\n'
            '        "main.py",\n'
            '        "dataset_loader.py",\n'
            '        "model.py",\n'
            '        "trainer.py",\n'
            '        "evaluation.py"\n'
            "    ],\n"
            '    "Data structures and interfaces": "\\nclassDiagram\\n    class Main {\\n        +__init__()\\n        +run_experiment()\\n    }\\n    class DatasetLoader {\\n        +__init__(config: dict)\\n        +load_data() -> Any\\n    }\\n    class Model {\\n        +__init__(params: dict)\\n        +forward(x: Tensor) -> Tensor\\n    }\\n    class Trainer {\\n        +__init__(model: Model, data: Any)\\n        +train() -> None\\n    }\\n    class Evaluation {\\n        +__init__(model: Model, data: Any)\\n        +evaluate() -> dict\\n    }\\n    Main --> DatasetLoader\\n    Main --> Trainer\\n    Main --> Evaluation\\n    Trainer --> Model\\n",\n'
            '    "Program call flow": "\\nsequenceDiagram\\n    participant M as Main\\n    participant DL as DatasetLoader\\n    participant MD as Model\\n    participant TR as Trainer\\n    participant EV as Evaluation\\n    M->>DL: load_data()\\n    DL-->>M: return dataset\\n    M->>MD: initialize model()\\n    M->>TR: train(model, dataset)\\n    TR->>MD: forward(x)\\n    MD-->>TR: predictions\\n    TR-->>M: training complete\\n    M->>EV: evaluate(model, dataset)\\n    EV->>MD: forward(x)\\n    MD-->>EV: predictions\\n    EV-->>M: metrics\\n",\n'
            '    "Anything UNCLEAR": "Need clarification on the exact dataset format and any specialised hyperparameters."\n'
            "}\n"
            "[/CONTENT]\n\n"
            "## Nodes: \"<node>: <type>  # <instruction>\"\n"
            "- Implementation approach: <class 'str'>  # Summarise the chosen solution strategy.\n"
            "- File list: typing.List[str]  # Only need relative paths. ALWAYS write a main.py or app.py here.\n"
            "- Data structures and interfaces: typing.Optional[str]  # Use mermaid classDiagram code syntax ...\n"
            "- Program call flow: typing.Optional[str]  # Use sequenceDiagram code syntax ...\n"
            "- Anything UNCLEAR: <class 'str'>  # Mention ambiguities and ask for clarifications.\n\n"
            "## Constraint\n"
            "Format: output wrapped inside [CONTENT][/CONTENT] like the format example, nothing else.\n\n"
            "## Action\n"
            "Follow the instructions for the nodes, generate the output, and ensure it follows the format example."
        ),
    }
]

task_list_msg = [
    {
        "role": "system",
        "content": """You are a JSON-only responder for task breakdowns. Output ONLY the [CONTENT] JSON [/CONTENT] block with valid JSON—no inline comments in JSON, no markdown, no extra text. 
Deviation = invalid. Fill with general task info based on the PRD/design: Required packages, Logic Analysis (file-desc pairs), Task list (prioritized array of plain file names: lowercase, underscores, .py extension, no descriptions or parentheses), etc. Ensure 'Task list' key is included."""
    },
    {
        "role": "user",
        "content": """Your goal is break down tasks according to PRD/technical design, generate a task list, and analyse task dependencies.
You will break down tasks, analyse dependencies.

You outlined a clear PRD/technical design for reproducing the paper’s method and experiments.

Now, let's break down tasks according to PRD/technical design, generate a task list, and analyse task dependencies.
The Logic Analysis should not only consider the dependencies between files but also provide detailed descriptions to assist 
in writing the code needed to reproduce the paper.

CRITICAL: Your ENTIRE response must be ONLY this JSON wrapped in [CONTENT][/CONTENT]. No markdown, no plans, no <begin or artifacts. Deviation = invalid.

Format Example (output exactly this structure, filled for the paper):
[CONTENT]
{
    "Required packages": [
        "numpy==1.21.0",
        "torch==1.9.0"
    ],
    "Required Other language third-party packages": [
        "No third-party dependencies required"
    ],
    "Logic Analysis": [
        [
            "data_preprocessing.py",
            "DataPreprocessing class ........"
        ],
        [
            "trainer.py",
            "Trainer ....... "
        ],
        [
            "dataset_loader.py",
            "Handles loading and ........"
        ],
        [
            "model.py",
            "Defines the model ......."
        ],
        [
            "evaluation.py",
            "Evaluation class ........ "
        ],
        [
            "main.py",
            "Entry point  ......."
        ]
    ],
    "Task list": [
        "dataset_loader.py",
        "model.py",
        "trainer.py",
        "evaluation.py",
        "main.py"
    ],
    "Full API spec": "openapi: 3.0.0 ...",
    "Shared Knowledge": "Both data_preprocessing.py and trainer.py share ........",
    "Anything UNCLEAR": "Clarification needed on recommended hardware configuration for large-scale experiments."
}
[/CONTENT]

## Nodes: "<node>: <type>  # <instruction>"
- Required packages: typing.Optional[typing.List[str]]  # Provide required third-party packages.
- Required Other language third-party packages: typing.List[str]  # Non-Python dependencies.
- Logic Analysis: typing.List[typing.List[str]]  # File list with detailed description.
- Task list: typing.List[str]  # Prioritised list of plain file names (lowercase, underscores, .py extension, no descriptions).
- Full API spec: <class 'str'>  # OpenAPI spec if needed.
- Shared Knowledge: <class 'str'>  # Common utilities.
- Anything UNCLEAR: <class 'str'>

## Constraint
Format: output wrapped inside [CONTENT][/CONTENT] like the format example, nothing else.

## Action
Follow the node instructions above, generate your output accordingly, and ensure it follows the example.
Generate now."""
    }
]
# -------- config_msg (DeepSeek-friendly, compact) --------------------------
config_msg: list[dict] = [
    {
        "role": "system",
        "content": (
            "You are an expert researcher and strategic planner with a deep understanding "
            "of experimental design and reproducibility in scientific research.\n"
            "You will receive a research paper in JSON format.\n"
            "Your task is to create a detailed and efficient plan to reproduce the "
            "experiments and methodologies described in the paper.\n"
            "This plan should align precisely with the paper's methodology, experimental "
            "setup, and evaluation metrics.\n\n"
            "Instructions:\n"
            "1. Align with the Paper: Your plan must strictly follow the methods, datasets, "
            "model configurations, hyperparameters, and experimental setups described in the paper.\n"
            "2. Be Clear and Structured: Present the plan in a well-organised and easy-to-follow "
            "format, breaking it down into actionable steps.\n"
            "3. Prioritise Efficiency: Optimise the plan for clarity and practical implementation "
            "while ensuring fidelity to the original experiments.\n\n"
            "YAML OUTPUT RULE: After planning internally, output ONLY a valid YAML file for config.yaml "
            "in a ```yaml code block. Use proper YAML syntax: keys followed by colon, no outer braces, "
            "indented nested structures with 2 spaces, inline comments with '#'. No JSON format, no [CONTENT] tags, "
            "no explanations. Example syntax:\n"
            "dataset: mnist  # comment\n"
            "model:\n"
            "  type: nsfg++\n"
            "training:\n"
            "  learning_rate: 0.0001  # assumed default\n"
            "Extract from paper: batch_size 256, epochs 20000, num_steps 5. Defaults: lr 0.0001, epsilon 0.1, blur 0.5.\n"
            "Your response must start with ```yaml and end with ```, nothing else."
        )
    },
    {
        "role": "user",
        "content": (
            "## Paper\n"
            "{'paper_id': 'NSGF', ... [full JSON]} \n\n"
            "Plan the NSGF reproduction, then output the config.yaml exactly as per rules."
        )
    }
]


@tracker.trace
def process_planning_stage(
    client: OpenAI,
    trajectories: list,
    instruction_msg: list,
    idx: int,
    responses: list,
    output_dir: str,
    gpt_version: str,
) -> Tuple[List[Any], List[Any]]:

    stage_map = {
        0: "overall_plan",
        1: "architecture_design",
        2: "logic_design",
        3: "config_generation",
    }
    stage_name = stage_map.get(idx, f"stage_{idx}")
    log.info(f"[STAGE] {stage_name}")

    # extend chat history
    trajectories.extend(instruction_msg)

    # choose safe call
    # GGUF-aware API call
    if "gguf" in gpt_version.lower():
        formatted_input = format_for_backend(trajectories, gpt_version)
        if isinstance(formatted_input, str):
            completion = completion = safe_api_call_prompt(
                client, formatted_input, gpt_version,
                max_context=args.max_context,
                max_output_tokens=args.max_output_tokens,
                reasoning_tokens=args.reasoning_tokens,
                safety_margin=args.safety_margin
            )
        else:
            completion = completion = safe_api_call(
                client, formatted_input, gpt_version,
                max_context=args.max_context,
                max_output_tokens=args.max_output_tokens,
                reasoning_tokens=args.reasoning_tokens,
                safety_margin=args.safety_margin
            )
    else:
        completion = completion = safe_api_call(
            client, trajectories, gpt_version,
            max_context=args.max_context,
            max_output_tokens=args.max_output_tokens,
            reasoning_tokens=args.reasoning_tokens,
            safety_margin=args.safety_margin
        )

    # save json
    fname = os.path.join(output_dir, f"{stage_name}.json")
    with open(fname, "w", encoding="utf-8") as f:
        json.dump(json.loads(completion.model_dump_json()), f, indent=2)
    log.info(f"[SAVE] {fname}")

    # update histories
    msg = completion.choices[0].message
    trajectories.append({"role": msg.role, "content": msg.content})
    responses.append(json.loads(completion.model_dump_json()))

    # metric
    if completion.usage:
        tracker.log_metrics({f"{stage_name}_tokens": completion.usage.total_tokens})

    return trajectories, responses

def main():
    log.info(f"[RESUME] stage index = {args.resume_stage_index}")
    trajectories, responses = restore_context(
        args.resume_stage_index,
        output_dir,
        args.kaggle_dataset_path,
        args.wandb_artifact,
    )

    instructions = [plan_msg, file_list_msg, task_list_msg, config_msg]

    for idx, instr in enumerate(instructions):
        if idx < args.resume_stage_index:
            continue
        trajectories, responses = process_planning_stage(
            client,
            trajectories,
            instr,
            idx,
            responses,
            output_dir,
            args.gpt_version,
        )
        tracker.save_artifact(output_dir=output_dir, stage="planning", phase="api_call", step=idx + 1, file_pattern="*.json")
       
    tracker.save_artifact(output_dir=output_dir, stage="planning", phase="final_save", step=len(instructions) + 1,  file_pattern="*.json")

    save_artifacts(output_dir, trajectories, responses, stage="planning")
    
    log.info("✅ Planning complete")

# ---------------------------------------------------------------------------
if __name__ == "__main__":
    main()