import json
import re
import os
import argparse
import shutil
from utils import extract_planning, content_to_json, format_json_data

parser = argparse.ArgumentParser()

parser.add_argument('--paper_name',type=str)
parser.add_argument('--output_dir',type=str, default="")

args    = parser.parse_args()

output_dir = args.output_dir

with open(f'{output_dir}/planning_trajectories.json', encoding='utf8') as f:
    traj = json.load(f)

yaml_raw_content = ""
for turn_idx, turn in enumerate(traj):
        if turn_idx == 8:
            yaml_raw_content = turn['content']   

if "</think>" in yaml_raw_content:
    yaml_raw_content = yaml_raw_content.split("</think>")[-1]

match = re.search(r"```yaml\n(.*?)\n```", yaml_raw_content, re.DOTALL)
if match:
    yaml_content = match.group(1)
    with open(f'{output_dir}/planning_config.yaml', 'w', encoding='utf8') as f:
        f.write(yaml_content)
else:
    # print("No YAML content found.")
    match2 = re.search(r"```yaml\\n(.*?)\\n```", yaml_raw_content, re.DOTALL)
    if match2:
        yaml_content = match2.group(1)
        with open(f'{output_dir}/planning_config.yaml', 'w', encoding='utf8') as f:
            f.write(yaml_content)
    else:
        print("No YAML content found.")

# ---------------------------------------

artifact_output_dir=f"{output_dir}/planning_artifacts"

os.makedirs(artifact_output_dir, exist_ok=True)

context_lst = extract_planning(f'{output_dir}/planning_trajectories.json')

arch_design = content_to_json(context_lst[1])
logic_design = content_to_json(context_lst[2])

formatted_arch_design = format_json_data(arch_design)
formatted_logic_design = format_json_data(logic_design)

with open(f"{artifact_output_dir}/1.1_overall_plan.txt", "w", encoding="utf-8") as f:
    f.write(context_lst[0])

with open(f"{artifact_output_dir}/1.2_arch_design.txt", "w", encoding="utf-8") as f:
    f.write(formatted_arch_design)

with open(f"{artifact_output_dir}/1.3_logic_design.txt", "w", encoding="utf-8") as f:
    f.write(formatted_logic_design)

# ...existing code...
config_src = os.path.join(output_dir, "planning_config.yaml")
config_dst = os.path.join(artifact_output_dir, "1.4_config.yaml")

try:
    # ensure destination directory exists (optional)
    os.makedirs(os.path.dirname(config_dst), exist_ok=True)

    shutil.copy(config_src, config_dst)
except FileNotFoundError:
    print(f"No planning_config.yaml found at {config_src}; skipping copy.")
    # Optionally create an empty placeholder:
    # with open(config_dst, "w", encoding="utf8") as f:
    #     f.write("")
except PermissionError as e:
    print(f"Permission error copying {config_src} to {config_dst}: {e}")
except Exception as e:
    print(f"Unexpected error copying {config_src} to {config_dst}: {e}")

