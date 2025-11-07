import json
import re
import os
from datetime import datetime
import logging
import subprocess

logger = logging.getLogger(__name__)  # Define logger

def extract_planning(trajectories_json_file_path):
    with open(trajectories_json_file_path) as f:
        traj = json.load(f)

    context_lst = []
    for turn in traj:
        if turn['role'] == 'assistant':
            # context_lst.append(turn['content'])
            content = turn['content']
            if "</think>" in content:
                content = content.split("</think>")[-1].strip()
            context_lst.append(content)


    context_lst = context_lst[:3]

    return context_lst



def content_to_json(data):
    clean_data = re.sub(r'\[CONTENT\]|\[/CONTENT\]', '', data).strip()

    clean_data = re.sub(r'(".*?"),\s*#.*', r'\1,', clean_data)

    clean_data = re.sub(r',\s*\]', ']', clean_data)

    clean_data = re.sub(r'\n\s*', '', clean_data)


    # JSON parsing
    try:
        json_data = json.loads(clean_data)
        return json_data
    except json.JSONDecodeError as e:
        # print(e)
        return content_to_json2(data)


def content_to_json2(data):
    # remove [CONTENT][/CONTENT]
    clean_data = re.sub(r'\[CONTENT\]|\[/CONTENT\]', '', data).strip()

    # "~~~~", #comment -> "~~~~",
    clean_data = re.sub(r'(".*?"),\s*#.*', r'\1,', clean_data)

    # "~~~~" #comment → "~~~~"
    clean_data = re.sub(r'(".*?")\s*#.*', r'\1', clean_data)


    # ("~~~~",] -> "~~~~"])
    clean_data = re.sub(r',\s*\]', ']', clean_data)

    clean_data = re.sub(r'\n\s*', '', clean_data)

    # JSON parsing
    try:
        json_data = json.loads(clean_data)
        return json_data

    except json.JSONDecodeError as e:
        # print("Json parsing error", e)
        return content_to_json3(data)

def content_to_json3(data):
    # remove [CONTENT] [/CONTENT]
    clean_data = re.sub(r'\[CONTENT\]|\[/CONTENT\]', '', data).strip()

    # "~~~~", #comment -> "~~~~",
    clean_data = re.sub(r'(".*?"),\s*#.*', r'\1,', clean_data)

    # "~~~~" #comment → "~~~~"
    clean_data = re.sub(r'(".*?")\s*#.*', r'\1', clean_data)

    # remove ("~~~~",] -> "~~~~"])
    clean_data = re.sub(r',\s*\]', ']', clean_data)

    clean_data = re.sub(r'\n\s*', '', clean_data)
    clean_data = re.sub(r'"""', '"', clean_data)  # Replace triple double quotes
    clean_data = re.sub(r"'''", "'", clean_data)  # Replace triple single quotes
    clean_data = re.sub(r"\\", "'", clean_data)  # Replace \

    # JSON parsing
    try:
        json_data = json.loads(f"""{clean_data}""")
        return json_data

    except json.JSONDecodeError as e:
        # print(e)

        # print(f"[DEBUG] utils.py > content_to_json3 ")
        # return None
        return content_to_json4(data)

def content_to_json4(data):
    # 1. Extract Logic Analysis, Task list
    pattern = r'"Logic Analysis":\s*(\[[\s\S]*?\])\s*,\s*"Task list":\s*(\[[\s\S]*?\])'
    match = re.search(pattern, data)

    if match:
        logic_analysis = json.loads(match.group(1))
        task_list = json.loads(match.group(2))

        result = {
            "Logic Analysis": logic_analysis,
            "Task list": task_list
        }
    else:
        result = {}

    # print(json.dumps(result, indent=2))
    return result

def extract_code_from_content(content):
    pattern = r'^```(?:\w+)?\s*\n(.*?)(?=^```)```'
    code = re.findall(pattern, content, re.DOTALL | re.MULTILINE)
    if len(code) == 0:
        return ""
    else:
        return code[0]

def extract_code_from_content2(content):
    pattern = r'```python\s*(.*?)```'
    result = re.search(pattern, content, re.DOTALL)

    if result:
        extracted_code = result.group(1).strip()
    else:
        extracted_code = ""
        print("[WARNING] No Python code found.")
    return extracted_code

def format_json_data(data):
    formatted_text = ""
    for key, value in data.items():
        formatted_text += "-" * 40 + "\n"
        formatted_text += "[" + key + "]\n"
        if isinstance(value, list):
            for item in value:
                formatted_text += f"- {item}\n"
        else:
            formatted_text += str(value) + "\n"
        formatted_text += "\n"
    return formatted_text



def cal_cost(completion_json, gpt_version):
    """
    Calculate the cost based on the completion JSON and model version.
    """
    if completion_json is None or not hasattr(completion_json, 'usage'):
        logger.warning("Completion JSON or usage data is missing. Returning default usage info.")
        return {
            'model_name': gpt_version,
            'actual_input_tokens': 0,
            'cached_tokens': 0,
            'cached_input_cost': 0.0,
            'output_tokens': 0,
            'input_cost': 0.0,
            'output_cost': 0.0,
            'total_cost': 0.0
        }

    response_json = completion_json
    usage_info = response_json.usage

    # Use API schema fields
    total_input_tokens = getattr(usage_info, 'input_tokens', 0)
    total_output_tokens = getattr(usage_info, 'output_tokens', 0)
    cached_tokens = getattr(usage_info, 'input_cached_tokens', 0)
    actual_input_tokens = total_input_tokens - cached_tokens if total_input_tokens > cached_tokens else total_input_tokens
    input_audio_tokens = getattr(usage_info, 'input_audio_tokens', 0)
    output_audio_tokens = getattr(usage_info, 'output_audio_tokens', 0)

    # Cost calculation (example rates, adjust based on Groq pricing)
    input_cost_per_token = 0.0000005  # Adjust based on gpt-oss-120b pricing
    output_cost_per_token = 0.0000015  # Adjust based on gpt-oss-120b pricing
    cached_input_cost_per_token = 0.0000001  # Hypothetical cached rate
    audio_input_cost_per_token = 0.0000002  # Hypothetical audio rate
    audio_output_cost_per_token = 0.0000003  # Hypothetical audio rate

    input_cost = actual_input_tokens * input_cost_per_token
    cached_input_cost = cached_tokens * cached_input_cost_per_token
    output_cost = total_output_tokens * output_cost_per_token
    audio_input_cost = input_audio_tokens * audio_input_cost_per_token
    audio_output_cost = output_audio_tokens * audio_output_cost_per_token
    total_cost = input_cost + cached_input_cost + output_cost + audio_input_cost + audio_output_cost

    return {
        'model_name': gpt_version,
        'actual_input_tokens': actual_input_tokens,
        'cached_tokens': cached_tokens,
        'cached_input_cost': cached_input_cost,
        'output_tokens': total_output_tokens,
        'input_cost': input_cost,
        'output_cost': output_cost,
        'total_cost': total_cost,
        'audio_input_tokens': input_audio_tokens,
        'audio_output_tokens': output_audio_tokens,
        'audio_input_cost': audio_input_cost,
        'audio_output_cost': audio_output_cost
    }



def load_accumulated_cost(accumulated_cost_file):
    if os.path.exists(accumulated_cost_file):
        with open(accumulated_cost_file, "r", encoding="utf-8") as f:
            data = json.load(f)
            return data.get("total_cost", 0.0)
    else:
        return 0.0

def save_accumulated_cost(accumulated_cost_file, cost):
    with open(accumulated_cost_file, "w", encoding="utf-8") as f:
        json.dump({"total_cost": cost}, f)

def print_response(completion_json, is_llm=False):
    print("============================================")
    if is_llm:
        print(completion_json['text'])
    else:
        print(completion_json['choices'][0]['message']['content'])
    print("============================================\n")

def print_log_cost(completion_json, gpt_version, current_stage, output_dir, total_accumulated_cost):
    usage_info = cal_cost(completion_json, gpt_version)

    current_cost = usage_info['total_cost']
    total_accumulated_cost += current_cost

    output_lines = []
    output_lines.append("🌟 Usage Summary 🌟")
    output_lines.append(f"{current_stage}")
    output_lines.append(f"🛠️ Model: {usage_info['model_name']}")
    output_lines.append(f"📥 Input tokens: {usage_info['actual_input_tokens']} (Cost: ${usage_info['input_cost']:.8f})")
    output_lines.append(f"📦 Cached input tokens: {usage_info['cached_tokens']} (Cost: ${usage_info['cached_input_cost']:.8f})")
    output_lines.append(f"📤 Output tokens: {usage_info['output_tokens']} (Cost: ${usage_info['output_cost']:.8f})")
    output_lines.append(f"💵 Current total cost: ${current_cost:.8f}")
    output_lines.append(f"🪙 Accumulated total cost so far: ${total_accumulated_cost:.8f}")
    output_lines.append("============================================\n")

    output_text = "\n".join(output_lines)

    print(output_text)

    with open(f"{output_dir}/cost_info.log", "a", encoding="utf-8") as f:
        f.write(output_text + "\n")

    return total_accumulated_cost


def num_tokens_from_messages(messages, model="gpt-4o-2024-08-06"):
    import tiktoken

    """Return the number of tokens used by a list of messages."""
    try:
        encoding = tiktoken.encoding_for_model(model)
    except KeyError:
        print("Warning: model not found. Using o200k_base encoding.")
        encoding = tiktoken.get_encoding("o200k_base")
    if model in {
        "gpt-3.5-turbo-0125",
        "gpt-4-0314",
        "gpt-4-32k-0314",
        "gpt-4-0613",
        "gpt-4-32k-0613",
        "gpt-4o-mini-2024-07-18",
        "gpt-4o-2024-08-06"
        }:
        tokens_per_message = 3
        tokens_per_name = 1
    elif "gpt-3.5-turbo" in model:
        print("Warning: gpt-3.5-turbo may update over time. Returning num tokens assuming gpt-3.5-turbo-0125.")
        return num_tokens_from_messages(messages, model="gpt-3.5-turbo-0125")
    elif "gpt-4o-mini" in model:
        print("Warning: gpt-4o-mini may update over time. Returning num tokens assuming gpt-4o-mini-2024-07-18.")
        return num_tokens_from_messages(messages, model="gpt-4o-mini-2024-07-18")
    elif "gpt-4o" in model:
        print("Warning: gpt-4o and gpt-4o-mini may update over time. Returning num tokens assuming gpt-4o-2024-08-06.")
        return num_tokens_from_messages(messages, model="gpt-4o-2024-08-06")

    elif "gpt-4" in model:
        print("Warning: gpt-4 may update over time. Returning num tokens assuming gpt-4-0613.")
        return num_tokens_from_messages(messages, model="gpt-4-0613")
    else:
        raise NotImplementedError(
            f"""num_tokens_from_messages() is not implemented for model {model}."""
        )
    num_tokens = 0
    for message in messages:
        num_tokens += tokens_per_message
        for key, value in message.items():
            # num_tokens += len(encoding.encode(value)
            num_tokens += len(encoding.encode(value, allowed_special={"<|endoftext|>"},disallowed_special=()))

            if key == "name":
                num_tokens += tokens_per_name
    num_tokens += 3  # every reply is primed with <|start|>assistant<|message|>
    return num_tokens



def read_all_files(directory, allowed_ext, is_print=True):
    """Recursively read all .py files in the specified directory and return their contents."""
    all_files_content = {}

    for root, _, files in os.walk(directory):  # Recursively traverse directories
        for filename in files:
            relative_path = os.path.relpath(os.path.join(root, filename), directory)  # Preserve directory structure

            # print(f"fn: {filename}\tdirectory: {directory}")
            _file_name, ext = os.path.splitext(filename)

            is_skip = False
            if len(directory) < len(root):
                root2 = root[len(directory)+1:]
                for dirname in root2.split("/"):
                    if dirname.startswith("."):
                        is_skip = True
                        break

            if filename.startswith(".") or "requirements.txt" in filename or ext == "" or is_skip:
                if is_print and ext == "":
                    print(f"[SKIP] {os.path.join(root, filename)}")
                continue

            if ext not in allowed_ext:
                if _file_name.lower() != "readme":
                    if is_print:
                        print(f"[SKIP] {os.path.join(root, filename)}")
                    continue

            try:
                filepath = os.path.join(root, filename)
                file_size = os.path.getsize(filepath) # bytes

                if file_size > 204800: # > 200KB
                    print(f"[BIG] {filepath} {file_size}")

                with open(filepath, "r") as file: # encoding="utf-8"
                    all_files_content[relative_path] = file.read()
            except Exception as e:
                print(e)
                print(f"[SKIP] {os.path.join(root, filename)}")


    return all_files_content

def read_python_files(directory):
    """Recursively read all .py files in the specified directory and return their contents."""
    python_files_content = {}

    for root, _, files in os.walk(directory):  # Recursively traverse directories
        for filename in files:
            if filename.endswith(".py"):  # Check if file has .py extension
                relative_path = os.path.relpath(os.path.join(root, filename), directory)  # Preserve directory structure
                with open(os.path.join(root, filename), "r", encoding="utf-8") as file:
                    python_files_content[relative_path] = file.read()

    return python_files_content


def extract_json_from_string(text):
    # Extract content inside ```yaml\n...\n```
    match = re.search(r"```json\n(.*?)\n```", text, re.DOTALL)

    if match:
        yaml_content = match.group(1)
        return yaml_content
    else:
        print("No JSON content found.")
        return ""


def get_now_str():
    now = datetime.now()
    now = str(now)
    now = now.split(".")[0]
    now = now.replace("-","").replace(" ","_").replace(":","")
    return now # now - "20250427_205124"


def save_input_variable(output_dir: str, var_name: str, var_value: any):
    """
    Save a single input variable as a JSON file.
    """
    json_filename = f"{output_dir}/{var_name}.json"
    try:
        with open(json_filename, "w", encoding="utf-8") as f:
            json.dump(var_value, f, indent=4, default=str)
        print(f"Saved {var_name} to {json_filename}")
    except TypeError as e:
        print(
            f"Error loading {var_name} to JSON: {e}. Saving as string representation instead."
        )
        with open(json_filename, "w", encoding="utf-8") as f:
            json.dump(str(var_value), f, indent=4)


def save_artifacts(output_dir, trajectories, responses):
    """Save final planning-stage outputs for reproducibility."""
    os.makedirs(output_dir, exist_ok=True)
    with open(f"{output_dir}/planning_trajectories.json", "w") as f:
        json.dump(trajectories, f, indent=2)
    with open(f"{output_dir}/planning_response.json", "w") as f:
        json.dump(responses, f, indent=2)
    print(f"✅ Saved artifacts to {output_dir}")

def restore_artifacts_planning(output_dir, resume_stage_index):
    """Restore trajectories and responses from previous planning stage."""
    trajectories, responses = [], []
    if resume_stage_index > 0:
        try:
            with open(f"{output_dir}/planning_trajectories.json", "r") as f:
                trajectories = json.load(f)
            with open(f"{output_dir}/planning_response.json", "r") as f:
                responses = json.load(f)
            print(f"✅ Restored context from stage {resume_stage_index - 1}")
        except Exception as e:
            print(f"⚠️ Could not restore artifacts: {e}")
    return trajectories, responses

def restore_artifacts_analyzing(output_dir, resume_stage_index):
    """Restore trajectories and responses from previous planning stage."""
    trajectories, responses = [], []
    if resume_stage_index > 0:
        try:
            with open(f"{output_dir}/analyzing_trajectories.json", "r") as f:
                trajectories = json.load(f)
            with open(f"{output_dir}/analyzing_response.json", "r") as f:
                responses = json.load(f)
            print(f"✅ Restored context from stage {resume_stage_index - 1}")
        except Exception as e:
            print(f"⚠️ Could not restore artifacts: {e}")
    return trajectories, responses

def snapshot_to_kaggle(output_dir, dataset_slug=None, dataset_title=None, is_public=False):
    """Create or update a Kaggle dataset from the output_dir."""
    os.makedirs(output_dir, exist_ok=True)
    default_name = os.path.basename(os.path.normpath(output_dir)).replace(" ", "_")
    dataset_slug = dataset_slug or f"roger/{default_name.lower()}-planning"
    dataset_title = dataset_title or f"{default_name} Planning Outputs"

    metadata_path = os.path.join(output_dir, 'dataset-metadata.json')
    metadata = {
        "title": dataset_title,
        "id": dataset_slug,
        "licenses": [{"name": "CC0-1.0"}],
        "isPrivate": not is_public
    }

    with open(metadata_path, "w") as f:
        json.dump(metadata, f, indent=2)

    try:
        subprocess.run(["kaggle", "datasets", "create", "-p", output_dir], check=True)
        print(f"✅ Created Kaggle dataset: {dataset_slug}")
    except subprocess.CalledProcessError:
        subprocess.run(["kaggle", "datasets", "version", "-p", output_dir, "-m", "Updated planning outputs"], check=True)
        print(f"🔁 Updated Kaggle dataset: {dataset_slug}")

def load_from_kaggle(dataset_path, output_dir):
    """Load planning artifacts from a Kaggle dataset path."""
    try:
        with open(f"{dataset_path}/planning_trajectories.json", "r") as f:
            trajectories = json.load(f)
            shutil.copy(f.name, output_dir)
        with open(f"{dataset_path}/planning_response.json", "r") as f:
            responses = json.load(f)
            shutil.copy(f.name, output_dir)
        print(f"✅ Loaded artifacts from Kaggle dataset: {dataset_path}")
        return trajectories, responses
    except Exception as e:
        print(f"⚠️ Failed to load from Kaggle dataset: {e}")
        return [], []