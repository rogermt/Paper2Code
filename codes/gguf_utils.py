def qwen3_gguf_format(messages):
    """Format messages for Qwen-3 GGUF multi-turn chat."""
    prompt_parts = ["<|im_start|>system\nYou are a helpful assistant."]
    for m in messages:
        if m["role"] == "system":
            prompt_parts[0] = f"<|im_start|>system\n{m['content']}"
        elif m["role"] == "user":
            prompt_parts.append(f"<|im_start|>user\n{m['content']}<|im_end|>")
        elif m["role"] == "assistant":
            prompt_parts.append(f"<|im_start|>assistant\n{m['content']}<|im_end|>")
    prompt_parts.append("<|im_start|>assistant\n")
    return "\n".join(prompt_parts)

def llama2_gguf_format(messages):
    """Format messages for LLaMA-2 GGUF multi-turn chat."""
    prompt_parts = ["<s>[INST] <<SYS>>\nYou are a helpful assistant.\n<</SYS>>\n"]
    for m in messages:
        if m["role"] == "system":
            prompt_parts[0] = f"<s>[INST] <<SYS>>\n{m['content']}\n<</SYS>>\n"
        elif m["role"] == "user":
            prompt_parts.append(f"{m['content']} [/INST]")
        elif m["role"] == "assistant":
            prompt_parts.append(f" {m['content']} </s><s>[INST] ")
    return "".join(prompt_parts)

def mistral_gguf_format(messages):
    """Format messages for Mistral GGUF multi-turn chat."""
    prompt_parts = []
    for m in messages:
        if m["role"] == "system":
            prompt_parts.append(f"[INST] {m['content']} [/INST]")
        elif m["role"] == "user":
            prompt_parts.append(f"[INST] {m['content']} [/INST]")
        elif m["role"] == "assistant":
            prompt_parts.append(m["content"])
    return "\n".join(prompt_parts)

def format_for_backend(messages, model_name):
    """
    Return either messages (cloud) or a flattened prompt (GGUF).
    Auto-detects Qwen, LLaMA-2, Mistral GGUF models.
    """
    lower_name = model_name.lower()
    if "gguf" not in lower_name:
        return messages  # Cloud/HF path

    if "qwen" in lower_name:
        return qwen3_gguf_format(messages)
    if "llama" in lower_name:
        return llama2_gguf_format(messages)
    if "mistral" in lower_name:
        return mistral_gguf_format(messages)

    # Fallback: just join contents
    return "\n".join(m["content"] for m in messages)