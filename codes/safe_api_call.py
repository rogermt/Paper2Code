from openai import OpenAI
import time
import logging
import tiktoken
from typing import List, Dict, Any
from openai.types.chat import ChatCompletion
from tokenizers import Tokenizer

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger(__name__)


def get_tokenizer_from_json(path: str) -> Tokenizer:
    try:
        tokenizer = Tokenizer.from_file(path)
        logger.info(f"Loaded tokenizer from {path}")
        return tokenizer
    except Exception as e:
        logger.error(f"Failed to load tokenizer from {path}: {e}")
        raise


def get_tokenizer(model: str):
    try:
        if model == "gpt-oss-120b":
            return tiktoken.get_encoding("o200k_base")
        elif model == "gpt-oss-20b":
            tokenizer_path = f"/kaggle/working/tokenizers/{model}/tokenizer.json"
            return get_tokenizer_from_json(tokenizer_path)
        elif model.startswith(("gpt-3.5", "gpt-4", "openai/")):
            return tiktoken.encoding_for_model(model)
        else:
            tokenizer_path = f"/kaggle/working/tokenizers/{model}/tokenizer.json"
            return get_tokenizer_from_json(tokenizer_path)
    except Exception as e:
        logger.warning(f"Tokenizer fallback triggered for model '{model}': {e}")
        return tiktoken.get_encoding("cl100k_base")


def is_hf_tokenizer(tokenizer) -> bool:
    try:
        return hasattr(tokenizer.encode("test"), "ids")
    except Exception:
        return False


def count_tokens(messages: List[Dict[str, str]], model: str) -> int:
    tokenizer = get_tokenizer(model)
    hf_mode = is_hf_tokenizer(tokenizer)
    total_tokens = 0
    for message in messages:
        content = message.get("content", "")
        token_ids = tokenizer.encode(content).ids if hf_mode else tokenizer.encode(content)
        total_tokens += len(token_ids)
    return total_tokens


def chunk_messages(messages: List[Dict[str, str]], max_tokens: int, model: str):
    tokenizer = get_tokenizer(model)
    hf_mode = is_hf_tokenizer(tokenizer)
    chunks = []
    current_chunk = []
    current_tokens = 0

    for msg in messages:
        content = msg.get("content", "")
        token_ids = tokenizer.encode(content).ids if hf_mode else tokenizer.encode(content)
        msg_tokens = len(token_ids)
        
        if msg_tokens > max_tokens:
            for i in range(0, len(token_ids), max_tokens):
                chunk_ids = token_ids[i : i + max_tokens]
                chunk_content = tokenizer.decode(chunk_ids)
                split_msg = {"role": msg["role"], "content": chunk_content}
                split_tokens = len(
                    tokenizer.encode(chunk_content).ids if hf_mode else tokenizer.encode(chunk_content)
                )
                if current_tokens + split_tokens > max_tokens and current_chunk:
                    chunks.append(current_chunk)
                    current_chunk = []
                    current_tokens = 0
                current_chunk.append(split_msg)
                current_tokens += split_tokens
        else:
            if current_tokens + msg_tokens > max_tokens and current_chunk:
                chunks.append(current_chunk)
                current_chunk = []
                current_tokens = 0
            current_chunk.append(msg)
            current_tokens += msg_tokens
    if current_chunk:
        chunks.append(current_chunk)
    return chunks


def aggregate_usage(completions, logger):
    if not completions:
        raise Exception("No completions to aggregate")
    
    total_prompt_tokens = sum(c.usage.prompt_tokens for c in completions if hasattr(c, "usage") and c.usage)
    total_completion_tokens = sum(c.usage.completion_tokens for c in completions if hasattr(c, "usage") and c.usage)
    total_total_tokens = sum(c.usage.total_tokens for c in completions if hasattr(c, "usage") and c.usage)

    total_cached_tokens = 0
    total_audio_input_tokens = 0
    total_audio_output_tokens = 0
    total_reasoning_tokens = 0

    for c in completions:
        if hasattr(c, "usage") and c.usage:
            if hasattr(c.usage, "prompt_tokens_details") and c.usage.prompt_tokens_details:
                if getattr(c.usage.prompt_tokens_details, "cached_tokens", 0):
                    total_cached_tokens += c.usage.prompt_tokens_details.cached_tokens
                if getattr(c.usage.prompt_tokens_details, "audio_tokens", 0):
                    total_audio_input_tokens += c.usage.prompt_tokens_details.audio_tokens
            if hasattr(c.usage, "completion_tokens_details") and c.usage.completion_tokens_details:
                if getattr(c.usage.completion_tokens_details, "audio_tokens", 0):
                    total_audio_output_tokens += c.usage.completion_tokens_details.audio_tokens
                if getattr(c.usage.completion_tokens_details, "reasoning_tokens", 0):
                    total_reasoning_tokens += c.usage.completion_tokens_details.reasoning_tokens

    usage_data = {
        "prompt_tokens": total_prompt_tokens,
        "completion_tokens": total_completion_tokens,
        "total_tokens": total_total_tokens,
    }

    if total_cached_tokens > 0 or total_audio_input_tokens > 0:
        from openai.types.completion_usage import PromptTokensDetails
        usage_data["prompt_tokens_details"] = PromptTokensDetails(
            cached_tokens=total_cached_tokens or None,
            audio_tokens=total_audio_input_tokens or None,
        )
    if total_audio_output_tokens > 0 or total_reasoning_tokens > 0:
        from openai.types.completion_usage import CompletionTokensDetails
        usage_data["completion_tokens_details"] = CompletionTokensDetails(
            audio_tokens=total_audio_output_tokens or None,
            reasoning_tokens=total_reasoning_tokens or None,
        )

    try:
        from openai.types.completion_usage import CompletionUsage
        return CompletionUsage(**usage_data)
    except ImportError:
        class SimpleUsage:
            def __init__(self, **kwargs):
                for k, v in kwargs.items():
                    setattr(self, k, v)
        return SimpleUsage(**usage_data)


def safe_api_call(
    client: OpenAI,
    messages: List[Dict[str, str]],
    model: str,
    max_context: int = 32768,
    max_output_tokens: int = 4096,
    reasoning_tokens: int = 1500,
    safety_margin: float = 0.1,
    min_input_tokens: int = 2000,
    max_retries: int = 3,
) -> ChatCompletion:
    reserved_tokens = max_output_tokens + reasoning_tokens
    reserved_tokens = int(reserved_tokens * (1 + safety_margin))
    max_input_tokens = max_context - reserved_tokens
    effective_token_limit = max(max_input_tokens, min_input_tokens)

    total_tokens = count_tokens(messages, model)
    logger.info(f"[safe_api_call] Model '{model}' | max_context={max_context} | input limit={effective_token_limit} | actual={total_tokens}")

    if total_tokens <= effective_token_limit:
        return client.chat.completions.create(
            model=model,
            messages=messages,
            max_tokens=max_output_tokens
        )

    logger.warning(f"Input too large ({total_tokens} > {effective_token_limit}). Chunking...")
    completions = []
    for i, chunk in enumerate(chunk_messages(messages, effective_token_limit, model), start=1):
        for attempt in range(max_retries):
            try:
                resp = client.chat.completions.create(
                    model=model,
                    messages=chunk,
                    max_tokens=max_output_tokens
                )
                completions.append(resp)
                break
            except Exception as e:
                logger.warning(f"Chunk {i}, attempt {attempt+1}/{max_retries} failed: {e}")
                if attempt == max_retries - 1:
                    raise
                time.sleep(2**attempt)

    merged_content = "\n".join(c.choices[0].message.content for c in completions if c.choices)
    completions[0].choices[0].message.content = merged_content
    completions[0].usage = aggregate_usage(completions, logger)
    return completions[0]


def safe_api_call_prompt(
    client,
    prompt: str,
    model: str,
    max_context: int = 32768,
    max_output_tokens: int = 4096,
    reasoning_tokens: int = 1500,
    safety_margin: float = 0.1,
    min_input_tokens: int = 2000,
    max_retries: int = 3,
):
    tokenizer = get_tokenizer(model)
    hf_mode = is_hf_tokenizer(tokenizer)
    token_ids = tokenizer.encode(prompt).ids if hf_mode else tokenizer.encode(prompt)
    total_tokens = len(token_ids)

    reserved_tokens = max_output_tokens + reasoning_tokens
    reserved_tokens = int(reserved_tokens * (1 + safety_margin))
    max_input_tokens = max_context - reserved_tokens
    effective_token_limit = max(max_input_tokens, min_input_tokens)

    logger.info(f"[safe_api_call_prompt] Model '{model}' | max_context={max_context} | input limit={effective_token_limit} | actual={total_tokens}")

    if total_tokens <= effective_token_limit:
        return client.chat.completions.create(
            model=model,
            messages=[{"role": "user", "content": prompt}],
            max_tokens=max_output_tokens
        )

    logger.warning(f"Prompt too large ({total_tokens} > {effective_token_limit}). Chunking...")
    completions = []
    for i in range(0, total_tokens, effective_token_limit):
        chunk_ids = token_ids[i:i+effective_token_limit]
        chunk_text = tokenizer.decode(chunk_ids)
        for attempt in range(max_retries):
            try:
                resp = client.chat.completions.create(
                    model=model,
                    messages=[{"role": "user", "content": chunk_text}],
                    max_tokens=max_output_tokens
                )
                completions.append(resp)
                break
            except Exception as e:
                logger.warning(f"Prompt chunk {i//effective_token_limit + 1}, attempt {attempt+1}/{max_retries} failed: {e}")
                if attempt == max_retries - 1:
                    raise
                time.sleep(2**attempt)

    merged_text = "\n".join(c.choices[0].message.content for c in completions if c.choices)
    completions[0].choices[0].message.content = merged_text
    completions[0].usage = aggregate_usage(completions, logger)
    return completions[0]