"""
PURE DSPY EVALUATION SCRIPT - USING PYDANTIC LISTS
Uses DSPy's native support for list[PydanticModel] - no JSON parsing, no hardcoded fields
Paper+Code chunking: properly chunks paper and code together, ensuring each chunk < max_context
"""
import os
import sys
import argparse
import time
import json
import re
import statistics
import traceback
from collections import defaultdict
from typing import Optional, List, Dict, Tuple, Any
from dataclasses import dataclass, asdict
from datetime import datetime

import dspy
import tiktoken
from pydantic import BaseModel, Field, ValidationError

from tracking import ExperimentTracker, WandbWeaveTracker
from utils import (
    content_to_json,
    extract_code_from_content2,
    extract_planning,
    read_all_files,
    read_python_files,
)


class CritiqueItemRefFree(BaseModel):
    """Single critique item for ref_free mode - Pydantic model for DSPy."""
    file_name: str = Field(..., description="Name of the file with the issue")
    func_name: str = Field(..., description="Function or class name (or 'global' if not applicable)")
    severity_level: str = Field(..., description="Severity: high, medium, or low")
    critique: str = Field(..., description="Specific critique description (1-2 sentences)")

class CritiqueItemRefBased(BaseModel):
    """Single critique item for ref_based mode - Pydantic model for DSPy."""
    gold_file_name: str = Field(..., description="Name of the gold repository file")
    gold_func_name: str = Field(..., description="Function or class name in gold repository (or 'global' if not applicable)")
    target_file_name: str = Field(..., description="Name of the target repository file with the issue")
    target_func_name: str = Field(..., description="Function or class name in target repository (or 'global' if not applicable)")
    severity_level: str = Field(..., description="Severity: high, medium, or low")
    critique: str = Field(..., description="Specific critique description comparing target to gold (1-2 sentences)")

class EvaluationConfig(BaseModel):
    """Configuration."""
    paper_name: str
    pdf_json_path: str
    data_dir: str = "../data"
    output_dir: str
    target_repo_dir: str
    eval_result_dir: str
    eval_type: str = "ref_free"
    generated_n: int = Field(8, ge=1)
    gpt_version: str = "qwen3-8b-GGUF-q8_0_local"
    papercoder: bool = False
    max_context: int = 40960
    max_output_tokens: int = 4096
    delay: float = Field(0.0, ge=0)
    temperature: float = Field(1.0, ge=0.0, le=2.0)
    wandb_run_id: Optional[str] = None
    gold_repo_dir: str = ""
    selected_file_path: Optional[str] = None
    max_workers: int = Field(3, ge=1)
    max_retries: int = Field(3, ge=0)
    retry_delay: float = Field(1.0, ge=0)
    cache_dir: Optional[str] = Field(None)
    reasoning_tokens: int = Field(1500)
    safety_margin: float = Field(0.1, ge=0.0, le=0.3)
    input_json_type: str = Field(default="standard", 
        description="Either `standard` or `dolphin-ocr`"
    )


@dataclass
class ChunkEvaluationResult:
    """Result from evaluating a single chunk."""
    chunk_index: int
    score: int
    critique_items: List[Any]  # Can be either CritiqueItemRefFree or CritiqueItemRefBased
    reasoning: str
    input_tokens: int
    output_tokens: int
    raw_response: str
    success: bool = True
    error: Optional[str] = None

@dataclass
class GenerationResult:
    """Result from a single generation (one of n runs)."""
    generation_index: int
    score: int
    critique_list: List[Any]  # Can be either CritiqueItemRefFree or CritiqueItemRefBased
    input_tokens: int
    output_tokens: int
    chunk_count: int
    success: bool = True
    error: Optional[str] = None

@dataclass
class EvaluationCheckpoint:
    """Checkpoint matching Paper2Code format."""
    paper_name: str
    target_repo_dir: str
    eval_type: str
    gold_repo_dir: str
    generated_n: int
    run_index: int
    valid: bool
    score: int
    critique_list: List[Dict]
    nudge: str
    temperature: float
    input_tokens: int
    output_tokens: int
    chunked: bool
    chunk_count: int


class EvaluateChunkRefFree(dspy.Signature):
    """Evaluate code chunk against research paper methodology.
    
    Provide a score (1-5) and a list of specific critiques.
    Each critique must include: file_name, func_name, severity_level (high/medium/low), and critique text.
    Severity levels:
    - high: Core concepts, algorithms, or fundamental components missing/incorrect
    - medium: Training logic, preprocessing, or significant functionality issues
    - low: Minor errors, suboptimal choices, or non-critical issues
    
    IMPORTANT: Score must be single integer = (1, 2, 3, 4, or 5).
    """
    
    paper: str = dspy.InputField(desc="Research paper content")
    code_chunk: str = dspy.InputField(desc="Code implementation to evaluate")
    evaluation_focus: str = dspy.InputField(desc="Specific evaluation focus")
    chunk_info: str = dspy.InputField(desc="Chunk context information")
    
    critique_list: list[CritiqueItemRefFree] = dspy.OutputField(
        desc="List of specific critiques (3-10 items recommended)"
    )
    score: int = dspy.OutputField(desc="Overall quality score from (1, 2, 3, 4, or 5) ")

class EvaluateChunkRefBased(dspy.Signature):
    """You will be given a research paper along with two corresponding code repositories: a gold repository and a target repository.

Your task is to compare the target repository against the gold repository, rate the target repository on one metric, and provide a critique highlighting key differences.

Please make sure you read and understand these instructions carefully. Keep this document open while reviewing, and refer to it as needed.

---

Evaluation Criteria:

Correctness (1-5): The quality of the target repository in accurately implementing the paper's concepts, methodology, and algorithms without logical errors, as compared to the gold repository. Additionally, provide a critique focusing on the completeness, accuracy, and implementation choices made in the target repository relative to the gold repository.

1: Very Poor. The target repository does not correctly implement the core concepts, methodology, or algorithms from the paper. Major logical errors or missing components are present, especially when compared to the gold repository.
2: Poor. The target repository attempts to implement the paper's concepts but contains significant mistakes or missing components, making the implementation incorrect when compared to the gold repository.
3: Fair. Some core components and concepts are correctly implemented in the target repository, but there are notable logical errors or inaccuracies compared to the gold repository.
4: Good. The target repository correctly implements the key components and methodology, with only minor inaccuracies or deviations from the gold repository.
5: Excellent. The target repository fully and accurately implements all relevant key components, methodology, and algorithms from the paper, matching the quality of the gold repository.

---

Evaluation Steps  

1. Identify Key Aspects of the Paper: Carefully read the research paper to understand its core concepts, methodology, and algorithms. Pay close attention to the key aspects that are crucial for implementing the paper's results (e.g., specific algorithms, data preprocessing steps, evaluation protocols).

2. Analyze the Gold Repository: Examine the gold repository to understand how these key aspects have been implemented. Use the gold repository as a reference for how the paper's methodology should be translated into code. Note the completeness, accuracy, and design choices in the gold repository that faithfully represent the paper's concepts.

3. Examine the Target Repository: Analyze the target repository to assess how well it implements the key aspects of the paper. Reference the gold repository as a guide for understanding these key aspects in the target repository. Focus on whether the target repository's core logic, algorithms, and structure align with the methodology and experiments described in the paper.

4. Identify Logical Errors and Deviations: Check for logical errors, missing steps, or deviations from the paper's methodology. Note any incorrect representations, inconsistencies, or incomplete implementations that could affect the correctness of the target repository.

5. Provide a Critique: Consider both the completeness and accuracy of the implementation relative to the paper's goals and the gold repository's standard. You do not need to analyze minor details like logging functions, script organization, or documentation quality. Instead, concentrate on the correctness of the logic and implementation that ensures the core concepts from the paper are fully reflected in the target repository. For each mismatch or deviation in implementation, note down specific critiques comparing relevant functions in the target repository to the corresponding functions in the gold repository. Highlight incorrect logic, missing steps, or deviations that affect the correct implementation of the paper's methodology.

5. Assess the Correctness: Determine whether the target repository includes all the critical elements described in the paper and implemented in the gold repository. Identify missing components, significant deviations, or incorrect implementations that could affect the correctness of the target repository.

6. Assign a Score: Based on your evaluation, provide a critique and assign a correctness score from 1 to 5 for the target repository, reflecting how well it implements the key aspects of the paper refer to the gold repository. Include a detailed critique in the specified JSON format.


---

Severity Level:  

Each identified critique will be assigned a severity level based on its impact on the correctness of the methodology implementation.  

- High: Missing or incorrect implementation of the paper's core concepts, major loss functions, or experiment components that are fundamental to reproducing the paper's methodology.  
  - Example: The main algorithm is missing or fundamentally incorrect.  
- Medium: Issues affecting training logic, data preprocessing, or other core functionalities that significantly impact performance but do not completely break the system.  
  - Example: Improper training loop structure, incorrect data augmentation, or missing essential components in data processing.  
- Low: Errors in specific features that cause deviations from expected results but can be worked around with modifications. Any errors in the evaluation process belong to this category unless they impact the core concepts. These include minor issues like logging, error handling mechanisms, configuration settings, evaluation steps that do not alter the fundamental implementation and additional implementations not explicitly stated in the paper.
  - Example: Suboptimal hyperparameter initialization, incorrect learning rate schedule, inaccuracies in evaluation metrics, using a different random seed, variations in batch processing, different weight initialization, issues in result logging or reporting, variations in evaluation dataset splits, improper error handling in non-critical steps, mismatches in secondary evaluation criteria, or additional implementation details not specified in the paper that do not interfere with core results.

---
    """
    
    paper: str = dspy.InputField(desc="Research paper content")
    gold_code: str = dspy.InputField(desc="Gold reference code repository")
    target_code: str = dspy.InputField(desc="Target code repository to evaluate")
    evaluation_focus: str = dspy.InputField(desc="Specific evaluation focus")
    chunk_info: str = dspy.InputField(desc="Chunk context information")
    
    critique_list: list[CritiqueItemRefBased] = dspy.OutputField(
        desc="List of specific critiques comparing target to gold (3-10 items recommended)"
    )
    score: int = dspy.OutputField(desc="Overall quality score from (1, 2, 3, 4, or 5) ")



def clean_score(raw: Any) -> int:
    """Extract and validate score 1-5."""
    try:
        if isinstance(raw, int):
            return max(1, min(raw, 5))
        match = re.search(r'\b([1-5])\b', str(raw))
        return int(match.group(1)) if match else 3
    except:
        return 3

def validate_critique_item_ref_free(item: CritiqueItemRefFree) -> bool:
    """Validate and clean critique item for ref_free mode."""
    if not item.file_name or item.file_name.lower() in ['none', 'n/a', '']:
        return False
    if not item.critique or item.critique.lower() in ['none', 'n/a', '']:
        return False
    
    # Normalize severity
    severity = item.severity_level.lower().strip()
    if 'high' in severity:
        item.severity_level = 'high'
    elif 'medium' in severity or 'med' in severity:
        item.severity_level = 'medium'
    elif 'low' in severity:
        item.severity_level = 'low'
    else:
        item.severity_level = 'medium'
    
    # Ensure func_name exists
    if not item.func_name or item.func_name.lower() in ['none', 'n/a', '']:
        item.func_name = 'unknown'
    
    # Truncate long fields
    item.file_name = item.file_name[:100]
    item.func_name = item.func_name[:100]
    item.critique = item.critique[:500]
    
    return True

def validate_critique_item_ref_based(item: CritiqueItemRefBased) -> bool:
    """Validate and clean critique item for ref_based mode."""
    if not item.target_file_name or item.target_file_name.lower() in ['none', 'n/a', '']:
        return False
    if not item.critique or item.critique.lower() in ['none', 'n/a', '']:
        return False
    
    # Normalize severity
    severity = item.severity_level.lower().strip()
    if 'high' in severity:
        item.severity_level = 'high'
    elif 'medium' in severity or 'med' in severity:
        item.severity_level = 'medium'
    elif 'low' in severity:
        item.severity_level = 'low'
    else:
        item.severity_level = 'medium'
    
    # Ensure func_names exist
    if not item.gold_file_name or item.gold_file_name.lower() in ['none', 'n/a', '']:
        item.gold_file_name = 'unknown'
    if not item.gold_func_name or item.gold_func_name.lower() in ['none', 'n/a', '']:
        item.gold_func_name = 'unknown'
    if not item.target_func_name or item.target_func_name.lower() in ['none', 'n/a', '']:
        item.target_func_name = 'unknown'
    
    # Truncate long fields
    item.gold_file_name = item.gold_file_name[:100]
    item.gold_func_name = item.gold_func_name[:100]
    item.target_file_name = item.target_file_name[:100]
    item.target_func_name = item.target_func_name[:100]
    item.critique = item.critique[:500]
    
    return True

def get_raw_response_content() -> str:
    """Get raw LM response."""
    try:
        history = dspy.settings.lm.history[-1] if dspy.settings.lm.history else None
        if not history:
            return ""
        
        if isinstance(history, dict):
            resp = history.get('response', '')
            if hasattr(resp, 'choices') and len(resp.choices) > 0:
                return resp.choices[0].message.content
            if isinstance(resp, dict):
                choices = resp.get('choices', [])
                if choices:
                    return choices[0].get('message', {}).get('content', '')
            return str(resp)
        
        if hasattr(history, 'choices') and len(history.choices) > 0:
            return history.choices[0].message.content
        
        return str(history)
    except Exception as e:
        print(f"[WARN] Failed to get raw response: {e}")
        return ""

def merge_critique_lists_ref_free(lists: List[List[CritiqueItemRefFree]], max_items: int = 50) -> List[CritiqueItemRefFree]:
    """Merge and deduplicate critique lists for ref_free mode."""
    grouped = defaultdict(list)
    
    for critique_list in lists:
        for item in critique_list:
            key = (item.file_name, item.func_name)
            grouped[key].append(item)
    
    merged = []
    severity_order = {'high': 0, 'medium': 1, 'low': 2}
    
    for key, items in grouped.items():
        # Take highest severity
        best = min(items, key=lambda x: severity_order.get(x.severity_level, 3))
        
        # Merge critique texts if duplicates
        if len(items) > 1:
            unique_texts = list(set(i.critique for i in items))
            best.critique = "; ".join(unique_texts)[:500]
        
        merged.append(best)
    
    # Sort by severity
    merged.sort(key=lambda x: severity_order.get(x.severity_level, 3))
    
    return merged[:max_items]

def merge_critique_lists_ref_based(lists: List[List[CritiqueItemRefBased]], max_items: int = 50) -> List[CritiqueItemRefBased]:
    """Merge and deduplicate critique lists for ref_based mode."""
    grouped = defaultdict(list)
    
    for critique_list in lists:
        for item in critique_list:
            key = (item.gold_file_name, item.gold_func_name, item.target_file_name, item.target_func_name)
            grouped[key].append(item)
    
    merged = []
    severity_order = {'high': 0, 'medium': 1, 'low': 2}
    
    for key, items in grouped.items():
        # Take highest severity
        best = min(items, key=lambda x: severity_order.get(x.severity_level, 3))
        
        # Merge critique texts if duplicates
        if len(items) > 1:
            unique_texts = list(set(i.critique for i in items))
            best.critique = "; ".join(unique_texts)[:500]
        
        merged.append(best)
    
    # Sort by severity
    merged.sort(key=lambda x: severity_order.get(x.severity_level, 3))
    
    return merged[:max_items]


class ChunkEvaluatorRefFree(dspy.Module):
    """Evaluate individual chunks using Pydantic list output for ref_free mode."""
    
    def __init__(self, config: EvaluationConfig):
        super().__init__()
        self.evaluator = dspy.ChainOfThought(EvaluateChunkRefFree)
        self.config = config
        self.tokenizer = tiktoken.get_encoding("cl100k_base")
    
    def forward(self, paper: str, code_chunk: str, 
                nudge: str, chunk_index: int, total_chunks: int) -> ChunkEvaluationResult:
        """Evaluate single chunk for ref_free mode."""
        
        for attempt in range(self.config.max_retries + 1):
            try:
                chunk_info = f"Chunk {chunk_index + 1}/{total_chunks}"
                
                response = self.evaluator(
                    paper=paper,
                    code_chunk=code_chunk,
                    evaluation_focus=nudge,
                    chunk_info=chunk_info
                )
                
                raw_response = get_raw_response_content()
                reasoning = getattr(response, 'rationale', '') or getattr(response, 'reasoning', '')
                
                # Get critique_list directly from response (DSPy handles parsing!)
                critique_list = response.critique_list if hasattr(response, 'critique_list') else []
                
                # Handle None case (when DSPy fails to parse)
                if critique_list is None:
                    critique_list = []
                
                # Validate and clean critiques
                valid_critiques = []
                for item in critique_list:
                    if isinstance(item, CritiqueItemRefFree) and validate_critique_item_ref_free(item):
                        valid_critiques.append(item)
                
                # Get score
                score = clean_score(getattr(response, 'score', 3))
                
                # Calculate tokens
                input_text = paper + code_chunk + nudge
                input_tokens = len(self.tokenizer.encode(input_text, disallowed_special=()))
                output_tokens = len(self.tokenizer.encode(raw_response, disallowed_special=()))
                
                print(f"[DEBUG] Chunk {chunk_index}: score={score}, critiques={len(valid_critiques)}")
                
                return ChunkEvaluationResult(
                    chunk_index=chunk_index,
                    score=score,
                    critique_items=valid_critiques,
                    reasoning=reasoning[:500] if reasoning else "",
                    input_tokens=input_tokens,
                    output_tokens=output_tokens,
                    raw_response=raw_response,
                    success=True
                )
                
            except Exception as e:
                if attempt < self.config.max_retries:
                    wait_time = self.config.retry_delay * (2 ** attempt)
                    print(f"[RETRY] Chunk {chunk_index} attempt {attempt+1} failed: {e}. Retrying in {wait_time}s...")
                    time.sleep(wait_time)
                else:
                    print(f"[ERROR] Chunk {chunk_index} failed after {self.config.max_retries+1} attempts")
                    traceback.print_exc()
                    
                    return ChunkEvaluationResult(
                        chunk_index=chunk_index,
                        score=1,
                        critique_items=[],
                        reasoning="",
                        input_tokens=0,
                        output_tokens=0,
                        raw_response="",
                        success=False,
                        error=str(e)
                    )

class ChunkEvaluatorRefBased(dspy.Module):
    """Evaluate individual chunks using Pydantic list output for ref_based mode."""
    
    def __init__(self, config: EvaluationConfig):
        super().__init__()
        self.evaluator = dspy.ChainOfThought(EvaluateChunkRefBased)
        self.config = config
        self.tokenizer = tiktoken.get_encoding("cl100k_base")
    
    def forward(self, paper: str, gold_code: str, target_code: str,
                nudge: str, chunk_index: int, total_chunks: int) -> ChunkEvaluationResult:
        """Evaluate single chunk for ref_based mode."""
        
        for attempt in range(self.config.max_retries + 1):
            try:
                chunk_info = f"Chunk {chunk_index + 1}/{total_chunks}"
                
                response = self.evaluator(
                    paper=paper,
                    gold_code=gold_code,
                    target_code=target_code,
                    evaluation_focus=nudge,
                    chunk_info=chunk_info
                )
                
                raw_response = get_raw_response_content()
                reasoning = getattr(response, 'rationale', '') or getattr(response, 'reasoning', '')
                
                # Get critique_list directly from response (DSPy handles parsing!)
                critique_list = response.critique_list if hasattr(response, 'critique_list') else []
                
                # Handle None case (when DSPy fails to parse)
                if critique_list is None:
                    critique_list = []
                
                # Validate and clean critiques
                valid_critiques = []
                for item in critique_list:
                    if isinstance(item, CritiqueItemRefBased) and validate_critique_item_ref_based(item):
                        valid_critiques.append(item)
                
                # Get score
                score = clean_score(getattr(response, 'score', 3))
                
                # Calculate tokens
                input_text = paper + gold_code + target_code + nudge
                input_tokens = len(self.tokenizer.encode(input_text, disallowed_special=()))
                output_tokens = len(self.tokenizer.encode(raw_response, disallowed_special=()))
                
                print(f"[DEBUG] Chunk {chunk_index}: score={score}, critiques={len(valid_critiques)}")
                
                return ChunkEvaluationResult(
                    chunk_index=chunk_index,
                    score=score,
                    critique_items=valid_critiques,
                    reasoning=reasoning[:500] if reasoning else "",
                    input_tokens=input_tokens,
                    output_tokens=output_tokens,
                    raw_response=raw_response,
                    success=True
                )
                
            except Exception as e:
                if attempt < self.config.max_retries:
                    wait_time = self.config.retry_delay * (2 ** attempt)
                    print(f"[RETRY] Chunk {chunk_index} attempt {attempt+1} failed: {e}. Retrying in {wait_time}s...")
                    time.sleep(wait_time)
                else:
                    print(f"[ERROR] Chunk {chunk_index} failed after {self.config.max_retries+1} attempts")
                    traceback.print_exc()
                    
                    return ChunkEvaluationResult(
                        chunk_index=chunk_index,
                        score=1,
                        critique_items=[],
                        reasoning="",
                        input_tokens=0,
                        output_tokens=0,
                        raw_response="",
                        success=False,
                        error=str(e)
                    )

class RepoEvaluator(dspy.Module):
    """Evaluate repository via chunking - handles both ref_free and ref_based modes."""
    
    def __init__(self, config: EvaluationConfig):
        super().__init__()
        self.config = config
        if config.eval_type == "ref_free":
            self.chunk_evaluator = ChunkEvaluatorRefFree(config)
        else:  # ref_based
            self.chunk_evaluator = ChunkEvaluatorRefBased(config)

    def forward(self, paper_chunks: List[str], code_chunks: List[str], gold_code_string: str,
                nudge: str) -> GenerationResult:
        """Evaluate all chunks and aggregate.
        
        For ref_based: gold_code_string is the ENTIRE gold repository (same for all chunks).
        For ref_free: gold_code_string is empty.
        """
        
        # Handle case with no chunks (no code files)
        if not paper_chunks:
            return GenerationResult(
                generation_index=0,
                score=1,
                critique_list=[],
                input_tokens=0,
                output_tokens=0,
                chunk_count=0,
                success=False,
                error="No code files to evaluate"
            )
        
        chunk_results = []
        
        for idx, (paper_chunk, code_chunk) in enumerate(zip(paper_chunks, code_chunks)):
            # All chunks should have code, so evaluate all of them
            if self.config.eval_type == "ref_free":
                result = self.chunk_evaluator(
                    paper=paper_chunk,
                    code_chunk=code_chunk,
                    nudge=nudge,
                    chunk_index=idx,
                    total_chunks=len(paper_chunks)
                )
            else:  # ref_based - same gold_code for ALL chunks
                result = self.chunk_evaluator(
                    paper=paper_chunk,
                    gold_code=gold_code_string,  # SAME for all chunks
                    target_code=code_chunk,
                    nudge=nudge,
                    chunk_index=idx,
                    total_chunks=len(paper_chunks)
                )
            
            chunk_results.append(result)
            
            status = "✓" if result.success else "✗"
            print(f"[PROGRESS] {status} Chunk {idx+1}/{len(paper_chunks)} - Score: {result.score}, Critiques: {len(result.critique_items)}")
        
        # Aggregate all results (all chunks contain code)
        successful = [r for r in chunk_results if r.success]
        
        if not successful:
            return GenerationResult(
                generation_index=0,
                score=1,
                critique_list=[],
                input_tokens=0,
                output_tokens=0,
                chunk_count=len(paper_chunks),
                success=False,
                error="All evaluated chunks failed"
            )
        
        # Arithmetic mean of scores
        avg_score = round(statistics.mean([r.score for r in successful]))
        
        # Merge critique lists based on eval type
        all_critiques = [r.critique_items for r in successful]
        if self.config.eval_type == "ref_free":
            merged_critiques = merge_critique_lists_ref_free(all_critiques, max_items=50)
        else:
            merged_critiques = merge_critique_lists_ref_based(all_critiques, max_items=50)
        
        total_input = sum(r.input_tokens for r in chunk_results)
        total_output = sum(r.output_tokens for r in chunk_results)
        
        return GenerationResult(
            generation_index=0,
            score=avg_score,
            critique_list=merged_critiques,
            input_tokens=total_input,
            output_tokens=total_output,
            chunk_count=len(paper_chunks),
            success=True
        )
    

def calculate_safe_chunk_size(config: EvaluationConfig) -> int:
    """Calculate maximum chunk size that fits in context."""
    # Reserve tokens for: output, reasoning, safety margin, and overhead
    reserved_tokens = config.max_output_tokens + config.reasoning_tokens
    reserved_tokens = int(reserved_tokens * (1 + config.safety_margin))
    
    # Maximum tokens available for input (paper + code combined)
    max_input_tokens = config.max_context - reserved_tokens
    
    # Ensure we have reasonable minimum
    return max(max_input_tokens, 2000)

def create_document_chunks(paper: str, code_files: Dict[str, str], 
                          gold_code_string: str, config: EvaluationConfig) -> Tuple[List[str], List[str]]:
    """
    CORRECT: Split paper into sections, distribute code files across sections.
    Gold code is passed as complete string (matching original behavior).
    
    Chunking order: paper + gold_code + target_code (gold BEFORE target)
    
    Returns: (paper_chunks, code_chunks)
    """
    
    tokenizer = tiktoken.get_encoding("cl100k_base")
    chunk_size = calculate_safe_chunk_size(config)
    
    paper_tokens = len(tokenizer.encode(paper, disallowed_special=()))
    gold_tokens = len(tokenizer.encode(gold_code_string, disallowed_special=())) if gold_code_string else 0
    
    print(f"[INFO] Paper: {paper_tokens} tokens, Gold code: {gold_tokens} tokens, Max chunk: {chunk_size} tokens")
    
    # STEP 1: Split paper into manageable sections
    paper_sections = []
    paragraphs = [p.strip() for p in paper.split('\n\n') if p.strip()]
    
    current_section = []
    current_tokens = 0
    
    # Reserve space for gold code (which will be in every chunk)
    available_for_paper_code = chunk_size - gold_tokens
    target_paper_size = available_for_paper_code // 2  # Reserve half for target code
    
    for para in paragraphs:
        para_tokens = len(tokenizer.encode(para, disallowed_special=()))
        
        if current_tokens + para_tokens > target_paper_size:
            if current_section:
                paper_sections.append('\n\n'.join(current_section))
                current_section = []
                current_tokens = 0
            
            # If single paragraph is too big, split it
            if para_tokens > target_paper_size:
                words = para.split()
                word_chunk = []
                word_tokens = 0
                
                for word in words:
                    test = ' '.join(word_chunk + [word])
                    test_tokens = len(tokenizer.encode(test, disallowed_special=()))
                    
                    if test_tokens <= target_paper_size:
                        word_chunk.append(word)
                        word_tokens = test_tokens
                    else:
                        if word_chunk:
                            paper_sections.append(' '.join(word_chunk))
                        word_chunk = [word]
                        word_tokens = len(tokenizer.encode(word, disallowed_special=()))
                
                if word_chunk:
                    paper_sections.append(' '.join(word_chunk))
            else:
                current_section.append(para)
                current_tokens = para_tokens
        else:
            current_section.append(para)
            current_tokens += para_tokens
    
    if current_section:
        paper_sections.append('\n\n'.join(current_section))
    
    print(f"[INFO] Split paper into {len(paper_sections)} sections")
    
    # STEP 2: Prepare target code files
    code_file_list = []
    for filename, code in code_files.items():
        code_text = f"```\n## File name: {filename}\n{code}\n```\n\n"
        code_file_list.append(code_text)
    
    print(f"[INFO] {len(code_file_list)} code files to distribute")
    
    # STEP 3: Combine paper sections with target code files
    paper_chunks = []
    code_chunks = []
    
    code_idx = 0
    
    for i, paper_section in enumerate(paper_sections):
        paper_section_tokens = len(tokenizer.encode(paper_section, disallowed_special=()))
        available_for_code = chunk_size - paper_section_tokens - gold_tokens
        
        # Collect target code files that fit with this paper section
        chunk_code_files = []
        used_tokens = paper_section_tokens + gold_tokens
        
        while code_idx < len(code_file_list):
            code_file = code_file_list[code_idx]
            code_tokens = len(tokenizer.encode(code_file, disallowed_special=()))
            
            if used_tokens + code_tokens <= chunk_size:
                chunk_code_files.append(code_file)
                used_tokens += code_tokens
                code_idx += 1
            else:
                # No more room
                break
        
        # Build the chunk
        paper_chunks.append(paper_section)
        
        if chunk_code_files:
            code_chunks.append("".join(chunk_code_files))
            status = "HAS CODE"
        else:
            code_chunks.append("no code")
            status = "NO CODE (skip)"
        
        print(f"[CHUNK] {i+1}/{len(paper_sections)}: {status} - {used_tokens} tokens")
    
    code_count = sum(1 for c in code_chunks if c != "no code")
    print(f"[INFO] Total: {len(paper_chunks)} chunks ({code_count} with code, {len(paper_chunks) - code_count} skipped)")
    
    return paper_chunks, code_chunks
    
        
def configure_dspy(config: EvaluationConfig, temperature: float):
    """Configure DSPy."""
    api_base = os.getenv("LITELLM_API_BASE", "http://localhost:4000")
    api_key = os.getenv("LITELLM_API_KEY", "anything")
    
    model = config.gpt_version
    if not model.startswith("openai/"):
        model = "openai/" + model
    
    dspy.settings.configure(lm=dspy.LM(
        model=model,
        api_base=api_base,
        api_key=api_key,
        max_tokens=config.max_output_tokens,
        temperature=temperature if temperature else config.temperature
    ))

def validate_config(config: EvaluationConfig):
    """Validate config."""
    checks = [
        (os.path.exists(config.target_repo_dir), "Target repo not found"),
    ]
    
    if config.eval_type == "ref_based":
        checks.append((config.gold_repo_dir and os.path.exists(config.gold_repo_dir), "Gold repo required"))
    
    for condition, msg in checks:
        if not condition:
            raise ValueError(msg)

def read_code_files(config: EvaluationConfig) -> Dict[str, str]:
    """Read code files as dict."""
    if config.papercoder:
        files = read_python_files(config.target_repo_dir)
        task_list_path = os.path.join(config.output_dir, "task_list.json")
        
        if os.path.exists(task_list_path):
            with open(task_list_path) as f:
                task_list = json.load(f)
        else:
            try:
                context_lst = extract_planning(os.path.join(config.output_dir, "planning_trajectories.json"))
                task_list = content_to_json(context_lst[2])
            except:
                task_list = {}
        
        return {f: files[f] for f in task_list.get("Task list", []) if not f.endswith(".yaml") and f in files}
    else:
        return read_all_files(
            config.target_repo_dir,
            allowed_ext=[".py", ".yaml", ".yml", ".md", ".sh", ".bash"],
            is_print=False
        )

def format_gold_code(config: EvaluationConfig) -> str:
    """
    Load and format gold code exactly like original OpenAI version.
    Returns formatted gold code string (ALL files concatenated) or empty string if ref_free.
    
    EXACTLY matches original lines 60-80.
    """
    # Only load gold code if ref_based AND gold_repo_dir exists
    if config.eval_type != "ref_based" or not config.gold_repo_dir:
        return ""
    
    # EXACT line from original code
    all_files_dict = read_all_files(config.gold_repo_dir, allowed_ext=[".py", ".yaml", ".yml", ".md", ".sh", ".bash"], is_print=False)
    
    goldcodes = ""
    gold_cnt = 0
    
    # Filter by selected_file_path if provided (matching original)
    if config.selected_file_path and len(config.selected_file_path) > 0:
        selected_file_lst = []
        with open(config.selected_file_path) as f:
            selected_file_lst = f.readlines()
        
        for s_idx in range(len(selected_file_lst)):
            selected_file_lst[s_idx] = selected_file_lst[s_idx].strip()
        
        for all_file, all_file_code in all_files_dict.items():
            if all_file not in selected_file_lst:
                continue
            goldcodes += f"```## File name: {all_file}\n{all_file_code}\n```\n\n"
            gold_cnt += 1
    else:
        # No filter - use all files (matching original)
        for all_file, all_file_code in all_files_dict.items():
            goldcodes += f"```## File name: {all_file}\n{all_file_code}\n```\n\n"
            gold_cnt += 1
    
    print(f"[INFO] Loaded {gold_cnt} gold files")
    
    return goldcodes
    

def load_and_prepare_paper(config: EvaluationConfig) -> str:
    """
    Loads the paper JSON.  
    If `input_json_type` == 'dolphin-ocr', strip bbox, reading_order, etc.
    Returns the final JSON as **string** (what the rest of the pipeline expects).
    """
    # 1. Read raw file
    with open(config.pdf_json_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    # 2. Optional clean-up
    if config.input_json_type == "dolphin-ocr":
        def clean_element(el):
            return {k: el[k] for k in ("label", "text") if k in el}

        def clean_page(pg):
            cleaned = {'elements': [clean_element(e) for e in pg.get("elements", [])]}
            if "page_number" in pg:
                cleaned["page_number"] = pg["page_number"]
            return cleaned

        if "pages" in data:
            data = {"pages": [clean_page(p) for p in data["pages"]]}
        else:
            data = {"elements": [clean_element(e) for e in data.get("elements", [])]}
    
        cleaned_path = os.path.join(config.output_dir, "paper_cleaned.json")
        with open(cleaned_path, "w", encoding="utf-8") as f:
            json.dump(data, f, indent=2, ensure_ascii=False)
        print(f"[INFO] Saved stripped JSON → {cleaned_path}")

    # 3. Return json.dumps() because the downstream code already relies on that
    return json.dumps(data, ensure_ascii=False)


NUDGE_VARIATIONS = [
    "Focus on critical flaws and implementation bugs",
    "Emphasize strengths and correct implementations",
    "Suggest concrete improvements and optimizations",
    "Identify edge cases and missing error handling",
    "Evaluate computational efficiency and scalability",
    "Check strict alignment with paper methodology",
    "Verify mathematical correctness and numerical stability",
    "Assess code completeness and missing features"
]

def get_run_parameters(run_index: int, config: EvaluationConfig) -> Tuple[float, str]:
    """Get temp and nudge."""
    is_deterministic = config.gpt_version.endswith("_local")
    
    if is_deterministic:
        if run_index == 0:
            return 0.0, "baseline"
        return config.temperature, NUDGE_VARIATIONS[(run_index - 1) % len(NUDGE_VARIATIONS)]
    else:
        if run_index == 0:
            return 0.0, "deterministic"
        return min(1.0, config.temperature + (run_index % 31) / 100.0), "stochastic"

def run_single_generation(
    paper_chunks: List[str],
    code_chunks: List[str],
    gold_code_string: str,
    run_index: int,
    config: EvaluationConfig
) -> GenerationResult:
    """Run single generation."""
    
    temp, nudge = get_run_parameters(run_index, config)
    configure_dspy(config, temp)
    
    print(f"\n{'='*60}")
    print(f"[GENERATION] {run_index + 1}/{config.generated_n}")
    print(f"[TEMPERATURE] {temp}")
    print(f"[NUDGE] {nudge}")
    print(f"[CHUNKS] {len(paper_chunks)}")
    print(f"{'='*60}")
    
    evaluator = RepoEvaluator(config)
    result = evaluator(paper_chunks, code_chunks, gold_code_string, nudge)
    result.generation_index = run_index
    
    if result.success:
        print(f"[RESULT] Score: {result.score}/5, Total Critiques: {len(result.critique_list)}")
    else:
        print(f"[ERROR] Generation failed: {result.error}")
    
    return result

def run_evaluation_suite(
    config: EvaluationConfig,
    paper: str,
    code_files: Dict[str, str],
    gold_code_string: str,
    tracker: ExperimentTracker
) -> List[GenerationResult]:
    """Run full suite with proper chunking."""
    
    # Create properly sized chunks
    paper_chunks, code_chunks = create_document_chunks(paper, code_files, gold_code_string, config)
    
    print(f"[INFO] Created {len(paper_chunks)} chunks (paper + target code combined)")
    
    # Run generations
    results = []
    
    for run_idx in range(config.generated_n):
        try:
            result = run_single_generation(paper_chunks, code_chunks, gold_code_string, run_idx, config)
            results.append(result)
            
            # Save checkpoint
            checkpoint = EvaluationCheckpoint(
                paper_name=config.paper_name,
                target_repo_dir=config.target_repo_dir,
                eval_type=config.eval_type,
                gold_repo_dir=config.gold_repo_dir,
                generated_n=config.generated_n,
                run_index=run_idx + 1,
                valid=result.success,
                score=result.score,
                critique_list=[item.model_dump() for item in result.critique_list],  # Pydantic v2
                nudge=get_run_parameters(run_idx, config)[1],
                temperature=get_run_parameters(run_idx, config)[0],
                input_tokens=result.input_tokens,
                output_tokens=result.output_tokens,
                chunked=True,
                chunk_count=result.chunk_count
            )
            
            checkpoint_file = os.path.join(
                config.eval_result_dir, "checkpoints",
                f"{config.paper_name}_gen{run_idx+1}.json"
            )
            os.makedirs(os.path.dirname(checkpoint_file), exist_ok=True)
            with open(checkpoint_file, 'w') as f:
                json.dump(asdict(checkpoint), f, indent=2)
            
            tracker.log_metrics({
                f"score_gen{run_idx+1}": result.score,
                f"critiques_gen{run_idx+1}": len(result.critique_list)
            })
            
        except KeyboardInterrupt:
            print("\n[INFO] Interrupted")
            break
        except Exception as e:
            print(f"[ERROR] Generation {run_idx+1} failed: {e}")
            traceback.print_exc()
    
    return results

def finalize_report(config: EvaluationConfig, results: List[GenerationResult], tracker: ExperimentTracker):
    """Generate final report."""
    
    if not results:
        print("[INFO] No results")
        return
    
    valid_results = [r for r in results if r.success]
    
    if not valid_results:
        print("[ERROR] All generations failed")
        return
    
    score_lst = [r.score for r in valid_results]
    avg_score = statistics.mean(score_lst)
    
    rationale_lst = [[item.model_dump() for item in r.critique_list] for r in valid_results]
    
    print(f"\n{'='*70}")
    print(f"{'🌟 EVALUATION COMPLETE 🌟':^70}")
    print(f"{'='*70}")
    print(f"📄 Paper: {config.paper_name}")
    print(f"📊 Eval Type: {config.eval_type}")
    print(f"🤖 Model: {config.gpt_version}")
    print(f"📈 Average Score: {avg_score:.4f}/5.0")
    print(f"✅ Valid: {len(valid_results)}/{config.generated_n}")
    print(f"📉 Range: {min(score_lst)}-{max(score_lst)}")
    print(f"🔢 Total Critiques: {sum(len(r.critique_list) for r in valid_results)}")
    print(f"{'='*70}\n")
    
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    final_file = os.path.join(
        config.eval_result_dir,
        f"{config.paper_name}_eval_{config.eval_type}_{timestamp}.json"
    )
    
    report = {
        'paper_name': config.paper_name,
        'target_repo_dir': config.target_repo_dir,
        'eval_type': config.eval_type,
        'gold_repo_dir': config.gold_repo_dir,
        'generated_n': config.generated_n,
        'valid_n': len(valid_results),
        'score': avg_score,
        'score_lst': score_lst,
        'rationale_lst': rationale_lst,
        'timestamp': datetime.now().isoformat(),
        'config': {
            'model': config.gpt_version,
            'temperature': config.temperature,
            'generated_n': config.generated_n
        }
    }
    
    with open(final_file, 'w') as f:
        json.dump(report, f, indent=2)
    
    print(f"[INFO] Report: {final_file}")
    
    tracker.log_metrics({'overall_avg_score': avg_score})
    tracker.save_artifact(
        output_dir=config.eval_result_dir,
        file_pattern=os.path.basename(final_file),
        artifact_type="file-checkpoints"
    )


def main(config: EvaluationConfig):
    """Main pipeline."""
    
    print(f"\n{'='*70}")
    print(f"{'🚀 DSPy Evaluation (Field-Based)':^70}")
    print(f"{'='*70}")
    print(f"Paper: {config.paper_name}")
    print(f"Mode: {config.eval_type}")
    print(f"Model: {config.gpt_version}")
    print(f"Generations: {config.generated_n}")
    print(f"{'='*70}\n")
    
    tracker = WandbWeaveTracker(config)
    
    print("[INFO] Loading & preprocessing paper JSON...")
    paper = load_and_prepare_paper(config)
    
    print("[INFO] Loading code files...")
    code_files = read_code_files(config)
    
    # Format gold code (matching original behavior - ALL files as one string)
    gold_code_string = format_gold_code(config)
    
    print(f"[INFO] Loaded {len(code_files)} target code files")
    
    results = run_evaluation_suite(config, paper, code_files, gold_code_string, tracker)
    
    finalize_report(config, results, tracker)
    
    print("\n✅ Complete!\n")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    
    parser.add_argument("--paper_name", required=True)
    parser.add_argument("--pdf_json_path", required=True)
    parser.add_argument("--target_repo_dir", required=True)
    parser.add_argument("--data_dir", default="../data")
    parser.add_argument("--output_dir", default="./output")
    parser.add_argument("--eval_result_dir", default="./eval_results")
    parser.add_argument("--eval_type", default="ref_free", choices=["ref_free", "ref_based"])
    parser.add_argument("--generated_n", type=int, default=8)
    parser.add_argument("--gpt_version", default="qwen3-8b-GGUF-q8_0_local")
    parser.add_argument("--papercoder", action="store_true")
    parser.add_argument("--max_context", type=int, default=40960)
    parser.add_argument("--max_output_tokens", type=int, default=4096)
    parser.add_argument("--delay", type=float, default=0.1)
    parser.add_argument("--temperature", type=float, default=1.0)
    parser.add_argument("--wandb_run_id", default=None)
    parser.add_argument("--gold_repo_dir", default="")
    parser.add_argument("--selected_file_path", default=None)
    parser.add_argument("--max_workers", type=int, default=3)
    parser.add_argument("--max_retries", type=int, default=3)
    parser.add_argument("--retry_delay", type=float, default=1.0)
    parser.add_argument("--cache_dir", default="/kaggle/working")
    parser.add_argument("--strict_original", action="store_true")
    parser.add_argument("--critiques_per_chunk", type=int, default=5)
    parser.add_argument("--input_json_type", default="standard",choices=["standard", "dolphin-ocr"],)
    
    args = parser.parse_args()
    
    try:
        config = EvaluationConfig(**vars(args))

        config.target_repo_dir = config.target_repo_dir or f"{output_dir}/code_output"

        validate_config(config)
        
        os.makedirs(config.eval_result_dir, exist_ok=True)
        os.makedirs(os.path.join(config.eval_result_dir, "checkpoints"), exist_ok=True)
        
        main(config)
        
    except (ValidationError, ValueError) as e:
        sys.exit(f"[CRITICAL] Error: {e}")
    except Exception as e:
        traceback.print_exc()
        sys.exit(f"[CRITICAL] Fatal: {e}")