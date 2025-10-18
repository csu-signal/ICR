# Standard library imports
import os
import sys
import re
import math
import random
import json
import copy
import shutil
import warnings
import inspect
import pickle
import time
import typing
from enum import Enum
from functools import partial, wraps
from dataclasses import dataclass, field
from collections import defaultdict
from typing import Any, Callable, Dict, List, Literal, Optional, Tuple, Union

# Third-party data science libraries
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# PyTorch
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset

# Transformers and related
from transformers import (
    AutoModelForCausalLM,
    AutoModelForSequenceClassification,
    AutoTokenizer,
    BitsAndBytesConfig,
    DataCollatorForLanguageModeling,
    GenerationMixin,
    HfArgumentParser,
    PreTrainedModel,
    PreTrainedTokenizerBase,
    PreTrainedTokenizerFast,
    Trainer,
    TrainingArguments,
    get_cosine_schedule_with_warmup,
    get_linear_schedule_with_warmup,
    pipeline,
    set_seed,
)
from transformers.trainer_utils import EvalLoopOutput, TrainOutput, speed_metrics
from transformers.trainer_callback import TrainerCallback

# Datasets and HuggingFace ecosystem
from datasets import Dataset, DatasetDict, load_dataset, load_from_disk
from accelerate import Accelerator, PartialState
from accelerate.utils import gather_object, is_deepspeed_available

# import the ICR trainer class for training loop
from ICR_trainer import ICRPPO_trainer

# TRL (Transformer Reinforcement Learning)
from trl import (
    AutoModelForCausalLMWithValueHead,
    ORPOConfig,
    ORPOTrainer,
    PPOConfig,
    PPOTrainer,
    RewardConfig,
    RewardTrainer,
    SFTConfig,
    SFTTrainer,
)
from trl.core import (
    LengthSampler,
    PPODecorators,
    WANDB_PADDING,
    clip_by_value,
    convert_to_scalar,
    entropy_from_logits,
    flatten_dict,
    logprobs_from_logits,
    masked_mean,
    masked_var,
    masked_whiten,
    stack_dicts,
    stats_to_np,
)
from trl.models import (
    SUPPORTED_ARCHITECTURES,
    PreTrainedModelWrapper,
    create_reference_model,
    unwrap_model_for_generation,
)
from trl.trainer.utils import RewardDataCollatorWithPadding, compute_accuracy, print_rich_table

# PEFT (Parameter Efficient Fine-Tuning)
from peft import LoraConfig, PeftModel, prepare_model_for_kbit_training, AutoPeftModelForCausalLM
import bitsandbytes as bnb

# Additional ML libraries
from sentence_transformers import SentenceTransformer, util
from evaluate import load as load_metric
from rouge_score import rouge_scorer
from safetensors import safe_open
from tqdm import tqdm

# Project-specific imports
from deli_reward_functions import calculate_ppo_deli_proxy_reward, calculate_ppo_deli_gold_reward

# Weights & Biases
import wandb
wandb.init(project="ICR_training_NIPS_2025_robust_collaborator")

# Configure pandas progress bar
tqdm.pandas()

# Debugging configuration
os.environ["TORCH_DISTRIBUTED_DEBUG"] = "DETAIL"


@dataclass
class ScriptArguments_2:
    """
    The arguments for the BC Collaborator (SFT agent) training
    """

    # data parameters
    beta: Optional[float] = field(default=0.1, metadata={"help": "the beta parameter for DPO loss"})
    base_model_name
    base_model_name: Optional[str] = field(
        default="../sft/results/final_checkpoint",
        metadata={"help": "the location of the SFT model name or path"},
    )

    learning_rate: Optional[float] = field(default=5e-6, metadata={"help": "optimizer learning rate"})
    lr_scheduler_type: Optional[str] = field(default="cosine", metadata={"help": "the lr scheduler type"})
    # warmup_steps: Optional[int] = field(default=10, metadata={"help": "the number of warmup steps"})
    weight_decay: Optional[float] = field(default=0.05, metadata={"help": "the weight decay"})
    optimizer_type: Optional[str] = field(default="paged_adamw_32bit", metadata={"help": "the optimizer type"})
    loss_type: Optional[str] = field(default="sigmoid", metadata={"help": "the loss type you want to test your policy on"})

    per_device_train_batch_size: Optional[int] = field(default=12, metadata={"help": "train batch size per device"})
    per_device_eval_batch_size: Optional[int] = field(default=5, metadata={"help": "eval batch size per device"})
    gradient_accumulation_steps: Optional[int] = field(
        default=4, metadata={"help": "the number of gradient accumulation steps"}
    )
    gradient_checkpointing: Optional[bool] = field(
        default=True, metadata={"help": "whether to use gradient checkpointing"}
    )

    

    gradient_checkpointing_use_reentrant: Optional[bool] = field(
        default=False, metadata={"help": "whether to use reentrant for gradient checkpointing"}
    )

    lora_alpha: Optional[float] = field(default=16, metadata={"help": "the lora alpha parameter"})
    lora_dropout: Optional[float] = field(default=0.05, metadata={"help": "the lora dropout parameter"})
    lora_r: Optional[int] = field(default=8, metadata={"help": "the lora r parameter"})
    dataset: Optional[str] = field(default="ultrafeedback_binarized", metadata={"help": "the dataset used for training and evaluation "})

    max_prompt_length: Optional[int] = field(default=512, metadata={"help": "the maximum prompt length"})
    max_length: Optional[int] = field(default=4096, metadata={"help": "the maximum sequence length"})
    max_new_tokens: Optional[int] = field(default=256, metadata={"help": "the maximum sequence length"})
    
    max_steps: Optional[int] = field(default=1000, metadata={"help": "max number of training steps"})
    logging_steps: Optional[int] = field(default=20, metadata={"help": "the logging frequency"})
    save_steps: Optional[int] = field(default=200, metadata={"help": "the saving frequency"})
    save_strategy: Optional[str] = field(default="no", metadata={"help": "whether to save intermediate steps during training"})
  
 
    eval_steps: Optional[int] = field(default=100, metadata={"help": "the evaluation frequency"})
    
    output_dir: Optional[str] = field(default="./results_falcon", metadata={"help": "the output directory"})
    log_freq: Optional[int] = field(default=1, metadata={"help": "the logging frequency"})
    load_in_4bit: Optional[bool] = field(default=True, metadata={"help": "whether to load the model in 4bit"})
    model_dtype: Optional[str] = field(
        default="float16", metadata={"help": "model_dtype[float16, bfloat16, float] for loading."}
    )

    # instrumentation
    sanity_check: Optional[bool] = field(default=False, metadata={"help": "only train on 1000 samples"})
    report_to: Optional[str] = field(
        default="wandb",
        metadata={
            "help": 'The list of integrations to report the results and logs to. Supported platforms are `"azure_ml"`,'
            '`"comet_ml"`, `"mlflow"`, `"neptune"`, `"tensorboard"`,`"clearml"` and `"wandb"`. '
            'Use `"all"` to report to all integrations installed, `"none"` for no integrations.'
        },
    )
    # debug argument for distributed training
    ignore_bias_buffers: Optional[bool] = field(
        default=False,
        metadata={
            "help": "fix for DDP issues with LM bias/mask buffers - invalid scalar type,`inplace operation. See"
            "https://github.com/huggingface/transformers/issues/22482#issuecomment-1595790992"
        },
    )
    seed: Optional[int] = field(
        default=0, metadata={"help": "Random seed that will be set at the beginning of training."}
    )

        
        
def transform_and_assign_preferences(example):
    """Prepare the intervention generation prompt and response for SFT training."""
    
    system_prompt_rm = (
    "You are an expert in collaborative task analysis and personality-driven communication. "
    "Your task is to generate nuanced intervention statements within a dialogue. "
    "Given the **dialogue history** involving three participants and the *game details*, "
    "generate a <intervention> statement that acts as indirect persuasion. This statement should "
    "encourage the participants to reevaluate their beliefs and assumptions about the task. "
    "Additionally, provide a <rationale> or explanation for your intervention statement. Base your reasoning "
    "on evidence from the dialogue, focusing on elements such as: "
    "- Incorrect assumptions "
    "- False beliefs "
    "- Rash decisions "
    "- Missing evidence ")

    friction_definition_game_definition_prompt_rm = (
        "*Game details and ground-truth*: The game is called 'Game of Weights.' The participants (P1, P2, and P3) are "
        "trying to determine the weight of various blocks. The blocks are of different colors and have specific weights in grams: "
        "the red block is 10 grams, the blue block is 10 grams, the green block is 20 grams, the purple block is 30 grams, and "
        "the yellow block is 50 grams. At the start of the game, participants are only allowed to weigh two blocks at a time, "
        "and they are told the weight of the red block. The participants must figure out how to determine the weight of each block. "
        "At the beginning of the game, they are unaware of the actual weights. Additionally, we inform the participants that they "
        "don’t need to use the scale's slider. The actual reason is that the blocks are in increments of 10 grams. "
        "The **dialogue history** is given below: "
    )
    system_prompt_rm = (
    "You are an expert in collaborative task analysis and personality-driven communication."
    "Your task is to generate nuanced intervention statements within a dialogue."
   )

    

    # "Be specific and ensure that your response clearly addresses the dynamics in the dialogue.")
    # Combine the prompts and context history
    # prompt = (system_prompt_rm + friction_definition_game_definition_prompt_rm + example['context']).replace('\n', ' ')
    prompt = (system_prompt_rm + example['context']).replace('\n', ' ')
    # Format the chosen response
    # example['chosen'] = example["chosen"].replace('\n', ' ')
    # example['rejected'] = example["rejected"].replace('\n', ' ')
    chosen_response_format = f"Answer: <intervention> {example['chosen']}. <rationale>: {example['chosen_rationale']}"
    rejected_response_format = f"Answer: <intervention> {example['rejected']}. <rationale>: {example['rejected_rationale']}"

    chosen_response = [
        {'content': prompt, 'role': 'user'},
        {'content': chosen_response_format, 'role': 'assistant'}
    ]

    # Format the rejected response
    rejected_response = [
        {'content': prompt, 'role': 'user'},
        {'content': rejected_response_format, 'role': 'assistant'}
    ]

    # Return the new structure with feedback weights
    return {
        'prompt': prompt,
        'chosen': chosen_response,
        'rejected': rejected_response,
    }



def transform_and_assign_preference_deli_with_cf(example, tokenizer):
    """
    Prepare the intervention generation prompt and response for SFT training with counterfactual scenarios.
    
    This function:
    1. Creates both factual and counterfactual versions of the messages
    2. Applies the chat template to both
    3. Returns properly formatted prompts and responses
    
    Args:
        example (dict): The input example with 'messages' field
        tokenizer: The tokenizer to apply chat template
        
    Returns:
        dict: Dictionary with original and counterfactual prompts and responses
    """
    # Get the original messages
    original_messages = example['messages']
    
    # Create a deep copy to avoid modifying the original
    cf_messages = copy.deepcopy(original_messages)
    
    # Modify the system message in the counterfactual copy
    for i, message in enumerate(cf_messages):
        if message['role'] == 'system':
            # Add the counterfactual assumption to the system prompt
            cf_messages[i]['content'] = message['content'] + "\n\nIMPORTANT: Assume that any intervention by Intervetion Agent will NOT increase common ground or improve collaboration between participants."
            break
    
    # Extract the dialogue context (user message content)
    dialogue_context = ""
    for message in original_messages:
        if message['role'] == 'user':
            dialogue_context = message['content']
            break
    
    # Extract the assistant's response
    assistant_response = ""
    for message in original_messages:
        if message['role'] == 'assistant':
            assistant_response = message['content']
            break
    
    # Apply chat template to both original and counterfactual messages
    # For factual prompt (original system + user, without assistant)
    factual_prompt_messages = [msg for msg in original_messages if msg['role'] != 'assistant']
    prompt = tokenizer.apply_chat_template(
        factual_prompt_messages,
        tokenize=False,
        add_generation_prompt=True
    )
    
    # For counterfactual prompt (modified system + original user, without assistant)
    cf_prompt_messages = [msg for msg in cf_messages if msg['role'] != 'assistant']
    cf_prompt_messages_with_assistant = [msg for msg in cf_messages]
    cf_prompt = tokenizer.apply_chat_template(
        cf_prompt_messages,
        tokenize=False,
        add_generation_prompt=True
    )
    
    # For the chosen response (complete conversation)
    chosen = tokenizer.apply_chat_template(
        original_messages,
        tokenize=False,
        add_generation_prompt=False
    )
    
    cf_chosen = tokenizer.apply_chat_template(
        cf_prompt_messages_with_assistant,
        tokenize=False,
        add_generation_prompt=False
    )
    
    return {
        'prompt': prompt,
        'cf_prompt': cf_prompt,
        'chosen': chosen,
        'cf_chosen':cf_chosen,
        'dialogue_context': dialogue_context,
        'assistant_response': assistant_response
    }


def build_friction_dataset_full_context(train_data, config, tokenizer, max_length=1024, max_query_length=800):
    """
    Build dataset for training a Causal model on intervention generation with counterfactual versions.
    
    Args:
        train_data (`list`): List of training samples with prompts and counterfactual prompts.
        config (`object`): Configuration object containing model_name.
        tokenizer: The tokenizer to use for encoding.
        max_length (`int`): Maximum token length for the input sequence.
        max_query_length (`int`): Maximum token length allowed for the query.

    Returns:
        dataset (`datasets.Dataset`): The filtered dataset with both factual and counterfactual queries.
    """
    def tokenize_friction_samples(sample):
        """
        Tokenizes factual and counterfactual prompts and returns the prepared tensors.
        """
        # Get the prompts and response
        factual_prompt = sample["prompt"]  # Original context with chat template applied
        counterfactual_prompt = sample["cf_prompt"]  # Counterfactual context with chat template
        assistant_response = sample["assistant_response"]  # Just the assistant's message
        
        # Tokenize factual prompt
        input_ids = tokenizer.encode(factual_prompt, truncation=True, max_length=max_length)
        attention_mask = [1] * len(input_ids)
        
        # Tokenize counterfactual prompt
        cf_input_ids = tokenizer.encode(counterfactual_prompt, truncation=True, max_length=max_length)
        cf_attention_mask = [1] * len(cf_input_ids)
        
        # For readability/debugging, decode back to text
        query = tokenizer.decode(input_ids, skip_special_tokens=True)
        cf_query = tokenizer.decode(cf_input_ids, skip_special_tokens=True)
        
        return {
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "cf_input_ids": cf_input_ids,
            "cf_attention_mask": cf_attention_mask,
            "label": 1,  # Following the original convention
            "query": query,
            "cf_query": cf_query,
            "golden_friction": assistant_response,
            "dialogue_context": sample["dialogue_context"]
        }

    # Tokenize all samples
    processed_samples = []
    for sample in tqdm(train_data, desc="Tokenizing Samples with Counterfactuals"):
        try:
            processed_sample = tokenize_friction_samples(sample)
            processed_samples.append(processed_sample)
        except Exception as e:
            print(f"Error processing sample: {e}")
            continue

    # Filter queries that are within the max_query_length
    filtered_samples = [
        sample for sample in processed_samples 
        if len(sample["input_ids"]) <= max_query_length and len(sample["cf_input_ids"]) <= max_query_length
    ]

    print(f"Filtered {len(processed_samples) - len(filtered_samples)} samples exceeding {max_query_length} tokens.")

    # Convert to Hugging Face Dataset
    dataset = Dataset.from_list(filtered_samples)

    # Set format for PyTorch compatibility
    dataset.set_format(
        type="torch", 
        columns=[
            "input_ids", "attention_mask", "cf_input_ids", "cf_attention_mask", 
            "label", "query", "cf_query", "golden_friction"
        ]
    )
    
    return dataset



def process_wtd_simulated_dataset(dataset, split, tokenizer, 
    sanity_check: bool = False,
    max_length: int = 1024,  # Maximum token length, not character length
    cache_dir: Optional[str] = None,
    num_proc=24):
    """
    Process and filter the dataset based on token lengths, not character lengths.
    
    Args:
        dataset: The dataset to process
        split: The dataset split (e.g., 'train')
        tokenizer: The tokenizer to use for token counting
        sanity_check: Whether to only process a small subset
        max_length: Maximum token length for prompts
        cache_dir: Optional cache directory
        num_proc: Number of processes for dataset mapping
        
    Returns:
        The processed dataset
    """
    if sanity_check:
        dataset = dataset.shuffle().select(range(1000))
    
    print(f"Original dataset size: {len(dataset)}")
    
    # First, let's analyze the token lengths to get a better understanding
    def analyze_token_lengths(example):
        """Count tokens in prompts and return sample with token counts"""
        prompt_tokens = len(tokenizer.encode(example["prompt"]))
        cf_prompt_tokens = len(tokenizer.encode(example["cf_prompt"]))
        
        return {
            "prompt_token_length": prompt_tokens,
            "cf_prompt_token_length": cf_prompt_tokens,
        }
    
    # Add token length analysis
    analysis_dataset = dataset.map(
        analyze_token_lengths,
        num_proc=num_proc,
    )
    
    # Print token length statistics
    prompt_lengths = analysis_dataset["prompt_token_length"]
    cf_prompt_lengths = analysis_dataset["cf_prompt_token_length"]
    
    print(f"Prompt token length - Min: {min(prompt_lengths)}, Max: {max(prompt_lengths)}, Mean: {sum(prompt_lengths)/len(prompt_lengths):.2f}")
    print(f"CF Prompt token length - Min: {min(cf_prompt_lengths)}, Max: {max(cf_prompt_lengths)}, Mean: {sum(cf_prompt_lengths)/len(cf_prompt_lengths):.2f}")
    
    # Print histogram of lengths
    prompt_length_counts = {}
    for length in prompt_lengths:
        bucket = (length // 100) * 100  # Bucket by hundreds
        prompt_length_counts[bucket] = prompt_length_counts.get(bucket, 0) + 1
    
    print("\nPrompt token length distribution:")
    for bucket in sorted(prompt_length_counts.keys()):
        print(f"{bucket}-{bucket+99}: {prompt_length_counts[bucket]} samples")
    
    # Now filter based on token counts using the tokenizer
    def check_token_length(example):
        prompt_tokens = example["prompt_token_length"]
        cf_prompt_tokens = example["cf_prompt_token_length"]
        return prompt_tokens <= max_length and cf_prompt_tokens <= max_length
    
    filtered_dataset = analysis_dataset.filter(check_token_length)
    
    print(f"\nFiltered dataset after token length constraint: {len(filtered_dataset)}")
    
    # If the filtered dataset is too small, try increasing the max_length
    if len(filtered_dataset) < 0.5 * len(dataset):
        print("\nWarning: More than 50% of samples were filtered out!")
        print("Consider increasing max_length or examining the prompt generation.")
        
        # Calculate what max_length would keep 95% of the data
        sorted_lengths = sorted(prompt_lengths + cf_prompt_lengths)
        suggested_length = sorted_lengths[int(0.95 * len(sorted_lengths))]
        print(f"Suggested max_length to keep 95% of data: {suggested_length}")
    
    # Remove analysis columns if not needed
    selected_columns = [col for col in dataset.column_names if col not in ["prompt_token_length", "cf_prompt_token_length"]]
    filtered_dataset = filtered_dataset.select_columns(selected_columns)
    
    return filtered_dataset

def build_friction_dataset_llama_full_context(train_data, config, max_length=1024, max_query_length=800):
    """
    Build dataset for training a Llama3  model on intervention classification using the full context up to "Answer: <intervention>",
    and filter queries exceeding the max_query_length.

    Args:
        train_data (`list`): List of training samples containing context and intervention statements.
        config (`object`): Configuration object containing model_name.
        max_length (`int`): Maximum token length for the input sequence.
        max_query_length (`int`): Maximum token length allowed for the query.

    Returns:
        dataset (`datasets.Dataset`): The filtered dataset ready for training.
    """
    tokenizer = AutoTokenizer.from_pretrained(config.model_name)
    tokenizer.pad_token = tokenizer.eos_token # for GPT 2
    
    # below processing is for llama 3
    # tokenizer.pad_token = "<|reserved_special_token_0|>" # new pad token for this run
    tokenizer.pad_token_id = tokenizer.convert_tokens_to_ids(tokenizer.pad_token)
    tokenizer.padding_side = 'right'
 

    def tokenize_friction_samples(sample):
        """
        Tokenizes and processes a sample into input, query, and label for training.

        Args:
            sample (`dict`): Dictionary containing context and intervention statements.

        Returns:
            `dict`: Processed sample with tokenized input, query, and label.
        """
        # Extract context and response
        context = sample["prompt"]  # Dialogue context
        response = sample["chosen"]  # Friction response
        dialogue_context = sample['dialogue_context']
   
        # friction_response = response.split("<rationale>:")[1].strip()

        friction_response = "<rationale>:" + response.rsplit("<rationale>:", 1)[-1].strip()
       
 
        input_sequence = context + " " + "### Assistant:"
        # Tokenize the input sequence
        input_ids = tokenizer.encode(input_sequence, truncation=True, max_length=max_length)
        # friction_input_ids = tokenizer.encode(friction_response, truncation=True, max_length=max_length)
        # Decode the input back into a query for readability
        query = tokenizer.decode(input_ids, skip_special_tokens=True, clean_up_tokenization_spaces=True)

       
        label = 1

        return {
            "input_ids": input_ids,
            "attention_mask": [1] * len(input_ids),
            "label": label,
            "query": query,
            "golden_friction": friction_response, 
            "dialogue_context":dialogue_context
        }

    # Tokenize all samples
    processed_samples = [
        tokenize_friction_samples(sample) for sample in tqdm(train_data, desc="Tokenizing Samples")
    ]

    # Filter queries that are within the max_query_length
    filtered_samples = [
        sample for sample in processed_samples if len(sample["input_ids"]) <= max_query_length
    ]

    print(f"Filtered {len(processed_samples) - len(filtered_samples)} samples exceeding {max_query_length} tokens.")

    # Convert to Hugging Face Dataset
    dataset = Dataset.from_list(filtered_samples)

    # Set format for PyTorch compatibility
    dataset.set_format(type="torch", columns=["input_ids", "attention_mask", "label", "query", "golden_friction"])

    return dataset


class Config:
    model_name = "gpt2"

def collator(data):
    return dict((key, [d[key] for d in data]) for key in data[0])

def calculate_batches(dataset_size, batch_size, num_epochs, drop_last=True):
    """
    Calculate the number of batches per epoch and total batches.
    
    Args:
        dataset_size: Size of the dataset
        batch_size: Batch size for training
        num_epochs: Number of epochs to train
        drop_last: Whether to drop the last incomplete batch
        
    Returns:
        batches_per_epoch: Number of batches in one epoch
        total_batches: Total number of batches over all epochs
    """
    if drop_last:
        batches_per_epoch = dataset_size // batch_size
    else:
        batches_per_epoch = (dataset_size + batch_size - 1) // batch_size
        
    total_batches = batches_per_epoch * num_epochs
    
    return batches_per_epoch, total_batches
 
def compute_probability(rewards_predicted_friction, rewards_golden_friction):
    # Compute the difference between the predicted and golden rewards
    reward_difference = rewards_predicted_friction - rewards_golden_friction
    
    # Normalize the difference to a range between -1 and 1
    # Here, we assume the difference is between 0 and 6 for normalization
    # normalized_score = 2 * (reward_difference - 0) / (5 - 0) - 1
    
    # # Clip the value to ensure it stays between -1 and 1
    # normalized_score = torch.clamp(normalized_score, -1.0, 1.0)
    
    return reward_difference

def check_ppo_inputs_for_collaboration(batch):

    print("Available keys in batch:", batch.keys())
    
    # Check the type of elements in the batch
    print(f"Type of input_ids: {type(batch['input_ids'])}")
    
    # Convert to tensors if needed, or handle list types
    if isinstance(batch['input_ids'], list):
        print(f"Length of input_ids list: {len(batch['input_ids'])}")
        if len(batch['input_ids']) > 0:
            print(f"Type of first element: {type(batch['input_ids'][0])}")
            print(f"Shape of first element: {batch['input_ids'][0].shape}")
    else:
        print(f"Factual query tensors shape: {batch['input_ids'].shape}")
    
    # Check if counterfactual queries are available
    if 'cf_input_ids' in batch:
        print("Counterfactual queries are available")
        if isinstance(batch['cf_input_ids'], list):
            print(f"Length of cf_input_ids list: {len(batch['cf_input_ids'])}")
        else:
            print(f"Counterfactual query tensors shape: {batch['cf_input_ids'].shape}")
    else:
        print("Counterfactual queries are NOT available in the batch")
    
    # Check other essential fields
    print("\nOther essential fields:")
    for field in ['label', 'query', 'cf_query', 'golden_friction']:
        print(f"- {field} available: {field in batch}")
 
    return None


def compute_rewards_from_classifier(model, tokenizer, queries, responses, max_length=512, response_length=100, device="cuda", golden_friction =None):
    """
    Computes rewards for responses based on a trained intervention classifier.

    Args:
        model: The trained BERT classification model.
        tokenizer: The tokenizer corresponding to the model.
        queries: List of query strings (prompts).
        responses: List of response strings (generated outputs).
        max_length: Maximum length for tokenization.
        response_length: Max tokens allocated to the response.
        device: Device to run the model on ('cpu' or 'cuda').

    Returns:
        rewards: List of rewards for each query-response pair.
    """
    model = model.to(device)
    model.eval()
    rewards = []
    system_prompt_rm = (
        "Please rate the following intervention intervention in light of the **dialogue history** of a *game* provided below. "
        "An intervention is a statement that acts as indirect persuasion and prompts participants to "
        "reevaluate their beliefs and assumptions about the task, primarily—but not exclusively—in response to new evidence "
        "that challenges their preconceived notions about the state of the task or the block weights."
    )

    def extract_text(input_string):
        # Define the regular expression pattern to match text between 'user' and '### Assistant:'
        pattern = r'user\n(.*?)### Assistant:'
        
        # Use re.search to find the matching part
        match = re.search(pattern, input_string, re.DOTALL)  # re.DOTALL allows dot (.) to match newlines
        
        # If a match is found, return the extracted text; otherwise, return None
        if match:
            return match.group(1).strip()
        else:
            return None


    with torch.no_grad():
        for query, response, gold_friction in zip(queries, responses, golden_friction):
            # print("query  in reward fetching", query)
            dialogue_context = extract_text(query)
            # print("dialogue context in reward fetching", dialogue_context)
            # prompt_and_dialogue_context = system_prompt_rm + dialogue_context
            prompt_and_dialogue_context = system_prompt_rm 
            # print("full prompt_and_dialogue_context context in reward fetching", len(golden_friction))

            predicted_response = f"</s> {response} </s>"
            gold_friction = f"</s> {gold_friction} </s>"
            # Tokenize response first to ensure it is fully included
            encoded_response = tokenizer(
                predicted_response,
                truncation=True,
                max_length=response_length,  # Cap response length
                padding=False,
                return_tensors="pt"
            )

            encoded_golden_response = tokenizer(
                gold_friction,
                truncation=True,
                max_length=response_length,  # Cap response length
                padding=False,
                return_tensors="pt"
            )


            # Calculate the remaining space for the query
            remaining_length = max_length - encoded_response["input_ids"].size(1) - 1  # Space for [SEP]
            remaining_length_golden = max_length - encoded_golden_response["input_ids"].size(1) - 1  # Space for [SEP]
            # Tokenize query with the remaining length
            # encoded_query = tokenizer(
            #     query,
            #     truncation=True,
            #     max_length=remaining_length,  # Truncate query to remaining space
            #     padding=False,
            #     return_tensors="pt"
            # )

            encoded_query = tokenizer(
                prompt_and_dialogue_context,
                truncation=True,
                max_length=remaining_length,  # Truncate query to remaining space: this one is with the change of prompts of rthe trained rewards model OPT 1.3
                padding=False,
                return_tensors="pt"
            )


            # Move query and response tensors to the same device
            encoded_query = {key: value.to(device) for key, value in encoded_query.items()}
            encoded_response = {key: value.to(device) for key, value in encoded_response.items()}
            encoded_golden_response = {key: value.to(device) for key, value in encoded_golden_response.items()}
      
            input_ids = torch.cat([
                encoded_query["input_ids"],
   
                encoded_response["input_ids"]
            ], dim=1)

            attention_mask = torch.cat([
                encoded_query["attention_mask"],
   
                encoded_response["attention_mask"]
            ], dim=1)
            

            # for golden intervention encoding
            input_ids_golden = torch.cat([
                encoded_query["input_ids"],
       
                encoded_golden_response["input_ids"]
            ], dim=1)

            attention_mask_golden = torch.cat([
                encoded_query["attention_mask"],
 
                encoded_golden_response["attention_mask"]
            ], dim=1)



            # Ensure the final length does not exceed max_length
            if input_ids.size(1) > max_length:
                input_ids = input_ids[:, :max_length]
                attention_mask = attention_mask[:, :max_length]
            if input_ids_golden.size(1) > max_length:
                input_ids_golden = input_ids_golden[:, :max_length]
                attention_mask_golden = attention_mask_golden[:, :max_length]   

            # Get model outputs
            # rewards_chosen = model(input_ids=inputs["input_ids_chosen"], attention_mask=inputs["attention_mask_chosen"])[0]
            rewards_predicted_friction = model(input_ids=input_ids, attention_mask = attention_mask)[0]
            rewards_golden_friction = model(input_ids=input_ids_golden, attention_mask = attention_mask_golden)[0]
            # Compute the probability
            probability = compute_probability(rewards_predicted_friction, rewards_golden_friction)
 

            print("Probability based on reward difference:", probability)
            rewards.append(probability)
            # #previous code below for BERT RM 
            # outputs = model(input_ids=input_ids, attention_mask=attention_mask)
            # logits = outputs.logits

            # # Compute reward as scaled "GOOD" score
            # good_score = torch.softmax(logits, dim=1)[0][0].item()  # "GOOD" class probability
            # reward = 2 * good_score - 1  # Scale to [-1, 1]
            # rewards.append(reward)

    return rewards, rewards_predicted_friction, rewards_golden_friction




def sample_df_for_logging(df, n=5):
    if len(df) > n:
        return df.sample(n)
    return df    

if __name__ == "__main__":
    parser = HfArgumentParser(ScriptArguments)


    parser = HfArgumentParser(ScriptArguments)
    script_args = parser.parse_args_into_dataclasses()[0]
    print(script_args)
 

    set_seed(script_args.seed)

    dataset = load_from_disk("DELI_collab_dict_cg_dataset")
    dataset = load_from_disk("DELI_collab_dict_cg_dataset_strict_json_instruction")
    
    tokenizer = AutoTokenizer.from_pretrained(script_args.model_name) # get tokenizer for processing 
    # train_dataset = dataset["train"]
    # eval_dataset = dataset["test"]

        # Transform the dataset
    train_data = dataset['train'].map(
        lambda example: transform_and_assign_preference_deli_with_cf(example, tokenizer)
    )

 
    print(f"Size of the train set: {len(train_data)}")

    train_dataset = process_wtd_simulated_dataset(train_data, split='train', tokenizer=tokenizer, max_length=1124)
    train_dataset = train_dataset.select(range(19808)) # 19808
    train_dataset = train_dataset.shuffle(seed=42) 
    print("size of DELI after processing and filtering: train_dataset", train_dataset)

    config = PPOConfig(
    model_name=script_args.model_name,
    learning_rate=3e-6,
    log_with="wandb",
    batch_size=32,  # Smaller batch size
    mini_batch_size=4,
    gradient_accumulation_steps=8,  # Ensure batch_size = mini_batch_size * gradient_accumulation_steps
)


    friction_dataset = build_friction_dataset_full_context(
    train_dataset, 
    config,
    tokenizer,
    max_length=1124, 
    max_query_length=900
)


    # Check an example from the dataset
    example = friction_dataset[0]
    print("Input IDs:", example["input_ids"])
    print("Query:", example["query"])
    print("Label:", example["label"])

    #load the lora and peft configs 


    lora_config = LoraConfig(
        r=16,
        lora_alpha=32,
        lora_dropout=0.05,
        bias="none",
        task_type="CAUSAL_LM",
    )
    nf4_config = BitsAndBytesConfig(
        load_in_4bit=True, bnb_4bit_quant_type="nf4", bnb_4bit_use_double_quant=True, bnb_4bit_compute_dtype=torch.bfloat16
    )
    ref_model = None
    device_map = {"": Accelerator().local_process_index}

    config = PPOConfig(
    model_name=script_args.model_name,
    learning_rate=8e-7,
    log_with="wandb",
    batch_size=32,  # Smaller batch size
    mini_batch_size=4,
    gradient_accumulation_steps=8,  # Ensure batch_size = mini_batch_size * gradient_accumulation_steps
    #   clip_range=0.2,
    #     entropy_coef=0.01       # Add entropy coefficient  

)
 

    print("Loading base model...")
    model = AutoModelForCausalLM.from_pretrained(
        script_args.base_model_name,
        device_map={"": Accelerator().local_process_index},
        torch_dtype=torch.bfloat16,
        low_cpu_mem_usage=True,
        trust_remote_code=True,
        load_in_4bit=script_args.load_in_4bit,  # Enable 4-bit quantization
     
    )

    # tokenizer = AutoTokenizer.from_pretrained("meta-llama/Meta-Llama-3-8B-Instruct")
    tokenizer = AutoTokenizer.from_pretrained(script_args.model_name)
     
    tokenizer.pad_token = tokenizer.eos_token # for quen , not for llama 3
    tokenizer.padding_side = "right"  # Fix weird overflow issue with fp16 training
    print("Loading LoRA adapter...")

    lora_merged_model = PeftModel.from_pretrained(
        model,
        script_args.model_name,
        torch_dtype=torch.bfloat16,
        device_map={"": Accelerator().local_process_index},
        low_cpu_mem_usage=True,
        trust_remote_code=True,
    )
    print("script args model name", lora_merged_model)


    print("script args model name", lora_merged_model)
    model = AutoModelForCausalLMWithValueHead.from_pretrained(
    lora_merged_model,
    device_map=device_map,
    peft_config=lora_config,
    quantization_config=nf4_config,
    # reward_adapter=script_args.rm_adapter,
    use_safetensors=script_args.use_safetensors,
)
    print("after lora_merged_model", model)
    tokenizer = AutoTokenizer.from_pretrained(config.model_name)

    tokenizer.pad_token = tokenizer.eos_token
    # tokenizer.pad_token = "<|reserved_special_token_0|>" # new pad token for this run
    # tokenizer.pad_token_id = tokenizer.convert_tokens_to_ids(tokenizer.pad_token)
    tokenizer.padding_side = 'right'
    
    # Initialize ppo trainer class
    friction_ppo_trainer = ICRPPO_trainer(config, model, ref_model, tokenizer, dataset=friction_dataset, data_collator=collator)
    sent_kwargs = {"return_all_scores": True, "function_to_apply": "none", "batch_size": 32}

    #now load the bert friction_classifier trained to assign rewards to good and rogue intervention samples
    DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
  
    CHECKPOINT_DIR = "friction_rm_DELI_output_dir_all_samples/checkpoint-3000" # DELI RM checkpoints
 


    # ppo needs response to be generated from policy, so configure generation below
    NUM_EPOCHS = 3  # Define the total number of epochs
    # tokenizer.pad_token = tokenizer.eos_token

    output_min_length = 260
    output_max_length = 356

    output_length_sampler = LengthSampler(output_min_length, output_max_length)

 
    generation_kwargs = {
        "min_length": -1,
        "top_k": 50,           # Add top_k filtering
        "top_p": 0.85,        # Slightly more conservative
        "do_sample": True,
        "pad_token_id": tokenizer.pad_token_id,
        "temperature": 0.8,    # Reduce for more stable outputs
    }

    #run the ppo training loop finally, get policy generation, compute rewards and take the ppo step 
    # Initialize variables for tracking rewards
all_rewards = []
average_rewards = []
steps = []

# Directory to save the final aggregated plot
# PLOT_DIR = "./ppo_reward_trajectory_deli_2"
# CHKPT_DIR = "./ppo_friction_checkpoints_deli_2"

#NIPS experiment May 10th, 2025 

PLOT_DIR = "./ppo_reward_trajectory_deli_nips"
CHKPT_DIR = "./ppo_friction_checkpoints_deli_nips"
os.makedirs(PLOT_DIR, exist_ok=True)  # Create the directory if it doesn't exist
os.makedirs(CHKPT_DIR, exist_ok=True)  # Create the directory if it doesn't exist



# Example usage with your dataset
dataset_size = len(friction_dataset)
batch_size = config.batch_size  # From your PPO config
num_epochs = NUM_EPOCHS  # Your defined number of epochs

batches_per_epoch, total_batches = calculate_batches(dataset_size, batch_size, num_epochs)

print(f"Dataset size: {dataset_size}")
print(f"Batch size: {batch_size}")
print(f"Number of epochs: {num_epochs}")
print(f"Batches per epoch: {batches_per_epoch}")
print(f"Total batches for training: {total_batches}")

# Estimate training time if you know average batch processing time
avg_seconds_per_batch = 5  # Example - replace with your actual timing
total_training_time_seconds = total_batches * avg_seconds_per_batch
total_training_time_hours = total_training_time_seconds / 3600

print(f"Estimated training time: {total_training_time_hours:.2f} hours")




# In your training loop
for epoch in range(NUM_EPOCHS):
    print(f"\nStarting Epoch {epoch + 1}/{NUM_EPOCHS}")

    for batch_idx, batch in tqdm(enumerate(friction_ppo_trainer.dataloader), desc=f"Training Epoch {epoch + 1}"):
        if batch_idx == 0:  # Just check the first batch, then continue
            check_ppo_inputs_for_collaboration(batch)
        query_tensors = batch["input_ids"]
        cf_query_tensors = batch["cf_input_ids"]  # Counterfactual queries
        dialogue_contexts = batch.get("dialogue_context", [None] * len(query_tensors))
        golden_friction = batch.get("golden_friction", [None] * len(query_tensors))

        query_tensors = batch["input_ids"]
        cf_query_tensors = batch["cf_input_ids"]  # Get counterfactual queries
        golden_friction = batch["golden_friction"]
        attention_masks = batch["attention_mask"]  # Ensure attention_mask is available
        cf_attention_masks = batch["cf_attention_mask"]
        print("keys in batch of dataloader", batch.keys())
        print("Dialogue Context in batch:", batch.get('dialogue_context'))

        
        # Move tensors to GPU
        query_tensors = [q.to(friction_ppo_trainer.current_device) for q in query_tensors]
        cf_query_tensors = [q.to(friction_ppo_trainer.current_device) for q in cf_query_tensors]
        
        # Generate responses using the model
        response_tensors = []
        responses = []  # Store decoded responses


        print("keys in batch of dataloader", batch.keys())
        print("Dialogue Context in batch:", batch.get('dialogue_context'))

        #### Generate Responses from Main policy (now we want this to be the collaborator policy)
        response_tensors = []
        print(f"Batch {batch_idx + 1}: Generating responses for queries...")

        for query_idx, query in enumerate(query_tensors):
            query_tensors = [q.to("cuda") for q in query_tensors]
            gen_len = output_length_sampler()
            remaining_space = 1124 - query.size(0)  # Calculate remaining space
            generation_kwargs["max_new_tokens"] = max(1, min(gen_len, remaining_space)) #to ensure max new tokens do not become zero as that will fail model.generate

            print("max_new_tokens", min(gen_len, remaining_space))
            response = friction_ppo_trainer.generate(
                query,
#                 attention_mask=attention_masks[query_idx].unsqueeze(0),  # Include if required
                **generation_kwargs,
            )


                 # Generate response
#             response = friction_ppo_trainer.generate(
#                 query=query.unsqueeze(0),
#                 attention_mask=attention_masks[query_idx].unsqueeze(0),
#                 **generation_kwargs,
#             )
            total_length = query.size(0) + response.size(1)
            print("query and response size", query.size(0), response.size(1))
            if total_length > 1124:
                # Truncate the query instead of the response
                truncate_length = total_length - 1124
                query = query[-(query.size(0) - truncate_length):]  # Keep only the last tokens
                print(f"Truncated query for Query {query_idx} to fit within max_length.")

            response_tensors.append(response.squeeze()[-generation_kwargs["max_new_tokens"]:])
            if query_idx % 5 == 0:  # Log every 5 queries
                print(f"Query {query_idx}: Response generated.")

        # Decode responses into text for reward computation
        batch["response"] = [tokenizer.decode(r.squeeze(), skip_special_tokens=True) for r in response_tensors]
        
        #### Compute Rewards Using Friction Classifier
        queries = [tokenizer.decode(q, skip_special_tokens=True) for q in query_tensors]
        responses = batch["response"]
        print("batch respones shape", responses[0])
    
        rewards = []
        rewards_proxy = []
        rewards_gold = []

        agreement_scores = []
        correctness_scores = []
        format_scores = []
        has_vowel_list = []
        has_odd_number_list = []
        has_both_correct_cards_list = []


        
        for i, (response, dialogue_context) in enumerate(zip(responses, dialogue_contexts)):
            # Calculate proxy reward (accuracy-focused)
            proxy_results = calculate_ppo_deli_proxy_reward(response, dialogue_context)
            proxy_reward = proxy_results["total_reward"]
            rewards_proxy.append(proxy_reward)
            
            # Calculate gold reward (agreement-focused) 
            gold_results = calculate_ppo_deli_gold_reward(response, dialogue_context)
            gold_reward = gold_results["total_reward"]
            rewards_gold.append(gold_reward)
            
            # Choose which reward to use for optimization (proxy in this case)
            rewards.append(float(proxy_reward))

              # Extract component scores if available
            agreement_score = proxy_results.get("agreement_score", 0.0)
            agreement_scores.append(float(agreement_score))
            
            correctness_score = proxy_results.get("correctness_score", 0.0)
            correctness_scores.append(float(correctness_score))
            
            format_score = proxy_results.get("format_score", 0.0)
            format_scores.append(float(format_score))
            
            has_vowel = 1.0 if proxy_results.get("has_vowel", False) else 0.0
            has_vowel_list.append(has_vowel)
            
            has_odd_number = 1.0 if proxy_results.get("has_odd_number", False) else 0.0
            has_odd_number_list.append(has_odd_number)
            
            has_both = 1.0 if proxy_results.get("has_both_correct_cards", False) else 0.0
            has_both_correct_cards_list.append(has_both)

        # Convert rewards to tensors
        rewards = [torch.tensor(reward, device=friction_ppo_trainer.current_device).squeeze() for reward in rewards]
        
        # Prepare data for logging
        # Complete DataFrame for local saving (all samples)
        complete_data = {
            "batch": [],
            "epoch": [],
            "sample_idx": [],
            "predicted_intervention": [],
            "Reward_Proxy": [],
            "Reward_Gold": [],
            "Agreement_Score": [],
            "Correctness_Score": [],
            "Format_Score": [],
            "Has_Vowel": [],
            "Has_Odd_Number": [],
            "Has_Both_Cards": [],
            "Golden_Friction_Sample": [],
        }

        # Filtered DataFrame for WandB (only threshold-meeting samples)
        wandb_data = {
            "batch": [],
            "epoch": [],
            "sample_idx": [],
            "predicted_intervention": [],
            "Reward_Proxy": [],
            "Reward_Gold": [],
            "Agreement_Score": [],
            "Correctness_Score": [],
            "Format_Score": [],
            "Has_Vowel": [],
            "Has_Odd_Number": [],
            "Has_Both_Cards": [],
            "Golden_Friction_Sample": [],
        }

        # Display each response with its rewards
        print("Responses and Rewards in this batch:")
        reward_lower_threshold = 0.25
        reward_upper_threshold = 4
        
                
        for idx, (response, reward, gold_friction, proxy_reward, gold_reward, 
                agreement, correctness, format_score, has_v, has_odd, has_both) in enumerate(
            zip(responses, rewards, golden_friction, rewards_proxy, rewards_gold,
                agreement_scores, correctness_scores, format_scores, 
                has_vowel_list, has_odd_number_list, has_both_correct_cards_list)
        ):


                    
            complete_data["batch"].append(batch_idx)
            complete_data["epoch"].append(epoch)
            complete_data["sample_idx"].append(idx)
            complete_data["predicted_intervention"].append(response)
            complete_data["Reward_Proxy"].append(proxy_reward)
            complete_data["Reward_Gold"].append(gold_reward)
            complete_data["Agreement_Score"].append(agreement)
            complete_data["Correctness_Score"].append(correctness)
            complete_data["Format_Score"].append(format_score)
            complete_data["Has_Vowel"].append(has_v)
            complete_data["Has_Odd_Number"].append(has_odd)
            complete_data["Has_Both_Cards"].append(has_both)
            complete_data["Golden_Friction_Sample"].append(gold_friction)
            # Only add to WandB DataFrame if sample meets threshold criteria
            if proxy_reward < reward_lower_threshold or proxy_reward > reward_upper_threshold:
                wandb_data["batch"].append(batch_idx)
                wandb_data["epoch"].append(epoch)
                wandb_data["sample_idx"].append(idx)
                wandb_data["predicted_intervention"].append(response)
                wandb_data["Reward_Proxy"].append(proxy_reward)
                wandb_data["Reward_Gold"].append(gold_reward)
                wandb_data["Agreement_Score"].append(agreement)
                wandb_data["Correctness_Score"].append(correctness)
                wandb_data["Format_Score"].append(format_score)
                wandb_data["Has_Vowel"].append(has_v)
                wandb_data["Has_Odd_Number"].append(has_odd)
                wandb_data["Has_Both_Cards"].append(has_both)
                wandb_data["Golden_Friction_Sample"].append(gold_friction)

        


        # Create and save the complete DataFrame locally
        complete_df = pd.DataFrame(complete_data)
        log_dir = os.path.join(PLOT_DIR, "detailed_logs")
        os.makedirs(log_dir, exist_ok=True)
        log_path = os.path.join(log_dir, f"complete_logs_epoch_{epoch}_batch_{batch_idx}.csv")
        complete_df.to_csv(log_path, index=False)

        # Additionally, append to a running log file for the entire training
        running_log_path = os.path.join(log_dir, "running_complete_logs.csv")
        if not os.path.exists(running_log_path):
            complete_df.to_csv(running_log_path, index=False)
        else:
            complete_df.to_csv(running_log_path, mode='a', header=False, index=False)

        print(f"Complete logs saved to {log_path}")
        # Append rewards for tracking
        all_rewards.extend([r.item() for r in rewards])
        
        # Log as DataFrame in WandB
        # Create the WandB DataFrame (filtered) and log it
        wandb_df = pd.DataFrame(wandb_data)
        wandb.log({"rewards_df": wandb.Table(dataframe=wandb_df)})

        # Periodically save summaries and statistics
        if batch_idx % 50 == 0:
            # Calculate summary statistics from the complete DataFrame
            summary_stats = {
                "avg_proxy_reward": complete_df["Reward_Proxy"].mean(),
                "avg_gold_reward": complete_df["Reward_Gold"].mean(),
                "avg_agreement_score": complete_df["Agreement_Score"].mean(),
                "avg_correctness_score": complete_df["Correctness_Score"].mean(),
                "avg_format_score": complete_df["Format_Score"].mean(),
                "pct_has_vowel": complete_df["Has_Vowel"].mean() * 100,
                "pct_has_odd_number": complete_df["Has_Odd_Number"].mean() * 100,
                "pct_has_both_cards": complete_df["Has_Both_Cards"].mean() * 100
            }
            
            # Log summary to wandb
            wandb.log(summary_stats)
            
            # Create a summary DataFrame and save locally
            summary_df = pd.DataFrame([summary_stats])
            summary_path = os.path.join(log_dir, f"summary_stats_epoch_{epoch}_batch_{batch_idx}.csv")
            summary_df.to_csv(summary_path, index=False)
                
        # Track rewards
        avg_reward = sum([r.item() for r in rewards]) / len(rewards)
        steps.append(len(steps) + 1)  # Incremental step count
        average_rewards.append(avg_reward)
        
        # Run PPO Step
        print(f"Number of query tensors before PPO step: {len(query_tensors)}")
        print(f"Number of response tensors before PPO step: {len(response_tensors)}")
        print(f"Number of rewards before PPO step: {len(rewards)}")
        print(len(rewards))
        
        # Call the step method with counterfactual queries
        stats = friction_ppo_trainer.step(
            queries=query_tensors,
            cf_queries=cf_query_tensors,
            responses=response_tensors,
            scores=rewards,
        )
        
        # Log stats
        friction_ppo_trainer.log_stats(stats, batch, rewards)
        
        # Intermediate logging
        if batch_idx % 10 == 0:
            avg_reward = sum([r.item() for r in rewards]) / len(rewards)
            print(f"  [Epoch {epoch + 1}, Batch {batch_idx + 1}] Avg Reward: {avg_reward:.4f}")
        
        # Plot rewards periodically
        if batch_idx % 50 == 0:
            plt.figure(figsize=(10, 6))
            plt.plot(steps, average_rewards, label="Average Reward", marker="o", color="b")
            plt.xlabel("Steps")
            plt.ylabel("Average Reward")
            plt.title("Trajectory of Rewards During Training")
            plt.legend()
            plt.grid()
            
            # Save the plot
            plot_path = os.path.join(PLOT_DIR, "aggregated_reward_plot.png")
            plt.savefig(plot_path)
            print(f"Aggregated plot saved to {plot_path}")
            plt.close()
        
        # Save checkpoints periodically
        if batch_idx % 200 == 0 and batch_idx > 0:
            save_path = f"{CHKPT_DIR}/ppo_checkpoint_epoch_{epoch + 1}_batch_{batch_idx}"
            os.makedirs(save_path, exist_ok=True)
            
            # Fixed saving logic
            if hasattr(friction_ppo_trainer.model, 'module'):
                # Model is wrapped, access the underlying model
                friction_ppo_trainer.model.module.save_pretrained(save_path)
            else:
                # Model is not wrapped, save directly
                friction_ppo_trainer.model.save_pretrained(save_path)
                
            friction_ppo_trainer.tokenizer.save_pretrained(save_path)
            print(f"Checkpoint saved for Epoch {epoch + 1}, Batch {batch_idx}")
    
    # Save epoch checkpoint
    save_path = f"{CHKPT_DIR}/ppo_checkpoint_epoch_{epoch + 1}"
    os.makedirs(save_path, exist_ok=True)
    
    # Fixed saving logic
    if hasattr(friction_ppo_trainer.model, 'module'):
        friction_ppo_trainer.model.module.save_pretrained(save_path)
    else:
        friction_ppo_trainer.model.save_pretrained(save_path)
        
    friction_ppo_trainer.tokenizer.save_pretrained(save_path)
    print(f"Checkpoint saved for Epoch {epoch + 1}")
    
    print(f"Epoch {epoch + 1} completed.")


