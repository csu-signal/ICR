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
class ScriptArguments:
    """
    The name of the Casual LM model we wish to fine with PPO
    """

    # model_name: Optional[str] = field(default="friction_sft_allsamples_weights_instruct", metadata={"help": "the model name"})
    
    model_name: Optional[str] = field(default="quen_1b_ins_deli_nl/checkpoint-3000", metadata={"help": "the model name"})
    # base_model_name: Optional[str] = field(default="llama3_8b_instruct", metadata={"help": "the model name"})
    base_model_name: Optional[str] = field(default="Qwen/Qwen2-0.5B-Instruct", metadata={"help": "the model name"})

    
    dataset_name: Optional[str] = field(default="Anthropic/hh-rlhf", metadata={"help": "the dataset name"})
    rm_adapter: Optional[str] = field(
        default="trl-lib/llama-7b-hh-rm-adapter", metadata={"help": "the rm adapter name"}
    )
    log_with: Optional[str] = field(default=None, metadata={"help": "use 'wandb' to log with wandb"})
    use_safetensors: Optional[bool] = field(default=False, metadata={"help": "Use safetensors"})
    seed: Optional[int] = field(default=0, metadata={"help": "the random seed"})
    use_score_scaling: Optional[bool] = field(default=False, metadata={"help": "Use score scaling"})
    use_score_norm: Optional[bool] = field(
        default=False, metadata={"help": "Use score normalization. Only applicable if use_score_scaling is True"}
    )
    score_clip: Optional[float] = field(default=None, metadata={"help": "Score clipping"})


 
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



class ICRPPO_trainer(PPOTrainer):
    """
    The ICRPPO_trainer uses Proximal Policy Optimization to optimize ICR agents to learn counterfactually invariant actions in collaboration. 

    Attributes:
        **config** (`PPOConfig`) -- Configuration object for PPOTrainer. Check the documentation of `PPOConfig` for more details.
        **model** (`PreTrainedModelWrapper`) -- Model to be optimized, Hugging Face transformer model with a value head. Check the documentation of `PreTrainedModelWrapper` for more details.
        **ref_model** (`PreTrainedModelWrapper`, *optional*) -- Reference model to be used for KL penalty, Hugging Face transformer model with a casual language modeling head. Check the documentation of `PreTrainedModelWrapper` for more details. If no reference model is provided, the trainer will create a reference model with the same architecture as the model to be optimized with shared layers.
        **tokenizer** (`PreTrainedTokenizerBase`) -- Tokenizer to be used for encoding the data. Check the documentation of `transformers.PreTrainedTokenizer` and `transformers.PreTrainedTokenizerFast` for more details.
        **dataset** (Union[`torch.utils.data.Dataset`, `datasets.Dataset`], *optional*) -- PyTorch dataset or Hugging Face dataset. This is used to create a PyTorch dataloader. If no dataset is provided, the dataloader must be created outside the trainer. Users need to design their own dataloader and ensure the batch size used is the same as the one specified in the configuration object.
        **optimizer** (`torch.optim.Optimizer`, *optional*) -- Optimizer to be used for training. If no optimizer is provided, the trainer will create an Adam optimizer with the learning rate specified in the configuration object.
        **data_collator** (DataCollatorForLanguageModeling, *optional*) -- Data collator to be used for training and passed along the dataloader.
        **num_shared_layers** (int, *optional*) -- Number of layers to be shared between the model and the reference model, if no reference model is passed. If no number is provided, all the layers will be shared.
        **lr_scheduler** (`torch.optim.lr_scheduler`, *optional*) -- Learning rate scheduler to be used for training.
    """

    def __init__(
        self,
        config: Optional[PPOConfig] = None,
        model: Optional[PreTrainedModelWrapper] = None,
        ref_model: Optional[PreTrainedModelWrapper] = None,
        tokenizer: Optional[PreTrainedTokenizerBase] = None,
        dataset: Optional[Union[torch.utils.data.Dataset, Dataset]] = None,
        optimizer: Optional[torch.optim.Optimizer] = None,
        data_collator: Optional[typing.Callable] = None,
        num_shared_layers: Optional[int] = None,
        lr_scheduler: Optional[torch.optim.lr_scheduler._LRScheduler] = None,
        training_data_collator: Optional[typing.Callable] = None,
    ):
        """
        Initialize DensePPOTrainer.

        Args:
            config (`PPOConfig`):
                Configuration object for PPOTrainer. Check the documentation of `PPOConfig` for more details.
            model (`PreTrainedModelWrapper`):
                Hugging Face transformer model with a value head.
            ref_model (`PreTrainedModelWrapper`):
                Hugging Face transformer model with a casual language modeling head. Used for KL penalty.
            tokenizer (`transformers.PreTrainedTokenizerBase`):
                Hugging Face tokenizer.
            dataset (Optional[Union[`torch.utils.data.Dataset`, `datasets.Dataset`]]):
                PyTorch dataset or Hugging Face dataset. If a Hugging Face dataset is passed, the dataset will be preprocessed by removing the columns that are not used by the model. If none is passed, a warning will be raised in a multi-GPU setting.
            optimizer (Optional[`torch.optim.Optimizer`]):
                Optimizer used for training. If `None`, the `Adam` is used as default.
            data_collator (Optional[function]):
                Data collator function that is going to be used for `prepare_dataloader` method. Note this collator is different from the one we use for training. Pass a valid `training_data_collator` instead.
            num_shared_layers (Optional[int]):
                Number of shared layers between the model and the reference model. If `None`, all layers are shared. Used only if `ref_model` is `None`.
            lr_scheduler (Optional[`torch.optim.lr_scheduler`]):
                Learning rate scheduler used for training.
            training_data_collator (Optional[function]):
                Custom data collator used for training.
        """
        super().__init__(
            config=config,
            model=model,
            ref_model=ref_model,
            tokenizer=tokenizer,
            dataset=dataset,
            optimizer=optimizer,
            data_collator=data_collator,
            num_shared_layers=num_shared_layers,
            lr_scheduler=lr_scheduler,
            training_data_collator=training_data_collator,
        )
        # Assign the dataset to an instance variable if it's passed in
        if dataset is not None:
            self.dataset = dataset
            print("Dataset is initialized:", self.dataset)
        else:
            self.dataset = None
            print("No dataset provided.")
        
        if self.dataset is not None:
            self.dataloader = self.prepare_dataloader(self.dataset, data_collator)
        self.intent_kl_coef = 0.1 # setting the KL coeff for the second Dkl term between the main and the counterfactual cf policy
        self.intent_kl_mode = 'token_align'  # Mode: 'token_align' or 'average'

 

    def prepare_dataloader(self, dataset: Union[torch.utils.data.Dataset, Dataset], data_collator=None):
        """
        Prepare the dataloader for training with custom collator that handles counterfactual fields.
        """

        def custom_data_collator(batch):
            # Include both factual and counterfactual fields
            return {
                'input_ids': torch.stack([item['input_ids'] for item in batch]),
                'attention_mask': torch.stack([item['attention_mask'] for item in batch]),
                'cf_input_ids': torch.stack([item['cf_input_ids'] for item in batch]),
                'cf_attention_mask': torch.stack([item['cf_attention_mask'] for item in batch]),
                'label': torch.stack([item['label'] for item in batch]),
                'query': [item['query'] for item in batch],
                'cf_query': [item['cf_query'] for item in batch],
                'golden_friction': [item['golden_friction'] for item in batch],
                'dialogue_context': [item.get('dialogue_context', '') for item in batch],
            }
        if isinstance(dataset, Dataset):
            dataset = self._remove_unused_columns(dataset)
        
        print("Dataset columns before creating dataloader and collator", dataset.column_names)
        
        # Use our custom collator
        data_collator = custom_data_collator if data_collator is None else data_collator
        
        dataloader = torch.utils.data.DataLoader(
            dataset,
            batch_size=self.config.batch_size,
            collate_fn=data_collator,
            shuffle=True,
            drop_last=True,
        )
        return dataloader


    def _set_signature_columns_if_needed(self):
        if self._signature_columns is None:
            # Inspect model forward signature to keep only the arguments it accepts.
            signature = inspect.signature(self.model.forward)
            self._signature_columns = list(signature.parameters.keys())
            # Add our custom fields including counterfactual ones
            self._signature_columns += [
                "label", "query", "cf_query", "response", "dialogue_context", 
                "golden_friction", "cf_input_ids", "cf_attention_mask"
            ]
            print("self signature cols for ppo", self._signature_columns)
    # Adapted from transformers.Trainer._remove_unused_columns
    def _remove_unused_columns(self, dataset: "Dataset"):
        if not self.config.remove_unused_columns:
            return dataset
        self._set_signature_columns_if_needed()
        signature_columns = self._signature_columns

        ignored_columns = list(set(dataset.column_names) - set(signature_columns))
        print("ignored cols",ignored_columns )

        columns = [k for k in signature_columns if k in dataset.column_names]
        print("dataset columns here", dataset.column_names)
        # if version.parse(datasets.__version__) < version.parse("1.4.0"):
        #     dataset.set_format(
        #         type=dataset.format["type"],
        #         columns=columns,
        #         format_kwargs=dataset.format["format_kwargs"],
        #     )
        #     return dataset
        # else:
        return dataset.remove_columns(ignored_columns) 

    def _step_safety_checker_with_cf(
        self, bs, queries, cf_queries, responses, scores, response_masks=None
    ):
        """
        Check inputs to step() method including counterfactual queries.
        """
        # Check for empty queries or responses
        if len(queries) == 0 or len(responses) == 0 or len(cf_queries) == 0:
            raise ValueError("Queries, counterfactual queries, and responses must be non-empty")
        if len(queries) != len(responses) or len(queries) != len(cf_queries):
            raise ValueError("Queries, counterfactual queries, and responses must have the same length")
        if len(scores) != len(queries):
            raise ValueError("Scores must have the same length as queries")
        if response_masks is not None and len(response_masks) != len(queries):
            raise ValueError("Response_masks must have the same length as queries")
        
        # Convert everything to lists of tensors if they aren't already
        if not isinstance(queries[0], torch.Tensor):
            queries = [torch.tensor(query, device=self.current_device) for query in queries]
        if not isinstance(cf_queries[0], torch.Tensor):
            cf_queries = [torch.tensor(query, device=self.current_device) for query in cf_queries]
        if not isinstance(responses[0], torch.Tensor):
            responses = [torch.tensor(response, device=self.current_device) for response in responses]
        if response_masks is None:
            response_masks = [torch.ones_like(response) for response in responses]
        
        # Truncate all lists to be at most batch_size
        if len(queries) > bs:
            queries = queries[:bs]
            cf_queries = cf_queries[:bs]
            responses = responses[:bs]
            scores = scores[:bs]
            response_masks = response_masks[:bs]
        
        return queries, cf_queries, responses, scores, response_masks
    
    @PPODecorators.empty_device_cache()
    def step(
        self,
        queries: List[torch.LongTensor],
        cf_queries: List[torch.LongTensor],  # Add counterfactual queries
        responses: List[torch.LongTensor],
        scores: List[torch.FloatTensor],
        response_masks: Optional[List[torch.LongTensor]] = None,
    ):
        """
        Run a PPO optimisation step with both factual and counterfactual queries.
        
        Args:
            queries (List[`torch.LongTensor`]):
                List of tensors containing the encoded factual queries
            cf_queries (List[`torch.LongTensor`]):
                List of tensors containing the encoded counterfactual queries
            responses (List[`torch.LongTensor`]):
                List of tensors containing the encoded responses
            scores (List[`torch.FloatTensor`]):
                List of tensors containing the scores.
            response_masks (List[`torch.FloatTensor`], *optional*)):
                List of tensors containing masks of the response tokens.

        Returns:
            `dict[str, Any]`: A summary of the training statistics
        """
        bs = self.config.batch_size
        # self.current_device = torch.device("cuda:1" if torch.cuda.is_available() else "cpu")
        queries, cf_queries, responses, scores, response_masks = self._step_safety_checker_with_cf(
        bs, queries, cf_queries, responses, scores, response_masks
    )

        print("Device check before prepare_model_inputs:")
        print(f"Queries device: {queries[0].device}")
        print(f"CF Queries device: {cf_queries[0].device}")
        print(f"Responses device: {responses[0].device}")
        print(f"Current device: {self.current_device}")
        scores = torch.tensor(scores, device=self.current_device)
        cf_queries = [q.to(self.current_device) for q in cf_queries]
        responses = [r.to(self.current_device) for r in responses]
        # Also move response_masks to the correct device
        if response_masks is not None:
            response_masks = [mask.to(self.current_device) for mask in response_masks]
        else:
            # If response_masks is None, create default masks on the correct device
            response_masks = [torch.ones_like(r, device=self.current_device) for r in responses]



        if self.config.use_score_scaling:
            # Score scaling
            scores_mean, scores_std = self.running.update(scores)
            tensor_to_kwargs = dict(dtype=scores.dtype, device=scores.device)
            score_scaling_factor = self.running.std.to(**tensor_to_kwargs) + torch.finfo(scores.dtype).eps
            if self.config.use_score_norm:
                scores = (scores - self.running.mean.to(**tensor_to_kwargs)) / score_scaling_factor
            else:
                scores /= score_scaling_factor

        if self.config.score_clip is not None:
            # Score clipping
            scores_dtype = scores.dtype
            scores = torch.clip(scores.float(), -self.config.score_clip, self.config.score_clip).to(dtype=scores_dtype)

        # if we want to push best model to the hub
        if hasattr(self, "highest_reward"):
            if self.compare_step % self.config.compare_steps == 0:
                curr_mean_reward = scores.mean()
                # if the best reward ever seen
                if curr_mean_reward > self.highest_reward:
                    self.highest_reward = curr_mean_reward
                    # push model to hub
                    self.push_to_hub(**self.push_to_hub_kwargs)
            self.compare_step += 1

        timing = dict()
        t0 = time.time()

        t = time.time()

            # Verify all tensors are now on the correct device
        print("Device check after moving CF queries:")
        print(f"Queries device: {queries[0].device}")
        print(f"CF Queries device: {cf_queries[0].device}")
        print(f"Responses device: {responses[0].device}")

        model_inputs = self.prepare_model_inputs(queries, responses)

        #prepare cf queries as well
        # Prepare model inputs for counterfactual queries (same responses)
        
        cf_model_inputs = self.prepare_model_inputs(cf_queries, responses)

        if self.is_distributed:
            pad_first = self.tokenizer.padding_side == "left"

            model_inputs["input_ids"] = self.accelerator.pad_across_processes(
                model_inputs["input_ids"],
                dim=1,
                pad_index=self.tokenizer.pad_token_id,
                pad_first=pad_first,
            )
            model_inputs["attention_mask"] = self.accelerator.pad_across_processes(
                model_inputs["attention_mask"], dim=1, pad_index=0, pad_first=pad_first
            )
            if self.is_encoder_decoder:
                model_inputs["decoder_input_ids"] = self.accelerator.pad_across_processes(
                    model_inputs["decoder_input_ids"],
                    dim=1,
                    pad_index=self.tokenizer.pad_token_id,
                    pad_first=pad_first,
                )
                model_inputs["decoder_attention_mask"] = self.accelerator.pad_across_processes(
                    model_inputs["decoder_attention_mask"],
                    dim=1,
                    pad_index=0,
                    pad_first=pad_first,
                )

        model_inputs_names = list(model_inputs.keys())

        full_kl_penalty = self.config.kl_penalty == "full"

        with torch.no_grad():
            all_logprobs, logits_or_none, values, masks = self.batched_forward_pass(
                self.model,
                queries,
                responses,
                model_inputs,
                response_masks=response_masks,
                return_logits=full_kl_penalty,
            )

                # Process counterfactual queries with the same model and responses
            cf_logprobs, cf_logits_or_none, _, _ = self.batched_forward_pass(
                self.model,
                cf_queries,
                responses,
                cf_model_inputs,
                response_masks=response_masks,
                return_logits=full_kl_penalty,
            )
            # Get reference model logprobs for standard KL calculation
            with self.optional_peft_ctx():
                ref_logprobs, ref_logits_or_none, _, _ = self.batched_forward_pass(
                    self.model if self.is_peft_model else self.ref_model,
                    queries,
                    responses,
                    model_inputs,
                    return_logits=full_kl_penalty,
                )

        timing["time/ppo/forward_pass"] = time.time() - t

        with torch.no_grad():
            t = time.time()
            if full_kl_penalty:
                print("running IF BLOCK: full_kl_penalty")
                active_full_logprobs = logprobs_from_logits(logits_or_none, None, gather=False)
                ref_full_logprobs = logprobs_from_logits(ref_logits_or_none, None, gather=False)
                cf_full_logprobs = logprobs_from_logits(cf_logits_or_none, None, gather=False) # get logprobs for the CF query
                # rewards, non_score_reward, kls = self.compute_rewards(
                #     scores, active_full_logprobs, ref_full_logprobs, masks
                # )

                # Modified to include intent KL (factual vs counterfactual)
                # rewards, non_score_reward, kls, intent_kls = self.compute_rewards_with_intent(
                #     scores, active_full_logprobs, ref_full_logprobs, cf_full_logprobs, masks
                # )

                # Updated call with queries and responses
                rewards, non_score_reward, kls, intent_kls = self.compute_rewards_with_intent(
                    scores, 
                    active_full_logprobs, 
                    ref_full_logprobs, 
                    cf_full_logprobs, 
                    masks,
                    # queries,  # Add queries parameter
                    # responses  # Add responses parameter
                )
            else:
                # rewards, non_score_reward, kls = self.compute_rewards(scores, all_logprobs, ref_logprobs, masks)
            #     rewards, non_score_reward, kls, intent_kls = self.compute_rewards_with_intent(
            #     scores, all_logprobs, ref_logprobs, cf_logprobs, masks
            # )

                 # Updated call for non-full_kl_penalty case

                print("running ELSE: full_kl_penalty")
                rewards, non_score_reward, kls, intent_kls = self.compute_rewards_with_intent(
                    scores, 
                    all_logprobs, 
                    ref_logprobs, 
                    cf_logprobs, 
                    masks,
                    # queries,  # Add queries parameter
                    # responses  # Add responses parameter
                )
            timing["time/ppo/compute_rewards"] = time.time() - t

            t = time.time()
            values, advantages, returns = self.compute_advantages(values, rewards, masks)
            timing["time/ppo/compute_advantages"] = time.time() - t

        # upcast to float32 to avoid dataset issues
        batch_dict = {
            "queries": queries,
            "responses": responses,
            "logprobs": all_logprobs.to(torch.float32),
            "values": values.to(torch.float32),
            "masks": masks,
            "advantages": advantages,
            "returns": returns,
        }
        batch_dict.update(model_inputs)

        t = time.time()
        all_stats = []
        early_stop = False
        for _ in range(self.config.ppo_epochs):
            if early_stop:
                break
            b_inds = np.random.permutation(bs)
            for backward_batch_start in range(0, bs, self.config.backward_batch_size):
                backward_batch_end = backward_batch_start + self.config.backward_batch_size
                backward_batch_inds = b_inds[backward_batch_start:backward_batch_end]

                for mini_batch_start in range(0, self.config.backward_batch_size, self.config.mini_batch_size):
                    mini_batch_end = mini_batch_start + self.config.mini_batch_size
                    mini_batch_inds = backward_batch_inds[mini_batch_start:mini_batch_end]
                    mini_batch_dict = {
                        "logprobs": batch_dict["logprobs"][mini_batch_inds],
                        "values": batch_dict["values"][mini_batch_inds],
                        "masks": batch_dict["masks"][mini_batch_inds],
                        # hacks: the queries and responses are ragged.
                        "queries": [batch_dict["queries"][i] for i in mini_batch_inds],
                        "responses": [batch_dict["responses"][i] for i in mini_batch_inds],
                        "advantages": batch_dict["advantages"][mini_batch_inds],
                        "returns": batch_dict["returns"][mini_batch_inds],
                    }
                    for k in model_inputs_names:
                        mini_batch_dict[k] = batch_dict[k][mini_batch_inds]
                    with self.accelerator.accumulate(self.model):
                        model_inputs = {k: mini_batch_dict[k] for k in model_inputs_names}

                        logprobs, logits, vpreds, _ = self.batched_forward_pass(
                            self.model,
                            mini_batch_dict["queries"],
                            mini_batch_dict["responses"],
                            model_inputs,
                            return_logits=True,
                        )
                        train_stats = self.train_minibatch(
                            mini_batch_dict["logprobs"],
                            mini_batch_dict["values"],
                            logprobs,
                            logits,
                            vpreds,
                            mini_batch_dict["masks"],
                            mini_batch_dict["advantages"],
                            mini_batch_dict["returns"],
                        )
                        all_stats.append(train_stats)

            # typically, early stopping is done at the epoch level
            if self.config.early_stopping:
                policykl = train_stats["policy/policykl"]
                early_stop = self._early_stop(policykl)
                if early_stop:
                    break

        timing["time/ppo/optimize_step"] = time.time() - t

        t = time.time()
        train_stats = stack_dicts(all_stats)

        # reshape advantages/ratios such that they are not averaged.
        train_stats["policy/advantages"] = torch.flatten(train_stats["policy/advantages"]).unsqueeze(0)
        train_stats["policy/advantages"] = torch.nan_to_num(train_stats["policy/advantages"], WANDB_PADDING)
        train_stats["policy/ratio"] = torch.flatten(train_stats["policy/ratio"]).unsqueeze(0)

        stats = self.record_step_stats(
            scores=scores,
            logprobs=all_logprobs,
            ref_logprobs=ref_logprobs,
            cf_logprobs = cf_logprobs,
            non_score_reward=non_score_reward,
            train_stats=train_stats,
            kl_coef=self.kl_ctl.value,
            intent_kl_coef = self.intent_kl_coef,
            masks=masks,
            queries=queries,
            responses=responses,
            kls=kls,
            intent_kls=intent_kls, 
        )
        # Gather/Reduce stats from all processes
        if self.is_distributed:
            stats = self.gather_stats(stats)
        stats = stats_to_np(stats)
        timing["time/ppo/calc_stats"] = time.time() - t
        stats["ppo/learning_rate"] = self.optimizer.param_groups[0]["lr"]

        # Update the KL control - multiply the batch_size by the number of processes
        self.kl_ctl.update(
            stats["objective/kl"],
            self.config.batch_size * self.accelerator.num_processes,
        )

        # Log the total ppo time
        timing["time/ppo/total"] = time.time() - t0
        stats.update(timing)

        # post-process stats for tensorboard and other loggers
        if self.config.log_with != "wandb":
            stats = convert_to_scalar(stats)

        if self.lr_scheduler is not None:
            self.lr_scheduler.step()

        return stats

    def compute_rewards_with_intent(
        self,
        scores: torch.FloatTensor,
        logprobs: torch.FloatTensor,
        ref_logprobs: torch.FloatTensor,
        cf_logprobs: torch.FloatTensor,
        masks: torch.LongTensor,
    ):
        """
        Compute intent KL using a simpler trimming approach.
        """
        rewards, non_score_rewards, kls, intent_kls = [], [], [], []
        intent_kl_coef = self.intent_kl_coef if hasattr(self, 'intent_kl_coef') else 0.1
        
        for score, logprob, ref_logprob, cf_logprob, mask in zip(
            scores, logprobs, ref_logprobs, cf_logprobs, masks
        ):
            # Standard KL with reference model (unchanged)
            kl = self._kl_penalty(logprob, ref_logprob)
            kls.append(kl)
            
            # Count response tokens (where mask=1)
            response_length = mask.sum().item()
            
            # Get response sections from the end of each sequence
            # This works because responses are at the end of the sequence
            factual_start = max(0, logprob.shape[0] - response_length)
            cf_start = max(0, cf_logprob.shape[0] - response_length)
            
            factual_resp = logprob[factual_start:]
            cf_resp = cf_logprob[cf_start:]
            
            # Compute intent KL on the response parts
            if len(factual_resp) != len(cf_resp):
                # If lengths still don't match, use the minimum length
                min_len = min(len(factual_resp), len(cf_resp))
                factual_resp = factual_resp[-min_len:]
                cf_resp = cf_resp[-min_len:]
            
            intent_kl_resp = self._kl_penalty(factual_resp, cf_resp)
            
            # Create full-sized intent KL tensor
            full_intent_kl = torch.zeros_like(logprob)
            full_intent_kl[factual_start:] = intent_kl_resp
            
            intent_kls.append(full_intent_kl)
            
            # Combined KL penalty
            non_score_reward = -self.kl_ctl.value * kl - intent_kl_coef * full_intent_kl
            non_score_rewards.append(non_score_reward)
            
            # Full reward with score
            reward = non_score_reward.clone()
            last_non_masked_index = mask.nonzero()[-1]
            reward[last_non_masked_index] += score
            rewards.append(reward)
        
        return (
            torch.stack(rewards), 
            torch.stack(non_score_rewards), 
            torch.stack(kls),
            torch.stack(intent_kls)
        )
