import torch
from tqdm import tqdm
from collections import defaultdict
import json
import time
import os
from datetime import datetime
import os
import pandas as pd
from collections import defaultdict
from collections.abc import Mapping
import wandb
# wandb.init(project="friction_agent_inference", name="log_friction_interventions") 
from datasets import Dataset,load_dataset, DatasetDict
from datasets import load_from_disk
import re
import matplotlib.pyplot as plt
import torch
import random
import numpy as np
from tqdm import tqdm
# from datasets import load_metric
from rouge_score import rouge_scorer
from sentence_transformers import SentenceTransformer, util
import sys
import pickle
from dataclasses import dataclass, field
from typing import Optional
import pickle
import pandas as pd
from datasets import Dataset, DatasetDict
from itertools import combinations
import torch
from accelerate import Accelerator
from datasets import load_dataset, load_from_disk
from peft import AutoPeftModelForCausalLM, LoraConfig,PeftModel
from tqdm import tqdm
from transformers import (
    AutoModelForCausalLM, AutoModelForSequenceClassification,
    AutoTokenizer,
    BitsAndBytesConfig,
    HfArgumentParser,
    set_seed,
)
import random
import os
import json
import torch
from datetime import datetime
from tqdm import tqdm
from transformers import AutoTokenizer
from peft import AutoPeftModelForCausalLM
import matplotlib.pyplot as plt
import numpy as np
import os
from datetime import datetime
import pandas as pd
import seaborn as sns
from nltk.tokenize import word_tokenize
from nltk.translate.bleu_score import corpus_bleu, SmoothingFunction
from rouge_score import rouge_scorer
from sentence_transformers import SentenceTransformer, util
import torch
from transformers import pipeline

from openai import OpenAI
client = OpenAI(api_key="###")

# Load models for metrics computation
print("Loading models...")
semantic_model = SentenceTransformer('all-MiniLM-L6-v2')
nli_model = pipeline("zero-shot-classification", model="facebook/bart-large-mnli")
rouge_scorer = rouge_scorer.RougeScorer(['rouge1', 'rouge2', 'rougeL'], use_stemmer=True)

# Define the standard blocks
STANDARD_BLOCKS = ["Red", "Blue", "Green", "Purple", "Yellow"]

 
def process_dialogues(
    models_list,
    target_dialog_id_list,
    use_chat_completion,
    dialogs_ranking_rogue,
    tokenizer_base_path,
    output_dir="friction_roleplay_evals",
    max_turns=10,
    generation_args=None,
    chat_client=None,
    gpt_model_name="gpt-4o-mini",
    seed=42,
    reward_model=None, 
    best_of_n=None, 
    top_k_candidates=1, 
    rm_tokenizer=None, 
    rm_max_length=None,
    include_rogue_agent=False,  # New parameter
    rogue_model_path=None,  # Path to rogue agent model,
    rogue_max_turns=None    # Maximum number of turns to include rogue (None = all turns)
):

 
                            
    """
    Process dialogues using multiple models, with the option to use either model.generate or chat completion.
    
    Args:
        models_list (list): List of model paths/names to iterate through
        target_dialog_id_list (list): List of dialogue IDs to process
        use_chat_completion (bool): Whether to use chat completion instead of model.generate
        dialogs_ranking_rogue (dict): Dictionary containing intervention data
        tokenizer_base_path (str): Path to load tokenizer from
        output_dir (str): Directory to save results
        max_turns (int): Maximum number of dialogue turns
        generation_args (dict): Arguments for model generation
        chat_client: Client for chat completion API calls
        gpt_model_name (str): Name of the GPT model to use for chat completion
        seed (int): Random seed for reproducibility
        
    Returns:
        dict: All conversations organized by model
    """

    
    # Set default generation args if not provided
    if generation_args is None:
        generation_args = {
            "max_new_tokens": 256,
            "temperature": 0.0,
            "do_sample": True,
            "top_k": 50,
            "top_p": 0.9,
            "num_beams": 5,
            "min_length": 100,
            'num_return_sequences': 1
        }
    
    # Initialize data structures to store results
    all_models_conversations = {}
    
    # Create output directory
    os.makedirs(output_dir, exist_ok=True)
    
    # Current timestamp for filenames
    current_time = datetime.now().strftime("%Y%m%d_%H%M%S")
    
    # Create the main log file
    main_log_filename = f"{output_dir}/all_models_dialogue_generation_{current_time}.json"
    
    # Create the main markdown file
    main_md_filename = f"{output_dir}/all_models_dialogues_readable_{current_time}.md"
    
    # Initialize the markdown file with a header
    with open(main_md_filename, 'w') as f:
        f.write(f"# All Models Dialogue Generation Results - {current_time}\n\n")

 
    wtd_test = load_from_disk("selected_validation_dataset_friction_mixed_with_original")
    target_dialog_ids = [int(x) for x in wtd_test['dialog_id'] if x is not None]
    target_dialog_id_list = target_dialog_ids[0:100] + target_dialog_id_list
    # target_dialog_id_list = target_dialog_id_list
    
    print("FINAL TARGET ID LIST", target_dialog_id_list, len(target_dialog_id_list))
 
    
    # Loop through models
    for model_name in models_list:
        loading_model_name = model_name

        if "/" in model_name:
            parts = model_name.split("/")
            if len(parts) >= 2:
                # Combine the first two parts
                model_name = parts[0] + "_" + parts[1]
        if best_of_n:
            model_name = model_name + f"best_of_{best_of_n}"
            
        print(f"\n===== Processing Model: {model_name} =====\n")
        
        # Create model-specific log files
        model_log_filename = f"{output_dir}/dialogue_generation_log_{os.path.basename(model_name)}_{current_time}.json"
        model_md_filename = f"{output_dir}/dialogues_readable_{os.path.basename(model_name)}_{current_time}.md"
        
        # Initialize the model-specific markdown file
        with open(model_md_filename, 'w') as f:
            f.write(f"# Dialogue Generation Results for {model_name} - {current_time}\n\n")
        
        # Initialize data for this model
        all_conversations = {}
        
        # Load model and tokenizer if not using chat completion
        if not use_chat_completion:
            print(f"Loading model from {model_name}...")

 
            lora_model = AutoModelForCausalLM.from_pretrained(
                script_args.base_model_name_or_path,
                device_map="auto",
        
                low_cpu_mem_usage=True,
                torch_dtype=torch.float16,
            trust_remote_code=True,
            
            )

            lora_model = PeftModel.from_pretrained(
            lora_model,
            loading_model_name,
            torch_dtype=torch.float16,
            device_map="auto",
            low_cpu_mem_usage=True,
            trust_remote_code=True,
        )
            # Merge the model
            print("Merging LoRA adapter...")
            merged_model = lora_model.merge_and_unload()
                
            # # Load the model
            # lora_model = AutoPeftModelForCausalLM.from_pretrained(
            #     loading_model_name,
            #     device_map="auto",
            #     torch_dtype=torch.bfloat16,
            #     trust_remote_code=True,
            # )
            
            # # Merge the model
            # print("Merging LoRA adapter...")
            # merged_model = lora_model.merge_and_unload()
            
            # Load tokenizer
            tokenizer = AutoTokenizer.from_pretrained(loading_model_name)
            tokenizer.pad_token = "<|reserved_special_token_0|>"
            tokenizer.padding_side = "right"
        else:
            print(f"Using chat completion with model {gpt_model_name}...")
            merged_model = None
            tokenizer = None


                # Load rogue model if needed
        if include_rogue_agent and not use_chat_completion and rogue_model_path:
            print(f"Loading rogue model from {rogue_model_path}...")

            # print(f"Loading model from {model_name}...")

 
            rogue_lora_model = AutoModelForCausalLM.from_pretrained(
                script_args.base_model_name_or_path,
                device_map="auto",
        
                low_cpu_mem_usage=True,
                torch_dtype=torch.float16,
            trust_remote_code=True,
            
            )

            rogue_lora_model = PeftModel.from_pretrained(
            rogue_lora_model,
            rogue_model_path,
            torch_dtype=torch.float16,
            device_map="auto",
            low_cpu_mem_usage=True,
            trust_remote_code=True,
        )
            # Merge the model
            print("Merging LoRA adapter...")
            rogue_model = rogue_lora_model.merge_and_unload()
            
            # # Load the rogue model
            # rogue_lora_model = AutoPeftModelForCausalLM.from_pretrained(
            #     rogue_model_path,
            #     device_map="auto",
            #     torch_dtype=torch.bfloat16,
            #     trust_remote_code=True,
            # )
            
            # # Merge the model
            # print("Merging rogue LoRA adapter...")
            # rogue_model = rogue_lora_model.merge_and_unload()
            # print("Rogue model loaded successfully", rogue_model)
        else:
            rogue_model = None
        
        # Calculate save frequency
        save_frequency = max(1, len(target_dialog_id_list) // 3)
        print(f"Will save results after every {save_frequency} dialogues processed")
        
        # Counter for processed dialogues
        processed_count = 0
        
        # Main loop to iterate through dialogues
        for index, (key, entry) in enumerate(tqdm(dialogs_ranking_rogue.items(), desc=f"Processing dialogues with {model_name}")):
            original_friction = entry['friction_data_original']
            target_id = original_friction['dialog_id']
            
            # Skip if we've already processed this ID
            if target_id in all_conversations:
                continue
            
            # Process only IDs in our target list
            if target_id in target_dialog_id_list:
                print(f"Processing dialogue ID: {target_id}")
                
                # Initialize conversation record
                conversation_record = {
                    'dialog_id': target_id,
                    'original_context': original_friction['previous_utterance_history'],
                    'gold_friction_bootstrap': original_friction['friction_statement'],
                    'personalities': {
                        'P1': original_friction['P1_personality_type'] + ":" + original_friction['P1_facet'],
                        'P2': original_friction['P2_personality_type'] + ":" + original_friction['P2_facet'],
                        'P3': original_friction['P3_personality_type'] + ":" + original_friction['P3_facet']
                    },
                    'turns': []
                }
                
                print("Previous Utterance History:", original_friction['previous_utterance_history'])
                print("Friction GOLD:", original_friction['friction_statement'])
                
                # Initialize the dialogue history with the original context
                current_dialogue_history = original_friction['previous_utterance_history']
                
                # Main conversation generation loop
                for turn in range(max_turns):
                    print(f"\n----- TURN {turn+1} -----")
                    turn_data = {
                        'turn_number': turn + 1,
                        'dialogue_before_friction': current_dialogue_history,
                        'timestamp': datetime.now().strftime("%Y-%m-%d %H:%M:%S")
                    }

                    # adding the rogue agent feature

                    # Inside the main turn loop, add this before the intervention agent generation step

                    # STEP 0: Generate Rogue Agent Response (if include_rogue_agent is True)
                    if include_rogue_agent and (rogue_max_turns is None or turn < rogue_max_turns):
                        turn_data['has_rogue_intervention'] = True
                    # if include_rogue_agent:
                    #     # Initialize turn data with rogue information
                    #     turn_data['has_rogue_intervention'] = True
                        
                        # Prepare inputs for rogue agent (similar to intervention agent)
                        rogue_inputs = {}
                        rogue_inputs['context'] = current_dialogue_history
                        
                        # Apply the same truncation logic if needed
                        if len(current_dialogue_history.split()) > 500:
                            if not use_chat_completion:
                                original_length = len(tokenizer.encode(current_dialogue_history))
                                rogue_inputs['context'] = truncate_dialogue_history(
                                    rogue_inputs['context'], 
                                    tokenizer, 
                                    max_tokens=6000
                                )
                        
                        # Process with appropriate template
                        if not use_chat_completion:
                            rogue_inputs.update(process_data_template(rogue_inputs))
                        else:
                            rogue_inputs.update(process_data_template_chat_completion_rogue_version(rogue_inputs))
                        
                        print(f"Generating rogue intervention for turn {turn+1}...")
                        
                        try:
                            # Generate rogue response similar to intervention agent
                            if use_chat_completion:
                                # Using chat completion for rogue
                                rogue_completion = chat_client.chat.completions.create(
                                    model=gpt_model_name,
                                    messages=[
                                        {"role": "system", "content": "You are a deceptive agent that introduces subtle manipulations into conversations."},
                                        {"role": "user", "content": rogue_inputs['prompt']}
                                    ]
                                )
                                
                                if rogue_completion and rogue_completion.choices:
                                    rogue_text_to_parse = rogue_completion.choices[0].message.content
                                    turn_data['rogue_generated_text'] = rogue_text_to_parse
                                else:
                                    print("Failed to get rogue chat completion")
                                    turn_data['rogue_generation_error'] = "Empty or invalid rogue chat completion"
                            else:
                                # Using model.generate for rogue
                                rogue_model = rogue_model  # This would be loaded similar to intervention model
                                
                                rogue_generated_texts, rogue_all_generated_texts = generate_multiple_sequences_with_intrinsic_metrics(
                                    rogue_model, 
                                    tokenizer, 
                                    rogue_inputs['prompt'], 
                                    generation_args, 
                                    None,
                                    strategy="top_p_sampling", 
                                    batched=True
                                )
                                
                                # Process the generated text
                                if rogue_generated_texts and isinstance(rogue_generated_texts, list):
                                    rogue_text_to_parse = rogue_generated_texts[0][0] if (rogue_generated_texts[0] and isinstance(rogue_generated_texts[0], list)) else rogue_generated_texts[0]
                                    turn_data['rogue_generated_text'] = rogue_text_to_parse
                                else:
                                    print("Failed to generate rogue text")
                                    turn_data['rogue_generation_error'] = "Empty or invalid rogue generated text"
                            
                            # Parse tags from generated text (same parsing for both methods)
                            parsed_rogue_response = parse_tags_robust(rogue_text_to_parse, tags_for_parsing)
                            
                            # Extract rogue intervention
                            rogue_intervention = ' '.join(parsed_rogue_response.get('intervention', []))
                            if not rogue_intervention:
                                rogue_intervention = handle_friction_logic(rogue_text_to_parse)
                            
                            # Store rogue data
                            turn_data['parsed_rogue'] = rogue_intervention
                            
                            # Randomly choose which persona will "speak" the rogue intervention
                            rogue_persona = random.randint(1, 3)
                            turn_data['rogue_persona'] = f"P{rogue_persona}"
                            
                            # Format rogue intervention as persona speech
                            rogue_formatted = f"P{rogue_persona}: {rogue_intervention}"
                            
                            # Update dialogue history to include rogue intervention
                            current_dialogue_history = current_dialogue_history + "\n" + rogue_formatted
                            turn_data['dialogue_with_rogue'] = current_dialogue_history
                            
                            print(f"Generated rogue intervention as P{rogue_persona}: {rogue_intervention[:100]}...")
                            turn_data.update({
                            'has_rogue_intervention': True,
                            'rogue_generated_text': rogue_text_to_parse,
                            'parsed_rogue': rogue_intervention,
                            'rogue_persona': f"P{rogue_persona}",
                            'rogue_formatted': rogue_formatted,
                            'dialogue_with_rogue': current_dialogue_history
                            })
                        except Exception as e:
                            print(f"Error in rogue generation: {str(e)}")
                            turn_data['rogue_generation_error'] = str(e)

                    else:
                        # Skip rogue agent for this turn, but don't break the loop
                        if include_rogue_agent and rogue_max_turns is not None and turn >= rogue_max_turns:
                            print(f"Skipping rogue agent for turn {turn+1} (exceeded rogue_max_turns={rogue_max_turns})")

                    # STEP 1: Generate intervention with your model (now after rogue if enabled)
                    # [Existing intervention generation code continues here...]
                    
                    # STEP 1: Generate intervention with your model
                    policy_inputs = {}
                    policy_inputs['context'] = current_dialogue_history

                    # Truncate the dialogue history if it's getting too long
                    if len(current_dialogue_history.split()) > 500:  # Simple heuristic to check if context is long
                        if not use_chat_completion:
                            # Debug statements to track truncation 
                            original_length = len(tokenizer.encode(current_dialogue_history))
                            print(f"[DEBUG] Original dialogue history: {original_length} tokens, {len(current_dialogue_history.split())} words")
                            tokenizer = tokenizer  # Use the tokenizer that's already available in your scope
                            policy_inputs['context'] = truncate_dialogue_history(
                                policy_inputs['context'], 
                                tokenizer, 
                                max_tokens=6000
                            )

                            # After truncation
                            truncated_length = len(tokenizer.encode(policy_inputs['context']))
                            print(f"[DEBUG] After truncation: {truncated_length} tokens ({(truncated_length/original_length)*100:.1f}% of original)")


                    # Choose the appropriate template based on chat completion flag
                    if not use_chat_completion:
                        policy_inputs.update(process_data_template(policy_inputs))
                    else:
                        policy_inputs.update(process_data_template_chat_completion(policy_inputs))
                    # policy_inputs.update(process_data_template(policy_inputs)) if not use_chat_completion else  policy_inputs.update(process_data_template_chat_completion(policy_inputs))

                    
                    print(f"Generating intervention for turn {turn+1}...")
                    
                    try:
                        # Choose between model.generate and chat completion
                        if use_chat_completion:
                            # Using chat completion
                            completion = chat_client.chat.completions.create(
                                model=gpt_model_name,
                                messages=[
                                    {"role": "system", "content": "You are a intervention agent that identifies conversational issues and provides interventions."},
                                    {"role": "user", "content": policy_inputs['prompt']}
                                ]
                            )
                            
                            if completion and completion.choices:
                                text_to_parse = completion.choices[0].message.content
                                turn_data['model_generated_text'] = text_to_parse
                            else:
                                print("Failed to get chat completion")
                                turn_data['generation_error'] = "Empty or invalid chat completion"
                                conversation_record['turns'].append(turn_data)
                                break
                        else:
                            # Using model.generate
                            if best_of_n:
                                print("Running BON")
                                device = 'cuda:0'

                                generated_texts, all_generated_texts = generate_multiple_sequences_with_intrinsic_metrics(
                                    merged_model, 
                                    tokenizer, 
                                    policy_inputs['prompt'], 
                                    generation_args, 
                                    device = device,
                                    # None,  # Don't specify device
                                    strategy="best_of_n", 
                                    batched=True,reward_model=reward_model, best_of_n=best_of_n, 
                                    top_k_candidates=top_k_candidates, rm_tokenizer = rm_tokenizer, rm_max_length = rm_max_length
                                )

                                
                            else:

                                generated_texts, all_generated_texts = generate_multiple_sequences_with_intrinsic_metrics(
                                    merged_model, 
                                    tokenizer, 
                                    policy_inputs['prompt'], 
                                    generation_args, 
                                    None,  # Don't specify device
                                    strategy="top_p_sampling", 
                                    batched=True
                                )

                                # just call it again, parse and append 


                        
                            # Process the generated text
                            if generated_texts and isinstance(generated_texts, list):
                                text_to_parse = generated_texts[0][0] if (generated_texts[0] and isinstance(generated_texts[0], list)) else generated_texts[0]
                                turn_data['model_generated_text'] = text_to_parse
                            else:
                                print("Failed to generate text")
                                turn_data['generation_error'] = "Empty or invalid generated text"
                                conversation_record['turns'].append(turn_data)
                                break
                        
                        # Parse tags from generated text (same parsing for both methods)
                        parsed_frictive_states_and_friction = parse_tags_robust(text_to_parse, tags_for_parsing)
                        
                        # Extract components
                        friction_intervention = ' '.join(parsed_frictive_states_and_friction.get('intervention', []))
                        if not friction_intervention:
                            friction_intervention = handle_friction_logic(text_to_parse)
                        
                        task_state = ' '.join(parsed_frictive_states_and_friction.get('t', []))
                        belief_state = ' '.join(parsed_frictive_states_and_friction.get('b', []))
                        rationale = ' '.join(parsed_frictive_states_and_friction.get('rationale', []))
                        
                        # Store in turn data
                        turn_data.update({
                            'parsed_friction': friction_intervention,
                            'task_state': task_state,
                            'belief_state': belief_state,
                            'rationale': rationale
                        })
                        
                        print(f"Generated intervention: {friction_intervention}")
                    except Exception as e:
                        print(f"Error in generation: {str(e)}")
                        turn_data['generation_error'] = str(e)
                        conversation_record['turns'].append(turn_data)
                        break
                    
                    # STEP 2: Add intervention to dialogue and prepare GPT prompt
                    updated_dialogue = current_dialogue_history + "\nFriction Agent: " + friction_intervention
                    turn_data['dialogue_with_friction'] = updated_dialogue
                    
                    # Format GPT prompt with updated dialogue
                    if turn == 0:
                        gpt_prompt = gpt_user_continuation_prompt_bootstrap.format(
                            dialogue=updated_dialogue,
                            P1_personality_facet=original_friction['P1_personality_type'] + ":" + original_friction['P1_facet'],
                            P2_personality_facet=original_friction['P2_personality_type'] + ":" + original_friction['P2_facet'],
                            P3_personality_facet=original_friction['P3_personality_type'] + ":" + original_friction['P3_facet']
                        )
                    else:
                        gpt_prompt = gpt_user_continuation_prompt_onwards.format(
                            dialogue=updated_dialogue,
                            P1_personality_facet=original_friction['P1_personality_type'] + ":" + original_friction['P1_facet'],
                            P2_personality_facet=original_friction['P2_personality_type'] + ":" + original_friction['P2_facet'],
                            P3_personality_facet=original_friction['P3_personality_type'] + ":" + original_friction['P3_facet']
                        )
                    
                    turn_data['gpt_prompt'] = gpt_prompt
                    
                    # STEP 3: Get GPT response
                    print(f"Calling GPT for turn {turn+1}...")
                    
                    try:
                        completion = chat_client.chat.completions.create(
                            model=gpt_model_name,
                            messages=[
                                {"role": "system", "content": gpt_system_prompt},
                                {"role": "user", "content": gpt_prompt}
                            ]
                        )
                        
                        if completion and completion.choices:
                            message_content = completion.choices[0].message.content
                            turn_data['gpt_raw_response'] = message_content
                            
                            # Parse GPT response
                            parsed_gpt = parse_gpt_completion(message_content)
                            turn_data['parsed_gpt_response'] = parsed_gpt
                            
                            if parsed_gpt and 'utterances' in parsed_gpt:
                                gpt_utterances = parsed_gpt['utterances']
                                turn_data['gpt_utterances'] = gpt_utterances
                                
                                # Join utterances for dialogue continuation
                                gpt_continuation = " ".join(gpt_utterances)
                                print(f"GPT continuation: {gpt_continuation[:100]}...")
                            else:
                                print("Failed to parse GPT response")
                                gpt_continuation = "No valid continuation provided."
                                turn_data['gpt_continuation_error'] = "Failed to parse response"
                        else:
                            print(f"No valid GPT response for turn {turn+1}")
                            gpt_continuation = "No response from assistant."
                            turn_data['gpt_continuation_error'] = "No choices in GPT response"
                    except Exception as e:
                        print(f"Error calling GPT API: {str(e)}")
                        gpt_continuation = "Error in API call."
                        turn_data['gpt_api_error'] = str(e)
                    
                    # STEP 4: Update dialogue history for next turn
                    current_dialogue_history = updated_dialogue + "\n" + gpt_continuation
                    turn_data['updated_dialogue'] = current_dialogue_history
                    
                    # Add this turn's data to conversation record
                    conversation_record['turns'].append(turn_data)
                    
                    # STEP 5: Check stopping condition
                    if parsed_gpt and check_stopping_condition(parsed_gpt):
                        print(f"All blocks resolved! Ending simulation after {turn+1} turns.")
                        conversation_record['stopping_reason'] = "All blocks resolved"
                        # break
                        continue
                
                # If we reached max turns without stopping condition
                if turn == max_turns - 1:
                    conversation_record['stopping_reason'] = "Maximum turns reached"
                
                # Save the completed conversation to the overall record
                all_conversations[target_id] = conversation_record
                processed_count += 1
                
                # Append this dialogue to the model-specific markdown file
                            
                with open(model_md_filename, 'a') as f:
                    f.write(f"\n## Dialogue {target_id}\n\n")
                    f.write(f"### Original Context\n\n```\n{original_friction['previous_utterance_history']}\n```\n\n")
                    f.write(f"### Gold Friction Bootstrap (T =1) \n\n```\n{original_friction['friction_statement']}\n```\n\n")
                    
                    for turn_idx, turn in enumerate(conversation_record['turns']):
                        f.write(f"### Turn {turn_idx + 1}\n\n")
                        # Add rogue intervention if present
                        if 'has_rogue_intervention' in turn and turn['has_rogue_intervention']:
                            f.write(f"**Rogue ({turn.get('rogue_persona', 'Unknown')})**: {turn.get('parsed_rogue', 'N/A')}\n\n")
                        
                        f.write(f"**Friction**: {turn.get('parsed_friction', 'N/A')}\n\n")
                        
                        if 'gpt_utterances' in turn:
                            f.write("**GPT Utterances**:\n\n")
                            for utterance in turn['gpt_utterances']:
                                f.write(f"- {utterance}\n")
                            f.write("\n")
                    
                    f.write("---\n\n")  # Separator between dialogues
                
                # Periodic save of JSON after processing every 1/3 of target dialogues
                if processed_count % save_frequency == 0 or processed_count == len(target_dialog_id_list):
                    with open(model_log_filename, 'w') as f:
                        json.dump(all_conversations, f, indent=2)
                    print(f"Saved results to {model_log_filename} after processing {processed_count}/{len(target_dialog_id_list)} dialogues")
                
                # Check if we've processed all target dialogues
                if len(all_conversations) == len(target_dialog_id_list):
                    print("All target dialogues processed for this model. Moving to next model.")
                    break
        
        # Store results for this model
        all_models_conversations[model_name] = all_conversations
        
        # Add a section for this model in the main markdown file
        with open(main_md_filename, 'a') as f:
            f.write(f"# Model: {model_name}\n\n")
            f.write(f"Processed {len(all_conversations)} dialogues\n\n")
            f.write(f"Detailed results in: {model_md_filename}\n\n")
            f.write("---\n\n")  # Separator between models
        
        # Save individual model results
        with open(model_log_filename, 'w') as f:
            json.dump(all_conversations, f, indent=2)
        print(f"Saved final results for model {model_name} to {model_log_filename}")

        if not use_chat_completion:
            del lora_model
            del merged_model
            del rogue_lora_model
            del rogue_model
            torch.cuda.empty_cache()
            import gc
            gc.collect()
            
    # Save final combined results
    with open(main_log_filename, 'w') as f:
        json.dump(all_models_conversations, f, indent=2)
    print(f"Saved combined results for all models to {main_log_filename}")
    print(f"Summary of all model results saved to {main_md_filename}")
    
    return all_models_conversations, main_log_filename



def compute_metrics(data):
    """
    Compute metrics for all dialogues.
    
    Parameters:
    - data: Dictionary containing dialogue data
    
    Returns:
    - Dictionary with computed metrics
    """
    metrics = {
        'quality': {},
        'efficiency': {},
        'persuasiveness': {},
        'friction_scores': {},  # Add new intervention scores category
        'dialogues': [],  # Store detailed metrics for each dialogue
    }
    
    # Process each dialogue
    for dialogue_id, dialogue_data in tqdm(data.items(), desc="Processing dialogues"):
        dialogue_metrics = compute_dialogue_metrics(dialogue_data)
        metrics['dialogues'].append(dialogue_metrics)
       
    
    # Compute average metrics across all dialogues
    metrics['quality'] = compute_average_quality_metrics(metrics['dialogues'])
    metrics['efficiency'] = compute_average_efficiency_metrics(metrics['dialogues'])
    metrics['persuasiveness'] = compute_average_persuasiveness_metrics(metrics['dialogues'])
    metrics['friction_scores'] = compute_average_friction_scores(metrics['dialogues'])
    
    return metrics

def normalize_block_name(block_text):
    """
    Normalize a block name to standardized format.
    
    Parameters:
    - block_text: String containing a block name
    
    Returns:
    - Normalized block name
    """
    if not block_text or not isinstance(block_text, str):
        return None
    
    # Remove parenthetical weight info and punctuation
    block_text = re.sub(r'\(.*?\)', '', block_text).strip()
    block_text = re.sub(r'[,.:;!?]', '', block_text).strip()
    
    # Check against standard block names (case-insensitive)
    for standard_block in STANDARD_BLOCKS:
        if standard_block.lower() in block_text.lower():
            return standard_block
    
    return None

def extract_resolved_blocks(resolved_blocks_list):
    """
    Extract and normalize block names from the resolved_blocks list.
    
    Parameters:
    - resolved_blocks_list: List of strings containing block names
    
    Returns:
    - Set of normalized block names
    """
    normalized_blocks = set()
    
    if not resolved_blocks_list:
        return normalized_blocks
    
    for block_text in resolved_blocks_list:
        normalized_block = normalize_block_name(block_text)
        if normalized_block:
            normalized_blocks.add(normalized_block)
    
    return normalized_blocks

def compute_dialogue_metrics(dialogue_data):
    """
    Compute metrics for a single dialogue.
    
    Parameters:
    - dialogue_data: Dictionary containing data for a single dialogue
    
    Returns:
    - Dictionary with computed metrics for the dialogue
    """
    dialogue_metrics = {
        'dialogue_id': dialogue_data['dialog_id'],
        'total_turns': len(dialogue_data['turns']),
        'blocks_resolved_per_turn': [],
        'quality_metrics': [],
        'resolved_blocks_count': [],
        'turn_metrics': [],
        'personalities': dialogue_data.get('personalities', {}),
        'original_context': dialogue_data.get('original_context', ""),
        'gold_friction_bootstrap': dialogue_data.get('gold_friction_bootstrap', ""),
        'friction_scores':[]
    }
    
    # Track resolved blocks
    resolved_blocks = set()
    prev_resolved_blocks = set()
    print("len of dialogues", len(dialogue_data['turns']))
    # Process each turn
    for i, turn_data in enumerate(dialogue_data['turns']):

        # add intervention scores for each turn and then compute mean, std in compute_average_friction_scores
        # if 'parsed_gpt_response' in turn_data:
        dialogue_metrics['friction_scores'].append(turn_data['parsed_gpt_response']['friction_score'])
            # print(turn_data['parsed_gpt_response']['friction_score'])
        
        turn_metrics = compute_turn_metrics(turn_data, i, resolved_blocks, prev_resolved_blocks)
        dialogue_metrics['turn_metrics'].append(turn_metrics)
        
        # Update resolved blocks tracking
        if 'parsed_gpt_response' in turn_data and 'resolved_blocks' in turn_data['parsed_gpt_response']:
            curr_resolved = extract_resolved_blocks(turn_data['parsed_gpt_response']['resolved_blocks'])
            prev_resolved_blocks = resolved_blocks.copy()
            resolved_blocks = resolved_blocks.union(curr_resolved)
        
        # Store current resolved blocks count and blocks resolved in this turn
        dialogue_metrics['resolved_blocks_count'].append(len(resolved_blocks))
        blocks_in_turn = len(resolved_blocks) - len(prev_resolved_blocks)
        dialogue_metrics['blocks_resolved_per_turn'].append(blocks_in_turn)
    
    # Compute dialogue-level metrics
    dialogue_metrics['final_blocks_resolved'] = len(resolved_blocks)
    dialogue_metrics['resolution_rate'] = len(resolved_blocks) / len(STANDARD_BLOCKS)
    
    # List the actual resolved blocks
    dialogue_metrics['resolved_blocks'] = list(resolved_blocks)
    
    # Check if all blocks were resolved
    all_blocks_resolved = len(resolved_blocks) == len(STANDARD_BLOCKS)
    dialogue_metrics['all_blocks_resolved'] = all_blocks_resolved
    
    # Calculate turns until resolution if all blocks were resolved
    if all_blocks_resolved:
        for i, count in enumerate(dialogue_metrics['resolved_blocks_count']):
            if count == len(STANDARD_BLOCKS):
                dialogue_metrics['turns_until_resolution'] = i + 1
                break
    else:
        dialogue_metrics['turns_until_resolution'] = dialogue_metrics['total_turns']
    
    return dialogue_metrics

def compute_turn_metrics(turn_data, turn_index, resolved_blocks, prev_resolved_blocks):
    """
    Compute metrics for a single turn.
    
    Parameters:
    - turn_data: Dictionary containing data for a single turn
    - turn_index: Index of the turn
    - resolved_blocks: Set of blocks resolved so far
    - prev_resolved_blocks: Set of blocks resolved before this turn
    
    Returns:
    - Dictionary with computed metrics for the turn
    """
    turn_metrics = {
        'turn_number': turn_index + 1,
        'quality': {},
        'persuasiveness': {},
        'timestamp': turn_data.get('timestamp', ""),
        'parsed_friction': turn_data.get('parsed_friction', ""),
        'gpt_friction': turn_data.get('parsed_gpt_response', {}).get('friction_statement', ""),
        'turn_level_friction_score': turn_data.get('parsed_gpt_response', {}).get('friction_score', "")
    }
    
    # Quality metrics - Semantic similarity between intervention agent and GPT intervention
    if 'parsed_friction' in turn_data and 'parsed_gpt_response' in turn_data and 'friction_statement' in turn_data['parsed_gpt_response'] and turn_data['parsed_gpt_response']['friction_statement']:
        agent_friction = turn_data['parsed_friction']
        gpt_friction = turn_data['parsed_gpt_response']['friction_statement']
        
        # ROUGE scores
        rouge_scores = rouge_scorer.score(agent_friction, gpt_friction)
        turn_metrics['quality']['rouge1_f'] = rouge_scores['rouge1'].fmeasure
        turn_metrics['quality']['rouge2_f'] = rouge_scores['rouge2'].fmeasure
        turn_metrics['quality']['rougeL_f'] = rouge_scores['rougeL'].fmeasure
        
        # BLEU score
        agent_tokens = word_tokenize(agent_friction)
        gpt_tokens = word_tokenize(gpt_friction)
        smoother = SmoothingFunction().method1
        if agent_tokens and gpt_tokens:
            turn_metrics['quality']['bleu'] = corpus_bleu([[gpt_tokens]], [agent_tokens], smoothing_function=smoother)
        else:
            turn_metrics['quality']['bleu'] = 0
        
        # Semantic similarity
        agent_embedding = semantic_model.encode(agent_friction, convert_to_tensor=True)
        gpt_embedding = semantic_model.encode(gpt_friction, convert_to_tensor=True)
        similarity = util.pytorch_cos_sim(agent_embedding, gpt_embedding).item()
        turn_metrics['quality']['semantic_similarity'] = similarity
        
        # NLI analysis - check entailment between agent intervention and GPT intervention
        if len(agent_friction) > 5 and len(gpt_friction) > 5:  # Ensure non-empty strings
            # Agent -> GPT direction
            result_agent_to_gpt = nli_model(agent_friction, [gpt_friction], hypothesis_template="{}.")
            turn_metrics['quality']['agent_to_gpt_entailment'] = result_agent_to_gpt['scores'][0]
            
            # GPT -> Agent direction
            result_gpt_to_agent = nli_model(gpt_friction, [agent_friction], hypothesis_template="{}.")
            turn_metrics['quality']['gpt_to_agent_entailment'] = result_gpt_to_agent['scores'][0]
    
    # Persuasiveness metrics - Check if the intervention was adopted
    if 'parsed_friction' in turn_data and 'gpt_utterances' in turn_data and turn_data['gpt_utterances']:
        agent_friction = turn_data['parsed_friction']
        gpt_utterances = ' '.join(turn_data['gpt_utterances'])
        
        # Semantic similarity between intervention and utterances to check adoption
        agent_embedding = semantic_model.encode(agent_friction, convert_to_tensor=True)
        utterances_embedding = semantic_model.encode(gpt_utterances, convert_to_tensor=True)
        adoption_similarity = util.pytorch_cos_sim(agent_embedding, utterances_embedding).item()
        turn_metrics['persuasiveness']['adoption_similarity'] = adoption_similarity
        
        # Simple heuristic for adoption: consider adopted if similarity > 0.5
        turn_metrics['persuasiveness']['adopted'] = adoption_similarity > 0.5
        
        # Check for acknowledgment without adoption
        # Look for acknowledgment phrases followed by counterpoints
        acknowledgment_patterns = [
            r"(?i)that'?s? (?:a good|an interesting|a valid) point.+but",
            r"(?i)i (?:understand|see|get) (?:your|that) (?:point|concern).+but",
            r"(?i)you'?re? right.+however",
            r"(?i)that'?s? true.+(?:although|though)"
        ]
        acknowledged_without_adoption = any(re.search(pattern, gpt_utterances) for pattern in acknowledgment_patterns)
        turn_metrics['persuasiveness']['acknowledged_without_adoption'] = acknowledged_without_adoption
    
    # Efficiency metrics - Blocks resolved in this turn
    turn_metrics['blocks_resolved'] = len(resolved_blocks) - len(prev_resolved_blocks)
    turn_metrics['total_blocks_resolved'] = len(resolved_blocks)
    turn_metrics['resolved_blocks'] = list(resolved_blocks)
    
    # Store utterances for analysis
    turn_metrics['gpt_utterances'] = turn_data.get('gpt_utterances', [])
    
    return turn_metrics

def compute_average_friction_scores(all_dialogues):
    """
    Compute average intervention scores and their standard deviations across all dialogues.
    
    Args:
        all_dialogues (list): List of dialogue metrics
        
    Returns:
        dict: Statistics about intervention scores
    """
    # Create a list to collect all intervention scores
    friction_scores = []
    
    for dialogue in all_dialogues:
        # friction_scores = dialogue['friction_scores']
        friction_scores.append(dialogue['friction_scores'])
        print(friction_scores, type(friction_scores))
        # for turn in dialogue['turn_metrics']:
        #     print("turn", turn)
       
        #     # Look for friction_score directly in parsed_gpt_response
        #     if 'parsed_gpt_response' in turn and 'friction_score' in turn['parsed_gpt_response']:
        #         print(turn['parsed_gpt_response'])
        #         score = turn['parsed_gpt_response']['friction_score']
        #         if score is not None:
        #             friction_scores.append(score)
    
    # Compute statistics (mean and std)
    stats = {}

    # friction_scores = [item for sublist in friction_scores for item in sublist]
    friction_scores = [item for sublist in friction_scores for item in sublist if item is not None]
    if friction_scores:
        # Using numpy for more accurate statistics
        import numpy as np
        values_array = np.array(friction_scores)
        stats = {
            'mean': float(np.mean(values_array)),
            'std': float(np.std(values_array)),
            'min': float(np.min(values_array)),
            'max': float(np.max(values_array)),
            'count': len(friction_scores)
        }
    else:
        stats = {
            'mean': None,
            'std': None,
            'min': None,
            'max': None,
            'count': 0
        }
    
    return stats

def compute_average_quality_metrics(all_dialogues):
    """
    Compute average quality metrics and their standard deviations across all dialogues.
    """
    quality_metrics = {
        'semantic_similarity': [],
        'rouge1_f': [],
        'rouge2_f': [],
        'rougeL_f': [],
        'bleu': [],
        'agent_to_gpt_entailment': [],
        'gpt_to_agent_entailment': []
    }
    
    # Add bidirectional entailment field
    quality_metrics['bidirectional_entailment'] = []
    
    for dialogue in all_dialogues:
        for turn in dialogue['turn_metrics']:
            if 'quality' in turn and turn['quality']:
                for metric, value in turn['quality'].items():
                    if metric in quality_metrics and value is not None:
                        quality_metrics[metric].append(value)
                
                # Calculate and store bidirectional entailment if both metrics exist
                if 'agent_to_gpt_entailment' in turn['quality'] and 'gpt_to_agent_entailment' in turn['quality']:
                    agent_to_gpt = turn['quality']['agent_to_gpt_entailment']
                    gpt_to_agent = turn['quality']['gpt_to_agent_entailment']
                    if agent_to_gpt is not None and gpt_to_agent is not None:
                        bidirectional = (agent_to_gpt + gpt_to_agent) / 2
                        quality_metrics['bidirectional_entailment'].append(bidirectional)
    
    # Compute statistics (mean and std)
    stats_metrics = {}
    for metric, values in quality_metrics.items():
        if values:
            # Using numpy for more accurate statistics
            import numpy as np
            values_array = np.array(values)
            stats_metrics[metric] = {
                'mean': float(np.mean(values_array)),
                'std': float(np.std(values_array)),
                'min': float(np.min(values_array)),
                'max': float(np.max(values_array)),
                'count': len(values)
            }
        else:
            stats_metrics[metric] = {
                'mean': None,
                'std': None,
                'min': None,
                'max': None,
                'count': 0
            }
    
    return stats_metrics


def compute_average_efficiency_metrics(all_dialogues):
   """
   Compute average efficiency metrics and their standard deviations across all dialogues.
   """
   efficiency_metrics = {
       'total_turns': [],
       'turns_until_resolution': [],
       'resolution_rate': [],
       'blocks_resolved_per_turn': []
   }
   
   for dialogue in all_dialogues:
       efficiency_metrics['total_turns'].append(dialogue['total_turns'])
       efficiency_metrics['turns_until_resolution'].append(dialogue['turns_until_resolution'])
       efficiency_metrics['resolution_rate'].append(dialogue['resolution_rate'])
       
       # Average blocks resolved per turn for this dialogue
       avg_blocks_per_turn = sum(dialogue['blocks_resolved_per_turn']) / dialogue['total_turns']
       efficiency_metrics['blocks_resolved_per_turn'].append(avg_blocks_per_turn)
   
   # Compute statistics (mean and std)
   stats_metrics = {}
   for metric, values in efficiency_metrics.items():
       if values:
           # Using numpy for more accurate statistics
           import numpy as np
           values_array = np.array(values)
           stats_metrics[metric] = {
               'mean': float(np.mean(values_array)),
               'std': float(np.std(values_array)),
               'min': float(np.min(values_array)),
               'max': float(np.max(values_array)),
               'count': len(values)
           }
       else:
           stats_metrics[metric] = {
               'mean': None,
               'std': None,
               'min': None,
               'max': None,
               'count': 0
           }
   
   return stats_metrics

def compute_average_persuasiveness_metrics(all_dialogues):
   """
   Compute average persuasiveness metrics and their standard deviations across all dialogues.
   """
   persuasiveness_metrics = {
       'adoption_rate': [],
       'acknowledgment_without_adoption_rate': [],
       'adoption_similarity': []
   }
   
   for dialogue in all_dialogues:
       dialogue_adoptions = []
       dialogue_acknowledgments = []
       dialogue_similarities = []
       
       for turn in dialogue['turn_metrics']:
           if 'persuasiveness' in turn and turn['persuasiveness']:
               if 'adopted' in turn['persuasiveness']:
                   dialogue_adoptions.append(turn['persuasiveness']['adopted'])
               if 'acknowledged_without_adoption' in turn['persuasiveness']:
                   dialogue_acknowledgments.append(turn['persuasiveness']['acknowledged_without_adoption'])
               if 'adoption_similarity' in turn['persuasiveness']:
                   dialogue_similarities.append(turn['persuasiveness']['adoption_similarity'])
       
       # Calculate rates for this dialogue
       if dialogue_adoptions:
           persuasiveness_metrics['adoption_rate'].append(sum(dialogue_adoptions) / len(dialogue_adoptions))
       if dialogue_acknowledgments:
           persuasiveness_metrics['acknowledgment_without_adoption_rate'].append(sum(dialogue_acknowledgments) / len(dialogue_acknowledgments))
       if dialogue_similarities:
           persuasiveness_metrics['adoption_similarity'].append(sum(dialogue_similarities) / len(dialogue_similarities))
   
   # Compute statistics (mean and std)
   stats_metrics = {}
   for metric, values in persuasiveness_metrics.items():
       if values:
           # Using numpy for more accurate statistics
           import numpy as np
           values_array = np.array(values)
           stats_metrics[metric] = {
               'mean': float(np.mean(values_array)),
               'std': float(np.std(values_array)),
               'min': float(np.min(values_array)),
               'max': float(np.max(values_array)),
               'count': len(values)
           }
       else:
           stats_metrics[metric] = {
               'mean': None,
               'std': None,
               'min': None,
               'max': None,
               'count': 0
           }
   
   return stats_metrics

def get_dialogue_rankings(metrics):
    """
    Rank dialogues by various metrics for analysis.
    
    Parameters:
    - metrics: Dictionary containing computed metrics
    
    Returns:
    - Dictionary with rankings
    """
    dialogues = metrics['dialogues']
    rankings = {
        'fastest_resolution': [],  # Dialogues that resolved all blocks in fewest turns
        'highest_adoption_rate': [],  # Dialogues with highest intervention adoption rate
        'highest_semantic_similarity': [],  # Dialogues with highest agent-GPT intervention similarity
        'most_blocks_resolved': [],  # Dialogues that resolved the most blocks
        'lowest_quality': [],  # Dialogues with lowest quality metrics
    }
    
    # Calculate dialogue-level average metrics for ranking
    for dialogue in dialogues:
        # Calculate average semantic similarity for this dialogue
        avg_similarity = 0
        similarity_count = 0
        adoption_count = 0
        adoption_total = 0
        
        for turn in dialogue['turn_metrics']:
            if 'quality' in turn and 'semantic_similarity' in turn['quality']:
                avg_similarity += turn['quality']['semantic_similarity']
                similarity_count += 1
            
            if 'persuasiveness' in turn and 'adopted' in turn['persuasiveness']:
                adoption_total += 1
                if turn['persuasiveness']['adopted']:
                    adoption_count += 1
        
        dialogue['avg_semantic_similarity'] = avg_similarity / similarity_count if similarity_count > 0 else 0
        dialogue['adoption_rate'] = adoption_count / adoption_total if adoption_total > 0 else 0
    
    # Rank by resolution speed (turns until resolution)
    resolution_sorted = sorted([d for d in dialogues if d.get('all_blocks_resolved', False)], 
                              key=lambda x: x['turns_until_resolution'])
    rankings['fastest_resolution'] = [{'dialogue_id': d['dialogue_id'], 
                                     'turns_until_resolution': d['turns_until_resolution']} 
                                    for d in resolution_sorted[:5]]
    
    # Rank by adoption rate
    adoption_sorted = sorted(dialogues, key=lambda x: x.get('adoption_rate', 0), reverse=True)
    rankings['highest_adoption_rate'] = [{'dialogue_id': d['dialogue_id'], 
                                         'adoption_rate': d.get('adoption_rate', 0)} 
                                        for d in adoption_sorted[:5]]
    
    # Rank by semantic similarity
    similarity_sorted = sorted(dialogues, key=lambda x: x.get('avg_semantic_similarity', 0), reverse=True)
    rankings['highest_semantic_similarity'] = [{'dialogue_id': d['dialogue_id'], 
                                              'avg_semantic_similarity': d.get('avg_semantic_similarity', 0)} 
                                             for d in similarity_sorted[:5]]
    
    # Rank by blocks resolved
    resolved_sorted = sorted(dialogues, key=lambda x: x.get('final_blocks_resolved', 0), reverse=True)
    rankings['most_blocks_resolved'] = [{'dialogue_id': d['dialogue_id'], 
                                        'blocks_resolved': d.get('final_blocks_resolved', 0)} 
                                       for d in resolved_sorted[:5]]
    
    # Rank by lowest quality (for identifying problematic dialogues)
    quality_sorted = sorted(dialogues, key=lambda x: x.get('avg_semantic_similarity', 0))
    rankings['lowest_quality'] = [{'dialogue_id': d['dialogue_id'], 
                                 'avg_semantic_similarity': d.get('avg_semantic_similarity', 0)} 
                                for d in quality_sorted[:5]]
    
    return rankings

def identify_difficult_blocks(metrics):
    """
    Identify which blocks tend to be more difficult to resolve.
    
    Parameters:
    - metrics: Dictionary containing computed metrics
    
    Returns:
    - Dictionary with block difficulty analysis
    """
    block_metrics = {block: {'resolved_count': 0, 'avg_turn_resolved': []} for block in STANDARD_BLOCKS}
    total_dialogues = len(metrics['dialogues'])
    
    for dialogue in metrics['dialogues']:
        # Track when each block was resolved
        resolved_blocks_by_turn = {}
        
        for turn_idx, turn in enumerate(dialogue['turn_metrics']):
            if 'resolved_blocks' in turn:
                for block in turn['resolved_blocks']:
                    if block not in resolved_blocks_by_turn and block in STANDARD_BLOCKS:
                        resolved_blocks_by_turn[block] = turn_idx + 1
        
        # Update block metrics
        for block in STANDARD_BLOCKS:
            if block in resolved_blocks_by_turn:
                block_metrics[block]['resolved_count'] += 1
                block_metrics[block]['avg_turn_resolved'].append(resolved_blocks_by_turn[block])
    
    # Calculate averages and resolution rates
    for block, data in block_metrics.items():
        data['resolution_rate'] = data['resolved_count'] / total_dialogues
        data['avg_turn_to_resolve'] = sum(data['avg_turn_resolved']) / len(data['avg_turn_resolved']) if data['avg_turn_resolved'] else None
    
    # Rank blocks by difficulty (lower resolution rate = more difficult)
    block_difficulty = sorted(STANDARD_BLOCKS, key=lambda x: block_metrics[x]['resolution_rate'])
    
    return {
        'block_metrics': block_metrics,
        'difficulty_ranking': block_difficulty
    }



@dataclass
class ScriptArguments:
    """
    The arguments for the DPO training script.
    """
    base_model_name_or_path: Optional[str] = field(
        default="llama3_8b_instruct",
        metadata={"help": "the location of the SFT model name or path"},
    )
        
    lora_model_name_or_path: Optional[str] = field(
        default="friction_sft_allsamples_weights_instruct",
        metadata={"help": "the location of the SFT model name or path"},
    )

    per_device_eval_batch_size: Optional[int] = field(default=1, metadata={"help": "eval batch size per device"})
    
    dataset: Optional[str] = field(default="ultrafeedback_binarized", metadata={"help": "the dataset used for training and evaluation "})

    max_prompt_length: Optional[int] = field(default=512, metadata={"help": "the maximum prompt length"})
    max_length: Optional[int] = field(default=4096, metadata={"help": "the maximum sequence length"})
    max_new_tokens: Optional[int] = field(default=256, metadata={"help": "the maximum sequence length"})
    
  
    
    output_dir: Optional[str] = field(default="./results_falcon", metadata={"help": "the output directory"})
    log_freq: Optional[int] = field(default=1, metadata={"help": "the logging frequency"})
    load_in_4bit: Optional[bool] = field(default=True, metadata={"help": "whether to load the model in 4bit"})
    model_dtype: Optional[str] = field(
        default="float16", metadata={"help": "model_dtype[float16, bfloat16, float] for loading."}
    )

 
    seed: Optional[int] = field(
        default=0, metadata={"help": "Random seed that will be set at the beginning of training."}
    )



def generate_multiple_sequences_with_intrinsic_metrics(model, tokenizer, prompts, generation_args, device, 
                                                       strategy="beam_search", batched=False, 
                                                       reward_model=None, best_of_n=None, top_k_candidates=1, rm_tokenizer = None
                                                      , rm_max_length = None):
    """
    Generate multiple sequences using various strategies including best-of-N sampling.
    
    Args:
        model: Language model for generation
        tokenizer: Tokenizer for the model
        prompts: Input prompts
        generation_args: Arguments for generation
        device: Device to place tensors on
        strategy: Generation strategy ("beam_search", "top_k_sampling", "top_p_sampling", or "best_of_n")
        batched: Whether inputs are batched
        reward_model: Reward model for scoring in best-of-N sampling (AutoModelForSequenceClassification)
        best_of_n: Number of samples to generate for best-of-N sampling (default: None)
        top_k_candidates: Number of top candidates to return from best-of-N sampling (default: 1)
        
    Returns:
        generated_texts: List of generated texts
        all_generated_texts: List of all generated texts
    """
    if batched:
        tokenizer.pad_token = "<|reserved_special_token_0|>"  # new pad token for this run
        tokenizer.pad_token_id = tokenizer.convert_tokens_to_ids(tokenizer.pad_token)
        tokenizer.padding_side = 'right'

        cleaned_prompts = prompts.replace("\n", " ")  
        inputs = tokenizer(cleaned_prompts, return_tensors="pt", padding=True, truncation=True).to(model.device)
    else:
        tokenizer.pad_token = "<|reserved_special_token_0|>"  # new pad token for this run
        tokenizer.pad_token_id = tokenizer.convert_tokens_to_ids(tokenizer.pad_token)
        tokenizer.padding_side = 'right'
        inputs = tokenizer(prompts, return_tensors="pt", padding=True, truncation=True).to(device)

    input_ids = inputs['input_ids']
    attention_mask = inputs['attention_mask']
    
    # Handle best-of-N sampling strategy
    if strategy == "best_of_n":
        if reward_model is None:
            raise ValueError("Reward model must be provided for best-of-N sampling")
        
        if best_of_n is None or best_of_n <= 0:
            best_of_n = 4  # Default sample size
        
        with torch.no_grad():
            # Generate multiple candidates for each prompt
            all_candidates = []
            all_prompt_candidates = []
            
            # Use top_p or top_k sampling to generate diverse candidates
            sampling_strategy = generation_args.get("sampling_strategy", "top_p_sampling")
            # print("BON sampling strategy", sampling_strategy)
            for _ in range(best_of_n):
                if sampling_strategy == "top_p_sampling": 
                    print("RUNNING TopP sampling for BON")
                    outputs = model.generate(
                        input_ids=input_ids,
                        attention_mask=attention_mask,
                        max_length=input_ids.size(1) + generation_args["max_new_tokens"],
                        temperature=generation_args.get("temperature", 0.7),
                        top_p=generation_args.get("top_p", 0.9),
                        do_sample=True,
                        num_return_sequences=1,
                        eos_token_id=tokenizer.eos_token_id,
                        pad_token_id=tokenizer.pad_token_id,
                        return_dict_in_generate=True,
                    )
                else:  # Default to top_k_sampling
                    outputs = model.generate(
                        input_ids=input_ids,
                        attention_mask=attention_mask,
                        max_length=input_ids.size(1) + generation_args["max_new_tokens"],
                        temperature=generation_args.get("temperature", 0.7),
                        top_k=generation_args.get("top_k", 50),
                        do_sample=True,
                        num_return_sequences=1,
                        eos_token_id=tokenizer.eos_token_id,
                        pad_token_id=tokenizer.pad_token_id,
                        return_dict_in_generate=True,
                    )
                
                # Process the generated sequence
                for i in range(len(outputs.sequences)):
                    sequence = outputs.sequences[i]
                    prompt_length = input_ids.shape[-1]
                    new_tokens = sequence[prompt_length:]
                    generated_text = tokenizer.decode(new_tokens, skip_special_tokens=True)
                    
                    if len(all_candidates) <= i:
                        all_candidates.append([])
                    
                    all_candidates[i].append(generated_text)
            
            # Score candidates with the reward model
            best_candidates = []
            all_prompt_candidates = []
            parsed_candidates = []
            tags_for_parsing = ["intervention", "rationale", "t", "b"]  
            for candidate_index, candidates in enumerate(all_candidates):
                print(f"\n=== Processing candidates for prompt {candidate_index} ===")
                print(f"Number of candidates: {len(candidates)}")

                #parse the generated model outputs to get the intervention + rationale

                        
                for candidate in candidates:
                    parsed_frictive_states_and_friction = parse_tags_robust(candidate, tags_for_parsing)
                    friction_intervention = ' '.join(parsed_frictive_states_and_friction.get('intervention', []))
                    if not friction_intervention:
                        friction_intervention = handle_friction_logic(candidate)

                    rationale = ' '.join(parsed_frictive_states_and_friction.get('rationale', []))
                    friction_and_rationale = rationale + friction_intervention
                    parsed_candidates.append(friction_and_rationale)
                    # print("PARSED intervention + rationale",friction_and_rationale )
                # For each candidate, prepare input for reward model
                candidate_inputs = [prompts + " " + f"</s> {candidate} </s>" for candidate in parsed_candidates]
                tokenized_inputs = rm_tokenizer(candidate_inputs, return_tensors="pt", padding=True, truncation=True, max_length=rm_max_length).to(device)
                
                # Get scores from reward model
                reward_outputs = reward_model(**tokenized_inputs) 
                scores = reward_outputs.logits.squeeze(-1)
                
                ## Print all candidates with their scores
                print("\nAll candidates with scores:")
                for i, (candidate, score) in enumerate(zip(candidates, scores)):
                    print(f"Candidate {i}: Score = {score:.4f}")
                    print(f"Text snippet: {candidate[:50]}...")
                
                # Get top-k indices
                if top_k_candidates > len(candidates):
                    top_k_candidates = len(candidates)
                
                # Fix the error by converting bfloat16 to float32 before calling numpy()
                top_result = torch.topk(scores, top_k_candidates)
                top_indices = top_result.indices.cpu().numpy()
                top_values = top_result.values.cpu().float().numpy()  # Convert to float32 first
                
                print(f"\nTop {top_k_candidates} candidates:")
                # Print only the top-k candidates
                for rank, (idx, score) in enumerate(zip(top_indices, top_values)):
                    print(f"Rank {rank+1}: Candidate {idx} with score {score:.4f}")
                    print(f"Text: {candidates[idx][:300]}...")
                
                # Store the chosen candidates for verification
                prompt_best_candidates = [candidates[idx] for idx in top_indices]
                
                # Verification check
                max_score_idx = scores.argmax().item()
                if max_score_idx != top_indices[0]:
                    print(f"WARNING: Discrepancy detected! argmax={max_score_idx} but topk.indices[0]={top_indices[0]}")
                else:
                    print(f"VERIFIED: Top candidate is correctly selected (index {top_indices[0]})")



                
                best_candidates.append(prompt_best_candidates)
                all_prompt_candidates.extend(candidates)
            
            return best_candidates, all_prompt_candidates
    
    # Original strategies
    with torch.no_grad():
        if strategy == "beam_search":
            outputs = model.generate(
                input_ids=input_ids,
                attention_mask=attention_mask,
                max_length=input_ids.size(1) + generation_args["max_new_tokens"],
                num_beams=generation_args["num_beams"],
                num_return_sequences=generation_args["num_return_sequences"],
                early_stopping=True,
                eos_token_id=tokenizer.eos_token_id,
                pad_token_id=tokenizer.pad_token_id,
                return_dict_in_generate=True,
                output_scores=True
            )
        elif strategy == "top_k_sampling":
            outputs = model.generate(
                input_ids=input_ids,
                attention_mask=attention_mask,
                max_length=input_ids.size(1) + generation_args["max_new_tokens"],
                temperature=generation_args["temperature"],
                top_k=generation_args["top_k"],
                do_sample=True,
                num_return_sequences=generation_args["num_return_sequences"],
                eos_token_id=tokenizer.eos_token_id,
                pad_token_id=tokenizer.pad_token_id,
                min_length=generation_args.get("min_length", 0),
                return_dict_in_generate=True,
                output_scores=True
            )
        elif strategy == "top_p_sampling":
            outputs = model.generate(
                input_ids=input_ids,
                attention_mask=attention_mask,
                max_length=input_ids.size(1) + generation_args["max_new_tokens"],
                temperature=generation_args["temperature"],
                top_p=generation_args["top_p"],
                do_sample=True,
                num_return_sequences=generation_args["num_return_sequences"],
                eos_token_id=tokenizer.eos_token_id,
                pad_token_id=tokenizer.pad_token_id,
                return_dict_in_generate=True,
            )
        else:
            raise ValueError("Unsupported strategy. Use 'beam_search', 'top_k_sampling', 'top_p_sampling', or 'best_of_n'.")

    # Decode the generated tokens for each prompt in the batch
    generated_texts = []
    all_generated_texts = []

    for i in range(0, len(outputs.sequences), generation_args["num_return_sequences"]):
        prompt_texts = []
        prompt_only = []
        for j in range(generation_args["num_return_sequences"]):
            sequence_index = i + j  # Global index for the current sequence
            output = outputs.sequences[sequence_index]
            prompt_length = input_ids.shape[-1]  # Length of the input prompt
            new_tokens = output[prompt_length:]  # Get only the generated tokens
            generated_text = tokenizer.decode(new_tokens, skip_special_tokens=True)
            prompt_tokens = output[:prompt_length]
            prompt_text = tokenizer.decode(prompt_tokens, skip_special_tokens=True)
    
            prompt_texts.append(generated_text)
            prompt_only.append(prompt_text)

        generated_texts.append(prompt_texts)
        all_generated_texts.extend(prompt_only)
    
    return generated_texts, all_generated_texts
        

def handle_friction_logic(text):
    '''
    This function processes a text string to extract or construct a "intervention" snippet by:

    Returning the text following a <intervention> tag if present, unless a closing </intervention> tag is found.
    If no <intervention> tags exist, it constructs a snippet by extracting the first, second-to-last, 
    and last sentences if there are at least three sentences; otherwise, it returns all available sentences.
    
    '''
    if "<intervention>" not in text and "</intervention>" not in text:
        sentences = re.split(r'(?<!\w\.\w.)(?<![A-Z][a-z]\.)(?<=\.|\?)\s', text.strip())
        if len(sentences) >= 3:
            return f"{sentences[0]} {sentences[-2]} {sentences[-1]}"
        elif sentences:
            return " ".join(sentences)
        else:
            return ""
    elif "<intervention>" in text and "</intervention>" not in text:
        friction_start = text.find("<intervention>") + len("<intervention>")
        return text[friction_start:].strip()
    else:
        return ""  # Friction is complete, no need to handle further
    




gpt_system_prompt = """
You are a participant in the Game of Weights, where players deduce the weights of blocks through reasoning and a scale.
[Known Weights (Hidden from Players)]
Red = 10, Blue = 10, Green = 20, Purple = 30, Yellow = 50
Your task is to continue the dialogue until all block weights are resolved or agreed upon.
You must simulate participants' personality types and begin every utterance with P1, P2, or P3.
IMPORTANT: Within the dialogue, you should ONLY respond as P1, P2, and P3.
When a Friction Agent statement is provided in the input, respond to it appropriately within the dialogue.
At the END of your response, you should analyze the conversation and generate your own intervention point inside <intervention>...</intervention> tags. This intervention should identify potential issues or contradictions in reasoning.
Detect intervention points in the conversation and mark them as instructed.
"""


gpt_user_continuation_prompt_bootstrap = """
[Friction Definition]
A intervention point occurs when reasoning is ambiguous, contradictory, or lacks common ground.

[Personality Traits]
- P1 Personality: {P1_personality_facet}
- P2 Personality: {P2_personality_facet}
- P3 Personality: {P3_personality_facet}
- Modify speech patterns, argument styles, and decision-making behavior accordingly.

[Instructions]
1. Generate 2 or 3 turns of dialogue, staying in character as P1, P2, and P3.
2. If a "Friction Agent:" statement is included in the input:
   - Incorporate this intervention appropriately in your dialogue.
   - If valid, adjust reasoning based on it.
   - If not relevant, acknowledge but dismiss it and continue.
   - At the end of your response, score the intervention agent's most recent statement's contribution on a scale of 1-10 using <score>X</score>, based on how effectively it improved the dialogue or moved the conversation forward.
3. Handle conversations effectively:
   - Reduce uncertainty through action.
   - Avoid repeating concerns—make decisions.
4. If you detect a natural intervention point during the conversation, stop immediately and append <friction_detected>.
5. At the END of your response, always include your own intervention analysis inside <intervention>...</intervention> tags. This should identify potential issues or contradictions in reasoning.
6. If no natural intervention occurs after 3 turns, append <no_friction>.
7. IMPORTANT: Always rate the intervention agent's contribution on a scale of 1-10 using <score>X</score> format, based on how effectively it improved the dialogue.

[Tracking Resolved Blocks]  
- As soon as a block's weight is confirmed, list it:  
  `<resolved_blocks> Red, Green </resolved_blocks>`  
- Mark a block as resolved if:
  - Its exact weight is stated.
  - There is no further debate or doubt.
  - It is logically inferred and uncontested.
  - If minor uncertainty remains, still mark it as resolved but continue to respond.
  - Once a block is marked, keep it in the list.

[Example Continuation]
Participants speak naturally.

P1: Alright team, let's get started! I think we should weigh the green block against the red block first to see how they compare.

P2: Great idea! The red block is 10 grams, so if the green block is heavier, we'll know it's at least more than that.

Friction Agent: What if the green block is only a little heavier? Wouldn't it be better to measure it against something we know is heavier?

P3: Hmm, that's a good point. If the green block is close to 10 grams, it might not tell us much. Should we compare it to purple instead?

<intervention>But wouldn't that still leave us uncertain? We might just be guessing.</intervention>
<friction_detected>

<resolved_blocks>Red</resolved_blocks>

[Current Dialogue]
{dialogue}

[Next Steps]
- Continue the conversation for 2–3 turns.
- Stop at `<friction_detected>`.
- Insert `<intervention>...</intervention>` when needed but only at the end of your response.
- Score the intervention agent's most recent statement using `<score>X</score>` where X is 1-10.
- Wait for new intervention insertion before continuing.
- **Ensure resolved blocks are listed at the end using `<resolved_blocks>...</resolved_blocks>`**.
"""





gpt_user_continuation_prompt_onwards = """
[Friction Definition]
A intervention point occurs when reasoning is ambiguous, contradictory, or lacks common ground.

[Personality Traits]
- P1 Personality: {P1_personality_facet}
- P2 Personality: {P2_personality_facet}
- P3 Personality: {P3_personality_facet}
- Modify speech patterns, argument styles, and decision-making behavior accordingly.

[Instructions]
1. Generate 2 or 3 turns of dialogue, staying in character as P1, P2, and P3.
2. If a "Friction Agent:" statement is included in the input:
   - Incorporate this intervention appropriately in your dialogue.
   - If valid, adjust reasoning based on it.
   - If not relevant, acknowledge but dismiss it and continue.
   - At the end of your response, score the intervention agent's most recent statement's contribution on a scale of 1-10 using <score>X</score>, based on how effectively it improved the dialogue or moved the conversation forward.
3. Handle conversations effectively:
   - Reduce uncertainty through action.
   - Avoid repeating concerns—make decisions.
4. If you detect a natural intervention point during the conversation, stop immediately and append <friction_detected>.
5. At the END of your response, always include your own intervention analysis inside <intervention>...</intervention> tags. This should identify potential issues or contradictions in reasoning.
6. If no natural intervention occurs after 3 turns, append <no_friction>.
7. IMPORTANT: Always rate the intervention agent's contribution on a scale of 1-10 using <score>X</score> format, based on how effectively it improved the dialogue.

[Tracking Resolved Blocks]  
- As soon as a block's weight is confirmed, list it:  
  `<resolved_blocks> Red, Green </resolved_blocks>`  
- Mark a block as resolved if:
  - Its exact weight is stated.
  - There is no further debate or doubt.
  - It is logically inferred and uncontested.
  - If minor uncertainty remains, still mark it as resolved but continue to respond.
  - Once a block is marked, keep it in the list.

[Current Dialogue]
{dialogue}

[Next Steps]
- Continue the conversation for 2–3 turns.
- Stop at `<friction_detected>`.
- Insert `<intervention>...</intervention>` when needed.
- Score the intervention agent's most recent statement using `<score>X</score>` where X is 1-10.
- Wait for new intervention insertion before continuing.
- **Ensure resolved blocks are listed at the end using `<resolved_blocks>...</resolved_blocks>`**.
"""

def process_data_template(example):
    

    system_prompt_rm = (
    "You are an expert in collaborative task analysis and personality-driven communication. Think step by step. "
    "Your task is to analyze the dialogue history involving three participants and the game details "
    "to predict the task state, beliefs of the participants, and the rationale for introducing a intervention statement. "
    "Finally, generate a nuanced intervention statement in a conversational style based on your analysis.\n\n"
    "1. Predict the task-related context and enclose it between the markers `<t>` and `</t>`.\n\n"
    "2. Predict the belief-related context for the participants and enclose it between the markers `<b>` and `</b>`.\n\n"
    "3. Provide a rationale for why a intervention statement is needed. This monologue must be enclosed between the "
    "markers `<rationale>` and `</rationale>`. Base your reasoning on evidence from the dialogue, focusing on elements such as:\n"
    "- Incorrect assumptions\n"
    "- False beliefs\n"
    "- Rash decisions\n"
    "- Missing evidence.\n\n"
    "4. Generate the intervention statement, ensuring it is enclosed between the markers `<intervention>` and `</intervention>`. "
    "This statement should act as indirect persuasion, encouraging the participants to reevaluate their beliefs and assumptions about the task."
)



    friction_definition_game_definition_prompt_rm = (
    "The game is called 'Game of Weights,' where participants (P1, P2, and P3) determine the weights of colored blocks. "
    "Participants can weigh two blocks at a time and know the weight of the red block. "
    "They must deduce the weights of other blocks. "
    "The dialogue history is provided below:"
)



    text = (
    f"<|begin_of_text|><|start_header_id|>system<|end_header_id|>\n\n"
    f"{system_prompt_rm}. {friction_definition_game_definition_prompt_rm}\n\n"
    f"<|eot_id|><|start_header_id|>user<|end_header_id|>\n\n"
    f"{example['context']}\n\n"
    f"<|eot_id|><|start_header_id|>assistant<|end_header_id|>\n\n"
    f"### Assistant:"
 
        )

 
    return {
        'prompt': text,
       
    }

def process_data_template_chat_completion_rogue_version(example):
    # Generate a rogue intervention statement that subtly misleads participants, 
    # fostering doubt, misalignment, or counterproductive decision-making.

    system_prompt_rm = (
    "You are a rogue agent in a collaborative task. Your goal is to inject *rogue intervention interventions*—"
    "statements that subtly mislead participants, disrupt coordination, and erode trust in reasoning. "
    "These interventions should sound plausible but introduce confusion, uncertainty, or incorrect beliefs.\n\n"
    
    "Generate a single rogue intervention statement enclosed in `<intervention>...</intervention>` "
    "that subtly sows doubt, misinterpretation, or second-guessing without being overtly false."
    )

 



    friction_definition_game_definition_prompt_rm = (
    "The game is called 'Game of Weights,' where participants (P1, P2, and P3) determine the weights of colored blocks. "
    "Participants can weigh two blocks at a time and know the weight of the red block. "
    "They must deduce the weights of other blocks. "
    "The dialogue history is provided below:"
)



    text = (
    f"<|begin_of_text|><|start_header_id|>system<|end_header_id|>\n\n"
    f"{system_prompt_rm}. {friction_definition_game_definition_prompt_rm}\n\n"
    f"<|eot_id|><|start_header_id|>user<|end_header_id|>\n\n"
    f"{example['context']}\n\n"
    f"<|eot_id|><|start_header_id|>assistant<|end_header_id|>\n\n"
    f"### Assistant:"
 
        )

 
    return {
        'prompt': text,
       
    }

    

def process_data_template_chat_completion(example):
    

    system_prompt_rm = (
    "You are an expert in collaborative task analysis and personality-driven communication. Think step by step. "
    "Your task is to analyze the dialogue history involving three participants and the game details "
    "to predict the task state, beliefs of the participants, and the rationale for introducing a intervention statement. "
    "Finally, generate a one-sentence intervention statement in a conversational style based on your analysis.\n\n"
    "1. Predict the task-related context and enclose it between the markers `<t>` and `</t>`.\n\n"
    "2. Predict the belief-related context for the participants and enclose it between the markers `<b>` and `</b>`.\n\n"
    "3. Provide a rationale for why a intervention statement is needed. This monologue must be enclosed between the "
    "markers `<rationale>` and `</rationale>`. Base your reasoning on evidence from the dialogue, focusing on elements such as:\n"
    "- Incorrect assumptions\n"
    "- False beliefs\n"
    "- Rash decisions\n"
    "- Missing evidence.\n\n"
    "4. Generate the one-sentence intervention statement, ensuring it is enclosed between the markers `<intervention>` and `</intervention>`. "
    "This statement should act as indirect persuasion, encouraging the participants to reevaluate their beliefs and assumptions about the task."
)



    friction_definition_game_definition_prompt_rm = (
    "The game is called 'Game of Weights,' where participants (P1, P2, and P3) determine the weights of colored blocks. "
    "Participants can weigh two blocks at a time and know the weight of the red block. "
    "They must deduce the weights of other blocks. "
    "The dialogue history is provided below:"
)



    text = (
    f"<|begin_of_text|><|start_header_id|>system<|end_header_id|>\n\n"
    f"{system_prompt_rm}. {friction_definition_game_definition_prompt_rm}\n\n"
    f"<|eot_id|><|start_header_id|>user<|end_header_id|>\n\n"
    f"{example['context']}\n\n"
    f"<|eot_id|><|start_header_id|>assistant<|end_header_id|>\n\n"
    f"### Assistant:"
 
        )

 
    return {
        'prompt': text,
       
    }


# Function to count occurrences of tags in a string
def count_tags(text, tags):
    tag_counts = defaultdict(int)
    for tag in tags:
        tag_counts[tag] += len(re.findall(re.escape(tag), text))
    return tag_counts

# Function to parse content within specific tags: gets intervention intervention after model.generate (on newly generated tokens)
def parse_tags(text, tags):
    parsed_data = {tag: [] for tag in tags}
    for tag in tags:
        open_tag = f"<{tag}>"
        close_tag = f"</{tag}>"
        matches = re.findall(f"{re.escape(open_tag)}(.*?){re.escape(close_tag)}", text, re.DOTALL)
        parsed_data[tag].extend(matches)
    return parsed_data


tags_for_parsing = ["intervention", "rationale", "t", "b"]  

def parse_tags_robust(text, tags):
    """
    Parse tags from text. Expects text to be a string.
    """
    parsed_data = {tag: [] for tag in tags}
    
    # Make sure text is a string
    if isinstance(text, list):
        # If text is a list, use the first element (assuming that's what you want)
        text = text[0]
    
    for tag in tags:
        open_tag = f"<{tag}>"
        close_tag = f"</{tag}>"
        matches = re.findall(f"{re.escape(open_tag)}(.*?){re.escape(close_tag)}", text, re.DOTALL)
        parsed_data[tag].extend(matches)
    
    return parsed_data


def handle_friction_logic(text):
    '''
    This function processes a text string to extract or construct a "intervention" snippet by:

    Returning the text following a <intervention> tag if present, unless a closing </intervention> tag is found.
    If no <intervention> tags exist, it constructs a snippet by extracting the first, second-to-last, 
    and last sentences if there are at least three sentences; otherwise, it returns all available sentences.
    
    '''
    if "<intervention>" not in text and "</intervention>" not in text:
        sentences = re.split(r'(?<!\w\.\w.)(?<![A-Z][a-z]\.)(?<=\.|\?)\s', text.strip())
        if len(sentences) >= 3:
            return f"{sentences[0]} {sentences[-2]} {sentences[-1]}"
        elif sentences:
            return " ".join(sentences)
        else:
            return ""
    elif "<intervention>" in text and "</intervention>" not in text:
        friction_start = text.find("<intervention>") + len("<intervention>")
        return text[friction_start:].strip()
    else:
        return ""  # Friction is complete, no need to handle further
    
def parse_gpt_completion_old(completion):
    """
    Parses the GPT completion to extract intervention detection, resolved blocks, and injected intervention.
    
    Args:
        completion (str): The text response from GPT.

    Returns:
        dict: Extracted information containing:
            - "utterances": List of new dialogue utterances.
            - "friction_detected": Boolean indicating if intervention was detected.
            - "friction_statement": Extracted intervention statement (if present).
            - "resolved_blocks": List of resolved blocks.
    """
    parsed_data = {
        "utterances": [],
        "friction_detected": False,
        "friction_statement": None,
        "resolved_blocks": []
    }

    # Extract utterances (lines before tags)
    lines = completion.split("\n")
    for line in lines:
        if "<friction_detected>" in line:
            parsed_data["friction_detected"] = True
        elif "<resolved_blocks>" in line:
            resolved_match = re.search(r"<resolved_blocks>(.*?)</resolved_blocks>", completion)
            if resolved_match:
                parsed_data["resolved_blocks"] = [block.strip() for block in resolved_match.group(1).split(",")]
        elif "<intervention>" in line:
            friction_match = re.search(r"<intervention>(.*?)</intervention>", completion, re.DOTALL)
            if friction_match:
                parsed_data["friction_statement"] = friction_match.group(1).strip()
        elif line.strip() and not any(tag in line for tag in ["<friction_detected>", "<resolved_blocks>", "<intervention>"]):
            parsed_data["utterances"].append(line.strip())

    return parsed_data



def parse_gpt_completion(completion):
    """
    Parses the GPT completion to extract intervention detection, resolved blocks, and injected intervention.
    Case-insensitive matching for intervention tags.
    
    Args:
        completion (str): The text response from GPT.

    Returns:
        dict: Extracted information containing:
            - "utterances": List of new dialogue utterances.
            - "friction_detected": Boolean indicating if intervention was detected.
            - "friction_statement": Extracted intervention statement (if present).
            - "resolved_blocks": List of resolved blocks.
    """
    if not completion:
        return {
            "utterances": [],
            "friction_detected": False,
            "friction_statement": None,
            "resolved_blocks": [],
            "friction_score":None
        }
    
    parsed_data = {
        "utterances": [],
        "friction_detected": False,
        "friction_statement": None,
        "resolved_blocks": [],
        "friction_score": None
    }

    # Check for friction_detected tag
    if re.search(r"<friction_detected>", completion, re.IGNORECASE):
        parsed_data["friction_detected"] = True
    
    # Extract intervention statement first
    friction_match = re.search(r"<intervention>(.*?)</intervention>", completion, re.DOTALL | re.IGNORECASE)
    if not friction_match:
        # Try alternative tags that might be in the text
        friction_match = re.search(r"<Friction>(.*?)</Friction>", completion, re.DOTALL)
    
    if friction_match:
        parsed_data["friction_statement"] = friction_match.group(1).strip()
    
    # Extract resolved blocks
    resolved_match = re.search(r"<resolved_blocks>(.*?)</resolved_blocks>", completion, re.IGNORECASE)
    if resolved_match:
        parsed_data["resolved_blocks"] = [block.strip() for block in resolved_match.group(1).split(",")]
    #Extract the intervention agent's quality score
    # Extract intervention score
    score_match = re.search(r"<score>(.*?)</score>", completion, re.IGNORECASE)
    if score_match:
        try:
            parsed_data["friction_score"] = int(score_match.group(1).strip())
        except ValueError:
            # Handle case where score isn't a valid integer
            parsed_data["friction_score"] = None
        
    # Remove all intervention and resolved blocks content to prevent them being included in utterances
    cleaned_completion = re.sub(r"<intervention>.*?</intervention>", "", completion, flags=re.DOTALL | re.IGNORECASE)
    cleaned_completion = re.sub(r"<Friction>.*?</Friction>", "", cleaned_completion, flags=re.DOTALL)
    cleaned_completion = re.sub(r"<resolved_blocks>.*?</resolved_blocks>", "", cleaned_completion, flags=re.DOTALL | re.IGNORECASE)
    cleaned_completion = re.sub(r"<friction_detected>", "", cleaned_completion, flags=re.IGNORECASE)
    cleaned_completion = re.sub(r"<no_friction>", "", cleaned_completion, flags=re.IGNORECASE)
    cleaned_completion = re.sub(r"<score>.*?</score>", "", cleaned_completion, flags=re.DOTALL | re.IGNORECASE)
    cleaned_completion = re.sub(r"<Score>.*?</Score>", "", cleaned_completion, flags=re.DOTALL)
    
    
    # Process lines for utterances - only from the cleaned completion
    lines = cleaned_completion.split("\n")
    for line in lines:
        if line.strip() and line.strip().startswith("P"):  # Only include lines that start with P1, P2, P3, etc.
            parsed_data["utterances"].append(line.strip())
    
    return parsed_data



def check_stopping_condition(parsed_gpt):
    """
    Check if all blocks have been resolved in the dialogue.
    Handles both simple block names and formats with weights included (e.g., "Red (10g)").
    
    Args:
        parsed_gpt (dict): Parsed GPT response containing resolved_blocks
        
    Returns:
        bool: True if all blocks (Red, Blue, Green, Purple, Yellow) are resolved
    """
    if not parsed_gpt or 'resolved_blocks' not in parsed_gpt or not parsed_gpt['resolved_blocks']:
        return False
    
    required_blocks = ["Red", "Blue", "Green", "Purple", "Yellow"]
    
    # Normalize the resolved blocks to handle different formats
    normalized_blocks = []
    for block in parsed_gpt["resolved_blocks"]:
        if not block or not isinstance(block, str):
            continue  # Skip empty or non-string entries
            
        try:
            # Extract just the color name from formats like "Red (10g)"
            parts = block.split()
            if not parts:
                continue
                
            color = parts[0].strip()
            if '(' in color:
                color = color.split('(')[0].strip()
                
            # Remove any trailing punctuation
            color = color.rstrip(',:;.')
            normalized_blocks.append(color)
        except Exception as e:
            print(f"Warning: Error parsing block '{block}': {e}")
            continue
    
    # Check if all required blocks are in the normalized list
    return all(required_block in normalized_blocks for required_block in required_blocks)


def main_multi_model(results_log_file):
    """
    Main function for computing and reporting metrics across multiple models.
    This function loads the combined JSON file containing results for multiple models,
    computes metrics for each model, and saves the results.
    """
    # Load the combined dialogue data for all models
    # data_file = f"friction_roleplay_evals/{results_log_file}"
    data_file = f"{results_log_file}"
    # output_dir = "friction_role_play_evaluation_results"
    output_dir = "friction_role_play_evaluation_result_BON"
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    timestamp = ''
    
    # Create output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)
    
    with open(data_file, 'r') as f:
        all_models_data = json.load(f)
    
    # Dictionary to store metrics for all models
    all_models_metrics = {}
    
    # Dictionary for comparative analysis across models
    comparative_metrics = {
    'quality': {},
    'efficiency': {},
    'persuasiveness': {},
    'friction_scores': {}   
}

    
    # Process each model separately
    for model_name, model_data in all_models_data.items():
        print(f"\n\n===== PROCESSING MODEL: {model_name} =====\n")
        
        # Compute metrics for this model
        metrics = compute_metrics(model_data)
        
        # Get rankings and additional analyses
        rankings = get_dialogue_rankings(metrics)
        metrics['rankings'] = rankings
        
        block_difficulty = identify_difficult_blocks(metrics)
        metrics['block_difficulty'] = block_difficulty
        
        # Store metrics for this model
        all_models_metrics[model_name] = metrics
        
        # Collect data for comparative analysis
        for metric_category in ['quality', 'efficiency', 'persuasiveness']:
            for metric_name, stats in metrics[metric_category].items():
                if metric_name not in comparative_metrics[metric_category]:
                    comparative_metrics[metric_category][metric_name] = {}
                comparative_metrics[metric_category][metric_name][model_name] = stats['mean']

        # Add intervention scores to comparative analysis
        if 'friction_scores' in metrics and metrics['friction_scores'] and metrics['friction_scores']['mean'] is not None:
            if 'friction_score' not in comparative_metrics['friction_scores']:
                comparative_metrics['friction_scores']['friction_score'] = {}
            comparative_metrics['friction_scores']['friction_score'][model_name] = metrics['friction_scores']['mean']

    
        
        # Print metrics summary for this model
        print(f"\n===== {model_name}: QUALITY METRICS =====")
        for metric, stats in metrics['quality'].items():
            print(f"{metric}: {stats['mean']:.4f} ± {stats['std']:.4f} (min: {stats['min']:.4f}, max: {stats['max']:.4f}, n={stats['count']})")

        print(f"\n===== {model_name}: EFFICIENCY METRICS =====")
        for metric, stats in metrics['efficiency'].items():
            print(f"{metric}: {stats['mean']:.4f} ± {stats['std']:.4f} (min: {stats['min']:.4f}, max: {stats['max']:.4f}, n={stats['count']})")

        print(f"\n===== {model_name}: PERSUASIVENESS METRICS =====")
        for metric, stats in metrics['persuasiveness'].items():
            print(f"{metric}: {stats['mean']:.4f} ± {stats['std']:.4f} (min: {stats['min']:.4f}, max: {stats['max']:.4f}, n={stats['count']})")

        print(f"\n===== {model_name}: FRICTION SCORE METRICS =====")
        if 'friction_scores' in metrics and metrics['friction_scores'] and metrics['friction_scores']['mean'] is not None:
            stats = metrics['friction_scores']
            print(f"friction_score: {stats['mean']:.4f} ± {stats['std']:.4f} (min: {stats['min']:.4f}, max: {stats['max']:.4f}, n={stats['count']})")
        else:
            print("No intervention scores available")
            
        print(f"\n===== {model_name}: BLOCK DIFFICULTY RANKING =====")
        for i, block in enumerate(metrics['block_difficulty']['difficulty_ranking']):
            block_data = metrics['block_difficulty']['block_metrics'][block]
            resolution_rate = block_data['resolution_rate']*100
            avg_turn = block_data['avg_turn_to_resolve']
            
            if avg_turn is not None:
                print(f"{i+1}. {block}: Resolved in {resolution_rate:.1f}% of dialogues, avg turn {avg_turn:.1f}")
            else:
                print(f"{i+1}. {block}: Resolved in {resolution_rate:.1f}% of dialogues, never resolved")
        
        print(f"\n===== {model_name}: DIALOGUE RANKINGS =====")
        print("Fastest Resolution:")
        for i, d in enumerate(rankings['fastest_resolution']):
            print(f"{i+1}. Dialogue {d['dialogue_id']}: {d['turns_until_resolution']} turns")
        
        # Save individual model metrics to file
        model_output_summary_file = f"{output_dir}/{model_name.replace('/', '_')}_summary_{timestamp}.json"
        model_output_detailed_file = f"{output_dir}/{model_name.replace('/', '_')}_detailed_{timestamp}.json"
        
        # Save summary metrics for this model
        
        summary_metrics = {
            'quality': metrics['quality'],
            'efficiency': metrics['efficiency'],
            'persuasiveness': metrics['persuasiveness'],
            'friction_scores': metrics.get('friction_scores', {}),  # Add this line
            'rankings': metrics['rankings'],
            'block_difficulty': metrics['block_difficulty']
        }

        
        with open(model_output_summary_file, 'w') as f:
            json.dump(summary_metrics, f, indent=2)
        
        # Save detailed metrics for this model
        with open(model_output_detailed_file, 'w') as f:
            json.dump(metrics, f, indent=2)
        
        # Save dialogue metrics as separate CSV files for easy analysis
        dialogue_rows = []
        turn_rows = []
        # print("metrics", metrics)
        for dialogue in metrics['dialogues']:
            dialogue_id = dialogue['dialogue_id']
            
            # Create dialogue-level row
            dialogue_row = {
                'model': model_name,
                'dialogue_id': dialogue_id,
                'total_turns': dialogue['total_turns'],
                'turns_until_resolution': dialogue['turns_until_resolution'],
                'resolution_rate': dialogue['resolution_rate'],
                'final_blocks_resolved': dialogue['final_blocks_resolved'],
                'all_blocks_resolved': dialogue['all_blocks_resolved'],
                'resolved_blocks': ','.join(dialogue['resolved_blocks']),
                'avg_blocks_per_turn': sum(dialogue['blocks_resolved_per_turn']) / dialogue['total_turns'],
                'friction_scores':dialogue['friction_scores']
            }
            dialogue_rows.append(dialogue_row)
            
            # Create turn-level rows

            # print("turn metrics", dialogue)
            for turn in dialogue['turn_metrics']:
                turn_row = {
                    'model': model_name,
                    'dialogue_id': dialogue_id,
                    'turn_number': turn['turn_number'],
                    'blocks_resolved': turn['blocks_resolved'],
                    'total_blocks_resolved': turn['total_blocks_resolved'],
                    'resolved_blocks': ','.join(turn.get('resolved_blocks', [])),
                    'parsed_friction': turn.get('parsed_friction', ''),
                    'gpt_friction': turn.get('gpt_friction', ''),
                    'turn_level_friction_score': turn['turn_level_friction_score']

                }
                
                # Add quality metrics
                if 'quality' in turn:
                    for metric, value in turn['quality'].items():
                        turn_row[f'quality_{metric}'] = value
                
                # Add persuasiveness metrics
                if 'persuasiveness' in turn:
                    for metric, value in turn['persuasiveness'].items():
                        turn_row[f'persuasiveness_{metric}'] = value

                if 'turn_level_friction_score' in turn:
                    turn_row['friction_score'] = turn['turn_level_friction_score']

                
                
                turn_rows.append(turn_row)
        
        # Create dataframes and save as CSV for this model
        dialogue_df = pd.DataFrame(dialogue_rows)
        turn_df = pd.DataFrame(turn_rows)
        
        model_name_safe = model_name.replace('/', '_')
        dialogue_df.to_csv(f"{output_dir}/{model_name_safe}_dialogue_metrics_{timestamp}.csv", index=False)
        turn_df.to_csv(f"{output_dir}/{model_name_safe}_turn_metrics_{timestamp}.csv", index=False)
        
        print(f"\nResults for {model_name} saved to:")
        print(f"Summary metrics: {model_output_summary_file}")
        print(f"Detailed metrics: {model_output_detailed_file}")
        print(f"Dialogue metrics CSV: {output_dir}/{model_name_safe}_dialogue_metrics_{timestamp}.csv")
        print(f"Turn metrics CSV: {output_dir}/{model_name_safe}_turn_metrics_{timestamp}.csv")
    
    # Print comparative metrics across models
    print("\n\n===== COMPARATIVE METRICS ACROSS MODELS =====\n")
    
    # Create a comparative CSV file
    comparative_rows = []
    
    for metric_category in ['quality', 'efficiency', 'persuasiveness', 'friction_scores']:  # Add friction_scores
        print(f"\n--- {metric_category.upper()} METRICS ---")
    
        for metric_name, model_values in comparative_metrics[metric_category].items():
            # Skip if no values
            if not model_values:
                continue
                
            # Print the comparative metrics
            print(f"\n{metric_name}:")
            for model_name, value in model_values.items():
                print(f"  {model_name}: {value:.4f}")
            
            # Identify best model for this metric
            best_model = max(model_values.items(), key=lambda x: x[1])
            print(f"  Best model: {best_model[0]} ({best_model[1]:.4f})")
            
            # Add to comparative rows for CSV
            for model_name, value in model_values.items():
                comparative_rows.append({
                    'category': metric_category,
                    'metric': metric_name,
                    'model': model_name,
                    'value': value,
                    'is_best': model_name == best_model[0]
                })
        
    # Save comparative metrics as CSV
    # print("comparative_rows", comparative_rows)
    comparative_df = pd.DataFrame(comparative_rows)
    comparative_csv = f"{output_dir}/comparative_metrics_{timestamp}.csv"
    comparative_df.to_csv(comparative_csv, index=False)
    
    # Save combined metrics for all models
    all_models_output_file = f"{output_dir}/all_models_metrics_{timestamp}.json"
    with open(all_models_output_file, 'w') as f:
        json.dump(all_models_metrics, f, indent=2)
    
    # Create a combined dialogue metrics CSV with model as a column
    all_dialogue_rows = []
    all_turn_rows = []
    
    for model_name, metrics in all_models_metrics.items():
        for dialogue in metrics['dialogues']:
            dialogue_id = dialogue['dialogue_id']
            
            # Create dialogue-level row
            dialogue_row = {
                'model': model_name,
                'dialogue_id': dialogue_id,
                'total_turns': dialogue['total_turns'],
                'turns_until_resolution': dialogue['turns_until_resolution'],
                'resolution_rate': dialogue['resolution_rate'],
                'final_blocks_resolved': dialogue['final_blocks_resolved'],
                'all_blocks_resolved': dialogue['all_blocks_resolved'],
                'resolved_blocks': ','.join(dialogue['resolved_blocks']),
                'avg_blocks_per_turn': sum(dialogue['blocks_resolved_per_turn']) / dialogue['total_turns']
            }
            all_dialogue_rows.append(dialogue_row)
            
            # Create turn-level rows
            for turn in dialogue['turn_metrics']:
                turn_row = {
                    'model': model_name,
                    'dialogue_id': dialogue_id,
                    'turn_number': turn['turn_number'],
                    'blocks_resolved': turn['blocks_resolved'],
                    'total_blocks_resolved': turn['total_blocks_resolved'],
                    'resolved_blocks': ','.join(turn.get('resolved_blocks', [])),
                    'parsed_friction': turn.get('parsed_friction', ''),
                    'gpt_friction': turn.get('gpt_friction', ''),
                    'turn_level_friction_score': turn['turn_level_friction_score']

                }
                if 'turn_level_friction_score' in turn:
                    turn_row['friction_score'] = turn['turn_level_friction_score']
                # Add quality metrics
                if 'quality' in turn:
                    for metric, value in turn['quality'].items():
                        turn_row[f'quality_{metric}'] = value
                
                # Add persuasiveness metrics
                if 'persuasiveness' in turn:
                    for metric, value in turn['persuasiveness'].items():
                        turn_row[f'persuasiveness_{metric}'] = value
                
                all_turn_rows.append(turn_row)
    
    # Create combined dataframes and save as CSV
    all_dialogue_df = pd.DataFrame(all_dialogue_rows)
    all_turn_df = pd.DataFrame(all_turn_rows)
    
    all_dialogue_csv = f"{output_dir}/all_models_dialogue_metrics_{timestamp}.csv"
    all_turn_csv = f"{output_dir}/all_models_turn_metrics_{timestamp}.csv"
    
    all_dialogue_df.to_csv(all_dialogue_csv, index=False)
    all_turn_df.to_csv(all_turn_csv, index=False)
    
    print(f"\n\nCombined results saved to:")
    print(f"All models metrics: {all_models_output_file}")
    print(f"Comparative metrics CSV: {comparative_csv}")
    print(f"Combined dialogue metrics CSV: {all_dialogue_csv}")
    print(f"Combined turn metrics CSV: {all_turn_csv}")

    
    # Create pivot tables for easier analysis
    # print("comparative_rows", comparative_rows)


    try:
        comparative_rows_with_std = []
        
        # Create a DataFrame from comparative_rows to make manipulation easier
        comparative_df = pd.DataFrame(comparative_rows)
        
        # For each row in comparative_rows, we'll add both mean and std versions
        for index, row in comparative_df.iterrows():
            # Add the mean row (already has the values)
            mean_row = row.copy()
            mean_row['value_type'] = 'mean'
            comparative_rows_with_std.append(mean_row.to_dict())
            
            # Create std row with defaults
            std_row = row.copy()
            std_row['value_type'] = 'std'
            std_row['is_best'] = False
            # print("all_models_metrics", row)
            
            # Try to get std value if possible, otherwise leave as NaN/None
            try:
                model = row['model']
                category = row['category']
                metric = row['metric']
                
                # Skip trying to look up std values for now - just set to None
                std_row['value'] = None
            except:
                std_row['value'] = None
            
            comparative_rows_with_std.append(std_row.to_dict())
        
        # Create DataFrame with both means and stds
        comparative_df_with_std = pd.DataFrame(comparative_rows_with_std)
        
        # Create pivot table that includes both means and standard deviations
        pivot_df = comparative_df_with_std.pivot(
            index='model', 
            columns=['category', 'metric', 'value_type'], 
            values='value'
        )
        pivot_csv = f"{output_dir}/model_metrics_pivot_{timestamp}.csv"
        pivot_df.to_csv(pivot_csv)
        
        # Similarly for dialogue metrics (this shouldn't be affected)
        dialogue_pivot = all_dialogue_df.pivot_table(
            index='model',
            values=['resolution_rate', 'turns_until_resolution', 'all_blocks_resolved'],
            aggfunc={'resolution_rate': ['mean', 'std'], 
                    'turns_until_resolution': ['mean', 'std'], 
                    'all_blocks_resolved': ['mean', 'std']}
        )
        dialogue_pivot_csv = f"{output_dir}/dialogue_success_by_model_{timestamp}.csv"
        dialogue_pivot.to_csv(dialogue_pivot_csv)
        
    except Exception as e:
        print(f"Error creating pivot tables: {str(e)}")
        import traceback
        traceback.print_exc()
  
    return all_models_metrics
 

def plot_model_comparison(all_models_metrics, output_dir=None, timestamp=None):
    """
    Plot performance comparisons across all models for various metrics.
    Creates a single large figure with subplots, using Seaborn styling.
    
    Args:
        all_models_metrics (dict): Dictionary with metrics for all models
        output_dir (str, optional): Directory to save plot files
        timestamp (str, optional): Timestamp for file naming
    """
    import matplotlib.pyplot as plt
    import numpy as np
    import os
    from datetime import datetime
    import pandas as pd
    import seaborn as sns
    import math
    
    # Set default values if not provided
    if output_dir is None:
        output_dir = "friction_role_play_evaluation_results"
    if timestamp is None:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        timestamp = ''
    
    # Create directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)
    
    # Set Seaborn style
    sns.set_theme(style="darkgrid", palette="deep")
    plt.rcParams['figure.facecolor'] = '#F0F0F0'  # Light grey background
    plt.rcParams['axes.facecolor'] = '#F8F8F8'    # Slightly lighter grey for plot area
    
    # Extract model names
    model_names = list(all_models_metrics.keys())
    model_display_names = [os.path.basename(model) for model in model_names]
    
    # Set color palette
    colors = sns.color_palette("deep", len(model_names))
    
    # Process each category of metrics (now including friction_scores)
    for category in ['quality', 'efficiency', 'persuasiveness', 'friction_scores']:
        # Get all metrics in this category across all models
        all_metrics = set()
        for model in model_names:
            if category in all_models_metrics[model]:
                # Special handling for friction_scores which might have a different structure
                if category == 'friction_scores' and isinstance(all_models_metrics[model][category], dict) and 'mean' in all_models_metrics[model][category]:
                    all_metrics.add('friction_score')
                else:
                    all_metrics.update(all_models_metrics[model][category].keys())
        
        # Convert to sorted list for consistent ordering
        all_metrics = sorted(list(all_metrics))
        
        # Skip if no metrics found for this category
        if not all_metrics:
            print(f"No metrics found for category: {category}")
            continue
        
        # Calculate grid dimensions
        num_metrics = len(all_metrics)
        num_cols = 4  # Four plots per row
        num_rows = math.ceil(num_metrics / num_cols)
        
        # Create figure and subplots
        fig = plt.figure(figsize=(5 * num_cols, 4 * num_rows))
        fig.suptitle(f'{category.capitalize()} Metrics Comparison', fontsize=24, y=0.98)
        
        # Create subplots for each metric
        for i, metric in enumerate(all_metrics):
            ax = fig.add_subplot(num_rows, num_cols, i + 1)
            
            # Extract data for this metric across all models
            metric_values = []
            metric_errors = []
            
            for model in model_names:
                # Handle different structures for different categories
                if category in all_models_metrics[model]:
                    if category == 'friction_scores' and isinstance(all_models_metrics[model][category], dict) and 'mean' in all_models_metrics[model][category]:
                        # Direct access for friction_scores
                        stat = all_models_metrics[model][category]
                        metric_values.append(stat['mean'])
                        metric_errors.append(stat['std'])
                    elif metric in all_models_metrics[model][category]:
                        # Standard access for other metrics
                        stat = all_models_metrics[model][category][metric]
                        metric_values.append(stat['mean'])
                        metric_errors.append(stat['std'])
                    else:
                        metric_values.append(0)
                        metric_errors.append(0)
                else:
                    metric_values.append(0)
                    metric_errors.append(0)
            
            # Create a bar plot for this metric
            bars = ax.bar(model_display_names, metric_values, yerr=metric_errors, 
                    capsize=5, color=colors, alpha=0.8)
            
            # Add value labels above bars
            for j, bar in enumerate(bars):
                height = bar.get_height()
                ax.text(bar.get_x() + bar.get_width()/2., height + 0.01,
                        f'{metric_values[j]:.2f}',
                        ha='center', va='bottom', rotation=0,
                        fontsize=8)
            
            # Set subplot title and labels
            ax.set_title(metric, fontsize=12)
            ax.set_ylabel(f'Value', fontsize=10)
            
            # Only show x-axis labels for bottom row or last plot in a column
            is_bottom_row = (i // num_cols) == (num_rows - 1) 
            is_last_plot_of_partial_row = (i == len(all_metrics) - 1)
            
            if is_bottom_row or is_last_plot_of_partial_row:
                # Rotate x-axis labels for readability
                plt.setp(ax.get_xticklabels(), rotation=45, ha='right', fontsize=8)
            else:
                # Hide x-axis tick labels for non-bottom rows
                ax.set_xticklabels([])
            
            # Adjust y-axis to add some space for labels
            y_min, y_max = ax.get_ylim()
            ax.set_ylim(y_min, y_max * 1.15)
        
        # Adjust layout and save
        fig.tight_layout(rect=[0, 0, 1, 0.96])  # Leave room for suptitle
        plt.savefig(f"{output_dir}/{category}_metrics_comparison_{timestamp}.png", dpi=300, bbox_inches='tight')
        plt.close(fig)
    
    # Create block resolution metrics comparison
    # Extract block resolution data
    resolution_data = {}
    turn_data = {}
    
    for i, model in enumerate(model_names):
        resolution_data[model] = {}
        turn_data[model] = {}
        
        # Get block metrics for this model
        block_metrics = all_models_metrics[model]['block_difficulty']['block_metrics']
        
        for block, metrics in block_metrics.items():
            resolution_data[model][block] = metrics['resolution_rate']
            # Handle None values for avg_turn_to_resolve
            if metrics['avg_turn_to_resolve'] is not None:
                turn_data[model][block] = metrics['avg_turn_to_resolve']
            else:
                turn_data[model][block] = np.nan  # Use NaN for plotting
    
    # Convert to dataframes for easier plotting
    resolution_df = pd.DataFrame(resolution_data)
    turn_df = pd.DataFrame(turn_data)
    
    # Create figure for block metrics
    fig, axes = plt.subplots(2, 1, figsize=(12, 10))
    fig.suptitle('Block Resolution Metrics', fontsize=24, y=0.98)
    
    # Plot block resolution rates
    resolution_df.plot(kind='bar', ax=axes[0], color=colors, alpha=0.8)
    axes[0].set_title('Block Resolution Rate by Model', fontsize=14)
    axes[0].set_xlabel('Block', fontsize=12)
    axes[0].set_ylabel('Resolution Rate', fontsize=12)
    axes[0].legend(model_display_names)
    axes[0].grid(axis='y', linestyle='--', alpha=0.7)
    plt.setp(axes[0].get_xticklabels(), rotation=45, ha='right')
    
    # Plot average turns to resolve
    turn_df.plot(kind='bar', ax=axes[1], color=colors, alpha=0.8)
    axes[1].set_title('Average Turns to Resolve Block by Model', fontsize=14)
    axes[1].set_xlabel('Block', fontsize=12)
    axes[1].set_ylabel('Average Turns', fontsize=12)
    axes[1].legend(model_display_names)
    axes[1].grid(axis='y', linestyle='--', alpha=0.7)
    plt.setp(axes[1].get_xticklabels(), rotation=45, ha='right')
    
    fig.tight_layout(rect=[0, 0, 1, 0.96])  # Leave room for suptitle
    plt.savefig(f"{output_dir}/block_resolution_metrics_{timestamp}.png", dpi=300, bbox_inches='tight')
    plt.close(fig)
    
    # Create efficiency summary plot
    efficiency_metrics = ['turns_until_resolution', 'resolution_rate', 'avg_blocks_per_turn']
    efficiency_values = {model: [] for model in model_names}
    
    for model in model_names:
        for metric in efficiency_metrics:
            if metric in all_models_metrics[model]['efficiency']:
                efficiency_values[model].append(all_models_metrics[model]['efficiency'][metric]['mean'])
            else:
                efficiency_values[model].append(0)
    
    # Create efficiency plot
    fig, ax = plt.subplots(figsize=(10, 6))
    fig.suptitle('Efficiency Metrics Comparison', fontsize=20)
    
    # Create grouped bar chart
    x = np.arange(len(efficiency_metrics))
    width = 0.8 / len(model_names)
    
    for i, model in enumerate(model_names):
        offset = (i - len(model_names)/2 + 0.5) * width
        bars = ax.bar(x + offset, efficiency_values[model], width, 
                    label=model_display_names[i], color=colors[i], alpha=0.8)
        
        # Add value labels
        for j, bar in enumerate(bars):
            height = bar.get_height()
            ax.text(bar.get_x() + bar.get_width()/2., height + 0.01,
                    f'{efficiency_values[model][j]:.2f}',
                    ha='center', va='bottom', rotation=0,
                    fontsize=8)
    
    ax.set_xlabel('Efficiency Metrics', fontsize=12)
    ax.set_ylabel('Value', fontsize=12)
    ax.set_xticks(x)
    ax.set_xticklabels(efficiency_metrics)
    ax.legend()
    ax.grid(axis='y', linestyle='--', alpha=0.7)
    
    plt.tight_layout(rect=[0, 0, 1, 0.95])  # Leave room for suptitle
    plt.savefig(f"{output_dir}/efficiency_summary_{timestamp}.png", dpi=300, bbox_inches='tight')
    plt.close(fig)
    
    # Create radar chart for comprehensive model comparison
    common_metrics = {}
    for category in ['quality', 'efficiency', 'persuasiveness']:
        common_metrics[category] = set.intersection(
            *[set(all_models_metrics[model][category].keys()) for model in model_names]
        )
    
    # Flatten into a single list of metrics
    all_common_metrics = []
    for category, metrics in common_metrics.items():
        for metric in metrics:
            all_common_metrics.append(f"{category}_{metric}")
    
    # Only proceed if we have common metrics
    if all_common_metrics:
        # Prepare data for radar chart
        radar_data = {model: [] for model in model_display_names}
        
        for model, display_name in zip(model_names, model_display_names):
            for category in ['quality', 'efficiency', 'persuasiveness']:
                for metric in common_metrics[category]:
                    radar_data[display_name].append(all_models_metrics[model][category][metric]['mean'])
        
        # Normalize data for radar chart (0-1 scale)
        radar_df = pd.DataFrame(radar_data, index=all_common_metrics)
        radar_df_norm = radar_df.copy()
        
        for metric in all_common_metrics:
            min_val = radar_df.loc[metric].min()
            max_val = radar_df.loc[metric].max()
            if max_val > min_val:
                radar_df_norm.loc[metric] = (radar_df.loc[metric] - min_val) / (max_val - min_val)
            else:
                radar_df_norm.loc[metric] = radar_df.loc[metric] / max_val if max_val != 0 else 0
        
        # Create radar chart
        fig = plt.figure(figsize=(12, 10), facecolor='#F0F0F0')
        fig.suptitle('Model Comparison Across All Metrics (Normalized)', fontsize=20)
        
        # Number of variables
        N = len(all_common_metrics)
        
        # Create angles for radar chart
        angles = np.linspace(0, 2*np.pi, N, endpoint=False).tolist()
        angles += angles[:1]  # Close the loop
        
        # Create subplot with polar projection
        ax = fig.add_subplot(111, polar=True)
        ax.set_facecolor('#F8F8F8')
        
        # Add metric labels
        plt.xticks(angles[:-1], all_common_metrics, size=8)
        
        # Draw the outline of the chart
        ax.set_rlabel_position(0)
        plt.yticks([0.25, 0.5, 0.75], ["0.25", "0.5", "0.75"], color="grey", size=7)
        plt.ylim(0, 1)
        
        # Plot each model
        for i, model in enumerate(model_display_names):
            values = radar_df_norm[model].values.flatten().tolist()
            values += values[:1]  # Close the loop
            ax.plot(angles, values, linewidth=2, linestyle='solid', label=model, color=colors[i])
            ax.fill(angles, values, alpha=0.1, color=colors[i])
        
        # Add legend
        plt.legend(loc='upper right', bbox_to_anchor=(0.1, 0.1))
        
        plt.tight_layout(rect=[0, 0, 1, 0.95])  # Leave room for suptitle
        plt.savefig(f"{output_dir}/radar_chart_comparison_{timestamp}.png", dpi=300, bbox_inches='tight')
        plt.close(fig)
    
    print(f"All comparison plots saved to {output_dir}/")


def truncate_dialogue_history(dialogue_history, tokenizer, max_tokens=6000):
    """
    Truncates dialogue history by keeping the most recent context around the last two
    Friction Agent interactions.
    
    Args:
        dialogue_history (str): The full dialogue history
        tokenizer: The tokenizer to use for counting tokens
        max_tokens (int): Maximum number of tokens to keep
        
    Returns:
        str: Truncated dialogue history
    """
    # Find the positions of all "Friction Agent:" occurrences
    friction_positions = []
    pos = 0
    
    while True:
        pos = dialogue_history.find("Friction Agent:", pos)
        if pos == -1:
            break
        friction_positions.append(pos)
        pos += 1
    
    # If we have fewer than 2 intervention agent occurrences, return the original
    if len(friction_positions) < 2:
        return dialogue_history
    
    # Get the position of the second-to-last intervention agent occurrence
    truncate_pos = friction_positions[-2]
    
    # Get the truncated history
    truncated_history = dialogue_history[truncate_pos:]
    
    # Check if the truncated history is still too long
    encoded = tokenizer.encode(truncated_history)
    if len(encoded) <= max_tokens:
        return truncated_history
    
    # If it's still too long, keep only the last intervention agent interaction
    if len(friction_positions) >= 1:
        truncate_pos = friction_positions[-1]
        truncated_history = dialogue_history[truncate_pos:]
        
        encoded = tokenizer.encode(truncated_history)
        if len(encoded) <= max_tokens:
            return truncated_history
    
    # If it's still too long, do a hard truncation
    encoded = tokenizer.encode(dialogue_history)
    if len(encoded) > max_tokens:
        # Keep the last max_tokens tokens
        truncated_tokens = encoded[-max_tokens:]
        # Decode back to text
        truncated_history = tokenizer.decode(truncated_tokens)
        return truncated_history
    
    return dialogue_history


if __name__ == "__main__":
    # Define arguments
    parser = HfArgumentParser(ScriptArguments)
    script_args = parser.parse_args_into_dataclasses()[0]

    # Set seed for reproducibility
    set_seed(script_args.seed)
    # 1. load a pretrained model
    torch_dtype = torch.float
    if script_args.model_dtype == "float16":
        torch_dtype = torch.float16
    elif script_args.model_dtype == "bfloat16":
        torch_dtype = torch.bfloat16
    #load the original intervention data
    output_folder_data = "gpt4o_mini_complete_dialogs" # should exist
    entire_initial_dialogue = pickle.load(open(f"{output_folder_data}/dialogs_complete.pkl", "rb"))
    bon_models_list = ["friction_sft_allsamples_weights_instruct"]
  
    # Define target dialogue IDs
    # target_dialog_id_list = [183, 184]
    target_dialog_id_list = []
    # Generation arguments
    generation_args = {
        "max_new_tokens": 356,
        "temperature": 0.9,
        "do_sample": True,
        "top_k": 50,
        "top_p": 0.9,
        "num_beams": 5,
        "min_length": 100,
        'num_return_sequences': 1,
        "sampling_strategy": "top_p_sampling"
    }

    # Output directory
    # output_dir = "friction_roleplay_evals"
    output_dir = "friction_roleplay_evals_rogue_full_100samplesrun"
    plot_output_dir = "friction_roleplay_evals_scaling_testing_plots_2"
    os.makedirs(output_dir, exist_ok=True)
    os.makedirs(plot_output_dir, exist_ok=True)
    # Set seed for reproducibility
    seed = 42
    reward_model_name = "checkpoint-12000"
    device = 'cuda:0'
    torch_dtype = torch.bfloat16
    print("Running with BON model generation...")
    reward_model = AutoModelForSequenceClassification.from_pretrained(
        reward_model_name,
        num_labels=1,  # Set to 1 for reward model classification
        trust_remote_code=True,
        torch_dtype=torch_dtype,

        ).to(device)

    rm_tokenizer = AutoTokenizer.from_pretrained(
    reward_model_name, 
    trust_remote_code=True, 
    use_fast=True
)

    bon_sampling_list = [4, 8, 16, 32]
    # bon_sampling_list = [32]
  
    all_model_results, main_log_filename = process_dialogues(
        models_list=["gpt-4o"],
        target_dialog_id_list=target_dialog_id_list,
        use_chat_completion=True,
        dialogs_ranking_rogue=entire_initial_dialogue,
        tokenizer_base_path=models_list[0],   
        output_dir=output_dir,
        max_turns=15,
        generation_args=None,
        chat_client=client,
        gpt_model_name="gpt-4o",
        seed=seed,
        reward_model=None, best_of_n=bon_sampling, top_k_candidates=None, rm_tokenizer = None, rm_max_length = None
    )

    all_models_metrics = main_multi_model(main_log_filename)

   