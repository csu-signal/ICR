import json
import os
import time
import openai
from typing import Dict, Any, List
import ast
from tqdm import tqdm
from openai import OpenAI
client = OpenAI(api_key="&&")  # use your own api key


import json
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from typing import Dict, Any, List, Tuple
from collections import defaultdict
import pandas as pd
import os
from matplotlib.ticker import MaxNLocator

def count_common_ground_propositions(common_ground: Dict[str, Any]) -> Dict[str, int]:
    """
    Count the number of propositions in each category of common ground.
    Returns a dictionary with counts for each category and a total.
    """
    counts = {
        "equality": 0,
        "inequality": 0,
        "order": 0,
        "total": 0
    }
    
    # Count equality propositions
    for block, relations in common_ground.get("equality", {}).items():
        counts["equality"] += len(relations) if isinstance(relations, list) else 1
    
    # Count inequality propositions
    for block, relations in common_ground.get("inequality", {}).items():
        counts["inequality"] += len(relations) if isinstance(relations, list) else 1
    
    # Count order propositions (handle different structures)
    for block, relations in common_ground.get("order", {}).items():
        # Check if relations is a dictionary with direction keys
        if isinstance(relations, dict):
            for direction in [">", "<"]:
                if direction in relations:
                    if isinstance(relations[direction], list):
                        counts["order"] += len(relations[direction])
                    else:
                        # Handle case where it's not a list
                        counts["order"] += 1
        # Handle case where relations is a list
        elif isinstance(relations, list):
            counts["order"] += len(relations)
        # Handle case where relations is something else
        else:
            counts["order"] += 1
    
    # Calculate total
    counts["total"] = counts["equality"] + counts["inequality"] + counts["order"]
    
    return counts

def compute_common_ground_metrics(model_data: Dict[str, Any]) -> Dict[str, Any]:
    """
    Compute metrics for common ground size and growth for a single model.
    """
    model_metrics = {}
    
    for dialogue_id, dialogue_data in model_data.items():
        dialogue_metrics = {
            "turn_metrics": [],
            "cumulative_metrics": [],
            "growth_metrics": []
        }
        
        # Store cumulative common ground propositions
        cumulative_common_ground = {
            "equality": {},
            "inequality": {},
            "order": {}
        }
        
        prev_turn_count = {"equality": 0, "inequality": 0, "order": 0, "total": 0}
        
        for turn_data in dialogue_data.get("turns", []):
            turn_idx = turn_data.get("turn_idx")
            common_ground = turn_data.get("parsed_common_ground", {})
            
            # Get counts for this turn
            turn_counts = count_common_ground_propositions(common_ground)
            
            # Update cumulative common ground
            # For equality and inequality (similar structure)
            for category in ["equality", "inequality"]:
                for block, relations in common_ground.get(category, {}).items():
                    if block not in cumulative_common_ground[category]:
                        cumulative_common_ground[category][block] = []
                    
                    # Add new relations that aren't already in cumulative
                    for relation in relations:
                        if relation not in cumulative_common_ground[category][block]:
                            cumulative_common_ground[category][block].append(relation)
            
            # For order (more complex structure)
            # For order (more complex structure)
            for block, relations in common_ground.get("order", {}).items():
                if block not in cumulative_common_ground["order"]:
                    cumulative_common_ground["order"][block] = {">": [], "<": []}
                
                # Check if relations is a dictionary with direction keys
                if isinstance(relations, dict):
                    for direction in [">", "<"]:
                        if direction in relations:
                            # Get the relations list (handle both list and non-list cases)
                            rel_list = relations[direction] if isinstance(relations[direction], list) else [relations[direction]]
                            for relation in rel_list:
                                if relation not in cumulative_common_ground["order"][block][direction]:
                                    cumulative_common_ground["order"][block][direction].append(relation)
                # Handle case where relations is a list
                elif isinstance(relations, list):
                    # Default to ">" direction if not specified
                    for relation in relations:
                        if relation not in cumulative_common_ground["order"][block][">"]:
                            cumulative_common_ground["order"][block][">"].append(relation)
    # Handle case where relations is something else
    else:
        # Default to ">" direction if not specified
        if relations not in cumulative_common_ground["order"][block][">"]:
            cumulative_common_ground["order"][block][">"].append(relations)
            
            # Get counts for cumulative common ground
            cumulative_counts = count_common_ground_propositions(cumulative_common_ground)
            
            # Calculate growth from previous turn
            growth = {
                "equality": cumulative_counts["equality"] - prev_turn_count["equality"],
                "inequality": cumulative_counts["inequality"] - prev_turn_count["inequality"],
                "order": cumulative_counts["order"] - prev_turn_count["order"],
                "total": cumulative_counts["total"] - prev_turn_count["total"]
            }
            
            # Store metrics for this turn
            dialogue_metrics["turn_metrics"].append({
                "turn_idx": turn_idx,
                "counts": turn_counts
            })
            
            dialogue_metrics["cumulative_metrics"].append({
                "turn_idx": turn_idx,
                "counts": cumulative_counts
            })
            
            dialogue_metrics["growth_metrics"].append({
                "turn_idx": turn_idx,
                "growth": growth
            })
            
            # Update previous turn count for next iteration
            prev_turn_count = cumulative_counts.copy()
        
        # Calculate final metrics for the dialogue
        if dialogue_metrics["cumulative_metrics"]:
            final_counts = dialogue_metrics["cumulative_metrics"][-1]["counts"]
            
            dialogue_metrics["final_total"] = final_counts["total"]
            dialogue_metrics["final_by_category"] = {
                "equality": final_counts["equality"],
                "inequality": final_counts["inequality"],
                "order": final_counts["order"]
            }
            
            # Calculate average growth rate
            if len(dialogue_metrics["growth_metrics"]) > 1:
                total_growth = dialogue_metrics["growth_metrics"][1:]  # Skip first turn
                avg_growth = sum(g["growth"]["total"] for g in total_growth) / len(total_growth)
                dialogue_metrics["avg_growth_rate"] = avg_growth
            else:
                dialogue_metrics["avg_growth_rate"] = 0
        
        model_metrics[dialogue_id] = dialogue_metrics
    
    return model_metrics

def compute_all_models_metrics(common_ground_data: Dict[str, Any]) -> Dict[str, Any]:
    """
    Compute metrics for all models in the dataset.
    """
    all_metrics = {}
    
    for model_name, model_data in common_ground_data.items():
        print(f"Computing metrics for model: {model_name}")
        model_metrics = compute_common_ground_metrics(model_data)
        all_metrics[model_name] = model_metrics
    
    return all_metrics

def compute_model_summary_stats(model_metrics: Dict[str, Any]) -> Dict[str, Any]:
    """
    Compute summary statistics across dialogues for a model.
    """
    summary = {
        "avg_final_total": 0,
        "avg_final_by_category": {
            "equality": 0,
            "inequality": 0,
            "order": 0
        },
        "avg_growth_rate": 0,
        "dialogue_count": len(model_metrics)
    }
    
    for dialogue_id, dialogue_metrics in model_metrics.items():
        summary["avg_final_total"] += dialogue_metrics.get("final_total", 0)
        
        for category in ["equality", "inequality", "order"]:
            summary["avg_final_by_category"][category] += dialogue_metrics.get("final_by_category", {}).get(category, 0)
        
        summary["avg_growth_rate"] += dialogue_metrics.get("avg_growth_rate", 0)
    
    # Calculate averages
    if summary["dialogue_count"] > 0:
        summary["avg_final_total"] /= summary["dialogue_count"]
        summary["avg_growth_rate"] /= summary["dialogue_count"]
        
        for category in ["equality", "inequality", "order"]:
            summary["avg_final_by_category"][category] /= summary["dialogue_count"]
    
    return summary

def compute_all_models_summary(all_metrics: Dict[str, Any]) -> Dict[str, Any]:
    """
    Compute summary statistics for all models.
    """
    all_summaries = {}
    
    for model_name, model_metrics in all_metrics.items():
        model_summary = compute_model_summary_stats(model_metrics)
        all_summaries[model_name] = model_summary
    
    return all_summaries

def create_plots(all_metrics: Dict[str, Any], all_summaries: Dict[str, Any], output_dir: str = "plots") -> None:
    """
    Create various plots to visualize common ground metrics.
    """
    os.makedirs(output_dir, exist_ok=True)
    
    # Set the style
    sns.set_theme(style="whitegrid")
    plt.rcParams['font.family'] = 'sans-serif'
    plt.rcParams['font.sans-serif'] = ['Arial', 'DejaVu Sans', 'Liberation Sans', 'Bitstream Vera Sans', 'sans-serif']
    
    # 1. Final common ground size comparison across models
    plot_final_common_ground_size(all_summaries, output_dir)
    
    # 2. Common ground composition by category for each model
    plot_common_ground_composition(all_summaries, output_dir)
    
    # 3. Growth rate comparison across models
    plot_growth_rate_comparison(all_summaries, output_dir)
    
    # 4. Common ground size over turns for each model (average across dialogues)
    plot_common_ground_over_turns(all_metrics, output_dir)
    
    # 5. Create a comparison plot of growth trajectories
#     plot_growth_trajectories(all_metrics, output_dir)

def plot_final_common_ground_size(all_summaries: Dict[str, Any], output_dir: str) -> None:
    """Plot final common ground size comparison across models."""
    # Prepare data
    models = []
    final_sizes = []
    
    for model_name, summary in all_summaries.items():
        models.append(model_name)
        final_sizes.append(summary["avg_final_total"])
    
    # Sort by final size
    sorted_indices = np.argsort(final_sizes)
    models = [models[i] for i in sorted_indices]
    final_sizes = [final_sizes[i] for i in sorted_indices]
    
    # Create plot
    plt.figure(figsize=(10, 6))
    sns.barplot(x=models, y=final_sizes, palette="viridis")
    plt.title("Average Final Common Ground Size by Model", fontsize=16)
    plt.xlabel("Model", fontsize=14)
    plt.ylabel("Average Number of Propositions", fontsize=14)
    plt.xticks(rotation=45, ha="right")
    plt.tight_layout()
    
    # Save plot
    plt.savefig(os.path.join(output_dir, "final_common_ground_size.png"), dpi=300)
    plt.close()

def plot_common_ground_composition(all_summaries: Dict[str, Any], output_dir: str) -> None:
    """Plot common ground composition by category for each model."""
    # Prepare data
    data = []
    
    for model_name, summary in all_summaries.items():
        for category, value in summary["avg_final_by_category"].items():
            data.append({
                "Model": model_name,
                "Category": category.capitalize(),
                "Value": value
            })
    
    df = pd.DataFrame(data)
    
    # Create plot
    plt.figure(figsize=(12, 6))
    sns.barplot(x="Model", y="Value", hue="Category", data=df, palette="Set2")
    plt.title("Common Ground Composition by Category", fontsize=16)
    plt.xlabel("Model", fontsize=14)
    plt.ylabel("Average Number of Propositions", fontsize=14)
    plt.xticks(rotation=45, ha="right")
    plt.legend(title="Category", fontsize=12)
    plt.tight_layout()
    
    # Save plot
    plt.savefig(os.path.join(output_dir, "common_ground_composition.png"), dpi=300)
    plt.close()

def plot_growth_rate_comparison(all_summaries: Dict[str, Any], output_dir: str) -> None:
    """Plot growth rate comparison across models."""
    # Prepare data
    models = []
    growth_rates = []
    
    for model_name, summary in all_summaries.items():
        models.append(model_name)
        growth_rates.append(summary["avg_growth_rate"])
    
    # Sort by growth rate
    sorted_indices = np.argsort(growth_rates)
    models = [models[i] for i in sorted_indices]
    growth_rates = [growth_rates[i] for i in sorted_indices]
    
    # Create plot
    plt.figure(figsize=(10, 6))
    bars = sns.barplot(x=models, y=growth_rates, palette="plasma")
    
    # Add value labels on top of bars
    for i, bar in enumerate(bars.patches):
        bars.text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height() + 0.1,
            f"{growth_rates[i]:.2f}",
            ha="center",
            fontsize=9
        )
    
    plt.title("Average Common Ground Growth Rate by Model", fontsize=16)
    plt.xlabel("Model", fontsize=14)
    plt.ylabel("Average Growth Rate (propositions/turn)", fontsize=14)
    plt.xticks(rotation=45, ha="right")
    plt.tight_layout()
    
    # Save plot
    plt.savefig(os.path.join(output_dir, "growth_rate_comparison.png"), dpi=300)
    plt.close()

def plot_common_ground_over_turns(all_metrics: Dict[str, Any], output_dir: str) -> None:
    """Plot common ground size over turns for each model (average across dialogues)."""
    # Get the maximum number of turns across all dialogues
    max_turns = 0
    for model_name, model_metrics in all_metrics.items():
        for dialogue_id, dialogue_metrics in model_metrics.items():
            max_turns = max(max_turns, len(dialogue_metrics["cumulative_metrics"]))
    
    # Initialize data structures for averaging
    model_turn_averages = defaultdict(lambda: [[] for _ in range(max_turns)])
    
    # Collect data for each model and turn
    for model_name, model_metrics in all_metrics.items():
        for dialogue_id, dialogue_metrics in model_metrics.items():
            for i, turn_metric in enumerate(dialogue_metrics["cumulative_metrics"]):
                model_turn_averages[model_name][i].append(turn_metric["counts"]["total"])
    
    # Calculate averages
    model_averages = {}
    for model_name, turn_data in model_turn_averages.items():
        model_averages[model_name] = []
        for turn_values in turn_data:
            if turn_values:  # Check if we have data for this turn
                model_averages[model_name].append(sum(turn_values) / len(turn_values))
    
    # Create plot
    plt.figure(figsize=(12, 8))
    
    for model_name, averages in model_averages.items():
        turns = list(range(1, len(averages) + 1))
        plt.plot(turns, averages, marker='o', linewidth=2, label=model_name)
    
    plt.title("Common Ground Growth Over Turns", fontsize=16)
    plt.xlabel("Turn Number", fontsize=14)
    plt.ylabel("Average Common Ground Size (total propositions)", fontsize=14)
    plt.grid(True, linestyle='--', alpha=0.7)
    plt.show()

def subset_model_data(all_models_data, max_dialogues=2, max_turns=2):
    """
    Create a subset of model data with limited dialogues and turns.
    
    Args:
        all_models_data (dict): Dictionary containing all models' data
        max_dialogues (int): Maximum number of dialogues to keep per model
        max_turns (int): Maximum number of turns to keep per dialogue
        
    Returns:
        dict: Subset of the original data with limited dialogues and turns
    """
    all_models_data_subset = {}
    
    # Debug information
    print(f"[INFO] Subsetting data: keeping {max_dialogues} dialogues with {max_turns} turns each")
    
    for model_name, model_data in all_models_data.items():
        # Take only the specified number of dialogues for each model
        model_dialogues_subset = list(model_data.values())[:max_dialogues]
        
        # Trim each dialogue to have only the specified number of turns
        for dialogue in model_dialogues_subset:
            if 'turns' in dialogue and len(dialogue['turns']) > max_turns:
                dialogue['turns'] = dialogue['turns'][:max_turns]
        
        # Add to the new subset
        all_models_data_subset[model_name] = {
            list(model_data.keys())[i]: model_dialogues_subset[i] 
            for i in range(min(max_dialogues, len(model_data)))
        }
    
    # Print summary of the resulting subset
    total_dialogues = sum(len(model_data) for model_data in all_models_data_subset.values())
    print(f"[INFO] Created subset with {len(all_models_data_subset)} models, {total_dialogues} total dialogues")
    
    return all_models_data_subset

 


def create_common_ground_prompt(dialogue: str) -> str:
    """Create a prompt for extracting common ground from dialogue."""
    prompt = """
Analyze the following dialogue about the weights task where participants are weighing blocks (red, blue, green, purple, yellow) on a scale. Only the red block's weight (10g) is initially known.

Extract ONLY the common ground (shared beliefs) about block weights and relations between ALL participants  NOT INCLUDING Friction Agent.
IMPORTANT: Extract common ground from participants only; Do  not include Friction Agent in your answer. 
Represent this as a dictionary with three categories:
1. "equality": Relations where blocks equal each other or a specific weight
2. "inequality": Relations where blocks are explicitly NOT equal
3. "order": Relations showing one block is heavier (>) or lighter (<) than another

Format examples:

Example 1 - Some common ground exists:
{
  "equality": {"red": ["blue", "10g"], "blue": ["red", "10g"]},
  "inequality": {"red": ["green"], "blue": ["green"]},
  "order": {"green": {">": ["red", "blue", "10g"], "<": ["purple"]}}
}

Example 2 - No common ground at all:
{
  "equality": {},
  "inequality": {},
  "order": {}
}

Example 3 - Partial common ground:
{
  "equality": {"red": ["10g"]},  # Some agreement exists here
  "inequality": {},              # Empty - no shared inequality beliefs
  "order": {}                    # Empty - no shared order relations
}

IMPORTANT:
- Only include propositions that at ALL participants explicitly state or clearly agree with
- If there is no common ground for a category, use an empty dictionary: "equality": {}
- If there is no common ground at all, return {"equality": {}, "inequality": {}, "order": {}}
- Do not infer agreement - only count explicit statements or clear acknowledgments
- Disagreements, uncertain claims, or proposals that aren't accepted by others should be excluded

Dialogue:
"""
    return prompt + dialogue


def create_participant_beliefs_prompt(dialogue: str) -> str:
    """Create a prompt for extracting individual participant beliefs about block weights."""
    prompt = """Analyze the following dialogue about the weights task where participants are weighing blocks (red, blue, green, purple, yellow) on a scale. Only the red block's weight (10g) is initially known.

Extract the INDIVIDUAL BELIEFS of EACH participant (P1, P2, and P3) about block weights and relations.
DO NOT extract common ground - instead, provide separate belief structures for each participant.
IMPORTANT: Do not include the Friction Agent in your analysis - focus only on P1, P2, and P3.

For each participant, represent their beliefs as a dictionary with three categories:
1. "equality": Relations where blocks equal each other or a specific weight
2. "inequality": Relations where blocks are explicitly NOT equal
3. "order": Relations showing one block is heavier (>) or lighter (<) than another

Format your response as a JSON object with keys for each participant:

{
  "P1": {
    "equality": {"red": ["blue", "10g"], "blue": ["red", "10g"]},
    "inequality": {"red": ["green"]},
    "order": {"green": {">": ["red", "blue", "10g"], "<": ["purple"]}}
  },
  "P2": {
    "equality": {"red": ["10g"]},
    "inequality": {},
    "order": {"yellow": {">": ["green"]}}
  },
  "P3": {
    "equality": {"red": ["10g"]},
    "inequality": {},
    "order": {"green": {">": ["red", "yellow"]}}
  }
}

IMPORTANT:
- Only include beliefs that the participant explicitly states or clearly endorses
- If a participant has no beliefs in a category, use an empty dictionary: "equality": {}
- If a participant expresses uncertainty, capture this as a separate "uncertain" category within their beliefs
- Include both direct statements and implicit agreements with others' statements
- For each proposition, indicate when participants express confidence levels ("definitely", "maybe", "uncertain", etc.)
- When a participant questions or doubts a proposition, do not include it in their beliefs
- Track how beliefs evolve through the dialogue

Dialogue: """
    return prompt + dialogue

def query_openai_api(prompt: str, model: str = "gpt-4-turbo") -> str:
    """Query OpenAI API and return the response."""
    try:
#         response = openai.ChatCompletion.create(
#             model=model,
#             messages=[{"role": "user", "content": prompt}],
#             temperature=0.0,  # Use low temperature for more consistent results
#             max_tokens=1000
#         )
        
        response = client.chat.completions.create(
                        model="gpt-4o",
                        messages=[
                            {"role": "system", "content": ""},
                            {"role": "user", "content": prompt}
                        ],
              temperature=0.0,  # Use low temperature for more consistent results
            max_tokens=1000
                    )
                                
        return response.choices[0].message.content
    except Exception as e:
        print(f"Error querying OpenAI API: {e}")
#         time.sleep(5)  # Wait before retrying
        return query_openai_api(prompt, model)  # Retry once

def parse_common_ground(response: str) -> Dict[str, Any]:
    """Parse the API response to extract the common ground dictionary."""
    try:
        # Look for dictionary in the response
        response = response.strip()
        # Extract dictionary structure - typically it's enclosed in {}
        start_idx = response.find('{')
        end_idx = response.rfind('}')
        
        if start_idx != -1 and end_idx != -1:
            dict_str = response[start_idx:end_idx+1]
            # Convert to Python dictionary using ast.literal_eval for safety
            common_ground = ast.literal_eval(dict_str)
            
            # Validate the structure
            if not isinstance(common_ground, dict):
                raise ValueError("Response is not a valid dictionary")
            
            # Ensure required keys are present
            for key in ["equality", "inequality", "order"]:
                if key not in common_ground:
                    common_ground[key] = {}
            
            return common_ground
        else:
            # Fallback - return empty dictionary if parsing fails
            return {"equality": {}, "inequality": {}, "order": {}}
    
    except Exception as e:
        print(f"Error parsing response: {e}")
        # Fallback - return empty dictionary
        return {"equality": {}, "inequality": {}, "order": {}}


def parse_individual_propositions(response: str) -> Dict[str, Dict[str, Any]]:
    """
    Parse the API response to extract individual propositions for each participant.
    
    Args:
        response: String response from the LLM containing participant propositions
        
    Returns:
        Dictionary with propositions for each participant (P1, P2, P3)
    """
    try:
        # Look for JSON dictionary in the response
        response = response.strip()
        # Extract dictionary structure - typically enclosed in {}
        start_idx = response.find('{')
        end_idx = response.rfind('}')
        
        if start_idx != -1 and end_idx != -1:
            dict_str = response[start_idx:end_idx+1]
            
            # Try using json.loads first (preferred for JSON)
            try:
                propositions = json.loads(dict_str)
            except json.JSONDecodeError:
                # Fallback to ast.literal_eval if JSON parsing fails
                propositions = ast.literal_eval(dict_str)
            
            # Validate the structure
            if not isinstance(propositions, dict):
                raise ValueError("Response is not a valid dictionary")
            
            # Ensure required participants and keys are present
            for participant in ["P1", "P2", "P3"]:
                if participant not in propositions:
                    propositions[participant] = {"equality": {}, "inequality": {}, "order": {}}
                else:
                    # Ensure each participant has the required keys
                    for key in ["equality", "inequality", "order"]:
                        if key not in propositions[participant]:
                            propositions[participant][key] = {}
            
            return propositions
        else:
            # Fallback - return empty structure if no dictionary found
            return {
                "P1": {"equality": {}, "inequality": {}, "order": {}},
                "P2": {"equality": {}, "inequality": {}, "order": {}},
                "P3": {"equality": {}, "inequality": {}, "order": {}}
            }
    
    except Exception as e:
        print(f"Error parsing propositions: {e}")
        # Fallback - return empty structure
        return {
            "P1": {"equality": {}, "inequality": {}, "order": {}},
            "P2": {"equality": {}, "inequality": {}, "order": {}},
            "P3": {"equality": {}, "inequality": {}, "order": {}}
        }


def process_all_models(all_models_data: Dict[str, Any], output_dir: str = "common_ground_results_WTD_ROGUE_NO_FRICTION_BASELINE", resume_from: Dict[str, int] = None) -> Dict[str, Any]:
    """Process all models and dialogues to extract common ground.
    Returns a combined dictionary of results for all models.
    
    resume_from: Optional dictionary with model_name: dialogue_id pairs to resume processing from.
    """
    # Create output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)
    
    # Dictionary to store results for all models
    all_results = {}
    
    if resume_from is None:
        resume_from = {}

    models_list = ['WTD_faaf_0.01',
    'WTD_faaf_1',
    'WTD_faaf_5',
    'FAAF_friction_run_with_69ksamples_good_vs_rogue_weights_checkpoint-3500',
    'dpo_friction_run_with_69ksamples_checkpoint-2000']
    
    # Process each model
    for model_name, model_data in tqdm(all_models_data.items(), desc="Processing models"):
        print(f"Processing model: {model_name}")
        # if model_name in models_list:
        #     continue
      
        model_results = {}
        print(f"Getting results for model {model_name}")
        
        # Check if we have a previously saved file to load from
        output_file = os.path.join(output_dir, f"{model_name}_common_ground.json")
        if os.path.exists(output_file):
            print(f"Found existing results file for {model_name}, loading it")
            try:
                with open(output_file, "r") as f:
                    model_results = json.load(f)
                print(f"Successfully loaded existing results for {model_name}")
            except Exception as e:
                print(f"Error loading existing results for {model_name}: {e}")
                print("Starting with empty results")
        
        # Get the dialogue ID to resume from (if any)
        resume_dialogue_id = resume_from.get(model_name, 0)
        print(f"Will resume processing from dialogue ID {resume_dialogue_id} for model {model_name}")
        
        # Get all dialogue IDs and sort them
        dialogue_ids = sorted([int(d_id) for d_id in model_data.keys()])
        
        # Filter to only process dialogues from the resume point
        dialogue_ids = [d_id for d_id in dialogue_ids if d_id >= resume_dialogue_id]
        
        # Process each remaining dialogue
        for dialogue_id in dialogue_ids:
            dialogue_id_str = str(dialogue_id)
            print(f"  Processing dialogue: {dialogue_id}")
            
            # Skip if we already have results for this dialogue
            if dialogue_id_str in model_results:
                print(f"  Dialogue {dialogue_id} already processed, skipping")
                continue
                
            dialogue_data = model_data[dialogue_id_str]
            dialogue_results = {
                "turns": []
            }

            # Process each turn in the dialogue
            for turn_idx, turn_data in enumerate(dialogue_data.get("turns", [])):
                utterances = turn_data.get("gpt_utterances", [])
                if isinstance(utterances, list) and all(isinstance(u, str) for u in utterances) and utterances:
                    dialogue_text = " ".join(utterances)

                    # Create prompt and query API
                    prompt = create_participant_beliefs_prompt(dialogue_text)
            
                    api_response = query_openai_api(prompt)

                    # Parse the response
                    common_ground = parse_individual_propositions(api_response)

                    # Store results for this turn
                    turn_results = {
                        "turn_idx": turn_idx,
                        "raw_response": api_response,
                        "parsed_common_ground": common_ground
                    }

                    dialogue_results["turns"].append(turn_results)
                else:
                    print(f"    Skipping turn {turn_idx} in dialogue {dialogue_id} — invalid or empty gpt_utterances")

            # Store results for this dialogue
            model_results[dialogue_id_str] = dialogue_results
            
            # Save intermediate results after each dialogue to avoid losing work
            with open(output_file, "w") as f:
                try:
                    json.dump(model_results, f, indent=2)
                except TypeError as e:
                    if "not JSON serializable" in str(e):
                        print(f"Serialization error in model {model_name}, dialogue {dialogue_id}. Converting sets to lists...")
                        
                        def convert_sets(obj):
                            if isinstance(obj, set):
                                return list(obj)
                            elif isinstance(obj, dict):
                                return {k: convert_sets(v) for k, v in obj.items()}
                            elif isinstance(obj, list):
                                return [convert_sets(item) for item in obj]
                            else:
                                return obj
                        
                        json.dump(convert_sets(model_results), f, indent=2)
                    else:
                        raise
            
            print(f"  Saved intermediate results for dialogue {dialogue_id}")

        # Save final results for this model
        with open(output_file, "w") as f:
            try:
                json.dump(model_results, f, indent=2)
            except TypeError as e:
                if "not JSON serializable" in str(e):
                    print(f"Serialization error in model {model_name}. Converting sets to lists...")
                    
                    def convert_sets(obj):
                        if isinstance(obj, set):
                            return list(obj)
                        elif isinstance(obj, dict):
                            return {k: convert_sets(v) for k, v in obj.items()}
                        elif isinstance(obj, list):
                            return [convert_sets(item) for item in obj]
                        else:
                            return obj
                    
                    json.dump(convert_sets(model_results), f, indent=2)
                else:
                    raise
                    
        print(f"Saved results for model {model_name} to {output_file}")

        # Add this model's results to the combined results
        all_results[model_name] = model_results

    # Save combined results for all models
    combined_output_file = os.path.join(output_dir, "all_models_common_ground.json")
    with open(combined_output_file, "w") as f:
        try:
            json.dump(all_results, f, indent=2)
        except TypeError as e:
            if "not JSON serializable" in str(e):
                print(f"Serialization error in combined results. Converting sets to lists...")
                
                def convert_sets(obj):
                    if isinstance(obj, set):
                        return list(obj)
                    elif isinstance(obj, dict):
                        return {k: convert_sets(v) for k, v in obj.items()}
                    elif isinstance(obj, list):
                        return [convert_sets(item) for item in obj]
                    else:
                        return obj
                
                json.dump(convert_sets(all_results), f, indent=2)
            else:
                raise
    
    print(f"Saved combined results for all models to {combined_output_file}")
    
    return all_results

def subset_model_data(all_models_data, max_dialogues=2, max_turns=2):
    """
    Create a subset of model data with limited dialogues and turns.
    
    Args:
        all_models_data (dict): Dictionary containing all models' data
        max_dialogues (int): Maximum number of dialogues to keep per model
        max_turns (int): Maximum number of turns to keep per dialogue
        
    Returns:
        dict: Subset of the original data with limited dialogues and turns
    """
    all_models_data_subset = {}
    
    # Debug information
    print(f"[INFO] Subsetting data: keeping {max_dialogues} dialogues with {max_turns} turns each")
    
    for model_name, model_data in all_models_data.items():
        # Take only the specified number of dialogues for each model
        model_dialogues_subset = list(model_data.values())[:max_dialogues]
        
        # Trim each dialogue to have only the specified number of turns
        for dialogue in model_dialogues_subset:
            if 'turns' in dialogue and len(dialogue['turns']) > max_turns:
                dialogue['turns'] = dialogue['turns'][:max_turns]
        
        # Add to the new subset
        all_models_data_subset[model_name] = {
            list(model_data.keys())[i]: model_dialogues_subset[i] 
            for i in range(min(max_dialogues, len(model_data)))
        }
    
    # Print summary of the resulting subset
    total_dialogues = sum(len(model_data) for model_data in all_models_data_subset.values())
    print(f"[INFO] Created subset with {len(all_models_data_subset)} models, {total_dialogues} total dialogues")
    
    return all_models_data_subset

if __name__ == "__main__":
    # use expert_iteration_results_wtd for the weights task
    json_data_1 = "expert_iteration_results_deli/all_models_combined.json"
    with open(json_data_1, 'r') as f:
        all_models_data = json.load(f)

    print("LOGGED DATA MODELS", all_models_data.keys())
    all_results = process_all_models(all_models_data, output_dir = "stance_beliefs")

    