import numpy as np
import pandas as pd
from collections import defaultdict, Counter
import logging
import math

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    filename='deli_metrics_log.txt'
)
logger = logging.getLogger(__name__)

def compute_deli_metrics(all_models_data):
    """
    Compute evaluation metrics for all models in the DELI dataset.
    
    Args:
        all_models_data: Dictionary with model names as keys and dialogue data as values
        
    Returns:
        Dictionary with computed metrics (mean and std) for each model
    """
    metrics = {}
    missing_data_counts = defaultdict(int)
    
    for model_name, model_data in all_models_data.items():
        model_metrics = {
            'final_solution_accuracy': [],
            'fine_grained_scores': [],
            'performance_gains': [],
            'initial_solution_diversity': [],
            'discussion_solution_diversity': [],
            'unique_transitions': [],
            'stuck_transitions': [],
            'circular_transitions': [],
            'participant_correct_solutions': defaultdict(list),  # Track per participant correct solutions
            'total_correct_solutions': 0,  # Add this line to track total correct solutions
            'total_solutions': 0  # Add this line to track total solutions
        }
        
        for dialogue_id, dialogue_data in model_data.items():
            # Check for missing data
            if not dialogue_data:
                missing_data_counts[f"{model_name}_missing_dialogue"] += 1
                logger.warning(f"Missing dialogue data for {dialogue_id} in model {model_name}")
                continue
                
            if 'turns' not in dialogue_data or not dialogue_data['turns']:
                missing_data_counts[f"{model_name}_missing_turns"] += 1
                logger.warning(f"Missing turns data for dialogue {dialogue_id} in model {model_name}")
                continue
                
            try:
                # Get the last turn data
                last_turn = dialogue_data['turns'][-1]
                if 'parsed_gpt_response' not in last_turn:
                    missing_data_counts[f"{model_name}_missing_parsed_gpt_response"] += 1
                    logger.warning(f"Missing parsed_gpt_response in last turn for dialogue {dialogue_id} in model {model_name}")
                    continue
                
                last_turn_data = last_turn['parsed_gpt_response']
                
                # 1. Final Solution Accuracy
                final_solution = last_turn_data.get('final_submission_mapped')
                if final_solution is None:
                    missing_data_counts[f"{model_name}_missing_final_submission"] += 1
                    logger.warning(f"Missing final_submission_mapped for dialogue {dialogue_id} in model {model_name}")
                    continue
                    
                is_correct = final_solution == 'OV'
                model_metrics['final_solution_accuracy'].append(int(is_correct))
                
                # 2. Fine-grained Scoring
                score = compute_fine_grained_score(final_solution)
                model_metrics['fine_grained_scores'].append(score)
                
                # 3. Performance Gain
                initial_solutions = dialogue_data.get('initial_solutions')
                if not initial_solutions:
                    missing_data_counts[f"{model_name}_missing_initial_solutions"] += 1
                    logger.info(f"Missing initial_solutions for dialogue {dialogue_id} in model {model_name}")
                else:
                    gain = compute_performance_gain(initial_solutions, final_solution)
                    if gain is not None:
                        model_metrics['performance_gains'].append(gain)
                
                # 4. Initial Solution Diversity
                if initial_solutions:
                    diversity = len(set(map_initial_solutions_to_framework(initial_solutions).values()))
                    model_metrics['initial_solution_diversity'].append(diversity)
                
                # 5. Discussion Solution Diversity
                all_solutions = set()
                
                # 6-8. Solution Transitions
                transitions = extract_solution_transitions(dialogue_data['turns'])
                
                # Track participant correct solutions across turns
                for turn_idx, turn in enumerate(dialogue_data['turns']):
                    if 'parsed_gpt_response' not in turn:
                        missing_data_counts[f"{model_name}_missing_turn_parsed_response"] += 1
                        logger.info(f"Missing parsed_gpt_response in turn {turn_idx} for dialogue {dialogue_id} in model {model_name}")
                        continue
                        
                    # Track solution diversity
                    if 'solution_mappings' in turn['parsed_gpt_response']:
                        solutions = turn['parsed_gpt_response']['solution_mappings'].values()
                        all_solutions.update(solutions)
                    
                    # Track participant correct solutions
                    if 'solution_mappings' in turn['parsed_gpt_response']:
                        for participant, solution in turn['parsed_gpt_response']['solution_mappings'].items():
                            is_correct = solution == 'OV'
                            model_metrics['participant_correct_solutions'][participant].append(int(is_correct))
                            
                            # Track total solutions and correct solutions
                            model_metrics['total_solutions'] += 1
                            if is_correct:
                                model_metrics['total_correct_solutions'] += 1
                
                model_metrics['discussion_solution_diversity'].append(len(all_solutions))
                
                # Analyze transitions
                unique_transitions, stuck_transitions, circular_transitions = analyze_transitions(transitions)
                model_metrics['unique_transitions'].append(unique_transitions)
                model_metrics['stuck_transitions'].append(stuck_transitions)
                model_metrics['circular_transitions'].append(circular_transitions)
                
            except Exception as e:
                logger.error(f"Error processing dialogue {dialogue_id} for model {model_name}: {e}", exc_info=True)
                continue
        
        # Calculate means and standard deviations for the model
        # Calculate means and standard error of the mean (SEM) for the model
        metrics[model_name] = {
            'final_solution_accuracy': {
                'mean': safe_mean(model_metrics['final_solution_accuracy']),
                'sem': safe_std(model_metrics['final_solution_accuracy']) / math.sqrt(len(model_metrics['final_solution_accuracy'])) if model_metrics['final_solution_accuracy'] else 0
            },
            'fine_grained_score': {
                'mean': safe_mean(model_metrics['fine_grained_scores']),
                'sem': safe_std(model_metrics['fine_grained_scores']) / math.sqrt(len(model_metrics['fine_grained_scores'])) if model_metrics['fine_grained_scores'] else 0
            },
            'performance_gain': {
                'mean': safe_mean(model_metrics['performance_gains']),
                'sem': safe_std(model_metrics['performance_gains']) / math.sqrt(len(model_metrics['performance_gains'])) if model_metrics['performance_gains'] else 0
            },
            'initial_solution_diversity': {
                'mean': safe_mean(model_metrics['initial_solution_diversity']),
                'sem': safe_std(model_metrics['initial_solution_diversity']) / math.sqrt(len(model_metrics['initial_solution_diversity'])) if model_metrics['initial_solution_diversity'] else 0
            },
            'discussion_solution_diversity': {
                'mean': safe_mean(model_metrics['discussion_solution_diversity']),
                'sem': safe_std(model_metrics['discussion_solution_diversity']) / math.sqrt(len(model_metrics['discussion_solution_diversity'])) if model_metrics['discussion_solution_diversity'] else 0
            },
            'unique_transitions': {
                'mean': safe_mean(model_metrics['unique_transitions']),
                'sem': safe_std(model_metrics['unique_transitions']) / math.sqrt(len(model_metrics['unique_transitions'])) if model_metrics['unique_transitions'] else 0
            },
            'stuck_transitions': {
                'mean': safe_mean(model_metrics['stuck_transitions']),
                'sem': safe_std(model_metrics['stuck_transitions']) / math.sqrt(len(model_metrics['stuck_transitions'])) if model_metrics['stuck_transitions'] else 0
            },
            'circular_transitions': {
                'mean': safe_mean(model_metrics['circular_transitions']),
                'sem': safe_std(model_metrics['circular_transitions']) / math.sqrt(len(model_metrics['circular_transitions'])) if model_metrics['circular_transitions'] else 0
            },
            'participant_correct_solutions': {},
            # Add total correct solutions statistics
            'total_correct_solutions_count': model_metrics['total_correct_solutions'],
            'total_solutions_count': model_metrics['total_solutions'],
            'total_correct_solutions_percentage': (model_metrics['total_correct_solutions'] / model_metrics['total_solutions'] * 100) if model_metrics['total_solutions'] > 0 else 0
        }
    
    # Log missing data summary
    for key, count in missing_data_counts.items():
        logger.info(f"{key}: {count} instances")
    
    return metrics

def compute_fine_grained_score(solution_mapping):
    """
    Compute fine-grained score based on the 0.25-point system.
    
    Args:
        solution_mapping: String representing the solution mapping (e.g., 'OV', 'EV', etc.)
        
    Returns:
        Float score between 0 and 1
    """
    if solution_mapping is None:
        return 0.0
    
    score = 0.0
    
    # Check for inclusion of target cards
    if 'O' in solution_mapping:
        score += 0.25
    if 'V' in solution_mapping:
        score += 0.25
    
    # Check for exclusion of unnecessary cards
    if 'E' not in solution_mapping:
        score += 0.25
    if 'C' not in solution_mapping:
        score += 0.25
    
    return score

def map_initial_solutions_to_framework(initial_solutions):
    """
    Map initial solutions to the CEOV framework.
    
    Args:
        initial_solutions: Dictionary mapping participant names to lists of selected cards
        
    Returns:
        Dictionary mapping participant names to solution framework strings
    """
    mapped_solutions = {}
    
    # Handle different formats of initial_solutions
    if isinstance(initial_solutions, str):
        try:
            import json
            initial_solutions = json.loads(initial_solutions.replace("'", '"'))
        except:
            logger.warning(f"Could not parse initial_solutions string: {initial_solutions}")
            return mapped_solutions
    
    if not isinstance(initial_solutions, dict):
        logger.warning(f"Initial solutions is not a dictionary: {type(initial_solutions)}")
        return mapped_solutions
    
    for participant, cards in initial_solutions.items():
        if not cards:  # Skip if no cards
            continue
            
        solution = ''
        
        # Check for consonant
        if any(card.isalpha() and card.upper() not in 'AEIOU' for card in cards):
            solution += 'C'
        
        # Check for even number
        if any(card.isdigit() and int(card) % 2 == 0 for card in cards):
            solution += 'E'
        
        # Check for odd number
        if any(card.isdigit() and int(card) % 2 == 1 for card in cards):
            solution += 'O'
        
        # Check for vowel
        if any(card.isalpha() and card.upper() in 'AEIOU' for card in cards):
            solution += 'V'
        
        mapped_solutions[participant] = solution if solution else 'none'
    
    return mapped_solutions

def compute_performance_gain(initial_solutions, final_solution):
    """
    Compute performance gain from initial to final solutions.
    
    Args:
        initial_solutions: Dictionary mapping participant names to lists of selected cards
        final_solution: String representing the final solution mapping
        
    Returns:
        Float representing the average performance gain
    """
    if not initial_solutions or final_solution is None:
        return None
    
    final_score = compute_fine_grained_score(final_solution)
    
    try:
        # Map initial solutions to framework and compute scores
        mapped_solutions = map_initial_solutions_to_framework(initial_solutions)
        
        # Calculate initial scores
        initial_scores = [compute_fine_grained_score(solution) for solution in mapped_solutions.values()]
        
        if not initial_scores:
            return None
        
        # Calculate average initial score
        avg_initial_score = sum(initial_scores) / len(initial_scores)
        
        # Calculate gain
        return final_score - avg_initial_score
    except Exception as e:
        logger.error(f"Error computing performance gain: {e}", exc_info=True)
        return None

def extract_solution_transitions(turns):
    """
    Extract solution transitions for each participant across turns.
    
    Args:
        turns: List of turn data
        
    Returns:
        Dictionary mapping participants to lists of solution sequences
    """
    participant_solutions = defaultdict(list)
    
    for turn in turns:
        if 'parsed_gpt_response' not in turn:
            continue
            
        solution_mappings = turn['parsed_gpt_response'].get('solution_mappings', {})
        
        for participant, solution in solution_mappings.items():
            if solution:  # Skip empty solutions
                participant_solutions[participant].append(solution)
    
    return participant_solutions

def analyze_transitions(participant_solutions):
    """
    Analyze solution transitions to count unique, stuck, and circular transitions.
    
    Args:
        participant_solutions: Dictionary mapping participants to lists of solution sequences
        
    Returns:
        Tuple of (unique transitions count, stuck transitions count, circular transitions count)
    """
    unique_transitions = set()
    stuck_transitions = 0
    circular_transitions = 0
    
    for participant, solutions in participant_solutions.items():
        if len(solutions) < 3:  # Need at least 3 solutions for a transition triple
            continue
            
        # Create transition triples
        for i in range(len(solutions) - 2):
            triple = f"{solutions[i]}-{solutions[i+1]}-{solutions[i+2]}"
            unique_transitions.add(triple)
            
            # Check for stuck transitions
            if solutions[i] == solutions[i+1] == solutions[i+2]:
                stuck_transitions += 1
            
            # Check for circular transitions
            if solutions[i] == solutions[i+2] and solutions[i] != solutions[i+1]:
                circular_transitions += 1
    
    return len(unique_transitions), stuck_transitions, circular_transitions

def safe_mean(values):
    """Calculate mean safely even with empty lists."""
    return float(np.mean(values)) if values else 0.0

def safe_std(values):
    """Calculate standard deviation safely even with empty lists."""
    return float(np.std(values)) if len(values) > 1 else 0.0

def print_metrics(metrics):
    """Print metrics in a readable format."""
    print("Model Performance Metrics:\n")
    
    for model_name, model_metrics in metrics.items():
        print(f"Model: {model_name}")
        print(f"  Final Solution Accuracy: {model_metrics['final_solution_accuracy']['mean']:.2f} ± {model_metrics['final_solution_accuracy']['sem']:.2f}")
        print(f"  Fine-grained Score: {model_metrics['fine_grained_score']['mean']:.2f} ± {model_metrics['fine_grained_score']['sem']:.2f}")
        print(f"  Performance Gain: {model_metrics['performance_gain']['mean']:.2f} ± {model_metrics['performance_gain']['sem']:.2f}")
        print(f"  Initial Solution Diversity: {model_metrics['initial_solution_diversity']['mean']:.2f} ± {model_metrics['initial_solution_diversity']['sem']:.2f}")
        print(f"  Discussion Solution Diversity: {model_metrics['discussion_solution_diversity']['mean']:.2f} ± {model_metrics['discussion_solution_diversity']['sem']:.2f}")
        print(f"  Unique Transitions: {model_metrics['unique_transitions']['mean']:.2f} ± {model_metrics['unique_transitions']['sem']:.2f}")
        print(f"  Stuck Transitions: {model_metrics['stuck_transitions']['mean']:.2f} ± {model_metrics['stuck_transitions']['sem']:.2f}")
        print(f"  Circular Transitions: {model_metrics['circular_transitions']['mean']:.2f} ± {model_metrics['circular_transitions']['sem']:.2f}")
        
        print(f"  Total Solutions: {model_metrics['total_solutions_count']} (Correct: {model_metrics['total_correct_solutions_count']}, {model_metrics['total_correct_solutions_percentage']:.2f}%)")

if __name__ == "__main__":
    import os

    # Define paths to collaboration logs
    deli_logs_path = "path/to/delidata_collaboration_logs.json"  
    wtd_logs_path = "path/to/weights_task_collaboration_logs.json" 

    # ---- Evaluate Delidata Task ----
    print("Evaluating Delidata collaboration logs...")
    deli_metrics = compute_deli_metrics(deli_logs_path)
    print("\nDelidata Evaluation Metrics:")
    print_metrics(deli_metrics)

    # ---- Evaluate Weights Task ----
    print("\nEvaluating Weights Task collaboration logs...")
    print("Starting belief alignment and reward analysis...")
    weights_results = analyze_individual_beliefs_rewards(wtd_logs_path)

    # Print overall summary statistics
    print("\nSummary Statistics:")
    print_results_summary(weights_results)

    # Print per-turn detailed statistics
    print("\nPer-Turn Statistics:")
    turn_stats = generate_statistics_per_turn(weights_results)
    print_per_turn_statistics(turn_stats)

    
 
    
 