
import json
import re


def preprocess_deli_response(response, dialogue_context):
    """
    Preprocess response to fix common JSON structure issues.
    
    Args:
        response: The model's response to fix
        dialogue_context: The dialogue context (to extract participant names)
    
    Returns:
        Preprocessed response with fixed structure
    """
    try:
        # Parse the response
        data = json.loads(response)
        
        # Extract participant names from dialogue context if available
        participant_names = extract_participant_names(dialogue_context)
        
        # Case 1: Missing "participants" wrapper
        if "participants" not in data and any(k.startswith("ParticipantName") for k in data.keys()):
            # Create new structure with participants wrapper
            new_data = {"participants": {}}
            
            # Process each participant
            for i, (key, value) in enumerate(data.items()):
                # Replace generic name with actual name if available
                if i < len(participant_names):
                    participant_name = participant_names[i]
                else:
                    participant_name = key
                
                cards = value.get("cards", [])
                stances = {}
                
                # Fix malformed stances with pipe separators
                for card, stance in value.get("stances", {}).items():
                    if "|" in stance:
                        # Take the first value before the pipe
                        stances[card] = stance.split("|")[0]
                    else:
                        stances[card] = stance
                
                new_data["participants"][participant_name] = {
                    "cards": cards,
                    "stances": stances
                }
            
            return json.dumps(new_data)
        
        # Case 2: Has participants but with malformed stances
        if "participants" in data:
            for participant, info in data["participants"].items():
                if "stances" in info:
                    for card, stance in info["stances"].items():
                        if "|" in stance:
                            info["stances"][card] = stance.split("|")[0]
            
            return json.dumps(data)
        
        # If data has ParticipantState structure (from previous example)
        if "ParticipantState" in data:
            # Handle this structure - implementation omitted for brevity
            pass
    
    except Exception as e:
        print(f"Error preprocessing response: {e}")
    
    # If we couldn't fix it, return original
    return response


def calculate_deli_format_quality(model_output):
    """
    Enhanced format quality assessment for DELI responses.
    """
    score = 0.0
    
    # Basic checks
    try:
        data = json.loads(model_output)
        score += 0.3  # Valid JSON
    except:
        return score
    
    # Structure checks
    if "participants" in data:
        score += 0.3
        
        # Check participants structure
        for participant, info in data["participants"].items():
            if isinstance(info, dict):
                score += 0.1
                
                if "cards" in info and isinstance(info["cards"], list):
                    score += 0.1
                    
                if "stances" in info and isinstance(info["stances"], dict):
                    score += 0.1
                    
                    # Check stances format
                    valid_stances = True
                    for card, stance in info["stances"].items():
                        if "|" in stance or stance not in ["support", "oppose", "unsure", "conditional"]:
                            valid_stances = False
                    
                    if valid_stances:
                        score += 0.1
    
    # Partial structure (without "participants" wrapper)
    elif any(k.startswith("Participant") for k in data.keys()):
        score += 0.1
        
        for key, info in data.items():
            if isinstance(info, dict) and "cards" in info and "stances" in info:
                score += 0.1
    
    return min(1.0, score)


# Helper functions
import json
import re


def robust_json_parse(model_output):
    """Parse JSON from model output with extreme fallback mechanisms."""
    if not model_output or not isinstance(model_output, str):
        return {}  # Return empty dict for empty input
    
    try:
        # Try to extract JSON if it's surrounded by other text
        match = re.search(r'\{.*\}', model_output, re.DOTALL)
        if match:
            json_str = match.group(0)
            return json.loads(json_str)
        else:
            return json.loads(model_output)
    except json.JSONDecodeError:
        # Fallback 1: Try with replaced quotes
        cleaned_output = model_output.replace("'", "\"")
        try:
            return json.loads(cleaned_output)
        except:
            # Fallback 2: Try with additional cleanup
            try:
                cleaned_output = cleanup_json(cleaned_output)
                return json.loads(cleaned_output)
            except:
                # Fallback 3: Try to extract any JSON-like object
                try:
                    # Look for patterns like "cards": ["A", "B"] or "stances": {...}
                    cards_match = re.search(r'"cards"\s*:\s*\[(.*?)\]', model_output, re.DOTALL)
                    stances_match = re.search(r'"stances"\s*:\s*\{(.*?)\}', model_output, re.DOTALL)
                    
                    extracted = {}
                    if cards_match:
                        try:
                            # Try to parse the cards array
                            cards_str = "[" + cards_match.group(1) + "]"
                            cards = json.loads(cards_str.replace("'", "\""))
                            extracted["cards"] = cards
                        except:
                            pass
                    
                    if stances_match:
                        try:
                            # Try to parse the stances object
                            stances_str = "{" + stances_match.group(1) + "}"
                            stances = json.loads(stances_str.replace("'", "\""))
                            extracted["stances"] = stances
                        except:
                            pass
                    
                    if extracted:
                        # Create a minimal valid JSON structure
                        return {"participants": {"extracted_participant": extracted}}
                    
                except:
                    pass
                
                # Fallback 4: Create a basic structure by regex matching all possible cards
                try:
                    # Look for any card references like "A", "7", etc.
                    vowel_cards = ["A", "E", "I", "O", "U"]
                    number_cards = ["1", "2", "3", "4", "5", "6", "7", "8", "9"]
                    all_cards = vowel_cards + number_cards
                    
                    found_cards = []
                    for card in all_cards:
                        if re.search(r'["\']\s*' + card + r'\s*["\']', model_output):
                            found_cards.append(card)
                    
                    if found_cards:
                        # Create a minimal valid JSON structure with found cards
                        return {
                            "participants": {
                                "extracted_participant": {
                                    "cards": found_cards,
                                    "stances": {card: "support" for card in found_cards}
                                }
                            }
                        }
                    
                except:
                    pass
                
                # Return an empty structure rather than raising an exception
                return {"participants": {}}



def cleanup_json(json_str):
    """Attempt to fix common JSON syntax errors."""
    # Remove trailing commas
    json_str = re.sub(r',\s*}', '}', json_str)
    json_str = re.sub(r',\s*]', ']', json_str)
    
    # Add missing closing braces/brackets
    open_braces = json_str.count('{')
    close_braces = json_str.count('}')
    if open_braces > close_braces:
        json_str += '}' * (open_braces - close_braces)
        
    open_brackets = json_str.count('[')
    close_brackets = json_str.count(']')
    if open_brackets > close_brackets:
        json_str += ']' * (open_brackets - close_brackets)
        
    # Fix missing quotes around keys
    json_str = re.sub(r'([{,]\s*)(\w+)(\s*:)', r'\1"\2"\3', json_str)
    
    return json_str

def parse_gold_standard(dialogue_context):
    """Parse the gold standard from the dialogue context."""
    # Check for None input
    if dialogue_context is None or not isinstance(dialogue_context, str) or not dialogue_context.strip():
        return {"participants": {}}
    
    try:
        # Extract JSON from the dialogue context
        match = re.search(r'{.*}', dialogue_context, re.DOTALL)
        if match:
            gold_json_str = match.group(0)
            return robust_json_parse(gold_json_str)
    except Exception as e:
        print(f"Warning: Failed to parse gold standard: {e}")
        
    # Fallback: return empty structure
    return {"participants": {}}

def calculate_ppo_deli_gold_reward(model_output, dialogue_context):
    """
    Calculate gold reward with improved robustness and partial agreement scoring.
    """
    # Check for None inputs first
    if model_output is None:
        print("Warning: model_output is None")
        return {"total_reward": -1.0, "error": "model_output is None", "fallback_used": True}
    
    if dialogue_context is None:
        print("Warning: dialogue_context is None")
        return {"total_reward": -1.0, "error": "dialogue_context is None", "fallback_used": True}
    
    try:
        # Use ultra-robust parsing that never fails
        model_json = robust_json_parse(model_output)
        
        # Check if parsing returned an empty or minimal structure
        if not model_json or not model_json.get("participants"):
            # If we got an empty result, use the fallback reward based on text pattern matching
            fallback_reward = calculate_fallback_reward(model_output, dialogue_context)
            return {
                "total_reward": fallback_reward,
                "error": "Empty JSON structure after parsing",
                "fallback_used": True
            }
        
        # Extract participant information
        model_participants = model_json.get("participants", {})
        
        # Parse the gold standard from the dialogue context
        gold_json = parse_gold_standard(dialogue_context)
        gold_participants = gold_json.get("participants", {})
        
        # If gold standard couldn't be parsed, use a very conservative reward
        if not gold_participants:
            print("Warning: Could not parse gold standard from dialogue context")
            return {
                "total_reward": -0.5,  # Conservative negative reward
                "error": "Could not parse gold standard",
                "fallback_used": True
            }
        
        # Calculate partial agreement scores
        participant_match_score = calculate_participant_match(model_participants, gold_participants)
        card_match_score = calculate_card_match(model_participants, gold_participants)
        stance_match_score = calculate_stance_match(model_participants, gold_participants)
        
        # Weighted combination of scores
        agreement_score = (
            participant_match_score * 0.3 +
            card_match_score * 0.4 +
            stance_match_score * 0.3
        )
        
        # Scale to desired range (-1 to 2)
        scaled_reward = (agreement_score * 3.0) - 1.0
        
        return {
            "total_reward": scaled_reward,
            "participant_match": participant_match_score,
            "card_match": card_match_score,
            "stance_match": stance_match_score,
            "agreement_score": agreement_score,
            "parsed_json": model_json
        }
        
    except Exception as e:
        print(f"Gold reward calculation failed (unhandled exception): {str(e)}")
        # Print stack trace for debugging
        import traceback
        traceback.print_exc()
        
        # Return some partial credit based on text similarity if JSON parsing fails
        try:
            fallback_reward = calculate_fallback_reward(model_output, dialogue_context)
            return {
                "total_reward": fallback_reward,
                "error": str(e),
                "fallback_used": True
            }
        except Exception as fallback_error:
            print(f"Even fallback reward calculation failed: {fallback_error}")
            # Last resort: return a fixed negative reward
            return {
                "total_reward": -0.8,
                "error": f"Failed to calculate reward: {e}, fallback also failed: {fallback_error}",
                "fallback_used": True
            }
def calculate_participant_match(model_participants, gold_participants):
    """Calculate how well the model matched participant names."""
    if not model_participants or not gold_participants:
        return 0.0
        
    # Extract just the participant names
    model_names = set(model_participants.keys())
    gold_names = set(gold_participants.keys())
    
    # If the model used generic names like "ParticipantName1", check for count match
    generic_prefix = "ParticipantName"
    if any(name.startswith(generic_prefix) for name in model_names):
        # Give partial credit for at least getting the count right
        return min(len(model_names), len(gold_names)) / max(len(model_names), len(gold_names))
    
    # Calculate Jaccard similarity for name matching
    intersection = len(model_names.intersection(gold_names))
    union = len(model_names.union(gold_names))
    
    return intersection / union if union > 0 else 0.0
    
def calculate_card_match(model_participants, gold_participants):
    """Calculate how well the model matched cards for each participant."""
    if not model_participants or not gold_participants:
        return 0.0
        
    total_score = 0.0
    count = 0
    
    for gold_name, gold_info in gold_participants.items():
        # Find best matching participant in model output
        best_match = find_best_matching_participant(gold_name, model_participants)
        
        if best_match:
            model_cards = set(model_participants[best_match].get("cards", []))
            gold_cards = set(gold_info.get("cards", []))
            
            # Calculate card match (Jaccard similarity)
            intersection = len(model_cards.intersection(gold_cards))
            union = len(model_cards.union(gold_cards))
            
            score = intersection / union if union > 0 else 0.0
            total_score += score
            count += 1
    
    return total_score / count if count > 0 else 0.0

def calculate_stance_match(model_participants, gold_participants):
    """Calculate how well the model matched stances for each participant's cards."""
    if not model_participants or not gold_participants:
        return 0.0
        
    total_score = 0.0
    count = 0
    
    for gold_name, gold_info in gold_participants.items():
        # Find best matching participant in model output
        best_match = find_best_matching_participant(gold_name, model_participants)
        
        if best_match:
            model_stances = model_participants[best_match].get("stances", {})
            gold_stances = gold_info.get("stances", {})
            
            # Track matching stances
            correct_stances = 0
            total_stances = 0
            
            # Check each card in gold stance
            for card, gold_stance in gold_stances.items():
                if card in model_stances:
                    model_stance = model_stances[card]
                    # Handle pipe-separated stance values
                    if isinstance(model_stance, str) and "|" in model_stance:
                        model_stance = model_stance.split("|")[0]
                    
                    if isinstance(model_stance, str) and model_stance.lower() == gold_stance.lower():
                        correct_stances += 1
                total_stances += 1
            
            if total_stances > 0:
                score = correct_stances / total_stances
                total_score += score
                count += 1
    
    return total_score / count if count > 0 else 0.0

def find_best_matching_participant(gold_name, model_participants):
    """Find the best matching participant in the model output for a gold participant."""
    if gold_name in model_participants:
        return gold_name
        
    # If exact match not found, try to find best match based on similarity
    # For simplicity, just return first participant if gold_name not found
    if model_participants:
        return list(model_participants.keys())[0]
        
    return None

def calculate_fallback_reward(model_output, dialogue_context):
    """Calculate a fallback reward when JSON parsing fails completely."""
    # Check for None inputs
    if model_output is None:
        model_output = ""
    if dialogue_context is None:
        dialogue_context = ""
    
    # Track scores for different aspects
    format_score = 0.0
    content_score = 0.0
    
    # 1. Check for JSON-like structure (even if invalid)
    if "{" in model_output and "}" in model_output:
        format_score += 0.1
    
    if "participants" in model_output:
        format_score += 0.1
    
    if "cards" in model_output:
        format_score += 0.1
    
    if "stances" in model_output:
        format_score += 0.1
        
    # 2. Extract card mentions for content evaluation
    vowel_cards = ["A", "E", "I", "O", "U"]
    odd_numbers = ["1", "3", "5", "7", "9"]
    
    has_vowel = any(re.search(r'["\']\s*' + card + r'\s*["\']', model_output) for card in vowel_cards)
    has_odd = any(re.search(r'["\']\s*' + num + r'\s*["\']', model_output) for num in odd_numbers)
    
    # 3. Extract participant names from dialogue - safely with null checks
    dialogue_names = extract_participant_names(dialogue_context)
    has_correct_names = any(name in model_output for name in dialogue_names) if dialogue_names else False
    
    # 4. Calculate content score based on card and participant mentions
    if has_vowel:
        content_score += 0.3
    
    if has_odd:
        content_score += 0.3
    
    if has_vowel and has_odd:
        content_score += 0.1
    
    if has_correct_names:
        content_score += 0.2
    
    # 5. Calculate total fallback score
    total_score = (format_score * 0.3) + (content_score * 0.7)
    
    # Apply a penalty for completely failed parsing
    format_penalty = 0.5
    final_score = total_score - format_penalty
    
    # Scale to appropriate range
    return max(-0.8, min(0.3, final_score))

def extract_participant_names(dialogue_context):
    """Extract participant names from dialogue context."""
    names = set()
    
    # Look for patterns like "Name: " at the start of lines
    for line in dialogue_context.split('\n'):
        match = re.match(r'^([A-Za-z]+):', line.strip())
        if match:
            names.add(match.group(1))
    
    return names

def extract_cards_from_malformed_json(model_output):
    """
    Extract card information even from malformed outputs.
    Returns a list of cards and whether they've been identified correctly.
    """
    # Look for common cards in Wason tasks
    vowel_cards = ["A", "E", "I", "O", "U"]
    number_cards = ["1", "2", "3", "4", "5", "6", "7", "8", "9"]
    
    # Check if any cards are mentioned in the text
    mentioned_cards = set()
    
    # Look for vowel cards
    has_vowel_card = False
    for card in vowel_cards:
        if f'"{card}"' in model_output or f"'{card}'" in model_output:
            mentioned_cards.add(card)
            has_vowel_card = True
    
    # Look for number cards
    has_odd_number = False
    for card in number_cards:
        if f'"{card}"' in model_output or f"'{card}'" in model_output:
            mentioned_cards.add(card)
            # Check if it's an odd number
            if card in ["1", "3", "5", "7", "9"]:
                has_odd_number = True
    
    return {
        "cards": list(mentioned_cards),
        "has_vowel": has_vowel_card,
        "has_odd_number": has_odd_number
    }

def calculate_ppo_deli_proxy_reward(model_output, dialogue_context):
    """
    Proxy reward function with improved card extraction even from malformed JSON.
    
    Prioritizes accuracy (correct cards) while still penalizing bad format.
    """
    try:
        # First try to parse normally
        try:
            parsed_beliefs = robust_json_parse(model_output)
            format_score = calculate_deli_format_quality(model_output)
            
            # Check if it has the minimum required structure
            is_valid = is_valid_deli_format(parsed_beliefs)
            
            if is_valid:
                # Calculate rewards normally
                correctness_data = calculate_deli_correctness(parsed_beliefs)
                agreement_data = calculate_deli_agreement(parsed_beliefs)
                
                # Weighted combination as before
                correctness_weight = 0.7
                agreement_weight = 0.2
                format_weight = 0.1
                
                normalized_reward = (
                    correctness_data["correctness_score"] * correctness_weight +
                    agreement_data["agreement_score"] * agreement_weight +
                    format_score * format_weight
                )
                
                # Scale to -1 to +2 range
                if normalized_reward < 0.3:
                    scaled_reward = (normalized_reward * 3.33) - 1.0
                else:
                    scaled_reward = (normalized_reward - 0.3) * 2.85
                    
                return {
                    "total_reward": scaled_reward,
                    "parsed_beliefs": parsed_beliefs,
                    "correctness_score": correctness_data["correctness_score"],
                    "agreement_score": agreement_data["agreement_score"],
                    "format_score": format_score
                }
            
        except Exception as parse_error:
            # Normal parsing failed, try extraction approach
            pass
        
        # If we're here, either parsing failed or structure was invalid
        # Try to extract card information even from malformed JSON
        card_info = extract_cards_from_malformed_json(model_output)
        
        # Calculate a basic correctness score based on card selection
        # Give partial credit for having vowel and odd number cards
        card_correctness = 0.0
        
        if card_info["has_vowel"]:
            card_correctness += 0.3  # 30% for having vowel card
        
        if card_info["has_odd_number"]:
            card_correctness += 0.3  # 30% for having odd number card
            
        # Additional 20% if both are present
        if card_info["has_vowel"] and card_info["has_odd_number"]:
            card_correctness += 0.2
        
        # Format penalty - still need to encourage proper format
        format_penalty = 0.5  # 50% penalty for bad format
        
        # Calculate final score
        total_reward = (card_correctness - format_penalty)
        
        # Ensure range is between -1 and +1 for malformed outputs
        total_reward = max(-0.8, min(0.3, total_reward))
        
        return {
            "total_reward": total_reward,
            "extracted_cards": card_info["cards"],
            "has_vowel": card_info["has_vowel"],
            "has_odd_number": card_info["has_odd_number"],
            "card_correctness": card_correctness,
            "format_penalty": format_penalty,
            "parse_error": "Invalid JSON structure"
        }
        
    except Exception as e:
        print(f"Proxy reward calculation failed: {e}")
        return {
            "total_reward": -1.0,
            "error": str(e)
        }


def robust_json_parse(model_output):
    """Parse JSON from model output with fallback mechanisms."""
    try:
        # Try to extract JSON if it's surrounded by other text
        match = re.search(r'\{.*\}', model_output, re.DOTALL)
        if match:
            json_str = match.group(0)
            return json.loads(json_str)
        else:
            return json.loads(model_output)
    except json.JSONDecodeError:
        # Fallback: Try to fix common JSON errors
        cleaned_output = model_output.replace("'", "\"")
        try:
            return json.loads(cleaned_output)
        except:
            raise ValueError("Failed to parse JSON")

def is_valid_deli_format(parsed_beliefs):
    """Validate the structure of the parsed beliefs."""
    if not isinstance(parsed_beliefs, dict):
        return False
    
    if "participants" not in parsed_beliefs:
        return False
    
    if not isinstance(parsed_beliefs["participants"], dict):
        return False
    
    # Check each participant has required fields
    for participant, data in parsed_beliefs["participants"].items():
        if not isinstance(data, dict):
            return False
        
        if "cards" not in data or "stances" not in data:
            return False
            
        if not isinstance(data["cards"], list) or not isinstance(data["stances"], dict):
            return False
    
    return True

def assess_deli_format_quality(model_output):
    """Assess the quality of the JSON format (0.0-1.0)."""
    # Simple heuristic scoring for format quality
    score = 0.0
    
    # Check for valid JSON structure
    try:
        json.loads(model_output)
        score += 0.5  # Major points for valid JSON
    except:
        try:
            # Try with single quote replacement
            json.loads(model_output.replace("'", "\""))
            score += 0.3  # Fewer points for fixable JSON
        except:
            return score  # Failed both checks
    
    # Check for expected fields
    if '"participants"' in model_output or "'participants'" in model_output:
        score += 0.1
    
    if '"cards"' in model_output or "'cards'" in model_output:
        score += 0.2
        
    if '"stances"' in model_output or "'stances'" in model_output:
        score += 0.2
    
    return min(1.0, score)

def calculate_deli_agreement(parsed_beliefs):
    """
    Calculate agreement (common ground) between participants on card stances.
    
    Returns:
        dict with agreement score and details
    """
    participants = parsed_beliefs["participants"]
    
    # Check if there are multiple participants to calculate agreement
    if len(participants) <= 1:
        return {
            "agreement_score": 0.0,
            "details": {"error": "Insufficient participants to calculate agreement"}
        }
    
    # Collect all cards mentioned by any participant
    all_cards = set()
    for participant, data in participants.items():
        for card in data.get("cards", []):
            all_cards.add(card)
    
    total_possible_agreements = len(all_cards) * len(participants)
    agreement_count = 0
    
    # Count agreements on stances
    card_agreements = {}
    for card in all_cards:
        stances = []
        participants_with_stance = []
        
        for participant, data in participants.items():
            stance = data.get("stances", {}).get(card)
            if stance:
                stances.append(stance)
                participants_with_stance.append(participant)
        
        # Calculate agreements for this card
        if len(stances) > 1:
            # Check if all stances are the same
            if len(set(stances)) == 1:
                # Full agreement
                agreement_count += len(stances)
                card_agreements[card] = {
                    "stance": stances[0],
                    "participants": participants_with_stance,
                    "type": "full_agreement"
                }
            else:
                # Partial agreement
                stance_counts = {}
                for stance in stances:
                    stance_counts[stance] = stance_counts.get(stance, 0) + 1
                
                # Find most common stance
                most_common_stance = max(stance_counts.items(), key=lambda x: x[1])
                agreement_count += most_common_stance[1]
                
                card_agreements[card] = {
                    "stance": most_common_stance[0],
                    "participants": [p for p, data in zip(participants_with_stance, stances) 
                                     if data == most_common_stance[0]],
                    "type": "partial_agreement"
                }
    
    # Calculate agreement score (0.0 to 1.0+)
    agreement_score = agreement_count / max(1, total_possible_agreements)
    
    return {
        "agreement_score": agreement_score,
        "details": {
            "card_agreements": card_agreements,
            "agreement_count": agreement_count,
            "total_possible": total_possible_agreements
        }
    }

def calculate_deli_correctness(parsed_beliefs):
    """
    Calculate correctness of card selections against the correct Wason task answer.
    
    Correct answer: 'A' and '7' cards
    
    Returns:
        dict with correctness score and details
    """
    participants = parsed_beliefs["participants"]
    
    # Correct solution:
    correct_cards = {"A", "7"}
    correct_stances = {
        "A": "support",
        "7": "support",
        "K": "oppose",
        "2": "oppose",
        "4": "oppose",
        "3": "oppose",
        "D": "oppose"
    }
    
    total_correctness = 0.0
    details = {"participants": {}}
    
    for participant, data in participants.items():
        participant_cards = set(data.get("cards", []))
        participant_stances = data.get("stances", {})
        
        # Track this participant's correctness
        participant_score = 0.0
        participant_details = {
            "cards": [],
            "score": 0.0
        }
        
        # Check each card's correctness
        for card, stance in participant_stances.items():
            card_score = 0.0
            card_detail = {"card": card, "stance": stance}
            
            # Check against correct solution
            if card in correct_cards:
                if stance == "support":
                    card_score = 1.0  # Correct positive selection
                    card_detail["assessment"] = "correct_selection"
                else:
                    card_score = -0.5  # Incorrect stance on key card
                    card_detail["assessment"] = "incorrect_stance_key_card"
            else:
                # Non-key card
                if stance == "oppose" or stance == "unsure":
                    card_score = 0.5  # Correctly not selected
                    card_detail["assessment"] = "correct_non_selection"
                elif stance == "conditional":
                    card_score = 0.2  # Partial credit for conditional
                    card_detail["assessment"] = "partial_conditional"
                else:
                    card_score = -0.5  # Incorrectly selected
                    card_detail["assessment"] = "incorrect_selection"
            
            # Bonus for matching correct stance exactly
            if card in correct_stances and stance == correct_stances[card]:
                card_score += 0.2
                card_detail["bonus"] = "exact_stance_match"
                
            participant_score += card_score
            card_detail["score"] = card_score
            participant_details["cards"].append(card_detail)
        
        # Normalize participant score
        max_possible = len(participant_stances) * 1.2  # Maximum score with all cards correct
        if max_possible > 0:
            participant_details["score"] = participant_score / max_possible
        else:
            participant_details["score"] = 0.0
            
        total_correctness += participant_details["score"]
        details["participants"][participant] = participant_details
    
    # Average correctness across participants
    if len(participants) > 0:
        avg_correctness = total_correctness / len(participants)
    else:
        avg_correctness = 0.0
    
    # Scale to -1.0 to 1.0 range
    scaled_correctness = (avg_correctness * 2.0) - 1.0
    
    return {
        "correctness_score": scaled_correctness,
        "details": details
    }
