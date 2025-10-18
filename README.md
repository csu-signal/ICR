# ICR Agent Codebase – Supplementary Submission

This repository contains code for training and evaluating our Interruptible Collaborative Roleplayer (ICR) agents from the paper, "Learning “Partner-Aware” Collaborators in Multi-Party Collaboration", accepted for publication at NeurIPS 2025 Main Conference track. The code supports both the DeliData Wason Card Task and the Weights Task for running all experiments in the main paper.

## File Overview

| File | Description |
|------|-------------|
| `requirements.txt` | Please install the packages and dependencies to run the code. |
| `BC_collaborator.py` | Implements behavior cloning (BC) or Supervised finetuning (SFT) for training expert collaborators from full trajectory data. |
| `expert_mamdp_collaboration.py` | Collects data using roleplay simulation using a fixed GPT-4o intervention agent and collaborator under the MAMDP setup. |
| `ICR_trainer.py` | Defines the main trainer class for ICR agents, including PPO optimization and counterfactual KL regularization. |
| `ICR_training.py` | Launch script for running ICR training across tasks. This script is also used to train the standard PPO baseline. |
| `no_press_stance_generation.py` | Generate stances or discrete propositions from collected expert data for "no-press" training (structured stance output without free-form language). |
| `reward_functions.py` | Implements proxy reward functions used for reward modeling and task-specific scoring. |
| `eval.py` | Computes final evaluation metrics, including reward-based scores and CG (common-ground) statistics from logged interaction trajectories. |

## Usage Notes

- Install the dependencies from the requirements.txt file using pip (preferred):
  ```bash
  pip install -r requirements.txt
- The PPO and ICR agents rely on the [HuggingFace TRL](https://github.com/huggingface/trl) and PEFT libraries for LoRA and PPO training.

 
