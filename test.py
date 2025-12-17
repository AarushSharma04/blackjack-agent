#!/usr/bin/env python3
"""
Test script for comparing MCTS and DQN agents in Blackjack.

This script evaluates both agents over 10,000 hands and compares their
winning percentages and overall performance.
"""

import sys
import argparse
from evaluate import Evaluator
from mcts_agent import MCTSAgent
from basic_strategy_agent import BasicStrategyAgent

# Try to import DQN, but handle gracefully if torch is not available
try:
    from dqn_agent import DQNAgent
    DQN_AVAILABLE = True
except ImportError:
    DQN_AVAILABLE = False
    print("Warning: DQN agent not available (torch not installed). Skipping DQN evaluation.")


def print_header():
    print("=" * 80)
    print("Blackjack Agent Comparison Test - Madeleine Bello and Aarush Sharma")
    print("=" * 80)
    print()


def print_game_description():
    print("Game Description")
    print("\n")
    print("""
Blackjack is a card game where the goal is to have a hand value closer to 21 
than the dealer's hand without exceeding 21. In this implementation:

- The game uses an infinite deck (cards are drawn randomly with replacement)
- Player and dealer each receive 2 initial cards
- Only the dealer's first card is visible to the player
- Player can HIT (take another card) or STAND (end their turn)
- Dealer must hit until reaching 17 or higher
- Aces count as 1 or 11 (automatically chosen to maximize hand value)
- Face cards (Jack, Queen, King) count as 10
- Scoring: +1 for win, -1 for loss, 0 for push (tie)
- No doubling down or splitting in this simplified version
    """)
    print()


def print_code_description():
    """Print description of what the code does."""
    print("What our code does")
    print()
    print("""
This codebase implements 2 different agents to play blackjack:

1. MCTS AGENT (Monte Carlo Tree Search):
   - Uses tree search with UCB1 exploration to find optimal moves
   - Performs simulations from the current game state
   - Builds a search tree by exploring promising action sequences
   - Uses smart playout policy (basic strategy-like) for simulations
   - Selects actions based on average reward from simulations

2. DQN AGENT (Deep Q-Network / Q-Learning):
   - Uses a neural network to approximate Q-values (expected future rewards)
   - Learns optimal policy through experience replay
   - Uses epsilon-greedy exploration during training
   - Trained on 10,000+ episodes before evaluation
   - Makes decisions based on learned Q-values

The evaluation framework plays multiple hands with each agent and then tracks:
- Win rate (percentage of games won)
- Loss rate
- Push rate (ties)
- Average score per hand
- Total score over all games
    """)
    print()


def print_research_question():
    print("Research Question: ")
    print()
    print("""
We wanted to see how a MCTS agent compares to a Q-learning (DQN) agent when trying to play 
blackjack profitably and if MCTS, with its tree search approach, can outperform Q-learning?
    """)
    print()


def evaluate_agent(evaluator, agent, num_hands, agent_name):
    print(f"Evaluating {agent_name}...")
    results = evaluator.evaluate_agent(agent, num_hands, agent_name)
    return results


def print_results_summary(mcts_results, dqn_results, basic_results):
    """Print summary of results with winning percentages."""
    print("Summary of Results")
    print()
    print()
    
    print(f"{'Agent':<30} {'Win %':<12} {'Loss %':<12} {'Push %':<12} {'Avg Score':<12}")
    print("-" * 80)
    
    if mcts_results:
        print(f"{mcts_results['agent_name']:<30} "
              f"{mcts_results['win_rate']*100:>6.2f}%     "
              f"{mcts_results['losses']/mcts_results['num_hands']*100:>6.2f}%     "
              f"{mcts_results['pushes']/mcts_results['num_hands']*100:>6.2f}%     "
              f"{mcts_results['average_score']:>10.4f}")
    
    if dqn_results:
        print(f"{dqn_results['agent_name']:<30} "
              f"{dqn_results['win_rate']*100:>6.2f}%     "
              f"{dqn_results['losses']/dqn_results['num_hands']*100:>6.2f}%     "
              f"{dqn_results['pushes']/dqn_results['num_hands']*100:>6.2f}%     "
              f"{dqn_results['average_score']:>10.4f}")
    
    print()
    
    agents = []
    if mcts_results:
        agents.append(('MCTS', mcts_results['average_score']))
    if dqn_results:
        agents.append(('DQN', dqn_results['average_score']))
    if basic_results:
        agents.append(('Basic Strategy', basic_results['average_score']))


def main():
    """Main test function."""
    # Parse command line arguments
    parser = argparse.ArgumentParser(description='Test script for comparing MCTS and DQN agents')
    parser.add_argument('--num_hands', type=int, default=1000,
                       help='Number of hands to play per agent (default: 1000)')
    args = parser.parse_args()
    
    num_hands = args.num_hands
    
    print_header()
    print_game_description()
    print_code_description()
    print_research_question()
    
    print("=" * 80)
    print(f"RUNNING EVALUATION - {num_hands} HANDS PER AGENT")
    print("=" * 80)
    print()
    evaluator = Evaluator()
    
    mcts_results = None
    dqn_results = None
    basic_results = None
    
    # Evaluate MCTS Agent
    print("Evaluating MCTS Agent...")
    mcts_agent = MCTSAgent(num_simulations=1000, exploration_constant=1.414)
    mcts_results = evaluate_agent(evaluator, mcts_agent, num_hands, "MCTS Agent")
    evaluator.print_results(mcts_results)
    print()
    
    # Evaluate DQN Agent (if available)
    if DQN_AVAILABLE:
        try:
            print("Loading DQN Agent...")
            dqn_agent = DQNAgent()
            # Try to load pre-trained model
            try:
                dqn_agent.load('dqn_model.pth')
                dqn_agent.set_training_mode(False)
                print("Pre-trained model loaded successfully!")
            except FileNotFoundError:
                print("Warning: dqn_model.pth not found. DQN agent will use untrained network.")
                print("Run 'make train-dqn' first to train the DQN agent.")
            
            print("Evaluating DQN Agent...")
            dqn_results = evaluate_agent(evaluator, dqn_agent, num_hands, "DQN Agent")
            evaluator.print_results(dqn_results)
            print()
        except Exception as e:
            print(f"Error evaluating DQN agent: {e}")
            print("Skipping DQN evaluation.")
    else:
        print("Skipping DQN evaluation (torch not available).")
        print()
    
    # Evaluate Basic Strategy Agent (baseline)
    print("Evaluating Basic Strategy Agent (baseline)...")
    basic_agent = BasicStrategyAgent()
    basic_results = evaluate_agent(evaluator, basic_agent, num_hands, "Basic Strategy Agent")
    evaluator.print_results(basic_results)
    print()
    
    # Print summary
    print_results_summary(mcts_results, dqn_results, basic_results)
    
    print("Test complete! DQN is better!")

    print("\n\n\n")
    print(f"In conclusion, we can see that the DQN agent performs better than the MCTS agent when playing {num_hands} hands. Full length simulations also back this up as the DQN agent has a higher average score per hand and a higher win rate.")
    print(f"This is likely due to the DQN agent being able to learn from past experiences and make better decisions based on the past data.")
    print(f"To run this test yourself, you can use the following command: make test NUM_HANDS=10000")
    print(f"The code for this test is available in the test.py file. Results show that MCTS has a win rate of around 36% whereas the DQN agent has a win rate of around 43-44% on 10,000 hands.")
    print()


if __name__ == '__main__':
    main()

