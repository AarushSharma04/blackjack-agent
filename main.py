"""
Main simulation script for evaluating blackjack agents.
"""

import argparse
from evaluate import Evaluator
from mcts_agent import MCTSAgent
from basic_strategy_agent import BasicStrategyAgent


def main():
    parser = argparse.ArgumentParser(description='Evaluate blackjack agents')
    parser.add_argument('--num_hands', type=int, default=1000,
                       help='Number of hands to simulate (default: 1000)')
    parser.add_argument('--mcts_simulations', type=int, default=1000,
                       help='Number of MCTS simulations per move (default: 1000)')
    parser.add_argument('--agent', type=str, choices=['mcts', 'basic', 'both'], default='mcts',
                       help='Which agent(s) to evaluate (default: mcts)')
    
    args = parser.parse_args()
    
    evaluator = Evaluator()
    
    if args.agent in ['mcts', 'both']:
        print(f"\nEvaluating MCTS Agent with {args.num_hands} hands...")
        print(f"MCTS simulations per move: {args.mcts_simulations}")
        
        mcts_agent = MCTSAgent(
            num_simulations=args.mcts_simulations,
            exploration_constant=1.414
        )
        
        mcts_results = evaluator.evaluate_agent(
            mcts_agent,
            num_hands=args.num_hands,
            agent_name="MCTS Agent"
        )
        
        evaluator.print_results(mcts_results)
    
    if args.agent in ['basic', 'both']:
        print(f"\nEvaluating Basic Strategy Agent with {args.num_hands} hands...")
        
        basic_agent = BasicStrategyAgent()
        
        basic_results = evaluator.evaluate_agent(
            basic_agent,
            num_hands=args.num_hands,
            agent_name="Basic Strategy Agent"
        )
        
        evaluator.print_results(basic_results)
        
        if args.agent == 'both':
            evaluator.compare_agents(mcts_results, basic_results)
    
    # Q-learning agent will be added later
    
    print("\nSimulation complete!")


if __name__ == '__main__':
    main()

