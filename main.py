"""
Main simulation script for evaluating blackjack agents.
"""

import argparse
from evaluate import Evaluator
from mcts_agent import MCTSAgent
from basic_strategy_agent import BasicStrategyAgent
from dqn_agent import DQNAgent, train_dqn


def main():
    parser = argparse.ArgumentParser(description='Evaluate blackjack agents')
    parser.add_argument('--num_hands', type=int, default=1000,
                       help='Number of hands to simulate (default: 1000)')
    parser.add_argument('--mcts_simulations', type=int, default=1000,
                       help='Number of MCTS simulations per move (default: 1000)')
    parser.add_argument('--agent', type=str, 
                       choices=['mcts', 'basic', 'dqn', 'both', 'all'], 
                       default='mcts',
                       help='Which agent(s) to evaluate (default: mcts)')
    parser.add_argument('--train_dqn', action='store_true',
                       help='Train DQN agent before evaluation')
    parser.add_argument('--dqn_episodes', type=int, default=10000,
                       help='Number of episodes to train DQN (default: 10000)')
    parser.add_argument('--dqn_model_path', type=str, default=None,
                       help='Path to load/save DQN model')
    
    args = parser.parse_args()
    
    evaluator = Evaluator()
    mcts_results = None
    basic_results = None
    dqn_results = None
    
    # Train or load DQN if needed
    dqn_agent = None
    if args.agent in ['dqn', 'both', 'all']:
        dqn_agent = DQNAgent()
        
        if args.train_dqn:
            # Train DQN
            train_dqn(dqn_agent, num_episodes=args.dqn_episodes, 
                     save_path=args.dqn_model_path)
        elif args.dqn_model_path:
            # Load pre-trained model
            print(f"\nLoading DQN model from {args.dqn_model_path}...")
            dqn_agent.load(args.dqn_model_path)
            dqn_agent.set_training_mode(False)
            print("Model loaded!")
    
    # Evaluate MCTS
    if args.agent in ['mcts', 'both', 'all']:
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
    
    # Evaluate Basic Strategy
    if args.agent in ['basic', 'both', 'all']:
        print(f"\nEvaluating Basic Strategy Agent with {args.num_hands} hands...")
        
        basic_agent = BasicStrategyAgent()
        
        basic_results = evaluator.evaluate_agent(
            basic_agent,
            num_hands=args.num_hands,
            agent_name="Basic Strategy Agent"
        )
        
        evaluator.print_results(basic_results)
    
    # Evaluate DQN
    if args.agent in ['dqn', 'both', 'all'] and dqn_agent is not None:
        print(f"\nEvaluating DQN Agent with {args.num_hands} hands...")
        
        dqn_results = evaluator.evaluate_agent(
            dqn_agent,
            num_hands=args.num_hands,
            agent_name="DQN Agent"
        )
        
        evaluator.print_results(dqn_results)
    
    # Compare agents
    if args.agent == 'both' and mcts_results and basic_results:
        evaluator.compare_agents(mcts_results, basic_results, show_regret=True)
    elif args.agent == 'all':
        # Compare all three agents
        if basic_results and mcts_results:
            evaluator.compare_agents(mcts_results, basic_results, show_regret=True)
        if basic_results and dqn_results:
            evaluator.compare_agents(dqn_results, basic_results, show_regret=True)
        if mcts_results and dqn_results:
            evaluator.compare_agents(mcts_results, dqn_results, show_regret=False)
    
    print("\nSimulation complete!")


if __name__ == '__main__':
    main()

