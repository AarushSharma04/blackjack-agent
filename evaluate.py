"""
Evaluation framework for comparing blackjack agents.
Scoring: +1 for wins, -1 for losses, 0 for pushes/ties.
"""

from typing import List, Dict
from blackjack import Blackjack, Action
from mcts_agent import MCTSAgent


class Evaluator:
    """Evaluates agent performance on blackjack."""
    
    def __init__(self):
        self.results: List[Dict] = []
    
    def evaluate_agent(self, agent, num_hands: int, agent_name: str = "Agent") -> Dict:
        """
        Evaluate an agent by playing a specified number of hands.
        
        Args:
            agent: The agent to evaluate (must have get_action method)
            num_hands: Number of hands to play
            agent_name: Name of the agent for reporting
        
        Returns:
            Dictionary with evaluation statistics
        """
        game = Blackjack()
        wins = 0
        losses = 0
        pushes = 0
        total_score = 0
        
        for hand_num in range(num_hands):
            # Reset game and agent
            state = game.reset()
            if hasattr(agent, 'reset'):
                agent.reset()
            
            # Play one hand
            reward = None
            while not game.state.terminal:
                if game.state.player_turn:
                    action = agent.get_action(game)
                    state, reward, done = game.step(action)
                    if done:
                        break
                else:
                    # Should not happen - dealer plays automatically in step()
                    break
            
            # Update statistics
            if reward is not None:
                total_score += reward
                if reward > 0:
                    wins += 1
                elif reward < 0:
                    losses += 1
                else:
                    pushes += 1
        
        # Calculate statistics
        win_rate = wins / num_hands if num_hands > 0 else 0.0
        average_score = total_score / num_hands if num_hands > 0 else 0.0
        
        results = {
            'agent_name': agent_name,
            'num_hands': num_hands,
            'wins': wins,
            'losses': losses,
            'pushes': pushes,
            'win_rate': win_rate,
            'total_score': total_score,
            'average_score': average_score
        }
        
        self.results.append(results)
        return results
    
    def print_results(self, results: Dict):
        """Print evaluation results in a readable format."""
        print(f"\n{'='*60}")
        print(f"Evaluation Results: {results['agent_name']}")
        print(f"{'='*60}")
        print(f"Number of hands: {results['num_hands']}")
        print(f"Wins: {results['wins']} ({results['win_rate']*100:.2f}%)")
        print(f"Losses: {results['losses']}")
        print(f"Pushes: {results['pushes']}")
        print(f"Total Score: {results['total_score']:.1f}")
        print(f"Average Score per Hand: {results['average_score']:.4f}")
        print(f"{'='*60}\n")
    
    def compare_agents(self, results1: Dict, results2: Dict):
        """Compare results from two agents."""
        print(f"\n{'='*60}")
        print("Agent Comparison")
        print(f"{'='*60}")
        print(f"{results1['agent_name']:30s} vs {results2['agent_name']:30s}")
        print(f"{'='*60}")
        print(f"Win Rate:     {results1['win_rate']*100:6.2f}%  vs  {results2['win_rate']*100:6.2f}%")
        print(f"Average Score: {results1['average_score']:6.4f}  vs  {results2['average_score']:6.4f}")
        print(f"Total Score:   {results1['total_score']:6.1f}  vs  {results2['total_score']:6.1f}")
        
        if results1['average_score'] > results2['average_score']:
            print(f"\nWinner: {results1['agent_name']} (by {results1['average_score'] - results2['average_score']:.4f} points per hand)")
        elif results2['average_score'] > results1['average_score']:
            print(f"\nWinner: {results2['agent_name']} (by {results2['average_score'] - results1['average_score']:.4f} points per hand)")
        else:
            print(f"\nTie!")
        print(f"{'='*60}\n")

