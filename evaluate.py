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
    
    def calculate_regret(self, agent_results: Dict, optimal_results: Dict) -> Dict:
        """
        Calculate regret: the difference between optimal and agent performance.
        
        Regret measures how much worse the agent performs compared to the optimal baseline.
        Lower regret is better (0 regret means the agent matches optimal performance).
        
        Args:
            agent_results: Results from the agent being evaluated
            optimal_results: Results from the optimal baseline (e.g., basic strategy)
        
        Returns:
            Dictionary with regret metrics
        """
        regret_per_hand = optimal_results['average_score'] - agent_results['average_score']
        total_regret = optimal_results['total_score'] - agent_results['total_score']
        regret_percentage = (regret_per_hand / abs(optimal_results['average_score']) * 100) if optimal_results['average_score'] != 0 else 0.0
        
        return {
            'regret_per_hand': regret_per_hand,
            'total_regret': total_regret,
            'regret_percentage': regret_percentage,
            'optimal_score': optimal_results['average_score'],
            'agent_score': agent_results['average_score']
        }
    
    def print_regret(self, agent_results: Dict, optimal_results: Dict):
        """
        Print regret analysis comparing an agent to the optimal baseline.
        
        Args:
            agent_results: Results from the agent being evaluated
            optimal_results: Results from the optimal baseline
        """
        regret = self.calculate_regret(agent_results, optimal_results)
        print(f"\n{'='*60}")
        print("Regret Analysis (vs Optimal Baseline)")
        print(f"{'='*60}")
        print(f"Optimal Score ({optimal_results['agent_name']}): {regret['optimal_score']:.4f}")
        print(f"Agent Score ({agent_results['agent_name']}): {regret['agent_score']:.4f}")
        print(f"Regret per Hand: {regret['regret_per_hand']:.4f}")
        print(f"Total Regret: {regret['total_regret']:.1f}")
        print(f"Regret Percentage: {regret['regret_percentage']:.2f}%")
        if regret['regret_per_hand'] < 0:
            print(f"Note: Negative regret means agent outperformed baseline!")
        print(f"{'='*60}\n")
    
    def compare_agents(self, results1: Dict, results2: Dict, show_regret: bool = False):
        """
        Compare results from two agents.
        
        Args:
            results1: First agent's results
            results2: Second agent's results
            show_regret: If True and one agent is "Basic Strategy Agent", show regret
        """
        print(f"\n{'='*60}")
        print("Agent Comparison")
        print(f"{'='*60}")
        print(f"{results1['agent_name']:30s} vs {results2['agent_name']:30s}")
        print(f"{'='*60}")
        print(f"Win Rate:     {results1['win_rate']*100:6.2f}%  vs  {results2['win_rate']*100:6.2f}%")
        print(f"Average Score: {results1['average_score']:6.4f}  vs  {results2['average_score']:6.4f}")
        print(f"Total Score:   {results1['total_score']:6.1f}  vs  {results2['total_score']:6.1f}")
        
        # Calculate and show regret if requested
        if show_regret:
            if "Basic Strategy" in results1['agent_name']:
                self.print_regret(results2, results1)
            elif "Basic Strategy" in results2['agent_name']:
                self.print_regret(results1, results2)
        
        if results1['average_score'] > results2['average_score']:
            print(f"\nWinner: {results1['agent_name']} (by {results1['average_score'] - results2['average_score']:.4f} points per hand)")
        elif results2['average_score'] > results1['average_score']:
            print(f"\nWinner: {results2['agent_name']} (by {results2['average_score'] - results1['average_score']:.4f} points per hand)")
        else:
            print(f"\nTie!")
        print(f"{'='*60}\n")

