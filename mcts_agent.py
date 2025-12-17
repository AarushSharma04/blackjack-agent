import math
import random
from typing import Dict, List, Optional, Tuple
from blackjack import Blackjack, Action, GameState


def abstract_state(state: GameState) -> Tuple[int, int, bool, bool]:
    if state.terminal:
        return (0, 0, False, True)
    
    # Calculate player hand value
    player_total = 0
    player_aces = 0
    for card in state.player_hand:
        if card == 1:
            player_aces += 1
        elif card >= 11:
            player_total += 10
        else:
            player_total += card
    
    player_value = player_total
    is_soft = False
    for _ in range(player_aces):
        if player_value + 11 <= 21:
            player_value += 11
            is_soft = True
        else:
            player_value += 1
    
    # Get dealer's visible card value
    dealer_visible = 0
    if state.dealer_hand:
        dealer_card = state.dealer_hand[0]
        if dealer_card == 1:
            dealer_visible = 11  # Ace shown as 11
        elif dealer_card >= 11:
            dealer_visible = 10
        else:
            dealer_visible = dealer_card
    
    return (player_value, dealer_visible, is_soft, state.terminal)


class MCTSNode:
    def __init__(self, state: GameState, parent: Optional['MCTSNode'] = None, 
                 action: Optional[Action] = None):
        self.state = state
        self.abstract_state = abstract_state(state)
        self.parent = parent
        self.action = action
        self.children: Dict[Action, 'MCTSNode'] = {}
        self.visits = 0
        self.total_reward = 0.0
        self.untried_actions = self._get_legal_actions()
    
    def _get_legal_actions(self) -> List[Action]:
        """Get legal actions from this state."""
        if self.state.terminal:
            return []
        if not self.state.player_turn:
            return []  # Dealer plays automatically
        return [Action.HIT, Action.STAND]
    
    def is_fully_expanded(self) -> bool:
        """Check if all actions have been tried."""
        return len(self.untried_actions) == 0
    
    def is_terminal(self) -> bool:
        """Check if this is a terminal node."""
        return self.state.terminal
    
    def ucb1_value(self, exploration_constant: float = 1.414) -> float:
        """Calculate UCB1 value for this node."""
        if self.visits == 0:
            return float('inf')
        
        if self.parent is None:
            return self.total_reward / self.visits
        
        exploitation = self.total_reward / self.visits
        exploration = exploration_constant * math.sqrt(
            math.log(self.parent.visits) / self.visits
        )
        return exploitation + exploration
    
    def best_child(self, exploration_constant: float = 1.414) -> 'MCTSNode':
        # Get the best child according to UCB1
        return max(self.children.values(), 
                  key=lambda child: child.ucb1_value(exploration_constant))
    
    def expand(self) -> 'MCTSNode':
        # expand node
        if not self.untried_actions:
            return self
        
        # chooseuntried action
        action = self.untried_actions.pop()
        
        temp_game = Blackjack()
        temp_game.state = self.state.copy()
        new_state, reward, done = temp_game.step(action)
        
        child = MCTSNode(new_state, parent=self, action=action)
        self.children[action] = child
        
        return child
    
    def backpropagate(self, reward: float):
        self.visits += 1
        self.total_reward += reward
        
        if self.parent:
            self.parent.backpropagate(reward)


class MCTSAgent:    
    def __init__(self, num_simulations: int = 1000, exploration_constant: float = 1.414):
        self.num_simulations = num_simulations
        self.exploration_constant = exploration_constant
        self.root: Optional[MCTSNode] = None
    
    def select(self, node: MCTSNode) -> MCTSNode:
        while not node.is_terminal() and node.is_fully_expanded():
            if not node.children:
                break
            node = node.best_child(self.exploration_constant)
        return node
    
    def _get_hand_value(self, hand: List[int]) -> int:
        # hand value
        total = 0
        aces = 0
        for card in hand:
            if card == 1:  # Ace
                aces += 1
            elif card >= 11:  # Face cards
                total += 10
            else:
                total += card

        # handle aces optimally
        for _ in range(aces):
            if total + 11 <= 21:
                total += 11
            else:
                total += 1
        return total
    
    def _smart_simulation_action(self, player_hand: List[int], dealer_visible: int) -> Action:
        player_value = self._get_hand_value(player_hand)
        
        # Always stand on 21
        if player_value >= 21:
            return Action.STAND
        
        # Always stand on 17 or higher (basic strategy)
        if player_value >= 17:
            return Action.STAND
        
        dealer_value = self._get_hand_value([dealer_visible])
        
        if player_value >= 12:
            if dealer_value <= 6:
                return Action.STAND
            else:
                return Action.HIT
        
        return Action.HIT
    
    def simulate(self, game: Blackjack, state: GameState) -> float:
        sim_game = Blackjack()
        sim_game.state = state.copy()
        
        while not sim_game.state.terminal:
            if sim_game.state.player_turn:
                # player turn.. use smart policy
                legal_actions = sim_game.get_legal_actions()
                if not legal_actions:
                    break
                
                # shown card
                dealer_visible = sim_game.state.dealer_hand[0] if sim_game.state.dealer_hand else 0
                action = self._smart_simulation_action(sim_game.state.player_hand, dealer_visible)
                
                new_state, reward, done = sim_game.step(action)
                if done:
                    return reward if reward is not None else 0.0
            else:
                # dealer turn to hit until 17 or bust
                while sim_game._calculate_hand_value(sim_game.state.dealer_hand) < 17:
                    sim_game.state.dealer_hand.append(sim_game._draw_card())
                    if sim_game._is_bust(sim_game.state.dealer_hand):
                        sim_game.state.terminal = True
                        return 1.0
                
                sim_game.state.terminal = True
                player_value = sim_game._calculate_hand_value(sim_game.state.player_hand)
                dealer_value = sim_game._calculate_hand_value(sim_game.state.dealer_hand)
                
                player_bj = sim_game._is_blackjack(sim_game.state.player_hand)
                dealer_bj = sim_game._is_blackjack(sim_game.state.dealer_hand)
                
                if player_bj and not dealer_bj:
                    return 1.0
                elif dealer_bj and not player_bj:
                    return -1.0
                elif player_bj and dealer_bj:
                    return 0.0
                
                if player_value > dealer_value:
                    return 1.0
                elif dealer_value > player_value:
                    return -1.0
                else:
                    return 0.0
        
        return 0.0
    
    def search(self, game: Blackjack):
        """Perform MCTS search from current game state."""
        current_state = game.get_state()
        self.root = MCTSNode(current_state)
        
        for _ in range(self.num_simulations):
            node = self.select(self.root)
            
            if not node.is_terminal() and not node.is_fully_expanded():
                node = node.expand()
            
            # Simulation: smart playout
            reward = self.simulate(Blackjack(), node.state)
            
            # Backpropagation: update statistics
            node.backpropagate(reward)
    
    def get_action(self, game: Blackjack) -> Action:
        """
        Get the best action according to MCTS.
        """
        # Perform MCTS search
        self.search(game)
        
        if not self.root or not self.root.children:
            # Fallback to smart action
            dealer_visible = game.state.dealer_hand[0] if game.state.dealer_hand else 0
            return self._smart_simulation_action(game.state.player_hand, dealer_visible)
        
        # Return action with highest average reward
        best_action = None
        best_avg_reward = float('-inf')
        
        for action, child in self.root.children.items():
            if child.visits > 0:
                avg_reward = child.total_reward / child.visits
                if avg_reward > best_avg_reward:
                    best_avg_reward = avg_reward
                    best_action = action
        
        # Fallback if no good action found (shouldn't happen, but safety)
        if best_action is None:
            dealer_visible = game.state.dealer_hand[0] if game.state.dealer_hand else 0
            return self._smart_simulation_action(game.state.player_hand, dealer_visible)
        
        return best_action
    
    def reset(self):

        self.root = None

