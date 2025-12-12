"""
Blackjack game environment with infinite deck.
Rules: No doubling, no splitting, aces automatically treated as 1 or 11.
"""

from enum import Enum
from typing import List, Tuple, Optional
import random


class Action(Enum):
    """Available actions in blackjack."""
    HIT = 0
    STAND = 1


class GameState:
    """Represents the state of a blackjack game."""
    
    def __init__(self, player_hand: List[int], dealer_hand: List[int], 
                 player_turn: bool = True, terminal: bool = False):
        self.player_hand = player_hand.copy()
        self.dealer_hand = dealer_hand.copy()
        self.player_turn = player_turn
        self.terminal = terminal
    
    def copy(self):
        """Create a deep copy of the game state."""
        return GameState(
            self.player_hand,
            self.dealer_hand,
            self.player_turn,
            self.terminal
        )
    
    def __eq__(self, other):
        if not isinstance(other, GameState):
            return False
        return (self.player_hand == other.player_hand and
                self.dealer_hand == other.dealer_hand and
                self.player_turn == other.player_turn and
                self.terminal == other.terminal)
    
    def __hash__(self):
        return hash((tuple(self.player_hand), tuple(self.dealer_hand), 
                     self.player_turn, self.terminal))
    
    def __repr__(self):
        return f"GameState(player={self.player_hand}, dealer={self.dealer_hand}, " \
               f"turn={'player' if self.player_turn else 'dealer'}, terminal={self.terminal})"


class Blackjack:
    """Blackjack game environment with infinite deck."""
    
    def __init__(self):
        self.reset()
    
    def reset(self) -> GameState:
        """Reset the game to initial state."""
        # Deal initial cards: player gets 2, dealer gets 2 (one hidden)
        self.state = GameState(
            player_hand=[self._draw_card(), self._draw_card()],
            dealer_hand=[self._draw_card(), self._draw_card()],
            player_turn=True,
            terminal=False
        )
        return self.state
    
    def _draw_card(self) -> int:
        """Draw a card from infinite deck (1-13, where 1=Ace, 11=Jack, 12=Queen, 13=King)."""
        return random.randint(1, 13)
    
    def _get_card_value(self, card: int) -> int:
        """Get the value of a card (Ace=1, face cards=10)."""
        if card == 1:  # Ace
            return 1  # Will be handled automatically as 1 or 11
        elif card >= 11:  # Face cards
            return 10
        else:
            return card
    
    def _calculate_hand_value(self, hand: List[int]) -> int:
        """Calculate the best possible value of a hand (Aces automatically 1 or 11)."""
        total = 0
        aces = 0
        
        for card in hand:
            if card == 1:  # Ace
                aces += 1
            else:
                total += self._get_card_value(card)
        
        # Add aces optimally
        for _ in range(aces):
            if total + 11 <= 21:
                total += 11
            else:
                total += 1
        
        return total
    
    def _is_bust(self, hand: List[int]) -> bool:
        """Check if a hand is bust (over 21)."""
        return self._calculate_hand_value(hand) > 21
    
    def _is_blackjack(self, hand: List[int]) -> bool:
        """Check if a hand is a blackjack (21 with exactly 2 cards)."""
        return len(hand) == 2 and self._calculate_hand_value(hand) == 21
    
    def step(self, action: Action) -> Tuple[GameState, Optional[float], bool]:
        """
        Execute an action and return (new_state, reward, done).
        Reward is None during the game, only set at terminal state.
        """
        if self.state.terminal:
            return self.state, 0.0, True
        
        if action == Action.HIT:
            if not self.state.player_turn:
                raise ValueError("Cannot hit on dealer's turn")
            
            # Player hits
            self.state.player_hand.append(self._draw_card())
            
            # Check if player busts
            if self._is_bust(self.state.player_hand):
                self.state.terminal = True
                return self.state, -1.0, True  # Player loses
            
            # Player can continue
            return self.state, None, False
        
        elif action == Action.STAND:
            if not self.state.player_turn:
                raise ValueError("Cannot stand on dealer's turn")
            
            # Player stands, dealer's turn
            self.state.player_turn = False
            
            # Dealer plays: must hit until 17 or higher
            while self._calculate_hand_value(self.state.dealer_hand) < 17:
                self.state.dealer_hand.append(self._draw_card())
                if self._is_bust(self.state.dealer_hand):
                    self.state.terminal = True
                    return self.state, 1.0, True  # Dealer busts, player wins
            
            # Both hands are complete, determine winner
            self.state.terminal = True
            player_value = self._calculate_hand_value(self.state.player_hand)
            dealer_value = self._calculate_hand_value(self.state.dealer_hand)
            
            # Check for blackjacks
            player_bj = self._is_blackjack(self.state.player_hand)
            dealer_bj = self._is_blackjack(self.state.dealer_hand)
            
            if player_bj and not dealer_bj:
                return self.state, 1.0, True  # Player blackjack wins
            elif dealer_bj and not player_bj:
                return self.state, -1.0, True  # Dealer blackjack wins
            elif player_bj and dealer_bj:
                return self.state, 0.0, True  # Both blackjack, push
            
            # Compare values
            if player_value > dealer_value:
                return self.state, 1.0, True  # Player wins
            elif dealer_value > player_value:
                return self.state, -1.0, True  # Dealer wins
            else:
                return self.state, 0.0, True  # Push
    
    def get_legal_actions(self) -> List[Action]:
        """Get legal actions for current state."""
        if self.state.terminal:
            return []
        if not self.state.player_turn:
            return []  # Dealer plays automatically
        return [Action.HIT, Action.STAND]
    
    def get_state(self) -> GameState:
        """Get current game state."""
        return self.state.copy()
    
    def get_observable_state(self) -> Tuple[Tuple[int, ...], int]:
        """
        Get observable state for agents.
        Returns: (player_hand_tuple, dealer_visible_card)
        """
        # Only first dealer card is visible to player
        dealer_visible = self.state.dealer_hand[0] if self.state.dealer_hand else 0
        return (tuple(self.state.player_hand), dealer_visible)

