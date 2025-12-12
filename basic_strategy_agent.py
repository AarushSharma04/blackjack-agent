"""
Basic strategy agent for blackjack (for comparison).
Uses optimal basic strategy rules.
"""

from blackjack import Blackjack, Action


class BasicStrategyAgent:
    """Basic strategy agent using optimal blackjack strategy."""
    
    def _get_hand_value(self, hand):
        """Calculate hand value."""
        total = 0
        aces = 0
        for card in hand:
            if card == 1:
                aces += 1
            elif card >= 11:
                total += 10
            else:
                total += card
        is_soft = False
        for _ in range(aces):
            if total + 11 <= 21:
                total += 11
                is_soft = True
            else:
                total += 1
        return total, is_soft
    
    def get_action(self, game: Blackjack) -> Action:
        """Get action using basic strategy."""
        player_value, is_soft = self._get_hand_value(game.state.player_hand)
        dealer_visible = game.state.dealer_hand[0] if game.state.dealer_hand else 0
        dealer_value = 11 if dealer_visible == 1 else (10 if dealer_visible >= 11 else dealer_visible)
        
        # Always stand on 21
        if player_value >= 21:
            return Action.STAND
        
        # Always stand on 17 or higher
        if player_value >= 17:
            return Action.STAND
        
        # Soft hands (ace counted as 11)
        if is_soft:
            if player_value >= 19:
                return Action.STAND
            if player_value == 18:
                # Stand on 18 vs dealer 2-8, hit vs 9-10-A
                if dealer_value <= 8:
                    return Action.STAND
                else:
                    return Action.HIT
            # Always hit soft 17 or lower
            return Action.HIT
        
        # Hard hands
        if player_value >= 17:
            return Action.STAND
        if player_value >= 13:
            # Stand on 13-16 vs dealer 2-6, hit vs 7+
            if dealer_value <= 6:
                return Action.STAND
            else:
                return Action.HIT
        if player_value == 12:
            # Stand on 12 vs dealer 4-6, hit otherwise
            if 4 <= dealer_value <= 6:
                return Action.STAND
            else:
                return Action.HIT
        
        # Always hit 11 or lower
        return Action.HIT
    
    def reset(self):
        """Reset agent (no-op for basic strategy)."""
        pass

