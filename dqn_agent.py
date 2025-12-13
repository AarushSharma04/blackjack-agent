"""
Deep Q-Network (DQN) agent for blackjack.
Uses neural networks to learn optimal Q-values through experience.
"""

import random
import numpy as np
from typing import List, Tuple, Optional, Deque
from collections import deque
import torch
import torch.nn as nn
import torch.optim as optim
from blackjack import Blackjack, Action, GameState


class DQN(nn.Module):
    """Neural network for approximating Q-values."""
    
    def __init__(self, input_size: int = 3, hidden_sizes: List[int] = [64, 64], output_size: int = 2):
        """
        Initialize DQN network.
        
        Args:
            input_size: Size of state representation (default: 3 for player_value, dealer_visible, is_soft)
            hidden_sizes: List of hidden layer sizes
            output_size: Number of actions (default: 2 for HIT, STAND)
        """
        super(DQN, self).__init__()
        
        layers = []
        prev_size = input_size
        
        for hidden_size in hidden_sizes:
            layers.append(nn.Linear(prev_size, hidden_size))
            layers.append(nn.ReLU())
            prev_size = hidden_size
        
        layers.append(nn.Linear(prev_size, output_size))
        self.network = nn.Sequential(*layers)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass through the network."""
        return self.network(x)


def state_to_features(state: GameState) -> np.ndarray:
    """
    Convert game state to feature vector for neural network.
    
    Returns: [player_value, dealer_visible_value, is_soft (0 or 1)]
    """
    if state.terminal:
        return np.array([0, 0, 0], dtype=np.float32)
    
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
    is_soft = 0
    for _ in range(player_aces):
        if player_value + 11 <= 21:
            player_value += 11
            is_soft = 1
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
    
    # Normalize values for better training
    # Player value: 4-21 (clamp to reasonable range)
    player_value = max(4, min(21, player_value))
    # Dealer visible: 2-11
    dealer_visible = max(2, min(11, dealer_visible))
    
    return np.array([player_value, dealer_visible, float(is_soft)], dtype=np.float32)


class ReplayBuffer:
    """Experience replay buffer for DQN training."""
    
    def __init__(self, capacity: int = 10000):
        self.buffer: Deque[Tuple] = deque(maxlen=capacity)
    
    def push(self, state: np.ndarray, action: int, reward: float, 
             next_state: np.ndarray, done: bool):
        """Add experience to buffer."""
        self.buffer.append((state, action, reward, next_state, done))
    
    def sample(self, batch_size: int) -> List[Tuple]:
        """Sample a batch of experiences."""
        return random.sample(self.buffer, batch_size)
    
    def __len__(self) -> int:
        return len(self.buffer)


class DQNAgent:
    """Deep Q-Network agent for blackjack."""
    
    def __init__(self, 
                 learning_rate: float = 0.001,
                 gamma: float = 0.99,
                 epsilon_start: float = 1.0,
                 epsilon_end: float = 0.01,
                 epsilon_decay: float = 0.995,
                 memory_size: int = 10000,
                 batch_size: int = 32,
                 target_update_freq: int = 100,
                 hidden_sizes: List[int] = [64, 64],
                 device: Optional[str] = None):
        """
        Initialize DQN agent.
        
        Args:
            learning_rate: Learning rate for optimizer
            gamma: Discount factor
            epsilon_start: Starting epsilon for epsilon-greedy
            epsilon_end: Minimum epsilon
            epsilon_decay: Epsilon decay rate per episode
            memory_size: Size of replay buffer
            batch_size: Batch size for training
            target_update_freq: Frequency to update target network
            hidden_sizes: Hidden layer sizes for network
            device: 'cuda' or 'cpu' (None for auto-detect)
        """
        self.gamma = gamma
        self.epsilon = epsilon_start
        self.epsilon_start = epsilon_start
        self.epsilon_end = epsilon_end
        self.epsilon_decay = epsilon_decay
        self.batch_size = batch_size
        self.target_update_freq = target_update_freq
        self.update_counter = 0
        
        # Set device
        if device is None:
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        else:
            self.device = torch.device(device)
        
        # Networks
        self.q_network = DQN(input_size=3, hidden_sizes=hidden_sizes, output_size=2).to(self.device)
        self.target_network = DQN(input_size=3, hidden_sizes=hidden_sizes, output_size=2).to(self.device)
        self.target_network.load_state_dict(self.q_network.state_dict())
        self.target_network.eval()
        
        # Optimizer
        self.optimizer = optim.Adam(self.q_network.parameters(), lr=learning_rate)
        
        # Replay buffer
        self.memory = ReplayBuffer(capacity=memory_size)
        
        # Training mode
        self.training_mode = True
    
    def get_state_features(self, game: Blackjack) -> np.ndarray:
        """Get state features from game."""
        state = game.get_state()
        return state_to_features(state)
    
    def select_action(self, state_features: np.ndarray, training: bool = True) -> Action:
        """
        Select action using epsilon-greedy policy.
        
        Args:
            state_features: State feature vector
            training: If True, use epsilon-greedy. If False, always exploit.
        """
        if training and self.training_mode and random.random() < self.epsilon:
            # Explore: random action
            return random.choice([Action.HIT, Action.STAND])
        
        # Exploit: use Q-network
        with torch.no_grad():
            state_tensor = torch.FloatTensor(state_features).unsqueeze(0).to(self.device)
            q_values = self.q_network(state_tensor)
            action_idx = q_values.argmax().item()
            return Action.HIT if action_idx == 0 else Action.STAND
    
    def get_action(self, game: Blackjack) -> Action:
        """Get action for current game state (for evaluation interface)."""
        state_features = self.get_state_features(game)
        return self.select_action(state_features, training=False)
    
    def store_transition(self, state: np.ndarray, action: Action, reward: float,
                        next_state: np.ndarray, done: bool):
        """Store transition in replay buffer."""
        action_idx = 0 if action == Action.HIT else 1
        self.memory.push(state, action_idx, reward, next_state, done)
    
    def train_step(self) -> Optional[float]:
        """
        Perform one training step on a batch from replay buffer.
        Returns loss if training occurred, None otherwise.
        """
        if len(self.memory) < self.batch_size:
            return None
        
        # Sample batch
        batch = self.memory.sample(self.batch_size)
        states, actions, rewards, next_states, dones = zip(*batch)
        
        # Convert to tensors
        states = torch.FloatTensor(np.array(states)).to(self.device)
        actions = torch.LongTensor(actions).to(self.device)
        rewards = torch.FloatTensor(rewards).to(self.device)
        next_states = torch.FloatTensor(np.array(next_states)).to(self.device)
        dones = torch.BoolTensor(dones).to(self.device)
        
        # Current Q values
        current_q_values = self.q_network(states).gather(1, actions.unsqueeze(1))
        
        # Next Q values from target network
        with torch.no_grad():
            next_q_values = self.target_network(next_states).max(1)[0]
            target_q_values = rewards + (self.gamma * next_q_values * ~dones)
        
        # Compute loss
        loss = nn.MSELoss()(current_q_values.squeeze(), target_q_values)
        
        # Optimize
        self.optimizer.zero_grad()
        loss.backward()
        # Gradient clipping for stability
        torch.nn.utils.clip_grad_norm_(self.q_network.parameters(), 1.0)
        self.optimizer.step()
        
        # Update target network periodically
        self.update_counter += 1
        if self.update_counter % self.target_update_freq == 0:
            self.target_network.load_state_dict(self.q_network.state_dict())
        
        return loss.item()
    
    def decay_epsilon(self):
        """Decay epsilon for epsilon-greedy exploration."""
        self.epsilon = max(self.epsilon_end, self.epsilon * self.epsilon_decay)
    
    def set_training_mode(self, training: bool):
        """Set training mode (affects epsilon-greedy in get_action)."""
        self.training_mode = training
    
    def reset(self):
        """Reset agent (no-op for DQN, but needed for interface)."""
        pass
    
    def save(self, filepath: str):
        """Save model to file."""
        torch.save({
            'q_network': self.q_network.state_dict(),
            'target_network': self.target_network.state_dict(),
            'optimizer': self.optimizer.state_dict(),
            'epsilon': self.epsilon,
        }, filepath)
    
    def load(self, filepath: str):
        """Load model from file."""
        checkpoint = torch.load(filepath, map_location=self.device)
        self.q_network.load_state_dict(checkpoint['q_network'])
        self.target_network.load_state_dict(checkpoint['target_network'])
        self.optimizer.load_state_dict(checkpoint['optimizer'])
        self.epsilon = checkpoint['epsilon']


def train_dqn(agent: DQNAgent, num_episodes: int = 10000, 
               print_freq: int = 1000, save_path: Optional[str] = None):
    """
    Train DQN agent by playing games and learning from experience.
    
    Args:
        agent: DQN agent to train
        num_episodes: Number of training episodes
        print_freq: Frequency to print training stats
        save_path: Path to save trained model (optional)
    """
    agent.set_training_mode(True)
    game = Blackjack()
    
    episode_rewards = []
    episode_losses = []
    
    print(f"\nTraining DQN agent for {num_episodes} episodes...")
    print(f"Starting epsilon: {agent.epsilon:.3f}")
    
    for episode in range(num_episodes):
        # Reset game
        state = game.reset()
        state_features = state_to_features(state)
        
        episode_reward = 0
        episode_loss = 0
        loss_count = 0
        
        # Play one episode
        while not game.state.terminal:
            # Select action
            action = agent.select_action(state_features, training=True)
            
            # Take step
            next_state, reward, done = game.step(action)
            next_state_features = state_to_features(next_state)
            
            # Store transition (use 0.0 for intermediate rewards, actual reward at terminal)
            reward_value = reward if reward is not None else 0.0
            agent.store_transition(state_features, action, reward_value, 
                                  next_state_features, done)
            
            # Train
            loss = agent.train_step()
            if loss is not None:
                episode_loss += loss
                loss_count += 1
            
            # Update state
            state_features = next_state_features
            episode_reward += reward if reward is not None else 0
        
        episode_rewards.append(episode_reward)
        if loss_count > 0:
            episode_losses.append(episode_loss / loss_count)
        
        # Decay epsilon
        agent.decay_epsilon()
        
        # Print progress
        if (episode + 1) % print_freq == 0:
            avg_reward = np.mean(episode_rewards[-print_freq:])
            avg_loss = np.mean(episode_losses[-print_freq:]) if episode_losses else 0
            print(f"Episode {episode + 1}/{num_episodes} | "
                  f"Avg Reward: {avg_reward:.4f} | "
                  f"Avg Loss: {avg_loss:.4f} | "
                  f"Epsilon: {agent.epsilon:.3f} | "
                  f"Buffer Size: {len(agent.memory)}")
    
    # Save model if path provided
    if save_path:
        agent.save(save_path)
        print(f"\nModel saved to {save_path}")
    
    agent.set_training_mode(False)
    print("\nTraining complete!")
    
    return episode_rewards, episode_losses

