.PHONY: help install test clean train-dqn

help:
	@echo "Blackjack Agent Project - Makefile Commands"
	@echo "============================================"
	@echo "make install         - Install required dependencies"
	@echo "make test            - Run test script (default: 1000 hands per agent)"
	@echo "make test NUM_HANDS=N - Run test with N hands (e.g., make test NUM_HANDS=10000)"
	@echo "make train-dqn       - Train the DQN agent (saves to dqn_model.pth)"
	@echo "make clean           - Remove generated files (pyc, __pycache__)"
	@echo ""

install:
	@echo "Installing dependencies..."
	pip3 install -r requirements.txt
	@echo "Installation complete!"

test:
	@if [ -z "$(NUM_HANDS)" ]; then \
		python3 test.py; \
	else \
		python3 test.py --num_hands $(NUM_HANDS); \
	fi

train-dqn:
	@echo "Training DQN agent..."
	python3 -c "from dqn_agent import DQNAgent, train_dqn; agent = DQNAgent(); train_dqn(agent, num_episodes=10000, save_path='dqn_model.pth')"

clean:
	@echo "Cleaning generated files..."
	find . -type d -name "__pycache__" -exec rm -r {} + 2>/dev/null || true
	find . -type f -name "*.pyc" -delete 2>/dev/null || true
	@echo "Clean complete!"

