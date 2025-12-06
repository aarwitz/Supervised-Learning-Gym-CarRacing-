import os
import sys
import argparse
import torch
import numpy as np
from collections import deque

from network import ActorCritic
from ppo_agent import PPOAgent
from environment import make_env
from utils import Logger, RewardTracker, save_checkpoint


def train(args):
    """
    Train the PPO agent on CarRacing-v3.
    """
    print("=" * 60)
    print("Training PPO Agent on CarRacing-v3")
    print("=" * 60)
    
    # Set device
    device = torch.device('cuda' if torch.cuda.is_available() and args.cuda else 'cpu')
    print(f"Using device: {device}")
    
    # Create environment
    render_mode = "human" if args.render else None
    env = make_env(frame_stack=4, frame_skip=args.frame_skip, render_mode=render_mode)
    print(f"Environment created with frame_skip={args.frame_skip}")
    
    # Create network and agent
    network = ActorCritic(num_inputs=4, num_actions=3)
    agent = PPOAgent(
        network=network,
        learning_rate=args.lr,
        gamma=args.gamma,
        gae_lambda=args.gae_lambda,
        clip_epsilon=args.clip_epsilon,
        value_coef=args.value_coef,
        entropy_coef=args.entropy_coef,
        max_grad_norm=args.max_grad_norm,
        ppo_epochs=args.ppo_epochs,
        mini_batch_size=args.batch_size,
        device=device
    )
    
    # Load checkpoint if specified
    start_episode = 0
    if args.checkpoint:
        if os.path.exists(args.checkpoint):
            agent.load(args.checkpoint)
            print(f"Loaded checkpoint from {args.checkpoint}")
        else:
            print(f"Checkpoint not found: {args.checkpoint}")
    
    # Create logger and reward tracker
    logger = Logger(log_dir=args.log_dir)
    reward_tracker = RewardTracker(window_size=100)
    
    best_avg_reward = -float('inf')
    
    print("\nStarting training...")
    print(f"Total episodes: {args.episodes}")
    print(f"Update frequency: every {args.update_freq} steps")
    print(f"Checkpoint directory: {args.checkpoint_dir}")
    print("=" * 60)
    
    total_steps = 0
    
    for episode in range(start_episode, args.episodes):
        state, info = env.reset()
        episode_reward = 0
        episode_steps = 0
        done = False
        
        while not done:
            # Select action
            action, value, action_log_prob = agent.select_action(state)
            
            # Process action for environment
            action_processed = network.process_action(action)
            
            # Take step in environment
            next_state, reward, terminated, truncated, info_step = env.step(action_processed)
            done = terminated or truncated
            
            # Extract scalars
            value = value.cpu().numpy()[0][0]
            log_prob = action_log_prob.cpu().numpy()[0][0]
            
            # Store transition
            agent.store_transition(
                state, 
                action.cpu().numpy()[0], 
                reward, 
                value, 
                log_prob, 
                done
            )
            
            episode_reward += reward
            episode_steps += 1
            total_steps += 1
            state = next_state
            
            # Update agent
            if total_steps % args.update_freq == 0 and len(agent.memory) > 0:
                losses = agent.update()
            
            # No need to render explicitly - it's handled by render_mode
        
        # Log why episode ended
        if episode % 10 == 0:  # Log every 10 episodes
            end_reason = "TERMINATED" if terminated else "TRUNCATED"
            print(f"Episode {episode} ended: {end_reason} after {episode_steps} steps, reward={episode_reward:.1f}")
            if 'TimeLimit.truncated' in info_step:
                print(f"  -> Hit TimeLimit (max_episode_steps)")
        
        # Track reward
        reward_tracker.add(episode_reward)
        avg_reward = reward_tracker.get_average()
        
        # Log progress
        if episode % args.update_freq == 0:
            losses = {'policy_loss': 0, 'value_loss': 0, 'entropy_loss': 0}
        logger.log(episode, episode_steps, episode_reward, avg_reward, losses if 'losses' in locals() else None)
        
        # Save checkpoints
        if (episode + 1) % args.save_freq == 0:
            checkpoint_path = os.path.join(args.checkpoint_dir, f'ppo_carracing_ep{episode+1}.pt')
            save_checkpoint(agent, episode + 1, episode_reward, checkpoint_path)
        
        # Save best model
        if avg_reward > best_avg_reward:
            best_avg_reward = avg_reward
            best_path = os.path.join(args.checkpoint_dir, 'ppo_carracing_best.pt')
            save_checkpoint(agent, episode + 1, avg_reward, best_path)
    
    print("\n" + "=" * 60)
    print("Training completed!")
    print(f"Best average reward: {best_avg_reward:.2f}")
    print("=" * 60)
    
    env.close()


def evaluate(args):
    """
    Evaluate trained agent.
    """
    print("=" * 60)
    print("Evaluating PPO Agent on CarRacing-v3")
    print("=" * 60)
    
    # Set device
    device = torch.device('cuda' if torch.cuda.is_available() and args.cuda else 'cpu')
    print(f"Using device: {device}")
    
    # Create environment
    env = make_env(frame_stack=4, frame_skip=2, render_mode="human")
    
    # Create network and agent
    network = ActorCritic(num_inputs=4, num_actions=3)
    agent = PPOAgent(network=network, device=device)
    
    # Load checkpoint
    if not args.checkpoint:
        args.checkpoint = os.path.join(args.checkpoint_dir, 'ppo_carracing_best.pt')
    
    if os.path.exists(args.checkpoint):
        agent.load(args.checkpoint)
    else:
        print(f"Error: Checkpoint not found at {args.checkpoint}")
        return
    
    # Evaluate
    print(f"\nRunning {args.eval_episodes} evaluation episodes...")
    print("=" * 60)
    
    total_rewards = []
    
    for episode in range(args.eval_episodes):
        state, info = env.reset()
        episode_reward = 0
        done = False
        steps = 0
        
        while not done:
            # Select action (deterministic)
            action, _, _ = agent.select_action(state, deterministic=True)
            
            # Process action
            action_processed = network.process_action(action)
            
            # Take step
            next_state, reward, terminated, truncated, _ = env.step(action_processed)
            done = terminated or truncated
            
            episode_reward += reward
            steps += 1
            state = next_state
            
            # Rendering is handled automatically by render_mode
        
        total_rewards.append(episode_reward)
        print(f"Episode {episode + 1:2d} | Steps: {steps:4d} | Reward: {episode_reward:8.2f}")
    
    print("=" * 60)
    print(f"Average Reward: {np.mean(total_rewards):.2f} +/- {np.std(total_rewards):.2f}")
    print(f"Min Reward: {np.min(total_rewards):.2f}")
    print(f"Max Reward: {np.max(total_rewards):.2f}")
    print("=" * 60)
    
    env.close()


def main():
    parser = argparse.ArgumentParser(description='PPO for CarRacing-v3')
    subparsers = parser.add_subparsers(dest='command', help='Command to run')
    
    # Train command
    train_parser = subparsers.add_parser('train', help='Train the agent')
    train_parser.add_argument('--episodes', type=int, default=2000, help='Number of training episodes')
    train_parser.add_argument('--lr', type=float, default=1e-4, help='Learning rate')
    train_parser.add_argument('--gamma', type=float, default=0.99, help='Discount factor')
    train_parser.add_argument('--gae-lambda', type=float, default=0.95, help='GAE lambda')
    train_parser.add_argument('--clip-epsilon', type=float, default=0.2, help='PPO clip epsilon')
    train_parser.add_argument('--value-coef', type=float, default=0.25, help='Value loss coefficient')
    train_parser.add_argument('--entropy-coef', type=float, default=0.005, help='Entropy coefficient')
    train_parser.add_argument('--max-grad-norm', type=float, default=0.5, help='Max gradient norm')
    train_parser.add_argument('--ppo-epochs', type=int, default=4, help='PPO epochs per update')
    train_parser.add_argument('--batch-size', type=int, default=32, help='Mini-batch size')
    train_parser.add_argument('--update-freq', type=int, default=1024, help='Update frequency (steps)')
    train_parser.add_argument('--frame-skip', type=int, default=2, help='Frame skip')
    train_parser.add_argument('--checkpoint', type=str, default=None, help='Checkpoint to load')
    train_parser.add_argument('--checkpoint-dir', type=str, default='checkpoints', help='Checkpoint directory')
    train_parser.add_argument('--log-dir', type=str, default='logs', help='Log directory')
    train_parser.add_argument('--save-freq', type=int, default=100, help='Save frequency (episodes)')
    train_parser.add_argument('--render', action='store_true', help='Render environment')
    train_parser.add_argument('--render-freq', type=int, default=50, help='Render frequency (episodes)')
    train_parser.add_argument('--cuda', action='store_true', default=True, help='Use CUDA')
    
    # Evaluate command
    eval_parser = subparsers.add_parser('evaluate', help='Evaluate the agent')
    eval_parser.add_argument('--checkpoint', type=str, default=None, help='Checkpoint to load')
    eval_parser.add_argument('--checkpoint-dir', type=str, default='checkpoints', help='Checkpoint directory')
    eval_parser.add_argument('--eval-episodes', type=int, default=5, help='Number of evaluation episodes')
    eval_parser.add_argument('--cuda', action='store_true', default=True, help='Use CUDA')
    
    args = parser.parse_args()
    
    if args.command == 'train':
        train(args)
    elif args.command == 'evaluate':
        evaluate(args)
    else:
        parser.print_help()


if __name__ == '__main__':
    main()
