import random
import gymnasium as gym
import numpy as np
import matplotlib.pyplot as plt
import torch
from torch import nn
from model import DQN, ReplayMemory


class MountainCarDQL():
    def __init__(self, learning_rate, discount_factor, network_sync_rate,
                 replay_memory_size,
                 mini_batch_size,
                 num_divisions, steps_per_epoch):
        # Hyperparameters (adjustable)
        # learning rate (alpha)
        self.learning_rate = learning_rate
        # discount rate (gamma)
        self.discount_factor = discount_factor
        # number of steps the agent takes before syncing the policy and target network
        self.network_sync_rate = network_sync_rate
        # size of replay memory
        self.replay_memory_size = replay_memory_size
        # size of the training data set sampled from the replay memory
        self.mini_batch_size = mini_batch_size
        self.num_divisions = num_divisions
        self.steps_per_epoch = steps_per_epoch

    # NN Loss function. MSE=Mean Squared Error can be swapped to something else.
    loss_fn = nn.MSELoss()
    # NN Optimizer. Initialize later.
    optimizer = None

    # Train the environment
    def train(self, episodes, render=False):
        # Create MountainCar instance
        env = gym.make('MountainCar-v0',
                       render_mode='human' if render else None)
        # expecting 2: position & velocity
        num_states = env.observation_space.shape[0]
        num_actions = env.action_space.n
        # Divide position and velocity into segments
        self.pos_space = np.linspace(
            # Between -1.2 and 0.6
            env.observation_space.low[0], env.observation_space.high[0], self.num_divisions)
        self.vel_space = np.linspace(
            # Between -0.07 and 0.07
            env.observation_space.low[1], env.observation_space.high[1], self.num_divisions)

        epsilon = 1  # 1 = 100% random actions
        memory = ReplayMemory(self.replay_memory_size)

        # Create policy and target network. Number of nodes in the hidden layer can be adjusted.
        policy_dqn = DQN(in_states=num_states, h1_nodes=10,
                         out_actions=num_actions)
        target_dqn = DQN(in_states=num_states, h1_nodes=10,
                         out_actions=num_actions)

        # Make the target and policy networks the same (copy weights/biases from one network to the other)
        target_dqn.load_state_dict(policy_dqn.state_dict())

        # Policy network optimizer. "Adam" optimizer can be swapped to something else.
        self.optimizer = torch.optim.Adam(
            policy_dqn.parameters(), lr=self.learning_rate)

        # List to keep track of rewards collected per episode. Initialize list to 0's.
        rewards_per_episode = []

        # List to keep track of epsilon decay
        epsilon_history = []

        # Track number of steps taken. Used for syncing policy => target network.
        step_count = 0
        goal_reached = False
        best_rewards = -200

        for i in range(episodes):
            state = env.reset()[0]  # Initialize to state 0
            terminated = False      # True when agent falls in hole or reached goal

            rewards = 0

            # Agent navigates map until it falls into hole/reaches goal (terminated), or has taken 200 actions (truncated).
            while (not terminated and rewards > -self.steps_per_epoch):

                # Select action based on epsilon-greedy
                if random.random() < epsilon:
                    # select random action
                    action = env.action_space.sample()  # actions: 0=left,1=idle,2=right
                else:
                    # select best action
                    with torch.no_grad():
                        action = policy_dqn(
                            self.state_to_dqn_input(state)).argmax().item()

                # Execute action
                new_state, reward, terminated, _, _ = env.step(action)

                # Accumulate reward
                rewards += reward

                # Save experience into memory
                memory.append((state, action, new_state, reward, terminated))

                # Move to the next state
                state = new_state

                # Increment step counter
                step_count += 1

            # Keep track of the rewards collected per episode.
            rewards_per_episode.append(rewards)
            if terminated:
                goal_reached = True

            # Graph training progress
            if (i != 0 and i % self.steps_per_epoch == 0):
                print(f'Episode {i} Epsilon {epsilon}')

                self.plot_progress(rewards_per_episode, epsilon_history)

            if rewards > best_rewards:
                best_rewards = rewards
                print(f'Best rewards so far: {best_rewards}')
                # Save policy
                torch.save(policy_dqn.state_dict(),
                           f"result_mountaincar_dql_{i}.pt")

            # Check if enough experience has been collected
            if len(memory) > self.mini_batch_size and goal_reached:
                mini_batch = memory.sample(self.mini_batch_size)
                self.optimize(mini_batch, policy_dqn, target_dqn)

                # Decay epsilon
                epsilon = max(epsilon - 1/episodes, 0)
                epsilon_history.append(epsilon)

                # Copy policy network to target network after a certain number of steps
                if step_count > self.network_sync_rate:
                    target_dqn.load_state_dict(policy_dqn.state_dict())
                    step_count = 0

        # Close environment
        env.close()

    def plot_progress(self, rewards_per_episode, epsilon_history):
        # Create new graph
        plt.figure(1)

        # Plot average rewards (Y-axis) vs episodes (X-axis)
        # rewards_curve = np.zeros(len(rewards_per_episode))
        # for x in range(len(rewards_per_episode)):
        # rewards_curve[x] = np.min(rewards_per_episode[max(0, x-10):(x+1)])
        plt.subplot(121)  # plot on a 1 row x 2 col grid, at cell 1
        # plt.plot(sum_rewards)
        plt.plot(rewards_per_episode)

        # Plot epsilon decay (Y-axis) vs episodes (X-axis)
        plt.subplot(122)  # plot on a 1 row x 2 col grid, at cell 2
        plt.plot(epsilon_history)

        # Save plots
        plt.savefig('mountaincar_dql.png')
    # Optimize policy network

    def optimize(self, mini_batch, policy_dqn, target_dqn):

        current_q_list = []
        target_q_list = []

        for state, action, new_state, reward, terminated in mini_batch:

            if terminated:
                # Agent receive reward of 0 for reaching goal.
                # When in a terminated state, target q value should be set to the reward.
                target = torch.FloatTensor([reward])
            else:
                # Calculate target q value
                with torch.no_grad():
                    target = torch.FloatTensor(
                        reward + self.discount_factor *
                        target_dqn(self.state_to_dqn_input(new_state)).max()
                    )

            # Get the current set of Q values
            current_q = policy_dqn(self.state_to_dqn_input(state))
            current_q_list.append(current_q)

            # Get the target set of Q values
            target_q = target_dqn(self.state_to_dqn_input(state))
            # Adjust the specific action to the target that was just calculated
            target_q[action] = target
            target_q_list.append(target_q)

        # Compute loss for the whole minibatch
        loss = self.loss_fn(torch.stack(current_q_list),
                            torch.stack(target_q_list))

        # Optimize the model
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

    '''
    Converts a state (position, velocity) to tensor representation.
    Example:
    Input = (0.3, -0.03)
    Return = tensor([16, 6])
    '''

    def state_to_dqn_input(self, state) -> torch.Tensor:
        state_p = np.digitize(state[0], self.pos_space)
        state_v = np.digitize(state[1], self.vel_space)

        return torch.FloatTensor([state_p, state_v])

    # Run the environment with the learned policy
    def run(self, episodes, model_filepath):
        # Create FrozenLake instance
        env = gym.make('MountainCar-v0', render_mode='human')
        num_states = env.observation_space.shape[0]
        num_actions = env.action_space.n

        self.pos_space = np.linspace(
            # Between -1.2 and 0.6
            env.observation_space.low[0], env.observation_space.high[0], self.num_divisions)
        self.vel_space = np.linspace(
            # Between -0.07 and 0.07
            env.observation_space.low[1], env.observation_space.high[1], self.num_divisions)

        # Load learned policy
        policy_dqn = DQN(in_states=num_states, h1_nodes=10,
                         out_actions=num_actions)
        policy_dqn.load_state_dict(torch.load(model_filepath))
        policy_dqn.eval()    # switch model to evaluation mode

        for i in range(episodes):
            state = env.reset()[0]  # Initialize to state 0
            terminated = False      # True when agent falls in hole or reached goal
            truncated = False       # True when agent takes more than 200 actions

            # Agent navigates map until it falls into a hole (terminated), reaches goal (terminated), or has taken 200 actions (truncated).
            while (not terminated and not truncated):
                # Select best action
                with torch.no_grad():
                    action = policy_dqn(
                        self.state_to_dqn_input(state)).argmax().item()

                # Execute action
                state, reward, terminated, truncated, _ = env.step(action)

        env.close()
