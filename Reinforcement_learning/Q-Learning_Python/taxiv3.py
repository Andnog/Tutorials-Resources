import gymnasium as gym
import numpy as np
import matplotlib.pyplot as plt

# Initialize the Taxi-v3 environment with render_mode set to "ansi" for text-based output
env = gym.make("Taxi-v3", render_mode="ansi")
env.reset()

# Parameters
alpha = 0.1       # Learning rate
gamma = 0.9       # Discount factor
epsilon = 0.1     # Exploration rate
epsilon_decay = 0.99  # Decay rate for epsilon
episodes = 1000   # Number of training episodes

# Initialize Q-Table with zeros
Q = np.zeros([env.observation_space.n, env.action_space.n])

# List to store total rewards per episode for visualization
reward_list = []

# Training the agent
for episode in range(episodes):
    state, _ = env.reset()
    done = False
    total_rewards = 0  # Track total rewards for this episode

    while not done:
        # Epsilon-greedy action selection
        if np.random.uniform(0, 1) < epsilon:
            action = env.action_space.sample()  # Explore
        else:
            action = np.argmax(Q[state])  # Exploit

        # Take the action, observe reward and next state
        next_state, reward, done, truncated, _ = env.step(action)
        total_rewards += reward

        # Q-Learning update
        Q[state, action] = Q[state, action] + alpha * (reward + gamma * np.max(Q[next_state]) - Q[state, action])

        # Move to the next state
        state = next_state

    # Decay epsilon after each episode
    epsilon *= epsilon_decay

    # Append total rewards for this episode to the reward list
    reward_list.append(total_rewards)

# Testing the agent
state, _ = env.reset()
done = False
total_rewards = 0

print("\nTesting the agent's performance:")
while not done:
    # Select the action with the highest Q-value in the current state
    action = np.argmax(Q[state])
    
    # Take the action and observe the reward and next state
    next_state, reward, done, truncated, _ = env.step(action)
    total_rewards += reward

    # Render the environment in text mode to visualize the agent's actions
    print(env.render())
    
    # Move to the next state
    state = next_state

print("Total Rewards:", total_rewards)

# Plot total rewards per episode to visualize learning progress
plt.plot(reward_list)
plt.xlabel("Episodes")
plt.ylabel("Total Rewards")
plt.title("Agent's Learning Progress")
plt.show()

# Optional: Smoothing the learning curve with a rolling average
window_size = 50
smoothed_rewards = [np.mean(reward_list[i-window_size:i+1]) if i >= window_size else np.mean(reward_list[:i+1]) for i in range(len(reward_list))]

plt.plot(smoothed_rewards)
plt.xlabel("Episodes")
plt.ylabel("Average Total Rewards")
plt.title("Smoothed Learning Progress of the Agent")
plt.show()
