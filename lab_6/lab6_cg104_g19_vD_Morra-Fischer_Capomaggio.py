import gymnasium as gym
import random as rand
import numpy as np

def get_q_table(train_iter, alpha, gamma, min_eps): 
    q_table = np.zeros((500, 6))
    eps = 100
    eps_step = (100 - min_eps) / train_iter
    env = gym.make("Taxi-v4")
    for _ in range(train_iter):
        eps = eps - eps_step
        state, _ = env.reset()
        for _ in range(200):
            if rand.random() < eps/100:
                action = env.action_space.sample()
            else:
                action = np.argmax(q_table[state])
            next_state, reward, terminated, truncated, info = env.step(action)
            q_table[state][action] += alpha * (reward + gamma*max(q_table[next_state]) - q_table[state][action])
            state = next_state
            if terminated or truncated:
                break
    env.close()
    return q_table


def testing(q_table):
    env = gym.make("Taxi-v4", render_mode="human")
    state, _ = env.reset()
    for _ in range(200):
        action = np.argmax(q_table[state])
        state, reward, terminated, truncated, info = env.step(action)
        if terminated or truncated:
            return
    env.close()

def evaluate(q_table, n_episodes):
    env = gym.make("Taxi-v4")
    total_rewards = []
    total_steps = []
    successes = 0

    for _ in range(n_episodes):
        state, _ = env.reset()
        ep_reward = 0
        ep_steps = 0

        for _ in range(200):
            action = np.argmax(q_table[state])
            state, reward, terminated, truncated, info = env.step(action)
            ep_reward += reward
            ep_steps += 1

            if terminated or truncated:
                if reward == 20:
                    successes += 1
                break

        total_rewards.append(ep_reward)
        total_steps.append(ep_steps)

    env.close()

    print(f"--> Results of {n_episodes} tests")
    print(f"Success rate : {successes/n_episodes*100:.1f}%")
    print(f"Avg reward   : {np.mean(total_rewards):.2f}")
    print(f"Best reward  : {np.max(total_rewards):.2f}")
    print(f"Worst reward : {np.min(total_rewards):.2f}")
    print(f"Avg step nb  : {np.mean(total_steps):.1f}")


alpha = 0.1
gamma = 0.95
min_eps = 5
train_iter = 100


q_t = get_q_table(train_iter, alpha, gamma, min_eps)
print(q_t)
evaluate(q_t, 30)