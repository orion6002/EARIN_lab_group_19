import gymnasium as gym
import random as rand
import numpy as np
import matplotlib.pyplot as plt

def get_q_table(train_iter, alpha, gamma, min_eps): 
    eps = 100
    eps_step = (100 - min_eps) / train_iter
    env = gym.make("Taxi-v4") # we use Taxi-v4 because v3 is depreciated and causes blocking errors to the execution
    q_table = np.zeros((env.observation_space.n, env.action_space.n))
    rewards_per_ep = []
    steps_per_ep = []

    for _ in range(train_iter):
        eps = eps - eps_step
        state, _ = env.reset()
        tot_reward = 0
        steps = 0
        for _ in range(200):
            if rand.random() < eps/100:
                action = env.action_space.sample()
            else:
                action = np.argmax(q_table[state])
            next_state, reward, terminated, truncated, info = env.step(action)
            update_q_table(q_table, alpha, gamma, state, action, reward, next_state)
            state = next_state
            tot_reward += reward
            steps += 1
            if terminated or truncated:
                break
        rewards_per_ep.append(tot_reward)
        steps_per_ep.append(steps)
    env.close()
    return q_table, rewards_per_ep, steps_per_ep

def update_q_table(q_table, alpha, gamma, state, action, reward, next_state):
    # q_table is a reference so no return
    q_table[state][action] += alpha * (reward + gamma*max(q_table[next_state]) - q_table[state][action])


# def evaluate(q_table, n_episodes):
#     """This is only for executing a number of run without ploting ang having stats in the end """
#     env = gym.make("Taxi-v4")
#     total_rewards = []
#     total_steps = []
#     successes = 0

#     for _ in range(n_episodes):
#         state, _ = env.reset()
#         ep_reward = 0
#         ep_steps = 0

#         for _ in range(200):
#             action = np.argmax(q_table[state])
#             state, reward, terminated, truncated, info = env.step(action)
#             ep_reward += reward
#             ep_steps += 1

#             if terminated or truncated:
#                 if reward == 20:
#                     successes += 1
#                 break

#         total_rewards.append(ep_reward)
#         total_steps.append(ep_steps)

#     env.close()

#     print(f"--> Results of {n_episodes} tests")
#     print(f"Success rate : {successes/n_episodes*100:.1f}%")
#     print(f"Avg reward   : {np.mean(total_rewards):.2f}")
#     print(f"Best reward  : {np.max(total_rewards):.2f}")
#     print(f"Worst reward : {np.min(total_rewards):.2f}")
#     print(f"Avg step nb  : {np.mean(total_steps):.1f}")

# #example of the usage of evaluate
# q_t = get_q_table(10000, 0.1, 0.95, 5)
# evaluate(q_t, 30)


def plot_learning_curves(results):
    """Helper function for plot"""
    plt.figure(figsize=(10, 6))
    for label, data in results.items():
        window = 50
        if len(data) > window:
            smoothed_data = np.convolve(data, np.ones(window)/window, mode='valid')
            plt.plot(smoothed_data, label=label)
        else:
            plt.plot(data, label=label)
            
    plt.title(f"Evolution of the total reward during training")
    plt.xlabel("Epidodes")
    plt.ylabel("total reward")
    plt.legend()
    plt.grid(True)
    plt.show()


def experiments():
    iterations = 2000 # quite small iteration number to see the impact of the different hyperparameters on graphs

    all_rewards = {}
    alphas = [0.01, 0.1, 0.5, 1]
    for a in alphas:
        print(f"Alpha = {a}...")
        _, rewards, _ = get_q_table(iterations, a, 0.95, 5)
        all_rewards[f"Alpha {a}"] = rewards
    plot_learning_curves(all_rewards)

    all_rewards_gamma = {}
    gammas = [0.3, 0.6, 0.9, 0.99]
    for g in gammas:
        print(f"Gamma = {g}...")
        _, rewards, _ = get_q_table(iterations, 0.1, g, 5)
        all_rewards_gamma[f"Gamma {g}"] = rewards
    plot_learning_curves(all_rewards_gamma)
    
    all_rewards_eps = {}
    eps = [0, 20, 50, 100]
    for e in eps:
        print(f"Eps = {e}...")
        _, rewards, _ = get_q_table(iterations, 0.1, 0.95, e)
        all_rewards_eps[f"Eps {e}"] = rewards
    plot_learning_curves(all_rewards_eps)

    all_rewards_iter = {}
    iter = [100, 1000, 10000, 50000]
    for i in iter:
        print(f"Iter = {i}...")
        _, rewards, _ = get_q_table(i, 0.1, 0.95, 5)
        all_rewards_iter[f"Iter {i}"] = rewards
    plot_learning_curves(all_rewards_iter)

experiments()