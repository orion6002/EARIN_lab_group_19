"""
Lab 6 - Reinforcement Learning - Variant 4: Taxi-v3

Requirements:
    pip install gymnasium numpy matplotlib

Run:
    python lab6_cg104_g19_vD_Morra-Fischer_Capomaggio.py
"""
import gymnasium as gym
import random as rand
import numpy as np
import matplotlib.pyplot as plt

ENV_NAME = "Taxi-v3"
MAX_STEPS = 200
SEED = 42

def get_q_table(train_iter, alpha, gamma, min_eps): 
    rand.seed(SEED)
    np.random.seed(SEED)

    eps = 100
    eps_step = (100 - min_eps) / train_iter
    env = gym.make(ENV_NAME)
    env.action_space.seed(SEED)
    q_table = np.zeros((env.observation_space.n, env.action_space.n))
    rewards_per_ep = []
    steps_per_ep = []

    for episode in range(train_iter):
        eps = eps - eps_step
        state, _ = env.reset(seed=SEED + episode)
        tot_reward = 0
        steps = 0
        for _ in range(MAX_STEPS):
            if rand.random() < eps/100:
                action = env.action_space.sample()
            else:
                action = np.argmax(q_table[state])
            next_state, reward, terminated, truncated, _ = env.step(action)
            update_q_table(q_table, alpha, gamma, state, action, reward, next_state, terminated)
            state = next_state
            tot_reward += reward
            steps += 1
            if terminated or truncated:
                break
        rewards_per_ep.append(tot_reward)
        steps_per_ep.append(steps)
    env.close()
    return q_table, rewards_per_ep, steps_per_ep

def update_q_table(q_table, alpha, gamma, state, action, reward, next_state, terminated):
    """Update one Q-value using the Q-Learning rule."""
    # q_table is a reference, so there is no need to return it
    current_value = q_table[state][action]

    if terminated:
        target = reward
    else:
        target = reward + gamma * np.max(q_table[next_state])

    q_table[state][action] = current_value + alpha * (target - current_value)

def evaluate(q_table, n_episodes):
    """Evaluate the learned policy without exploration."""
    rand.seed(SEED)
    np.random.seed(SEED)

    env = gym.make(ENV_NAME)
    env.action_space.seed(SEED)
    total_rewards = []
    total_steps = []
    successes = 0

    for episode in range(n_episodes):
        state, _ = env.reset(seed=SEED + episode)
        ep_reward = 0
        ep_steps = 0

        for _ in range(MAX_STEPS):
            action = np.argmax(q_table[state])
            state, reward, terminated, truncated, _ = env.step(action)
            ep_reward += reward
            ep_steps += 1

            if terminated or truncated:
                if terminated and reward == 20:
                    successes += 1
                break

        total_rewards.append(ep_reward)
        total_steps.append(ep_steps)

    env.close()

    print(f"--> Results of {n_episodes} tests")
    print(f"Success rate : {successes / n_episodes * 100:.1f}%")
    print(f"Avg reward   : {np.mean(total_rewards):.2f}")
    print(f"Best reward  : {np.max(total_rewards):.2f}")
    print(f"Worst reward : {np.min(total_rewards):.2f}")
    print(f"Avg step nb  : {np.mean(total_steps):.1f}")

    return {
        "success_rate": round(float(successes / n_episodes * 100), 2),
        "avg_reward": round(float(np.mean(total_rewards)), 2),
        "best_reward": int(np.max(total_rewards)),
        "worst_reward": int(np.min(total_rewards)),
        "avg_steps": round(float(np.mean(total_steps)), 2),
    }

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
            
    plt.title("Evolution of the total reward during training")
    plt.xlabel("Episodes")
    plt.ylabel("Total reward")
    plt.legend()
    plt.grid(True)
    plt.show()


def plot_iteration_results(iteration_results):
    """Plot final evaluation metrics for different training iteration counts."""
    labels = list(iteration_results.keys())
    success_rates = [iteration_results[label]["success_rate"] for label in labels]
    avg_rewards = [iteration_results[label]["avg_reward"] for label in labels]
    avg_steps = [iteration_results[label]["avg_steps"] for label in labels]

    x = np.arange(len(labels))

    plt.figure(figsize=(10, 6))
    plt.bar(x, success_rates)
    plt.xticks(x, labels)
    plt.title("Success rate after different numbers of training episodes")
    plt.xlabel("Training episodes")
    plt.ylabel("Success rate (%)")
    plt.grid(axis="y")
    plt.show()

    print("\n=== Iteration experiment summary ===")
    print(f"{'Training episodes':<20} {'Success rate':<15} {'Avg reward':<12} {'Avg steps':<10}")
    for label in labels:
        metrics = iteration_results[label]
        print(
            f"{label:<20} "
            f"{metrics['success_rate']:<15.2f} "
            f"{metrics['avg_reward']:<12.2f} "
            f"{metrics['avg_steps']:<10.2f}"
        )

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

    iteration_results = {}
    iterations_list = [100, 1000, 10000, 50000]
    for i in iterations_list:
        print(f"Iter = {i}...")
        q_table, _, _ = get_q_table(i, 0.1, 0.95, 5)
        iteration_results[f"Iter {i}"] = evaluate(q_table, 100)
    plot_iteration_results(iteration_results)

    print("\n=== Final evaluation of the selected baseline configuration ===")
    q_table, _, _ = get_q_table(10000, 0.1, 0.95, 5)
    evaluate(q_table, 100)

if __name__ == "__main__":
    experiments()