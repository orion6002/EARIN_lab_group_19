import gymnasium as gym
import random as rand
import numpy as np

def get_q_table(train_iter, alpha, gamma, eps_step): 
    q_table = np.zeros((500, 6))
    eps = 100 + eps_step
    for iter in train_iter:
        eps = eps - eps_step
        for _ in range(200):
            env = gym.make("Taxi-v4", render_mode="ansi")
            state, _ = env.reset()
            if rand.rand() < eps/100:
                action = choose_best_action(state)
            action = env.action_space.sample()
            state, reward, terminated, truncated, info = env.step(action)

            if terminated or truncated:
                state, info = env.reset()
    
    env.close()

    return q_table

def calc_new_q_value(q_table, alpha, gamma, reward, state):
    return

def choose_best_action(state):
    return