"""
Q-Learning on FrozenLake from Scratch

Assembled from your step-by-step solutions.
"""

import numpy as np
import time
import pygame

def init_q_table(num_states, num_actions):
    """Return a zero-initialized Q-table of shape (num_states, num_actions)."""
    
    return np.zeros((num_states,num_actions))


def max_q_value(q_table, state):
    """Return the maximum Q value across all actions for the given state."""
   
    return q_table[state].max()

def greedy_action(q_table, state):
    """Return the action index with the highest Q value at the given state."""

    return int(np.argmax(q_table[state]))

def sample_random_action(action_space):
    return int(action_space.sample())

def should_explore(epsilon, rng):
    """Return True with probability epsilon using the provided numpy Generator."""
    # rng is a generator engine to generate numbers, not the number itself
    rng_val= rng.random() 
    return epsilon> rng_val or epsilon== rng_val

def epsilon_greedy_action(q_table, state, epsilon, action_space, rng):
    """Return an epsilon-greedy action for the given state."""
    
    if should_explore(epsilon,rng):
        return sample_random_action(action_space)

    all_max= np.where(q_table[state]== q_table[state].max())[0]

    return rng.choice(all_max).item()

def decay_epsilon(epsilon, decay_rate, min_epsilon):
    
    return max(min_epsilon, epsilon * decay_rate)

def td_target(reward, gamma, q_table, next_state, done):
    if not done:
        return reward + (gamma * max_q_value(q_table,next_state))

    return reward

def td_error(target, q_table, state, action):
    return target- q_table[state][action]

def q_learning_update(q_table, state, action, reward, next_state, done, alpha, gamma):
    q_table[state][action]+= alpha * td_error((td_target(reward, gamma , q_table, next_state, done)), q_table, state, action)
    return float(q_table[state][action])

def interaction_step(env, q_table, state, epsilon, alpha, gamma, rng):
    action= epsilon_greedy_action(q_table, state, epsilon, env.action_space, rng)
    
    next_state,reward, terminated, truncated,_ = env.step(action)
    done= terminated or truncated

    q_learning_update(q_table, state, action, reward, next_state, done, alpha, gamma)

    return (next_state, float(reward), done)

def run_training_episode(env, q_table, epsilon, alpha, gamma, rng, should_watch, max_steps=200):
    state, _ = env.reset()
    done= False
    steps=0
    tot_reward=0.0

    env.unwrapped.render_mode="human" if should_watch else None
    if should_watch:
         
        print(f"Visualizing Episode...")
        time.sleep(1.0) 


    while not done and steps < max_steps:
        if env.render_mode == "human":
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    print("\n'X' clicked during training! Closing safely...")
                    env.close()
                    exit()
        next_state, reward, done= interaction_step(env, q_table, state, epsilon, alpha, gamma, rng)
        
        tot_reward+= reward
        state= next_state

        steps+=1

    return tot_reward


def train_q_learning(env, num_episodes, alpha=0.8, gamma=0.95, epsilon_start=1.0, epsilon_min=0.01, epsilon_decay=0.99, seed=0, max_steps=200):
    
    rng=np.random.default_rng(seed)
    env.action_space.seed(seed)
    q_table=init_q_table(16,4)
    episode_returns=[]


    state, _= env.reset(seed=seed)
   
    for ep in range(num_episodes):
        should_watch = (ep % 200 == 0) or (ep == num_episodes - 1)
        episode_returns.append(run_training_episode(env, q_table, epsilon_start, alpha, gamma, rng,should_watch, max_steps=200))
        epsilon_start=max(epsilon_min, epsilon_start * epsilon_decay)

      

    return (q_table,episode_returns)

def extract_greedy_policy(q_table):
    return np.array([greedy_action(q_table, state) for state in range(len(q_table))], dtype= np.int64)


def run_greedy_episode(env, policy, seed=None, max_steps=200):
    """Run one greedy episode and return True if the goal was reached."""
   
    state,_ = env.reset(seed=seed)
    
    steps=0
    end_ep= False
    success=False

    while not end_ep and steps < max_steps :
       
        if env.render_mode == "human":
    
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    print("\n'X' clicked! Closing environment safely...")
                    env.close()
                    exit() 

        action_taken= policy[state]    

        next_state, reward,terminated, truncated, _ = env.step(action_taken)    
        
        if reward>0:
            success= True
            
        state=next_state
        end_ep= truncated or terminated
        steps+=1

        

    return success

def evaluate_success_rate(env, policy, num_episodes, seed=0, max_steps=200):

    return np.average([run_greedy_episode(env,policy, seed + ep, max_steps) for ep in range(num_episodes)])

