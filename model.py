"""
Q-Learning on FrozenLake from Scratch

Assembled from your step-by-step solutions.
"""

import numpy as np

# Step 1 - init_q_table
import numpy as np

def init_q_table(num_states, num_actions):
    """Return a zero-initialized Q-table of shape (num_states, num_actions)."""
    # TODO: build a 2D float64 numpy array of zeros sized by states and actions.
    
    return np.zeros((num_states,num_actions))

# Step 2 - max_q_value
import numpy as np

def max_q_value(q_table, state):
    """Return the maximum Q value across all actions for the given state."""
    # TODO: index the row for `state` and return its maximum value
    return q_table[state].max()

# Step 3 - greedy_action
import numpy as np

def greedy_action(q_table, state):
    """Return the action index with the highest Q value at the given state."""
    # TODO: return argmax over the action axis for this state's Q values
    return int(np.argmax(q_table[state]))

# Step 4 - sample_random_action
def sample_random_action(action_space):
    # TODO: draw a uniformly random action from the given Gymnasium action space
    return int(action_space.sample())

# Step 5 - should_explore
def should_explore(epsilon, rng):
    """Return True with probability epsilon using the provided numpy Generator."""
    # rng is a generator engine to generate numbers, not the number itself
    rng_val= rng.random() 
    return epsilon> rng_val or epsilon== rng_val

# Step 6 - epsilon_greedy_action
import numpy as np

def epsilon_greedy_action(q_table, state, epsilon, action_space, rng):
    """Return an epsilon-greedy action for the given state."""
    # TODO: with prob epsilon explore via action_space, else pick a max-Q action (random among ties)
    
    if should_explore(epsilon,rng):
        return sample_random_action(action_space)

    all_max= np.where(q_table[state]== q_table[state].max())[0]

    return rng.choice(all_max).item()

# Step 7 - decay_epsilon
def decay_epsilon(epsilon, decay_rate, min_epsilon):
    
    return max(min_epsilon, epsilon * decay_rate)

# Step 8 - td_target
def td_target(reward, gamma, q_table, next_state, done):
    # TODO: compute r + gamma * max_a Q(next_state, a), zeroing the bootstrap when done.
    if not done:
        return reward + (gamma * max_q_value(q_table,next_state))

    return reward

# Step 9 - td_error
def td_error(target, q_table, state, action):
    # TODO: return the TD error: target minus current Q(state, action)
    return target- q_table[state][action]

# Step 10 - q_learning_update
def q_learning_update(q_table, state, action, reward, next_state, done, alpha, gamma):
    # TODO: apply Q(s,a) += alpha * (target - Q(s,a)) in place and return the new Q value
    q_table[state][action]+= alpha * td_error((td_target(reward, gamma , q_table, next_state, done)), q_table, state, action)
    return float(q_table[state][action])

# Step 11 - interaction_step
def interaction_step(env, q_table, state, epsilon, alpha, gamma, rng):
    # TODO: select epsilon-greedy action, step env, apply Q-learning update, return (next_state, reward, done)
    greedy_action= epsilon_greedy_action(q_table, state, epsilon, env.action_space, rng)
    
    next_state,reward, terminated, truncated,_ = env.step(greedy_action)
    done= terminated or truncated

    if done:
       td_target= reward
    else:
        td_target= reward + gamma*np.max(q_table[next_state])

    td_error= td_target - q_table[state][greedy_action]
    q_learning_update(q_table, state, greedy_action, reward, next_state, done, alpha, gamma)

    return (next_state, float(reward), done)

# Step 12 - run_training_episode
def run_training_episode(env, q_table, epsilon, alpha, gamma, rng, max_steps=200):
    # TODO: reset env, then repeatedly call interaction_step until done or max_steps, returning total reward.
    state, _ = env.reset()
    done= False
    steps=0
    tot_reward=0.0

    while not done and steps < max_steps:
        next_state, reward, done= interaction_step(env, q_table, state, epsilon, alpha, gamma, rng)
        
        tot_reward+= reward
        state= next_state

        steps+=1

    return tot_reward

# Step 13 - train_q_learning
import numpy as np

def train_q_learning(env, num_episodes, alpha=0.8, gamma=0.95, epsilon_start=1.0, epsilon_min=0.01, epsilon_decay=0.99, seed=0, max_steps=200):
    # TODO: train a Q-learning agent for num_episodes; return (q_table, returns)
    
    rng=np.random.default_rng(seed)
    env.action_space.seed(seed)
    q_table=init_q_table(16,4)
    episode_returns=[]

    state, _= env.reset(seed=seed)

    
    for ep in range(num_episodes):

        episode_returns.append(run_training_episode(env, q_table, epsilon_start, alpha, gamma, rng, max_steps=200))
        epsilon_start=max(epsilon_min, epsilon_start * epsilon_decay)



    return (q_table,episode_returns)

# Step 14 - extract_greedy_policy
def extract_greedy_policy(q_table):
    # TODO: return a 1D int64 array mapping each state to its best (argmax) action.
    return np.array([greedy_action(q_table, state) for state in range(len(q_table))], dtype= np.int64)

# Step 15 - run_greedy_episode
def run_greedy_episode(env, policy, seed=None, max_steps=200):
    """Run one greedy episode and return True if the goal was reached."""
    # TODO: reset env, follow policy[state] each step, return bool(success)
    state,_ = env.reset(seed=seed)
    
    steps=0
    end_ep= False
    success=False

    tot_reward= 0

    while not end_ep and steps < max_steps :
        action_taken= policy[state]    

        next_state, reward,terminated, truncated, _ = env.step(action_taken)    
        
        if reward>0:
            success= True
            
        state=next_state
        end_ep= truncated or terminated
        steps+=1

        

    return success

# Step 16 - evaluate_success_rate (not yet solved)
# TODO: implement

