import gym
import numpy as np
import agent
from preprocess import FeatExt

def evaluate_agent(eval_agent, eval_env):
    feat_ext_e = FeatExt(s, eval_env)
    cum_return = 0
    for kk in range(5):
        eval_env.reset()
        edone = False
        for k in range(50):
            es, er, edone, _ = eval_env.step(np.asarray([0.0, 0.0, 0.0]))
        while not edone:
            eval_env.render()
            eval_s, ks = feat_ext_e.feat2state(es, eval_env)
            eval_action = eval_agent.action_selection(eval_s.astype(float), False)
            eval_env_action = feat_ext.networkAct2envAct(eval_action)
            es, er, edone, _ = eval_env.step(eval_env_action)
            if ks == 1:
                edone, er = True, -1
            cum_return += er

    return cum_return / 5

train_id = 121
f = open("td3_car_race_evaluate"+str(train_id)+".txt", "w")
env = gym.make("CarRacing-v0")
eval_env = gym.make("CarRacing-v0")
s = env.reset()
feat_ext = FeatExt(s, env)
for k in range(50):
    s, r, done, info = env.step(np.asarray([0.0, 0.0, 0.0]))
dum_state, _ = feat_ext.feat2state(s, env)
gamma, tau, actorlr, criticlr, variance, action_dim = 0.99, 0.0001, 2e-4, 3e-4, 0.1, 2
mem_size, batch_size, state_dim, env_name, exploration_steps = int(5e5), 256, dum_state.shape[0], "car_racing", 25000
agent = agent.Agent(gamma, tau, actorlr, criticlr, variance, action_dim,mem_size, batch_size, state_dim, env_name, exploration_steps)
maxsteps = 20e6
trainsteps = 0
while trainsteps < maxsteps:
    done = False
    s = env.reset()
    for k in range(50):
        s, r, done, info = env.step(np.asarray([0.0, 0.0, 0.0]))
    state, killsig = feat_ext.feat2state(s, env)
    while not done:
        if trainsteps % 10000 == 0:
            print("MODEL SAVED")
            cum_ret = evaluate_agent(agent, eval_env)
            print(f"Training Steps: {trainsteps:10d}, Cumulative Return: {cum_ret}")
            f.write(f"Training Steps: {trainsteps}, Cumulative Return: {cum_ret} \n")
            f.flush()
            agent.save_models(trainsteps, train_id)
        trainsteps += 1
        #env.render()
        action = agent.action_selection(state.astype(float), True)
        env_action = feat_ext.networkAct2envAct(action)
        if trainsteps < exploration_steps:
            env_action = env.action_space.sample()
        next_s, reward, done, info = env.step(env_action)
        next_state, killsig = feat_ext.feat2state(next_s, env)
        if killsig == 1:
            done, reward = True, -1
        # memory store state, action, reward, next_state
        agent.memory.store(state=state, action=action, reward=reward, next_state=next_state, terminal=done)
        if trainsteps > exploration_steps:
            agent.update()
        state = next_state
