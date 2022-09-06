# Confidential, Copyright 2020, Sony Corporation of America, All rights reserved.
import random
from tqdm import trange
import numpy as np

import pandemic_simulator as ps
from pandemic_simulator.environment.interfaces import InfectionSummary
from pandemic_simulator.environment.done import ORDone, DoneFunctionFactory, DoneFunctionType 


def init_pandemic_env():
    # init globals
    ps.init_globals(seed=112358)
    sim_config = ps.sh.small_town_config
    done_threshold = sim_config.max_hospital_capacity * 3
    done_fn = ORDone(
            done_fns=[
                DoneFunctionFactory.default(
                    DoneFunctionType.INFECTION_SUMMARY_ABOVE_THRESHOLD,
                    summary_type=InfectionSummary.CRITICAL,
                    threshold=done_threshold,
                ),
                DoneFunctionFactory.default(DoneFunctionType.NO_PANDEMIC, num_days=40),
            ]
        )

    env = ps.env.PandemicGymEnv3Act.from_config(sim_config=sim_config, 
                                                pandemic_regulations=ps.sh.austin_regulations,
                                                done_fn=done_fn)
    return env

def eval_policy(policy, env, n_episodes=5):
    rets = []
    ep_lens = []
    for i in trange(n_episodes, desc='Simulating episode'):
        cumu_reward = 0
        obs = env.reset()
        done = False
        ep_len = 0
        while not done:
            action = policy(obs)
            obs, reward, done, aux = env.step(action=action)
            cumu_reward += reward
            ep_len += 1

        rets.append(cumu_reward)
        ep_lens.append(ep_len)
    return np.mean(rets), np.std(rets), rets, ep_lens


if __name__ == '__main__':
    n_eval_episodes = 5
    env = init_pandemic_env()

    policy = lambda obs: -1
    mean_rets, std_rets, rets, ep_lens = eval_policy(policy, env, n_eval_episodes)
    print(f"MEAN/STD RETURN OF MOST LENIENT POLICY: {mean_rets}, {std_rets}\n")

    policy = lambda obs: random.randint(-1, 1)
    mean_rets, std_rets, rets, ep_lens = eval_policy(policy, env, n_eval_episodes)
    print(f"MEAN/STD RETURN OF RANDOM POLICY: {mean_rets}, {std_rets}\n")
    
    policy = lambda obs: 1
    mean_rets, std_rets, rets, ep_lens = eval_policy(policy, env, n_eval_episodes)
    print(f"MEAN/STD RETURN OF MOST STRICT POLICY: {mean_rets}, {std_rets}\n")

