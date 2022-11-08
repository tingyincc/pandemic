# Confidential, Copyright 2020, Sony Corporation of America, All rights reserved.

from tqdm import trange

import pandemic_simulator as ps
import random


def run_pandemic_gym_env() -> None:
    """Here we execute the gym envrionment wrapped simulator using austin regulations,
    a small town config and default person routines."""

    print('\nA tutorial that runs the OpenAI Gym environment wrapped simulator', flush=True)

    # init globals
    ps.init_globals(seed=104923490)

    # select a simulator config
    # sim_config = ps.sh.small_town_config
    sim_config = ps.sh.my_town_config


    # make env

    wrap = ps.env.PandemicGymEnv.from_config(sim_config = sim_config, pandemic_regulations=ps.sh.italian_regulations)
    start_day = 29 # march_1

    # setup viz
    viz = ps.viz.GymViz.from_config(sim_config=sim_config)
    sim_viz = ps.viz.SimViz.from_config(sim_config=sim_config)

    # run stage-0 action steps in the environment
    wrap.reset()
    Reward = 0
    for i in trange(120, desc='Simulating day'):
        
        
        if i==0:
            action = 0 

        else:
            if i%10==0:
                viz.plot()
                sim_viz.plot()
                
            #######################################################################################################################################            
            #Replace the code in the below if-else statement with your own policy, based on observation variables
            # if obs.time_day[...,0]>20:
            #     action = 1
            # elif not obs.infection_above_threshold:
            #     action = 0
            # else:
            #     action = 4

            
            if obs.time_day [...,0]>=start_day and obs.time_day [...,0]< start_day+4:
                action = 1
            elif obs.time_day [...,0]>=start_day+4 and obs.time_day [...,0]<start_day+9:
                action = 2
            elif obs.time_day [...,0]>=start_day+9 and obs.time_day [...,0]<start_day+21: # march 21
                action = 3
            elif obs.time_day [...,0]>=start_day+21 and obs.time_day [...,0]<start_day+57: # april 26
                action = 4
            elif obs.time_day [...,0]>=start_day+57 and obs.time_day [...,0]<start_day+73: # may 13
                action = 3
            elif obs.time_day [...,0]>=start_day+74:
                action = 2

            ########################################################################################################################################

            # italian_strategy = [ps.data.StageSchedule(stage=0, end_day=3),
            #             ps.data.StageSchedule(stage=1, end_day=8),
            #             ps.data.StageSchedule(stage=2, end_day=13),
            #             ps.data.StageSchedule(stage=3, end_day=25),
            #             ps.data.StageSchedule(stage=4, end_day=59),
            #             ps.data.StageSchedule(stage=3, end_day=79),
            #             ps.data.StageSchedule(stage=2, end_day=None)]

        obs, reward, done, aux = wrap.step(action=int(action))  # here the action is the discrete regulation stage identifier
        print(obs)
        Reward += reward
        viz.record((obs, reward))
        sim_viz.record_state(state = wrap.pandemic_sim.state)
    # generate plots
    viz.plot()
    sim_viz.plot()
    print('Reward:'+str(Reward))


if __name__ == '__main__':
    run_pandemic_gym_env()

