# Confidential, Copyright 2020, Sony Corporation of America, All rights reserved.

# training script heavily based off tianshou lunarlander dqn example
import argparse
import os
import pprint
import datetime

import numpy as np
import torch
from torch.utils.tensorboard import SummaryWriter

from tianshou.data import Collector, ReplayBuffer
from tianshou.env import DummyVectorEnv
from tianshou.policy import DQNPolicy
from tianshou.trainer import offpolicy_trainer
from tianshou.utils import TensorboardLogger
from tianshou.utils.net.common import Net

import pandemic_simulator as ps
from pandemic_simulator.environment.interfaces import InfectionSummary
from pandemic_simulator.environment.done import ORDone, DoneFunctionFactory, DoneFunctionType 


def get_args():
    parser = argparse.ArgumentParser()
    # experiment settings
    parser.add_argument('--task', type=str, default='pansim')
    parser.add_argument('--seed', type=int, default=112358)
    parser.add_argument('--device', type=str, 
        default='cuda' if torch.cuda.is_available() else 'cpu')
    # logging
    parser.add_argument('--logdir', type=str, default='log')
    parser.add_argument('--log_interval', type=int, default=500, help="timesteps between logging returns")
    parser.add_argument('--expt_name', type=str, default="", help="optional additional name for logs")
    # dqn hyperparameters
    parser.add_argument('--eps_test', type=float, default=0.0)
    parser.add_argument('--eps_train_init', type=float, default=0.2)
    parser.add_argument('--eps_train_final', type=float, default=0.01)
    parser.add_argument('--buffer_size', type=int, default=10000)
    parser.add_argument('--lr', type=float, default=0.013)
    parser.add_argument('--gamma', type=float, default=0.99)
    parser.add_argument('--n_step', type=int, default=4, )
    parser.add_argument('--target_update_freq', type=int, default=50, 
        help="update target network every <target_update_freq> updates")
    parser.add_argument('--n_epoch', type=int, default=100)
    parser.add_argument('--init_random_steps', type=int, default=600)
    parser.add_argument('--step_per_epoch', type=int, default=3000)
    parser.add_argument('--step_per_collect', type=int, default=360)
    parser.add_argument('--update_per_step', type=float, default=0.0625)
    parser.add_argument('--batch_size', type=int, default=32)
    parser.add_argument('--hidden_sizes', type=int, nargs='*',
        default=[64, 64])
    parser.add_argument('--dueling_q_hidden_sizes', type=int, nargs='*',
        default=[64, 64])
    parser.add_argument('--dueling_v_hidden_sizes', type=int, nargs='*',
        default=[64, 64])
    return parser.parse_args()


def train_dqn(args=get_args()):
    # init env
    ps.init_globals(seed=args.seed)
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
    args.state_shape = env.observation_space.shape
    args.action_shape = env.action_space.shape

    # set random seed
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    env.seed(args.seed)

    # specify network structure
    Q_param = {"hidden_sizes": args.dueling_q_hidden_sizes}
    V_param = {"hidden_sizes": args.dueling_v_hidden_sizes}
    net = Net(
        args.state_shape,
        args.action_shape,
        hidden_sizes=args.hidden_sizes,
        device=args.device,
        dueling_param=(Q_param, V_param)
    ).to(args.device)
    optim = torch.optim.Adam(net.parameters(), lr=args.lr)
    policy = DQNPolicy(
        net,
        optim,
        args.gamma,
        args.n_step,
        target_update_freq=args.target_update_freq
    )

    # initialize collector
    train_collector = Collector(
        policy,
        env,
        ReplayBuffer(args.buffer_size),
        exploration_noise=True
    )
    # we do not use a test collector because we cannot initialize more than 1 env at a time
    test_collector = None

    # setup logging
    date_time = datetime.datetime.now().strftime("%m-%d-%H-%M-%S")
    log_path = os.path.join(args.logdir, args.task, f"dqn_{args.expt_name}_{date_time}")
    writer = SummaryWriter(log_path)
    logger = TensorboardLogger(writer, 
        train_interval=args.log_interval, # this interval is over the number of training timesteps
        update_interval=5) # this interval is over the number of updates

    def save_best_fn(policy):
        torch.save(policy.state_dict(), os.path.join(log_path, "policy.pth"))

    def stop_fn(mean_rewards):
        '''
        If desired, can specify a reward threshold to stop at with the following line:        
        return mean_rewards >= reward_threshold
        '''
        return False

    def train_fn(epoch, env_step):  # exp decay
        eps = max(args.eps_train_init * (1 - 5e-6)**env_step, args.eps_train_final)
        policy.set_eps(eps)

    def test_fn(epoch, env_step):
        policy.set_eps(args.eps_test)

    # prefill buffer with some random data
    policy.set_eps(1)
    train_collector.collect(n_step=args.init_random_steps)

    # main training loop
    result = offpolicy_trainer(
        policy=policy,
        train_collector=train_collector,
        test_collector=test_collector,
        max_epoch=args.n_epoch,
        step_per_epoch=args.step_per_epoch,
        step_per_collect=args.step_per_collect,
        episode_per_test=args.test_num,
        batch_size=args.batch_size,
        update_per_step=args.update_per_step,
        stop_fn=stop_fn,
        train_fn=train_fn,
        test_fn=test_fn,
        save_best_fn=save_best_fn,
        logger=logger
    )

    return result


if __name__ == '__main__':
    args = get_args()
    result = train_dqn(args)
    print("Final result: ")
    pprint.pprint(result)