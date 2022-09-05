# training script heavily based off tianshou lunarlander dqn example
import argparse
import os
import pprint
import datetime

import gym
import numpy as np
import torch
from torch.utils.tensorboard import SummaryWriter

from tianshou.data import Collector, VectorReplayBuffer, ReplayBuffer
from tianshou.env import DummyVectorEnv, SubprocVectorEnv
from tianshou.policy import DQNPolicy
from tianshou.trainer import offpolicy_trainer
from tianshou.utils import TensorboardLogger
from tianshou.utils.net.common import Net

# our imports
import pandemic_simulator as ps
from pandemic_simulator.environment.interfaces import InfectionSummary
from pandemic_simulator.environment.done import ORDone, DoneFunctionFactory, DoneFunctionType 

sim_config = ps.sh.small_town_config


def get_args():
    parser = argparse.ArgumentParser()
    # the parameters are found by Optuna
    parser.add_argument('--task', type=str, default='pansim')
    parser.add_argument('--seed', type=int, default=0)
    parser.add_argument('--eps-test', type=float, default=0.0)
    parser.add_argument('--eps-train-init', type=float, default=0.2)
    parser.add_argument('--eps-train-final', type=float, default=0.01)
    parser.add_argument('--buffer-size', type=int, default=1000)
    parser.add_argument('--lr', type=float, default=0.013)
    parser.add_argument('--gamma', type=float, default=0.99)
    parser.add_argument('--n-step', type=int, default=4)
    parser.add_argument('--target-update-freq', type=int, default=500) # this is every 500 UPDATES; test value=2
    parser.add_argument('--n-epoch', type=int, default=100)
    parser.add_argument('--step-per-epoch', type=int, default=3000)
    parser.add_argument('--step-per-collect', type=int, default=360)
    parser.add_argument('--update-per-step', type=float, default=0.0625)
    parser.add_argument('--batch-size', type=int, default=32)
    parser.add_argument('--hidden-sizes', type=int, nargs='*', default=[128, 128])
    parser.add_argument(
        '--dueling-q-hidden-sizes', type=int, nargs='*', default=[128, 128]
    )
    parser.add_argument(
        '--dueling-v-hidden-sizes', type=int, nargs='*', default=[128, 128]
    )
    parser.add_argument('--training-num', type=int, default=1)
    parser.add_argument('--test-num', type=int, default=1)
    parser.add_argument('--logdir', type=str, default='log')
    parser.add_argument('--log-interval', type=int, default=500, help="timesteps between logging")
    parser.add_argument('--expt-name', type=str, default="", help="optional name for logs")
    parser.add_argument('--render', type=float, default=0.)
    parser.add_argument(
        '--device', type=str, default='cuda' if torch.cuda.is_available() else 'cpu'
    )
    return parser.parse_args()


def train_dqn(args=get_args()):
    ps.init_globals(seed=args.seed)
    # env = ps.env.PandemicGymEnv3Act.from_config(sim_config=sim_config, 
        # pandemic_regulations=ps.sh.austin_regulations)

    args.state_shape = (1, 1, 13) # env.observation_space.shape or env.observation_space.n
    args.action_shape = 5 # env.action_space.shape or env.action_space.n
    # del env   

    # train_envs = gym.make(args.task)
    # you can also use tianshou.env.SubprocVectorEnv
    done_threshold = sim_config.max_hospital_capacity * 3
    done_fn = ORDone(
            done_fns=[
                DoneFunctionFactory.default(
                    DoneFunctionType.INFECTION_SUMMARY_ABOVE_THRESHOLD,
                    summary_type=InfectionSummary.CRITICAL,
                    threshold=done_threshold,
                ),
                DoneFunctionFactory.default(DoneFunctionType.NO_PANDEMIC, num_days=40),
                # InfectionSummaryAboveThresholdDone, 
            ]
        )

    train_envs = ps.env.PandemicGymEnv3Act.from_config(sim_config=sim_config, 
                 pandemic_regulations=ps.sh.austin_regulations,
                 done_fn=done_fn)
    # make_env = lambda: ps.env.PandemicGymEnv3Act.from_config(sim_config=sim_config, 
    #     pandemic_regulations=ps.sh.austin_regulations,
    #     done_fn=done_fn)

    # train_envs = SubprocVectorEnv(
    #     [make_env for _ in range(args.training_num)]
    # )
    # test_envs = SubprocVectorEnv(
    #     [make_env for _ in range(args.test_num)]
    # )
    # seed
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    train_envs.seed(args.seed)
    # test_envs.seed(args.seed)
    # model
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
    # collector
    train_collector = Collector(
        policy,
        train_envs,
        ReplayBuffer(args.buffer_size),
        # VectorReplayBuffer(args.buffer_size, len(train_envs)),
        exploration_noise=True
    )
    # test_collector = Collector(policy, test_envs, exploration_noise=True)
    test_collector = None
    # policy.set_eps(1)
    train_collector.collect(n_step=args.batch_size * args.training_num)
    # log
    date_time = datetime.datetime.now().strftime("%m-%d-%H-%M-%S")
    log_path = os.path.join(args.logdir, args.task, f"dqn_{args.expt_name}_{date_time}")
    writer = SummaryWriter(log_path)
    logger = TensorboardLogger(writer, train_interval=args.log_interval)

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

    # trainer
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
    # # Let's watch its performance!
    # policy.eval()
    # policy.set_eps(args.eps_test)
    # test_envs.seed(args.seed)
    # # test_collector.reset()
    # train_collector.reset()
    # # result = test_collector.collect(n_episode=args.test_num, render=args.render)
    # result = train_collector.collect(n_episode=args.test_num, render=args.render)

    # rews, lens = result["rews"], result["lens"]
    # print(f"Final reward: {rews.mean()}, length: {lens.mean()}")