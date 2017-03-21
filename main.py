from __future__ import absolute_import, division, print_function

import os
import argparse

import torch
import torch.multiprocessing as mp

from envs import create_atari_env
from model import ES
from train import do_2n_rollouts, do_rollouts, gradient_update, render_env

parser = argparse.ArgumentParser(description='ES')
parser.add_argument('--env-name', default='PongDeterministic-v3',
                    metavar='ENV', help='environment')
parser.add_argument('--lr', type=float, default=3e-3, metavar='LR',
                    help='learning rate')
parser.add_argument('--lr-decay', type=float, default=1, metavar='LRD',
                    help='learning rate decay')
parser.add_argument('--sigma', type=float, default=0.05, metavar='SD',
                    help='noise standard deviation')
parser.add_argument('--wd', type=float, default=0.996, metavar='WD',
                    help='amount of weight decay')
parser.add_argument('--num-processes', type=int, default=10,
                    metavar='NP', help='how many training processes to use')
parser.add_argument('--n', type=int, default=1, metavar='N',
                    help='number of perturbations per process, per update')
parser.add_argument('--max-episode-length', type=int, default=100000,
                    metavar='MEL', help='maximum length of an episode')
parser.add_argument('--max-gradient-updates', type=int, default=1000,
                    metavar='MGU', help='maximum number of updates')
parser.add_argument('--restore', default='', metavar='RES',
                    help='checkpoint from which to restore')
parser.add_argument('--small-net', action='store_true',
                    help='Use simple MLP on CartPole')
parser.add_argument('--variable-ep-len', action='store_true',
                    help="Change max episode length during training")
parser.add_argument('--silent', action='store_true',
                    help='Silence print statements during training')
parser.add_argument('--test', action='store_true',
                    help='Just render the env, no training')


def flatten(raw_results, index):
    notflat_results = [result[index] for result in raw_results]
    return [item for sublist in notflat_results for item in sublist]


def find_best(chkpt_dir):
    rewards = [float(s) for s in os.listdir(chkpt_dir)]
    return '{:.6f}'.format(max(rewards))

if __name__ == '__main__':
    args = parser.parse_args()

    if args.small_net and args.env_name not in ['CartPole-v0', 'CartPole-v1',
                                                'MountainCar-v0']:
        args.env_name = 'CartPole-v1'
        print('Switching env to CartPole')

    env = create_atari_env(args.env_name)
    chkpt_dir = 'checkpoints/%s/' % args.env_name
    if not os.path.exists(chkpt_dir):
        os.makedirs(chkpt_dir)
    synced_model = ES(env.observation_space.shape[0],
                      env.action_space, args.small_net)
    for param in synced_model.parameters():
        param.requires_grad = False
    if args.restore:
        state_dict = torch.load(args.restore)
        synced_model.load_state_dict(state_dict)

    if args.test:
        render_env(args, synced_model, env)

    print("Num params in network %d" % synced_model.count_parameters())
    num_eps = 0
    total_num_frames = 0
    for _ in range(args.max_gradient_updates):
        processes = []
        return_queue = mp.Queue()
        for rank in range(0, args.num_processes):
            p = mp.Process(target=do_2n_rollouts, args=(rank, args,
                                                        synced_model,
                                                        return_queue, env))
            p.start()
            processes.append(p)
        p = mp.Process(target=do_rollouts, args=(args, [synced_model], [-1],
                                                 return_queue, env))
        p.start()
        processes.append(p)
        for p in processes:
            p.join()
        raw_results = [return_queue.get() for p in processes]
        seeds, results, num_frames = [flatten(raw_results, index)
                                      for index in [0, 1, 2]]
        # Separate the unperturbed model from the perturbed models
        unperturbed_index = seeds.index(-1)
        seeds.pop(unperturbed_index)
        unperturbed_results = results.pop(unperturbed_index)
        num_frames.pop(unperturbed_index)
        total_num_frames += sum(num_frames)
        num_eps += len(results)
        gradient_update(args, synced_model, results, seeds, num_eps,
                        total_num_frames, chkpt_dir, unperturbed_results)
        if args.variable_ep_len:
            args.max_episode_length = int(2*sum(num_frames)/len(num_frames))
