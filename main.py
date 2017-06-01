from __future__ import absolute_import, division, print_function

import os
import argparse

import torch

from envs import create_atari_env
from model import ES
from train import train_loop, render_env

parser = argparse.ArgumentParser(description='ES')
parser.add_argument('--env-name', default='PongDeterministic-v4',
                    metavar='ENV', help='environment')
parser.add_argument('--lr', type=float, default=0.3, metavar='LR',
                    help='learning rate')
parser.add_argument('--lr-decay', type=float, default=1, metavar='LRD',
                    help='learning rate decay')
parser.add_argument('--sigma', type=float, default=0.05, metavar='SD',
                    help='noise standard deviation')
parser.add_argument('--useAdam', action='store_true',
                    help='bool to determine if to use adam optimizer')
parser.add_argument('--n', type=int, default=40, metavar='N',
                    help='batch size, must be even')
parser.add_argument('--max-episode-length', type=int, default=100000,
                    metavar='MEL', help='maximum length of an episode')
parser.add_argument('--max-gradient-updates', type=int, default=100000,
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



if __name__ == '__main__':
    args = parser.parse_args()
    assert args.n % 2 == 0
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
    else:
        train_loop(args, synced_model, env, chkpt_dir)
