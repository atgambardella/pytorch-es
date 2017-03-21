from __future__ import absolute_import, division, print_function

import os
import math
import numpy as np

import torch
import torch.nn.functional as F
from torch.autograd import Variable

from envs import create_atari_env
from model import ES


def do_rollouts(args, models, random_seeds, return_queue, env):
    all_returns = []
    all_num_frames = []
    for model in models:
        if not args.small_net:
            cx = Variable(torch.zeros(1, 256))
            hx = Variable(torch.zeros(1, 256))
        state = env.reset()
        state = torch.from_numpy(state)
        this_model_return = 0
        this_model_num_frames = 0
        # Rollout
        for step in range(args.max_episode_length):
            if args.small_net:
                state = state.float()
                state = state.view(1, env.observation_space.shape[0])
                logit = model(Variable(state, volatile=True))
            else:
                logit, (hx, cx) = model(
                    (Variable(state.unsqueeze(0), volatile=True),
                     (hx, cx)))

            prob = F.softmax(logit)
            action = prob.max(1)[1].data.numpy()
            state, reward, done, _ = env.step(action[0, 0])
            this_model_return += reward
            this_model_num_frames += 1
            if done:
                break
            state = torch.from_numpy(state)
        all_returns.append(this_model_return)
        all_num_frames.append(this_model_num_frames)
    return_queue.put((random_seeds, all_returns, all_num_frames))


def do_2n_rollouts(rank, args, synced_model, return_queue, env):
    def perturb_model(model, random_seed):
        """
        Modifies the given model with a pertubation of its parameters,
        as well as the negative perturbation, and returns both perturbed
        models.
        """
        new_model = ES(env.observation_space.shape[0],
                       env.action_space, args.small_net)
        anti_model = ES(env.observation_space.shape[0],
                        env.action_space, args.small_net)
        new_model.load_state_dict(model.state_dict())
        anti_model.load_state_dict(model.state_dict())
        np.random.seed(random_seed)
        for (k, v), (anti_k, anti_v) in zip(new_model.es_params(),
                                            anti_model.es_params()):
            eps = np.random.normal(0, 1, v.size())
            v += torch.from_numpy(args.sigma*eps).float()
            anti_v += torch.from_numpy(args.sigma*-eps).float()
        return [new_model, anti_model]

    unperturbed_model = ES(env.observation_space.shape[0],
                           env.action_space, args.small_net)
    unperturbed_model.load_state_dict(synced_model.state_dict())
    all_models = []
    returns = []
    seeds = []
    num_frames = []
    # Get 2*args.n perturbed models,
    # one for each perturbation and its negative.
    for i in range(args.n):
        np.random.seed()
        random_seed = np.random.randint(2**30)
        # Get 2 models from one seed so add the seed twice
        seeds.append(random_seed)
        seeds.append(random_seed)
        all_models += perturb_model(unperturbed_model, random_seed)
    do_rollouts(args, all_models, seeds, return_queue, env)


def gradient_update(args, synced_model, returns, random_seeds,
                    num_eps, num_frames, chkpt_dir, unperturbed_results):
    def fitness_shaping(returns):
        """
        A rank transformation on the rewards, which reduces the chances
        of falling into local optima early in training.
        """
        sorted_returns_backwards = sorted(returns)[::-1]
        lamb = len(returns)
        shaped_returns = []
        denom = sum([max(0, math.log2(lamb/2 + 1) -
                         math.log2(sorted_returns_backwards.index(r) + 1))
                     for r in returns])
        for r in returns:
            num = max(0, math.log2(lamb/2 + 1) -
                      math.log2(sorted_returns_backwards.index(r) + 1))
            shaped_returns.append(num/denom + 1/lamb)
        return shaped_returns

    def unperturbed_rank(returns, unperturbed_results):
        nth_place = 1
        for r in returns:
            if r > unperturbed_results:
                nth_place += 1
        rank_diag = '%d out of %d (1 is bad %d is best)' % (nth_place,
                                                            len(returns) + 1,
                                                            len(returns) + 1)
        return rank_diag, nth_place

    batch_size = len(returns)
    assert batch_size == 2*args.n*args.num_processes
    assert len(random_seeds) == batch_size
    shaped_returns = fitness_shaping(returns)
    rank_diag, rank = unperturbed_rank(returns, unperturbed_results)
    if not args.silent:
        print('Episode num: %d\n'
              'Average reward: %f\n'
              'Variance in rewards: %f\n'
              'Batch size: %d\n'
              'Max episode length %d\n'
              'Sigma: %f\n'
              'Learning rate: %f\n'
              'Total num frames seen: %d\n'
              'Unperturbed reward: %f\n'
              'Unperturbed rank: %s\n\n' %
              (num_eps, np.mean(returns), np.var(returns), batch_size,
               args.max_episode_length, args.sigma, args.lr, num_frames,
               unperturbed_results, rank_diag))
    # For each model, generate the same random numbers as we did
    # before, and update parameters. We apply weight decay once.
    for i in range(args.n*args.num_processes):
        np.random.seed(random_seeds[2*i])
        for k, v in synced_model.es_params():
            eps = np.random.normal(0, 1, v.size())
            v += torch.from_numpy(args.lr/(batch_size*args.sigma) *
                                  (shaped_returns[2*i]*eps)).float()
            v += torch.from_numpy(args.lr/(batch_size*args.sigma) *
                                  (shaped_returns[2*i + 1]*-eps)).float()
    for k, v in synced_model.es_params():
        v *= args.wd
    args.lr *= args.lr_decay
    torch.save(synced_model.state_dict(),
               os.path.join(chkpt_dir, 'latest.pth'))


def render_env(args, model, env):
    while True:
        state = env.reset()
        state = torch.from_numpy(state)
        this_model_return = 0
        if not args.small_net:
            cx = Variable(torch.zeros(1, 256))
            hx = Variable(torch.zeros(1, 256))
        done = False
        while not done:
            if args.small_net:
                state = state.float()
                state = state.view(1, env.observation_space.shape[0])
                logit = model(Variable(state, volatile=True))
            else:
                logit, (hx, cx) = model(
                    (Variable(state.unsqueeze(0), volatile=True),
                     (hx, cx)))

            prob = F.softmax(logit)
            action = prob.max(1)[1].data.numpy()
            state, reward, done, _ = env.step(action[0, 0])
            env.render()
            this_model_return += reward
            state = torch.from_numpy(state)
        print('Reward: %f' % this_model_return)
