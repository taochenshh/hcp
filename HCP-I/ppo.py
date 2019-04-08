import time
from collections import deque

import numpy as np
from runner import Runner

from model import PPOModel
from mutils import explained_variance, safemean
from util import logger


class PPO:
    def __init__(self, policy, env, nsteps, num_iters,
                 ent_coef, lr, hid1_dim, hid2_dim,
                 weight_decay, save_dir,
                 vf_coef=0.5, max_grad_norm=0.5,
                 gamma=0.99, lam=0.95, log_interval=10,
                 nminibatches=4, noptepochs=4, cliprange=0.2,
                 save_interval=0, resume=False, resume_step=None,
                 test=False, ob_rms=False, pretrain_dir=None,
                 embed_dim=32, robot_num=0, with_embed=True):
        if not test:
            nenvs = env.num_envs
            self.nbatch = nenvs * nsteps
            self.nbatch_train = self.nbatch // nminibatches

            assert self.nbatch % nminibatches == 0
            self.noptepochs = noptepochs
            self.log_interval = log_interval
            self.save_interval = save_interval
            self.num_iters = num_iters
        self.with_embed = with_embed
        self.nsteps = nsteps
        self.env = env
        ob_space = env.observation_space
        ac_space = env.action_space
        self.model = PPOModel(policy=policy,
                              ob_space=ob_space.shape[0],
                              ac_space=ac_space.shape[0],
                              ent_coef=ent_coef,
                              vf_coef=vf_coef,
                              max_grad_norm=max_grad_norm,
                              cliprange=cliprange,
                              hid1_dim=hid1_dim,
                              hid2_dim=hid2_dim,
                              weight_decay=weight_decay,
                              lr=lr,
                              save_dir=save_dir,
                              ob_rms=ob_rms,
                              robot_num=robot_num,
                              embed_dim=embed_dim,
                              with_embed=with_embed)
        self.ob_rms = ob_rms

        if resume or test or pretrain_dir is not None:
            self.model.load_model(step=resume_step,
                                  pretrain_dir=pretrain_dir)
        if not test:
            self.runner = Runner(env=env, model=self.model,
                                 nsteps=nsteps, gamma=gamma, lam=lam)

    def train(self):
        epinfobuf = deque(maxlen=20)
        tfirststart = time.time()

        for update in range(self.num_iters):
            tstart = time.time()
            res = self.runner.run()
            obs, returns, dones, actions, values, acts_neglog, epinfos = res
            if self.ob_rms:
                self.model.ob_rms.update(obs)
            epinfobuf.extend(epinfos)
            lossvals = {'policy_loss': [],
                        'value_loss': [],
                        'policy_entropy': [],
                        'approxkl': [],
                        'clipfrac': []}

            inds = np.arange(self.nbatch)
            for _ in range(self.noptepochs):
                np.random.shuffle(inds)
                for start in range(0, self.nbatch, self.nbatch_train):
                    end = start + self.nbatch_train
                    mbinds = inds[start:end]
                    slices = (arr[mbinds]
                              for arr in (obs, actions, returns,
                                          acts_neglog, values))
                    info = self.model.train(*slices)
                    lossvals['policy_loss'].append(info['pg_loss'])
                    lossvals['value_loss'].append(info['vf_loss'])
                    lossvals['policy_entropy'].append(info['entropy'])
                    lossvals['approxkl'].append(info['approxkl'])
                    lossvals['clipfrac'].append(info['clipfrac'])

            tnow = time.time()
            fps = int(self.nbatch / (tnow - tstart))
            if update % self.log_interval == 0:
                ev = explained_variance(values, returns)
                logger.logkv("Learning rate",
                             self.model.optimizer.param_groups[0]['lr'])
                logger.logkv("serial_timesteps",
                             update * self.nsteps)
                logger.logkv("nupdates", update)
                logger.logkv("total_timesteps",
                             update * self.nbatch)
                logger.logkv("fps", fps)
                logger.logkv("explained_variance", float(ev))
                logger.logkv('eprewmean',
                             safemean([epinfo['reward']
                                       for epinfo in epinfobuf]))
                logger.logkv('eplenmean',
                             safemean([epinfo['steps']
                                       for epinfo in epinfobuf]))
                logger.logkv('time_elapsed', tnow - tfirststart)
                for name, value in lossvals.items():
                    logger.logkv(name, np.mean(value))
                logger.dumpkvs()
            if self.save_interval and \
                    update % self.save_interval == 0 and \
                    logger.get_dir():
                self.model.log_model_weights()
                avg_steps, avg_reward = self.runner.test()
                logger.logkv("nupdates", update)
                logger.logkv("test/total_timesteps", update * self.nbatch)
                logger.logkv('test/step', avg_steps)
                logger.logkv('test/reward', avg_reward)
                if not self.with_embed:
                    res = self.runner.test(train=True)
                    train_avg_steps, train_avg_reward = res
                    logger.logkv('train/step', train_avg_steps)
                    logger.logkv('train/reward', train_avg_reward)
                logger.dumpkvs()
                if avg_reward > self.model.best_rewards:
                    self.model.best_rewards = avg_reward
                    is_best = True
                else:
                    is_best = False
                self.model.save_model(is_best=is_best, step=update)
        self.env.close()

    def test(self, render):
        env_steps = []
        env_rewards = []
        cond_now = 0
        while True:
            ob = self.env.test_reset(conds=[cond_now])
            if np.any(np.isnan(ob)):
                break
            sum_reward = 0
            for st in range(self.nsteps):
                action, _, _, _ = self.model.step(ob, sample=False)
                ob, reward, done, info = self.env.step(action)
                sum_reward += reward
                if render:
                    self.env.render()
                if done:
                    break
            print('Step: ', st)
            env_rewards.append(sum_reward)
            env_steps.append(st)
            cond_now += 1
        import json
        with open('./data.json', 'w') as f:
            json.dump(env_rewards, f)
