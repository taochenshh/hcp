import os

import numpy as np
import torch
from torch import optim
from torch.autograd import Variable

import mutils
from util import logger
from util.running_mean_std import RunningMeanStd


class PPOModel(object):
    def __init__(self, policy, ob_space, ac_space, ent_coef,
                 vf_coef, max_grad_norm, cliprange, hid1_dim,
                 hid2_dim, weight_decay, lr, save_dir, ob_rms,
                 robot_num, embed_dim, with_embed):
        self.with_embed = with_embed
        self.cliprange = cliprange
        self.ent_coef = ent_coef
        self.vf_coef = vf_coef
        self.max_grad_norm = max_grad_norm
        self.model = policy(in_dim=ob_space,
                            act_dim=ac_space,
                            hid1_dim=hid1_dim,
                            hid2_dim=hid2_dim,
                            robot_num=robot_num,
                            embed_dim=embed_dim,
                            with_embed=with_embed)
        self.model.cuda()
        self.optimizer = self.get_optimizer(weight_decay, lr)
        self.global_step = 0
        self.model_dir = os.path.join(save_dir, 'model')
        os.makedirs(self.model_dir, exist_ok=True)
        self.best_step = 0
        self.best_rewards = -np.inf
        if ob_rms:
            self.ob_rms = RunningMeanStd(shape=ob_space)
        else:
            self.ob_rms = None

    def normalize_ob(self, ob):
        if self.ob_rms is not None:
            ob = np.clip((ob - self.ob_rms.mean) /
                         np.sqrt(self.ob_rms.var + 1e-8),
                         -5.0, 5.0)
        return ob

    def step(self, obs, sample=True):
        self.model.eval()
        if self.with_embed:
            ob = obs[:, :-1]
            robot_id = obs[:, -1]
            robot_id = Variable(torch.from_numpy(robot_id)).long().cuda()
        else:
            ob = obs
            robot_id = None
        ob = self.normalize_ob(ob)
        ob = Variable(torch.from_numpy(ob)).float().cuda()

        act_neglogp = None
        val = None
        entropy = None
        if sample:
            res = self.model.step_and_eval(ob, robot_id)
            act, act_neglogp, val, entropy = res
            act = act.cpu().data.numpy()
            act_neglogp = act_neglogp.cpu().data.numpy()
            val = val.cpu().data.numpy()
            entropy = entropy.cpu().data.numpy()
        else:
            act = self.model.policy(ob, robot_id)
            act = act.cpu().data.numpy()
        return act, act_neglogp, val, entropy

    def value(self, obs):
        if self.with_embed:
            ob = obs[:, :-1]
            robot_id = obs[:, -1]
            robot_id = Variable(torch.from_numpy(robot_id)).long().cuda()
        else:
            ob = obs
            robot_id = None
        ob = self.normalize_ob(ob)
        ob = Variable(torch.from_numpy(ob)).float().cuda()
        return self.model.value(ob, robot_id).cpu().data.numpy()

    def train(self, obs, actions, returns, old_acts_neglogp, values):
        self.model.train()
        if self.with_embed:
            ob = obs[:, :-1]
            robot_id = obs[:, -1]
            robot_id = Variable(torch.from_numpy(robot_id)).long().cuda()
        else:
            ob = obs
            robot_id = None
        ob = self.normalize_ob(ob)
        ob = Variable(torch.from_numpy(ob)).float().cuda()
        actions = Variable(torch.from_numpy(actions)).float().cuda()
        returns = Variable(torch.from_numpy(returns)).float().cuda()
        old_acts_neglogp = Variable(torch.from_numpy(old_acts_neglogp))
        old_acts_neglogp = old_acts_neglogp.float().cuda()
        values = Variable(torch.from_numpy(values)).float().cuda()
        advs = returns - values
        advs = (advs - advs.mean()) / (advs.std() + 1e-8)
        res = self.model.step_and_eval(ob, robot_id, actions)
        _, acts_neglogp, vals_pred, entropy = res

        val_pred_clipped = values + torch.clamp(vals_pred - values,
                                                min=-self.cliprange,
                                                max=self.cliprange)

        vf_loss1 = torch.pow(vals_pred - returns, 2)
        vf_loss2 = torch.pow(val_pred_clipped - returns, 2)
        vf_loss = 0.5 * torch.mean(torch.max(vf_loss1, vf_loss2))

        ratio = torch.exp(old_acts_neglogp - acts_neglogp)
        pg_loss1 = -advs * ratio
        pg_loss2 = -advs * torch.clamp(ratio,
                                       1 - self.cliprange,
                                       1 + self.cliprange)
        pg_loss = torch.mean(torch.max(pg_loss1, pg_loss2))

        loss = pg_loss - entropy * self.ent_coef + vf_loss * self.vf_coef

        approxkl = 0.5 * torch.mean(torch.pow(acts_neglogp -
                                              old_acts_neglogp,
                                              2))
        clipfrac = np.mean(np.abs(ratio.cpu().data.numpy()
                                  - 1.0) > self.cliprange)

        self.optimizer.zero_grad()
        loss.backward()
        if self.max_grad_norm is not None:
            torch.nn.utils.clip_grad_norm_(self.model.parameters(),
                                           self.max_grad_norm)
        self.optimizer.step()
        self.global_step += 1
        return {'pg_loss': pg_loss.cpu().data.numpy(),
                'vf_loss': vf_loss.cpu().data.numpy(),
                'entropy': entropy.cpu().data.numpy(),
                'approxkl': approxkl.cpu().data.numpy(),
                'clipfrac': clipfrac}

    def load_model(self, step=None, pretrain_dir=None):
        if pretrain_dir is not None:
            ckpt_file = os.path.join(pretrain_dir, 'model_best.pth')
        else:
            if step is None:
                ckpt_file = os.path.join(self.model_dir, 'model_best.pth')
            else:
                ckpt_file = os.path.join(self.model_dir,
                                         'ckpt_{:08d}.pth'.format(step))
        if not os.path.isfile(ckpt_file):
            raise ValueError("No checkpoint found at '{}'".format(ckpt_file))
        mutils.print_yellow('Loading checkpoint {}'.format(ckpt_file))
        checkpoint = torch.load(ckpt_file)
        if step is None:
            mutils.print_yellow('Checkpoint step: {}'
                                ''.format(checkpoint['ckpt_step']))

        # if pretrain_dir is not None (i.e., we have new robots here)
        if pretrain_dir is not None:
            model_dict = self.model.state_dict()
            pretrained_dict = {k: v for k, v
                               in checkpoint['state_dict'].items()
                               if k in model_dict and
                               'embedding' not in k}
            model_dict.update(pretrained_dict)
            self.model.load_state_dict(model_dict)
        else:
            self.global_step = checkpoint['global_step']
            self.model.load_state_dict(checkpoint['state_dict'])
            self.optimizer.load_state_dict(checkpoint['optimizer'])
        for name, param in self.model.named_parameters():
            print(name, param.requires_grad)

        if self.ob_rms:
            self.ob_rms.mean = checkpoint['ob_rms_mean']
            self.ob_rms.var = checkpoint['ob_rms_var']
        mutils.print_yellow('Checkpoint loaded...')

    def log_model_weights(self):
        for name, param in self.model.named_parameters():
            logger.logkv(name, param.clone().cpu().data.numpy())

    def save_model(self, is_best, step=None):
        if step is None:
            step = self.global_step
        ckpt_file = os.path.join(self.model_dir,
                                 'ckpt_{:08d}.pth'.format(step))
        data_to_save = {
            'ckpt_step': step,
            'global_step': self.global_step,
            'state_dict': self.model.state_dict(),
            'optimizer': self.optimizer.state_dict()
        }
        if self.ob_rms:
            data_to_save['ob_rms_mean'] = self.ob_rms.mean
            data_to_save['ob_rms_var'] = self.ob_rms.var

        mutils.print_yellow('Saving checkpoint: %s' % ckpt_file)
        torch.save(data_to_save, ckpt_file)
        if is_best:
            torch.save(data_to_save, os.path.join(self.model_dir,
                                                  'model_best.pth'))

    def get_optimizer(self, weight_decay, lr):
        params = mutils.add_weight_decay([self.model],
                                         weight_decay=weight_decay)
        optimizer = optim.Adam(params,
                               lr=lr,
                               weight_decay=weight_decay)
        return optimizer
