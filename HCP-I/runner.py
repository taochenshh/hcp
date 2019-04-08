import numpy as np


class Runner(object):
    def __init__(self, env, model, nsteps, gamma, lam):
        self.env = env
        self.model = model
        self.nenv = env.num_envs
        self.gamma = gamma
        self.lam = lam
        self.nsteps = nsteps

    def test(self, train=False):
        """
        :param train: whether test on the training robots
        """
        cond_now = 0
        env_step_nums = []
        env_rewards = []
        while True:
            idx = np.arange(cond_now * self.nenv, (cond_now + 1) * self.nenv)
            if train:
                obs = self.env.train_test_reset(conds=idx)
            else:
                obs = self.env.test_reset(conds=idx)
            if np.any(np.isnan(obs)):
                break
            done_step_sub = [-1 for i in range(self.nenv)]
            done_reward_sub = [0.0 for i in range(self.nenv)]
            reward_add = [True for i in range(self.nenv)]
            for step_num in range(self.nsteps):
                res = self.model.step(obs, sample=False)
                actions, _, _, _ = res
                obs, rewards, dones, infos = self.env.step(actions)
                for reward_id in range(rewards.size):
                    if reward_add[reward_id]:
                        done_reward_sub[reward_id] += rewards[reward_id]
                for done_id in range(dones.size):
                    if dones[done_id] and done_step_sub[done_id] == -1:
                        done_step_sub[done_id] = step_num
                        reward_add[done_id] = False
                if np.all(dones):
                    break
            for done_id in range(dones.size):
                if done_step_sub[done_id] == -1:
                    done_step_sub[done_id] = step_num
            env_step_nums.extend(done_step_sub)
            env_rewards.extend(done_reward_sub)
            cond_now += 1
        return np.mean(env_step_nums), np.mean(env_rewards)

    def run(self):
        mb_obs, mb_rewards, mb_actions = [], [], []
        mb_values, mb_dones, mb_acts_neglog = [], [], []
        epinfos = [{'reward': 0.0, 'steps': 0.0}
                   for i in range(self.env.num_envs)]
        # epinfos = []
        dones = np.zeros(self.nenv, dtype=np.bool)
        obs = self.env.reset()
        book_keeping = np.ones(self.nenv, dtype=np.bool)
        for _ in range(self.nsteps):
            res = self.model.step(obs, sample=True)
            actions, acts_neglogp, values, entropy = res
            mb_obs.append(obs.copy())
            mb_actions.append(actions)
            mb_values.append(values)
            mb_acts_neglog.append(acts_neglogp)
            obs, rewards, dones, infos = self.env.step(actions)
            for idx, info in enumerate(infos):
                if book_keeping[idx]:
                    if not dones[idx]:
                        epinfos[idx]['reward'] += info['reward']
                        epinfos[idx]['steps'] += info['step']
                    else:
                        book_keeping[idx] = False
            mb_dones.append(dones)
            mb_rewards.append(rewards)

        mb_obs = np.asarray(mb_obs, dtype=obs.dtype)
        mb_rewards = np.asarray(mb_rewards, dtype=np.float32)
        mb_actions = np.asarray(mb_actions)
        mb_values = np.asarray(mb_values, dtype=np.float32)
        mb_acts_neglog = np.asarray(mb_acts_neglog, dtype=np.float32)
        mb_dones = np.asarray(mb_dones, dtype=np.bool)
        last_values = self.model.value(obs)

        mb_advs = np.zeros_like(mb_rewards)
        lastgaelam = 0
        for t in reversed(range(self.nsteps)):
            if t == self.nsteps - 1:
                nextnonterminal = 1.0 - dones
                nextvalues = last_values
            else:
                nextnonterminal = 1.0 - mb_dones[t + 1]
                nextvalues = mb_values[t + 1]
            delta = mb_rewards[t]
            delta += self.gamma * nextvalues.flatten() * nextnonterminal
            delta -= mb_values[t]
            tmp = delta
            tmp += self.gamma * self.lam * nextnonterminal * lastgaelam
            lastgaelam = tmp
            mb_advs[t] = lastgaelam
        mb_returns = mb_advs + mb_values
        arrs = (mb_obs, mb_returns, mb_dones,
                mb_actions, mb_values, mb_acts_neglog)
        return (*map(sf01, arrs), epinfos)


def sf01(arr):
    """
    swap and then flatten axes 0 and 1
    """
    s = arr.shape
    return arr.swapaxes(0, 1).reshape(s[0] * s[1], *s[2:])
