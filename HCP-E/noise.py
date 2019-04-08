import random

import numpy as np


class AdaptiveParamNoiseSpec(object):
    def __init__(self, initial_stddev=0.1,
                 desired_action_stddev=0.1,
                 adoption_coefficient=1.01):
        self.initial_stddev = initial_stddev
        self.desired_action_stddev = desired_action_stddev
        self.adoption_coefficient = adoption_coefficient

        self.current_stddev = initial_stddev

    def adapt(self, distance):
        if distance > self.desired_action_stddev:
            # Decrease stddev.
            self.current_stddev /= self.adoption_coefficient
        else:
            # Increase stddev.
            self.current_stddev *= self.adoption_coefficient

    def get_stats(self):
        stats = {
            'param_noise_stddev': self.current_stddev,
        }
        return stats

    def __repr__(self):
        fmt = 'AdaptiveParamNoiseSpec(initial_stddev={}, ' \
              'desired_action_stddev={}, adoption_coefficient={})'
        return fmt.format(self.initial_stddev,
                          self.desired_action_stddev,
                          self.adoption_coefficient)


class ActionNoise(object):
    def reset(self):
        pass


class NormalActionNoise(ActionNoise):
    def __init__(self, mu, sigma):
        self.mu = mu
        self.sigma = sigma

    def __call__(self, val):
        return val + np.random.normal(self.mu, self.sigma)

    def __repr__(self):
        return 'NormalActionNoise(mu={}, sigma={})'.format(self.mu, self.sigma)


# Based on http://math.stackexchange.com/questions/1287634/
# implementing-ornstein-uhlenbeck-in-matlab
class OrnsteinUhlenbeckActionNoise(ActionNoise):
    def __init__(self, mu, sigma, theta=.15, dt=1e-2, x0=None):
        self.theta = theta
        self.mu = mu
        self.sigma = sigma
        self.dt = dt
        self.x0 = x0
        self.reset()

    def __call__(self, val):
        ran = np.random.normal(size=self.mu.shape)
        x = self.x_prev + \
            self.theta * (self.mu - self.x_prev) * self.dt + \
            self.sigma * np.sqrt(self.dt) * ran
        self.x_prev = x
        val += x
        return val

    def reset(self):
        self.x_prev = self.x0 if self.x0 is not None \
            else np.zeros_like(self.mu)

    def __repr__(self):
        return 'OrnsteinUhlenbeckActionNoise(mu={},' \
               ' sigma={})'.format(self.mu, self.sigma)


class UniformNoise(ActionNoise):
    def __init__(self, low_limit, high_limit, dec_step):
        self.low_limit = low_limit
        self.high_limit = high_limit
        self.dec_step = dec_step
        self.noise_level = random.uniform(self.low_limit, self.high_limit)

    def __call__(self, val):
        noise_val = np.random.normal(size=val.shape)
        val = val * (1 - self.noise_level) + noise_val * self.noise_level
        return val

    def reset(self):
        self.noise_level = random.uniform(self.low_limit, self.high_limit)
        self.high_limit -= self.dec_step
        self.high_limit = max(self.high_limit, self.low_limit)

    def __repr__(self):
        return 'UniformNoise(low_limit={}, ' \
               'high_limit={}, dec_step={})'.format(self.low_limit,
                                                    self.high_limit,
                                                    self.dec_step)
