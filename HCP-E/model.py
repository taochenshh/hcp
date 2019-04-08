import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


def fanin_init(size, fanin=None, init_method='normal'):
    fanin = fanin or size[0]
    v = 1. / np.sqrt(fanin)
    if init_method == 'uniform':
        return torch.Tensor(size).uniform_(-v, v)
    else:
        return torch.Tensor(size).normal_(0, v)


class Actor(nn.Module):
    def __init__(self, ob_dim, act_dim,
                 hid1_dim=400, hid2_dim=300, hid3_dim=300,
                 init_method='normal'):
        super(Actor, self).__init__()
        self.pol_net_hid1 = nn.Linear(ob_dim, hid1_dim)
        print('model ob dim:', self.pol_net_hid1.in_features)
        self.pol_net_hid2 = nn.Linear(hid1_dim, hid2_dim)
        self.pol_net_hid3 = nn.Linear(hid2_dim, hid3_dim)
        self.pol_net_out = nn.Linear(hid3_dim, act_dim)
        self.reset_parameters(init_w=0.003, init_method=init_method)

    def forward(self, ob):
        pol_out = F.selu(self.pol_net_hid1(ob))
        pol_out = F.selu(self.pol_net_hid2(pol_out))
        pol_out = F.selu(self.pol_net_hid3(pol_out))
        out = torch.tanh(self.pol_net_out(pol_out))
        return out

    def reset_parameters(self, init_w, init_method):
        w_size = self.pol_net_hid1.weight.data.size()
        self.pol_net_hid1.weight.data = fanin_init(w_size,
                                                   init_method=init_method)
        w_size = self.pol_net_hid2.weight.data.size()
        self.pol_net_hid2.weight.data = fanin_init(w_size,
                                                   init_method=init_method)
        w_size = self.pol_net_hid3.weight.data.size()
        self.pol_net_hid3.weight.data = fanin_init(w_size,
                                                   init_method=init_method)
        if init_method == 'uniform':
            self.pol_net_out.weight.data.uniform_(-init_w, init_w)
        else:
            self.pol_net_out.weight.data.normal_(0, init_w)


class Critic(nn.Module):
    def __init__(self, ob_dim, act_dim,
                 hid1_dim, hid2_dim, hid3_dim,
                 init_method='normal'):
        super(Critic, self).__init__()
        self.val_net_hid1 = nn.Linear(ob_dim, hid1_dim)
        self.val_net_hid2 = nn.Linear(hid1_dim + act_dim, hid2_dim)
        self.val_net_hid3 = nn.Linear(hid2_dim, hid3_dim)
        self.val_net_out = nn.Linear(hid3_dim, 1)
        self.reset_parameters(init_w=0.003, init_method=init_method)

    def forward(self, ob, act):
        val_out = F.selu(self.val_net_hid1(ob))
        val_out = torch.cat([val_out, act], dim=1)
        val_out = F.selu(self.val_net_hid2(val_out))
        val_out = F.selu(self.val_net_hid3(val_out))
        out = self.val_net_out(val_out)
        return out

    def reset_parameters(self, init_w, init_method):
        w_size = self.val_net_hid1.weight.data.size()
        self.val_net_hid1.weight.data = fanin_init(w_size,
                                                   init_method=init_method)
        w_size = self.val_net_hid2.weight.data.size()
        self.val_net_hid2.weight.data = fanin_init(w_size,
                                                   init_method=init_method)
        w_size = self.val_net_hid3.weight.data.size()
        self.val_net_hid3.weight.data = fanin_init(w_size,
                                                   init_method=init_method)
        if init_method == 'uniform':
            self.val_net_out.weight.data.uniform_(-init_w, init_w)
        else:
            self.val_net_out.weight.data.normal_(0, init_w)
