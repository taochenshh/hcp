import torch
import torch.nn as nn
import torch.nn.functional as F
from distribution import DiagNormal


class MLPPolicy(nn.Module):
    def __init__(self, in_dim, act_dim, hid1_dim,
                 hid2_dim, robot_num, embed_dim=32,
                 with_embed=False):
        super(MLPPolicy, self).__init__()
        self.with_embed = with_embed
        if self.with_embed:
            print('Embedding dim:', embed_dim)
            self.embedding = nn.Embedding(robot_num, embed_dim)
            # we minus the input dim by 1 here because
            # the last element in observation is the robot id,
            # which we removed before feeding ob to the net
            self.pol_net_hid1 = nn.Linear(in_dim + embed_dim - 1,
                                          hid1_dim)
            self.val_net_hid1 = nn.Linear(in_dim + embed_dim - 1,
                                          hid1_dim)
        else:
            self.pol_net_hid1 = nn.Linear(in_dim, hid1_dim)
            self.val_net_hid1 = nn.Linear(in_dim, hid1_dim)
        print('Network input dim:', self.pol_net_hid1.in_features)
        self.pol_net_hid2 = nn.Linear(hid1_dim, hid2_dim)
        self.pol_net_out = nn.Linear(hid2_dim, act_dim)

        self.val_net_hid2 = nn.Linear(hid1_dim, hid2_dim)
        self.val_net_out = nn.Linear(hid2_dim, 1)

        self.logstd = nn.Parameter(data=torch.zeros(1, act_dim),
                                   requires_grad=True)
        self.reset_parameters()

    def forward(self, ob, robot_id, name):
        if self.with_embed:
            embed_out = self.embedding(robot_id)
            ob = torch.cat((ob, embed_out), dim=-1)
        if name == 'pol':
            pol_out = F.selu(self.pol_net_hid1(ob))
            pol_out = F.selu(self.pol_net_hid2(pol_out))
            out = self.pol_net_out(pol_out)
        elif name == 'val':
            val_out = F.selu(self.val_net_hid1(ob))
            val_out = F.selu(self.val_net_hid2(val_out))
            out = self.val_net_out(val_out)
        else:
            raise ValueError('Unrecognized name: %s', name)
        return out

    def value(self, ob, robot_id):
        return self.forward(ob, robot_id, name='val')

    def policy(self, ob, robot_id):
        return self.forward(ob, robot_id, name='pol')

    def step_and_eval(self, ob, robot_id, act=None):
        pi = self.policy(ob, robot_id)
        val = torch.squeeze(self.value(ob, robot_id), dim=1)
        self.pd = DiagNormal(mean=pi, logstd=self.logstd)
        if act is None:
            act = self.pd.sample(sample_shape=pi.size())
        act_neglogp = self.pd.neglogp(act)
        entropy = self.pd.entropy()
        return act, act_neglogp, val, entropy

    def reset_parameters(self):
        for name, param in self.named_modules():
            if name.find('net_hid') != -1:
                nn.init.xavier_normal_(param.weight.data)
            elif name.find('pol_net_out') != -1:
                nn.init.xavier_normal_(param.weight.data)
                param.weight.data.mul_(0.01)
            elif name.find('val_net_out') != -1:
                nn.init.xavier_normal_(param.weight.data)
            if 'bias' in param.state_dict().keys():
                param.bias.data.fill_(0)
