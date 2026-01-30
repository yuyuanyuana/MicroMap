import torch
import torch.nn.functional as F
from torch import nn
import os
import numpy as np



# ============================================================================ #
# activation function

exp_act = lambda x: torch.exp(x)

def acti_fun(name):
    if name == "relu":
        return F.relu
    elif name == "leakyrelu":
        return F.leaky_relu
    elif name == "silu" or name == "swish":
        return F.silu
    elif name == "tanh":
        return torch.tanh
    elif name == "softplus":
        return F.softplus
    elif name == "gelu":
        return F.gelu
    elif name == 'exp':
        return exp_act


# ============================================================================ #
# DSBatchNorm

class DSBatchNorm(nn.Module):
    def __init__(self, num_features, n_domain, eps=1e-5, momentum=0.1):
        super().__init__()
        self.n_domain = n_domain
        self.num_features = num_features
        self.bns = nn.ModuleList([nn.BatchNorm1d(num_features, eps=eps, momentum=momentum) for i in range(n_domain)])
    def reset_running_stats(self):
        for bn in self.bns:
            bn.reset_running_stats()        
    def reset_parameters(self):
        for bn in self.bns:
            bn.reset_parameters()     
    def _check_input_dim(self, input):
        raise NotImplementedError   
    def forward(self, x, y=None):
        out = torch.zeros(x.size(0), self.num_features, device=x.device) #, requires_grad=False)
        for i in range(self.n_domain):
            indices = np.where(y.cpu().numpy()==i)[0]
            if len(indices) > 1:
                out[indices] = self.bns[i](x[indices])
            elif len(indices) == 1:
                out[indices] = x[indices]
        return out



# ============================================================================ #
# layer

class FC_Layer(nn.Module): 
    def __init__(self, in_features, out_features, bn=False, nbatch=1, activate='relu', dropout=0.):
        super(FC_Layer, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.bn = bn
        self.activate = activate
        self.dropout = dropout
        self.layer = nn.Linear(in_features, out_features)
        self.bn_layer = nn.BatchNorm1d(out_features)
        self.dsbn_layer = DSBatchNorm(out_features, nbatch)
        # layer = nn.Linear(2048,128)
        # bn_layer = nn.BatchNorm1d(128)
        # dsbn_layer = DSBatchNorm(128, nbatch)
    def forward(self, x, y=None):
        x = self.layer(x)
        if self.bn=='DSB':
            x = self.dsbn_layer(x, y)
        elif self.bn=='bn':
            x = self.bn_layer(x)
        if self.dropout!=0:
            return F.dropout(acti_fun(self.activate)(x), p=self.dropout, training=self.training)
        return acti_fun(self.activate)(x)       
    def __repr__(self):
        return self.__class__.__name__ + ' (' + str(self.in_features) + ' -> ' + str(self.out_features) + ')'



# ============================================================================ #
# model

class SpotPriorNet(nn.Module):
    def __init__(self, input_dim=100, hidden_dim=64, latent_dim=32):
        super(SpotPriorNet, self).__init__()
        # Encoder
        self.encoder_fc = nn.Linear(input_dim, hidden_dim)
        self.fc_mu = nn.Linear(hidden_dim, latent_dim)
        self.fc_logvar = nn.Linear(hidden_dim, latent_dim)
        # Decoder
        self.decoder_fc1 = nn.Linear(latent_dim, hidden_dim)
        self.decoder_out = nn.Linear(hidden_dim, input_dim)  # Reconstruction target = PCA 输入维度
    def encode(self, x):
        h = torch.relu(self.encoder_fc(x))
        mu = self.fc_mu(h)
        logvar = self.fc_logvar(h)
        return mu, logvar
    def reparameterize(self, mu, logvar):
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std
    def decode(self, z):
        h = torch.relu(self.decoder_fc1(z))
        recon_x = self.decoder_out(h)
        return recon_x
    def forward(self, x):
        mu, logvar = self.encode(x)
        z = self.reparameterize(mu, logvar)
        recon_x = self.decode(z)
        return recon_x, mu, logvar




class Token2Expr(nn.Module):
    def __init__(self, input_dim, hidden_dim, latent_dim, output_dim, n_batch=1, dropout=0.2):
        super(Token2Expr, self).__init__()
        self.layer1 = FC_Layer(input_dim, hidden_dim, bn='bn', activate='gelu', dropout=dropout)
        self.layer2 = FC_Layer(hidden_dim, latent_dim, bn='bn', activate='gelu', dropout=dropout)
        self.fc_mu = nn.Linear(latent_dim, latent_dim)
        self.fc_log_var = nn.Linear(latent_dim, latent_dim)
        self.layer3 = FC_Layer(latent_dim, hidden_dim, nbatch=n_batch, activate='relu')
        self.layer4 = FC_Layer(hidden_dim, output_dim, nbatch=n_batch, activate='exp')
        self.logits = torch.nn.Parameter(torch.randn(n_batch, output_dim))
        self.size0 = FC_Layer(input_dim, hidden_dim, bn='bn', activate='gelu', dropout=0)
        self.size1 = FC_Layer(hidden_dim, 1, activate='softplus', dropout=0)
    def reparameterize(self, mu, log_var, logvar_scale=1):
        scaled_log_var = log_var + torch.log(torch.tensor(logvar_scale, device=log_var.device))
        std = torch.exp(0.5 * scaled_log_var)
        eps = torch.randn_like(std)
        return mu + eps * std
    def forward(self, x, batch_tensor=None, logvar_scale=1, n_samples=1, infer_mode=False): 
        x1 = self.layer1(x)
        x2 = self.layer2(x1)
        mu = self.fc_mu(x2)
        log_var = self.fc_log_var(x2)
        if infer_mode:
            z = mu
            rate_scaled= self.layer4(self.layer3(z, batch_tensor), batch_tensor)
        else:
            rate_list = []
            z_list = []
            for _ in range(n_samples):
                z = self.reparameterize(mu, log_var, logvar_scale)
                z_list.append(z)
                rate = self.layer4(self.layer3(z, batch_tensor), batch_tensor)
                rate_list.append(rate)
            rate_scaled = torch.stack(rate_list, dim=0).mean(dim=0)  # [n_samples, B, G] -> [B, G]
            z = torch.stack(z_list, dim=0).mean(dim=0)          # Optional: mean z for analysis
        x_size0 = self.size0(x)
        size = self.size1(x_size0)
        # batch_size = x.size(0)  
        # if batch_size > self.max_samples:
        #     raise ValueError("Batch size exceeds the maximum allowed samples")
        # size = self.size[:batch_size]
        return rate_scaled, self.logits.exp(), size, z, mu, log_var













