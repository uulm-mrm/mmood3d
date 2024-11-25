# SPDX-License-Identifier: AGPL-3.0

import torch
from torch import Tensor, nn

from mmood3d.registry import MODELS


class FCNN(nn.Module):
    """
    Simple fully connected neural network.
    """

    def __init__(self, in_dim, out_dim, num_hidden):
        super().__init__()
        self.network = nn.Sequential(
            nn.Linear(in_dim, num_hidden),
            nn.LeakyReLU(),
            #nn.ReLU(),
            nn.Linear(num_hidden, num_hidden),
            nn.LeakyReLU(),
            #nn.ReLU(),
            nn.Linear(num_hidden, out_dim),
            nn.Tanh(), # orig is tanh, before was sigmoid
            #nn.Sigmoid(),
            #nn.ReLU(),
        )

    def forward(self, x):
        return self.network(x)

class FCNN_Tanh(nn.Module):
    """
    Simple fully connected neural network.
    """

    def __init__(self, in_dim, out_dim, num_hidden):
        super().__init__()
        self.network = nn.Sequential(
            nn.Linear(in_dim, num_hidden),
            nn.Tanh(),
            nn.Linear(num_hidden, num_hidden),
            nn.Tanh(),
            nn.Linear(num_hidden, out_dim),
        )

    def forward(self, x):
        return self.network(x)


class New_FCNN(nn.Module):
    """
    Simple fully connected neural network.
    """

    def __init__(self, in_dim, out_dim, num_hidden):
        super().__init__()
        self.network = nn.Sequential(
            nn.Linear(in_dim, num_hidden),
            nn.LeakyReLU(),
            nn.Linear(num_hidden, num_hidden),
            nn.LeakyReLU(),
            nn.Linear(num_hidden, out_dim),
            #nn.Tanh(),
        )

    def forward(self, x):
        return self.network(x)


"""
class RealNVPTransform(nn.Module):
    def __init__(self, num_features: int, num_hidden: int, base_network=FCNN_Tanh):#FCNN):
        super().__init__()

        self.nn1 = base_network(num_features // 2, num_features // 2, num_hidden)
        self.nn2 = base_network(num_features // 2, num_features // 2, num_hidden)

    def forward(self, x: Tensor):
        x1, x2 = x.chunk(2, dim=-1)
        y1 = x1 * torch.exp(self.nn1(x2))
        y2 = x2 * torch.exp(self.nn2(y1))
        y = torch.cat([y1, y2], dim=-1)

        log_det = torch.sum(self.nn1(x2), dim=-1) + torch.sum(self.nn2(y1), dim=-1)

        return y, log_det


@MODELS.register_module()
class NormFlowsRealNVP(nn.Module):
    def __init__(self, num_flows: int, num_features: int, num_hidden: int):
        super().__init__()
        self.num_flows = num_flows

        self.transforms = nn.ModuleList([
            RealNVPTransform(num_features, num_hidden) for _ in range(self.num_flows)
        ])

        #self.log_scales = nn.Parameter(torch.zeros(self.num_flows, num_features))
        self.log_scales = nn.Parameter(torch.full((self.num_flows, num_features), fill_value=0.01))
        self.translation = nn.Parameter(torch.zeros(self.num_flows, num_features))
 
    def forward(self, features: Tensor):
        log_prob = features.new_zeros(features.shape[0])
        transformed_features = features.clone()

        for i in range(self.num_flows):
            features_masked = transformed_features * torch.exp(-self.log_scales[i])
            t, log_det_jacobian = self.transforms[i](features_masked)
            transformed_features = t * torch.exp(-self.log_scales[i]) + self.translation[i]
            log_prob += -torch.sum(self.log_scales[i]) - log_det_jacobian

        return log_prob
    

class NewRealNVPTransform(nn.Module):
    def __init__(self, num_features: int, num_hidden: int, base_network=FCNN_Tanh):
        super().__init__()

        self.t1 = base_network(num_features // 2, num_features // 2, num_hidden)
        self.s1 = base_network(num_features // 2, num_features // 2, num_hidden)
        self.t2 = base_network(num_features // 2, num_features // 2, num_hidden)
        self.s2 = base_network(num_features // 2, num_features // 2, num_hidden)

    def forward(self, x: Tensor):
        lower, upper = x.chunk(2, dim=-1)

        t1_transformed = self.t1(lower)
        s1_transformed = self.s1(lower)
        upper = t1_transformed + upper * torch.exp(s1_transformed)

        t2_transformed = self.t2(upper)
        s2_transformed = self.s2(upper)
        lower = t2_transformed + lower * torch.exp(s2_transformed)

        transformed_x = torch.cat([lower, upper], dim=-1)
        log_det = torch.sum(s1_transformed, dim=-1) + torch.sum(s2_transformed, dim=-1)

        return transformed_x, log_det


@MODELS.register_module()
class NewNormFlowsRealNVP(nn.Module):
    def __init__(self, num_flows: int, num_features: int, num_hidden: int):
        super().__init__()
        self.num_flows = num_flows

        self.transforms = nn.ModuleList([
 #           NewRealNVPTransform(num_features, num_hidden) for _ in range(self.num_flows)
            RealNVPTransform(num_features, num_hidden, base_network=FCNN) for _ in range(self.num_flows)
        ])
 
    def forward(self, features: Tensor):
        log_prob = features.new_zeros(features.shape[0])
        transformed_features = features

        for i, transform in enumerate(self.transforms):
            transformed_features, log_det_jacobian = transform(transformed_features)
            log_prob -= log_det_jacobian
            
        return log_prob
"""    


class RealNVPCouplingLayer(nn.Module):
    def __init__(self, num_features, hidden_dim, mask, base_network=FCNN_Tanh):
        super().__init__()
        self.register_buffer('mask', mask)
        self.scale_translate_net = base_network(
            in_dim=num_features,
            out_dim=num_features * 2,  # Outputting both scale (s) and translate (t)
            num_hidden=hidden_dim
        )

    def forward(self, x):
        # Apply mask to input
        x_masked = x * self.mask
        # Pass masked input through the scale and translate network
        s_t = self.scale_translate_net(x_masked)
        s, t = s_t.chunk(2, dim=-1)  # Split into scale and translate components

        # Constrain the scaling factors to avoid numerical issues
        s = torch.tanh(s) * (1 - self.mask)
        scale = torch.exp(s)

        # Transform the non-masked part of the input
        y = x_masked + (1 - self.mask) * (x * scale + t)

        # Compute the log-determinant of the Jacobian
        log_det = ((1 - self.mask) * s).sum(dim=-1)
        return y, log_det

class UpdatedNormFlowsRealNVP(nn.Module):
    def __init__(self, num_flows, num_features, hidden_dim, base_network=New_FCNN):
        super().__init__()
        self.transforms = nn.ModuleList()
        for i in range(num_flows):
            mask = self._create_mask(num_features, parity=i % 2)
            self.transforms.append(
                RealNVPCouplingLayer(num_features, hidden_dim, mask, base_network)
            )

    def _create_mask(self, num_features, parity):
        mask = torch.arange(num_features) % 2
        if parity:
            mask = 1 - mask
        return mask.float()

    def forward(self, x):
        log_prob = x.new_zeros(x.size(0))
        transformed_x = x
        for transform in self.transforms:
            transformed_x, log_det = transform(transformed_x)
            log_prob += log_det
        return transformed_x, log_prob
