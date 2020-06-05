# Copyright 2019-2020 Stanislav Pidhorskyi
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#  http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================

import torch


__all__ = ['kl', 'reconstruction', 'discriminator_logistic_simple_gp',
           'discriminator_gradient_penalty', 'generator_logistic_non_saturating']


def kl(mu, log_var):
    return -0.5 * torch.mean(torch.mean(1 + log_var - mu.pow(2) - log_var.exp(), 1))


def reconstruction(recon_x, x, lod=None):
    return torch.mean((recon_x - x)**2)


def discriminator_logistic_simple_gp(d_result_fake, d_result_real, reals, r1_gamma=10.0):
    loss = (torch.nn.functional.softplus(d_result_fake) + torch.nn.functional.softplus(-d_result_real))

    if r1_gamma != 0.0:
        real_loss = d_result_real.sum()
        real_grads = torch.autograd.grad(real_loss, reals, create_graph=True, retain_graph=True)[0]
        r1_penalty = torch.sum(real_grads.pow(2.0), dim=[1, 2, 3])
        loss = loss + r1_penalty * (r1_gamma * 0.5)
    return loss.mean()


def discriminator_gradient_penalty(d_result_real, reals, r1_gamma=10.0):
    real_loss = d_result_real.sum()
    real_grads = torch.autograd.grad(real_loss, reals, create_graph=True, retain_graph=True)[0]
    r1_penalty = torch.sum(real_grads.pow(2.0), dim=[1, 2, 3])
    loss = r1_penalty * (r1_gamma * 0.5)
    return loss.mean()


def generator_logistic_non_saturating(d_result_fake):
    return torch.nn.functional.softplus(-d_result_fake).mean()


class BaseLoss(torch.nn.Module):
    def __init__(self):
        super(BaseLoss, self).__init__()
        self.loss_stats_ = dict()

    @property
    def loss_stats(self):
        return self.loss_stats_


class CriticLoss(BaseLoss):
    def __init__(self):
        super(CriticLoss, self).__init__()

    def forward(self, critic_outputs_t, critic_outputs_s):
        loss = (critic_outputs_t - critic_outputs_s).mean()
        self.loss_stats_['total'] = loss
        return loss


class CycleLoss(BaseLoss):
    def __init__(self):
        super(CycleLoss, self).__init__()

    def forward(self, z_t, z_t_hat, z_s, z_s_restored):
        z_t_l1_loss = torch.nn.L1Loss()(z_t_hat, z_t)
        z_s_l1_loss = torch.nn.L1Loss()(z_s_restored, z_s)
        loss = z_t_l1_loss + z_s_l1_loss

        self.loss_stats_ = dict()
        self.loss_stats_['z_t_l1_loss'] = z_t_l1_loss
        self.loss_stats_['z_s_l1_loss'] = z_s_l1_loss
        self.loss_stats_['total'] = loss
        return loss


class GeneratorLoss(BaseLoss):
    def __init__(self):
        super(GeneratorLoss, self).__init__()
        self.cycle_criterion = CycleLoss()

    def forward(self, critic_outputs_t, z_t, z_t_hat, z_s, z_s_restored):
        cycle_loss = self.cycle_criterion(z_t, z_t_hat, z_s, z_s_restored)
        loss = -critic_outputs_t.mean() + cycle_loss

        self.loss_stats_ = dict()
        cycle_loss_stats = self.cycle_criterion.loss_stats
        self.loss_stats_.update({
            f'cycle_loss/{key}': cycle_loss_stats[key] for key in cycle_loss_stats.keys()
        })
        self.loss_stats_['total'] = loss
        return loss


class FaceRotationModelLoss(BaseLoss):
    def __init__(self):
        super(FaceRotationModelLoss, self).__init__()
        self.critic_criterion = CriticLoss()
        self.generator_criterion = GeneratorLoss()

    def forward(self, outputs, x):
        critic_outputs_t, critic_outputs_s, z_t, z_t_hat, z_s, z_s_restored = \
            outputs['critic_outputs_t'], outputs['critic_outputs_s'], outputs['z_t'], outputs['z_t_hat'], \
            outputs['z_s'], outputs['z_s_restored']

        critic_loss = self.critic_criterion(critic_outputs_t, critic_outputs_s)
        generator_loss = self.generator_criterion(critic_outputs_t, z_t, z_t_hat, z_s, z_s_restored)
        losses = (critic_loss, generator_loss)

        self.loss_stats_ = dict()
        self.loss_stats_['critic_t'] = critic_outputs_t.mean()
        self.loss_stats_['critic_s'] = critic_outputs_s.mean()
        critic_loss_stats = self.critic_criterion.loss_stats
        self.loss_stats_.update({
            f'critic_loss/{key}': critic_loss_stats[key] for key in critic_loss_stats.keys()
        })
        generator_loss_stats = self.generator_criterion.loss_stats
        self.loss_stats_.update({
            f'generator_loss/{key}': generator_loss_stats[key] for key in generator_loss_stats.keys()
        })
        return losses
