import torch
import torch.utils.data
import os
script_directory = os.path.dirname(os.path.abspath(__file__))

if 'solution' in script_directory:
    from solution import utils as ut
    from solution.models import nns
else:
    from submission import utils as ut
    from submission.models import nns

from torch import nn

class FSVAE(nn.Module):
    def __init__(self, nn='v2', name='fsvae'):
        super().__init__()
        self.name = name
        self.z_dim = 10
        self.y_dim = 10
        nn = getattr(nns, nn)
        self.enc = nn.Encoder(self.z_dim, self.y_dim) # from ./nns/v2.py
        self.dec = nn.Decoder(self.z_dim, self.y_dim) # from ./nns/v2.py

        # Set prior as fixed parameter attached to Module
        self.z_prior_m = torch.nn.Parameter(torch.zeros(1), requires_grad=False)
        self.z_prior_v = torch.nn.Parameter(torch.ones(1), requires_grad=False)
        self.z_prior = (self.z_prior_m, self.z_prior_v)

    def negative_elbo_bound(self, x, y):
        """
            Computes the Evidence Lower Bound, KL and, Reconstruction costs

            Args:
                x: tensor: (batch, dim): Observations
                y: tensor: (batch, y_dim): Labels

            Returns:
                nelbo: tensor: (): Negative evidence lower bound
                kl_z: tensor: (): ELBO KL divergence to prior for latent variable z
                rec: tensor: (): ELBO Reconstruction term
        """
        ################################################################################
        # TODO: Modify/complete the code here
        # Compute negative Evidence Lower Bound and its KL and Rec decomposition
        #
        # Note that we are interested in the ELBO of ln p(x | y)
        #
        # Note that nelbo = kl + rec
        #
        # Outputs should all be tensor scalars
        #
        # Return:
        #   nelbo, kl_z, rec
        ################################################################################
        ### START CODE HERE ###
        m_z, v_z = self.enc(x, y)
        z = ut.sample_gaussian(m_z, v_z) # z ~ q_\phi(z | x, y)
        log_prob_x = ut.log_normal(x, self.dec(y, z), 0.1 * torch.ones(x.shape)) # logp_\theta(x | y, z)

        # calc rec
        rec = -torch.mean(log_prob_x)

        # calc kl
        # D_{KL}(q\phi(z|x, y) || p(z))
        kl = torch.mean(ut.kl_normal(m_z, v_z, torch.ones_like(z) * self.z_prior_m, torch.ones_like(z) * self.z_prior_v))
        
        nelbo = kl + rec
        return nelbo, kl, rec
        ### END CODE HERE ###
        ################################################################################
        # End of code modification
        ################################################################################
        raise NotImplementedError

    def loss(self, x, y):
        nelbo, kl_z, rec = self.negative_elbo_bound(x, y)
        loss = nelbo

        summaries = dict((
            ('train/loss', loss),
            ('gen/elbo', -nelbo),
            ('gen/kl_z', kl_z),
            ('gen/rec', rec),
        ))

        return loss, summaries

    def compute_mean_given(self, z, y):
        return self.dec(z, y)

    def sample_z(self, batch):
        return ut.sample_gaussian(self.z_prior[0].expand(batch, self.z_dim),
                                  self.z_prior[1].expand(batch, self.z_dim))
