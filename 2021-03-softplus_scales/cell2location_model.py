import numpy as np
import pyro
import pyro.distributions as dist
import torch
import torch.nn as nn
from pyro.infer.autoguide import AutoNormal, init_to_mean
from pyro.nn import PyroModule, pyro_method
from pyro.infer import SVI, Trace_ELBO, JitTrace_ELBO
import pyro.optim as optim
from pyro.distributions.torch_distribution import TorchDistributionMixin

from numbers import Number

from torch.distributions import constraints
from torch.distributions.exp_family import ExponentialFamily
from torch.distributions.utils import broadcast_all

from tqdm.auto import tqdm

def _convert_mean_disp_to_counts_logits(mu, theta, eps=1e-8):
    r"""
    NB parameterizations conversion.

    Parameters
    ----------
    mu
        mean of the NB distribution.
    theta
        inverse overdispersion.
    eps
        constant used for numerical log stability. (Default value = 1e-6)

    Returns
    -------
    type
        the number of failures until the experiment is stopped
        and the success probability.
    """
    if not (mu is None) == (theta is None):
        raise ValueError(
            "If using the mu/theta NB parameterization, both parameters must be specified"
        )
    logits = (mu + eps).log() - (theta + eps).log()
    total_count = theta
    return total_count, logits

def _standard_gamma(alpha):
    return torch._standard_gamma(alpha)


class TorchGamma(ExponentialFamily):
    r"""
    Creates a Gamma distribution parameterized by shape :attr:`mu` and :attr:`sigma`.
    Example::
        >>> m = Gamma(torch.tensor([1.0]), torch.tensor([1.0]))
        >>> m.sample()  # Gamma distributed with mu=1 and sigma=1
        tensor([ 1.012])
    Args:
        mu (float or Tensor): shape parameter of the distribution
            (often referred to as alpha)
        sigma (float or Tensor): rate = 1 / scale of the distribution
            (often referred to as beta)
    """
    arg_constraints = {"alpha": constraints.positive, "beta": constraints.positive,
                       "mu": constraints.positive, "sigma": constraints.positive}
    support = constraints.positive
    has_rsample = True
    _mean_carrier_measure = 0

    @property
    def mean(self):
        return self.alpha / self.beta

    @property
    def variance(self):
        return self.alpha / self.beta.pow(2)

    def __init__(self, alpha=None, beta=None, mu=None, sigma=None, validate_args=None):
        if alpha != None and beta != None and mu != None and sigma != None:
            raise ValueError(
                "Gamma requires alpha and beta OR mu and sigma (not all of them)"
            )
        elif alpha is not None and beta is not None:
            mu = alpha / beta
            sigma = (alpha / beta.pow(2)).sqrt()
        elif mu is not None and sigma is not None:
            alpha = mu ** 2 / sigma ** 2
            beta = mu / sigma ** 2
        else:
            raise ValueError(
                "Gamma requires alpha and beta OR mu and sigma (define two)"
            )
        self.alpha, self.beta = broadcast_all(alpha, beta)
        self.mu, self.sigma = broadcast_all(mu, sigma)
        if isinstance(alpha, Number) and isinstance(beta, Number):
            batch_shape = torch.Size()
        else:
            batch_shape = self.alpha.size()
        super(TorchGamma, self).__init__(batch_shape, validate_args=validate_args)

    def expand(self, batch_shape, _instance=None):
        new = self._get_checked_instance(Gamma, _instance)
        batch_shape = torch.Size(batch_shape)
        new.alpha = self.alpha.expand(batch_shape)
        new.beta = self.beta.expand(batch_shape)
        super(TorchGamma, new).__init__(batch_shape, validate_args=False)
        new._validate_args = self._validate_args
        return new

    def rsample(self, sample_shape=torch.Size()):
        shape = self._extended_shape(sample_shape)
        value = _standard_gamma(self.alpha.expand(shape)) / self.beta.expand(shape)
        value.detach().clamp_(
            min=torch.finfo(value.dtype).tiny
        )  # do not record in autograd graph
        return value

    def log_prob(self, value):
        value = torch.as_tensor(value, dtype=self.beta.dtype, device=self.beta.device)
        if self._validate_args:
            self._validate_sample(value)
        return (
            self.alpha * torch.log(self.beta)
            + (self.alpha - 1) * torch.log(value)
            - self.beta * value
            - torch.lgamma(self.alpha)
        )

    def entropy(self):
        return (
            self.alpha
            - torch.log(self.beta)
            + torch.lgamma(self.alpha)
            + (1.0 - self.alpha) * torch.digamma(self.alpha)
        )

    @property
    def _natural_params(self):
        return (self.alpha - 1, -self.beta)

    def _log_normalizer(self, x, y):
        return torch.lgamma(x + 1) + (x + 1) * torch.log(-y.reciprocal())


class Gamma(TorchGamma, TorchDistributionMixin):
    """Adds pyro TorchDistributionMixin methods."""

    def conjugate_update(self, other):
        assert isinstance(other, TorchGamma)
        alpha = self.alpha + other.alpha - 1
        beta = self.beta + other.beta
        updated = TorchGamma(alpha, beta)

        def _log_normalizer(d):
            c = d.alpha
            return d.beta.log() * c - c.lgamma()

        log_normalizer = (
            _log_normalizer(self) + _log_normalizer(other) - _log_normalizer(updated)
        )
        return updated, log_normalizer


class LocationModelLinearDependentWMultiExperimentModel(PyroModule):
    def __init__(
        self,
        n_obs,
        n_var,
        n_fact,
        n_exper,
        cell_state_mat,
        batch_size=None,
        n_comb: int = 50,
        m_g_gene_level_prior={"mean": 1 / 2, "sd": 1 / 4},
        m_g_gene_level_var_prior={"mean_var_ratio": 1},
        cell_number_prior={
            "N_cells_per_location": 8,
            "A_factors_per_location": 7,
            "Y_combs_per_location": 7,
        },
        cell_number_var_prior={
            "N_cells_mean_var_ratio": 1,
            "A_factors_mean_var_ratio": 1,
            "Y_combs_mean_var_ratio": 1,
        },
        alpha_g_phi_hyp_prior={"mean": 3, "sd": 1},
        gene_add_alpha_hyp_prior={"mean": 3, "sd": 1},
        gene_add_mean_hyp_prior={"alpha": 1, "beta": 100},
        w_sf_mean_var_ratio=5,
    ):

        super().__init__()

        self.n_obs = n_obs
        self.n_var = n_var
        self.n_fact = n_fact
        self.n_exper = n_exper
        self.batch_size = batch_size
        self.n_comb = n_comb

        for k in m_g_gene_level_var_prior.keys():
            m_g_gene_level_prior[k] = m_g_gene_level_var_prior[k]
        #for k in m_g_gene_level_prior.keys():
        #    m_g_gene_level_prior[k] = m_g_gene_level_prior[k]

        # compute hyperparameters from mean and sd
        self.m_g_gene_level_prior = m_g_gene_level_prior
        self.m_g_shape = (
            self.m_g_gene_level_prior["mean"] ** 2
            / self.m_g_gene_level_prior["sd"] ** 2
        )
        self.m_g_rate = (
            self.m_g_gene_level_prior["mean"] / self.m_g_gene_level_prior["sd"] ** 2
        )
        self.m_g_shape_var = torch.tensor(
            self.m_g_shape / self.m_g_gene_level_prior["mean_var_ratio"]
        )
        self.m_g_rate_var = torch.tensor(self.m_g_rate / self.m_g_gene_level_prior["mean_var_ratio"])
        self.m_g_shape = torch.tensor(self.m_g_shape)
        self.m_g_rate = torch.tensor(self.m_g_rate)

        self.alpha_g_phi_hyp_prior = alpha_g_phi_hyp_prior
        self.w_sf_mean_var_ratio = w_sf_mean_var_ratio
        self.gene_add_alpha_hyp_prior = gene_add_alpha_hyp_prior
        self.gene_add_mean_hyp_prior = gene_add_mean_hyp_prior

        cell_number_prior["factors_per_combs"] = (
            cell_number_prior["A_factors_per_location"]
            / cell_number_prior["Y_combs_per_location"]
        )
        for k in cell_number_var_prior.keys():
            cell_number_prior[k] = cell_number_var_prior[k]
        #for k in cell_number_prior.keys():
        #    cell_number_prior[k] = np.array(cell_number_prior[k]).reshape((1, 1))
        self.cell_number_prior = cell_number_prior

        self.cell_state_mat = cell_state_mat
        # self.register_buffer("cell_state", torch.tensor(cell_state_mat.T))
        self.cell_state = torch.tensor(cell_state_mat.T)
        # self.register_buffer("ones", torch.ones((1, 1)))
        self.ones = torch.ones((1, 1))

    @staticmethod
    def _get_fn_args_from_batch(tensor_dict):
        x_data = tensor_dict[_CONSTANTS.X_KEY]
        ind_x = tensor_dict["ind_x"].long().squeeze()
        obs2sample = tensor_dict["obs2sample"]
        return (x_data, ind_x, obs2sample), {}

    def create_plates(self, x_data, idx, obs2sample):

        if self.batch_size is None:
            # to support training on full data
            obs_axis = pyro.plate("obs_axis", self.n_obs, dim=-2)
        else:
            obs_axis = pyro.plate(
                "obs_axis",
                self.n_obs,
                dim=-2,
                subsample_size=self.batch_size,
                subsample=idx,
            )
        return [
            obs_axis,
            pyro.plate("var_axis", self.n_var, dim=-1),
            pyro.plate("factor_axis", self.n_fact, dim=-1),
            pyro.plate("combination_axis", self.n_comb, dim=-3),
            pyro.plate("experim_axis", self.n_exper, dim=-2),
        ]

    def forward(self, x_data, idx, obs2sample):

        (
            obs_axis,
            var_axis,
            factor_axis,
            combination_axis,
            experim_axis,
        ) = self.create_plates(x_data, idx, obs2sample)

        # =====================Gene expression level scaling m_g======================= #
        # Explains difference in sensitivity for each gene between single cell and spatial technology

        m_g_alpha_hyp = pyro.sample(
            "m_g_alpha_hyp",
            Gamma(mu=self.ones * self.m_g_shape, sigma=self.ones * self.m_g_shape_var),
        )

        m_g_beta_hyp = pyro.sample(
            "m_g_beta_hyp",
            Gamma(mu=self.ones * self.m_g_rate, sigma=self.ones * self.m_g_rate_var),
        )
        with var_axis:
            m_g = pyro.sample("m_g", Gamma(alpha=m_g_alpha_hyp, beta=m_g_beta_hyp))

        # =====================Cell abundances w_sf======================= #
        # factorisation prior on w_sf models similarity in locations
        # across cell types f and reflects the absolute scale of w_sf
        with obs_axis:
            n_s_cells_per_location = pyro.sample(
                "n_s_cells_per_location",
                Gamma(
                    mu=self.ones * self.cell_number_prior["N_cells_per_location"],
                    sigma=self.ones
                    * np.sqrt(
                        self.cell_number_prior["N_cells_per_location"]
                        / self.cell_number_prior["N_cells_mean_var_ratio"]
                    ),
                ),
            )

            y_s_combs_per_location = pyro.sample(
                "y_s_combs_per_location",
                Gamma(
                    mu=self.ones * self.cell_number_prior["Y_combs_per_location"],
                    sigma=self.ones
                    * np.sqrt(
                        self.cell_number_prior["Y_combs_per_location"]
                        / self.cell_number_prior["Y_combs_mean_var_ratio"]
                    ),
                ),
            )

        with combination_axis, obs_axis:
            shape = y_s_combs_per_location / self.n_comb
            rate = (
                torch.ones([self.n_comb, 1, 1])
                / n_s_cells_per_location
                * y_s_combs_per_location
            )
            z_sr_combs_factors = pyro.sample(
                "z_sr_combs_factors", Gamma(alpha=shape, beta=rate)
            )  # (n_comb, n_obs, 1)
        with combination_axis:
            k_r_factors_per_combs = pyro.sample(
                "k_r_factors_per_combs",
                Gamma(
                    mu=self.ones * self.cell_number_prior["factors_per_combs"],
                    sigma=self.ones
                    * np.sqrt(
                        self.cell_number_prior["factors_per_combs"]
                        / self.cell_number_prior["A_factors_mean_var_ratio"]
                    ),
                ),
            )  # self.n_comb, 1, 1)

            c2f_shape = k_r_factors_per_combs / self.n_fact

        with factor_axis, combination_axis:
            x_fr_comb2fact = pyro.sample(
                "x_fr_comb2fact", Gamma(alpha=c2f_shape, beta=k_r_factors_per_combs)
            )  # (self.n_comb, 1, self.n_fact)

        with obs_axis, factor_axis:
            w_sf_mu = z_sr_combs_factors.squeeze(-1).T @ x_fr_comb2fact.squeeze(-2)
            w_sf_sigma = w_sf_mu / self.w_sf_mean_var_ratio
            w_sf = pyro.sample(
                "w_sf", Gamma(mu=w_sf_mu, sigma=w_sf_sigma)
            )  # (self.n_obs, self.n_fact)

        # =====================Location-specific additive component======================= #
        l_s_add_alpha = pyro.sample(
            "l_s_add_alpha", Gamma(alpha=self.ones, beta=self.ones)
        )
        l_s_add_beta = pyro.sample(
            "l_s_add_beta", Gamma(alpha=self.ones, beta=self.ones)
        )

        with obs_axis:
            l_s_add = pyro.sample(
                "l_s_add", Gamma(alpha=l_s_add_alpha, beta=l_s_add_beta)
            )  # (self.n_obs, 1)

        # =====================Gene-specific additive component ======================= #
        # per gene molecule contribution that cannot be explained by
        # cell state signatures (e.g. background, free-floating RNA)
        s_g_gene_add_alpha_hyp = pyro.sample(
            "s_g_gene_add_alpha_hyp",
            Gamma(
                mu=self.ones * self.gene_add_alpha_hyp_prior["mean"],
                sigma=self.ones * self.gene_add_alpha_hyp_prior["sd"],
            ),
        )
        with experim_axis:
            s_g_gene_add_mean = pyro.sample(
                "s_g_gene_add_mean",
                Gamma(
                    alpha=self.ones * self.gene_add_mean_hyp_prior["alpha"],
                    beta=self.ones * self.gene_add_mean_hyp_prior["beta"],
                ),
            )
            s_g_gene_add_alpha_e_inv = pyro.sample(
                "s_g_gene_add_alpha_e_inv", dist.Exponential(s_g_gene_add_alpha_hyp)
            )
            s_g_gene_add_alpha_e = self.ones / torch.pow(s_g_gene_add_alpha_e_inv, 2)
        with experim_axis, var_axis:
            s_g_gene_add = pyro.sample(
                "s_g_gene_add",
                Gamma(s_g_gene_add_alpha_e, s_g_gene_add_alpha_e / s_g_gene_add_mean),
            )

        # =====================Gene-specific overdispersion ======================= #
        alpha_g_phi_hyp = pyro.sample(
            "alpha_g_phi_hyp",
            Gamma(
                mu=self.ones * self.alpha_g_phi_hyp_prior["mean"],
                sigma=self.ones * self.alpha_g_phi_hyp_prior["sd"],
            ),
        )
        with experim_axis, var_axis:
            alpha_g_inverse = pyro.sample(
                "alpha_g_inverse", dist.Exponential(alpha_g_phi_hyp)
            )  # (self.n_exper, self.n_var)

        # =====================Expected expression ======================= #
        # expected expression
        mu = (w_sf @ self.cell_state) * m_g + (obs2sample @ s_g_gene_add) + l_s_add
        theta = obs2sample @ (self.ones / alpha_g_inverse.pow(2))
        # convert mean and overdispersion to total count and logits
        total_count, logits = _convert_mean_disp_to_counts_logits(mu, theta, eps=1e-8)

        # =====================DATA likelihood ======================= #
        # Likelihood (sampling distribution) of data_target & add overdispersion via NegativeBinomial
        with obs_axis, var_axis:
            data_target = pyro.sample(
                "data_target",
                dist.NegativeBinomial(total_count=total_count, logits=logits),
                obs=x_data,
            )

        # =====================Compute mRNA count from each factor in locations  ======================= #
        mRNA = w_sf * (self.cell_state * m_g).sum(-1)
        u_sf_mRNA_factors = pyro.deterministic("u_sf_mRNA_factors", mRNA)

    def compute_expected(self, obs2sample):
        r"""Compute expected expression of each gene in each location. Useful for evaluating how well
        the model learned expression pattern of all genes in the data.
        """
        self.mu = (
            np.dot(self.samples["post_sample_means"]["w_sf"], self.cell_state_mat.T)
            * self.samples["post_sample_means"]["m_g"]
            + np.dot(obs2sample, self.samples["post_sample_means"]["s_g_gene_add"])
            + self.samples["post_sample_means"]["l_s_add"]
        )
        self.alpha = np.dot(
            obs2sample,
            1
            / (
                self.samples["post_sample_means"]["alpha_g_inverse"]
                * self.samples["post_sample_means"]["alpha_g_inverse"]
            ),
        )


class LocationModelLinearDependentWMultiExperiment(nn.Module):
    def __init__(self, **kwargs):

        super().__init__()
        self.hist = []
        
        self.model = LocationModelLinearDependentWMultiExperimentModel(**kwargs)
        self.guide = AutoNormal(
            self.model.forward,
            init_loc_fn=init_to_mean, init_scale=0.1,
            create_plates=self.model.create_plates,
        )

    def train_full_data(self, x_data, obs2sample,
                        n_epochs=20000, lr=0.002):

        idx = np.arange(x_data.shape[0]).astype("int64")

        device = torch.device('cuda')
        idx = torch.tensor(idx).to(device)
        x_data = torch.tensor(x_data).to(device)
        obs2sample = torch.tensor(obs2sample).to(device)

        pyro.clear_param_store()
        self.guide(x_data, idx, obs2sample)

        svi = SVI(self.model.forward, self.guide,
                  optim.ClippedAdam({"lr": lr, 'clip_norm': 200}),
                  loss=JitTrace_ELBO())

        iter_iterator = tqdm(range(n_epochs))
        hist = []
        for it in iter_iterator:

            loss = svi.step(x_data, idx, obs2sample)
            iter_iterator.set_description('Epoch ' + '{:d}'.format(it) +
                                          ', -ELBO: ' + '{:.4e}'.format(loss))
            hist.append(loss)

            if it % 500 == 0:
                torch.cuda.empty_cache()
                
        self.hist = hist

            
### Build cell state signature matrix ###
import anndata
import numpy as np
import pandas as pd
from scipy.sparse import csr_matrix


# +
def get_cluster_averages(adata_ref, cluster_col):
    """
    :param adata_ref: AnnData object of reference single-cell dataset
    :param cluster_col: Name of adata_ref.obs column containing cluster labels
    :returns: pd.DataFrame of cluster average expression of each gene
    """
    if not adata_ref.raw:
        raise ValueError("AnnData object has no raw data")
    if sum(adata_ref.obs.columns == cluster_col) != 1:
        raise ValueError("cluster_col is absent in adata_ref.obs or not unique")

    all_clusters = np.unique(adata_ref.obs[cluster_col])
    averages_mat = np.zeros((1, adata_ref.raw.X.shape[1]))

    for c in all_clusters:
        sparse_subset = csr_matrix(adata_ref.raw.X[np.isin(adata_ref.obs[cluster_col], c), :])
        aver = sparse_subset.mean(0)
        averages_mat = np.concatenate((averages_mat, aver))
    averages_mat = averages_mat[1:, :].T
    averages_df = pd.DataFrame(data=averages_mat,
                               index=adata_ref.raw.var_names,
                               columns=all_clusters)

    return averages_df
