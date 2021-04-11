import numpy as np
import pyro
import pyro.distributions as dist
import pyro.optim as optim
import torch
from pyro.infer import SVI, JitTrace_ELBO
from pyro.infer.autoguide import AutoNormal, init_to_mean
from pyro.nn import PyroModule
from tqdm.auto import tqdm

def _convert_mean_disp_to_counts_logits(mu, theta, eps=1e-6):
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
        m_g_gene_level_var_prior={"mean_var_ratio": 1.0},
        cell_number_prior={
            "N_cells_per_location": 8.0,
            "A_factors_per_location": 7.0,
            "Y_combs_per_location": 7.0,
        },
        cell_number_var_prior={
            "N_cells_mean_var_ratio": 1.0,
        },
        alpha_g_phi_hyp_prior={"alpha": 9.0, "beta": 3.0},
        gene_add_alpha_hyp_prior={"alpha": 9.0, "beta": 3.0},
        gene_add_mean_hyp_prior={"alpha": 1.0, "beta": 100.0},
        w_sf_mean_var_ratio=5.0,
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
        self.cell_number_prior = cell_number_prior

        # compute hyperparameters from mean and sd
        self.m_g_gene_level_prior = m_g_gene_level_prior
        self.register_buffer(
            "m_g_shape",
            torch.tensor(
                (self.m_g_gene_level_prior["mean"] ** 2)
                / (self.m_g_gene_level_prior["sd"] ** 2)
            ),
        )
        self.register_buffer(
            "m_g_rate",
            torch.tensor(
                self.m_g_gene_level_prior["mean"]
                / (self.m_g_gene_level_prior["sd"] ** 2)
            ),
        )
        self.register_buffer(
            "m_g_mean_var", torch.tensor(self.m_g_gene_level_prior["mean_var_ratio"])
        )
        self.register_buffer("eps", torch.tensor(1e-8))

        self.cell_state_mat = cell_state_mat
        self.register_buffer("cell_state", torch.tensor(cell_state_mat.T))

        self.register_buffer(
            "N_cells_per_location",
            torch.tensor(self.cell_number_prior["N_cells_per_location"]),
        )
        self.register_buffer(
            "factors_per_combs",
            torch.tensor(self.cell_number_prior["factors_per_combs"]),
        )
        self.register_buffer(
            "Y_combs_per_location",
            torch.tensor(self.cell_number_prior["Y_combs_per_location"]),
        )
        self.register_buffer(
            "N_cells_mean_var_ratio",
            torch.tensor(self.cell_number_prior["N_cells_mean_var_ratio"]),
        )

        self.register_buffer(
            "alpha_g_phi_hyp_prior_alpha",
            torch.tensor(self.alpha_g_phi_hyp_prior["alpha"]),
        )
        self.register_buffer(
            "alpha_g_phi_hyp_prior_beta",
            torch.tensor(self.alpha_g_phi_hyp_prior["beta"]),
        )
        self.register_buffer(
            "gene_add_alpha_hyp_prior_alpha",
            torch.tensor(self.gene_add_alpha_hyp_prior["alpha"]),
        )
        self.register_buffer(
            "gene_add_alpha_hyp_prior_beta",
            torch.tensor(self.gene_add_alpha_hyp_prior["beta"]),
        )
        self.register_buffer(
            "gene_add_mean_hyp_prior_alpha",
            torch.tensor(self.gene_add_mean_hyp_prior["alpha"]),
        )
        self.register_buffer(
            "gene_add_mean_hyp_prior_beta",
            torch.tensor(self.gene_add_mean_hyp_prior["beta"]),
        )

        self.register_buffer(
            "w_sf_mean_var_ratio_tensor", torch.tensor(self.w_sf_mean_var_ratio)
        )

        self.register_buffer("n_fact_tensor", torch.tensor(self.n_fact))
        self.register_buffer("n_comb_tensor", torch.tensor(self.n_comb))

        self.register_buffer("ones", torch.ones((1, 1)))
        self.register_buffer("ones_n_comb_1_1", torch.ones([self.n_comb, 1, 1]))

    @staticmethod
    def _get_fn_args_from_batch(tensor_dict):
        x_data = tensor_dict[_CONSTANTS.X_KEY]
        ind_x = tensor_dict["ind_x"].long().squeeze()
        batch_index = tensor_dict[_CONSTANTS.BATCH_KEY]
        return (x_data, ind_x, batch_index), {}

    def create_plates(self, x_data, idx, batch_index):

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

    def forward(self, x_data, idx, batch_index):

        obs2sample = batch_index  # one_hot(batch_index, self.n_exper)

        (
            obs_axis,
            var_axis,
            factor_axis,
            combination_axis,
            experim_axis,
        ) = self.create_plates(x_data, idx, batch_index)

        # =====================Gene expression level scaling m_g======================= #
        # Explains difference in sensitivity for each gene between single cell and spatial technology

        m_g_alpha_hyp = pyro.sample(
            "m_g_alpha_hyp",
            dist.Gamma(self.m_g_shape * self.m_g_mean_var, self.m_g_mean_var),
        )

        m_g_beta_hyp = pyro.sample(
            "m_g_beta_hyp",
            dist.Gamma(self.m_g_rate * self.m_g_mean_var, self.m_g_mean_var),
        )
        with var_axis:
            m_g = pyro.sample("m_g", dist.Gamma(m_g_alpha_hyp, m_g_beta_hyp))

        # =====================Cell abundances w_sf======================= #
        # factorisation prior on w_sf models similarity in locations
        # across cell types f and reflects the absolute scale of w_sf
        with obs_axis:
            n_s_cells_per_location = pyro.sample(
                "n_s_cells_per_location",
                dist.Gamma(
                    self.N_cells_per_location * self.N_cells_mean_var_ratio,
                    self.N_cells_mean_var_ratio,
                ),
            )

            y_s_combs_per_location = pyro.sample(
                "y_s_combs_per_location",
                dist.Gamma(self.Y_combs_per_location, self.ones),
            )

        with combination_axis, obs_axis:
            shape = y_s_combs_per_location / self.n_comb_tensor
            rate = self.ones_n_comb_1_1 / (
                n_s_cells_per_location / y_s_combs_per_location
            )
            z_sr_combs_factors = pyro.sample(
                "z_sr_combs_factors", dist.Gamma(shape, rate)
            )  # (n_comb, n_obs, 1)
        with combination_axis:
            k_r_factors_per_combs = pyro.sample(
                "k_r_factors_per_combs", dist.Gamma(self.factors_per_combs, self.ones)
            )  # self.n_comb, 1, 1)

            c2f_shape = k_r_factors_per_combs / self.n_fact_tensor

        with factor_axis, combination_axis:
            x_fr_comb2fact = pyro.sample(
                "x_fr_comb2fact", dist.Gamma(c2f_shape, k_r_factors_per_combs)
            )  # (self.n_comb, 1, self.n_fact)

        with obs_axis, factor_axis:
            w_sf_mu = z_sr_combs_factors.squeeze(-1).T @ x_fr_comb2fact.squeeze(-2)
            w_sf = pyro.sample(
                "w_sf",
                dist.Gamma(
                    w_sf_mu * self.w_sf_mean_var_ratio_tensor,
                    self.w_sf_mean_var_ratio_tensor,
                ),
            )  # (self.n_obs, self.n_fact)

        # =====================Location-specific additive component======================= #
        l_s_add_alpha = pyro.sample("l_s_add_alpha", dist.Gamma(self.ones, self.ones))
        l_s_add_beta = pyro.sample("l_s_add_beta", dist.Gamma(self.ones, self.ones))

        with obs_axis:
            l_s_add = pyro.sample(
                "l_s_add", dist.Gamma(l_s_add_alpha, l_s_add_beta)
            )  # (self.n_obs, 1)

        # =====================Gene-specific additive component ======================= #
        # per gene molecule contribution that cannot be explained by
        # cell state signatures (e.g. background, free-floating RNA)
        s_g_gene_add_alpha_hyp = pyro.sample(
            "s_g_gene_add_alpha_hyp",
            dist.Gamma(
                self.gene_add_alpha_hyp_prior_alpha, self.gene_add_alpha_hyp_prior_beta
            ),
        )
        with experim_axis:
            s_g_gene_add_mean = pyro.sample(
                "s_g_gene_add_mean",
                dist.Gamma(
                    self.gene_add_mean_hyp_prior_alpha,
                    self.gene_add_mean_hyp_prior_beta,
                ),
            )
            s_g_gene_add_alpha_e_inv = pyro.sample(
                "s_g_gene_add_alpha_e_inv", dist.Exponential(s_g_gene_add_alpha_hyp)
            )
            s_g_gene_add_alpha_e = self.ones / torch.pow(s_g_gene_add_alpha_e_inv, 2)
        with experim_axis, var_axis:
            s_g_gene_add = pyro.sample(
                "s_g_gene_add",
                dist.Gamma(
                    s_g_gene_add_alpha_e, s_g_gene_add_alpha_e / s_g_gene_add_mean
                ),
            )

        # =====================Gene-specific overdispersion ======================= #
        alpha_g_phi_hyp = pyro.sample(
            "alpha_g_phi_hyp",
            dist.Gamma(
                self.alpha_g_phi_hyp_prior_alpha, self.alpha_g_phi_hyp_prior_beta
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
        total_count, logits = _convert_mean_disp_to_counts_logits(
            mu, theta, eps=self.eps
        )

        # =====================DATA likelihood ======================= #
        # Likelihood (sampling distribution) of data_target & add overdispersion via NegativeBinomial
        with obs_axis, var_axis:
            pyro.sample(
                "data_target",
                dist.NegativeBinomial(total_count=total_count, logits=logits),
                obs=x_data,
            )

        # =====================Compute mRNA count from each factor in locations  ======================= #
        mRNA = w_sf * (self.cell_state * m_g).sum(-1)
        pyro.deterministic("u_sf_mRNA_factors", mRNA)

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


class LocationModelLinearDependentWMultiExperiment(torch.nn.Module):
    def __init__(self, **kwargs):

        super().__init__()
        self.hist = []

        self._model = LocationModelLinearDependentWMultiExperimentModel(**kwargs)
        self._guide = AutoNormal(
            self.model,
            init_loc_fn=init_to_mean,
            create_plates=self.model.create_plates,
        )

    @property
    def model(self):
        return self._model

    @property
    def guide(self):
        return self._guide

    def _train_full_data(self, x_data, obs2sample, n_epochs=20000, lr=0.002):

        idx = np.arange(x_data.shape[0]).astype("int64")

        device = torch.device("cuda")
        idx = torch.tensor(idx).to(device)
        x_data = torch.tensor(x_data).to(device)
        obs2sample = torch.tensor(obs2sample).to(device)

        self.to(device)

        pyro.clear_param_store()
        self.guide(x_data, idx, obs2sample)

        svi = SVI(
            self.model,
            self.guide,
            optim.ClippedAdam({"lr": lr, "clip_norm": 200}),
            loss=JitTrace_ELBO(),
        )

        iter_iterator = tqdm(range(n_epochs))
        hist = []
        for it in iter_iterator:

            loss = svi.step(x_data, idx, obs2sample)
            iter_iterator.set_description(
                "Epoch " + "{:d}".format(it) + ", -ELBO: " + "{:.4e}".format(loss)
            )
            hist.append(loss)

            if it % 500 == 0:
                torch.cuda.empty_cache()

        self.hist = hist

        
import pandas as pd
from scipy.sparse import csr_matrix
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
