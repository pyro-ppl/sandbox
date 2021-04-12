import numpy as np
import numpyro as pyro
import numpyro.distributions as dist
import numpyro.optim as optim
import jax.numpy as jnp
from jax import device_put
from jax import random, jit, lax

from numpyro.infer import SVI, Trace_ELBO
from numpyro.infer.autoguide import AutoNormal
from tqdm.auto import tqdm

from jax import hessian, lax, random, tree_map
from numpyro.infer import init_to_feasible, init_to_median

from functools import partial


def init_to_mean(site=None):
    """
    Initialize to the prior mean; fallback to median if mean is undefined.
    """
    if site is None:
        return partial(init_to_mean)

    try:
        # Try .mean() method.
        if site['type'] == 'sample' and not site['is_observed'] and not site['fn'].is_discrete:
            value = site["fn"].mean
            # if jnp.isnan(value):
            #    raise ValueError
            if hasattr(site["fn"], "_validate_sample"):
                site["fn"]._validate_sample(value)
            return np.array(value)
    except (NotImplementedError, ValueError):
        # Fall back to a median.
        # This is required for distributions with infinite variance, e.g. Cauchy.
        return init_to_median(site)


class LocationModelLinearDependentWMultiExperimentModel():
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
        self.m_g_shape = jnp.array(
            (self.m_g_gene_level_prior["mean"] ** 2)
            / (self.m_g_gene_level_prior["sd"] ** 2)
        )
        self.m_g_rate = jnp.array(
            self.m_g_gene_level_prior["mean"]
            / (self.m_g_gene_level_prior["sd"] ** 2)
        )
        self.m_g_mean_var = jnp.array(self.m_g_gene_level_prior["mean_var_ratio"])
        self.eps = jnp.array(1e-8)

        self.cell_state_mat = cell_state_mat
        self.cell_state = jnp.array(cell_state_mat.T)

        self.N_cells_per_location = jnp.array(self.cell_number_prior["N_cells_per_location"])
        self.factors_per_combs = jnp.array(self.cell_number_prior["factors_per_combs"])
        self.Y_combs_per_location = jnp.array(self.cell_number_prior["Y_combs_per_location"])
        self.N_cells_mean_var_ratio = jnp.array(self.cell_number_prior["N_cells_mean_var_ratio"])

        self.alpha_g_phi_hyp_prior_alpha = jnp.array(self.alpha_g_phi_hyp_prior["alpha"])
        self.alpha_g_phi_hyp_prior_beta = jnp.array(self.alpha_g_phi_hyp_prior["beta"])
        self.gene_add_alpha_hyp_prior_alpha = jnp.array(self.gene_add_alpha_hyp_prior["alpha"])
        self.gene_add_alpha_hyp_prior_beta = jnp.array(self.gene_add_alpha_hyp_prior["beta"])
        self.gene_add_mean_hyp_prior_alpha = jnp.array(self.gene_add_mean_hyp_prior["alpha"])
        self.gene_add_mean_hyp_prior_beta = jnp.array(self.gene_add_mean_hyp_prior["beta"])
        self.w_sf_mean_var_ratio_tensor = jnp.array(self.w_sf_mean_var_ratio)

        self.n_fact_tensor = jnp.array(self.n_fact)
        self.n_comb_tensor = jnp.array(self.n_comb)

        self.ones = jnp.ones((1, 1))
        self.ones_n_comb_1_1 = jnp.ones([self.n_comb, 1, 1])

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

        # obs2sample = batch_index  # one_hot(batch_index, self.n_exper)

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
            s_g_gene_add_alpha_e = self.ones / jnp.power(s_g_gene_add_alpha_e_inv, 2)
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
        theta = obs2sample @ (self.ones / jnp.power(alpha_g_inverse, 2))

        # =====================DATA likelihood ======================= #
        # Likelihood (sampling distribution) of data_target & add overdispersion via NegativeBinomial
        with obs_axis, var_axis:
            pyro.sample(
                "data_target",
                dist.GammaPoisson(concentration=theta, rate=theta / mu),
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


class LocationModelLinearDependentWMultiExperiment():

    def __init__(self, device='gpu',
                 init_loc_fn=init_to_mean, init_scale=0.1,
                 **kwargs):

        super().__init__()
        pyro.set_platform(platform=device)

        self.hist = []

        self._model = LocationModelLinearDependentWMultiExperimentModel(**kwargs)
        self._guide = AutoNormal(
            self.model.forward,
            init_loc_fn=init_loc_fn,
            init_scale=init_scale,
            create_plates=self.model.create_plates
        )

    @property
    def model(self):
        return self._model

    @property
    def guide(self):
        return self._guide

    def _train_full_data(self, x_data, obs2sample, n_epochs=20000, lr=0.002,
                         progressbar=True, random_seed=1):

        idx = np.arange(x_data.shape[0]).astype("int64")

        # move data to default device
        x_data = device_put(jnp.array(x_data))
        extra_data = {'idx': device_put(jnp.array(idx)),
                      'obs2sample': device_put(jnp.array(obs2sample))}

        # initialise SVI inference method
        svi = SVI(self.model.forward, self.guide,
                  # limit the gradient step from becoming too large
                  optim.ClippedAdam(clip_norm=jnp.array(200),
                                    **{'step_size': jnp.array(lr)}),
                  loss=Trace_ELBO())
        init_state = svi.init(random.PRNGKey(random_seed),
                              x_data=x_data, **extra_data)
        self.state = init_state

        if not progressbar:
            # Training in one step
            epochs_iterator = tqdm(range(1))
            for e in epochs_iterator:
                state, losses = lax.scan(lambda state_1, i: svi.update(state_1,
                                                                       x_data=self.x_data, **extra_data),
                                         # TODO for minibatch DataLoader goes here
                                         init_state, jnp.arange(n_epochs))
                # print(state)
                epochs_iterator.set_description('ELBO Loss: ' + '{:.4e}'.format(losses[::-1][0]))

            self.state = state
            self.hist = losses

        else:
            # training using for-loop

            jit_step_update = jit(lambda state_1: svi.update(state_1, x_data=x_data, **extra_data))
            # TODO figure out minibatch static_argnums https://github.com/pyro-ppl/numpyro/issues/869

            ### very slow
            epochs_iterator = tqdm(range(n_epochs))
            for e in epochs_iterator:
                self.state, loss = jit_step_update(self.state)
                self.hist.append(loss)
                epochs_iterator.set_description('ELBO Loss: ' + '{:.4e}'.format(loss))

        self.state_param = svi.get_params(self.state).copy()


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
