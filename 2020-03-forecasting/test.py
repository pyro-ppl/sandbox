import numpy as np
import math
import torch
import torch.distributions.constraints as constraints

import pyro
from pyro.contrib.forecast import ForecastingModel, backtest, Forecaster
from pyro.nn import PyroParam, PyroModule
from pyro.infer.reparam import SymmetricStableReparam, StudentTReparam, LinearHMMReparam, StableReparam
from pyro.distributions import StudentT, Stable, Normal, LinearHMM

LinearHMM(self._get_init_dist(), self.trans_matrix, self._get_trans_dist(),
                                 obs_matrix_multiplier * self.obs_matrix, self._get_obs_dist(obs_scale)
