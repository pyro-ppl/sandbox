import argparse
import os

import torch

import pyro
import pyro.distributions as dist
from pyro.infer import SVI, Trace_ELBO
from pyro.infer.autoguide import AutoLowRankMultivariateNormal
from pyro.optim import ClippedAdam


class Model:
    def __init__(self, dim, rank):
        self.dim = dim
        self.rank = rank
        self.loc1 = dist.Laplace(0, 1).sample((dim,))
        self.scale1 = dist.Exponential(1).sample((dim,))
        self.loc2 = dist.Laplace(0, 1).sample((rank,))
        self.scale2 = dist.Exponential(1).sample((rank,))
        self.mat = dist.Normal(0, 1).sample((dim, rank))

    def __call__(self):
        z = pyro.sample("z",
                        dist.Normal(self.loc1, self.scale1)
                            .expand([self.dim]).to_event(1))
        pyro.sample("x",
                    dist.Normal(self.loc2, self.scale2)
                        .expand([self.rank]).to_event(1),
                    obs=z @ self.mat)


def train(args):
    model = Model(args.dim, 2 * args.rank)
    guide = AutoLowRankMultivariateNormal(model, rank=args.rank, init_scale=0.01)
    optim = ClippedAdam({"lr": args.learning_rate})
    elbo = Trace_ELBO()
    svi = SVI(model, guide, optim, elbo)
    losses = []
    for step in range(args.num_steps):
        loss = svi.step() / args.dim
        losses.append(loss)
        if step % 100 == 0:
            print("step {: >4} loss = {:0.8g}".format(step, loss))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="experiment runner")
    parser.add_argument("-d", "--dim", default=100, type=int)
    parser.add_argument("-r", "--rank", default=10, type=int)
    parser.add_argument("-n", "--num-steps", default=1000, type=int)
    parser.add_argument("-lr", "--learning-rate", default=0.01, type=float)
    args = parser.parse_args()

    train(args)
