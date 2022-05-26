import skopt
from distutils.version import LooseVersion
import numpy as np
np.random.seed(2022)

def gen_dataset(bounds, N_sample, method="pseudo", ndim=1):

    if ndim == 0:
        return bounds[0] * np.ones([N_sample, 1])
    elif ndim == 1:
        if method == 'uniform':
            return np.linspace(bounds[0], bounds[1], N_sample)
        else:
            X = sample(N_sample, ndim, method)
            return X * (np.array(bounds)[1]-np.array(bounds)[0]) + np.array(bounds)[0]
    elif ndim == 2:
        if method == 'uniform':
            n = int(np.sqrt(N_sample))
            x0 = np.linspace(bounds[0, 0], bounds[0, 1], n)
            x1 = np.linspace(bounds[1, 0], bounds[1, 1], n)
            X = np.concatenate([np.tile(x0, 1, n), np.tile(x1, 1, n)], axis=-1)
            return X
        else:
            X = sample(N_sample, ndim, method)

            return X * (np.array(bounds)[1, :]-np.array(bounds)[0,:]) + np.array(bounds)[0,:]


################################ hpinns文章中生成数据方式  ################################################
def sample(n_samples, dimension, sampler="pseudo"):
    """Generate random or quasirandom samples in [0, 1]^dimension.

    Args:
        n_samples (int): The number of samples.
        dimension (int): Space dimension.
        sampler (string): One of the following: "pseudo" (pseudorandom), "LHS" (Latin
            hypercube sampling), "Halton" (Halton sequence), "Hammersley" (Hammersley
            sequence), or "Sobol" (Sobol sequence).
    """
    if sampler == "pseudo":
        return pseudo(n_samples, dimension)
    if sampler in ["LHS", "Halton", "Hammersley", "Sobol"]:
        return quasirandom(n_samples, dimension, sampler)
    raise ValueError("f{sampler} sampler is not available.")


def pseudo(n_samples, dimension):
    """Pseudo random."""
    # If random seed is set, then the rng based code always returns the same random
    # number, which may not be what we expect.
    # rng = np.random.default_rng(config.random_seed)
    # return rng.random(size=(n_samples, dimension), dtype=config.real(np))
    return np.random.random(size=(n_samples, dimension)).astype(dtype=np.float32)


def quasirandom(n_samples, dimension, sampler):
    if sampler == "LHS":
        sampler = skopt.sampler.Lhs(
            lhs_type="centered", criterion="maximin", iterations=1000
        )
    elif sampler == "Halton":
        sampler = skopt.sampler.Halton(min_skip=-1, max_skip=-1)
    elif sampler == "Hammersley":
        sampler = skopt.sampler.Hammersly(min_skip=-1, max_skip=-1)
    elif sampler == "Sobol":
        # Remove the first point [0, 0, ...] and the second point [0.5, 0.5, ...], which
        # are too special and may cause some error.
        if LooseVersion(skopt.__version__) < LooseVersion("0.9"):
            sampler = skopt.sampler.Sobol(min_skip=2, max_skip=2, randomize=False)
        else:
            sampler = skopt.sampler.Sobol(skip=0, randomize=False)
            space = [(0.0, 1.0)] * dimension
            return np.array(
                sampler.generate(space, n_samples + 2)[2:], dtype=np.float32
            )
    space = [(0.0, 1.0)] * dimension
    return np.array(sampler.generate(space, n_samples), dtype=np.float32)


