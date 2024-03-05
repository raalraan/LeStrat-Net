import numpy as np
import matplotlib.pyplot as plt
from matplotlib.ticker import MaxNLocator


def get_lims(res, divs, res_min=0.0, tstdv=200):
    # * n2 is the first number of divisions, for the points that will be used
    #   for interpolation, maybe twice the requested divisions.
    # * res will be the resulting function value for the current set of
    #   points (probably randomly uniformly distributed)
    # * If not uniformly distributed we need to multiply by some weight
    n2 = tstdv
    rmin = np.max([res.min(), res_min])
    lims = np.linspace(rmin, res.max(), n2)
    counts = []
    wcounts = []
    # n1_nz = (res >= 1e-10).sum()
    n1_nz = (res > res_min).sum()
    for j in range(n2 - 1):
        if j == 0:
            # Correct for missing one point in the first step
            correction = 1
        else:
            correction = 0
        fltr = (res > lims[j])*(res <= lims[j + 1])
        # print(j, fltr.sum())
        counts += [(fltr.sum() + correction)/n1_nz]
        if counts[j] == 0.0:
            w = 0.0
        else:
            w = res[fltr].mean()
        wcounts += [w*counts[j]]

    cumul = [0]
    wcumul = [0]
    for j in range(len(counts)):
        cumul += [cumul[j] + counts[j]]
        wcumul += [wcumul[j] + wcounts[j]]

    dum_lims = np.linspace(0, 1, divs + 1)
    new_lims = np.interp(dum_lims, cumul, lims)

    wdum_lims = np.linspace(0, 1, divs + 1)
    wnew_lims = np.interp(wdum_lims, np.array(wcumul)/wcumul[-1], lims)

    return new_lims, wnew_lims


def get_lims2(res, divs, res_min=0.0):
    ras_i = res.argsort()
    ras = res[ras_i]
    n_p_div = int(np.round(ras.shape[0]/divs))
    lims = list(ras[(j + 1)*n_p_div - 1] for j in range(divs - 1))
    lims = [res_min] + lims
    lims = lims + [res.max()]
    return np.array(lims)


def xgenerate(n, bounds):
    bnds_here = np.array(bounds)
    x = np.random.uniform(
        bnds_here[:, 0],
        bnds_here[:, 1],
        (int(n), len(bounds))
    )
    return x


def divindx(res, lims):
    indices = np.full(res.shape, -1)
    counts = np.zeros(len(lims) - 1)
    for j in range(len(lims) - 1):
        if j == 0:
            indices[(res <= lims[1])] = 0
        elif j == len(lims) - 2:
            indices[(res > lims[j])] = j
        else:
            indices[(res > lims[j])*(res <= lims[j + 1])] = j
        counts[j] = (indices == j).sum()
    # Correct for point exactly at lims[0]
    # if (res == lims[0]).sum() == 1:
    #     counts[0] += 1
    # indices[res <= lims[0]] = 0
    # indices[res > lims[-1]] = len(lims) - 2
    return indices.reshape((indices.shape[0], 1)), counts.astype(int)


def lbins(limini, limend, nbins=100):
    return np.logspace(
        np.log10(limini),
        np.log10(limend),
        nbins + 1
    )


def get_train_xy(momenta, weights, lims):
    sdind, cnts = divindx(weights, lims)
    nreg = len(lims) - 1
    boored = int(weights.shape[0]/nreg)
    xtrain = np.empty((0, momenta.shape[1]))
    wtrain = np.empty((0))
    ytrain = np.empty((0, 1))
    for j in range(len(lims) - 1):
        shrange = np.random.permutation(np.arange(int(cnts[j])))
        test = momenta[sdind.flatten() == j][shrange[:boored]]
        testw = weights[sdind.flatten() == j][shrange[:boored]]
        if test.shape[0] < boored:
            test = np.tile(test, (int(boored/test.shape[0]) + 1, 1))
            testw = np.tile(testw, int(boored/testw.shape[0]) + 1)
        xtrain = np.append(xtrain, test[:boored], axis=0)
        ytrain = np.append(ytrain, np.array([[j]]*boored), axis=0)
        wtrain = np.append(wtrain, testw[:boored])
    return xtrain, ytrain, wtrain
