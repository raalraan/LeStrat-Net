import numpy as np


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


def get_lim_err(
    weights, target,
    fromlim=0.0,
    ntest=100,
    vtotal=None,
    nreal=None,
    verbose=0,
    tstscale=None
):
    if tstscale == 'log' or tstscale is None:
        lims_tst = np.logspace(
            np.log10(weights[weights > fromlim].min()),
            np.log10(weights.max()),
            ntest
        )
    elif tstscale == 'linear':
        lims_tst = np.linspace(
            weights[weights > fromlim].min(),
            weights.max(),
            ntest
        )
    else:
        print("'tstscale' must be 'log', 'linear' or None")
        return -1

    if nreal is None:
        nreal = len(weights)
    if vtotal is None:
        vtotal = 1.0

    theerrel = []
    l_ind = 1

    while not theerrel or theerrel[-1] < target:
        lim = lims_tst[l_ind]
        fltr = (weights > fromlim)*(weights <= lim)
        wstd_l = weights[fltr].std()
        vreg = vtotal*fltr.sum()/nreal
        theerrel.append(wstd_l*vreg)
        l_ind += 1
        if l_ind == len(lims_tst):
            print(
                "Target could not be reached.",
                "Last value is:", theerrel[-1]
            )
            return -1

    if l_ind == 2:
        print(
            "Limit found in one jump, consider increasing ntest.",
            "Used ntest={}".format(ntest)
        )

    thelim = np.interp(target, theerrel, lims_tst[1:len(theerrel) + 1])

    if verbose > 0:
        print(
            "New limit estimated using",
            ((weights > fromlim)*(weights <= thelim)).sum(),
            "out of",
            weights.shape[0]
        )
        print(
            "with weights in the range",
            fromlim, "-", thelim
        )

    return thelim


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


# TODO Add an option to increase the use the size of the largest class instead
def get_train_xy(momenta, weights, lims, largest_size=False):
    sdind, cnts = divindx(weights, lims)
    nreg = len(lims) - 1
    if largest_size:
        boored = max(cnts)
    else:
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


def to_multilabel(labels, nregs=None, activation_out=None):
    if nregs is None:
        nregs = int(labels.max() - labels.min()) + 1
    yout = np.empty((labels.shape[0], nregs - 1))
    yint = labels.astype(int)
    ymax = int(yint.max())
    if activation_out is None or activation_out == "sigmoid":
        for j in range(len(yint)):
            yout[j] = np.append(
                np.ones((yint[j][0])),
                np.zeros((ymax - yint[j][0]))
            )
    elif activation_out == "tanh":
        for j in range(len(yint)):
            yout[j] = np.append(
                np.full((yint[j][0]), 1),
                np.full((ymax - yint[j][0]), -1)
            )
    return yout


def get_sigvol_size(weights, limits):
    """Function to get variance(weights)*volume(region) according to limits"""
    sigvols = []
    sizes = []
    for j in range(len(limits) - 1):
        fltr = (weights > limits[j])*(weights <= limits[j + 1])
        sigvols.append(weights[fltr].std()*fltr.sum()/weights.shape[0])
        sizes.append(fltr.sum())

    return sigvols, sizes


def get_rdev_size(weights, limits):
    """Function to get std/average in all regions according to limits"""
    rdevs = []
    sizes = []
    for j in range(len(limits) - 1):
        fltr = (weights > limits[j])*(weights <= limits[j + 1])
        rdevs.append(weights[fltr].std()/abs(weights[fltr].mean()))
        sizes.append(fltr.sum())

    return rdevs, sizes


def get_rat_upfromval(weights, valtst, verbose=0):
    """Function to get the ratio of (number of points)*(standard deviation)
    for two regions of weights with limit valtst
    """
    fltrs = (weights <= valtst, weights > valtst)
    n1, n2 = fltrs[0].sum(), fltrs[1].sum()
    # Random uniform distribution is assumed so only n1 and n2 is enough
    # TODO If another type of distribution is used it may be necessary to
    # add also weights for the distribution
    # v1, v2 = n1/ntst, n2/ntst
    sig1, sig2 = weights[fltrs[0]].std(), weights[fltrs[1]].std()
    if verbose > 0:
        if any(np.array([n2, sig2]) == 0.0):
            print("Division by zero", [n2, sig2])

    # Random uniform distribution is assumed so only n1 and n2 is enough
    # TODO If another type of distribution is used it may be necessary to
    # add also weights for the distribution
    # ratio = v1*sig1/v2/sig2
    ratio = n1*sig1/n2/sig2
    return ratio, n1*sig1, n2*sig2


# If ratio is not close to 1, then use larger ntest
def get_lim_eqvar(
    weights,
    ntest=100, verbose=0,
    maxsteps=np.inf
):
    """Function to get a new limit that divides weights in two regions with
    equal variance
    """
    valtst = np.linspace(weights.min(), weights.max(), ntest + 2)[1:-1]
    dtst = []

    ninval = ((weights >= valtst[0])*(weights <= valtst[-1])).sum()
    if ninval < ntest:
        valtst = np.sort(
            weights[(weights >= valtst[0])*(weights <= valtst[-1])]
        )

    for j in range(valtst.shape[0]):
        ratio, nsig1, nsig2 = get_rat_upfromval(
            weights, valtst[j], verbose=verbose
        )

        if verbose > 1:
            print(
                "Test value and ratio [{}]:".format(j),
                valtst[j], ratio
            )
        if ratio > 1:
            break
        dtst.append(ratio)
    if verbose > 0:
        print("Stopped at", valtst[j], ratio)

    k = 0
    using_weights = False
    while k < maxsteps:
        try:
            invalfltr = (weights >= valtst[j - 1])*(weights <= valtst[j])
        except IndexError:
            print(
                "Not enough test values, increase `ntest`. Current value is",
                ntest
            )
        ninval = invalfltr.sum()
        if ninval == 0:
            raise Exception(
                "Not enough test values, "
                "increase `ntest`. Used ntest={}".format(ntest)
            )

        if verbose > 0:
            print(
                ninval, "in range",
                valtst[j - 1],
                valtst[j],
            )

        if ninval < ntest:
            valtst = np.sort(
                weights[invalfltr]
            )
            if using_weights or ninval == 2:
                ratio_1, nsig1_1, nsig2_1 = get_rat_upfromval(
                    weights, valtst[-2])
                ratio_2, nsig1_2, nsig2_2 = get_rat_upfromval(
                    weights, valtst[-1])

                if verbose > 0:
                    print(
                        "Interpolating new limit:",
                        "using ratio {} at {}".format(ratio_1, valtst[-2]),
                        "and ratio {} at {}".format(ratio_2, valtst[-1]),
                    )
                    print(
                        "n*sig was:\n", nsig1_1, nsig2_1, '\n',
                        nsig1_2, nsig2_2
                    )
                newlim = np.interp(1.0, [ratio_1, ratio_2], valtst[-2:])

                return newlim, 1.0
            using_weights = True
        else:
            valtst = np.linspace(valtst[j - 1], valtst[j], ntest)

        for j in range(valtst.shape[0]):
            ratio, nsig1, nsig2 = get_rat_upfromval(
                weights, valtst[j], verbose=verbose)
            if verbose > 1:
                print(
                    "Test value and ratio [{}, {}]:".format(k, j),
                    valtst[j], ratio
                )
            if ratio > 1:
                break
            dtst.append(ratio)

        if verbose > 0:
            print("Step {}:".format(k + 1), "stopped at", valtst[j], ratio)
        k += 1

    return valtst[j], ratio


def get_lim_halfs(weights, ntest=10, maxsteps=np.inf):
    """Function to get a new limit that divides weights in two regions with
    similar size
    """
    nw = weights.shape[0]

    trysize = np.linspace(weights.min(), weights.max(), ntest)
    for j in range(trysize.shape[0]):
        fraclim = (weights <= trysize[j]).sum()/nw
        if fraclim == 0.5:
            return trysize[j]
        elif fraclim > 0.5:
            break
    k = 0
    usingset = False
    while k < maxsteps:
        inlimfltr = (weights >= trysize[max(0, j - 2)])*(weights <= trysize[j])
        # print(inlimfltr.sum(), trysize[max(0, j - 2)], trysize[j])
        if inlimfltr.sum() < ntest:
            trysize = np.sort(weights[inlimfltr])
            usingset = True
            # print(trysize, usingset)

            if inlimfltr.sum() == 1:
                return trysize[0]
            elif inlimfltr.sum() == 2:
                frac1 = (weights <= trysize[-2]).sum()/nw
                frac2 = (weights <= trysize[-1]).sum()/nw
                return np.interp(0.5, [frac1, frac2], trysize)
        else:
            trysize = np.linspace(trysize[max(0, j - 2)], trysize[j], ntest)
            # print(trysize, usingset)

        for j in range(trysize.shape[0]):
            fraclim = (weights <= trysize[j]).sum()/nw
            # print(j, trysize[j], fraclim, usingset)
            if fraclim == 0.5:
                return trysize[j]
            elif fraclim > 0.5:
                break
        if usingset:
            if j == 0:
                return trysize[0]
            frac1 = (weights <= trysize[j - 1]).sum()/nw
            frac2 = (weights <= trysize[j]).sum()/nw
            return np.interp(0.5, [frac1, frac2], trysize[j-1:j+1])
        if inlimfltr.sum() <= 2:
            break
        k += 1


def get_lims_halvings(weights, steps, borders=True, prelimits=None):
    """Function to create new limits of regions for weight by a sequence of
    halving previous regions
    """
    if prelimits is None:
        lim0 = get_lim_halfs(weights)
        lims = [-np.inf, lim0, np.inf]
    else:
        lims = list(prelimits)

    for k in range(steps):
        sigsvol = []
        sigs = []
        means = []
        wsets = []
        # sizes = []
        for j in range(len(lims) - 1):
            wsets.append(weights[(weights > lims[j])*(weights <= lims[j + 1])])
            sigs.append(wsets[j].std())
            means.append(wsets[j].mean())
            sigsvol.append(sigs[j]*wsets[j].shape[0]/weights.shape[0])
            # print(k, j, "wsets.shape", wsets[j].shape)
            # sizes.append(wsets[j].shape[0])

        # print("sigsvol", sigsvol)
        # print("sizes", sizes)
        # Use volume times standard deviation
        maxsig = np.argmax(sigsvol)
        # Use relative deviation
        # maxsig = np.argmax(np.array(sigs)/np.abs(means))

        limnext = get_lim_halfs(wsets[maxsig])
        # print(limnext)
        # print("count", (weights < limnext).sum())
        # print(maxsig)
        # print(lims[:maxsig + 1], [limnext], lims[maxsig + 1:])

        lims = lims[:maxsig + 1] + [limnext] + lims[maxsig + 1:]

    if borders:
        return lims
    else:
        return lims[1:-1]


# TODO Check if I need a condition to avoid mergin limits with similar sizes
def merge_lim(limits, numbers):
    """Function to merge limits for regions where `numbers` is a number that is
    considerably smaller (order of magnitud) than the others.
    """
    limits_ls = list(limits)
    if len(limits_ls) != len(numbers) + 1:
        print("`numbers` must be 1 element larger than `limits`")
        return 0

    numbers_red = list(numbers)

    logsv = np.log10(numbers)
    logsvmean = logsv.mean()
    logsmall = logsv < logsvmean

    mergeleft = False
    if any(logsmall):
        minind = logsv.argmin()
        if minind == 0:
            mergeleft = False
        elif minind == logsv.shape[0] - 1:
            mergeleft = True
        else:
            logsv_left = logsv[minind - 1]
            logsv_right = logsv[minind + 1]
            # print(logsv_left, logsv[minind], logsv_right)
            if logsv_left < logsv_right:
                mergeleft = True
    else:
        print(
            "All the differences between regions are below merging criterion"
        )
        return limits

    if mergeleft:
        lims_red = limits_ls[:minind] + limits_ls[minind + 1:]
    else:
        lims_red = limits_ls[:minind + 1] + limits_ls[minind + 2:]

    numbers_red.remove(min(numbers_red))

    if mergeleft:
        # print("Merged left")
        regmerg = (minind, minind - 1)
    else:
        # print("Merged right")
        regmerg = (minind, minind + 1)

    if type(limits) is np.ndarray:
        return np.array(lims_red), np.array(numbers_red), regmerg
    else:
        return lims_red, numbers_red, regmerg


def get_lims_backforth(weights, steps):
    """Function to create and merge limits by halvings in a way that reduces
    sigma*volume while keeping regions large enough to have data for training
    """
    lims_rediv = get_lims_halvings(weights, steps)
    for j in range(steps):
        sigvols, sizes = get_sigvol_size(weights, lims_rediv)
        lims_rediv, _, _ = merge_lim(lims_rediv, sigvols)

    return lims_rediv


def get_dist_margin(xdata, maxcompare=100):
    """TODO (not used)"""
    dpdsv = []
    ptscompare = min(maxcompare, xdata.shape[0])
    shufind = np.arange(xdata.shape[0])
    np.random.shuffle(shufind)
    for k in range(ptscompare):
        min_dpd = np.abs(
            xdata[shufind[k]] - np.delete(xdata, (shufind[k]), axis=0)
        ).min(axis=0)
        dpdsv.append(min_dpd)
    return np.array(dpdsv).max(axis=0)


def gaussian_gen(dim, mu, sig):
    """Create a random number generator with a gaussian distribution"""
    if type(sig) is not np.ndarray:
        if type(sig) is list:
            sig_a = np.array(sig)
        else:
            sig_a = np.array([sig]*dim)
    else:
        sig_a = sig
    if type(mu) is not np.ndarray:
        if type(mu) is list:
            mu_a = np.array(mu)
        else:
            mu_a = np.array([mu]*dim)
    else:
        mu_a = mu

    norm = np.sqrt((2.0*np.pi)**dim)*sig_a.prod()

    def thegaussian(x):
        arg1 = ((x - mu_a)**2)/sig_a**2
        arg2 = -0.5*arg1.sum(axis=1)
        res = np.exp(arg2)/norm
        return res

    return thegaussian


def x_gen(ndim, lows, highs, seed=None):
    """Uniform random number generator"""
    if seed is None:
        seed = 42

    if type(lows) is float or type(lows) is int:
        lows_ = [lows]*ndim
    else:
        lows_ = lows
    if type(lows) is float or type(lows) is int:
        highs_ = [highs]*ndim
    else:
        highs_ = highs

    def _x_gen(npts=None, lows=lows_, highs=highs_):
        if npts is not None:
            return np.random.uniform(
                low=lows, high=highs,
                size=(npts, ndim)
            )
        else:
            return lows, highs

    return _x_gen
