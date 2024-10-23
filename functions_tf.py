import numpy as np
from math import sqrt, exp, ceil
from functions import divindx, get_train_xy

import tensorflow as tf
from tensorflow.keras import Sequential
from tensorflow.keras.layers import Dense
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.utils import to_categorical


import gc


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
    return tf.convert_to_tensor(yout, tf.float32)


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

    tfrand = tf.random.Generator.from_seed(seed)

    def _x_gen(npts=None, lows=lows_, highs=highs_):
        if npts is not None:
            return tfrand.uniform([npts, ndim], minval=lows, maxval=highs)
        else:
            return lows, highs

    return _x_gen


def reg_pred_gen(model, data_transform=None):
    """Generate functions that use the neural network to predict regions"""
    def reg_pred(xdata=None, batch_size=int(1e6), verbose=0):
        if type(model) is list:
            nregs = len(model) + 1
            mdconf = model[0].get_config()
            # TODO This depends on trusting that model.get_config() is stable
            try:
                ndim = mdconf['layers'][0]['config']['batch_input_shape'][1]
            except KeyError:
                ndim = mdconf['layers'][0]['config']['batch_shape'][1]
        else:
            mdconf = model.get_config()
            try:
                ndim = mdconf['layers'][0]['config']['batch_input_shape'][1]
            except KeyError:
                ndim = mdconf['layers'][0]['config']['batch_shape'][1]
            activation_out = mdconf['layers'][-1]['config']['activation']
            if activation_out == 'softmax':
                nregs = mdconf['layers'][-1]['config']['units']
            elif activation_out == 'sigmoid' or activation_out == 'tanh':
                nregs = mdconf['layers'][-1]['config']['units'] + 1

        # If no input is given then just return information on dimensions and
        # regions
        if xdata is None:
            return ndim, nregs

        if data_transform is not None:
            xdata_tf = data_transform(xdata)
        else:
            xdata_tf = xdata

        # TODO Check what would be good locations for gc.collect()
        gc.collect()
        if type(model) is list:
            regres_pre = []
            modres_out = []
            for j in range(len(model)):
                # with tf.device('CPU:0'):
                # gc.collect()
                modres_here = model[j].predict(
                    xdata_tf,
                    batch_size=batch_size,
                    verbose=verbose
                )
                gc.collect()
                modres_out.append(modres_here)
                mdconf = model[j].get_config()
                activation_out = mdconf['layers'][-1]['config']['activation']
                if activation_out == 'tanh':
                    regres_pre.append(
                        np.round((modres_here + 1)/2).astype(int)
                    )
                elif activation_out == 'sigmoid':
                    regres_pre.append(np.round(modres_here).astype(int))
            regres = np.asarray(regres_pre).reshape(
                (len(model), xdata_tf.shape[0])
            ).T.sum(axis=1)
        else:
            # with tf.device('CPU:0'):
            # gc.collect()
            modres = model.predict(
                xdata_tf,
                batch_size=batch_size,
                verbose=verbose
            )
            gc.collect()
            activation_out = mdconf['layers'][-1]['config']['activation']
            if activation_out == 'tanh':
                modres_out = np.abs(modres).min(axis=1)
                regres = np.round((modres + 1)/2).sum(axis=1).astype(int)
            elif activation_out == 'sigmoid':
                modres_out = (2*np.abs(modres - 0.5)).min(axis=1)
                regres = np.round(modres).sum(axis=1).astype(int)
            elif activation_out == 'softmax':
                modres_out = (modres.max(axis=1) - 1/nregs)/(1 - 1/nregs)
                regres = np.argmax(modres, axis=1)
            else:
                raise ValueError(
                    "Unexpected value for activation function:",
                    activation_out
                )
        return regres, modres_out
    return reg_pred


# Notes: Lower ntest with large ntarget and maxiter improve estimation of
# volume
def sample_gen_nn_single_reg(
    nnfun, xgenerator, regindex, ntarget,
    ntest=None, batch_size=None, fmargin=5, maxiter=1000, maxretries=100,
    verbose=0, sample_seed=None, Vrerror_target=None
):
    """Generate a sample for a single region using a neural network"""
    if ntarget == 0 and Vrerror_target is None:
        return [], 0, 0

    if Vrerror_target is None:
        Vrerror_target = np.inf

    orig_min, orig_max = xgenerator()
    Vtot = (np.array(orig_max) - np.array(orig_min)).prod()

    if ntest is None:
        ntest = ntarget
    if batch_size is None:
        batch_size = min(int(1e6), ntarget)

    if sample_seed is not None:
        if sample_seed[2] is None:
            seed_nnregs, seed_nnconf = nnfun(
                sample_seed[0], batch_size=batch_size
            )
        else:
            seed_nnregs = sample_seed[2]
        sd_rfltr = seed_nnregs == regindex
        sample_seed_inreg = sample_seed[0][sd_rfltr]

    xpre_dum = xgenerator(ntest)
    # ntried = ntest
    ntried = xpre_dum.shape[0]
    nnregs, nnconf = nnfun(xpre_dum, batch_size=batch_size)

    if not isinstance(xpre_dum, np.ndarray):
        xpre_dum = xpre_dum.numpy()

    rfltr = nnregs == regindex
    retries = 0
    while rfltr.sum() < 1:
        if verbose > 0:
            print(
                "Not enough points found in region:",
                rfltr.sum(),
                "in",
                ntried
            )
        xpre_dum_r = xgenerator(ntest)
        # ntried += ntest
        ntried += xpre_dum_r.shape[0]
        nnregs_r, nnconf_r = nnfun(xpre_dum_r, batch_size=batch_size)
        retries += 1

        if not isinstance(xpre_dum_r, np.ndarray):
            xpre_dum_r = xpre_dum_r.numpy()

        if retries == maxretries:
            print(
                "Region {}:".format(regindex),
                "getting 1 points in region took longer than maxretries.",
                "Giving up.",
                "`maxretries` is:", maxretries
            )
            if sample_seed is None or sd_rfltr.sum() == 0:
                return np.empty((0, xpre_dum_r.shape[1])), 0, 0
            else:
                print(
                    "Will use points from `sample_seed` ",
                    "to determine test limits"
                )

        xpre_dum = np.append(
            xpre_dum,
            xpre_dum_r,
            axis=0
        )
        nnregs = np.append(
            nnregs,
            nnregs_r
        )
        nnconf = np.append(
            nnconf,
            nnconf_r
        )
        rfltr = nnregs == regindex

    if rfltr.sum() > 0 and sample_seed is None:
        xmaxs_pre = xpre_dum[rfltr].max(axis=0)
        xmins_pre = xpre_dum[rfltr].min(axis=0)
    elif rfltr.sum() > 0 and sample_seed is not None and sd_rfltr.sum() > 0:
        xmaxs_pre0 = xpre_dum[rfltr].max(axis=0)
        xmins_pre0 = xpre_dum[rfltr].min(axis=0)
        xmaxs_pre1 = sample_seed_inreg.max(axis=0)
        xmins_pre1 = sample_seed_inreg.min(axis=0)
        xmaxs_pre = np.max(np.array([
            xmaxs_pre0,
            xmaxs_pre1
        ]), axis=0)
        xmins_pre = np.min(np.array([
            xmins_pre0,
            xmins_pre1
        ]), axis=0)
    elif rfltr.sum() == 0 and sample_seed is not None and sd_rfltr.sum() > 0:
        xmaxs_pre = sample_seed_inreg.max(axis=0)
        xmins_pre = sample_seed_inreg.min(axis=0)
    else:
        print(
            "Cannot use `sample_seed` to optimize test limits.",
            "There were no points in region."
        )
        return np.empty((0, xpre_dum_r.shape[1])), 0, 0

    xaccumul = xpre_dum[rfltr]

    nin = xaccumul.shape[0]
    in_tried_r = nin/ntried
    Verror = Vtot*sqrt((in_tried_r - in_tried_r**2)/ntried)

    if verbose > 0:
        print(
            "Attempt 0:",
            "Volume (this run, total):",
            Vtot*(nnregs == regindex).sum()/ntried,
            Vtot*nin/ntried,
            "points found",
            xaccumul.shape[0]
        )

    new_min = np.max(np.array([
        orig_min + np.abs(np.array(orig_min) - np.array(xmins_pre))/fmargin,
        orig_min
    ]), axis=0)
    new_max = np.min(np.array([
        orig_max - np.abs(np.array(orig_max) - np.array(xmaxs_pre))/fmargin,
        orig_max
    ]), axis=0)

    if xaccumul.shape[0] < ntarget:
        for j in range(maxiter):
            xpre1_dum = xgenerator(ntest, new_min, new_max)
            # Effective number of tried points if I had used the full space
            # ntreff = ntest*Vtot/(new_max - new_min).prod()
            ntreff = xpre1_dum.shape[0]*Vtot/(new_max - new_min).prod()
            ntried += ntreff
            nnregs1, nnconf1 = nnfun(xpre1_dum, batch_size=batch_size)

            if not isinstance(xpre1_dum, np.ndarray):
                xpre1_dum = xpre1_dum.numpy()

            rfltr1 = nnregs1 == regindex
            while rfltr1.sum() < 1:
                xpre1_dum_r = xgenerator(ntest, new_min, new_max)
                # ntried += ntest*Vtot/(new_max - new_min).prod()
                # ntreff += ntest*Vtot/(new_max - new_min).prod()
                ntried += xpre1_dum_r.shape[0]*Vtot/(new_max - new_min).prod()
                ntreff += ntried
                nnregs1_r, nnconf1_r = nnfun(
                    xpre1_dum_r,
                    batch_size=batch_size
                )

                if not isinstance(xpre1_dum_r, np.ndarray):
                    xpre1_dum_r = xpre1_dum_r.numpy()

                xpre1_dum = np.append(
                    xpre1_dum,
                    xpre1_dum_r,
                    axis=0
                )
                nnregs1 = np.append(
                    nnregs1,
                    nnregs1_r
                )
                nnconf1 = np.append(
                    nnconf1,
                    nnconf1_r
                )
                rfltr1 = nnregs1 == regindex

            nin += rfltr1.sum()

            if xaccumul.shape[0] < ntarget:
                xaccumul = np.append(
                    xaccumul,
                    xpre1_dum[rfltr1],
                    axis=0
                )

            if verbose > 0:
                print(
                    "Attempt {}:".format(j + 1),
                    "Volume (this run, total):",
                    Vtot*(nnregs1 == regindex).sum()/ntreff,
                    Vtot*nin/ntried,
                    "points found",
                    nin
                )

            # print(
            #     "got {} points in region".format((nnregs1 == regindex).sum())
            # )
            xmaxs1_pre = np.max(np.array([
                xpre1_dum[rfltr1].max(axis=0),
                xmaxs_pre
            ]), axis=0)
            xmins1_pre = np.min(np.array([
                xpre1_dum[rfltr1].min(axis=0),
                xmins_pre
            ]), axis=0)

            new_min = np.max(np.array([
                new_min + np.abs(
                    np.array(new_min) - np.array(xmins1_pre)
                )/fmargin,
                new_min
            ]), axis=0)
            new_max = np.min(np.array([
                new_max - np.abs(
                    np.array(new_max) - np.array(xmaxs1_pre)
                )/fmargin,
                new_max
            ]), axis=0)

            xmaxs_pre = np.copy(xmaxs1_pre)
            xmins_pre = np.copy(xmins1_pre)

            in_tried_r = nin/ntried
            Verror = Vtot*sqrt((in_tried_r - in_tried_r**2)/ntried)

            # print(nin, Verror/Vtot/in_tried_r, Vrerror_target)
            if nin >= ntarget and Verror/Vtot/in_tried_r < Vrerror_target:
                break

        if j == maxiter - 1:
            print("Reached `maxiter`:", maxiter)

    # print(Vot, in_tried_r, ntried, Verror, Verror/Vtot/in_tried_r)
    return xaccumul[:ntarget], Vtot*in_tried_r, Verror


# nnfun: a function that returns number of region based on neural network,
# generated with the function reg_pred_gen
# Output is:
#   Parameter space points, list of arrays: One array per region as classified
#     by the network.
#   Network confidence, list of arrays: One array per region as classified by
#     the network.
#   Volumes for regions, list.
#   Number of points tested, list.
def sample_gen_nn(
    nnfun, xgenerator, npts, ntest,
    batch_size=int(1e6), Vtot=None, verbose=0
):
    """Generate a sample for all regions using a neural network"""
    if Vtot is None:
        orig_min, orig_max = xgenerator()
        Vtot = (np.array(orig_max) - np.array(orig_min)).prod()

    D, nregs = nnfun()
    if type(npts) is int:
        npts_ls = [npts]*nregs
    else:
        npts_ls = npts
    # xpre = np.random.uniform(
    #     low=0, high=xside, size=(ntest, D)
    # )
    xpre = xgenerator(ntest)
    regs_pred, xconfs = nnfun(xpre, batch_size=batch_size)
    xaccumul = [np.empty((0, D)) for j in range(nregs)]
    xconfs_out = [np.empty((0,)) for j in range(nregs)]

    if not isinstance(xpre, np.ndarray):
        xpre = xpre.numpy()

    # TODO use np.empty instead
    vols = [None]*nregs
    found = [None]*nregs
    tried = [None]*nregs
    for j in range(nregs):
        xaccumul[j] = xpre[regs_pred == j][:npts_ls[j]]
        xconfs_out[j] = xconfs[regs_pred == j][:npts_ls[j]]
        found[j] = (regs_pred == j).sum()
        tried[j] = xpre.shape[0]
        vols[j] = Vtot*found[j]/tried[j]
    isdone = [xaccumul[k].shape[0] >= npts_ls[k] for k in range(nregs)]
    # print(isdone)
    while not all(isdone):
        xpre = xgenerator(ntest)
        regs_pred, xconfs = nnfun(xpre)

        if not isinstance(xpre, np.ndarray):
            xpre = xpre.numpy()

        for j in range(nregs):
            if not isdone[j]:
                xconfs_out[j] = np.append(
                    xconfs_out[j],
                    xconfs[regs_pred == j][:npts_ls[j] - xaccumul[j].shape[0]]
                )
                xaccumul[j] = np.append(
                    xaccumul[j],
                    xpre[regs_pred == j][:npts_ls[j] - xaccumul[j].shape[0]],
                    axis=0
                )
            found[j] += (regs_pred == j).sum()
            tried[j] += xpre.shape[0]
            vols[j] = Vtot*found[j]/tried[j]
            # print(j, xaccumul[j].shape[0], vols[j], found[j], tried[j])
        isdone = [xaccumul[k].shape[0] >= npts_ls[k] for k in range(nregs)]
        # print(isdone)
    return xaccumul, xconfs_out, vols, tried


def sample_gen_nn2(
    nnfun, xgenerator, npts, ntest,
    batch_size=int(1e6), verbose=0,
    maxiter=1000, fmargin=5, Vtot=None,
    sample_seed=None, maxretries=100, Vrerror_target=None
):
    """Generate a sample for all regions using a neural network.  Simpler
    version that uses the single region generator.  A little inefficient.
    """
    if Vtot is None:
        orig_min, orig_max = xgenerator()
        Vtot = (np.array(orig_max) - np.array(orig_min)).prod()

    D, nregs = nnfun()
    if type(npts) is int:
        npts_ls = [npts]*nregs
    else:
        npts_ls = npts

    xaccumul = []
    vols = []
    volserror = []
    for j in range(nregs):
        if verbose > 0:
            print("Region", j)
        sam_gen = sample_gen_nn_single_reg(
            nnfun, xgenerator, j, npts_ls[j],
            ntest=ntest, batch_size=batch_size, fmargin=fmargin,
            maxiter=maxiter, verbose=verbose, sample_seed=sample_seed,
            maxretries=maxretries, Vrerror_target=Vrerror_target
        )
        xaccumul.append(sam_gen[0])
        vols.append(sam_gen[1])
        volserror.append(sam_gen[2])

    vols = np.array(vols)
    # TODO What to do when total volume is estimated different from true
    # volume which is known

    return xaccumul, vols, volserror


def sample_gen_nn3(
    nnfun, xgenerator, npts, ntest,
    batch_size=None, verbose=0,
    maxiter=1000, exp_den=10, Vtot=None,
    sample_seed=None, maxretries=100, Vrerror_target=None
):
    """Generate a sample for all regions using a neural network.  Improved
    for cases where some regions are surrounded by other regions.
    """
    orig_min, orig_max = xgenerator()
    Vtot = (np.array(orig_max) - np.array(orig_min)).prod()
    D, nregs = nnfun()

    if Vrerror_target is None:
        Vrerror_target = np.inf

    if batch_size is None:
        batch_size = min(int(1e6), npts)

    if type(npts) in [float, int]:
        npts_ls = [npts]*nregs
    elif len(npts) == 1:
        npts_ls = [npts[0]]*nregs
    elif len(npts) != nregs:
        raise ValueError(
            'The number of requested points in '
            '"npts" should be a single number'
            'or a list/array with one element per region: {}'.format(nregs)
        )
    else:
        npts_ls = npts

    if type(Vrerror_target) in [float, int]:
        Vrerror_target_ls = [Vrerror_target]*nregs
    elif len(Vrerror_target) == 1:
        Vrerror_target_ls = [Vrerror_target[0]]*nregs
    elif len(Vrerror_target) != nregs:
        raise ValueError(
            'The number of values in relative error targets '
            '"Vrerror_target" should be a single number '
            'or a list/array with one element per region: {}'.format(nregs)
        )
    else:
        Vrerror_target_ls = Vrerror_target

    xaccumul = [np.empty((0, D)) for j in range(nregs)]

    # 1. First full sample
    xdum = xgenerator(ntest)
    rdum, _ = nnfun(xdum, batch_size=batch_size)

    # 2.1 Determine number of points in region
    nin = np.fromiter((
        (rdum == k).sum() for k in range(nregs)
    ), float)
    # rlrgst = nin.argmax()

    for k in range(nregs):
        if nin[k] > 0:
            xaccumul[k] = np.append(
                xaccumul[k],
                xdum[rdum == k][:npts_ls[k]],
                axis=0
            )

    # 2.2 Determine number of tried points
    ntried = np.ones([nregs])*xdum.shape[0]

    intst_rat = nin/xdum.shape[0]

    # 2.3 Determine volumes
    Vest = Vtot*intst_rat

    # 2.4 Determine variance
    Vevar = Vtot**2/xdum.shape[0]*(intst_rat - (intst_rat)**2)

    # 2.4.1 Determine relative error
    rerr = Vevar**0.5/Vest
    rerr[Vest == 0] = np.inf

    # 3. Determine cubes limits and volumes
    cubes = [None]*nregs
    Vcubes = np.empty([nregs])
    for k in range(nregs):
        if nin[k] > 0:
            cubes[k] = [
                xdum[rdum == k].numpy().min(axis=0),
                xdum[rdum == k].numpy().max(axis=0)
            ]
            Vcubes[k] = (cubes[k][1] - cubes[k][0]).prod()
        else:
            cubes[k] = [
                -1,
                -1
            ]
            Vcubes[k] = 0

    # 4. Determine the largest cube
    clrgst = Vcubes.argmax()

    # Direction of refinement: +1 if refining upwards, -1 if refining downwards
    for dirct in [+1, -1]:
        if dirct == +1:
            regseq = range(clrgst, nregs)
        else:
            regseq = range(clrgst, -1, -1)

        for regref in regseq:
            vregref = regref
            if verbose > 0:
                print(
                    "Region", vregref,
                    "rel. error {:.3e}".format(rerr[vregref]),
                    "target {:.3e}".format(Vrerror_target_ls[vregref]),
                    "pts accumulated",
                    xaccumul[vregref].shape[0]
                )
                # print(Vcubes)
            thistry = 0
            while (
                (
                    rerr[vregref] > Vrerror_target_ls[vregref]
                    or xaccumul[vregref].shape[0] < npts_ls[vregref]
                )
                and thistry < maxretries
            ):
                # 5. Determine the estimated limits of largest cube
                if vregref != clrgst:
                    mins = cubes[vregref][0]
                    maxs = cubes[vregref][1]
                    mins_prev = cubes[vregref - dirct*1][0]
                    maxs_prev = cubes[vregref - dirct*1][1]
                    fac_h = (1 - exp(-(thistry/exp_den)**3))
                    new_min = mins_prev + (mins - mins_prev)*fac_h
                    new_max = maxs_prev + (maxs - maxs_prev)*fac_h
                    # print(mins, '\n', new_min)
                    # print(maxs, '\n', new_max)
                else:
                    new_min = np.array(orig_min)
                    new_max = np.array(orig_max)

                # 6. Determine regions without points outside cube and exclude
                # empty regions
                nout_cube = np.empty([nregs])
                for k in range(nregs):
                    if nin[k] > 0:
                        nout_cube[k] = (
                            np.any(xdum[rdum == k].numpy() < new_min, axis=1)
                            * np.any(xdum[rdum == k].numpy() > new_max, axis=1)
                        ).sum()
                    else:
                        nout_cube[k] = -1

                # 7. Resample with new limits
                xdum_next = xgenerator(ntest, new_min, new_max)
                ntreff = xdum_next.shape[0]*Vtot/(new_max - new_min).prod()
                rdum_next, _ = nnfun(xdum_next, batch_size=batch_size)

                nin_next = np.fromiter((
                    (rdum_next == k).sum() for k in range(nregs)
                ), float)

                if all(new_min == orig_min) and all(new_max == orig_max):
                    regs_2_ref = range(nregs)
                elif vregref > clrgst:
                    regs_2_ref = range(vregref, nregs)
                elif vregref < clrgst:
                    regs_2_ref = range(vregref + 1)

                for k in regs_2_ref:
                    # if nout_cube[k] == 0:
                    nin[k] += nin_next[k]
                    ntried[k] += ntreff
                    Vest[k] = Vtot*nin[k]/ntried[k]
                    Vevar[k] = Vtot**2/ntried[k]*(
                        nin[k]/ntried[k] - (nin[k]/ntried[k])**2)
                    if Vest[k] == 0:
                        rerr[k] = np.inf
                    else:
                        rerr[k] = Vevar[k]**0.5/Vest[k]

                    # CUBES ============
                    if nin_next[k] > 0:
                        if type(cubes[k][0]) is int and cubes[k][0] == -1:
                            cubes[k] = [
                                xdum_next[rdum_next == k].numpy().min(axis=0),
                                xdum_next[rdum_next == k].numpy().max(axis=0)
                            ]
                        else:
                            cube_min = np.min([
                                xdum_next[rdum_next == k].numpy().min(axis=0),
                                cubes[k][0]
                            ], axis=0)
                            cube_max = np.max([
                                xdum_next[rdum_next == k].numpy().max(axis=0),
                                cubes[k][1]
                            ], axis=0)
                            cubes[k] = [
                                cube_min,
                                cube_max
                            ]
                        Vcubes[k] = (cubes[k][1] - cubes[k][0]).prod()

                        # Also accumulate points
                        if xaccumul[k].shape[0] < npts_ls[k]:
                            mss2npts = npts_ls[k] - xaccumul[k].shape[0]
                            xaccumul[k] = np.append(
                                xaccumul[k],
                                xdum_next[rdum_next == k][:mss2npts],
                                axis=0
                            )
                    # ==================

                thistry += 1
                if verbose > 0:
                    print(
                        "Region", vregref,
                        "rel. error {:.3e}".format(rerr[vregref]),
                        "target {:.3e}".format(Vrerror_target_ls[vregref]),
                        "pts accumulated",
                        xaccumul[vregref].shape[0]
                    )
                    # print(Vcubes)
    return xaccumul, Vest, Vevar**0.5


# TODO Replace xside by something related to full integration space
def sample_integrate(nnfun, ffun, xgenerator, nptsreg, ntest, Vtot=None):
    if Vtot is None:
        orig_min, orig_max = xgenerator()
        Vtot = (np.array(orig_max) - np.array(orig_min)).prod()

    xacc, _, vols, tried = sample_gen_nn(
        nnfun, xgenerator, nptsreg, ntest, Vtot=Vtot
    )
    """Integrate function using a sample from the neural network.
    STILL NOT THE FINAL VERSION
    """

    contribs = np.empty((len(vols),))
    varncs = np.empty((len(vols),))
    # sig2_tst = np.empty((len(vols),))
    # varncs_tst = np.empty((len(vols),))
    nvals = np.empty((len(vols),))
    for j in range(len(vols)):
        fvals = ffun(xacc[j])
        nval = xacc[j].shape[0]
        contribs[j] = vols[j]*fvals.mean()
        varncs[j] = vols[j]**2*fvals.var()/nval
        # sig2_tst[j] = np.sum(fvals**2)/nval - (np.sum(fvals)/nval)**2
        # varncs_tst[j] = vols[j]**2*sig2_tst[j]/nval
        nvals[j] = nval
    int_res = np.sum(contribs)
    err_res = np.sum(varncs)**0.5
    # err_res2 = np.sum(varncs_tst)**0.5
    return int_res, err_res, contribs, varncs, nvals, vols, tried, xacc


# sample_seed is an initial sample that may have been obtained from previous
# runs
def sample_integrate2(
    nnfun, ffun, xgenerator, nptsreg, ntest,
    Vtot=None, maxiter=1000, sample_seed=None,
    maxretries=100, Vrerror_target=None
):
    """Integrate function using a sample from the neural network.
    STILL NOT THE FINAL VERSION
    """
    if type(nptsreg) is list:
        nptsreg_try = np.array(nptsreg)
    else:
        nptsreg_try = nptsreg

    this_sample = sample_gen_nn2(
        nnfun, xgenerator, nptsreg_try, ntest,
        maxiter=maxiter, fmargin=50,
        sample_seed=sample_seed, maxretries=maxretries,
        Vrerror_target=Vrerror_target
    )

    vols = this_sample[1]
    volserr = this_sample[2]
    nregs = len(vols)
    wout = []

    if type(nptsreg) is int:
        nptsreg_ls = [nptsreg]*nregs
    else:
        nptsreg_ls = nptsreg

    contribs = np.empty((len(vols),))
    varncs = np.empty((len(vols),))
    sample_res = []
    for j in range(nregs):
        # Randomize sample
        shufind = np.arange(this_sample[0][j].shape[0])
        np.random.shuffle(shufind)
        wout.append(ffun(this_sample[0][j][shufind[:nptsreg_ls[j]]]))
        sample_res.append(this_sample[0][j][shufind[:nptsreg_ls[j]]])
        contribs[j] = wout[j].mean()*vols[j]
        varncs[j] = wout[j].var()*vols[j]**2

    int_res = np.sum(contribs)
    err_res = (np.sum(varncs/np.array(nptsreg_ls)))**0.5

    return int_res, err_res, contribs, varncs**0.5, vols, volserr, \
        sample_res, wout


# CUSTOM METRICS
def my_metric(name='jumping', activation_out='sigmoid'):
    def _my_metric(y_true, y_pred):

        if activation_out == 'sigmoid':
            y_true_r = tf.math.round(y_true)
            y_pred_r = tf.math.round(y_pred)
            r_true = tf.math.reduce_sum(y_true_r, axis=1)
            r_pred = tf.math.reduce_sum(y_pred_r, axis=1)
        elif activation_out == 'tanh':
            y_true_r = tf.math.round((y_true + 1)/2)
            y_pred_r = tf.math.round((y_pred + 1)/2)
            r_true = tf.math.reduce_sum(y_true_r, axis=1)
            r_pred = tf.math.reduce_sum(y_pred_r, axis=1)
        elif activation_out == 'softmax':
            r_true = tf.cast(tf.math.argmax(y_true, axis=1), y_true.dtype)
            r_pred = tf.cast(tf.math.argmax(y_pred, axis=1), y_true.dtype)

        r_diff = r_true - r_pred
        if name == 'jumping':
            r_adiff = tf.math.abs(r_diff)
            res = tf.math.reduce_mean(r_adiff)
        elif name == 'accuracy':
            r_ok = tf.cast(r_true == r_pred, tf.float32)
            res = tf.math.reduce_mean(r_ok)
        return res

    _my_metric.__name__ = name

    return _my_metric


def model_create(
    dimensions,
    activation_out,
    nodes_start,
    nodes_out,
    loss,
    learning_rate=0.001,
    use_metrics=False
):
    """Create a model based on parameters"""
    D = dimensions
    model = Sequential()
    model.add(Dense(nodes_start, input_shape=(D,), activation='relu'))
    model.add(Dense(int(nodes_start/2 + 0.5), activation='relu'))
    model.add(Dense(nodes_out, activation=activation_out))
    if use_metrics:
        model.compile(
            optimizer=Adam(learning_rate=learning_rate),
            loss=loss,
            metrics=[
                my_metric('accuracy', activation_out),
                my_metric('jumping', activation_out)
            ]
        )
    else:
        model.compile(
            optimizer=Adam(learning_rate=learning_rate),
            loss=loss
        )

    return model


# Confusion matrix
def conf_mat(r_true, r_pred, normalize=False):
    """Create confusion matrix from true and predicted labels"""
    rtmax = int(r_true.max())
    rpmax = int(tf.math.reduce_max(r_pred))
    normit = 1.0
    dumm1 = tf.fill(r_pred.shape, -1.0)
    conf_mat = np.empty((rtmax + 1, rpmax + 1))
    r_true_rs = r_true.reshape((-1,))
    for i in range(rtmax + 1):
        if normalize:
            normit = (r_true_rs == i).sum()
        rpredthis = tf.where(
            r_true_rs == i,
            r_pred,
            dumm1
        )
        for j in range(rpmax + 1):
            conf_mat[i, j] = tf.math.reduce_sum(
                tf.cast(rpredthis == j, dtype=rpredthis.dtype)
            )/normit
    return conf_mat


# TODO This should probably go into a plotting submodule
# def plt_confusion_matrix(
#     ax, thematrix,
#     cmap=None, overlay_values=False, ticks_step=1
# ):
#     nrows, ncols = thematrix.shape
#     pcmesh = ax.pcolormesh(
#         np.arange(0, ncols),
#         np.arange(0, nrows),
#         thematrix,
#         cmap=cmap,
#         antialiased=False,
#         snap=True,
#         rasterized=False
#     )
#     if overlay_values:
#         for i in range(nrows):
#             for j in range(ncols):
#                 ax.text(j, i, "{:.2f}".format(thematrix[i, j]),
#                         ha="center", va="center", color="w")

#     ax.set_yticks(
#         np.arange(0, nrows, ticks_step),
#         np.arange(0, nrows, ticks_step)
#     )
#     ax.set_xticks(
#         np.arange(0, ncols, ticks_step),
#         np.arange(0, ncols, ticks_step)
#     )

#     ax.set_ylim(ax.get_ylim()[::-1])
#     ax.tick_params(top=True, labeltop=True, bottom=False, labelbottom=False)
#     ax.xaxis.set_label_position("top")
#     return pcmesh


# Batched prediction using calls to model instead of keras model.predict()
# interface. NOT USED
def my_predict(
    model, data,
    data_transform=None, batch_size=100,
    device=None, gc_every=5, verbose=0
):
    gc.collect()

    if device is None:
        device = tf.config.list_physical_devices(device_type='GPU')[0][0][17:]

    data_size = data.shape[0]
    nbatches = ceil(data_size/batch_size)

    if data_transform is not None:
        data_tr = data_transform(data)
    else:
        data_tr = data

    with tf.device(device):
        rpred = []
        for k in range(nbatches):
            start_at = k*batch_size
            end_at = (k + 1)*batch_size
            if verbose > 0:
                print(
                    start_at, "-", end_at,
                    "{}%".format(int(100*end_at/data_size)), end='\n'
                )
            rpred.append(model(data_tr[start_at:end_at]))
            if (k % gc_every) == (gc_every - 1):
                # print(
                #     k, "performing garbage collection",
                # )
                gc.collect()
        if verbose > 0:
            print("\n")
        fullpred = tf.concat(rpred, axis=0)
    gc.collect()

    return fullpred


# Function for repeated training of network, with restart of the network if
# necessary
def train_sel(xpool, weights, limits, ntrain, verbose=0):
    # TODO This assumes the first and last limits are extreme like +-np.inf
    # limdists = np.abs(
    #     np.array([limits[1:-1]]*len(weights)) - weights.reshape(-1, 1)
    # )
    # limdists_min = limdists.min(axis=1)

    # # PREPARE TRAINING DATA
    # xsel = xpool[np.argsort(limdists_min)][:int(ntrain/2)]
    # wsel = weights[np.argsort(limdists_min)][:int(ntrain/2)]
    # xsel = np.append(
    #     xsel,
    #     xpool[np.argsort(limdists_min)][-ntrain + int(ntrain/2):],
    #     axis=0
    # )
    # wsel = np.append(
    #     wsel,
    #     weights[np.argsort(limdists_min)][-ntrain + int(ntrain/2):]
    # )

    # if verbose > 0:
    #     for j in range(len(limits) - 1):
    #         print(
    #             "points in region {}:".format(j),
    #             ((wsel >= limits[j])*(wsel < limits[j + 1])).sum()
    #         )

    gind, gcnt = divindx(weights, limits)
    xsel = np.empty((0, xpool.shape[1]))
    wsel = np.empty((0,))

    for j in range(len(limits) - 1):
        xsel = np.append(
            xsel,
            xpool[(gind == j).flatten()][:int(ntrain/(len(limits) - 1))],
            axis=0
        )
        wsel = np.append(
            wsel,
            weights[(gind == j).flatten()][:int(ntrain/(len(limits) - 1))]
        )

    xgetind = get_train_xy(xsel, wsel, limits)

    # xtrain = np.append(
    #     xgetind[0][xgetind[1].flatten() == 0][:int(ntrain/2)],
    #     xgetind[0][xgetind[1].flatten() == 1][:int(ntrain/2)],
    #     axis=0
    # )
    # ytrain = np.append(
    #     xgetind[1][xgetind[1].flatten() == 0][:int(ntrain/2)],
    #     xgetind[1][xgetind[1].flatten() == 1][:int(ntrain/2)],
    #     axis=0
    # )

    xtrain = xgetind[0]
    ytrain = xgetind[1]
    wtrain = xgetind[2]

    return xtrain, ytrain, wtrain


def retrain_sel(
    nnfun, xpool, weights, trueregs,
    ffun, xgenerator, limits, nadd,
    verbose=0, maxiter=1000, sample_seed=None,
    maxretries=100
):
    guessregs, confguess = nnfun(xpool)
    arewrong = trueregs.flatten() - guessregs != 0
    wrongfrac = arewrong.sum()/arewrong.shape[0]

    print(
        "Fraction of wrong classifications:",
        wrongfrac
    )

    if arewrong.sum() == 0:
        print(
            "All the points in pool have been correctly classified.\n",
            "Remember to check for overfitting."
        )
        return 0, 0, 0

    # limits of guessed regions
    # Lower limits
    llo = np.array(limits[:-1])[guessregs.flatten()]
    # Higher limits
    lhi = np.array(limits[1:])[guessregs.flatten()]

    # Distances between limits of guessed regions and weight
    min_max_dist = np.abs(
        np.array([llo, lhi]).T - weights.reshape(-1, 1)
    )
    # Only keep distances for wrong classifications
    gldist = min_max_dist.min(axis=1)[arewrong]
    if gldist.min() == 0.0:
        # Shift all distances if some point happens to be exactly at the limit
        gldist = gldist + gldist[gldist > 0].min()

    toappend = gldist/gldist.max() + gldist.min()/gldist

    repar = np.round(nadd*toappend/toappend.sum()).astype(int)

    fac = 1
    # In case the actual number of points selected for retraining is zero even
    # though there are wrong classifications
    while repar.sum() < nadd:
        fac *= 2
        repar = np.round(fac*nadd*toappend/toappend.sum()).astype(int)

    xretrain = np.repeat(xpool[arewrong], repar, axis=0)
    # Weight could also be useful for further selection of points
    wretrain = np.repeat(weights[arewrong], repar, axis=0)
    yretrain = np.repeat(trueregs[arewrong], repar, axis=0)

    if wrongfrac < 2/3 and xgenerator is not None:
        # morepts = sample_gen_nn2(
        #     nnfun, xgenerator, nadd, int(2*nadd*100),
        #     batch_size=int(2*nadd*10), maxiter=maxiter,
        #     verbose=verbose, sample_seed=sample_seed,
        #     maxretries=maxretries
        # )

        if sample_seed[2] is None:
            seedregs, seedconf = nnfun(sample_seed[0])
        else:
            seedregs = sample_seed[2]
        morepts = [None]*seedregs.max()
        morewgh = [None]*seedregs.max()
        for j in range(seedregs.max()):
            shufind = np.arange((seedregs == j).sum())
            np.random.shuffle(shufind)
            morepts[j] = sample_seed[0][seedregs == j][shufind[:nadd]]
            morewgh[j] = sample_seed[1][seedregs == j][shufind[:nadd]]

        # xmpts = np.concatenate(morepts[0], axis=0)
        # wmpts = ffun(xmpts)
        # ympts, _ = divindx(wmpts, limits)

        xmpts = np.concatenate(morepts, axis=0)
        wmpts = np.concatenate(morewgh)
        ympts, _ = divindx(wmpts, limits)

        wrongfltr = ympts.flatten() != nnfun(xmpts)[0]

        xretrain = np.append(xretrain, xmpts[wrongfltr], axis=0)
        # Weight could also be useful for further selection of points
        wretrain = np.append(wretrain, wmpts[wrongfltr])
        yretrain = np.append(yretrain, ympts[wrongfltr], axis=0)

    return xretrain, yretrain, wretrain


def model_fit(
    xpool, weights, ffun, limits, xtrain_size,
    activation_out, loss,
    xgenerator=None,
    data_transform=None,
    batch_size=int(1e6),
    epochs_part=1000, ntrains=5,
    nadd_retrain=1000,
    verbose=0,
    learning_rate=0.001,
    model_restart=False,
    use_metrics=False,
    sample_seed=None,
    maxretries=100,
    callbacks=None
):
    # global weights_as_tensor

    if sample_seed is None:
        sample_seed = (xpool, weights, None)

    ndim = xpool.shape[1]
    nreg = len(limits) - 1
    regs, _ = divindx(weights, limits)

    if activation_out == 'softmax':
        nodes_out = nreg
    else:
        nodes_out = nreg - 1

    xtrain, ytrain, wtrain = train_sel(xpool, weights, limits, xtrain_size)

    # MODEL STUFF
    this_mdl = model_create(
        ndim, activation_out, nreg*2*ndim, nodes_out, loss,
        learning_rate=learning_rate, use_metrics=use_metrics
    )

    if activation_out == 'softmax':
        ytrain_trans = to_categorical(ytrain)
    else:
        ytrain_trans = to_multilabel(ytrain, activation_out=activation_out)

    # weights_as_tensor = tf.convert_to_tensor(wtrain, dtype=tf.float32)

    print("Training with", xtrain.shape[0], "data points")
    this_mdl.fit(
        data_transform(xtrain),
        ytrain_trans,
        epochs=epochs_part,
        batch_size=batch_size,
        verbose=verbose,
        callbacks=callbacks
    )

    nnfun = reg_pred_gen(this_mdl, data_transform)

    if ntrains > 1:
        xretr = np.copy(xtrain)
        yretr = np.copy(ytrain)
        wretr = np.copy(wtrain)

        for k in range(ntrains - 1):
            sample_seed_2, _ = nnfun(sample_seed[0])
            sample_seed_here = (sample_seed[0], sample_seed[1], sample_seed_2)
            xtoadd, ytoadd, wtoadd = retrain_sel(
                nnfun, xpool, weights, regs, ffun, xgenerator, limits,
                nadd_retrain, sample_seed=sample_seed_here,
                maxretries=maxretries
            )

            if type(xtoadd) is int and xtoadd == 0:
                print(
                    "All the points in pool have been correctly classified.\n",
                    "Remember to check for overfitting."
                )
                return nnfun, this_mdl

            xretr = np.append(xretr, xtoadd, axis=0)
            yretr = np.append(yretr, ytoadd, axis=0)
            wretr = np.append(wretr, wtoadd)

            # REPEAT SMALL SAMPLE
            cntr = []
            for j in range(int(np.max(yretr)) + 1):
                cntr.append((yretr == j).sum())

            cntr_max = max(cntr)
            cntr_min = min(cntr)

            if cntr_max != cntr_min:
                xretr_pdd = np.empty((0, xretr.shape[1]))
                yretr_pdd = np.empty((0, yretr.shape[1]))
                for j in range(int(np.max(yretr)) + 1):
                    xretr_dum = np.copy(xretr[(yretr == j).flatten()])
                    yretr_dum = np.copy(yretr[(yretr == j).flatten()])
                    while xretr_dum.shape[0] < cntr_max:
                        xretr_dum = np.append(
                            xretr_dum,
                            xretr[(yretr == j).flatten()],
                            axis=0
                        )
                        yretr_dum = np.append(
                            yretr_dum,
                            yretr[(yretr == j).flatten()],
                            axis=0
                        )

                    xretr_pdd = np.append(
                        xretr_pdd,
                        xretr_dum[:cntr_max],
                        axis=0
                    )
                    yretr_pdd = np.append(
                        yretr_pdd,
                        yretr_dum[:cntr_max],
                        axis=0
                    )
                xretr = xretr_pdd
                yretr = yretr_pdd
            # ===================

            if activation_out == 'softmax':
                yretr_t = to_categorical(yretr)
            else:
                yretr_t = to_multilabel(yretr, activation_out=activation_out)

            if model_restart:
                this_mdl = model_create(
                    # ndim, activation_out, 10*16*ndim, nodes_out, loss,
                    ndim, activation_out, nreg*2*ndim, nodes_out, loss,
                    learning_rate=learning_rate, use_metrics=use_metrics
                )
                nnfun = reg_pred_gen(this_mdl, data_transform)

            # weights_as_tensor = tf.convert_to_tensor(wretr, dtype=tf.float32)

            print("Training with", xretr.shape[0], "data points")
            this_mdl.fit(
                data_transform(xretr),
                yretr_t,
                epochs=epochs_part,
                batch_size=batch_size,
                verbose=verbose,
                callbacks=callbacks
            )

    return nnfun, this_mdl
