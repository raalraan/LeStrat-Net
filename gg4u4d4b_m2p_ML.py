# import ROOT
# from array import array
import numpy as np
import matplotlib.pyplot as plt

import TGPS_m2p
# from TGPS_m2p import ft_gen_ph_spc, ft_get_mg5weights

import tensorflow as tf
# import tensorflow.math as tfm
from tensorflow.keras import Sequential
from tensorflow.keras.layers import Dense
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import ReduceLROnPlateau, EarlyStopping

# from functions import get_lims
import functions
import uproot as ur

ENERGY = TGPS_m2p.ENERGY

# Actual independent indices in 4-momentum input to madgraph
indindx = np.delete(list(range(8, 13*4)), range(0, 11*4, 4))
# Add quark energy
indindx = np.append([0, 4], indindx)
indim = indindx.shape[0]

p23 = TGPS_m2p.p23

# %% Relevant definitions


# Keep only independent components of 4-momentum input
def inputtrans(x):
    # x_nb = x[:, 2:-1, 1]
    # y_nb = x[:, 2:-1, 2]
    # rho_nb = np.sqrt(x_nb**2 + y_nb**2)
    # prephi = np.arcsin(y_nb/rho_nb)
    # phi = prephi
    # phi[(x_nb < 0)*(y_nb >= 0)] = - prephi[(x_nb < 0)*(y_nb >= 0)] + np.pi
    # phi[(x_nb < 0)*(y_nb < 0)] = - prephi[(x_nb < 0)*(y_nb < 0)] - np.pi
    # phi[rho_nb == 0] = 0
    # z = x[:, :-1, 3]
    # xind = np.concatenate(
    #     (
    #         z/ENERGY*2,
    #         rho_nb/ENERGY,
    #         phi/np.pi
    #     ),
    #     axis=1
    # )
    nx = x.shape[0]
    newx = x.reshape(nx, 14*4)
    xind = newx[:, indindx]/ENERGY*2
    return xind


def get_weights(momenta, weights, factors=None):
    ndata = momenta.shape[0]
    # Use s = Q^2 for PDFS
    p1p2 = momenta[:, 0] + momenta[:, 1]
    q2s = p1p2[:, 0]**2 - (p1p2[:, 1]**2 + p1p2[:, 2]**2 + p1p2[:, 3]**2)

    alphas = [p23.alphasQ2(s) for s in q2s]
    wgws = TGPS_m2p.ft_get_mg5weights(momenta, alphas)

    x1 = 2*momenta[:, 0, 3]/ENERGY
    x2 = -2*momenta[:, 1, 3]/ENERGY

    pdf1 = np.empty(ndata)
    pdf2 = np.empty(ndata)
    for j in range(ndata):
        pdf1[j] = p23.xfxQ2(21, x1[j], q2s[j])/x1[j]/momenta[j, 0, 0]
        pdf2[j] = p23.xfxQ2(21, x2[j], q2s[j])/x2[j]/momenta[j, 1, 0]

    tws = weights*wgws*pdf1*pdf2
    tws[tws < 0.0] = 0.0
    if factors is None:
        return tws
    else:
        return tws, wgws, pdf1, pdf2


# %%
NNREG = 2
THRESHOLD = 0.2


# The best so far
def myloss4(ytr, ypr):
    global NNREG
    ydiff = tf.math.abs(ytr - ypr)
    cdiff = ydiff*(NNREG - 1)

    cdiff_shift = cdiff - THRESHOLD
    fltr = cdiff_shift > 0
    summed = tf.math.reduce_sum(cdiff[fltr])

    maxdiff = 1 + tf.math.ceil(tf.math.reduce_max(cdiff))
    return maxdiff*summed/tf.cast(tf.shape(ypr)[0], tf.float32)


def myloss5(ytr, ypr):
    global NNREG
    ydiff = tf.math.abs(ytr - ypr)
    cdiff = ydiff*(NNREG - 1)
    cdiff_re = (cdiff/0.333)**2
    return tf.math.reduce_mean(cdiff_re)


def myloss_mae(ytr, ypr):
    ydiff = tf.math.abs(ytr - ypr)
    ydiff_max = tf.math.reduce_max(ydiff)
    ydiff_mean = tf.math.reduce_mean(ydiff)
    ydiff_std = tf.math.reduce_std(ydiff)
    # return ydiff_max*ydiff_mean
    return (ydiff_mean + 3*ydiff_std)*ydiff_max


reduce_lr = ReduceLROnPlateau(
    monitor='loss',
    factor=0.9,
    min_delta=0.001,
    patience=300,
    min_lr=0.00001
)

stopper = EarlyStopping(
    monitor='loss',
    min_delta=0.01,
    patience=200,
)

# indim = x0train.shape[1]
# outdim = 1

# %%


def get_train_xy(momenta, weights, lims):
    momenta_t = inputtrans(momenta)
    sdind, cnts = functions.divindx(weights, lims)
    nreg = len(lims) - 1
    boored = int(weights.shape[0]/nreg)
    xtrain = np.empty((0, momenta_t.shape[1]))
    wtrain = np.empty((0))
    ytrain = np.empty((0, 1))
    for j in range(len(lims) - 1):
        shrange = np.random.permutation(np.arange(int(cnts[j])))
        test = momenta_t[sdind.flatten() == j][shrange[:boored]]
        testw = weights[sdind.flatten() == j][shrange[:boored]]
        if test.shape[0] < boored:
            test = np.tile(test, (int(boored/test.shape[0]) + 1, 1))
            testw = np.tile(testw, int(boored/testw.shape[0]) + 1)
        xtrain = np.append(xtrain, test[:boored], axis=0)
        ytrain = np.append(ytrain, np.array([[j]]*boored), axis=0)
        wtrain = np.append(wtrain, testw[:boored])
    return xtrain, ytrain, wtrain


def find_lim(data, weights, margn=15, datamin=None):
    dmean = np.average(
        data,
        weights=weights
    )
    dstd = np.sqrt(np.average(
        (data - dmean)**2,
        weights=weights
    ))
    if datamin is not None:
        minlim = datamin
    else:
        minlim = dmean - margn*dstd
    return [minlim, dmean + margn*dstd]


# Get a limit for a set of weights (assumed uniform distribution) with some
# target relative importance.
# For example, `target=10.0` here means that the division will be put where
# one region has 10 times more importance than the other
def get_lims_test(weights, target=10.0):
    asrt = np.argsort(weights)
    wsrt = weights[asrt]
    lw = len(weights)
    for j in range(lw):
        # sum upper part
        upsum = wsrt[lw - j:].sum()
        # sum lower part
        lpsum = wsrt[:lw - j].sum()
        rat_post = upsum/lpsum
        if rat_post > target:
            # print(
            #     lw - j,
            #     wsrt[lw - j],
            #     wsrt[lw - j:].sum()/wsrt[:lw - j].sum()
            # )
            # print(
            #     lw - j + 1,
            #     wsrt[lw - j + 1],
            #     wsrt[lw - j + 1:].sum()/wsrt[:lw - j + 1].sum()
            # )
            rat_prev = wsrt[lw - j + 1:].sum()/wsrt[:lw - j + 1].sum()
            break
        if j == lw - 1 and upsum/lpsum < target:
            print(
                "get_lims_test: target could not get achieved",
                upsum/lpsum
            )
    # print([rat_prev, rat_post])
    # print([wsrt[lw - j + 1], wsrt[lw - j]])
    # print("get_lims_test:", lw, j, rat_prev, rat_post, target)
    tarinterp = np.interp(
        target,
        [rat_prev, rat_post],
        [wsrt[lw - j], wsrt[lw - j - 1]]
    )
    return tarinterp

# %%

# TODO It may never actually get to 1
def get_lim_var(weights, target=1, space_survey='log', ntest=100):
    if space_survey == 'log':
        tlims = np.logspace(
            np.log10(weights[weights > 0].min()),
            np.log10(weights.max()),
            ntest
        )
    elif space_survey == 'linear':
        tlims = np.linspace(
            weights.min(),
            weights.max(),
            ntest
        )
    wstds_n = np.fromiter(
        (weights[(weights < tlims[j])].std() for j in range(1, ntest)),
        dtype=float
    )
    wmeans_n = np.fromiter(
        (weights[(weights < tlims[j])].mean() for j in range(1, ntest)),
        dtype=float
    )
    smrat = wstds_n/wmeans_n
    if (smrat <= target).sum() == 0:
        print(
            "target is never reached, outputting limit at mininum: std/mean =",
            smrat.min()
        )
    attrgt = np.abs(wstds_n/wmeans_n - target).argmin() + 1
    # plt.plot(
    #     tlims[1:],
    #     wstds_n/wmeans_n
    # )
    # plt.plot(
    #     tlims[1:],
    #     np.abs(wstds_n/wmeans_n - target)
    # )
    # plt.xscale(space_survey)
    # plt.yscale(space_survey)
    return tlims[attrgt]



# %%


class puringko:
    def __init__(
        self,
        nreg,
        indim,
        hlayers=2,
        nodes_max=35,
        learning_rate=0.0001,
        activation_out='sigmoid',
        optimizer=None,
        loss=None
    ):
        self.nreg = nreg
        self.indim = indim
        self.hlayers = hlayers
        self.learning_rate = learning_rate
        self.nodes_max = nodes_max
        self.activation_out = activation_out
        self.model_fitted = [False]*(nreg - 1)
        self.optimizer = optimizer
        self.limits = [None]*(nreg - 1)
        self.Egbounds = np.zeros((nreg - 1, 2))
        if loss is None:
            if self.activation_out == 'sigmoid':
                self.loss='binary_crossentropy'
            elif self.activation_out == 'tanh':
                self.loss='hinge'
                # Could also be:
                # self.loss='squared_hinge'
            else:
                raise ValueError("use `loss=` to set the appropriate loss function")
        else:
            self.loss = loss

        # lnodes = np.linspace(1, nodes_max, hlayers + 1).astype(int)
        lnodes = (2**np.arange(hlayers)*(nodes_max/2**(hlayers - 1))).astype(int)
        lnodes = np.append(1, lnodes)
        self.lnodes = lnodes[::-1]

        self.models = [None]*(nreg - 1)
        for j in range(nreg - 1):
            self.start_model(j)

    def start_model(self, which):
        lnodes = self.lnodes
        indim = self.indim
        hlayers = self.hlayers
        loss = self.loss
        activation_out = self.activation_out

        if self.optimizer is None:
            optimizer = Adam(learning_rate=self.learning_rate)
        else:
            optimizer = self.optimizer

        thismdl = Sequential()
        thismdl.add(Dense(lnodes[0], input_shape=(indim,), activation='relu'))
        for k in range(hlayers - 1):
            thismdl.add(Dense(lnodes[k + 1], activation='relu'))
        thismdl.add(Dense(lnodes[-1], activation=activation_out))
        thismdl.compile(optimizer=optimizer, loss=loss)

        self.model_fitted[which] = False
        self.models[which] = thismdl

    def restart_model(self, which):
        print(
            "Restarting model {} (index {})".format(which + 1, which),
            "of", len(self.models)
        )
        self.start_model(which)

    def predict(self, fmomenta, batch_size=10000, verbose=1):
        nreg = self.nreg
        fmomentat = inputtrans(fmomenta)
        preds = [None]*(nreg - 1)
        rpreds = [None]*(nreg - 1)
        for j in range(nreg - 1):
            with tf.device('CPU:0'):
                preds[j] = self.models[j].predict(fmomentat, batch_size=batch_size, verbose=verbose).flatten()
                if self.activation_out == 'tanh':
                    preds[j] = (preds[j] + 1)/2
            rpreds[j] = np.round(preds[j]).astype(int)
        rpredsT = np.array(rpreds).T
        # return rpredsT, rpredsT.sum(axis=1), np.array(preds).T, np.packbits(rpredsT[:, ::-1], bitorder='little', axis=1)

        return rpredsT, np.array(preds).T

    def model_fit(
            self, which, x, y,
            epochs=1000, batch_size=10000, verbose=1, callbacks=None,
            learning_rate=None
    ):
        if learning_rate is None:
            learning_rate = self.learning_rate

        if self.activation_out == 'tanh':
            yadj = y*2 - 1
        else:
            yadj = y
        self.models[which].optimizer.learning_rate.assign(learning_rate)
        with tf.device('GPU:0'):
            hist = self.models[which].fit(
                x, yadj, epochs=epochs, batch_size=batch_size, verbose=verbose,
                callbacks=callbacks
            )
        self.model_fitted[which] = True
        return hist


testseq = [1000000, 100]

# %%

def myloss5_2(ytr, ypr):
    ydiff = tf.math.abs(ytr - ypr)
    ydiff_re = (ydiff/0.2)**2
    return tf.math.reduce_mean(ydiff_re)


def myloss4_2(ytr, ypr):
    ydiff = tf.math.abs(ytr - ypr)
    ydiff_shift = ydiff - 0.2
    fltr = ydiff_shift > 0
    summed = tf.math.reduce_sum(ydiff[fltr])
    return summed/tf.cast(tf.shape(ypr)[0], tf.float32)


# %%


def trainloop(
    purobject,
    which,
    nsteps,
    ntest=int(1e7),
    ntrain=int(1e5)
):
    # =======================================
    # Training loop
    thist = [None]*nsteps
    reg1sv = np.empty((0, 14, 4))
    w1sv = np.empty((0))
    fmmnta_train = np.empty((0, 14, 4))
    weights_train = np.empty((0))
    if which == 0:
        Eglim_here = ENERGY
    else:
        Eglim_here = purobject.Egbounds[which - 1]

    for j in range(nsteps):
        # PHASE SPACE GENERATOR
        td4mg1_n, w1_n, ncut1_n = TGPS_m2p.gg4u4d4b_gen_ph_spc_fast(
            energy=Eglim_here, npts=ntest)

        if which > 0:
            p1, p2 = purobject.predict(td4mg1_n, verbose=0)
            prefltr = p1[:, which - 1] == 1
            td4mg1_n = td4mg1_n[prefltr]
            w1_n = w1_n[prefltr]

        if purobject.model_fitted[which] or j > 0:
            # PREDICTION
            p1, p2 = purobject.predict(td4mg1_n, verbose=0)

            # SELECTION =============================
            # Select a limited amount of points predicted in upper half
            fm_n = td4mg1_n[(p2[:, which] >= 0.5)][:int(ntrain/2)]
            w_n = w1_n[(p2[:, which] >= 0.5)][:int(ntrain/2)]

            # Select a limited amount of points predicted in lower half but
            # not at 0
            fm_n = np.append(
                fm_n,
                td4mg1_n[(p2[:, which] != 0)*(p2[:, which] < 0.5)][:int(ntrain/2)],
                axis=0
            )
            w_n = np.append(
                w_n,
                w1_n[(p2[:, which] != 0)*(p2[:, which] < 0.5)][:int(ntrain/2)]
            )

            # Complete with points predicted at 0 if non-zero predictions are
            # not enough
            # if fm_n.shape[0] < int(ntrain/2)*2:
            fm_n = np.append(
                fm_n,
                td4mg1_n[p2[:, which] == 0][:ntrain - fm_n.shape[0]],
                axis=0
            )
            w_n = np.append(
                w_n,
                w1_n[p2[:, which] == 0][:ntrain - w_n.shape[0]]
            )
            # =======================================

            # Restart model if it was ruined by training and select random
            # data set for next training
            if (
                (p2[:, which] < 0.5).sum() == p2[:, which].shape[0]
                or (p2[:, which] > 0.5).sum() == p2[:, which].shape[0]
            ):
                purobject.restart_model(which)
                fm_n = td4mg1_n[:ntrain]
                w_n = w1_n[:ntrain]
        else:
            fm_n = td4mg1_n[:ntrain]
            w_n = w1_n[:ntrain]

        tw1_conf = get_weights(fm_n, w_n)

        purobject.Egbounds[which] = np.array([
            purobject.Egbounds[which],
            fm_n[tw1_conf > purobject.limits[which]][:, 0:2, 0].max(axis=0)
        ]).max(axis=0)

        reg1sv = np.append(
            reg1sv,
            fm_n[tw1_conf > purobject.limits[which]],
            axis=0
        )
        w1sv = np.append(
            w1sv,
            tw1_conf[tw1_conf > purobject.limits[which]]
        )

        # fm = 1.0
        # if purobject.model_fitted[which]:
        #     fm_n_re = fm_n[tw1_conf >= fm*purobject.limits[which]]
        #     tw1_conf_re = tw1_conf[tw1_conf >= fm*purobject.limits[which]]
        #     fm_n_re = np.append(
        #         fm_n_re,
        #         fm_n[tw1_conf < fm*purobject.limits[which]][:fm_n_re.shape[0]],
        #         axis=0
        #     )
        #     tw1_conf_re = np.append(
        #         tw1_conf_re,
        #         tw1_conf[
        #             tw1_conf < fm*purobject.limits[which]
        #         ][:tw1_conf_re.shape[0]]
        #     )
        # else:
        #     fm_n_re = fm_n
        #     tw1_conf_re = tw1_conf
        fm_n_re = fm_n
        tw1_conf_re = tw1_conf

        print("Adding points for next training:", fm_n_re.shape[0])
        fmmnta_train = np.append(
            fmmnta_train,
            fm_n_re,
            axis=0
        )
        weights_train = np.append(
            weights_train,
            tw1_conf_re,
        )

        xn, yn, _ = get_train_xy(
            fmmnta_train, weights_train, [0.0, purobject.limits[which], np.inf]
        )

        thist[j] = purobject.model_fit(
            which,
            xn, yn,
            # learning_rate=0.001,
            epochs=3000, batch_size=100000, verbose=0, callbacks=stopper
        )

        p1_tst, p2_tst = purobject.predict(td4mg1_tst, verbose=0)
        print(
            "loss", purobject.loss,
            "Eff:",
            (tw1_tst[p1_tst[:, which] == 1] > purobject.limits[which]).sum()/(
                p1_tst[:, which] == 1).sum(),
            "Frac. missed:",
            1 - (tw1_tst[p1_tst[:, which] == 1] > purobject.limits[which]).sum()/(
                tw1_tst > purobject.limits[which]).sum(),
            ",",
            (tw1_tst[p1_tst[:, which] == 1] > purobject.limits[which]).sum()/(
                tw1_tst > purobject.limits[which]).sum()
        )
    return reg1sv, w1sv, thist


# %%

# SET FOR TESTING
n_tst = int(1e6)
td4mg1_tst, w1_tst, ncut1_tst = TGPS_m2p.gg4u4d4b_gen_ph_spc_fast(
    energy=ENERGY, npts=n_tst)
tw1_tst = get_weights(td4mg1_tst, w1_tst)

wbins = functions.lbins(
    tw1_tst[tw1_tst > 0].min(),
    tw1_tst[tw1_tst > 0].max(),
    100
)

# SET TO ESTIMATE LIMIT FROM
n1 = int(1e5)
td4mg1, w1, ncut1 = TGPS_m2p.gg4u4d4b_gen_ph_spc_fast(
    energy=ENERGY, npts=n1)
tw1 = get_weights(td4mg1, w1)

# %%

ptest = puringko(
    3, indim, hlayers=6, nodes_max=128, activation_out='tanh',
    loss='squared_hinge'
)
ptest.limits[0] = get_lims_test(
    tw1,
    target=100_000_000
)
# mlim = get_lim_var(tw1, target=1, ntest=1000)

# %%

trainloop(ptest, 0, 4, ntest=int(2e7), ntrain=500000)

p1_tst, p2_tst = ptest.predict(td4mg1_tst, verbose=0)

plt.hist(
    tw1_tst, bins=wbins,
    histtype='step',
    label='test sample'
)
plt.hist(
    tw1_tst[tw1_tst > ptest.limits[0]], bins=wbins,
    histtype='step',
    label='sample above limit'
)
plt.hist(
    tw1_tst[p1_tst[:, 0] == 1], bins=wbins,
    histtype='step', linestyle='dashed',
    label='guessed above limit'
)
plt.hist(
    tw1_tst[tw1_tst > ptest.limits[0]][p1_tst[tw1_tst > ptest.limits[0], 0] == 1],
    bins=wbins,
    histtype='step', linestyle='dashed',
    label='guessed above limit, with cut'
)
plt.legend()
plt.xscale('log')
plt.yscale('log')
plt.show()

# %%

td4mg1_2, w1_2, ncut1_2 = TGPS_m2p.gg4u4d4b_gen_ph_spc_fast(
    energy=ptest.Egbounds[0], npts=int(1e7)
)
p1_2, p2_2 = ptest.predict(td4mg1_2, verbose=0)
print((p1_2[:, 0] == 1).sum())
f5 = 5e5/(p1_2[:, 0] == 1).sum()

td4mg1_2, w1_2, ncut1_2 = TGPS_m2p.gg4u4d4b_gen_ph_spc_fast(
    energy=ptest.Egbounds[0], npts=int(f5*1e7)
)
p1_2, p2_2 = ptest.predict(td4mg1_2, verbose=0)
print((p1_2[:, 0] == 1).sum())

tw2 = get_weights(td4mg1_2[p1_2[:, 0] == 1], w1_2[p1_2[:, 0] == 1])
ptest.limits[1] = get_lims_test(
    tw2[tw2 > ptest.limits[0]],
    target=10_000_000
)
trainloop(ptest, 1, 6, ntest=int(3*f5*1e7), ntrain=400000)

p1_tst, p2_tst = ptest.predict(td4mg1_tst, verbose=0)
plt.hist(
    tw1_tst, bins=wbins,
    histtype='step',
    label='test sample'
)
plt.hist(
    tw1_tst[tw1_tst > ptest.limits[1]], bins=wbins,
    histtype='step',
    label='sample above limit'
)
plt.hist(
    tw1_tst[p1_tst[:, 0] == 1], bins=wbins,
    histtype='step', linestyle='dashed',
    label='guessed above limit'
)
plt.hist(
    tw1_tst[tw1_tst > ptest.limits[0]][p1_tst[tw1_tst > ptest.limits[0], 0] == 1],
    bins=wbins,
    histtype='step', linestyle='dashed',
    label='guessed above limit, with cut'
)
plt.hist(
    tw1_tst[p1_tst[:, 1] == 1], bins=wbins,
    histtype='step', linestyle='dashed',
    label='guessed above limit'
)
plt.hist(
    tw1_tst[tw1_tst > ptest.limits[1]][p1_tst[tw1_tst > ptest.limits[1], 1] == 1],
    bins=wbins,
    histtype='step', linestyle='dashed',
    label='guessed above limit, with cut'
)
plt.legend()
plt.xscale('log')
plt.yscale('log')
plt.show()

# %%

# Steps on training the next model
# 1. Obtain a new limit
#   * First obtain a sample using previous model such that the density of high
#   importance points is increased

# %%


thvals = np.linspace(0, 1, 102)
thvals = thvals[1:-1]
thtst1 = [None]*len(thvals)
thtst2 = [None]*len(thvals)
for j in range(len(thvals)):
    thtst1[j] = (tw1_tst[p2_tst[:, 0] > thvals[j]] > mlim).sum()/(p2_tst[:, 0] > thvals[j]).sum()
    thtst2[j] = (tw1_tst[p2_tst[:, 0] > thvals[j]] > mlim).sum()/(tw1_tst > mlim).sum()

plt.plot(
    thvals,
    thtst1
)
plt.plot(
    thvals,
    thtst2
)
plt.plot(
    thvals,
    np.multiply(thtst1, thtst2)
)
# plt.xscale('log')
plt.show()

# TODO Something similar but with a moving limit

lvals = np.logspace(
    np.log10(mlim),
    np.log10(tw1_tst.max()),
    100
)
ltst1 = [None]*len(lvals)
ltst2 = [None]*len(lvals)
for j in range(len(lvals)):
    ltst1[j] = (
        tw1_tst[p2_tst[:, 0] > 0.5] > lvals[j]
    ).sum()/(p2_tst[:, 0] > 0.5).sum()
    ltst2[j] = (
        tw1_tst[p2_tst[:, 0] > 0.5] > lvals[j]
    ).sum()/(tw1_tst > lvals[j]).sum()

plt.plot(
    lvals,
    ltst1
)
plt.plot(
    lvals,
    ltst2
)
plt.xscale('log')
plt.show()

# %%

plt.hist(
    np.log10(tw1_tst[tw1_tst > 0][p2_tst[tw1_tst > 0, 0] > 0.5]),
    bins=100,
    histtype='step',
    density=True
)
# plt.hist(
#     np.log10(tw1_tst[tw1_tst > 0][p2_tst[tw1_tst > 0, 0] > 0.75]),
#     bins=100,
#     histtype='step'
# )
# plt.hist(
#     np.log10(tw1_tst[tw1_tst > 0][p2_tst[tw1_tst > 0, 0] > 0.9]),
#     bins=100,
#     histtype='step'
# )
plt.hist(
    np.log10(tw1_tst[tw1_tst > 0][p2_tst[tw1_tst > 0, 0] == 1]),
    bins=100,
    histtype='step',
    density=True
)
# plt.vlines([np.log10(mlim)], ymin=0, ymax=300, color='k')
plt.show()

plt.hist(
    p2_tst[:, 0],
    bins=10
)
plt.yscale('log')
plt.show()

# %%

td4mg1_0, w1_0, ncut1_0 = TGPS_m2p.gg4u4d4b_gen_ph_spc_fast(
    energy=ENERGY, npts=int(1e7))
p1_0, p2_0 = ptest.predict(td4mg1_0, verbose=0)

tw1_0 = get_weights(
    td4mg1_0[p1_0[:, 0] == 1][:100000],
    w1_0[p1_0[:, 0] == 1][:100000]
)

# mlim1 = get_lims_test(
#     tw1_0[tw1_0 > mlim],
#     target=100_000
# )
mlim1 = get_lim_var(tw1_0[tw1_0 > mlim], target=2.5, ntest=1000)

xini1, yini1, wini1 = get_train_xy(
    td4mg1_0[p1_0[:, 0] == 1][:100000], tw1_0, [0.0, mlim1, np.inf])

ptest.model_fit(
    1,
    xini1, yini1,
    epochs=1000, batch_size=100000, verbose=0, callbacks=stopper
)
p1_tst, p2_tst = ptest.predict(td4mg1_tst, verbose=0)
print(
    "Eff:",
    (tw1_tst[p1_tst[:, 1] == 1] > mlim1).sum()/(p1_tst[:, 1] == 1).sum(),
    "Frac. missed:",
    ((tw1_tst > mlim1).sum() - (tw1_tst[p1_tst[:, 1] == 1] > mlim1).sum())/(tw1_tst > mlim1).sum()
)

# %%

fmmnta_train = td4mg1_0[p1_0[:, 1] == 1][:100000]
weights_train = tw1_0

reg2sv = np.empty((0, 14, 4))
w2sv = np.empty((0))
for j in range(5):
    fm_n = np.empty((0, 14, 4))
    w_n = np.empty((0))
    added_low = 0
    tries = 0
    while fm_n.shape[0] < 100000 and tries < 20:
        td4mg1_n, w1_n, ncut1_n = TGPS_m2p.gg4u4d4b_gen_ph_spc_fast(
            energy=ENERGY, npts=int(2e7))
        p1, p2 = ptest.predict(td4mg1_n, verbose=0)

        # =============================
        fm_n = np.append(
            fm_n,
            td4mg1_n[(p2[:, 1] > 0.5)][:50000],  # Not bad
            axis=0
        )
        w_n = np.append(
            w_n,
            w1_n[(p2[:, 1] > 0.5)][:50000]
        )

        # if added_low < 50000:
        fm_n = np.append(
            fm_n,
            # td4mg1_n[p2[:, 0] == 0.0][:50000],
            td4mg1_n[(p2[:, 1] != 0)*(p2[:, 1] < 0.5)][:50000],  # Not bad
            axis=0
        )
        w_n = np.append(
            w_n,
            # w1_n[p2[:, 0] == 0.0][:50000]
            w1_n[(p2[:, 1] != 0)*(p2[:, 1] < 0.5)][:50000]
        )
        added_low += min((p2[:, 1] == 0.0).sum(), 50000)
        # =============================
        if fm_n.shape[0] < 100000:
            fm_n = np.append(
                fm_n,
                td4mg1_n[p2[:, 1] == 0][:120000 - fm_n.shape[0]],  # Not bad
                axis=0
            )
            w_n = np.append(
                w_n,
                # w1_n[p2[:, 0] == 0.0][:50000]
                w1_n[p2[:, 1] == 0][:120000 - w_n.shape[0]]
            )
        # =============================

        if (p2[:, 1] < 0.5).sum() == p2[:, 1].shape[0] or (p2[:, 1] > 0.5).sum() == p2[:, 1].shape[0]:
            ptest = puringko(3, hlayers=6, nodes_max=128, activation_out='tanh', loss=loss)
            fm_n = td4mg1_n[:int(1e5)]
            w_n = w1_n[:int(1e5)]
        tries += 1

    print("Adding points for next training:", fm_n.shape[0])

    tw1_conf = get_weights(
        fm_n,
        w_n
    )

    reg2sv = np.append(
        reg2sv,
        fm_n[tw1_conf > mlim1],
        axis=0
    )
    w2sv = np.append(
        w2sv,
        tw1_conf[tw1_conf > mlim1]
    )

    # plt.hist(np.log10(tw1[tw1 > 0]), bins=100, density=True)
    # plt.hist(np.log10(tw1_conf[tw1_conf > 0]), bins=100, density=True)
    # plt.show()

    fmmnta_train = np.append(
        fmmnta_train,
        fm_n,
        axis=0
    )
    weights_train = np.append(
        weights_train,
        tw1_conf,
    )

    xn, yn, _ = get_train_xy(fmmnta_train, weights_train, [0.0, mlim1, np.inf])

    ptest.model_fit(
        0,
        xn, yn,
        learning_rate=0.01,
        epochs=3000, batch_size=100000, verbose=0, callbacks=stopper
    )

    p1_tst, p2_tst = ptest.predict(td4mg1_tst, verbose=0)
    print(
        "loss", loss,
        "Eff:",
        (tw1_tst[p1_tst[:, 1] == 1] > mlim1).sum()/(p1_tst[:, 1] == 1).sum(),
        "Frac. missed:",
        1 - (tw1_tst[p1_tst[:, 1] == 1] > mlim1).sum()/(tw1_tst > mlim1).sum(), ",",
        (tw1_tst[p1_tst[:, 1] == 1] > mlim1).sum()**2/(p1_tst[:, 1] == 1).sum()/(tw1_tst > mlim1).sum()
    )
    # plt.hist(
    #     np.log10(tw1_tst[tw1_tst > 0]),
    #     density=True, histtype='step'
    # )
    # plt.hist(
    #     np.log10(tw1_tst[tw1_tst > mlim]),
    #     density=True, histtype='step'
    # )
    # plt.hist(
    #     np.log10(tw1_tst[tw1_tst > 0][p1_tst[tw1_tst > 0, 0] == 1]),
    #     density=True, histtype='step'
    # )
    # plt.hist(
    #     np.log10(tw1_tst[tw1_tst > mlim][p1_tst[tw1_tst > mlim, 0] == 1]),
    #     density=True, histtype='step'
    # )
    # plt.yscale('log')
    # plt.show()

    # plt.scatter(
    #     tw1_tst[tw1_tst > mlim][p1_tst[tw1_tst > mlim, 0] == 0],
    #     p2_tst[tw1_tst > mlim][p1_tst[tw1_tst > mlim, 0] == 0, 0]
    # )
    # plt.xscale('log')
    # # plt.yscale('log')
    # plt.show()

plt.hist(
    np.log10(tw1_tst[tw1_tst > mlim]),
    density=True, histtype='step', linestyle='dashed',
    label='guessed above limit'
)
plt.hist(
    np.log10(tw1_tst[tw1_tst > 0][p1_tst[tw1_tst > 0, 1] == 1]),
    density=True, histtype='step', linestyle='dashed',
    label='guessed above limit'
)
plt.hist(
    np.log10(tw1_tst[tw1_tst > mlim][p1_tst[tw1_tst > mlim, 1] == 1]),
    density=True, histtype='step', linestyle='dashed',
    label='guessed above limit, with cut'
)
plt.legend()
plt.yscale('log')
plt.show()

# %%

td4mg1_0, w1_0, ncut1_0 = TGPS_m2p.gg4u4d4b_gen_ph_spc_fast(
    energy=ENERGY, npts=int(1e7))
p1_0, p2_0 = ptest.predict(td4mg1_0, verbose=0)

# %%


# Second proposal for iterative training
def itertrain2(
    model, trainsteps, subdivsteps,
    epochs=2000,
    batch_size=5000,
    npts=int(1e5),
    npts_iter=int(1e5),
    uwevents=100000,
    loss=myloss5,
    verbose=1,
    callbacks=None,
    learning_rate=0.0001,
):
    nreg = ptest.nreg
    nini = int(npts)
    fmmnta, preweights, ncut = TGPS_m2p.gg4u4d4b_gen_ph_spc_fast(
        energy=ENERGY, npts=nini
    )
    weights = get_weights(fmmnta, preweights)
    # The first limit should separate a region with enough importance for
    # unweighted events
    mlim = get_lims_test(weights, target=testseq[0])
    lims = np.array([0.0, mlim, weights.max()])

    xini, yini, wini = get_train_xy(fmmnta, weights, lims)

    # =================================================

    ptest.model_fit(
        0, xini, yini,
        epochs=epochs, batch_size=batch_size, verbose=verbose,
        callbacks=callbacks
    )

    x_n = [None]*(ptest.nreg - 1)
    y_n = [None]*(ptest.nreg - 1)
    w_n = [None]*(ptest.nreg - 1)
    x_n[0] = xini
    y_n[0] = yini
    w_n[0] = wini
    # Set that will be stored as sample
    weights_sv = np.copy(weights)
    fmmnta_sv = np.copy(fmmnta)
    # Set that will be used for training
    weights_train = [np.copy(weights)]
    fmmnta_train = [np.copy(fmmnta)]
    Eglims = [
        [ENERGY/2.0, ENERGY/2.0],
        [
            x_n[0][:, 0][y_n[0].flatten() == 1].max()*ENERGY/2.0,
            x_n[0][:, 1][y_n[0].flatten() == 1].max()*ENERGY/2.0
        ]
    ]

    # =============== LOOP WILL START HERE =================
    # tstep = 0

    # TODO test with limits on gluon energy
    tam = 1
    for tstep in range(trainsteps):
        nlims = len(Eglims)
        mfitted = np.sum(ptest.model_fitted)


        fmmnta_n, preweights_n, ncut_n = TGPS_m2p.gg4u4d4b_gen_ph_spc_fast(
            energy=np.array(Eglims[nlims - 2]), npts=int(npts_iter*tam)
        )
        guessr_n, guess_n = ptest.predict(fmmnta_n, batch_size=batch_size)

        fltr_highconf = [None]*mfitted
        fltr_low_notconf = [None]*mfitted
        fltr_all = [None]*mfitted
        fltr_allregs = np.full((fmmnta_n.shape[0],), False)
        for j in range(mfitted):
            fltr_highconf[j] = guess_n[:, j] >= 1/3
            fltr_low_notconf[j] = guess_n[:, j] < 1/3
            # Clip the number of values in lower class
            ksum = guessr_n[:, j].sum()
            fltr_lnc_clip = fltr_low_notconf[j][:ksum]
            while (fltr_lnc_clip).sum() < guessr_n[:, j].sum():
                fltr_lnc_clip = fltr_low_notconf[j][:ksum]
                ksum += 1
            fltr_lnc_clip = np.append(
                fltr_lnc_clip,
                np.full(
                    (fltr_low_notconf[j].shape[0] - fltr_lnc_clip.shape[0],),
                    False
                )
            )
            fltr_all[j] = np.logical_or(fltr_highconf[j], fltr_lnc_clip)

            fltr_allregs = np.logical_or(fltr_allregs, fltr_all[j])

        # SELECTION
        # GET TRUE VALUES for all confusing
        # GET TRUE VALUES for upper trained network ~1
        # GET TRUE VALUES for a number of points with ~0 in all networks

        fmmnta_nf = fmmnta_n[fltr_allregs]
        preweights_nf = preweights_n[fltr_allregs]
        dumweights = np.full((fmmnta_n.shape[0],), 0.0, dtype=float)
        weights_allregs = get_weights(fmmnta_nf, preweights_nf)
        dumweights[fltr_allregs] = weights_allregs

        # fmmnta_n[fltr_all[j]]
        # dumweights[fltr_all[j]]

        hweights = dumweights[guessr_n[:, mfitted - 1].astype(bool)]
        print(
            "TEST hweights:",
            hweights.min(),
            hweights.max(),
            dumweights[guessr_n[:, 0].astype(bool)].max(),
            dumweights[guessr_n[:, 1].astype(bool)].max(),
            weights_allregs.max(),
            dumweights.max(),
            (weights_allregs > dumweights[guessr_n[:, 0].astype(bool)].max()).sum(),
            (weights_allregs > dumweights[guessr_n[:, 1].astype(bool)].max()).sum(),
        )

        weights_sv = np.append(weights_sv, hweights, axis=0)

        for j in range(mfitted):
            dumlims = np.array([0.0, lims[j + 1], weights_sv.max()])

            fmmnta_train[j] = np.append(
                fmmnta_train[j],
                fmmnta_n[fltr_all[j]],
                axis=0
            )
            weights_train[j] = np.append(
                weights_train[j],
                dumweights[fltr_all[j]],
                axis=0
            )
            x_n[j], y_n[j], w_n[j] = get_train_xy(
                fmmnta_train[j], weights_train[j], dumlims
            )
            ptest.model_fit(
                j, x_n[j], y_n[j],
                epochs=epochs, batch_size=batch_size, verbose=verbose,
                callbacks=callbacks
            )


        if all(ptest.model_fitted) is False:
            fmmnta_train += [np.empty((0, fmmnta.shape[1], fmmnta.shape[2]))]
            weights_train += [np.empty((0))]
            fmmnta_train[mfitted] = np.append(
                fmmnta_train[mfitted],
                fmmnta_n[fltr_all[mfitted - 1]],
                axis=0
            )
            weights_train[mfitted] = np.append(
                weights_train[mfitted],
                dumweights[fltr_all[mfitted - 1]],
                axis=0
            )

            mlim = get_lims_test(
                weights_sv[weights_sv > lims[-2]],
                target=testseq[tstep + 1]
            )
            lims = np.array(list(lims[:-1]) + [mlim] + [weights_sv.max()])

            dumlims = np.array([0.0, lims[mfitted + 1], weights_sv.max()])
            x_next, y_next, w_next = get_train_xy(
                    fmmnta_train[mfitted], weights_train[mfitted], dumlims
                )
            ptest.model_fit(
                mfitted, x_next, y_next,
                epochs=epochs, batch_size=batch_size, verbose=verbose,
                callbacks=callbacks
            )

    return nreg, lims, w_n, x_n, y_n, fmmnta_train, weights_train, np.array(Eglims)


# %% TESTING

# loss = 'mae'
# loss = 'mse'
loss = myloss4
learning_rate = 0.0001
# nreg = 7

# TODO Define a model here or recreate the model when more limits are created?
# mdl = Sequential()
# mdl.add(Dense(nreg*64, input_shape=(indim,), activation='relu'))
# mdl.add(Dense(nreg*32, activation='relu'))
# mdl.add(Dense(nreg*16, activation='relu'))
# mdl.add(Dense(nreg*8, activation='relu'))
# mdl.add(Dense(nreg*4, activation='relu'))
# mdl.add(Dense(nreg*2, activation='relu'))
# relu or sigmoid?
# if relu, put values larger than higher class into higher class
# mdl.add(Dense(1, activation='relu'))
# if sigmoid...
# mdl.add(Dense(1, activation='sigmoid'))
# adam = Adam(learning_rate=learning_rate)
# mdl.compile(optimizer=adam, loss=loss)

epochs = 1000
batch_size = 100000
verbose = 1
# xtst, ytst, fmtst, wtst, thels = itertrain(
#     mdl, 3, nreg,
#     npts=1e5,
#     epochs=epochs,
#     batch_size=batch_size,
#     verbose=verbose,
# )
# 10: Train 10 times
# 4: of those 10, 4 make a new division, since we start with 2 divisions this
#   means 6 divisions or 5 regions
# npts: number of points used in first run and training
# npts_iter: number of points tested in iteration steps.  They will be
#   filtered by the NN
# uwevents: Number of events we will attempt to create.  Right now is used to
#   decide where to create the first division
# epochs, batch_size, verbose, callbacks follow their meaning in fit function
#   of keras
nreg, lims, wtst, xtst, ytst, fmtst, wghttst, Eglim = itertrain2(
    None, 4, 1,
    npts=1e5,
    npts_iter=1e6,
    epochs=1000,
    loss=loss,
    uwevents=100000,
    batch_size=batch_size,
    verbose=0,
    callbacks=stopper
)

# Check results for training data
# diff = np.abs(np.round(mdl(xtst).numpy()*(nreg - 1)) - ytst)
# plt.hist(diff, bins=int(diff.max() - diff.min()))
# plt.show()
# print((diff == 0).sum()/diff.shape[0])

# mdl.save_weights('mdl_gg4u4d4b.h5')

# nreg = 4
# Eglim = np.array([
#     [6500.        , 6500.        ],
#     [6473.28932618, 6377.66043912],
#     [3706.86719091, 3111.72263959],
#     [3623.48939825, 3055.90059537]
# ])

# mdl = Sequential()
# mdl.add(Dense(nreg*128, input_shape=(indim,), activation='relu'))
# mdl.add(Dense(nreg*64, activation='relu'))
# mdl.add(Dense(nreg*32, activation='relu'))
# mdl.add(Dense(nreg*16, activation='relu'))
# mdl.add(Dense(nreg*8, activation='relu'))
# mdl.add(Dense(nreg*4, activation='relu'))
# # relu or sigmoid?
# # if relu, put values larger than higher class into higher class
# # mdl.add(Dense(1, activation='relu'))
# # if sigmoid...
# mdl.add(Dense(1, activation='sigmoid'))

# adam = Adam(learning_rate=learning_rate)
# mdl.compile(optimizer=adam, loss=loss)

# mdl.load_weights('mdl_gg4u4d4b.h5')
print("FINISHED")


# %%

n1 = int(1e5)
td4mg1, w1, ncut1 = TGPS_m2p.gg4u4d4b_gen_ph_spc_fast(energy=Eglim[1], npts=n1)
tw1 = get_weights(td4mg1, w1)

sdind1, cnts1 = functions.divindx(tw1, lims)
# td4mgt1 = inputtrans(td4mg1)

guess1, guess2 = ptest.predict(td4mg1, batch_size=100000)
# guess1 = mdl(td4mgt1).numpy()
# guessr1 = np.round(guess1*(nreg - 1)).astype(int)
guessr1 = guess1.sum(axis=1).reshape((guess1.shape[0], 1))
diff = np.abs(guessr1 - sdind1)

# plt.hist(diff)
# plt.show()

print((diff == 0).sum()/diff.shape[0])
# print("Put limit at", get_lims_test(tw1, target=100000))

# %%

nreg = 3
# plt.hist(ytst, bins=nreg, histtype='step')
# plt.show()

# plt.hist(np.log10(wtst[wtst > 0.0]), bins=100, histtype='step')
# plt.hist(np.log10(wghttst[wghttst > 0.0]), bins=100, histtype='step')
# plt.hist(np.log10(tw1[tw1 > 0.0]), bins=100, histtype='step')
# plt.show()
print(
    "region",
    "\nactual number of points",
    "\nguessed number of points",
    "\npercentage guesed in wrong region (by same real region)",
    "\npercentage guesed in wrong region (by same guessed region)"
)
for j in range(nreg):
    print(
        j, (sdind1 == j).sum(), (guess1 == j).sum(),
        (guessr1[sdind1 == j] != j).sum()/(sdind1 == j).sum()*100,
        (sdind1[guessr1 == j] != j).sum()/(guessr1 == j).sum()*100
    )

plt.title("Dispersion of guesses per region")
for j in range(nreg):
    disphist = guessr1[sdind1 == j]
    plt.hist(
        disphist,
        bins=disphist.max() - disphist.min() + 1,
        histtype='step',
        linewidth=2,
        linestyle='dashed',
        label=j
    )
plt.yscale('log')
plt.legend()
plt.show()

# %% VOLUMES AND AVERAGES

vollrg = 1.0
vols = np.empty(nreg)
avergs = np.empty(nreg)
avergsnow = np.empty(nreg)
avergsprev = np.empty(nreg)
njregnow = np.empty(nreg)
njregnew = np.empty(nreg)
njregprev = np.empty(nreg)
nvols = int(1e6)

for j in range(nreg):
    td4mgvol = np.empty((0, 14, 4))
    wvol = np.empty((0, 14, 4))
    guessvol = np.empty((0, 1))
    nvolssum = 0
    while (guessvol == j).sum() < 100:
        td4mgvol_pre, wvol_pre, ncutvol_pre = TGPS_m2p.gg4u4d4b_gen_ph_spc_fast(
            energy=Eglim[j], npts=nvols)
        td4mgtvol = inputtrans(td4mgvol_pre)
        predvol = mdl.predict(td4mgtvol, batch_size=100000)
        # predvol = mdl(td4mgtvol).numpy()
        guessvol_pre = np.round(predvol*(nreg - 1)).astype(int)
        td4mgvol = np.append(
            td4mgvol,
            td4mgvol_pre[guessvol_pre.flatten() >= j],
            axis=0
        )
        wvol = np.append(
            wvol,
            wvol_pre[guessvol_pre.flatten() >= j]
        )
        guessvol = np.append(
            guessvol,
            guessvol_pre[guessvol_pre.flatten() >= j],
            axis=0
        )
        nvolssum += nvols
        print(0, (guessvol == 0).sum())
        print(1, (guessvol == 1).sum())
        print(2, (guessvol == 2).sum())
        print(3, (guessvol == 3).sum())
        print((guessvol == j).sum(), guessvol.shape[0], nvolssum)

    if j + 1 < nreg:
        Eglim[j + 1] = np.array([
                Eglim[j + 1],
                td4mgvol[guessvol.flatten() == j + 1][:, :2, 0].max(axis=0)
            ]).max(axis=0)

    vols[j] = vollrg*(guessvol == j).sum()/(guessvol >= j).sum()
    vollrg -= vols[j]

    # AVERAGES
    for k in range(j, nreg):
        if (guessvol.flatten() == k).sum() > 0:
            avergsnow[k] = get_weights(
                    td4mgvol[guessvol.flatten() == k],
                    wvol[guessvol.flatten() == k],
                ).mean()
            njregnow[k] = (guessvol.flatten() == k).sum()

            if j > 0:
                njregnew[k] = (njregnow[k] + njregprev[k])
                avergs[k] = (njregnow[k]*avergsnow[k] + njregprev[k]*avergsprev[k])/njregnew[k]
            else:
                njregnew[k] = njregnow[k]
                avergs[k] = avergsnow[k]

    avergsprev = np.copy(avergs)
    njregprev = np.copy(njregnew)
    if j > 1:
        nvols = nvols*10

imprtncs1 = (avergs*vols)/(avergs*vols).sum()
print("Region importances", imprtncs1)

# %%

n2 = int(1e7)
td4mg2, w2, ncut2 = TGPS_m2p.gg4u4d4b_gen_ph_spc_fast(energy=Eglim[1], npts=n2)

td4mgt2 = inputtrans(td4mg2)
# guess2 = np.round(mdl(td4mgt2).numpy()*(nreg - 1)).astype(int)
guess2 = np.round(mdl.predict(td4mgt2, batch_size=10000)*(nreg - 1)).astype(int)
weights2_f = get_weights(td4mg2[guess2.flatten() > 0], w2[guess2.flatten() > 0])

# %%

# If one wants to compare agains importances from single sampling
# wsums = np.array(
#     [weights2_f[guess2[guess2.flatten() > 0].flatten() == k].sum() for k in range(1, nreg)]
# )

# imprtncs1_th = wsums/wsums.sum()
# print("Importances from single sampling:", imprtncs1_th)

# %% REPEATED SAMPLING WITH CHANGING ENERGY LIMITS

nstart = int(4e5)
fmmntsv = np.empty((0, 4, 4))
weightssv = np.empty((0))
guesssv = np.empty((0, 1))
# Eglim = [[ENERGY/2.0, ENERGY/2.0]]
# =======================
for j in range(nreg - 1):
    td4mg_0, w_0, ncut_0 = TGPS_m2p.gg4u4d4b_gen_ph_spc_fast(energy=Eglim[j], npts=nstart)
    td4mgt_0 = inputtrans(td4mg_0)
    guess_0 = np.round(mdl.predict(td4mgt_0, batch_size=10000)*(nreg - 1)).astype(int)
    weights_0 = get_weights(td4mg_0, w_0)

    fmmntsv = np.append(fmmntsv, td4mg_0, axis=0)
    weightssv = np.append(weightssv, weights_0, axis=0)
    guesssv = np.append(guesssv, guess_0, axis=0)

    if j + 1 < nreg:
        Eglim[j + 1] = np.array([
                Eglim[j + 1],
                fmmntsv[guesssv.flatten() == j + 1][:, :2, 0].max(axis=0)
            ]).max(axis=0)


# %%

# ratios = np.array([wghtave[j + 1]/wghtave[j] for j in range(nreg - 2)])
nevents = 100000
prereqevs = imprtncs1*nevents
reqevs_r = np.round(prereqevs).astype(int)
reqevs_c = np.ceil(prereqevs).astype(int)
reqevs = np.copy(reqevs_r)

diffevs = nevents - reqevs_r.sum()
if diffevs > 0:
    # Add missing number of events to places where it affects the least,
    # regions with larger number of points
    reqevs[reqevs_r.argmax()] = reqevs_r.max() + 1

# %% Acceptance rejection: Get the efficiency

filt_ar = []
eff_ar = []
punif = []
for k in range(nreg):
    pnorm = weightssv[guesssv.flatten() == k]/lims[k + 1]
    fpnorm = pnorm <= 1.0
    punif += [np.random.rand(pnorm.shape[0])]
    filt_ar += [(pnorm > punif[k])*fpnorm]
    eff_ar += [filt_ar[k].sum()/filt_ar[k].shape[0]]

# %% OVERSAMPLE REGIONS THAT NEED MORE POINTS

# number of region to oversample
for k in range(nreg):
    for j in range(200):
        # TODO Use efficiency when acceptance rejection uses uniform distribution
        if (guesssv == k).sum() > np.round(reqevs[k]/eff_ar[k]):
            print(
                "Region {}:".format(k),
                "Requested number of points has been reached"
            )
            break
        # Number of region that needs oversampling minus 1 (k - 1 = 5)
        # TODO IDEA pick either k or k - 1 at random, 0.5-0.5 (coin)
        if np.random.rand() > 0.5:
            uselims = k - 1
        else:
            uselims = k
        td4mg_0, w_0, ncut_0 = TGPS_m2p.gg4u4d4b_gen_ph_spc_fast(energy=Eglim[uselims], npts=nstart)
        td4mgt_0 = inputtrans(td4mg_0)
        guess_0 = np.round(mdl.predict(td4mgt_0, batch_size=10000)*(nreg - 1)).astype(int)

        # Number of region that needs oversampling (k)
        fltr = (guess_0.flatten() == k)

        weights_0 = get_weights(td4mg_0[fltr], w_0[fltr])

        fmmntsv = np.append(fmmntsv, td4mg_0[fltr], axis=0)
        weightssv = np.append(weightssv, weights_0, axis=0)
        guesssv = np.append(guesssv, guess_0[fltr], axis=0)

        # Update quarks energy limits
        Eglim[k] = fmmntsv[guesssv.flatten() == k][:, :2, 0].max(axis=0)
        # Eglim[k, 1] = fmmntsv[guesssv.flatten() == k][:, 1, 0].max()

        lims[-1] = max(lims[-1], max(weightssv))

        print(
            j, (guesssv == k).sum(),
            "(+{}, {}),".format(fltr.sum(), uselims),
            "New limits:", Eglim[k]
        )

meesv = mee_invariant(fmmntsv)

# %% Acceptance rejection: Get the final results

filt_ar = []
eff_ar = []
punif = []
for k in range(nreg):
    pnorm = weightssv[guesssv.flatten() == k]/lims[k + 1]
    fpnorm = pnorm <= 1.0
    punif += [np.random.rand(pnorm.shape[0])]
    filt_ar += [(pnorm > punif[k])*fpnorm]
    eff_ar += [filt_ar[k].sum()/filt_ar[k].shape[0]]

# %%

fmmnt_accrej = np.empty((0, 4, 4))
for k in range(nreg):
    fmmnt_accrej = np.append(
        fmmnt_accrej,
        fmmntsv[guesssv.flatten() == k][filt_ar[k]][:int(reqevs[k])],
        axis=0
    )

np.savetxt("gg4u4d4b_mlevents.csv", fmmnt_accrej.reshape((nevents, 14*4)))
# read with np.loadtxt("mlevents.csv").reshape((nevents, 14, 4))
