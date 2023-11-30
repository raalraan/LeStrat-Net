# import ROOT
# from array import array
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as colors

import TGPS_m2p
# from TGPS_m2p import ft_gen_ph_spc, ft_get_mg5weights
import lhapdf

import tensorflow as tf
# import tensorflow.math as tfm
from tensorflow.keras import Sequential
from tensorflow.keras.layers import Dense, Dropout
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import ReduceLROnPlateau, EarlyStopping

# from functions import get_lims
import functions
import uproot as ur

ENERGY = TGPS_m2p.ENERGY

# Actual independent indices in 4-momentum input to madgraph
# indindx = np.delete(list(range(8, 4*4)), range(0, 2*4, 4))
indindx = np.array([0, 4, 9, 10, 11])
# p23 = lhapdf.mkPDF("NNPDF23_lo_as_0130_qed", 0)
p23 = TGPS_m2p.p23

# %% Relevant definitions


# Keep only independent components of 4-momentum input
def inputtrans(x):
    nx = x.shape[0]
    newx = x.reshape(nx, 4*4)
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
        pdf1[j] = p23.xfxQ2(2, x1[j], q2s[j])/x1[j]/momenta[j, 0, 0]
        pdf2[j] = p23.xfxQ2(-2, x2[j], q2s[j])/x2[j]/momenta[j, 1, 0]

    tws = weights*wgws*pdf1*pdf2
    tws[tws < 0.0] = 0.0
    if factors is None:
        return tws
    else:
        return tws, wgws, pdf1, pdf2


def lbins(limini, limend, nbins=100):
    return np.logspace(
        np.log10(limini),
        np.log10(limend),
        nbins
    )


def mee_invariant(momenta):
    fm_sum = momenta[:, 2] + momenta[:, 3]
    mee_inv = np.sqrt(
        fm_sum[:, 0]**2 - (fm_sum[:, 1]**2 + fm_sum[:, 2]**2 + fm_sum[:, 3]**2)
    )
    return mee_inv


# %%

n0 = int(1e6)
# thedata4mg, weights, ncut = TGPS_m2p.qqee_gen_ph_spc(energy=ENERGY, npts=n0)
thedata4mg, weights, ncut = TGPS_m2p.qqee_gen_ph_spc_fast(energy=ENERGY, npts=n0)
totweights = get_weights(thedata4mg, weights)

td4mgtrans = inputtrans(thedata4mg)

pre = thedata4mg[:, 2] + thedata4mg[:, 3]
mee = np.sqrt(pre[:, 0]**2 - (pre[:, 1]**2 + pre[:, 2]**2 + pre[:, 3]**2))
plt.hist(
    mee[totweights > 0.0],
    weights=totweights[totweights > 0.0],
    bins=lbins(20, 1200, 100)
)
plt.xlim(20, 1200)
plt.xlabel('$m_{ee}$')
plt.xscale('log')
plt.yscale('log')
plt.show()

plt.hist(
    mee[totweights > 0.0],
    weights=totweights[totweights > 0.0],
    bins=lbins(20, 600, 100)
)
plt.xlabel('$m_{ee}$')
plt.xlim(20, 600)
plt.xscale('log')
plt.yscale('log')
plt.show()

# %%

nreg = 10
lims1, lims2 = functions.get_lims(totweights, nreg)
lims3 = functions.get_lims2(totweights, nreg)

hbins, ebins, p = plt.hist(
    np.log10(totweights[totweights > 0.0]),
    bins=100,
    density=True,
    histtype="step",
    linewidth=1.5
)
# plt.vlines(np.log10(lims1), 0, hbins.max()*1.05, colors='r')
# plt.vlines(np.log10(lims2), 0, hbins.max()*1.05, colors='g')
plt.vlines(np.log10(lims3), 0, hbins.max()*1.05, colors='g')
plt.ylim(0, hbins.max()*1.05)
# plt.vlines(np.log10(lims2), 0, hbins.max()*1.05, colors='g', ls='dotted')
plt.show()

lims = lims3

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

indim = indindx.shape[0]
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


# Training function
def itertrain(
    model, trainsteps, nreg,
    epochs=2000,
    batch_size=5000,
    npts=int(1e4),
    verbose=1,
    callbacks=None,
    learning_rate=None,
):
    global NNREG
    NNREG = nreg
    nini = int(npts)
    fmmnta, weights, ncut = TGPS_m2p.qqee_gen_ph_spc_fast(
        energy=ENERGY,
        npts=nini
    )
    weights_tot = get_weights(fmmnta, weights)
    lims = functions.get_lims2(weights_tot, nreg)

    xini, yini, _ = get_train_xy(fmmnta, weights_tot, lims)

    if learning_rate is not None:
        model.optimizer.learning_rate.assign(learning_rate)
    model.fit(
        xini, yini/(nreg - 1),
        epochs=epochs, batch_size=batch_size, verbose=verbose,
        callbacks=callbacks
    )

    x_n = xini
    y_n = yini
    weights_sv = weights_tot
    fmmnta_sv = fmmnta
    for tstep in range(trainsteps - 1):
        fmmnta_n, weights_n, ncut_n = TGPS_m2p.qqee_gen_ph_spc_fast(
            energy=ENERGY, npts=nini)
        # Get guessed classes
        guess_n = model(inputtrans(fmmnta_n)).numpy()*(nreg - 1)
        guessr_n = np.round(guess_n)
        # filter points above half and confusing
        # TODO Better selection of regions
        # fltr_half = guessr_n > min(tstep, nreg - 2)
        fltr_higher = guessr_n > nreg - 4
        # fltr_cut = guessr_n == 0
        # fltr_hicut = np.logical_or(fltr_higher, fltr_cut)
        fltr_conf = np.abs(guess_n - guessr_n) > THRESHOLD
        fltr_both = np.logical_or(fltr_higher, fltr_conf).flatten()
        # fltr_both = fltr_higher.flatten()
        # for j in range(nreg):
        #     guessr_n == 0

        fmmnta_f = fmmnta_n[fltr_both]
        weights_f = weights_n[fltr_both]

        weights_totf = get_weights(fmmnta_f, weights_f)

        fmmnta_ft = inputtrans(fmmnta_f)
        sdind_n, cnts_n = functions.divindx(weights_totf, lims)

        x_n = np.append(x_n, fmmnta_ft, axis=0)
        y_n = np.append(y_n, sdind_n, axis=0)
        weights_sv = np.append(
            weights_sv,
            weights_n[fltr_higher.flatten()],
            axis=0
        )
        fmmnta_sv = np.append(
            fmmnta_sv,
            fmmnta_n[fltr_higher.flatten()],
            axis=0
        )
        # TODO Better scheduling for learning rate
        # model.optimizer.learning_rate.assign(0.00005)
        model.optimizer.learning_rate.assign(0.0001/(tstep + 1)/2)
        model.fit(
            x_n, y_n/(nreg - 1),
            epochs=epochs, batch_size=batch_size, verbose=verbose,
            callbacks=callbacks
        )

    lims[-1] = weights_sv.max()

    return x_n, y_n, fmmnta_sv, weights_sv, lims


def get_lims_test(weights, target=10.0):
    asrt = np.argsort(weights)
    wsrt = weights[asrt]
    lw = len(weights)
    for j in range(lw):
        # sum upper part
        upsum = wsrt[lw - j:].sum()
        # sum lower part
        lpsum = wsrt[:lw - j].sum()
        if upsum/lpsum > target:
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
            rat_post = wsrt[lw - j:].sum()/wsrt[:lw - j].sum()
            break
    # print([rat_prev, rat_post])
    # print([wsrt[lw - j + 1], wsrt[lw - j]])
    tarinterp = np.interp(
        target,
        [rat_prev, rat_post],
        [wsrt[lw - j + 1], wsrt[lw - j]]
    )
    return tarinterp


def itertrain2(
    model, trainsteps, subdivsteps, nreg=2,
    epochs=2000,
    batch_size=5000,
    npts=int(1e5),
    npts_iter=int(1e5),
    uwevents=100000,
    verbose=1,
    callbacks=None,
    learning_rate=0.0001,
):
    global NNREG
    NNREG = 2
    nini = int(npts)
    fmmnta, preweights, ncut = TGPS_m2p.qqee_gen_ph_spc_fast(energy=ENERGY, npts=nini)
    weights = get_weights(fmmnta, preweights)
    # The first limit should separate a region with enough importance for
    # unweighted events
    mlim = get_lims_test(weights, target=uwevents)
    lims = np.array([0.0, mlim, weights.max()])

    xini, yini, wini = get_train_xy(fmmnta, weights, lims)

    # =================================================
    modelh = Sequential()
    modelh.add(Dense(NNREG*64, input_shape=(indim,), activation='relu'))
    modelh.add(Dense(NNREG*32, activation='relu'))
    modelh.add(Dense(NNREG*16, activation='relu'))
    modelh.add(Dense(NNREG*8, activation='relu'))
    modelh.add(Dense(NNREG*4, activation='relu'))
    modelh.add(Dense(NNREG*2, activation='relu'))
    # relu or sigmoid?
    # if relu, put values larger than higher class into higher class
    # modelh.add(Dense(1, activation='relu'))
    # if sigmoid...
    modelh.add(Dense(1, activation='sigmoid'))

    adam = Adam(learning_rate=learning_rate)
    modelh.compile(optimizer=adam, loss=loss)
    # =================================================

    if learning_rate is not None:
        modelh.optimizer.learning_rate.assign(learning_rate)
    modelh.fit(
        xini, yini/(nreg - 1),
        epochs=epochs, batch_size=batch_size, verbose=verbose,
        callbacks=callbacks
    )

    nreg_n = nreg
    x_n = xini
    y_n = yini
    w_n = wini
    # Set that will be stored as sample
    weights_sv = np.copy(weights)
    fmmnta_sv = np.copy(fmmnta)
    # Set that will be used for training
    weights_train = np.copy(weights)
    fmmnta_train = np.copy(fmmnta)
    # return nnreg, lims, w_n, x_n, y_n
    Eqlims = [
        [ENERGY/2.0, ENERGY/2.0],
        [
            x_n[:, 0][y_n.flatten() == 1].max()*ENERGY/2.0,
            x_n[:, 1][y_n.flatten() == 1].max()*ENERGY/2.0
        ]
    ]

    # =============== LOOP WILL START HERE =================
    # tstep = 0

    # TODO test with limits on quark energy
    for tstep in range(trainsteps):
        nlims = len(Eqlims)
        fmmnta_n, preweights_n, ncut_n = TGPS_m2p.qqee_gen_ph_spc_fast(
            energy=np.array(Eqlims[nlims - 2]), npts=int(npts_iter)
        )

        guess_n = modelh(inputtrans(fmmnta_n)).numpy()*(nreg_n - 1)
        guessr_n = np.round(guess_n)
        fltr_conf = np.abs(guess_n - guessr_n) > 0.2
        # TODO Can this selection of points be improved
        fltr_higher = guessr_n > max(0, nlims - 3)
        print("Added confusing points:", fltr_conf.sum())
        print("Added high importance points:", fltr_higher.sum())
        print("Points in higher importance level:", (guessr_n > nlims - 2).sum())
        fltr_both = np.logical_or(fltr_higher, fltr_conf).flatten()

        fmmnta_nf = fmmnta_n[fltr_both]
        preweights_nf = preweights_n[fltr_both]
        weights_nf = get_weights(fmmnta_nf, preweights_nf)
        fltr_nreg2 = weights_nf > lims[-2]
        fmmnta_sv = np.append(fmmnta_sv, fmmnta_nf[fltr_nreg2], axis=0)
        weights_sv = np.append(weights_sv, weights_nf[fltr_nreg2], axis=0)
        fmmnta_train = np.append(fmmnta_train, fmmnta_nf, axis=0)
        weights_train = np.append(weights_train, weights_nf, axis=0)

        if tstep < subdivsteps:
            # TODO Improve the recalculation of target, should depend on
            # target number of regions
            mlim = get_lims_test(
                weights_sv[weights_sv > lims[-2]],
                target=uwevents/(np.sqrt(10)**(tstep + 1))
            )
            lims = np.array(list(lims[:-1]) + [mlim] + [weights_sv.max()])
            NNREG = lims.shape[0] - 1
            nreg_n = NNREG
        else:
            lims[-1] = weights_sv.max()

        x_n, y_n, w_n = get_train_xy(fmmnta_train, weights_train, lims)

        # Update limits
        for j in range(1, nlims):
            Eqlims[j] = [
                x_n[:, 0][y_n.flatten() == j].max()*ENERGY/2.0,
                x_n[:, 1][y_n.flatten() == j].max()*ENERGY/2.0
            ]
        # Add one new limit
        if tstep < subdivsteps:
            Eqlims += [
                [
                    x_n[:, 0][y_n.flatten() == nlims].max()*ENERGY/2.0,
                    x_n[:, 1][y_n.flatten() == nlims].max()*ENERGY/2.0
                ]
            ]

        # =================================================
        if tstep < subdivsteps:
            modelh = Sequential()
            modelh.add(Dense(NNREG*64, input_shape=(indim,), activation='relu'))
            modelh.add(Dense(NNREG*32, activation='relu'))
            modelh.add(Dense(NNREG*16, activation='relu'))
            modelh.add(Dense(NNREG*8, activation='relu'))
            modelh.add(Dense(NNREG*4, activation='relu'))
            modelh.add(Dense(NNREG*2, activation='relu'))
            # relu or sigmoid?
            # if relu, put values larger than higher class into higher class
            # modelh.add(Dense(1, activation='relu'))
            # if sigmoid...
            modelh.add(Dense(1, activation='sigmoid'))

            adam = Adam(learning_rate=learning_rate)
            modelh.compile(optimizer=adam, loss=loss)
        # =================================================

        # modelh.optimizer.learning_rate.assign(0.0001/(tstep + 1)/2)
        modelh.fit(
            x_n, y_n/(nreg_n - 1),
            epochs=epochs, batch_size=batch_size, verbose=verbose,
            callbacks=callbacks
        )

    return nreg_n, lims, w_n, x_n, y_n, fmmnta_train, weights_train, modelh, Eqlims


# %% TESTING

# loss = 'mae'
# loss = 'mse'
loss = myloss4
learning_rate = 0.0001
# nreg = 7

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
batch_size = 5000
verbose = 0
# xtst, ytst, fmtst, wtst, thels = itertrain(
#     mdl, 3, nreg,
#     npts=1e5,
#     epochs=epochs,
#     batch_size=batch_size,
#     verbose=verbose,
# )
nreg, lims, wtst, xtst, ytst, fmtst, wghttst, mdl, Eqlim = itertrain2(
    None, 13, 9, 2,
    npts=1e5,
    npts_iter=1e6,
    epochs=1000,
    uwevents=100000,
    batch_size=batch_size,
    verbose=verbose,
    callbacks=stopper
)

diff = np.abs(np.round(mdl(xtst).numpy()*(nreg - 1)) - ytst)
# plt.hist(diff, bins=int(diff.max() - diff.min()))
# plt.show()
print((diff == 0).sum()/diff.shape[0])

# %%

n1 = int(1e6)
td4mg1, w1, ncut1 = TGPS_m2p.qqee_gen_ph_spc_fast(energy=Eqlim[1], npts=n1)
tw1 = get_weights(td4mg1, w1)

sdind1, cnts1 = functions.divindx(tw1, lims)
td4mgt1 = inputtrans(td4mg1)

guess1 = mdl(td4mgt1).numpy()
guessr1 = np.round(guess1*(nreg - 1)).astype(int)
diff = np.abs(guessr1 - sdind1)

# plt.hist(diff)
# plt.show()

print((diff == 0).sum()/diff.shape[0])
# print("Put limit at", get_lims_test(tw1, target=100000))

# %%

# plt.hist(ytst, bins=nreg, histtype='step')
# plt.show()

# plt.hist(np.log10(wtst[wtst > 0.0]), bins=100, histtype='step')
# plt.hist(np.log10(wghttst[wghttst > 0.0]), bins=100, histtype='step')
# plt.hist(np.log10(tw1[tw1 > 0.0]), bins=100, histtype='step')
# plt.show()

for j in range(nreg):
    print(
        j, (sdind1 == j).sum(), (guessr1 == j).sum(),
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
    td4mgvol, wvol, ncutvol = TGPS_m2p.qqee_gen_ph_spc_fast(
        energy=Eqlim[j], npts=nvol)
    td4mgtvol = inputtrans(td4mgvol)
    # guess2 = np.round(mdl(td4mgt2).numpy()*(nreg - 1)).astype(int)
    guessvol = np.round(mdl.predict(td4mgtvol, batch_size=10000)*(nreg - 1)).astype(int)

    if j + 1 < nreg:
        Eqlim[j + 1] = np.array([
                Eqlim[j + 1],
                td4mgvol[guessvol.flatten() == j + 1][:, :2, 0].max(axis=0)
            ]).max(axis=0)

    vols[j] = vollrg*(guessvol == j).sum()/(guessvol >= j).sum()
    vollrg -= vols[j]

    # AVERAGES
    for k in range(j, nreg):
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

imprtncs1 = (avergs*vols)/(avergs*vols).sum()

# %%

n2 = int(1e8)
td4mg2, w2, ncut2 = TGPS_m2p.qqee_gen_ph_spc_fast(energy=Eqlim[1], npts=n2)

td4mgt2 = inputtrans(td4mg2)
# guess2 = np.round(mdl(td4mgt2).numpy()*(nreg - 1)).astype(int)
guess2 = np.round(mdl.predict(td4mgt2, batch_size=10000)*(nreg - 1)).astype(int)
weights2_f = get_weights(td4mg2[guess2.flatten() > 0], w2[guess2.flatten() > 0])

mee2_f = mee_invariant(td4mg2[guess2.flatten() > 0])

# %%

wsums = np.array(
    [weights2_f[guess2[guess2.flatten() > 0].flatten() == k].sum() for k in range(1, nreg)]
)

imprtncs1 = wsums/wsums.sum()
print(imprtncs1)

# %% REPEATED SAMPLING WITH CHANGING ENERGY LIMITS

nstart = int(4e5)
fmmntsv = np.empty((0, 4, 4))
weightssv = np.empty((0))
wghtsum = []
guesssv = np.empty((0, 1))
# Eqlim = [[ENERGY/2.0, ENERGY/2.0]]
# =======================
for j in range(nreg - 1):
    td4mg_0, w_0, ncut_0 = TGPS_m2p.qqee_gen_ph_spc_fast(energy=Eqlim[j], npts=nstart)
    td4mgt_0 = inputtrans(td4mg_0)
    guess_0 = np.round(mdl.predict(td4mgt_0, batch_size=10000)*(nreg - 1)).astype(int)
    weights_0 = get_weights(td4mg_0, w_0)

    fmmntsv = np.append(fmmntsv, td4mg_0, axis=0)
    weightssv = np.append(weightssv, weights_0, axis=0)
    guesssv = np.append(guesssv, guess_0, axis=0)

    wghtsum += [weightssv[guesssv.flatten() == j + 1].sum()]

    Eqlim[j] = fmmntsv[guesssv.flatten() == j][:, :2, 0].max(axis=0)

# wghtsum = [weightssv[guesssv.flatten() == j + 1].sum() for j in range(nreg - 1)]

wghtsum = np.array(wghtsum)

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
        if (guesssv == k).sum() > np.round(reqevs[k - 1]/eff_ar[k]):
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
        td4mg_0, w_0, ncut_0 = TGPS_m2p.qqee_gen_ph_spc_fast(energy=Eqlim[uselims], npts=nstart)
        td4mgt_0 = inputtrans(td4mg_0)
        guess_0 = np.round(mdl.predict(td4mgt_0, batch_size=10000)*(nreg - 1)).astype(int)

        # Number of region that needs oversampling (k)
        fltr = (guess_0.flatten() == k)

        weights_0 = get_weights(td4mg_0[fltr], w_0[fltr])

        fmmntsv = np.append(fmmntsv, td4mg_0[fltr], axis=0)
        weightssv = np.append(weightssv, weights_0, axis=0)
        guesssv = np.append(guesssv, guess_0[fltr], axis=0)

        # Update quarks energy limits
        Eqlim[k] = fmmntsv[guesssv.flatten() == k][:, :2, 0].max(axis=0)
        # Eqlim[k, 1] = fmmntsv[guesssv.flatten() == k][:, 1, 0].max()

        lims[-1] = max(lims[-1], max(weightssv))

        print(
            j, (guesssv == k).sum(),
            "(+{}, {}),".format(fltr.sum(), uselims),
            "New limits:", Eqlim[k]
        )

meesv = mee_invariant(fmmntsv)

# %%

dummeesv = np.empty((0))
for k in range(nreg):
    dummeesv = np.append(
        dummeesv,
        meesv[guesssv.flatten() == k][filt_ar[k]][:int(reqevs[k])]
    )

fmmnt_accrej = np.empty((0, 4, 4))
for k in range(nreg):
    fmmnt_accrej = np.append(
        fmmnt_accrej,
        fmmntsv[guesssv.flatten() == k][filt_ar[k]][:int(reqevs[k])],
        axis=0
    )

# %%

meesv_divs = [
    meesv[guesssv.flatten() == j][filt_ar[j]][:int(reqevs[j])]
    for j in range(nreg)
]

plt.hist(
    meesv_divs,
    bins=lbins(20, 1200, 100),
    stacked=True,
    # density=True,
    # histtype='step'
)
plt.xlim(20, 1200)
plt.yscale('log')
plt.xscale('log')
plt.show()

# %% Vanity plots

# plt.figure(figsize=(4, 4))
# plt.scatter(
#     fmmntsv[:, 0, 0][guesssv.flatten() == 10],
#     fmmntsv[:, 1, 0][guesssv.flatten() == 10],
#     s=1
# )
# plt.xlabel(r'$E_{u}$ [GeV]')
# plt.ylabel(r'$E_{\bar{u}}$ [GeV]')
# plt.savefig('Eqq10.png', bbox_inches='tight', dpi=200)

# plt.figure(figsize=(4, 4))
# plt.scatter(
#     fmmntsv[:, 0, 0][guesssv.flatten() == 9],
#     fmmntsv[:, 1, 0][guesssv.flatten() == 9],
#     s=1
# )
# plt.xlabel(r'$E_{u}$ [GeV]')
# plt.ylabel(r'$E_{\bar{u}}$ [GeV]')
# plt.savefig('Eqq09.png', bbox_inches='tight', dpi=200)

# plt.figure(figsize=(4, 4))
# plt.scatter(
#     fmmntsv[:, 0, 0][guesssv.flatten() == 8],
#     fmmntsv[:, 1, 0][guesssv.flatten() == 8],
#     s=1
# )
# plt.xlabel(r'$E_{u}$ [GeV]')
# plt.ylabel(r'$E_{\bar{u}}$ [GeV]')
# plt.savefig('Eqq08.png', bbox_inches='tight', dpi=200)

# plt.figure(figsize=(4, 4))
# plt.scatter(
#     fmmntsv[:, 0, 0][guesssv.flatten() == 6],
#     fmmntsv[:, 1, 0][guesssv.flatten() == 6],
#     s=1
# )
# plt.xlabel(r'$E_{u}$ [GeV]')
# plt.ylabel(r'$E_{\bar{u}}$ [GeV]')
# plt.savefig('Eqq06.png', bbox_inches='tight', dpi=200)

# plt.figure(figsize=(4, 4))
# plt.scatter(
#     fmmntsv[:, 0, 0][guesssv.flatten() == 7],
#     fmmntsv[:, 1, 0][guesssv.flatten() == 7],
#     s=1
# )
# plt.xlabel(r'$E_{u}$ [GeV]')
# plt.ylabel(r'$E_{\bar{u}}$ [GeV]')
# plt.savefig('Eqq07.png', bbox_inches='tight', dpi=200)

# plt.figure(figsize=(4, 4))
# plt.scatter(
#     fmmntsv[:, 0, 0][guesssv.flatten() == 5],
#     fmmntsv[:, 1, 0][guesssv.flatten() == 5],
#     s=1
# )
# plt.xlabel(r'$E_{u}$ [GeV]')
# plt.ylabel(r'$E_{\bar{u}}$ [GeV]')
# plt.savefig('Eqq05.png', bbox_inches='tight', dpi=200)
# # %%

# plt.figure(figsize=(4, 4))
# plt.hist(
#     totweights,
#     bins=lbins(1e-30, 8.61, 100),
# )
# plt.xscale('log')
# plt.xlabel('Weight values')
# plt.savefig('weightall.png', bbox_inches='tight', dpi=200)

# plt.figure(figsize=(4, 4))
# plt.hist(
#     wghttst,
#     bins=lbins(1e-30, 8.61, 100),
# )
# plt.vlines(lims, 0, 180000, colors='g')
# plt.xscale('log')
# plt.xlabel('Weight values')
# plt.savefig('weighttrain.png', bbox_inches='tight', dpi=200)


# plt.figure(figsize=(4, 4))
# plt.scatter(
#     fmmntsv[:, 2, 1][guesssv.flatten() == 10],
#     fmmntsv[:, 2, 2][guesssv.flatten() == 10],
#     s=1
# )
# plt.xlabel(r'$p_{e,x}$ [GeV]')
# plt.ylabel(r'$p_{e,y}$ [GeV]')
# plt.show()

# %%

plt.hist(
    np.log10(weightssv[weightssv > 0.0]),
    weights=weightssv[weightssv > 0.0],
    bins=1000
)
# plt.yscale('log')
# plt.xscale('log')
plt.show()


# %% MADGRAPH

myevents = ur.open("../../../qqee_rw/Events/run_02/unweighted_events.root")

qqee_evs = myevents['LHEF;1']['Particle'].arrays(library="np")
nevs = qqee_evs['Particle.E'].shape[0]

qqee_evs_ak = myevents['LHEF;1']['Particle'].arrays(library="ak")
# nevs = qqee_evs_ak['Particle.E'].shape[0]

Uev = qqee_evs_ak[qqee_evs_ak['Particle.PID'] == 2]
Ubev = qqee_evs_ak[qqee_evs_ak['Particle.PID'] == -2]
Elev = qqee_evs_ak[qqee_evs_ak['Particle.PID'] == 11]
Poev = qqee_evs_ak[qqee_evs_ak['Particle.PID'] == -11]
# Zev = qqee_evs_ak[qqee_evs_ak['Particle.PID'] == -23]

Eu1 = Uev['Particle.E'].to_numpy().flatten()
Eu2 = Ubev['Particle.E'].to_numpy().flatten()
Ee1 = Elev['Particle.E'].to_numpy().flatten()
Ee2 = Poev['Particle.E'].to_numpy().flatten()

Pxu1 = Uev['Particle.Px'].to_numpy().flatten()
Pyu1 = Uev['Particle.Py'].to_numpy().flatten()
Pzu1 = Uev['Particle.Pz'].to_numpy().flatten()

Pxu2 = Ubev['Particle.Px'].to_numpy().flatten()
Pyu2 = Ubev['Particle.Py'].to_numpy().flatten()
Pzu2 = Ubev['Particle.Pz'].to_numpy().flatten()

Pxe1 = Elev['Particle.Px'].to_numpy().flatten()
Pye1 = Elev['Particle.Py'].to_numpy().flatten()
Pze1 = Elev['Particle.Pz'].to_numpy().flatten()
P2e1 = Pxe1**2 + Pye1**2 + Pze1**2

Pxe2 = Poev['Particle.Px'].to_numpy().flatten()
Pye2 = Poev['Particle.Py'].to_numpy().flatten()
Pze2 = Poev['Particle.Pz'].to_numpy().flatten()
P2e2 = Pxe2**2 + Pye2**2 + Pze2**2

P2ee = (Pxe1 + Pxe2)**2 + (Pye1 + Pye2)**2 + (Pze1 + Pze2)**2

meemg = np.sqrt((Ee1 + Ee2)**2 - P2ee)

# %%

hlbins = lbins(20, 600, 50)

fig, (hist, error) = plt.subplots(
    2,
    height_ratios=[3./4., 1./4.],
    figsize=(5, 5*4/3)
)
fig.suptitle(r'$u\bar{u} \to e^+ e^-$ $10^5$ events')
mlh, mlbc, _ = hist.hist(
    dummeesv,
    bins=hlbins,
    # density=True,
    histtype='step',
    label="This work"
)
mgh, mgbc, _ = hist.hist(
    meemg,
    bins=hlbins,
    # density=True,
    histtype='step',
    label="MadGraph"
)

# mlh, mlbc, _ = plt.hist(
#     mee2[totw2 > 0.0],
#     weights=totw2[totw2 > 0.0],
#     bins=hlbins,
#     density=True,
#     histtype='step'
# )
thh, thbc, _ = hist.hist(
    mee2_f,
    bins=hlbins,
    histtype='step',
    weights=weights2_f*nevents/weights2_f.sum(),
    label="Theory"
)
hist.set_ylabel('Number of events')
hist.legend()
hist.set_xlim(20, 600)
hist.set_ylim(9e-1, 1e5)
hist.set_xscale('log')
hist.set_yscale('log')
# plt.show()

# plt.figure(figsize=(5, 2))
# error.plot(
#     0.5*(hlbins[:-1] + hlbins[1:]),
#     np.abs((mgh - mlh)/mgh),
#     label="Madgraph - this work"
# )
error.plot(
    0.5*(hlbins[:-1] + hlbins[1:]),
    np.abs((thh - mlh)/thh),
    label="Theory - this work"
)
error.plot(
    0.5*(hlbins[:-1] + hlbins[1:]),
    np.abs((thh - mgh)/thh),
    label="Theory - Madgraph"
)
error.legend(loc='upper center', ncol=2)
error.set_xlabel('$m_{ee}$ [GeV]')
error.set_ylabel('error')
error.set_xlim(20, 600)
error.set_ylim(8e-4, 10)
error.set_xscale('log')
error.set_yscale('log')
# plt.xscale('log')
# plt.yscale('log')

# %%

plt.title("$u$ energy")
plt.hist(
    Eu1,
    bins=50,
    density=True,
    histtype='step'
)
plt.hist(
    td4mg2f[:, 0, 0][totw2 > 0.0],
    weights=totw2[totw2 > 0.0],
    bins=50,
    density=True,
    histtype='step'
)
plt.yscale('log')
plt.show()

plt.title(r"$\bar{u}$ energy")
plt.hist(
    Eu2,
    bins=100,
    density=True,
    histtype='step'
)
plt.hist(
    td4mg2f[:, 1, 0][totw2 > 0.0],
    weights=totw2[totw2 > 0.0],
    bins=100,
    density=True,
    histtype='step'
)
plt.yscale('log')
plt.show()

# %%

plt.hist(
    np.log10(Pzu1),
    bins=50,
    density=True,
    histtype='step'
)
plt.hist(
    np.log10(td4mg2f[:, 0, 3][totw2 > 0.0]),
    weights=totw2[totw2 > 0.0],
    bins=50,
    density=True,
    histtype='step'
)
plt.show()

plt.hist(
    np.log10(-Pzu2),
    bins=50,
    density=True,
    histtype='step'
)
plt.hist(
    np.log10(-td4mg2f[:, 1, 3][totw2 > 0.0]),
    weights=totw2[totw2 > 0.0],
    bins=50,
    density=True,
    histtype='step'
)
plt.show()


# %% Test madgraph against my weights function

data_rf = np.array(
    [
        [Eu1, Eu2, Ee1, Ee2],
        [Pxu1, Pxu2, Pxe1, Pxe2],
        [Pyu1, Pyu2, Pye1, Pye2],
        [Pzu1, Pzu2, Pze1, Pze2],
    ]
).T

ws_rf = get_weights(data_rf, np.full(data_rf.shape[0], 1))

# %%

plt.figure(figsize=(7, 7))
plt.scatter(
    Pzu1,
    -Pzu2,
    c=np.log10(ws_rf),
    cmap="turbo",
    s=1
)
# plt.xlim(0, 100)
# plt.ylim(0, 100)
# plt.yscale('log')
plt.show()

plt.hist(
    np.log10(ws_rf),
    bins=100,
    density=True,
    histtype='step'
)
plt.show()


# %% Elements of event file

myevents['LHEF;1']['Event']['Event.CouplingQCD'].arrays(library="np")
#['Rwgt.fUniqueID']
# 6 branches:
# Rwgt - Rwgt.fBits, Rwgt.Weight, Rwgt.fUniqueID: All empty
# Event
# - Event.fBits: all 50331648)
# - Event.Number: indices
# - Event.Weight: 179.642
# - Event.ScalePDF This seems to be s
# - Event.fUniqueID 0
# - Event.ProcessID 1
# - Event.Nparticles Number of particles in event, 4 or 5
# - Event.CouplingQED 0.00754677
# - Event.CouplingQCD TODO Find out how to get these values, maybe using s?
# Particle
# Rwgt_size
# Event_size
# Particle_size

# %%

muPDF = np.concatenate(
    myevents['LHEF;1']['Event']['Event.ScalePDF'].arrays(library="np")['Event.ScalePDF']
)
plt.hist(muPDF, bins=100)
plt.yscale('log')
plt.show()

alphaQCD = np.concatenate(
    myevents['LHEF;1']['Event']['Event.CouplingQCD'].arrays(library="np")['Event.CouplingQCD']
)
plt.hist(alphaQCD, bins=100)
# plt.yscale('log')
plt.show()

# %% alpha_s test

p23 = lhapdf.mkPDF("NNPDF23_lo_as_0130_qed", 0)
pdfalphas = np.array([p23.alphasQ(x) for x in muPDF])

plt.scatter(
    alphaQCD,
    alphaQCD/pdfalphas
)

# %% Pseudorapidity test

cth1 = Pze1/np.sqrt(P2e1)
th1 = np.arccos(cth1)
eta1 = -np.log(np.tan(th1/2))

plt.scatter(
    Pze1/np.sqrt(P2e1),
    eta1
)

