# TODO Unify test depending on number of dimensions
# syntax, connect to f64c0b88-1129-46b8-8357-944ee0c6a00a
import numpy as np
import matplotlib.pyplot as plt
from numpy import pi
from math import sqrt
import functions

import tensorflow as tf
# import tensorflow.math as tfm
from tensorflow.keras import Sequential
from tensorflow.keras.layers import Dense
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.utils import to_categorical

import os
os.environ["CUDA_VISIBLE_DEVICES"] = "0"

import matplotlib as mpl
mpl.use("pgf")
mpl.rcParams.update({
    "pgf.texsystem": "lualatex",
    "font.family": "serif",
    "text.usetex": True,
    "pgf.rcfonts": False,
})


def conegen(D, radii, centers):
    def thefun(x):
        fres_prt = []
        for j in range(len(radii)):
            fres_prt.append(
                radii[j] - np.sqrt(((x - centers[j])**2).sum(axis=1))
            )
            fres_prt[j][fres_prt[j] < 0] = 0
        fres = np.sum(fres_prt, axis=0)
        return fres
    return thefun


def sample_gen(npts, ntest, D, xside, lim, fun, verbose=0):
    xthis = np.empty((0, D))
    fthis = np.empty((0))
    ntried = 0
    while xthis.shape[0] < npts:
        xpre = np.random.uniform(
            low=lim, high=xside - lim, size=(ntest, D)
        )
        fpre = fun(xpre)
        xthis = np.append(
            xthis,
            xpre[fpre > lim],
            axis=0
        )
        fthis = np.append(
            fthis,
            fpre[fpre > lim],
        )
        ntried += ntest*xside**D/(xside - 2*lim)**D
        if verbose > 0:
            print(xthis.shape[0], end="\r")
    if verbose > 0:
        print()
    return xthis, fthis, ntried


def cone_vol(D, radius):
    vol = 1
    for j in range(int(D/2 + 0.5)):
        if D - j*2 == 1:
            vol *= 2*radius
        else:
            vol *= 2*pi*radius**2/(D - j*2)

    itrue = 1/(D + 1)*vol*radius
    return itrue


def plt_confusion_matrix(
    ax, thematrix,
    cmap=None, overlay_values=False, ticks_step=1
):
    nrows, ncols = thematrix.shape
    pcmesh = ax.pcolormesh(
        np.arange(0, ncols),
        np.arange(0, nrows),
        thematrix,
        cmap=cmap,
        antialiased=False,
        snap=True,
        rasterized=False
    )
    if overlay_values:
        for i in range(nrows):
            for j in range(ncols):
                ax.text(j, i, "{:.2f}".format(thiscomat[i, j]),
                        ha="center", va="center", color="w")

    ax.set_yticks(
        np.arange(0, nrows, ticks_step),
        np.arange(0, nrows, ticks_step)
    )
    ax.set_xticks(
        np.arange(0, ncols, ticks_step),
        np.arange(0, ncols, ticks_step)
    )

    ax.set_ylim(ax.get_ylim()[::-1])
    ax.tick_params(top=True, labeltop=True, bottom=False, labelbottom=False)
    ax.xaxis.set_label_position("top")
    return pcmesh


# %%

# Test integration with random points
D = 2
xside = 10
radius = xside/4
centers = [[xside/4]*D, [3*xside/4]*D]

radii = [radius]*len(centers)
ntest = int(1e8)
Vtot = xside**D

xvals = np.random.rand(ntest, D)*xside
ffun2 = conegen(D, radii, centers)
fres = ffun2(xvals)

iest = fres.mean()*Vtot
ierr = fres.std()/sqrt(ntest)*Vtot

# Two cones
itrue = 2*cone_vol(D, radius)

print(
    "Actual value of integral:",
    itrue,
    "Estimated value:",
    iest,
    "Error of estimation:",
    ierr
)

# %%

lims = [0.0]
for j in range(10):
    lims.append(
        functions.get_lim_err(
            fres[fres > lims[-1]], 0.24,
            fromlim=lims[-1],
            ntest=1000,
            nreal=ntest,
            vtotal=Vtot,
            tstscale='linear'
        )
    )
    print(j, lims[-1]/fres.max())

# %%

# Setup common to all networks
hlayers = 2
optimizer = 'adam'
learning_rate = 0.001
batch_size = int(1e6)
verbose = 1

# %% TWO DIMENSIONS

D = 2
xside = 10
radius = xside/4
centers = [[xside/4]*D, [3*xside/4]*D]

radii = [radius]*len(centers)
ntest = int(1e7)
Vtot2 = xside**D

xvals = np.random.rand(ntest, D)*xside
ffun2 = conegen(D, radii, centers)
fres = ffun2(xvals)

# %%

xgetind = functions.get_train_xy(xvals, fres, [-1] + lims + [2.5])
xohetrain = np.empty((0, D))
yohetrain = np.empty((0, 1))
fohetrain = np.empty((0))
for j in range(12):
    xohe_inreg = xgetind[0][xgetind[1].flatten() == j][:1000]
    xohetrain = np.append(
        xohetrain,
        xohe_inreg,
        axis=0
    )
    fohetrain = np.append(
        fohetrain,
        xgetind[2][xgetind[1].flatten() == j][:xohe_inreg.shape[0]]
    )
    yohetrain = np.append(
        yohetrain,
        np.full((xohe_inreg.shape[0], 1), j),
        axis=0
    )

# %% 2D: Multiple networks, predict single level

activation_out = 'sigmoid'
loss = 'binary_crossentropy'
xtrain = [None]*len(lims)
ytrain = [None]*len(lims)
for j in range(len(lims)):
    xtrain[j], ytrain[j], _ = functions.get_train_xy(
        xohetrain, fohetrain, [-1, lims[j], 2.5]
    )
    if activation_out == "tanh":
        ytrain[j] = ytrain[j]*2.0 - 1.0

mnsrsig2 = [None]*len(lims)
for j in range(len(lims)):
    mnsrsig2[j] = Sequential()
    mnsrsig2[j].add(Dense(10*16*D, input_shape=(D,), activation='relu'))
    mnsrsig2[j].add(Dense(10*8*D, activation='relu'))
    mnsrsig2[j].add(Dense(1, activation=activation_out))
    mnsrsig2[j].compile(
        optimizer=Adam(learning_rate=learning_rate),
        loss=loss
    )
    mnsrsig2[j].fit(
        xtrain[j]/xside,
        ytrain[j],
        epochs=4000,
        batch_size=batch_size,
        verbose=verbose
    )

activation_out = 'tanh'
loss = "squared_hinge"
xtrain = [None]*len(lims)
ytrain = [None]*len(lims)
for j in range(len(lims)):
    xtrain[j], ytrain[j], _ = functions.get_train_xy(
        xohetrain, fohetrain, [-1, lims[j], 2.5]
    )
    if activation_out == "tanh":
        ytrain[j] = ytrain[j]*2.0 - 1.0

mnsrtanh2 = [None]*len(lims)
for j in range(len(lims)):
    mnsrtanh2[j] = Sequential()
    mnsrtanh2[j].add(Dense(10*16*D, input_shape=(D,), activation='relu'))
    mnsrtanh2[j].add(Dense(10*8*D, activation='relu'))
    mnsrtanh2[j].add(Dense(1, activation=activation_out))
    mnsrtanh2[j].compile(
        optimizer=Adam(learning_rate=learning_rate),
        loss=loss
    )
    mnsrtanh2[j].fit(
        xtrain[j]/xside,
        ytrain[j],
        epochs=4000,
        batch_size=batch_size,
        verbose=verbose
    )

# %% 2D: Single network, softmax

activation_out = 'softmax'
loss = 'categorical_crossentropy'
snsm2 = Sequential()
snsm2.add(Dense(10*16*D, input_shape=(D,), activation='relu'))
snsm2.add(Dense(10*8*D, activation='relu'))
snsm2.add(Dense(len(lims) + 1, activation=activation_out))
snsm2.compile(
    optimizer=Adam(learning_rate=learning_rate),
    loss=loss
)
snsm2.fit(
    xohetrain/xside,
    to_categorical(yohetrain),
    epochs=6000,
    batch_size=batch_size,
    verbose=verbose
)

# %% Functions that will be used for multilabel test

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


tf_keras_bce = tf.keras.losses.BinaryCrossentropy()
tf_keras_sh = tf.keras.losses.SquaredHinge()
tf_keras_cce = tf.keras.losses.CategoricalCrossentropy()


def my_loss(
    name="binary_crossentropy",
    jumping=None
):
    if name == "binary_crossentropy":
        first_loss = tf.keras.losses.BinaryCrossentropy()

        def get_rdiff(y_true, y_pred):
            y_true_r = tf.math.round(y_true)
            y_pred_r = tf.math.round(y_pred)
            r_true = tf.math.reduce_sum(y_true_r, axis=1)
            r_pred = tf.math.reduce_sum(y_pred_r, axis=1)
            r_diff = r_true - r_pred
            return r_diff

    elif name == "squared_hinge":
        first_loss = tf.keras.losses.SquaredHinge()

        def get_rdiff(y_true, y_pred):
            y_true_r = tf.math.round((y_true + 1)/2)
            y_pred_r = tf.math.round((y_pred + 1)/2)
            r_true = tf.math.reduce_sum(y_true_r, axis=1)
            r_pred = tf.math.reduce_sum(y_pred_r, axis=1)
            r_diff = r_true - r_pred
            return r_diff

    elif name == "categorical_crossentropy":
        first_loss = tf.keras.losses.CategoricalCrossentropy()

        def get_rdiff(y_true, y_pred):
            r_true = tf.cast(tf.math.argmax(y_true, axis=1), y_pred.dtype)
            r_pred = tf.cast(tf.math.argmax(y_pred, axis=1), y_pred.dtype)
            r_diff = r_true - r_pred
            return r_diff
    else:
        raise ValueError(
            "Only `binary_crossentropy`, `squared_hinge` and "
            + "`categorical_crossentropy` have been configured "
            + "to use by `name`."
        )

    def _my_loss(y_true, y_pred):
        firstl = first_loss(y_true, y_pred)
        if jumping == "squared":
            rdiff = get_rdiff(y_true, y_pred)
            res = tf.math.reduce_mean(rdiff**2)
        elif jumping == "absolute":
            rdiff = get_rdiff(y_true, y_pred)
            res = tf.math.reduce_mean(tf.math.abs(rdiff))
        else:
            res = 1.0

        return firstl*res

    return _my_loss


def bce_jumping(y_true, y_pred):
    bce = tf_keras_bce(y_true, y_pred)
    y_true_r = tf.math.round(y_true)
    y_pred_r = tf.math.round(y_pred)
    r_true = tf.math.reduce_sum(y_true_r, axis=1)
    r_pred = tf.math.reduce_sum(y_pred_r, axis=1)
    r_diff = r_true - r_pred
    r_adiff = tf.math.abs(r_diff)
    res = tf.math.reduce_mean(r_adiff)
    return bce*res


def bce_nononsense(y_true, y_pred):
    rdum = tf.math.round(y_pred)
    rnum = rdum.shape[1]
    rdumsum = tf.math.reduce_sum(rdum, axis=1)
    rwhere = tf.reshape(tf.where(rdumsum), [-1])
    rdum_no0 = tf.gather(rdum, rwhere)
    rsum_no0 = tf.gather(rdumsum, rwhere)

    rrdum = tf.gather(rdum_no0, list(range(rnum - 1, -1, -1)), axis=1)
    rargmax = tf.cast(tf.math.argmax(rrdum, axis=1), rdumsum.dtype)
    rdiff = (rnum - rargmax - rsum_no0)
    nssum = tf.math.reduce_sum(rdiff)
    # tf.print(nssum)
    bce = tf_keras_bce(y_true, y_pred)
    npred = tf.cast(tf.shape(y_pred)[0], dtype=tf.float32)
    return bce*(1.0 + nssum/npred)


def jumping_loss(y_true, y_pred):
    y_true_r = tf.math.round(y_true)
    y_pred_r = tf.math.round(y_pred)
    r_true = tf.math.reduce_sum(y_true_r, axis=1)
    r_pred = tf.math.reduce_sum(y_pred_r, axis=1)
    r_diff = r_true - r_pred
    r_adiff = tf.math.abs(r_diff)
    res = tf.math.reduce_sum(r_adiff)

    bce = tf_keras_bce(y_true, y_pred)
    return bce + res


def bce_inaccuracy(y_true, y_pred):
    y_true_r = tf.math.round(y_true)
    y_pred_r = tf.math.round(y_pred)
    r_true = tf.math.reduce_sum(y_true_r, axis=1)
    r_pred = tf.math.reduce_sum(y_pred_r, axis=1)
    r_bad = tf.cast(r_true != r_pred, tf.float32)

    ntrain = tf.cast(tf.shape(y_pred)[0], dtype=tf.float32)
    inacc = tf.math.reduce_sum(r_bad)/ntrain
    bce = tf_keras_bce(y_true, y_pred)
    return bce*(1.0 + inacc)


def sh_jumping(y_true, y_pred):
    y_true_r = tf.math.round((y_true + 1)/2)
    y_pred_r = tf.math.round((y_pred + 1)/2)
    r_true = tf.math.reduce_sum(y_true_r, axis=1)
    r_pred = tf.math.reduce_sum(y_pred_r, axis=1)
    r_diff = r_true - r_pred
    r_adiff = tf.math.abs(r_diff)
    res = tf.math.reduce_mean(r_adiff)

    sh = tf_keras_sh(y_true, y_pred)
    return sh*res  # Best result at the moment


def sh_jumping2(y_true, y_pred):
    y_true_r = tf.math.round((y_true + 1)/2)
    y_pred_r = tf.math.round((y_pred + 1)/2)
    r_true = tf.math.reduce_sum(y_true_r, axis=1)
    r_pred = tf.math.reduce_sum(y_pred_r, axis=1)
    r_diff = r_true - r_pred
    res = tf.math.reduce_mean(r_diff**2)

    sh = tf_keras_sh(y_true, y_pred)
    return sh*res  # Best result at the moment


def sh_inaccuracy(y_true, y_pred):
    y_true_r = tf.math.round((y_true + 1)/2)
    y_pred_r = tf.math.round((y_pred + 1)/2)
    r_true = tf.math.reduce_sum(y_true_r, axis=1)
    r_pred = tf.math.reduce_sum(y_pred_r, axis=1)
    r_bad = tf.cast(r_true != r_pred, tf.float32)

    ntrain = tf.cast(tf.shape(y_pred)[0], dtype=tf.float32)
    inacc = tf.math.reduce_sum(r_bad)/ntrain
    sh = tf_keras_sh(y_true, y_pred)
    return sh*(1.0 + inacc)


def sh_nononsense(y_true, y_pred):
    rdum = tf.math.round((y_pred + 1.0)/2.0)
    rnum = rdum.shape[1]
    rdumsum = tf.math.reduce_sum(rdum, axis=1)
    rwhere = tf.reshape(tf.where(rdumsum), [-1])
    rdum_no0 = tf.gather(rdum, rwhere)
    rsum_no0 = tf.gather(rdumsum, rwhere)

    rrdum = tf.gather(rdum_no0, list(range(rnum - 1, -1, -1)), axis=1)
    rargmax = tf.cast(tf.math.argmax(rrdum, axis=1), rdumsum.dtype)
    rdiff = (rnum - rargmax - rsum_no0)
    nssum = tf.math.reduce_sum(rdiff)
    # tf.print(nssum)
    sh = tf_keras_sh(y_true, y_pred)
    npred = tf.cast(tf.shape(y_pred)[0], dtype=tf.float32)
    return sh*(1.0 + nssum/npred)


def cce_jumping(y_true, y_pred):
    r_true = tf.cast(tf.math.argmax(y_true, axis=1), y_pred.dtype)
    r_pred = tf.cast(tf.math.argmax(y_pred, axis=1), y_pred.dtype)
    r_diff = r_true - r_pred
    r_adiff = tf.math.abs(r_diff)
    res = tf.math.reduce_mean(r_adiff)

    cce = tf_keras_cce(y_true, y_pred)
    return cce*res


def cce_jumping2(y_true, y_pred):
    r_true = tf.cast(tf.math.argmax(y_true, axis=1), y_pred.dtype)
    r_pred = tf.cast(tf.math.argmax(y_pred, axis=1), y_pred.dtype)
    r_diff = r_true - r_pred
    res = tf.math.reduce_mean(r_diff**2)
    cce = tf_keras_cce(y_true, y_pred)
    return cce*res


# %% 2D: Single network, multilabel

activation_out = 'tanh'
loss = 'squared_hinge'
snmrtanh2 = Sequential()
snmrtanh2.add(Dense(10*16*D, input_shape=(D,), activation='relu'))
snmrtanh2.add(Dense(10*8*D, activation='relu'))
snmrtanh2.add(Dense(len(lims), activation=activation_out))
snmrtanh2.compile(
    optimizer=Adam(learning_rate=learning_rate),
    loss=loss
)
snmrtanh2.fit(
    xohetrain/xside,
    to_multilabel(yohetrain, activation_out=activation_out),
    epochs=6000,
    batch_size=batch_size,
    verbose=verbose
)

activation_out = 'sigmoid'
loss = 'binary_crossentropy'
snmrsig2 = Sequential()
snmrsig2.add(Dense(10*16*D, input_shape=(D,), activation='relu'))
snmrsig2.add(Dense(10*8*D, activation='relu'))
snmrsig2.add(Dense(len(lims), activation=activation_out))
snmrsig2.compile(
    optimizer=Adam(learning_rate=learning_rate),
    loss=loss
)
snmrsig2.fit(
    xohetrain/xside,
    to_multilabel(yohetrain, activation_out=activation_out),
    epochs=6000,
    batch_size=batch_size,
    verbose=verbose
)

# %% FOUR DIMENSIONS

D = 4
xside = 10
radius = xside/4
centers = [[xside/4]*D, [3*xside/4]*D]

radii = [radius]*len(centers)
ntest = int(1e6)
Vtot4 = xside**D

xvals = np.random.rand(ntest, D)*xside
ffun4 = conegen(D, radii, centers)
fres = ffun4(xvals)

lims4 = [0.0]
lims4.append(
    functions.get_lim_err(
        fres[fres > lims4[-1]], 0.24,
        fromlim=lims4[-1],
        ntest=1000,
        nreal=ntest,
        vtotal=Vtot4,
        tstscale='linear'
    )
)

# Collect a training set and create limits
# This is not the process that will be used for actual results, this is for
# comparing different architectures
for j in range(9):
    xdum, fdum, ndum = sample_gen(20000, 10000000, D, xside, lims4[-1], ffun4)
    xvals = np.append(
        xvals,
        xdum,
        axis=0
    )
    fres = np.append(
        fres,
        fdum,
    )
    lims4.append(
        functions.get_lim_err(
            fdum, 2.0,
            fromlim=lims4[-1],
            ntest=1000,
            nreal=ndum,
            vtotal=Vtot4,
            tstscale='linear'
        )
    )
    print(lims4[-1])
    if lims4[-1] == -1:
        break

# %%

xgetind = functions.get_train_xy(xvals, fres, [-1] + lims4 + [2.5])
xohetrain = np.empty((0, D))
yohetrain = np.empty((0, 1))
fohetrain = np.empty((0))
for j in range(12):
    xohe_inreg = xgetind[0][xgetind[1].flatten() == j][:1000]
    xohetrain = np.append(
        xohetrain,
        xohe_inreg,
        axis=0
    )
    fohetrain = np.append(
        fohetrain,
        xgetind[2][xgetind[1].flatten() == j][:xohe_inreg.shape[0]]
    )
    yohetrain = np.append(
        yohetrain,
        np.full((xohe_inreg.shape[0], 1), j),
        axis=0
    )

# %% 4D: Multiple networks, predict single level

activation_out = "sigmoid"
loss = "binary_crossentropy"
xtrain = [None]*len(lims4)
ytrain = [None]*len(lims4)
for j in range(len(lims4)):
    xtrain_pre, ytrain_pre, _ = functions.get_train_xy(
        xohetrain, fohetrain, [-1, lims4[j], 2.5]
    )
    if activation_out == "tanh":
        ytrain_pre = ytrain_pre*2.0 - 1.0
    indum = np.arange(xtrain_pre.shape[0])
    np.random.shuffle(indum)
    xtrain[j] = xtrain_pre[indum]
    ytrain[j] = ytrain_pre[indum]

mnsrsig4 = [None]*len(lims4)
for j in range(len(lims4)):
    mnsrsig4[j] = Sequential()
    mnsrsig4[j].add(Dense(5*16*D, input_shape=(D,), activation='relu'))
    mnsrsig4[j].add(Dense(5*8*D, activation='relu'))
    mnsrsig4[j].add(Dense(12, activation='relu'))
    mnsrsig4[j].add(Dense(1, activation=activation_out))
    mnsrsig4[j].compile(
        optimizer=Adam(learning_rate=learning_rate),
        loss=loss
    )
    mnsrsig4[j].fit(
        xtrain[j]/xside,
        ytrain[j],
        epochs=4000,
        batch_size=batch_size,
        verbose=verbose
    )

activation_out = "tanh"
loss = "squared_hinge"
xtrain = [None]*len(lims4)
ytrain = [None]*len(lims4)
for j in range(len(lims4)):
    xtrain_pre, ytrain_pre, _ = functions.get_train_xy(
        xohetrain, fohetrain, [-1, lims4[j], 2.5]
    )
    if activation_out == "tanh":
        ytrain_pre = ytrain_pre*2.0 - 1.0
    indum = np.arange(xtrain_pre.shape[0])
    np.random.shuffle(indum)
    xtrain[j] = xtrain_pre[indum]
    ytrain[j] = ytrain_pre[indum]

mnsrtanh4 = [None]*len(lims4)
for j in range(len(lims4)):
    mnsrtanh4[j] = Sequential()
    mnsrtanh4[j].add(Dense(5*16*D, input_shape=(D,), activation='relu'))
    mnsrtanh4[j].add(Dense(5*8*D, activation='relu'))
    mnsrtanh4[j].add(Dense(12, activation='relu'))
    mnsrtanh4[j].add(Dense(1, activation=activation_out))
    mnsrtanh4[j].compile(
        optimizer=Adam(learning_rate=learning_rate),
        loss=loss
    )
    mnsrtanh4[j].fit(
        xtrain[j]/xside,
        ytrain[j],
        epochs=4000,
        batch_size=batch_size,
        verbose=verbose
    )

# %% 4D: Single network, softmax

activation_out = 'softmax'
losssm = 'categorical_crossentropy'
snsm4 = Sequential()
snsm4.add(Dense(10*16*D, input_shape=(D,), activation='relu'))
snsm4.add(Dense(10*8*D, activation='relu'))
snsm4.add(Dense(len(lims4) + 1, activation=activation_out))
snsm4.compile(
    optimizer=Adam(learning_rate=learning_rate),
    loss=losssm
)
snsm4.fit(
    xohetrain/xside,
    to_categorical(yohetrain),
    epochs=12000,
    batch_size=batch_size,
    verbose=verbose
)

# %% 4D: Single network, multilabel

activation_out = 'tanh'
loss = 'squared_hinge'
snmrtanh4 = Sequential()
snmrtanh4.add(Dense(10*16*D, input_shape=(D,), activation='relu'))
snmrtanh4.add(Dense(10*8*D, activation='relu'))
snmrtanh4.add(Dense(len(lims), activation=activation_out))
snmrtanh4.compile(
    optimizer=Adam(learning_rate=learning_rate),
    loss=loss
)
snmrtanh4.fit(
    xohetrain/xside,
    to_multilabel(yohetrain, activation_out=activation_out),
    epochs=12000,
    batch_size=batch_size,
    verbose=verbose
)

activation_out = 'sigmoid'
loss = 'binary_crossentropy'
# loss = bce_nononsense
snmrsig4 = Sequential()
snmrsig4.add(Dense(10*16*D, input_shape=(D,), activation='relu'))
snmrsig4.add(Dense(10*8*D, activation='relu'))
snmrsig4.add(Dense(len(lims), activation=activation_out))
snmrsig4.compile(
    optimizer=Adam(learning_rate=learning_rate),
    loss=loss
)
snmrsig4.fit(
    xohetrain/xside,
    to_multilabel(yohetrain, activation_out=activation_out),
    epochs=12000,
    batch_size=batch_size,
    verbose=verbose
)

# %% SIX DIMENSIONS

D = 6
xside = 10
radius = xside/4
centers = [[xside/4]*D, [3*xside/4]*D]

radii = [radius]*len(centers)
ntest = int(1e6)
Vtot6 = xside**D

xvals = np.random.rand(ntest, D)*xside
ffun6 = conegen(D, radii, centers)
fres = ffun6(xvals)

lims6 = [0.0]
lims6.append(
    functions.get_lim_err(
        fres[fres > lims6[-1]], 0.24,
        fromlim=lims6[-1],
        ntest=1000,
        nreal=ntest,
        vtotal=Vtot6,
        tstscale='linear'
    )
)

# Collect a training set and create limits
# This is not the process that will be used for actual results, this is for
# comparing different architectures
for j in range(9):
    xdum, fdum, ndum = sample_gen(20000, 10000000, D, xside, lims6[-1], ffun6)
    xvals = np.append(
        xvals,
        xdum,
        axis=0
    )
    fres = np.append(
        fres,
        fdum,
    )
    lims6.append(
        functions.get_lim_err(
            fdum, 10.4,
            fromlim=lims6[-1],
            ntest=1000,
            nreal=ndum,
            vtotal=Vtot6,
            tstscale='linear'
        )
    )
    print(lims6[-1])
    if lims6[-1] == -1:
        break


xdum, fdum, ndum = sample_gen(10000, int(1e7), D, xside, lims6[-1], ffun6, verbose=1)
xvals = np.append(
    xvals,
    xdum,
    axis=0
)
fres = np.append(
    fres,
    fdum,
)

# %%

xgetind = functions.get_train_xy(xvals, fres, [-1] + lims6 + [2.5])
xohetrain = np.empty((0, D))
yohetrain = np.empty((0, 1))
fohetrain = np.empty((0))
for j in range(12):
    xohe_inreg = xgetind[0][xgetind[1].flatten() == j][:1000]
    xohetrain = np.append(
        xohetrain,
        xohe_inreg,
        axis=0
    )
    fohetrain = np.append(
        fohetrain,
        xgetind[2][xgetind[1].flatten() == j][:xohe_inreg.shape[0]]
    )
    yohetrain = np.append(
        yohetrain,
        np.full((xohe_inreg.shape[0], 1), j),
        axis=0
    )

# %% Randomize data sorting

ind_shuf = np.arange(xohetrain.shape[0])
np.random.shuffle(ind_shuf)

xtr_sh = xohetrain[ind_shuf][:6000]
ytr_sh = yohetrain[ind_shuf][:6000]

xval_sh = xohetrain[ind_shuf][6000:]
yval_sh = yohetrain[ind_shuf][6000:]

# %% 6D: Multiple networks, predict single level
# Note increase in number of epochs

# activation_out = 'tanh'
# loss = 'squared_hinge'
# xtrain = [None]*len(lims6)
# ytrain = [None]*len(lims6)
# for j in range(len(lims6)):
#     xtrain_pre, ytrain_pre, _ = functions.get_train_xy(
#         xohetrain, fohetrain, [-1, lims6[j], 2.5]
#     )
#     if activation_out == "tanh":
#         ytrain_pre = ytrain_pre*2.0 - 1.0
#     indum = np.arange(xtrain_pre.shape[0])
#     np.random.shuffle(indum)
#     xtrain[j] = xtrain_pre[indum]
#     ytrain[j] = ytrain_pre[indum]

# mnsrtanh6 = [None]*len(lims6)
# for j in range(len(lims6)):
#     mnsrtanh6[j] = Sequential()
#     mnsrtanh6[j].add(Dense(5*16*D, input_shape=(D,), activation='relu'))
#     mnsrtanh6[j].add(Dense(5*8*D, activation='relu'))
#     mnsrtanh6[j].add(Dense(12, activation='relu'))
#     mnsrtanh6[j].add(Dense(1, activation=activation_out))
#     mnsrtanh6[j].compile(
#         optimizer=Adam(learning_rate=learning_rate),
#         loss=loss
#     )
#     mnsrtanh6[j].fit(
#         xtrain[j]/xside,
#         ytrain[j],
#         epochs=8000,
#         batch_size=batch_size,
#         verbose=verbose
#     )

# activation_out = 'sigmoid'
# loss = 'binary_crossentropy'
# xtrain = [None]*len(lims6)
# ytrain = [None]*len(lims6)
# for j in range(len(lims6)):
#     xtrain_pre, ytrain_pre, _ = functions.get_train_xy(
#         xohetrain, fohetrain, [-1, lims6[j], 2.5]
#     )
#     if activation_out == "tanh":
#         ytrain_pre = ytrain_pre*2.0 - 1.0
#     indum = np.arange(xtrain_pre.shape[0])
#     np.random.shuffle(indum)
#     xtrain[j] = xtrain_pre[indum]
#     ytrain[j] = ytrain_pre[indum]

# mnsrsig6 = [None]*len(lims6)
# for j in range(len(lims6)):
#     mnsrsig6[j] = Sequential()
#     mnsrsig6[j].add(Dense(5*16*D, input_shape=(D,), activation='relu'))
#     mnsrsig6[j].add(Dense(5*8*D, activation='relu'))
#     mnsrsig6[j].add(Dense(12, activation='relu'))
#     mnsrsig6[j].add(Dense(1, activation=activation_out))
#     mnsrsig6[j].compile(
#         optimizer=Adam(learning_rate=learning_rate),
#         loss=loss
#     )
#     mnsrsig6[j].fit(
#         xtrain[j]/xside,
#         ytrain[j],
#         epochs=8000,
#         batch_size=batch_size,
#         verbose=verbose
#     )


# %%
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


# Confusion matrix
def conf_mat(r_true, r_pred, normalize=False):
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


def model_create(
    dimensions,
    activation_out,
    nodes_start,
    nodes_out,
    loss,
    learning_rate=learning_rate
):
    D = dimensions
    model = Sequential()
    model.add(Dense(nodes_start, input_shape=(D,), activation='relu'))
    model.add(Dense(int(nodes_start/2 + 0.5), activation='relu'))
    model.add(Dense(nodes_out, activation=activation_out))
    model.compile(
        optimizer=Adam(learning_rate=learning_rate),
        loss=loss,
        metrics=[
            my_metric('accuracy', activation_out),
            my_metric('jumping', activation_out)
        ]
    )

    return model


def model_fit(
    model,
    xtrain,
    ytrain,
    xval,
    yval,
    epochs=2000,
    batch_size=batch_size,
    verbose=verbose
):
    fit_hist = model.fit(
        xtrain,
        ytrain,
        epochs=epochs,
        batch_size=batch_size,
        verbose=verbose,
        validation_data=(
            xval,
            yval
        )
    )
    return fit_hist


# %%
D = 6
# Number of distinct trainings to get average accuracies and jumpings
nruns = 10

# %% 6D: Single network, softmax
activation_out = 'softmax'
main_loss = "categorical_crossentropy"

loss = my_loss(name=main_loss)
snsm61_hists = [None]*nruns
snsm61_comat = [None]*nruns
for j in range(nruns):
    snsm61 = model_create(
        D, activation_out, 20*16*D, len(lims6) + 1,
        loss
    )

    snsm61_hists[j] = model_fit(
        snsm61,
        xtr_sh/xside,
        to_categorical(ytr_sh),
        xval_sh/xside,
        to_categorical(yval_sh)
    )

    rpred = tf.cast(tf.math.argmax(snsm61(xval_sh/xside), axis=1), dtype=tf.float32)
    snsm61_comat[j] = conf_mat(yval_sh, rpred, normalize=True)

loss = my_loss(name=main_loss, jumping="absolute")
snsm62_hists = [None]*nruns
snsm62_comat = [None]*nruns
for j in range(nruns):
    snsm62 = model_create(
        D, activation_out, 20*16*D, len(lims6) + 1,
        loss
    )

    snsm62_hists[j] = model_fit(
        snsm62,
        xtr_sh/xside,
        to_categorical(ytr_sh),
        xval_sh/xside,
        to_categorical(yval_sh)
    )

    rpred = tf.cast(tf.math.argmax(snsm62(xval_sh/xside), axis=1), dtype=tf.float32)
    snsm62_comat[j] = conf_mat(yval_sh, rpred, normalize=True)

loss = my_loss(name=main_loss, jumping="squared")
snsm63_hists = [None]*nruns
snsm63_comat = [None]*nruns
for j in range(nruns):
    snsm63 = model_create(
        D, activation_out, 20*16*D, len(lims6) + 1,
        loss
    )

    snsm63_hists[j] = model_fit(
        snsm63,
        xtr_sh/xside,
        to_categorical(ytr_sh),
        xval_sh/xside,
        to_categorical(yval_sh)
    )

    rpred = tf.cast(tf.math.argmax(snsm63(xval_sh/xside), axis=1), dtype=tf.float32)
    snsm63_comat[j] = conf_mat(yval_sh, rpred, normalize=True)

# %% tanh
activation_out = 'tanh'
main_loss = "squared_hinge"

loss = my_loss(name=main_loss)
snmltanh61_hists = [None]*nruns
snmltanh61_comat = [None]*nruns
for j in range(nruns):
    snmltanh61 = model_create(
        D, activation_out, 20*16*D, len(lims6),
        loss
    )

    snmltanh61_hists[j] = model_fit(
        snmltanh61,
        xtr_sh/xside,
        to_multilabel(ytr_sh, activation_out=activation_out),
        xval_sh/xside,
        to_multilabel(yval_sh, activation_out=activation_out)
    )

    rpred2 = tf.math.reduce_sum(tf.math.round((snmltanh61(xval_sh/xside) + 1.0)/2.0), axis=1)
    snmltanh61_comat[j] = conf_mat(yval_sh, rpred2, normalize=True)

loss = my_loss(name=main_loss, jumping="absolute")
snmltanh62_hists = [None]*nruns
snmltanh62_comat = [None]*nruns
for j in range(nruns):
    snmltanh62 = model_create(
        D, activation_out, 20*16*D, len(lims6),
        loss
    )

    snmltanh62_hists[j] = model_fit(
        snmltanh62,
        xtr_sh/xside,
        to_multilabel(ytr_sh, activation_out=activation_out),
        xval_sh/xside,
        to_multilabel(yval_sh, activation_out=activation_out)
    )

    rpred2 = tf.math.reduce_sum(tf.math.round((snmltanh62(xval_sh/xside) + 1.0)/2.0), axis=1)
    snmltanh62_comat[j] = conf_mat(yval_sh, rpred2, normalize=True)

loss = my_loss(name=main_loss, jumping="squared")
snmltanh63_hists = [None]*nruns
snmltanh63_comat = [None]*nruns
for j in range(nruns):
    snmltanh63 = model_create(
        D, activation_out, 20*16*D, len(lims6),
        loss
    )

    snmltanh63_hists[j] = model_fit(
        snmltanh63,
        xtr_sh/xside,
        to_multilabel(ytr_sh, activation_out=activation_out),
        xval_sh/xside,
        to_multilabel(yval_sh, activation_out=activation_out)
    )

    rpred2 = tf.math.reduce_sum(tf.math.round((snmltanh63(xval_sh/xside) + 1.0)/2.0), axis=1)
    snmltanh63_comat[j] = conf_mat(yval_sh, rpred2, normalize=True)

# %% sigmoid
activation_out = 'sigmoid'
main_loss = "binary_crossentropy"

loss = my_loss(name=main_loss)
snmlsig61_hists = [None]*nruns
snmlsig61_comat = [None]*nruns
for j in range(nruns):
    snmlsig61 = model_create(
        D, activation_out, 20*16*D, len(lims6),
        loss
    )

    snmlsig61_hists[j] = model_fit(
        snmlsig61,
        xtr_sh/xside,
        to_multilabel(ytr_sh, activation_out=activation_out),
        xval_sh/xside,
        to_multilabel(yval_sh, activation_out=activation_out)
    )
    rpred3 = tf.math.reduce_sum(tf.math.round(snmlsig61(xval_sh/xside)), axis=1)
    snmlsig61_comat[j] = conf_mat(yval_sh, rpred3, normalize=True)

loss = my_loss(name=main_loss, jumping="absolute")
snmlsig62_hists = [None]*nruns
snmlsig62_comat = [None]*nruns
for j in range(nruns):
    snmlsig62 = model_create(
        D, activation_out, 20*16*D, len(lims6),
        loss
    )

    snmlsig62_hists[j] = model_fit(
        snmlsig62,
        xtr_sh/xside,
        to_multilabel(ytr_sh, activation_out=activation_out),
        xval_sh/xside,
        to_multilabel(yval_sh, activation_out=activation_out)
    )
    rpred3 = tf.math.reduce_sum(tf.math.round(snmlsig62(xval_sh/xside)), axis=1)
    snmlsig62_comat[j] = conf_mat(yval_sh, rpred3, normalize=True)

loss = my_loss(name=main_loss, jumping="squared")
snmlsig63_hists = [None]*nruns
snmlsig63_comat = [None]*nruns
for j in range(nruns):
    snmlsig63 = model_create(
        D, activation_out, 20*16*D, len(lims6),
        loss
    )

    snmlsig63_hists[j] = model_fit(
        snmlsig63,
        xtr_sh/xside,
        to_multilabel(ytr_sh, activation_out=activation_out),
        xval_sh/xside,
        to_multilabel(yval_sh, activation_out=activation_out)
    )
    rpred3 = tf.math.reduce_sum(tf.math.round(snmlsig63(xval_sh/xside)), axis=1)
    snmlsig63_comat[j] = conf_mat(yval_sh, rpred3, normalize=True)


# %%
# TODO Put together deviations and averages for jumping and accuracy
def build_stats(histories):
    nhistrs = len(histories)

    jump_test = []
    accu_test = []
    jump_train = []
    accu_train = []
    for j in range(nhistrs):
        jump_test.append(histories[j].history['val_jumping'])
        accu_test.append(histories[j].history['val_accuracy'])
        jump_train.append(histories[j].history['jumping'])
        accu_train.append(histories[j].history['accuracy'])

    jumpmean_t = np.mean(jump_test, axis=0)
    accumean_t = np.mean(accu_test, axis=0)
    jumpstd_t = np.std(jump_test, axis=0)
    accustd_t = np.std(accu_test, axis=0)

    return jumpmean_t, jumpstd_t, accumean_t, accustd_t


def history_mean_std(histories):
    nhistrs = len(histories)

    keys = list(histories[0].history.keys())
    # keys = [
    #     'loss', 'jumping', 'accuracy',
    #     'val_loss', 'val_jumping', 'val_accuracy'
    # ]

    histout = {}
    for key in keys:
        gthred = []
        for j in range(nhistrs):
            gthred.append(histories[j].history[key])
        histout[key + '_mean'] = np.mean(gthred, axis=0)
        histout[key + '_std'] = np.std(gthred, axis=0)

        if key in ['loss', 'jumping', 'val_loss', 'val_jumping']:
            kbest = np.argmin(
                [histories[k].history[key][-1] for k in range(nhistrs)]
            )
            kworst = np.argmax(
                [histories[k].history[key][-1] for k in range(nhistrs)]
            )
        else:
            kbest = np.argmax(
                [histories[k].history[key][-1] for k in range(nhistrs)]
            )
            kworst = np.argmin(
                [histories[k].history[key][-1] for k in range(nhistrs)]
            )
        histout[key + '_best'] = histories[kbest].history[key]
        histout[key + '_best_index'] = kbest
        histout[key + '_worst'] = histories[kworst].history[key]
        histout[key + '_worst_index'] = kworst

    return histout


# %%

snsm61_hms = history_mean_std(snsm61_hists)
snsm62_hms = history_mean_std(snsm62_hists)

snmltanh61_hms = history_mean_std(snmltanh61_hists)
snmltanh62_hms = history_mean_std(snmltanh62_hists)
# snmltanh63_hms = history_mean_std(snmltanh63_hists)

snmlsig61_hms = history_mean_std(snmlsig61_hists)
snmlsig62_hms = history_mean_std(snmlsig62_hists)
# snmlsig63_hms = history_mean_std(snmlsig63_hists)

# snsm63_hms = history_mean_std(snsm62_hists)

# %% softmax jumping
figsize = (4, 4)

btl1 = snsm61_hms['loss_best_index']
btl2 = snsm62_hms['loss_best_index']
plt.figure(figsize=figsize)
plt.title("softmax, CCE: categorical cross-entropy")
plt.plot(
    np.arange(1, 2001),
    snsm61_hists[btl1].history['jumping'],
    color='C0',
    alpha=0.7,
    label='CCE, training'
)
plt.plot(
    np.arange(1, 2001),
    # snsm61_hms['val_jumping_best'],
    snsm61_hists[btl1].history['val_jumping'],
    color='C2',
    alpha=0.9,
    label='CCE, validation'
)
plt.plot(
    np.arange(1, 2001),
    snsm62_hists[btl2].history['jumping'],
    alpha=0.9,
    color='C1',
    label='CCE$\\times$jumps, train.'
)
plt.plot(
    np.arange(1, 2001),
    snsm62_hists[btl2].history['val_jumping'],
    color='C3',
    alpha=0.7,
    label='CCE$\\times$jumps, val.'
)
ax = plt.gca()
plt.text(
    0.01,
    0.02,
    "Best training loss out of 10",
    transform=ax.transAxes
)
plt.ylabel("Average jumping")
plt.xlabel("Epochs")
plt.xlim(1, 2000)
plt.legend(frameon=False)
plt.yscale('log')
plt.yticks([0.5, 1, 5], [0.5, 1, 5])
plt.savefig("sm_jump_best.pgf", bbox_inches="tight")
plt.close('all')

btl1 = snsm61_hms['loss_worst_index']
btl2 = snsm62_hms['loss_worst_index']
plt.figure(figsize=figsize)
plt.title("softmax, CCE: categorical cross-entropy")
plt.plot(
    np.arange(1, 2001),
    snsm61_hists[btl1].history['jumping'],
    color='C0',
    alpha=0.7,
    label='CCE, training'
)
plt.plot(
    np.arange(1, 2001),
    # snsm61_hms['val_jumping_best'],
    snsm61_hists[btl1].history['val_jumping'],
    color='C2',
    alpha=0.9,
    label='CCE, validation'
)
plt.plot(
    np.arange(1, 2001),
    snsm62_hists[btl2].history['jumping'],
    alpha=0.9,
    color='C1',
    label='CCE$\\times$jumps, train.'
)
plt.plot(
    np.arange(1, 2001),
    snsm62_hists[btl2].history['val_jumping'],
    color='C3',
    alpha=0.7,
    label='CCE$\\times$jumps, val.'
)
ax = plt.gca()
plt.text(
    0.01,
    0.02,
    "Worst training loss out of 10",
    transform=ax.transAxes
)
plt.ylabel("Average jumping")
plt.xlabel("Epochs")
plt.xlim(1, 2000)
plt.legend(loc='upper left', frameon=False)
plt.yscale('log')
plt.yticks([0.5, 1, 5], [0.5, 1, 5])
plt.savefig("sm_jump_worst.pgf", bbox_inches="tight")
plt.close('all')

# %% tanh jumping

btl1 = snmltanh61_hms['loss_best_index']
btl2 = snmltanh62_hms['loss_best_index']
plt.figure(figsize=figsize)
plt.title("tanh, SH: squared hinge")
plt.plot(
    np.arange(1, 2001),
    snmltanh61_hists[btl1].history['jumping'],
    color='C0',
    alpha=0.7,
    label='SH, training'
)
plt.plot(
    np.arange(1, 2001),
    # snmltanh61_hms['val_jumping_best'],
    snmltanh61_hists[btl1].history['val_jumping'],
    color='C2',
    alpha=0.9,
    label='SH, validation'
)
plt.plot(
    np.arange(1, 2001),
    snmltanh62_hists[btl2].history['jumping'],
    alpha=0.9,
    color='C1',
    label='SH$\\times$jumps, train.'
)
plt.plot(
    np.arange(1, 2001),
    snmltanh62_hists[btl2].history['val_jumping'],
    color='C3',
    alpha=0.7,
    label='SH$\\times$jumps, val.'
)
ax = plt.gca()
plt.text(
    0.01,
    0.02,
    "Best training loss out of 10",
    transform=ax.transAxes
)
plt.ylabel("Average jumping")
plt.xlabel("Epochs")
plt.xlim(1, 2000)
plt.legend(frameon=False)
plt.yscale('log')
plt.yticks([0.5, 1, 5], [0.5, 1, 5])
plt.savefig("snmltanh_jump_best.pgf", bbox_inches="tight")
plt.close('all')

btl1 = snmltanh61_hms['loss_worst_index']
btl2 = snmltanh62_hms['loss_worst_index']
plt.figure(figsize=figsize)
plt.title("tanh, SH: squared hinge")
plt.plot(
    np.arange(1, 2001),
    snmltanh61_hists[btl1].history['jumping'],
    color='C0',
    alpha=0.7,
    label='SH, training'
)
plt.plot(
    np.arange(1, 2001),
    # snmltanh61_hms['val_jumping_best'],
    snmltanh61_hists[btl1].history['val_jumping'],
    color='C2',
    alpha=0.9,
    label='SH, validation'
)
plt.plot(
    np.arange(1, 2001),
    snmltanh62_hists[btl2].history['jumping'],
    alpha=0.9,
    color='C1',
    label='SH$\\times$jumps, train.'
)
plt.plot(
    np.arange(1, 2001),
    snmltanh62_hists[btl2].history['val_jumping'],
    color='C3',
    alpha=0.7,
    label='SH$\\times$jumps, val.'
)
ax = plt.gca()
plt.text(
    0.01,
    0.02,
    "Worst training loss out of 10",
    transform=ax.transAxes
)
plt.ylabel("Average jumping")
plt.xlabel("Epochs")
plt.xlim(1, 2000)
plt.legend(loc='upper center', frameon=False)
plt.yscale('log')
plt.yticks([0.5, 1, 5], [0.5, 1, 5])
plt.savefig("snmltanh_jump_worst.pgf", bbox_inches="tight")
plt.close('all')

# %% sigmoid jumping

btl1 = snmlsig61_hms['loss_best_index']
btl2 = snmlsig62_hms['loss_best_index']
plt.figure(figsize=figsize)
plt.title("sigmoid, BCE: binary cross-entropy")
plt.plot(
    np.arange(1, 2001),
    snmlsig61_hists[btl1].history['jumping'],
    color='C0',
    alpha=0.7,
    label='BCE, training'
)
plt.plot(
    np.arange(1, 2001),
    # snmlsig61_hms['val_jumping_best'],
    snmlsig61_hists[btl1].history['val_jumping'],
    color='C2',
    alpha=0.9,
    label='BCE, validation'
)
plt.plot(
    np.arange(1, 2001),
    snmlsig62_hists[btl2].history['jumping'],
    alpha=0.9,
    color='C1',
    label='BCE$\\times$jumps, train.'
)
plt.plot(
    np.arange(1, 2001),
    snmlsig62_hists[btl2].history['val_jumping'],
    color='C3',
    alpha=0.7,
    label='BCE$\\times$jumps, val.'
)
ax = plt.gca()
plt.text(
    0.01,
    0.02,
    "Best training loss out of 10",
    transform=ax.transAxes
)
plt.ylabel("Average jumping")
plt.xlabel("Epochs")
plt.xlim(1, 2000)
plt.legend(frameon=False)
plt.yscale('log')
plt.yticks([0.5, 1, 5], [0.5, 1, 5])
plt.savefig("snmlsig_jump_best.pgf", bbox_inches="tight")
plt.close('all')

btl1 = snmlsig61_hms['loss_worst_index']
btl2 = snmlsig62_hms['loss_worst_index']
plt.figure(figsize=figsize)
plt.title("sigmoid, BCE: binary cross-entropy")
plt.plot(
    np.arange(1, 2001),
    snmlsig61_hists[btl1].history['jumping'],
    color='C0',
    alpha=0.7,
    label='BCE, training'
)
plt.plot(
    np.arange(1, 2001),
    # snmlsig61_hms['val_jumping_best'],
    snmlsig61_hists[btl1].history['val_jumping'],
    color='C2',
    alpha=0.9,
    label='BCE, validation'
)
plt.plot(
    np.arange(1, 2001),
    snmlsig62_hists[btl2].history['jumping'],
    alpha=0.9,
    color='C1',
    label='BCE$\\times$jumps, train.'
)
plt.plot(
    np.arange(1, 2001),
    snmlsig62_hists[btl2].history['val_jumping'],
    color='C3',
    alpha=0.7,
    label='BCE$\\times$jumps, val.'
)
ax = plt.gca()
plt.text(
    0.01,
    0.02,
    "Worst training loss out of 10",
    transform=ax.transAxes
)
plt.ylabel("Average jumping")
plt.xlabel("Epochs")
plt.xlim(1, 2000)
plt.legend(loc='upper center', frameon=False)
plt.yscale('log')
plt.yticks([0.5, 1, 5], [0.5, 1, 5])
plt.savefig("snmlsig_jump_worst.pgf", bbox_inches="tight")
plt.close('all')

# %% softmax accuracy

btl1 = snsm61_hms['loss_best_index']
btl2 = snsm62_hms['loss_best_index']
plt.figure(figsize=figsize)
plt.title("softmax, CCE: categorical cross-entropy")
plt.plot(
    np.arange(1, 2001),
    snsm61_hists[btl1].history['accuracy'],
    color='C0',
    alpha=0.7,
    label='CCE, training'
)
plt.plot(
    np.arange(1, 2001),
    # snsm61_hms['val_accuracy_best'],
    snsm61_hists[btl1].history['val_accuracy'],
    color='C2',
    alpha=0.9,
    label='CCE, validation'
)
plt.plot(
    np.arange(1, 2001),
    snsm62_hists[btl2].history['accuracy'],
    alpha=0.9,
    color='C1',
    label='CCE$\\times$jumps, train.'
)
plt.plot(
    np.arange(1, 2001),
    snsm62_hists[btl2].history['val_accuracy'],
    color='C3',
    alpha=0.7,
    label='CCE$\\times$jumps, val.'
)
ax = plt.gca()
plt.text(
    0.01,
    0.95,
    "Best training loss out of 10",
    transform=ax.transAxes
)
plt.ylabel("Accuracy")
plt.xlabel("Epochs")
plt.xlim(1, 2000)
plt.legend(frameon=False)
plt.savefig("sm_accu_best.pgf", bbox_inches="tight")
plt.close('all')

btl1 = snsm61_hms['loss_worst_index']
btl2 = snsm62_hms['loss_worst_index']
plt.figure(figsize=figsize)
plt.title("softmax, CCE: categorical cross-entropy")
plt.plot(
    np.arange(1, 2001),
    snsm61_hists[btl1].history['accuracy'],
    color='C0',
    alpha=0.7,
    label='CCE, training'
)
plt.plot(
    np.arange(1, 2001),
    # snsm61_hms['val_accuracy_best'],
    snsm61_hists[btl1].history['val_accuracy'],
    color='C2',
    alpha=0.9,
    label='CCE, validation'
)
plt.plot(
    np.arange(1, 2001),
    snsm62_hists[btl2].history['accuracy'],
    alpha=0.9,
    color='C1',
    label='CCE$\\times$jumps, train.'
)
plt.plot(
    np.arange(1, 2001),
    snsm62_hists[btl2].history['val_accuracy'],
    color='C3',
    alpha=0.7,
    label='CCE$\\times$jumps, val.'
)
plt.ylabel("Accuracy")
plt.xlabel("Epochs")
plt.xlim(1, 2000)
ax = plt.gca()
plt.text(
    0.01,
    0.95,
    "Worst training loss out of 10",
    transform=ax.transAxes
)
plt.legend(loc=(0.11, 0), frameon=False)
plt.savefig("sm_accu_worst.pgf", bbox_inches="tight")
plt.close('all')

# %% tanh accuracy

btl1 = snmltanh61_hms['loss_best_index']
btl2 = snmltanh62_hms['loss_best_index']
plt.figure(figsize=figsize)
plt.title("tanh, SH: squared hinge")
plt.plot(
    np.arange(1, 2001),
    snmltanh61_hists[btl1].history['accuracy'],
    color='C0',
    alpha=0.7,
    label='SH, training'
)
plt.plot(
    np.arange(1, 2001),
    # snmltanh61_hms['val_accuracy_best'],
    snmltanh61_hists[btl1].history['val_accuracy'],
    color='C2',
    alpha=0.9,
    label='SH, validation'
)
plt.plot(
    np.arange(1, 2001),
    snmltanh62_hists[btl2].history['accuracy'],
    alpha=0.9,
    color='C1',
    label='SH$\\times$jumps, train.'
)
plt.plot(
    np.arange(1, 2001),
    snmltanh62_hists[btl2].history['val_accuracy'],
    color='C3',
    alpha=0.7,
    label='SH$\\times$jumps, val.'
)
ax = plt.gca()
plt.text(
    0.01,
    0.95,
    "Best training loss out of 10",
    transform=ax.transAxes
)
plt.ylabel("Accuracy")
plt.xlabel("Epochs")
plt.xlim(1, 2000)
plt.legend(frameon=False)
plt.savefig("snmltanh_accu_best.pgf", bbox_inches="tight")
plt.close('all')

btl1 = snmltanh61_hms['loss_worst_index']
btl2 = snmltanh62_hms['loss_worst_index']
plt.figure(figsize=figsize)
plt.title("tanh, SH: squared hinge")
plt.plot(
    np.arange(1, 2001),
    snmltanh61_hists[btl1].history['accuracy'],
    color='C0',
    alpha=0.7,
    label='SH, training'
)
plt.plot(
    np.arange(1, 2001),
    # snmltanh61_hms['val_accuracy_best'],
    snmltanh61_hists[btl1].history['val_accuracy'],
    color='C2',
    alpha=0.9,
    label='SH, validation'
)
plt.plot(
    np.arange(1, 2001),
    snmltanh62_hists[btl2].history['accuracy'],
    alpha=0.9,
    color='C1',
    label='SH$\\times$jumps, train.'
)
plt.plot(
    np.arange(1, 2001),
    snmltanh62_hists[btl2].history['val_accuracy'],
    color='C3',
    alpha=0.7,
    label='SH$\\times$jumps, val.'
)
plt.ylabel("Accuracy")
plt.xlabel("Epochs")
plt.xlim(1, 2000)
ax = plt.gca()
plt.text(
    0.01,
    0.95,
    "Worst training loss out of 10",
    transform=ax.transAxes
)
plt.legend(loc='lower center', frameon=False)
plt.savefig("snmltanh_accu_worst.pgf", bbox_inches="tight")
plt.close('all')

# %% sigmoid accuracy

btl1 = snmlsig61_hms['loss_best_index']
btl2 = snmlsig62_hms['loss_best_index']
plt.figure(figsize=figsize)
plt.title("sigmoid, BCE: binary cross-entropy")
plt.plot(
    np.arange(1, 2001),
    snmlsig61_hists[btl1].history['accuracy'],
    color='C0',
    alpha=0.7,
    label='BCE, training'
)
plt.plot(
    np.arange(1, 2001),
    # snmlsig61_hms['val_accuracy_best'],
    snmlsig61_hists[btl1].history['val_accuracy'],
    color='C2',
    alpha=0.9,
    label='BCE, validation'
)
plt.plot(
    np.arange(1, 2001),
    snmlsig62_hists[btl2].history['accuracy'],
    alpha=0.9,
    color='C1',
    label='BCE$\\times$jumps, train.'
)
plt.plot(
    np.arange(1, 2001),
    snmlsig62_hists[btl2].history['val_accuracy'],
    color='C3',
    alpha=0.7,
    label='BCE$\\times$jumps, val.'
)
ax = plt.gca()
plt.text(
    0.01,
    0.95,
    "Best training loss out of 10",
    transform=ax.transAxes
)
plt.ylabel("Accuracy")
plt.xlabel("Epochs")
plt.xlim(1, 2000)
plt.legend(frameon=False)
plt.savefig("snmlsig_accu_best.pgf", bbox_inches="tight")
plt.close('all')

btl1 = snmlsig61_hms['loss_worst_index']
btl2 = snmlsig62_hms['loss_worst_index']
plt.figure(figsize=figsize)
plt.title("sigmoid, BCE: binary cross-entropy")
plt.plot(
    np.arange(1, 2001),
    snmlsig61_hists[btl1].history['accuracy'],
    color='C0',
    alpha=0.7,
    label='BCE, training'
)
plt.plot(
    np.arange(1, 2001),
    # snmlsig61_hms['val_accuracy_best'],
    snmlsig61_hists[btl1].history['val_accuracy'],
    color='C2',
    alpha=0.9,
    label='BCE, validation'
)
plt.plot(
    np.arange(1, 2001),
    snmlsig62_hists[btl2].history['accuracy'],
    alpha=0.9,
    color='C1',
    label='BCE$\\times$jumps, train.'
)
plt.plot(
    np.arange(1, 2001),
    snmlsig62_hists[btl2].history['val_accuracy'],
    color='C3',
    alpha=0.7,
    label='BCE$\\times$jumps, val.'
)
plt.ylabel("Accuracy")
plt.xlabel("Epochs")
plt.xlim(1, 2000)
ax = plt.gca()
plt.text(
    0.01,
    0.95,
    "Worst training loss out of 10",
    transform=ax.transAxes
)
plt.legend(loc='lower center', frameon=False)
plt.savefig("snmlsig_accu_worst.pgf", bbox_inches="tight")
plt.close('all')

# %% Confusion matrices

cbar_rect = [0.94, 0.11, 0.045, 0.7699]

cmats = [
    np.mean(snsm62_comat, axis=0),
    np.mean(snmltanh62_comat, axis=0),
    np.mean(snmlsig61_comat, axis=0)
]
cmat_fns = [
    "snsm2_cm.pgf",
    "snmltanh2_cm.pgf",
    "snmlsig1_cm.pgf"
]
cmat_titles = [
    "softmax, CCE$\\times$jumps",
    "tanh, SH$\\times$jumps",
    "sigmoid, BCE",
]

for j in range(len(cmats)):
    fig, ax = plt.subplots(figsize=figsize)
    im = plt_confusion_matrix(
        ax, cmats[j],
        cmap='binary',
        ticks_step=2
    )

    ax.set_title(cmat_titles[j])
    ax.set_xlabel("Predicted region index")
    ax.set_ylabel("True region index")
    cb_ax = fig.add_axes(cbar_rect)
    fig.colorbar(im, cax=cb_ax)
    plt.savefig(cmat_fns[j], bbox_inches="tight")
    plt.close('all')

# %%

losssm = my_loss(name="categorical_crossentropy", jumping="absolute")
snsm62 = Sequential()
snsm62.add(Dense(20*16*D, input_shape=(D,), activation='relu'))
snsm62.add(Dense(20*8*D, activation='relu'))
snsm62.add(Dense(len(lims6) + 1, activation='softmax'))
snsm62.compile(
    optimizer=Adam(learning_rate=learning_rate),
    loss=losssm,
    metrics=[my_metric('accuracy', 'softmax'), my_metric('jumping', 'softmax')]
)
snsm62_hist = snsm62.fit(
    tf.convert_to_tensor(xtr_sh/xside, tf.float32),
    to_categorical(ytr_sh),
    epochs=2000,
    batch_size=batch_size,
    verbose=verbose,
    validation_data=(
        tf.convert_to_tensor(xval_sh/xside, tf.float32), to_categorical(yval_sh))
)

losssm = my_loss(name="categorical_crossentropy", jumping="squared")
snsm63 = Sequential()
snsm63.add(Dense(20*16*D, input_shape=(D,), activation='relu'))
snsm63.add(Dense(20*8*D, activation='relu'))
snsm63.add(Dense(len(lims6) + 1, activation='softmax'))
snsm63.compile(
    optimizer=Adam(learning_rate=learning_rate),
    loss=losssm,
    metrics=[my_metric('accuracy', 'softmax'), my_metric('jumping', 'softmax')]
)
snsm63_hist = snsm63.fit(
    tf.convert_to_tensor(xtr_sh/xside, tf.float32),
    to_categorical(ytr_sh),
    epochs=2000,
    batch_size=batch_size,
    verbose=verbose,
    validation_data=(
        tf.convert_to_tensor(xval_sh/xside, tf.float32),
        to_categorical(yval_sh))
)


# %% 6D: Single network, multilabel
# %% TANH
activation_out = 'tanh'
snmltanh61 = model_create(
    D, activation_out, 20*16*D, len(lims6),
    my_loss(name="squared_hinge")
)

nruns = 10
snmltanh61_hists = [None]*nruns
for j in range(nruns):
    snmltanh61_hists[j] = model_fit(
        snmltanh61,
        xtr_sh/xside,
        to_multilabel(ytr_sh, activation_out=activation_out),
        xval_sh/xside,
        to_multilabel(yval_sh, activation_out=activation_out)
    )

# ============================

loss = my_loss(name="squared_hinge", jumping="absolute")
snmltanh62 = Sequential()
snmltanh62.add(Dense(20*16*D, input_shape=(D,), activation='relu'))
snmltanh62.add(Dense(20*8*D, activation='relu'))
snmltanh62.add(Dense(len(lims6), activation=activation_out))
snmltanh62.compile(
    optimizer=Adam(learning_rate=learning_rate),
    loss=loss,
    metrics=[
        my_metric('accuracy', activation_out),
        my_metric('jumping', activation_out)
    ]
)
snmltanh62_hist = snmltanh62.fit(
    xtr_sh/xside,
    to_multilabel(ytr_sh, activation_out=activation_out),
    epochs=2000,
    batch_size=batch_size,
    verbose=verbose,
    validation_data=(
        xval_sh/xside,
        to_multilabel(yval_sh, activation_out=activation_out)
    )
)

loss = my_loss(name="squared_hinge", jumping="squared")
snmltanh63 = Sequential()
snmltanh63.add(Dense(20*16*D, input_shape=(D,), activation='relu'))
snmltanh63.add(Dense(20*8*D, activation='relu'))
snmltanh63.add(Dense(len(lims6), activation=activation_out))
snmltanh63.compile(
    optimizer=Adam(learning_rate=learning_rate),
    loss=loss,
    metrics=[
        my_metric('accuracy', activation_out),
        my_metric('jumping', activation_out)
    ]
)
snmltanh63_hist = snmltanh63.fit(
    xtr_sh/xside,
    to_multilabel(ytr_sh, activation_out=activation_out),
    epochs=2000,
    batch_size=batch_size,
    verbose=verbose,
    validation_data=(
        xval_sh/xside,
        to_multilabel(yval_sh, activation_out=activation_out)
    )
)

# %% SIGMOID
loss = my_loss(name="binary_crossentropy")
snmlsig61 = Sequential()
snmlsig61.add(Dense(20*16*D, input_shape=(D,), activation='relu'))
snmlsig61.add(Dense(20*8*D, activation='relu'))
snmlsig61.add(Dense(len(lims6), activation=activation_out))
snmlsig61.compile(
    optimizer=Adam(learning_rate=learning_rate),
    loss=loss,
    metrics=[
        my_metric('accuracy', activation_out),
        my_metric('jumping', activation_out)
    ]
)
snmlsig61_hist = snmlsig61.fit(
    xtr_sh/xside,
    to_multilabel(ytr_sh, activation_out=activation_out),
    epochs=2000,
    batch_size=batch_size,
    verbose=verbose,
    validation_data=(
        xval_sh/xside,
        to_multilabel(yval_sh, activation_out=activation_out)
    )
)

loss = my_loss(name="binary_crossentropy", jumping="absolute")
# loss = bce_nononsense
snmlsig62 = Sequential()
snmlsig62.add(Dense(20*16*D, input_shape=(D,), activation='relu'))
snmlsig62.add(Dense(20*8*D, activation='relu'))
snmlsig62.add(Dense(len(lims6), activation=activation_out))
snmlsig62.compile(
    optimizer=Adam(learning_rate=learning_rate),
    loss=loss,
    metrics=[
        my_metric('accuracy', activation_out),
        my_metric('jumping', activation_out)
    ]
)
snmlsig62_hist = snmlsig62.fit(
    xtr_sh/xside,
    to_multilabel(ytr_sh, activation_out=activation_out),
    epochs=2000,
    batch_size=batch_size,
    verbose=verbose,
    validation_data=(
        xval_sh/xside,
        to_multilabel(yval_sh, activation_out=activation_out)
    )
)

loss = my_loss(name="binary_crossentropy", jumping="squared")
# loss = bce_nononsense
snmlsig63 = Sequential()
snmlsig63.add(Dense(20*16*D, input_shape=(D,), activation='relu'))
snmlsig63.add(Dense(20*8*D, activation='relu'))
snmlsig63.add(Dense(len(lims6), activation=activation_out))
snmlsig63.compile(
    optimizer=Adam(learning_rate=learning_rate),
    loss=loss,
    metrics=[
        my_metric('accuracy', activation_out),
        my_metric('jumping', activation_out)
    ]
)
snmlsig63_hist = snmlsig63.fit(
    xtr_sh/xside,
    to_multilabel(ytr_sh, activation_out=activation_out),
    epochs=2000,
    batch_size=batch_size,
    verbose=verbose,
    validation_data=(
        xval_sh/xside,
        to_multilabel(yval_sh, activation_out=activation_out)
    )
)

# %%

# plt.title("Validation loss")
# plt.plot(
#     np.arange(1000) + 1,
#     np.array(snsm6_hist.history['val_loss'])
# )
# plt.plot(
#     np.arange(1000) + 1,
#     np.array(snmrtanh6_hist.history['val_loss'])
# )
# plt.plot
#     np.arange(1000) + 1,
#     np.array(snmrsig6_hist.history['val_loss'])
# )
# plt.plot(
#     np.arange(1000) + 1,
#     np.array(snmrsig62_hist.history['val_loss'])
# )
# plt.yscale('log')
# # plt.xscale('log')
# plt.xlabel("Epoch")
# plt.ylabel("Loss value")
# plt.show()

plt.title("Average region jumping with validation set")
plt.plot(
    np.arange(2000) + 1,
    np.array(snsm6_hist.history['val_jumping'])/xval_sh.shape[0]
)
plt.plot(
    np.arange(2000) + 1,
    np.array(snsm62_hist.history['val_jumping'])/xval_sh.shape[0]
)
plt.plot(
    np.arange(2000) + 1,
    np.array(snsm63_hist.history['val_jumping'])/xval_sh.shape[0]
)
# plt.plot(
#     np.arange(2000) + 1,
#     np.array(snmrtanh6_hist.history['val_jumping'])/xval_sh.shape[0]
# )
# plt.plot(
#     np.arange(1000) + 1,
#     np.array(snmrsig6_hist.history['val_jumping'])/xval_sh.shape[0]
# )
# plt.plot(
#     np.arange(1000) + 1,
#     np.array(snmrsig62_hist.history['val_jumping'])/xval_sh.shape[0]
# )
# plt.plot(
#     np.arange(2000) + 1,
#     np.array(snmrsig63_hist.history['val_jumping'])/xval_sh.shape[0]
# )
# plt.plot(
#     np.arange(2000) + 1,
#     np.array(snmrsig64_hist.history['val_jumping'])/xval_sh.shape[0]
# )
# plt.plot(
#     np.arange(2000) + 1,
#     np.array(snmrsig65_hist.history['val_jumping'])/xval_sh.shape[0]
# )
plt.yscale('log')
# plt.xscale('log')
# plt.ylim(0.5, 1)
plt.xlabel("Epoch")
plt.ylabel("Average jump size")
plt.show()

plt.title("Validation accuracy")
plt.plot(
    np.arange(2000) + 1,
    np.array(snsm6_hist.history['val_accuracy'])
)
plt.plot(
    np.arange(2000) + 1,
    np.array(snsm62_hist.history['val_accuracy'])
)
plt.plot(
    np.arange(2000) + 1,
    np.array(snsm63_hist.history['val_accuracy'])
)
# plt.plot(
#     np.arange(2000) + 1,
#     np.array(snmrtanh6_hist.history['val_accuracy'])
# )
# plt.plot(
#     np.arange(1000) + 1,
#     np.array(snmrsig6_hist.history['val_accuracy'])
# )
# plt.plot(
#     np.arange(1000) + 1,
#     np.array(snmrsig62_hist.history['val_accuracy'])
# )
# plt.plot(
#     np.arange(2000) + 1,
#     np.array(snmrsig63_hist.history['val_accuracy'])
# )
# plt.plot(
#     np.arange(2000) + 1,
#     np.array(snmrsig64_hist.history['val_accuracy'])
# )
# plt.plot(
#     np.arange(2000) + 1,
#     np.array(snmrsig65_hist.history['val_accuracy'])
# )
plt.yscale('log')
# plt.xscale('log')
plt.xlabel("Epoch")
plt.ylabel("Accuracy")
plt.show()

# plt.title("Validation loss")
# plt.plot(
#     np.arange(2000) + 1,
#     np.array(snsm6_hist.history['val_loss'])
# )
# plt.plot(
#     np.arange(2000) + 1,
#     np.array(snsm62_hist.history['val_loss'])
# )
# plt.plot(
#     np.arange(2000) + 1,
#     np.array(snmrtanh6_hist.history['val_loss'])
# )
# # plt.plot(
# #     np.arange(1000) + 1,
# #     np.array(snmrsig6_hist.history['val_loss'])
# # )
# # plt.plot(
# #     np.arange(1000) + 1,
# #     np.array(snmrsig62_hist.history['val_loss'])
# # )
# plt.plot(
#     np.arange(2000) + 1,
#     np.array(snmrsig63_hist.history['val_loss'])
# )
# plt.plot(
#     np.arange(2000) + 1,
#     np.array(snmrsig64_hist.history['val_loss'])
# )
# plt.plot(
#     np.arange(2000) + 1,
#     np.array(snmrsig65_hist.history['val_loss'])
# )
# plt.yscale('log')
# # plt.xscale('log')
# plt.xlabel("Epoch")
# plt.ylabel("loss")
# plt.show()

# %% 2D plotting of regions

figsize = (8, 8)

# TODO Add plotting part from notebook
nmtst_2D = 1000000
xmtst_2D = np.random.rand(nmtst_2D, 2)*xside

# Multiple network, tanh + squared_hinge

regtst_2Dtanh = []
with tf.device('CPU:0'):
    for j in range(len(lims)):
        regtst_2Dtanh.append(
            np.round((mnsrtanh2[j](xmtst_2D/xside).numpy() + 1)/2.0)
        )

regtst_2Dtanh = np.asarray(regtst_2Dtanh)
nreg_2Dtanh = regtst_2Dtanh.reshape((len(lims), nmtst_2D)).T.sum(axis=1)


plt.figure(figsize=figsize)
plt.title("Multiple networks, single limit, tanh")
for j in range(int(nreg_2Dtanh.max() + 1)):
    plt.scatter(
        xmtst_2D[nreg_2Dtanh == j][:, 0],
        xmtst_2D[nreg_2Dtanh == j][:, 1],
        s=1,
    )
plt.savefig("2Dregs_mn-tanh.png")

# Multiple network, sigmoid + binary_crossentropy

regtst_2Dsig = []
for j in range(len(lims)):
    regtst_2Dsig.append(
        np.round(mnsrsig2[j](xmtst_2D/xside).numpy())
    )

regtst_2Dsig = np.asarray(regtst_2Dsig)
nreg_2Dsig = regtst_2Dsig.reshape((len(lims), nmtst_2D)).T.sum(axis=1)

plt.figure(figsize=figsize)
plt.title("Multiple networks, single limit, sigmoid")
for j in range(int(nreg_2Dsig.max() + 1)):
    plt.scatter(
        xmtst_2D[nreg_2Dsig == j][:, 0],
        xmtst_2D[nreg_2Dsig == j][:, 1],
        s=1,
    )
plt.savefig("2Dregs_mn-sig.png")

# Single network, softmax + categorical_crossentropy

regtst_2Dsnsm = snsm2(xmtst_2D/xside)
nreg_2Dsnsm = np.argmax(regtst_2Dsnsm.numpy(), axis=1)

plt.figure(figsize=figsize)
plt.title("Single network with softmax")
for j in range(int(nreg_2Dsnsm.max() + 1)):
    plt.scatter(
        xmtst_2D[nreg_2Dsnsm == j][:, 0],
        xmtst_2D[nreg_2Dsnsm == j][:, 1],
        s=1,
    )
plt.savefig("2Dregs_sn-softmax.png")

# Single network, multilabel, tanh + squared_hinge

regtst_2Dsntanh = snmrtanh2(xmtst_2D/xside)
nregs_2Dsntanh = np.round((regtst_2Dsntanh + 1)/2).sum(axis=1)

plt.figure(figsize=(8, 8))
plt.title("Single network with multilabel, tanh")
for j in range(int(nregs_2Dsntanh.max()) + 1):
    plt.scatter(
        xmtst_2D[nregs_2Dsntanh == j][:, 0],
        xmtst_2D[nregs_2Dsntanh == j][:, 1],
        s=1,
    )
plt.savefig("2Dregs_sn-mltanh.png")

# Single network, multilabel, sigmoid + binary_crossentropy

regtst_2Dsnsig = snmrsig2(xmtst_2D/xside)
nregs_2Dsnsig = np.round(regtst_2Dsnsig).sum(axis=1)

plt.figure(figsize=(8, 8))
plt.title("Single network with multilabel, sigmoid")
for j in range(int(nregs_2Dsnsig.max()) + 1):
    plt.scatter(
        xmtst_2D[nregs_2Dsnsig == j][:, 0],
        xmtst_2D[nregs_2Dsnsig == j][:, 1],
        s=1,
    )
plt.savefig("2Dregs_sn-mlsig.png")


# Plot error
noverb = 0

# ******************** 2D ********************
# Use more points
nmtst2 = int(1e7)
xmtst2 = np.random.rand(nmtst2, 2)*xside

# softmax
regtst = snsm2.predict(xmtst2/xside, batch_size=batch_size, verbose=noverb)
nregs_sm = np.argmax(regtst, axis=1)
# ==========

# multilabel
# tanh:
reg_mlml = snmrtanh2.predict(xmtst2/xside, batch_size=batch_size, verbose=noverb)
nregs_mlml = np.round((reg_mlml + 1)/2).sum(axis=1).astype(int)

# sigmoid
reg_mlml2 = snmrsig2.predict(xmtst2/xside, batch_size=batch_size, verbose=noverb)
nregs_mlml2 = np.round(reg_mlml2).sum(axis=1).astype(int)
# ==========

# multinetwork, tanh
regtanh = []
for j in range(len(lims)):
    regtanh.append(
        np.round((mnsrtanh2[j].predict(xmtst2/xside, batch_size=batch_size, verbose=noverb) + 1)/2.0)
    )

regtanh = np.asarray(regtanh).reshape((len(lims), nmtst2)).T
nregstanh = regtanh.sum(axis=1)

regsig = []
for j in range(len(lims)):
    regsig.append(
        np.round(mnsrsig2[j].predict(xmtst2/xside, batch_size=batch_size, verbose=noverb))
    )

regsig = np.asarray(regsig).reshape((len(lims), nmtst2)).T
nregssigh = regsig.sum(axis=1)
# ==========

# %%

mnsigerr = []  # Multiple network, sigmoid
mntanherr = []  # Multiple network, tanh
smerr = []
smlerr = []
sml2err = []
for j in range(int(nregs_sm.max() + 1)):
    mnsigerr.append(
        Vtot2*(nregssigh == j).sum()*ffun2(xmtst2[nregssigh == j]).std()/nmtst2
    )
    mntanherr.append(
        Vtot2*(nregstanh == j).sum()*ffun2(xmtst2[nregstanh == j]).std()/nmtst2
    )
    smerr.append(
        Vtot2*(nregs_sm == j).sum()*ffun2(xmtst2[nregs_sm == j]).std()/nmtst2
    )
    smlerr.append(
        Vtot2*(nregs_mlml == j).sum()*ffun2(xmtst2[nregs_mlml == j]).std()/nmtst2
    )
    sml2err.append(
        Vtot2*(nregs_mlml2 == j).sum()*ffun2(xmtst2[nregs_mlml2 == j]).std()/nmtst2
    )

plt.figure()
plt.title("2 dimensions")
plt.plot(
    smerr,
    'rx',
    label="one-hot-encoding",
)
plt.plot(
    mnsigerr,
    'g+',
    label="Multi network, sigmoid",
)
plt.plot(
    mntanherr,
    'b+',
    label="Multi network, tanh",
)
plt.plot(
    smlerr,
    'g.',
    label="Multi label, sigmoid",
)
plt.plot(
    sml2err,
    'b.',
    label="Multi label, tanh",
)
plt.xlabel("Region index")
plt.ylabel(r"$\sigma_j V_j$")
plt.legend()
plt.savefig("2D_error.pdf")

# ******************** 4D ********************
nmtst4 = int(1e7)
xmtst4 = np.random.rand(nmtst4, 4)*xside

# Predict on CPU, avoid filling up GPU ram (useful when testing multiple times)
with tf.device('CPU:0'):
    # softmax
    restst = snsm4.predict(xmtst4/xside, batch_size=int(1e6), verbose=noverb)
    nregs_sm = np.argmax(restst, axis=1)
    # ==========

    # tanh
    restanh = []
    for j in range(len(lims)):
        restanh.append(
            np.round((mnsrtanh4[j].predict(xmtst4/xside, batch_size=int(1e5), verbose=noverb) + 1)/2.0)
        )

    restanh = np.asarray(restanh).reshape((len(lims), nmtst4)).T
    nregstanh = restanh.sum(axis=1)

    ressig = []
    for j in range(len(lims)):
        ressig.append(
            np.round(mnsrsig4[j].predict(xmtst4/xside, batch_size=int(1e5), verbose=noverb))
        )

    ressig = np.asarray(ressig).reshape((len(lims), nmtst4)).T
    nregssigh = ressig.sum(axis=1)
    # ==========

    # mlml
    # tanh: Seems to work bad
    res_mlml = snmrtanh4.predict(xmtst4/xside, batch_size=int(1e5), verbose=noverb)
    nregs_mlml = np.round((res_mlml + 1)/2).sum(axis=1).astype(int)

    # sigmoid
    res_mlml2 = snmrsig4.predict(xmtst4/xside, batch_size=int(1e5), verbose=noverb)
    nregs_mlml2 = np.round(res_mlml2).sum(axis=1).astype(int)
    # ==========

mnsigerr = []  # Multiple network, sigmoid
mntanherr = []  # Multiple network, tanh
smerr = []
smlerr = []
sml2err = []
for j in range(int(nregs_sm.max() + 1)):
    mnsigerr.append(
        Vtot4*(nregssigh == j).sum()*ffun4(xmtst4[nregssigh == j]).std()/nmtst4
    )
    mntanherr.append(
        Vtot4*(nregstanh == j).sum()*ffun4(xmtst4[nregstanh == j]).std()/nmtst4
    )
    smerr.append(
        Vtot4*(nregs_sm == j).sum()*ffun4(xmtst4[nregs_sm == j]).std()/nmtst4
    )
    smlerr.append(
        Vtot4*(nregs_mlml == j).sum()*ffun4(xmtst4[nregs_mlml == j]).std()/nmtst4
    )
    sml2err.append(
        Vtot4*(nregs_mlml2 == j).sum()*ffun4(xmtst4[nregs_mlml2 == j]).std()/nmtst4
    )

plt.figure()
plt.title("4 dimensions")
plt.plot(
    smerr,
    'rx',
    label="one-hot-encoding",
)
plt.plot(
    mnsigerr,
    'g+',
    label="Multi network, sigmoid",
)
plt.plot(
    mntanherr,
    'b+',
    label="Multi network, tanh",
)
plt.plot(
    smlerr,
    'g.',
    label="Multi label, sigmoid",
)
plt.plot(
    sml2err,
    'b.',
    label="Multi label, tanh",
)
plt.xlabel("Region index")
plt.ylabel(r"$\sigma_j V_j$")
plt.legend()
plt.savefig("4D_error.pdf")

# ******************** 6D ********************
nmtst6 = int(1e7)
xmtst6 = np.random.rand(nmtst6, 6)*xside

# Predict on CPU, avoid filling up GPU ram (useful when testing multiple times)
with tf.device('CPU:0'):
    # softmax
    restst = snsm6.predict(xmtst6/xside, batch_size=int(1e5), verbose=0)
    nregs_sm = np.argmax(restst, axis=1)
    # ==========

    # tanh
    restanh = []
    for j in range(len(lims)):
        restanh.append(
            np.round((mnsrtanh6[j].predict(xmtst6/xside, batch_size=int(1e5), verbose=0) + 1)/2.0)
        )

    restanh = np.asarray(restanh).reshape((len(lims), nmtst6)).T
    nregstanh = restanh.sum(axis=1)

    ressig = []
    for j in range(len(lims)):
        ressig.append(
            np.round(mnsrsig6[j].predict(xmtst6/xside, batch_size=int(1e5), verbose=0))
        )

    ressig = np.asarray(ressig).reshape((len(lims), nmtst6)).T
    nregssigh = ressig.sum(axis=1)
    # ==========

    # mlml
    # tanh: Seems to work bad
    res_mlml = snmrtanh6.predict(xmtst6/xside, batch_size=int(1e5), verbose=0)
    nregs_mlml = np.round((res_mlml + 1)/2).sum(axis=1).astype(int)

    # sigmoid
    res_mlml2 = snmrsig6.predict(xmtst6/xside, batch_size=int(1e5), verbose=0)
    nregs_mlml2 = np.round(res_mlml2).sum(axis=1).astype(int)

    res_mlml22 = snmrsig62.predict(xmtst6/xside, batch_size=int(1e5), verbose=0)
    nregs_mlml22 = np.round(res_mlml22).sum(axis=1).astype(int)
    # ==========

# %%

mnsigerr = []  # Multiple network, sigmoid
mntanherr = []  # Multiple network, tanh
smerr = []
smlerr = []
sml2err = []
sml22err = []
for j in range(int(nregs_sm.max() + 1)):
    mnsigerr.append(
        Vtot6*(nregssigh == j).sum()*ffun6(xmtst6[nregssigh == j]).std()/nmtst6
    )
    mntanherr.append(
        Vtot6*(nregstanh == j).sum()*ffun6(xmtst6[nregstanh == j]).std()/nmtst6
    )
    smerr.append(
        Vtot6*(nregs_sm == j).sum()*ffun6(xmtst6[nregs_sm == j]).std()/nmtst6
    )
    smlerr.append(
        Vtot6*(nregs_mlml == j).sum()*ffun6(xmtst6[nregs_mlml == j]).std()/nmtst6
    )
    sml2err.append(
        Vtot6*(nregs_mlml2 == j).sum()*ffun6(xmtst6[nregs_mlml2 == j]).std()/nmtst6
    )
    sml22err.append(
        Vtot6*(nregs_mlml22 == j).sum()*ffun6(xmtst6[nregs_mlml22 == j]).std()/nmtst6
    )

plt.figure()
plt.title("6 dimensions")
plt.plot(
    smerr,
    'rx',
    label="one-hot-encoding",
)
plt.plot(
    mnsigerr,
    'g+',
    label="Multi network, sigmoid",
)
plt.plot(
    mntanherr,
    'b+',
    label="Multi network, tanh",
)
plt.plot(
    smlerr,
    'g.',
    label="Multi label, sigmoid",
)
plt.plot(
    sml2err,
    'b.',
    label="Multi label, tanh",
)
plt.plot(
    sml22err,
    'r.',
    label="Multi label, sigmoid, custom loss",
)
plt.xlabel("Region index")
plt.ylabel(r"$\sigma_j V_j$")
plt.legend()
plt.savefig("6D_error.pdf")

# %% Group testing, make it easier to organize!

def test_networks(
    dim=2,
    xside=10,
    radius=10/4,
    ntest=int(1e8),
    error_target=0.1,
    regs_max=12,
    verbose=1,
    learning_rate = 0.001,
    batch_size = int(1e6)
):
    # function setup
    D = dim
    centers = [[xside/4]*D, [3*xside/4]*D]

    radii = [radius]*len(centers)
    Vtot = xside**D

    xvals = np.random.rand(ntest, D)*xside
    ffun = conegen(D, radii, centers)
    fres = ffun(xvals)

    lims = [0.0]
    print("Limit 1:", lims[-1])
    lims.append(
        functions.get_lim_err(
            fres[fres > lims[-1]], error_target,
            fromlim=lims[-1],
            ntest=1000,
            nreal=ntest,
            vtotal=Vtot,
            tstscale='linear'
        )
    )
    print("Limit 2:", lims[-1])

    for j in range(regs_max - 3):
        xdum, fdum, ndum = sample_gen(20000, ntest, D, xside, lims[-1], ffun)
        xvals = np.append(
            xvals,
            xdum,
            axis=0
        )
        fres = np.append(
            fres,
            fdum,
        )
        lims.append(
            functions.get_lim_err(
                fdum, error_target,
                fromlim=lims[-1],
                ntest=1000,
                nreal=ndum,
                vtotal=Vtot,
                tstscale='linear'
            )
        )
        print("Limit {}:".format(j + 3), lims[-1])
        if lims[-1] == -1:
            break

    lims = lims[:-1]
    nregs = len(lims) + 1

    # Get training set and labels
    xgetind = functions.get_train_xy(xvals, fres, [-1] + lims + [2.5])
    xohetrain = np.empty((0, D))
    yohetrain = np.empty((0, 1))
    fohetrain = np.empty((0))
    for j in range(12):
        xohe_inreg = xgetind[0][xgetind[1].flatten() == j][:1000]
        xohetrain = np.append(
            xohetrain,
            xohe_inreg,
            axis=0
        )
        fohetrain = np.append(
            fohetrain,
            xgetind[2][xgetind[1].flatten() == j][:xohe_inreg.shape[0]]
        )
        yohetrain = np.append(
            yohetrain,
            np.full((xohe_inreg.shape[0], 1), j),
            axis=0
        )

    # %% Multiple networks, predict single level
    activation_out = "sigmoid"
    loss = "binary_crossentropy"
    xtrain = [None]*len(lims)
    ytrain = [None]*len(lims)
    for j in range(len(lims)):
        xtrain_pre, ytrain_pre, _ = functions.get_train_xy(
            xohetrain, fohetrain, [-1, lims[j], 2.5]
        )
        if activation_out == "tanh":
            ytrain_pre = ytrain_pre*2.0 - 1.0
        indum = np.arange(xtrain_pre.shape[0])
        np.random.shuffle(indum)
        xtrain[j] = xtrain_pre[indum]
        ytrain[j] = ytrain_pre[indum]

    mnsrsig = [None]*len(lims)
    for j in range(len(lims)):
        mnsrsig[j] = Sequential()
        mnsrsig[j].add(Dense(16*D, input_shape=(D,), activation='relu'))
        mnsrsig[j].add(Dense(8*D, activation='relu'))
        mnsrsig[j].add(Dense(nregs, activation='relu'))
        mnsrsig[j].add(Dense(1, activation=activation_out))
        mnsrsig[j].compile(
            optimizer=Adam(learning_rate=learning_rate),
            loss=loss
        )
        mnsrsig[j].fit(
            xtrain[j]/xside,
            ytrain[j],
            epochs=4000,
            batch_size=batch_size,
            verbose=verbose
        )

    activation_out = "tanh"
    loss = "squared_hinge"
    xtrain = [None]*len(lims)
    ytrain = [None]*len(lims)
    for j in range(len(lims)):
        xtrain_pre, ytrain_pre, _ = functions.get_train_xy(
            xohetrain, fohetrain, [-1, lims[j], 2.5]
        )
        if activation_out == "tanh":
            ytrain_pre = ytrain_pre*2.0 - 1.0
        indum = np.arange(xtrain_pre.shape[0])
        np.random.shuffle(indum)
        xtrain[j] = xtrain_pre[indum]
        ytrain[j] = ytrain_pre[indum]

    mnsrtanh = [None]*len(lims)
    for j in range(len(lims)):
        mnsrtanh[j] = Sequential()
        mnsrtanh[j].add(Dense(16*D, input_shape=(D,), activation='relu'))
        mnsrtanh[j].add(Dense(8*D, activation='relu'))
        mnsrtanh[j].add(Dense(nregs, activation='relu'))
        mnsrtanh[j].add(Dense(1, activation=activation_out))
        mnsrtanh[j].compile(
            optimizer=Adam(learning_rate=learning_rate),
            loss=loss
        )
        mnsrtanh[j].fit(
            xtrain[j]/xside,
            ytrain[j],
            epochs=2000,
            batch_size=batch_size,
            verbose=verbose
        )

    # %% Single network, softmax

    activation_out = 'softmax'
    losssm = 'categorical_crossentropy'
    snsm = Sequential()
    snsm.add(Dense((nregs - 1)*16*D, input_shape=(D,), activation='relu'))
    snsm.add(Dense((nregs - 1)*8*D, activation='relu'))
    snsm.add(Dense(len(lims) + 1, activation=activation_out))
    snsm.compile(
        optimizer=Adam(learning_rate=learning_rate),
        loss=losssm
    )
    snsm.fit(
        xohetrain/xside,
        to_categorical(yohetrain),
        epochs=6000,
        batch_size=batch_size,
        verbose=verbose
    )

    # %% Single network, multilabel

    activation_out = 'tanh'
    loss = 'squared_hinge'
    snmltanh = Sequential()
    snmltanh.add(Dense((nregs - 1)*16*D, input_shape=(D,), activation='relu'))
    snmltanh.add(Dense((nregs - 1)*8*D, activation='relu'))
    snmltanh.add(Dense(len(lims), activation=activation_out))
    snmltanh.compile(
        optimizer=Adam(learning_rate=learning_rate),
        loss=loss
    )
    snmltanh.fit(
        xohetrain/xside,
        to_multilabel(yohetrain, activation_out=activation_out),
        epochs=6000,
        batch_size=batch_size,
        verbose=verbose
    )

    activation_out = 'sigmoid'
    loss = 'binary_crossentropy'
    # loss = bce_nononsense
    snmlsig = Sequential()
    snmlsig.add(Dense((nregs - 1)*16*D, input_shape=(D,), activation='relu'))
    snmlsig.add(Dense((nregs - 1)*8*D, activation='relu'))
    snmlsig.add(Dense(len(lims), activation=activation_out))
    snmlsig.compile(
        optimizer=Adam(learning_rate=learning_rate),
        loss=loss
    )
    snmlsig.fit(
        xohetrain/xside,
        to_multilabel(yohetrain, activation_out=activation_out),
        epochs=6000,
        batch_size=batch_size,
        verbose=verbose
    )

    return mnsrsig, mnsrtanh, snsm, snmlsig, snmltanh, lims


# %%

xside = 10

mnsig, mntanh, snsm, snmlsig, snmltanh, lims2 =  test_networks(
    dim=2, error_target=0.24, xside=xside, radius=xside/4
)

# %%

# Need to recalculate limits for 4 and 6 D!
mnsig4, mntanh4, snsm4, snmlsig4, snmltanh4, lims4 =  test_networks(
    dim=4, error_target=1.6, xside=xside, radius=xside/4
)

# %%

mnsig6, mntanh6, snsm6, snmlsig6, snmltanh6, lims6 =  test_networks(
    dim=6, error_target=8.6, xside=xside, radius=xside/4
)

# %%
# Integration


def data_transform(xdata):
    return xdata/xside


# Generate functions that use the neural network to predict regions
def reg_pred_gen(model, data_transf=None):
    def reg_pred(xdata=None, batch_size=int(1e6), verbose=0):
        if xdata is None:
            if type(model) is list:
                nregs = len(model) + 1
                ndim = model[0].get_config()['layers'][0]['config']['batch_input_shape'][1]
            else:
                ndim = model.get_config()['layers'][0]['config']['batch_input_shape'][1]
                activation_out = model.get_config()['layers'][-1]['config']['activation']
                if activation_out == 'softmax':
                    nregs = model.get_config()['layers'][-1]['config']['units']
                elif activation_out == 'sigmoid' or activation_out == 'tanh':
                    nregs = model.get_config()['layers'][-1]['config']['units'] + 1
            return ndim, nregs
        if data_transf is not None:
            xdata_tf = data_transf(xdata)
        else:
            xdata_tf = xdata
        if type(model) is list:
            regres_pre = []
            for j in range(len(model)):
                modres_here = model[j].predict(
                    xdata_tf,
                    batch_size=batch_size,
                    verbose=verbose
                )
                activation_out = model[j].get_config()['layers'][-1]['config']['activation']
                if activation_out == 'tanh':
                    regres_pre.append(
                        np.round((modres_here + 1)/2).astype(int)
                    )
                elif activation_out == 'sigmoid':
                    regres_pre.append(np.round(modres_here).astype(int))
            regres = np.asarray(regres_pre).reshape((len(model), xdata_tf.shape[0])).T.sum(axis=1)
        else:
            modres = model.predict(
                xdata_tf,
                batch_size=batch_size,
                verbose=verbose
            )
            activation_out = model.get_config()['layers'][-1]['config']['activation']
            if activation_out == 'tanh':
                regres = np.round((modres + 1)/2).sum(axis=1).astype(int)
            elif activation_out == 'sigmoid':
                regres = np.round(modres).sum(axis=1).astype(int)
            elif activation_out == 'softmax':
                regres = np.argmax(modres, axis=1)
            else:
                raise ValueError(
                    "Unexpected value for activation function:",
                    activation_out
                )
        return regres
    return reg_pred


# %%
# nnfun: a function that returns number of region based on neural network,
# generated with the function reg_pred_gen
# TODO Replace xside by something related to full integration space
def sample_gen_nn(nnfun, npts, ntest, xside, Vtot=1, verbose=0):
    D, nregs = nnfun()
    if type(npts) is int:
        npts_ls = [npts]*nregs
    xpre = np.random.uniform(
        low=0, high=xside, size=(ntest, D)
    )
    regs_pred = nnfun(xpre)
    xaccumul = [np.empty((0, D)) for j in range(nregs)]
    # TODO use np.empty instead
    vols = [None]*nregs
    found = [None]*nregs
    tried = [None]*nregs
    for j in range(nregs):
        xaccumul[j] = xpre[regs_pred == j][:npts_ls[j]]
        vols[j] = Vtot*(regs_pred == j).sum()/ntest
        found[j] = (regs_pred == j).sum()
        tried[j] = ntest
    isdone = [xaccumul[k].shape[0] >= npts for k in range(nregs)]
    # print(isdone)
    while not all(isdone):
        for j in range(nregs):
            if not isdone[j]:
                xpre = np.random.uniform(
                    low=0, high=xside, size=(ntest, D)
                )
                regs_pred = nnfun(xpre)
                xaccumul[j] = np.append(
                    xaccumul[j],
                    xpre[regs_pred == j][:npts_ls[j] - xaccumul[j].shape[0]],
                    axis=0
                )
                found[j] += (regs_pred == j).sum()
                tried[j] += ntest
                vols[j] = Vtot*found[j]/tried[j]
            # print(j, xaccumul[j].shape[0], vols[j], found[j], tried[j])
        isdone = [xaccumul[k].shape[0] >= npts for k in range(nregs)]
        # print(isdone)
    return xaccumul, vols, tried


def sample_integrate(nnfun, ffun, nptsreg, ntest, xside, Vtot=1):
    xacc, vols, tried = sample_gen_nn(nnfun, nptsreg, ntest, xside, Vtot=Vtot)
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
    return int_res, err_res, contribs, varncs, nvals, vols, tried


reg_pred_snsm = reg_pred_gen(snsm, data_transform)

reg_pred_snsm4 = reg_pred_gen(snsm4, data_transform)
reg_pred_snmlsig4 = reg_pred_gen(snmlsig4, data_transform)
reg_pred_snmltanh4 = reg_pred_gen(snmltanh4, data_transform)

reg_pred_snsm6 = reg_pred_gen(snsm6, data_transform)
reg_pred_snmlsig6 = reg_pred_gen(snmlsig6, data_transform)
reg_pred_snmltanh6 = reg_pred_gen(snmltanh6, data_transform)

# %%
xside = 10
radius = xside/4

D = 4
centers = [[radius]*D, [3*radius]*D]
radii = [radius]*len(centers)
Vtot4 = xside**D
ffun4 = conegen(D, radii, centers)

D = 6
centers = [[radius]*D, [3*radius]*D]
radii = [radius]*len(centers)
Vtot6 = xside**D
ffun6 = conegen(D, radii, centers)

itrue = 2*cone_vol(4, radius)
itrue_6 = 2*cone_vol(6, radius)

# %%
nptsreg = 100
ntest = int(1e6)

nattempts = 100
intval = np.empty((nattempts,))
errval = np.empty((nattempts,))
errvalst = np.empty((nattempts,))
# errval2 = np.empty((nattempts,))
cntrbs = np.empty((nattempts, 11))
varncs = np.empty((nattempts, 11))
nvals = np.empty((nattempts, 11))
vols = np.empty((nattempts, 11))
intvalt = np.empty((nattempts,))
errvalt = np.empty((nattempts,))
errvaltst = np.empty((nattempts,))
# errval2 = np.empty((nattempts,))
cntrbst = np.empty((nattempts, 11))
varncst = np.empty((nattempts, 11))
nvalst = np.empty((nattempts, 11))
volst = np.empty((nattempts, 11))
intval_6 = np.empty((nattempts,))
errval_6 = np.empty((nattempts,))
errvalst_6 = np.empty((nattempts,))
# errval2 = np.empty((nattempts,))
cntrbs_6 = np.empty((nattempts, 11))
varncs_6 = np.empty((nattempts, 11))
nvals_6 = np.empty((nattempts, 11))
vols_6 = np.empty((nattempts, 11))
intvalt_6 = np.empty((nattempts,))
errvalt_6 = np.empty((nattempts,))
errvaltst_6 = np.empty((nattempts,))
# errval2 = np.empty((nattempts,))
cntrbst_6 = np.empty((nattempts, 11))
varncst_6 = np.empty((nattempts, 11))
nvalst_6 = np.empty((nattempts, 11))
volst_6 = np.empty((nattempts, 11))
for j in range(nattempts):
    intval[j], errval[j], cntrbs[j], varncs[j], nvals[j], vols[j] = sample_integrate(
        reg_pred_snsm4, ffun4, nptsreg, ntest, xside, Vtot=Vtot4)
    intvalt[j], errvalt[j], cntrbst[j], varncst[j], nvalst[j], volst[j] = sample_integrate(
        reg_pred_snmltanh4, ffun4, nptsreg, ntest, xside, Vtot=Vtot4)
    intval_6[j], errval_6[j], cntrbs_6[j], varncs_6[j], nvals_6[j], vols_6[j] = sample_integrate(
        reg_pred_snsm6, ffun6, nptsreg, ntest, xside, Vtot=Vtot6)
    intvalt_6[j], errvalt_6[j], cntrbst_6[j], varncst_6[j], nvalst_6[j], volst_6[j] = sample_integrate(
        reg_pred_snmltanh6, ffun6, nptsreg, ntest, xside, Vtot=Vtot6)

print("DONE")
# errval.mean() 0.8486
# errvalt.mean() 0.74078
# errval_6.mean() 12.13
# errvalt_6.mean() 5.81

# %%

plt.plot(
    [cntrbs[:, j].var()/cntrbs[:, j].mean() for j in range(cntrbs[0].shape[0])]
)
plt.plot(
    [cntrbst[:, j].var()/cntrbs[:, j].mean() for j in range(cntrbst[0].shape[0])]
)
plt.show()

cntrbst[:, j].std()

# %%

print(
    intval_6.std(),
    np.sqrt(np.sum([cntrbs_6[:, j].std() for j in range(cntrbs_6[0].shape[0])]))
)

# %%
# Calculate the variances on volumes
volsvar = vols.std(axis=0)
volstvar = volst.std(axis=0)
vols_6var = vols_6.std(axis=0)
volst_6var = volst_6.std(axis=0)

plt.plot(
    volsvar**0.5
)
plt.plot(
    volstvar**0.5
)
plt.show()

plt.plot(
    vols_6var**0.5
)
plt.plot(
    volst_6var**0.5
)
plt.show()

# %%
# TEST ERROR ON VOLUME ESTIMATES


varteve = [None]*10
avgnteve = [None]*10
tevetries = 10
for k in range(10):
    ntest = int(np.random.rand()*1e7)
    vteve = [None]*tevetries
    nteve = [None]*tevetries
    for j in range(tevetries):
        _, vteve[j], nteve[j] = sample_gen_nn(
            reg_pred_snsm4, nptsreg, ntest, xside, Vtot=Vtot4, verbose=0)

    vteve_a = np.asarray(vteve)
    nteve_a = np.asarray(nteve)

    varteve[k] = vteve_a.var(axis=0)
    avgnteve[k] = nteve_a.mean(axis=0)

# %%
varteve_a = np.array(varteve)
avgnteve_a = np.array(avgnteve)
var_t_n = varteve_a*avgnteve_a

plt.plot(
    var_t_n.T
)
plt.yscale('log')
plt.show()

# %%

j = 10
theNj = vteve_a[:, j]*nteve_a[:, j]/Vtot4
Vtot4**2*(theNj**2 - theNj.mean()**2).mean()/nteve_a[:, j].mean()

# %%
for j in range(len(cntrbs.T)):
    plt.title(
        "{}, {}, {}".format(
            (cntrbst_6[:, j]/itrue_6).std(),
            sqrt(varncst_6[:, j].mean()),
            sqrt(varncst_6[:, j].mean())/(cntrbst_6[:, j]/itrue_6).std(),
        )
    )
    plt.hist(
        cntrbst_6[:, j]/itrue_6,
        bins=100
    )
    plt.show()

# %%

for j in range(len(cntrbs.T)):
    plt.title(
        "target {}, {}".format(
            cntrbst_6[:, j].std(),
            sqrt(varncst_6[:, j].mean())
        )
    )
    plt.hist(
        varncst[:, j],
        bins=100
    )
    plt.show()

# %%

fvarncs = varncs_6/vols_6**2
fexpctd = cntrbs_6/vols_6

fvarncs2 = fvarncs.mean(axis=0)
fexpctd2 = fexpctd.mean(axis=0)

vvarncs = vols_6.var(axis=0)
vexpctd = vols_6.mean(axis=0)

totvar = vvarncs*fvarncs2 + vvarncs*fexpctd2**2 + fvarncs2*vexpctd**2

plt.plot(
    totvar,
    'kx'
)
plt.plot(
    vvarncs*fexpctd2**2,

)
plt.plot(
    fvarncs2*vexpctd**2
)
# plt.plot(
#     tval1,
#     linestyle='dashed'
# )
plt.plot(
    tval2,
    linestyle='dashed'
)
# plt.ylim(0, 20)
plt.show()

# %%

tval1 = np.empty((len(cntrbs.T)))
tval2 = np.empty((len(cntrbs.T)))
tval3 = np.empty((len(cntrbs.T)))
for j in range(len(cntrbs.T)):
    tval1[j] = varncs_6.mean(axis=0)[j]
    tval2[j] = cntrbs_6.var(axis=0)[j]
    print(
        tval2[j]/tval1[j],
    )
# plt.plot(
#     tval1**0.5/tval2
# )
# plt.plot(
#     tval1/tval2**2
# )
# plt.show()

# plt.plot(
#     tval3
# )
# plt.show()

plt.plot(
    tval2,
    'rx'
)
plt.plot(
    totvar,
    'g+'
)
# plt.plot(
#     varncs.mean(axis=0),
#     'b*'

# )
# plt.ylim(0, 20)
plt.show()

# %%

plt.plot(
    vols_6.mean(axis=0),
    vvarncs
)
plt.xscale('log')
plt.show()

# %%

# print(intval.mean(), intval.std(), errval.mean())
plt.hist(
    intval_6/itrue,
    bins=40,
    histtype='step'
)
plt.hist(
    intvalt_6/itrue,
    bins=40,
    histtype='step'
)
plt.show()

plt.hist(
    errval,
    bins=20,
    histtype='step'
)
plt.show()

# %%

for j in range(cntrbs.shape[1]):
    print(
        cntrbs[:, j].std()**2,
        varncs[:, j].mean()
    )

# %%

plt.plot(
    cntrbs.std(axis=0)
)
plt.plot(
    varncs.mean(axis=0)**0.5
)
plt.show()

plt.plot(
    cntrbs.std(axis=0)/varncs.mean(axis=0)**0.5
)
plt.plot(
    cntrbs_6.std(axis=0)/varncs_6.mean(axis=0)**0.5
)
plt.show()

plt.plot(
    varncs_6.std(axis=0)/varncs.std(axis=0)
)
plt.show()



# %%

print(
    intval,
    intval/itrue,
    errval,
)

# %%

int_vals = []
err_vals = []
intreg_vals = []
varreg_vals = []
int_vals_6 = []
err_vals_6 = []
intreg_vals_6 = []
varreg_vals_6 = []
for k in range(100):
    xaccdum, vols = sample_gen_nn(100, int(1e6), 11, 4, xside, reg_pred_snsm4)
    contribs = [Vtot4*vols[j]*ffun4(xaccdum[j]).mean() for j in range(len(vols))]
    errors = [Vtot4**2*vols[j]**2*ffun4(xaccdum[j]).std()**2/xaccdum[j].shape[0] for j in range(len(vols))]
    int_vals.append(np.sum(contribs))
    err_vals.append(sqrt(np.sum(errors)))
    intreg_vals.append(contribs)
    varreg_vals.append(errors)

    xaccdum_6, vols_6 = sample_gen_nn(100, int(1e6), 11, 6, xside, reg_pred_snsm6)
    contribs_6 = [Vtot6*vols_6[j]*ffun6(xaccdum_6[j]).mean() for j in range(len(vols_6))]
    errors_6 = [Vtot6**2*vols_6[j]**2*ffun6(xaccdum_6[j]).std()**2/xaccdum_6[j].shape[0] for j in range(len(vols_6))]
    int_vals_6.append(np.sum(contribs_6))
    err_vals_6.append(sqrt(np.sum(errors_6)))
    intreg_vals_6.append(contribs_6)
    varreg_vals_6.append(errors_6)

xaccdum2, vols2 = sample_gen_nn(100, int(1e6), 11, 4, xside, reg_pred_snmlsig4)
xaccdum3, vols3 = sample_gen_nn(100, int(1e6), 11, 4, xside, reg_pred_snmlsig4)

xaccdum_6, vols_6 = sample_gen_nn(100, int(1e6), 11, 6, xside, reg_pred_snsm6)
xaccdum2_6, vols2_6 = sample_gen_nn(100, int(1e6), 11, 6, xside, reg_pred_snmlsig6)
xaccdum3_6, vols3_6 = sample_gen_nn(100, int(1e6), 11, 6, xside, reg_pred_snmlsig6)

# %%

plt.hist(
    int_vals,
    bins=40
)
plt.show()


# %%

contribs = [Vtot4*vols[j]*ffun4(xaccdum[j]).mean() for j in range(len(vols))]
contribs2 = [Vtot4*vols2[j]*ffun4(xaccdum2[j]).mean() for j in range(len(vols2))]
contribs3 = [Vtot4*vols3[j]*ffun4(xaccdum3[j]).mean() for j in range(len(vols3))]

errors = [Vtot4**2*vols[j]**2*ffun4(xaccdum[j]).std()**2/xaccdum[j].shape[0] for j in range(len(vols))]
errors2 = [Vtot4**2*vols2[j]**2*ffun4(xaccdum2[j]).std()**2/xaccdum2[j].shape[0] for j in range(len(vols2))]
errors3 = [Vtot4**2*vols3[j]**2*ffun4(xaccdum3[j]).std()**2/xaccdum3[j].shape[0] for j in range(len(vols3))]

# %%

contribs_6 = [Vtot6*vols_6[j]*ffun6(xaccdum_6[j]).mean() for j in range(len(vols_6))]
contribs2_6 = [Vtot6*vols2_6[j]*ffun6(xaccdum2_6[j]).mean() for j in range(len(vols2_6))]
contribs3_6 = [Vtot6*vols3_6[j]*ffun6(xaccdum3_6[j]).mean() for j in range(len(vols3_6))]

errors_6 = [Vtot6**2*vols_6[j]**2*ffun6(xaccdum_6[j]).std()**2/xaccdum_6[j].shape[0] for j in range(len(vols_6))]
errors2_6 = [Vtot6**2*vols2_6[j]**2*ffun6(xaccdum2_6[j]).std()**2/xaccdum2_6[j].shape[0] for j in range(len(vols2_6))]
errors3_6 = [Vtot6**2*vols3_6[j]**2*ffun6(xaccdum3_6[j]).std()**2/xaccdum3_6[j].shape[0] for j in range(len(vols3_6))]

# %%

plt.scatter(
    xaccdum3[1][:, 0],
    xaccdum3[1][:, 1],
    s=1
)
plt.show()

# %%

contribs.append(np.sum(contribs))
contribs2.append(np.sum(contribs2))
contribs3.append(np.sum(contribs3))

errors.append(np.sum(errors))
errors2.append(np.sum(errors2))
errors3.append(np.sum(errors3))

contribs_6.append(np.sum(contribs_6))
contribs2_6.append(np.sum(contribs2_6))
contribs3_6.append(np.sum(contribs3_6))

errors_6.append(np.sum(errors_6))
errors2_6.append(np.sum(errors2_6))
errors3_6.append(np.sum(errors3_6))

itrue = 2*cone_vol(4, radius)
itrue_6 = 2*cone_vol(6, radius)

# %%
plt.figure()
plt.title('4D: Contribution to final integral value')
plt.plot(
    contribs,
    'rx',
    label="softmax"
)
plt.plot(
    contribs2,
    'g.',
    label="multilabel, sigmoid"
)
plt.plot(
    contribs3,
    'b+',
    label="multilabel, tanh"
)
plt.plot(
    [12],
    itrue,
    'k+',
    label="multilabel, tanh"
)
plt.legend()
plt.show()

plt.figure()
plt.title('6D: Contribution to final integral value')
plt.plot(
    contribs_6,
    'rx',
    label="softmax"
)
plt.plot(
    contribs2_6,
    'g.',
    label="multilabel, sigmoid"
)
plt.plot(
    contribs3_6,
    'b+',
    label="multilabel, tanh"
)
plt.plot(
    [12],
    itrue_6,
    'k+',
    label="multilabel, tanh"
)
plt.legend()
plt.show()

# %%

plt.figure()
plt.title('4D: Contribution to variance')
plt.plot(
    errors,
    'rx',
    label="softmax"
)
plt.plot(
    errors2,
    'g.',
    label="multilabel, sigmoid"
)
plt.plot(
    errors3,
    'b+',
    label="multilabel, tanh"
)
# plt.plot(
#     [12],
#     itrue,
#     'k+',
#     label="multilabel, tanh"
# )
plt.legend()
plt.show()

plt.figure()
plt.title('6D: Contribution to variance')
plt.plot(
    errors_6,
    'rx',
    label="softmax"
)
plt.plot(
    errors2_6,
    'g.',
    label="multilabel, sigmoid"
)
plt.plot(
    errors3_6,
    'b+',
    label="multilabel, tanh"
)
# plt.plot(
#     [12],
#     itrue,
#     'k+',
#     label="multilabel, tanh"
# )
plt.legend()
plt.show()
