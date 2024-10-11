import numpy as np
# import matplotlib.pyplot as plt

import tensorflow as tf

from functions import gaussian_gen, get_lims_backforth, merge_lim
from functions_tf import x_gen, model_fit, sample_gen_nn2, sample_gen_nn3
from loss_experiment import loss_setup

# Notes: I use a phase space generator that outputs the upper and lower
# limits of the space if no number of points is requested

# %%
ndim = 7
side = 10


def data_transform(x):
    return tf.convert_to_tensor(x/side/2)


# mu1_rand = 2*np.random.rand(ndim)
# Seven random values used for figure in paper
mu1_rand = np.array([
    1.42872452, 0.83847923, 1.58055441,
    0.30422123, 1.7291153, 1.27053504,
    1.54311618
])
mu1_rand = mu1_rand[:ndim]
sig1 = [0.3]*ndim
# mu2_rand = -2*np.random.rand(ndim)
# Seven random values used for figure in paper
mu2_rand = np.array([
    -1.99715329, -1.46426389, -1.77558595,
    -0.76466156, -1.09619859, -0.62166405,
    -1.69391713
])
mu2_rand = mu2_rand[:ndim]
sig2 = [0.3]*ndim

g1 = gaussian_gen(ndim, mu1_rand, sig1)
g2 = gaussian_gen(ndim, mu2_rand, sig2)
bg = gaussian_gen(ndim, [0.0]*ndim, [side/10]*ndim)
my_x_gen = x_gen(ndim, -side/2, side/2)


# The big cancellation function
def my_cancel(x):
    if not isinstance(x, np.ndarray):
        x = x.numpy()
    return 100*g1(x) - 100*g2(x) + 0.1*bg(x)


# %%


n0 = int(2e5)
x0 = my_x_gen(n0)
res0 = my_cancel(x0)

lims = get_lims_backforth(res0, 3)

loss = loss_setup(loss='squared_hinge')
activation_out = 'tanh'
model_restart = True

xtrain_size_reg = 5000
epochs_part = 3000
nptsreg = int(2e5)

# %%

runs = 12
maxregs = 30
testfun = [None]*runs
testmdl = [None]*runs
limits = [lims] + [None]*runs
limits_mrg = [lims] + [None]*runs
xpool = [x0] + [None]*runs
fpool = [res0] + [None]*runs

# TODO Wrap the following loop into a more convenient function
print(
    "Run 0:",
    "Will use {} regions with limits".format(len(limits[0]) - 1),
    limits[0]
)
for j in range(runs):
    testfun[j], testmdl[j] = model_fit(
        # xpool[j], fpool[j], my_cancel, limits[j], xtrain_size,
        np.concatenate(xpool[:j + 1], axis=0),
        np.concatenate(fpool[:j + 1]),
        my_cancel, limits[j], xtrain_size_reg*(len(limits[j]) - 1),
        activation_out, loss,
        epochs_part=epochs_part, ntrains=4, data_transform=data_transform,
        model_restart=model_restart, xgenerator=my_x_gen,
        sample_seed=(
            np.concatenate(xpool[:j + 1], axis=0),
            np.concatenate(fpool[:j + 1]),
            None
        ),
        maxretries=5
    )

    if len(limits) - 1 >= maxregs:
        break

    # Use the accumulated sample to help refine sampling
    preseed0 = np.concatenate(xpool[:j + 1], axis=0)
    preseed1 = np.concatenate(fpool[:j + 1])
    preseed2, _ = testfun[j](preseed0)

    thisseed = (preseed0, preseed1, preseed2)

    # Create a sample of points using the neural network
    morepts = sample_gen_nn2(
        testfun[j], my_x_gen, nptsreg, int(1e7),
        batch_size=int(1e6), maxiter=1000,
        sample_seed=thisseed,
        maxretries=5
    )

    morefvals = [my_cancel(morepts[0][n]) for n in range(len(morepts[0]))]

    # Averages in regions
    means = [
        morefvals[n].mean()
        for n in range(len(morepts[0]))
    ]

    # Deviation in regions
    sigs = [
        morefvals[n].std()
        for n in range(len(morepts[0]))
    ]

    thesum = (np.array(morepts[1])*np.array(means)).sum()
    theerr = ((np.array(morepts[1])**2)*(np.array(sigs))**2/nptsreg).sum()**0.5

    devs_corr = []
    for k in range(len(limits[j]) - 1):
        fltr = (
            morefvals[k] > limits[j][k]
        )*(
            morefvals[k] < limits[j][k + 1]
        )
        devs_corr.append(
            morefvals[k][fltr].std()*morepts[1][k]
        )
    devs_corr = np.array(devs_corr)

    print("means", means)
    print("sigs", sigs)
    print("[1msum and total error[0m", thesum, theerr)
    print(
        "sigma(f)*Vol",
        np.array(morepts[1])*np.array(sigs)
    )
    print("Corrected sigma(f)*Vol", devs_corr)

    l10rdevs = np.log10(devs_corr)
    l10rdevs_mean = l10rdevs.mean()

    # WHAT TO MERGE
    # TODO Probably should merge only if variance is several orders of
    # magnitude smaller, like more than 10^5 smaller
    # reg_merge = l10rdevs < l10rdevs_mean - 3
    # reg_merge = l10rdevs < l10rdevs_mean - 2
    # TODO This number (0.5) needs to be smarter, maybe depend on average
    reg_merge = devs_corr**2 < 0.1

    if reg_merge.sum() > 0:
        limits_mrg[j], devs_corr_cp, indmrg = merge_lim(limits[j], devs_corr)
        print("Merged region {} to {}".format(*indmrg))
    else:
        limits_mrg[j] = list(limits[j])
        devs_corr_cp = list(devs_corr)
        indmrg = (np.inf, np.inf)
    # IF THERE WAS ANYTHING TO MERGE, IT WAS MERGED

    l10rdevs_cp = np.log10(devs_corr_cp)
    l10rdevs_cp_mean = l10rdevs_cp.mean()
    l10rdevs_cp_std = l10rdevs_cp.std()

    rdc_cp_mean = np.mean(devs_corr_cp)
    rdc_cp_std = np.std(devs_corr_cp)

    logmargin = 0.5
    # TODO This number (10) needs to be smarter, maybe depend on average
    reg_rediv = devs_corr**2 > 10
    reg_rediv_old = devs_corr**2 > 10

    if reg_rediv.sum() == 0 and reg_merge.sum() == 0:
        print("No more regions to merge or divide. Stopping at run number", j)
        break

    # HIGH VARIANCE*VOL**2 REGIONS DIVIDED HERE
    new_limits = list(limits_mrg[j])
    kshift = 0
    for k in range(len(limits_mrg[j]) - 1):
        if k >= indmrg[0]:
            krediv = k + 1
        else:
            krediv = k
        if reg_rediv[krediv]:
            print("Redivide region", krediv)
            fltr_corr = (
                (morefvals[krediv] > limits[j][krediv])
                * (morefvals[krediv] <= limits[j][krediv + 1])
            )
            lims_rediv = get_lims_backforth(morefvals[krediv][fltr_corr], 3)
            new_limits = (
                new_limits[:k + kshift + 1]
                + [lims_rediv[1]]
                + new_limits[k + kshift + 1:]
            )
            kshift += 1
    limits[j + 1] = new_limits
    # REGIONS HAVE BEEN DIVIDED

    # Boost samples in redivided regions:
    if reg_rediv.sum() > 0:
        print(
            "Adding samples for regions:",
            list(np.arange(len(morepts[0]))[reg_rediv_old])
        )

        nptsreg_rediv = nptsreg*reg_rediv_old
        morepts_rediv0 = sample_gen_nn2(
            testfun[j], my_x_gen, nptsreg_rediv, int(1e7),
            batch_size=int(1e6), maxiter=1000,
            sample_seed=thisseed,
            maxretries=5
        )
        # morepts_rediv0 = sample_gen_nn3(
        #     testfun[j], my_x_gen, nptsreg_rediv, int(1e7),
        #     batch_size=int(1e6), maxiter=1000,
        #     maxretries=5
        # )
        morepts_rediv = [
            morepts_rediv0[0][n]
            for n in list(np.arange(len(morepts_rediv0[0]))[reg_rediv_old])
        ]
        morefvals_rediv = [
            my_cancel(morepts_rediv[n])
            for n in range(len(morepts_rediv))
        ]
    else:
        morepts_rediv = []
        morefvals = []

    print(
        "Run {}:".format(j + 1),
        "Will use {} regions with limits".format(len(limits[j + 1]) - 1),
        limits[j + 1]
    )

    xpool[j + 1] = np.concatenate(morepts[0] + morepts_rediv, axis=0)
    fpool[j + 1] = np.concatenate(morefvals + morefvals_rediv)

    # TODO Check that this `if` is not necessary when this works properly
    if xpool[j + 1].max() > side/2 or xpool[j + 1].min() < -side/2:
        print("[1mPOINTS OUTSIDE OF RANGE GENERATED![0m")
        break


# %%
# Select a model for testing
mdlindx = 7
mynnfun = testfun[mdlindx]
nregs = mynnfun()[1]

samgentst_tst = sample_gen_nn3(
    # testfun[11], my_x_gen, ntgt, int(1e7),
    mynnfun, my_x_gen, int(2e5), int(1e7),
    batch_size=int(1e6), verbose=1,
    maxretries=1000, Vrerror_target=None
)
fests_tst = np.empty((len(samgentst_tst[0])))
vests_tst = np.empty((len(samgentst_tst[0])))
fvars_tst = np.empty((len(samgentst_tst[0])))
fvars_pre_tst = np.empty((len(samgentst_tst[0])))
vvars_tst = np.empty((len(samgentst_tst[0])))
for k in range(len(samgentst_tst[0])):
    fests_tst[k] = my_cancel(samgentst_tst[0][k]).mean()
    vests_tst[k] = samgentst_tst[1][k]
    fvars_pre_tst[k] = my_cancel(samgentst_tst[0][k]).var()
    fvars_tst[k] = fvars_pre_tst[k]/samgentst_tst[0][k].shape[0]
    vvars_tst[k] = samgentst_tst[2][k]**2

# FOR WHEN I GET THEM FROM TEST SET (_tst suffix)
fvar2_vol2 = np.array(fvars_pre_tst)*vests_tst**2
# FOR WHEN I GET THEM FROM TRAINING
# fvar2_vol2 = (np.array(morepts[1])*np.array(sigs))**2
epstgt = 5e-2
# TODO Try also to share the error between parts
# epstgt = 2.5e-2
ntgt = ((nregs/(epstgt**2))*fvar2_vol2).astype(int)
ntgt[ntgt < 1000] = 1000

# FOR WHEN I GET THEM FROM TEST SET (_tst suffix)
Vrertgt = (0.1*np.array(fvars_pre_tst)/ntgt/np.array(fests_tst)**2)**0.5
# FOR WHEN I GET THEM FROM TRAINING
# Vrertgt = (0.1*np.array(sigs)**2/ntgt/np.array(means)**2)**0.5


inttest = []
err1test = []
err2test = []
errfulltest = []

ntests = 20
fests_int = np.empty((ntests, nregs))
vests_int = np.empty((ntests, nregs))
fvars_int = np.empty((ntests, nregs))
vvars_int = np.empty((ntests, nregs))

# Perform integration using the last training for ntests times
for itest in range(ntests):
    samgentst_int = sample_gen_nn3(
        # testfun[11], my_x_gen, ntgt, int(1e7),
        mynnfun, my_x_gen, ntgt, int(1e7),
        batch_size=int(1e6), verbose=1,
        maxretries=1000, Vrerror_target=Vrertgt
    )

    # fests_int = np.empty((len(samgentst_int[0])))
    # vests_int = np.empty((len(samgentst_int[0])))
    # fvars_int = np.empty((len(samgentst_int[0])))
    fvars_pre_int = np.empty((nregs))
    # vvars_int = np.empty((len(samgentst_int[0])))
    for k in range(len(samgentst_int[0])):
        reshere = my_cancel(samgentst_int[0][k])
        fests_int[j, k] = reshere.mean()
        vests_int[j, k] = samgentst_int[1][k]
        fvars_pre_int[k] = reshere.var()
        fvars_int[j, k] = fvars_pre_int[k]/samgentst_int[0][k].shape[0]
        vvars_int[j, k] = samgentst_int[2][k]**2
        # print(
        #     samgentst_int[1][k]*my_cancel(samgentst_int[0][k]).mean(),
        #     samgentst_int[1][k]**2*my_cancel(samgentst_int[0][k]).var()/samgentst_int[0][k].shape[0],
        #     samgentst_int[1][k]**2*,
        #     my_cancel(samgentst_int[0][k]).var()/my_cancel(samgentst_int[0][k]).mean()**2/samgentst_int[0][k].shape[0],
        # )

    print(
        np.sum(fests_int[j]*vests_int[j]),
        (vests_int[j]**2*fvars_int[j]).sum()**0.5,
        (vvars_int[j]*fests_int[j]**2).sum()**0.5,
        (
            vests_int[j]**2*fvars_int[j]
            + vvars_int[j]*fests_int[j]**2
        ).sum()**0.5,
    )
    inttest.append(np.sum(fests_int[j]*vests_int[j]))
    err1test.append((vests_int[j]**2*fvars_int[j]).sum()**0.5)
    err2test.append((vvars_int[j]*fests_int[j]**2).sum()**0.5)
    errfulltest.append(
        (
            vests_int[j]**2*fvars_int[j]
            + vvars_int[j]*fests_int[j]**2
        ).sum()**0.5
    )

np.savetxt("inttest_SH.csv", inttest)
np.savetxt("err1test_SH.csv", err1test)
np.savetxt("err2test_SH.csv", err2test)
np.savetxt("errfulltest_SH.csv", errfulltest)
np.savetxt("numbers_SH.csv", [
    nregs,
    np.concatenate(xpool[:mdlindx]).shape[0],
    np.concatenate(samgentst_int[0]).shape[0],

])
np.savetxt("sample_SH.csv", np.concatenate(samgentst_int[0]))
# TODO DATA SAVING FOR THESE DOES NOT WORK BUT I DO NOT KNOW WHY, TEST
np.savetxt("fests_int_SH.csv", fests_int)
np.savetxt("vests_int_SH.csv", vests_int)
np.savetxt("fvars_int_SH.csv", fvars_int)
np.savetxt("vvars_int_SH.csv", vvars_int)
