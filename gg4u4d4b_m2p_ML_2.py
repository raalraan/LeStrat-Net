import numpy as np
import TGPS_m2p

from functions import get_lims_backforth, merge_lim
from functions_tf import model_fit, sample_gen_nn2
from loss_experiment import loss_setup

# %%

ENERGY = TGPS_m2p.ENERGY

# Actual independent indices in 4-momentum input to madgraph
indindx = np.delete(list(range(8, 13*4)), range(0, 11*4, 4))
# Add quark energy
indindx = np.append([0, 4], indindx)

p23 = TGPS_m2p.p23

# %% Relevant definitions


# Keep only independent components of 4-momentum input
def inputtrans(x):
    nx = x.shape[0]
    newx = x.reshape(nx, 14*4)
    xind = 2*newx[:, indindx]/ENERGY
    return xind


def inputtrans_w(x, w):
    nx = x.shape[0]
    newx = x.reshape(nx, 14*4)
    xind = 2*newx[:, indindx]/ENERGY
    return np.append(xind, w.reshape(-1, 1), axis=1)


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


# My other code expects a phase space generator that takes:
#   1st argument: number of points to generate
#   2nd argument: boundary at minimum for all parameters
#   3rd argument: boundary at maximum for all parameters
# Also, if no argument is given, it should output the default boundaries
def psg_wrap(npts=None, energy_min=None, energy_max=None):
    if npts is None and energy_min is None and energy_max is None:
        return [-1]*36, [1]*35 + [1e6]

    if energy_max is None:
        energy0 = ENERGY/2
        energy1 = ENERGY/2
    else:
        if type(energy_max) is list or type(energy_max) is np.ndarray \
                and len(energy_max) > 1:
            energy0 = energy_max[0]*ENERGY/2
            energy1 = energy_max[1]*ENERGY/2
        else:
            energy0 = energy_max*ENERGY/2
            energy1 = energy_max*ENERGY/2

    fmmnta, weights, ncut = TGPS_m2p.gg4u4d4b_gen_ph_spc_fast(
        energy=[energy0, energy1],
        npts=npts
    )

    # return inputtrans_w(fmmnta, weights), fmmnta, weights, ncut
    return inputtrans_w(fmmnta, weights)


def get_weights_wrap(x):
    fmmnta, weights = data_detransform(x)
    weights[np.isnan(weights)] = 0.0
    return get_weights(fmmnta, weights)


# This should return coordinates suitable for training the network
# At the moment it does nothing but is used to prevent error when passing None
# as function to transform data
# TODO Consider in functions_tf.py that data_transform may no be needed
def data_transform(x):
    return x


def data_detransform(x):
    ui1, ui2 = np.unravel_index(indindx, [14, 4])
    xps = x[:, :-1]*ENERGY/2
    wps = x[:, -1]

    xdt = np.zeros([x.shape[0], 14, 4])
    for k in range(len(ui1)):
        xdt[:, ui1[k], ui2[k]] = xps[:, k]
        xdt[:, 0, 3] = xdt[:, 0, 0]
        xdt[:, 1, 3] = -xdt[:, 1, 0]
        xdt[:, -1, 1:-1] = -xdt[:, 2:-1, 1:-1].sum(axis=1)
        xdt[:, -1, -1] = -xdt[:, 2:-1, -1].sum(axis=1) \
            + xdt[:, :2, -1].sum(axis=1)
        xdt[:, 2:, 0] = np.sqrt((xdt[:, 2:, 1:]**2).sum(axis=2))

    return xdt, wps


# %%

n0 = int(2e5)
fwn = psg_wrap(n0)

x0 = data_transform(fwn)
res0 = get_weights_wrap(fwn)

lims = get_lims_backforth(res0, 3)

loss = loss_setup(loss='squared_hinge')
activation_out = 'tanh'
model_restart = True

xtrain_size_reg = 10000
epochs_part = 3000
nptsreg = int(2e5)

# %%

runs = 3
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
        np.concatenate(xpool[:j + 1], axis=0),
        np.concatenate(fpool[:j + 1]),
        get_weights, limits[j], xtrain_size_reg*(len(limits[j]) - 1),
        activation_out, loss,
        epochs_part=epochs_part, ntrains=4, data_transform=data_transform,
        model_restart=model_restart, xgenerator=psg_wrap,
        sample_seed=(
            np.concatenate(xpool[:j + 1], axis=0),
            np.concatenate(fpool[:j + 1]),
            None
        ),
        maxretries=5,
        verbose=0
    )

    if len(limits) - 1 >= maxregs:
        break

    # Use the accumulated sample to help refine sampling
    preseed0 = np.concatenate(xpool[:j + 1], axis=0)
    preseed1 = np.concatenate(fpool[:j + 1])
    preseed2, _ = testfun[j](preseed0)

    thisseed = (preseed0, preseed1, preseed2)

    print("Training finished. Creating a sample using trained network...")
    # Create a sample of points using the neural network
    morepts = sample_gen_nn2(
        testfun[j], psg_wrap, nptsreg, int(1e7),
        batch_size=int(1e6), maxiter=1000,
        sample_seed=thisseed,
        maxretries=5
    )
    print("Sample created!")
    print("Calculating weights and estimates using sample")

    morefvals = [
        get_weights_wrap(morepts[0][n]) for n in range(len(morepts[0]))
    ]

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
    theerr = (
        (np.array(morepts[1])**2)*(np.array(sigs))**2/nptsreg
    ).sum()**0.5

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
    reg_merge = l10rdevs < l10rdevs_mean - 3
    # TODO This number (0.5) needs to be smarter, maybe depend on average
    # reg_merge = devs_corr**2 < 0.1

    # Do not merge 2 regions into one
    if len(l10rdevs) == 2:
        reg_merge[reg_merge] = False

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
    # reg_rediv = devs_corr**2 > 10
    # reg_rediv_old = devs_corr**2 > 10
    reg_rediv = l10rdevs_cp > l10rdevs_cp_mean + 0.5
    reg_rediv_old = l10rdevs_cp > l10rdevs_cp_mean

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
    # TODO This should only be needed if the number of points per region is
    # not enough to complete the requested number of points per region for
    # training when considering the new division
    if reg_rediv.sum() > 0:
        print(
            "Adding samples for regions:",
            list(np.arange(len(morepts[0]))[reg_rediv_old])
        )

        nptsreg_rediv = nptsreg*reg_rediv_old
        morepts_rediv0 = sample_gen_nn2(
            testfun[j], psg_wrap, nptsreg_rediv, int(1e7),
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
            get_weights_wrap(morepts_rediv[n])
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
    if xpool[j + 1].max() > 1.0 or xpool[j + 1].min() < -1.0:
        print("[1mPOINTS OUTSIDE OF RANGE GENERATED![0m")
        break

# Save data but remember that it needs to be detransformed
np.savetxt("gg4u4d4b_xpool.csv", np.concatenate(xpool, axis=0))
np.savetxt("gg4u4d4b_fpool.csv", np.concatenate(fpool))
