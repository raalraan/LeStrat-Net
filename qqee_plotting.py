import numpy as np
from functions import lbins
import uproot as ur


# TODO Import from qqee_m2p_ML but you have to remove main calculation from
# there
def mee_invariant(momenta):
    fm_sum = momenta[:, 2] + momenta[:, 3]
    mee_inv = np.sqrt(
        fm_sum[:, 0]**2 - (fm_sum[:, 1]**2 + fm_sum[:, 2]**2 + fm_sum[:, 3]**2)
    )
    return mee_inv


def ee_angle(momenta):
    pep = momenta[:, 2, 1:4]
    pem = momenta[:, 3, 1:4]
    pdot = (pep*pem).sum(axis=1)
    p1p2abs = np.sqrt(
        (pep**2).sum(axis=1)*(pem**2).sum(axis=1)
    )
    costh = pdot/p1p2abs
    costh1 = pep[:, 2]/np.sqrt((pep**2).sum(axis=1))
    costh2 = pem[:, 2]/np.sqrt((pem**2).sum(axis=1))
    return costh, costh1, costh2


def momntm_boost(momenta, boostvelvec):
    vel = np.sqrt(
        boostvelvec[:, 0]**2 + boostvelvec[:, 1]**2 + boostvelvec[:, 2]**2
    )
    # beta = vel
    velx = boostvelvec[:, 0]
    vely = boostvelvec[:, 1]
    velz = boostvelvec[:, 2]
    uvx = velx/vel
    uvy = vely/vel
    uvz = velz/vel
    # speed of light set to one
    gamma = 1/np.sqrt(1 - vel**2)
    gamma_b = gamma - 1
    theboost = np.array([
        [gamma, -gamma*velx, -gamma*vely, -gamma*velz],
        [-gamma*velx, 1 + gamma_b*uvx**2, gamma_b*uvx*uvy, gamma_b*uvx*uvz],
        [-gamma*vely, gamma_b*uvy*uvx, 1 + gamma_b*uvy**2, gamma_b*uvy*uvz],
        [-gamma*velz, gamma_b*uvz*uvx, gamma_b*uvy*uvz, 1 + gamma_b*uvz**2]
    ]).T
    el1 = (theboost[:, 0]*momenta).sum(axis=1)
    el2 = (theboost[:, 1]*momenta).sum(axis=1)
    el3 = (theboost[:, 2]*momenta).sum(axis=1)
    el4 = (theboost[:, 3]*momenta).sum(axis=1)
    return np.array([el1, el2, el3, el4]).T


preevs = np.loadtxt("mlevents.csv")
events = preevs.reshape((preevs.shape[0], 4, 4))

mee = mee_invariant(events)
cth, cth1, cth2 = ee_angle(events)

# %%

beam = events[:, 0] + events[:, 1]
vbeam = beam[:, 1:]/beam[:, 0].reshape(beam.shape[0], 1)

pe1com = momntm_boost(events[:, 2], vbeam)
pe2com = momntm_boost(events[:, 3], vbeam)

cth1com = pe1com[:, 3]/np.sqrt(pe1com[:, 2]**2 + pe1com[:, 1]**2)
cth2com = pe2com[:, 3]/np.sqrt(pe2com[:, 2]**2 + pe2com[:, 1]**2)

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
Pt1 = np.sqrt(Pxe1**2 + Pye1**2)
Pt2 = np.sqrt(Pxe2**2 + Pye2**2)

meemg = np.sqrt((Ee1 + Ee2)**2 - P2ee)

cthmg = (Pxe1*Pxe2 + Pye1*Pye2 + Pze1*Pze2)/(Ee1*Ee2)
cth1mg = Pze1/Ee1
cth2mg = Pze2/Ee2

fltrPt20 = (Pt1 > 40)*(Pt2 > 40)
facPt20 = fltrPt20.shape[0]/fltrPt20.sum()

mgbeam = np.array([Eu1 + Eu2, Pxu1 + Pxu2, Pyu1 + Pyu2, Pzu1 + Pzu2]).T
vmgbeam = mgbeam[:, 1:]/mgbeam[:, 0].reshape(mgbeam.shape[0], 1)

pe1com_mg = momntm_boost(np.array([Ee1, Pxe1, Pye1, Pze1]).T, vmgbeam)
pe2com_mg = momntm_boost(np.array([Ee2, Pxe2, Pye2, Pze2]).T, vmgbeam)

cth1com_mg = pe1com_mg[:, 3]/np.sqrt(pe1com_mg[:, 2]**2 + pe1com_mg[:, 1]**2)
cth2com_mg = pe2com_mg[:, 3]/np.sqrt(pe2com_mg[:, 2]**2 + pe2com_mg[:, 1]**2)

# %%

hlbins = lbins(20, 1200, 71)

fig, (hist, error) = plt.subplots(
    2,
    height_ratios=[3./4., 1./4.],
    figsize=(5, 5*4/3)
)
fig.suptitle(r'$u\bar{u} \to e^+ e^-$ $10^5$ events')
mlh, mlbc, _ = hist.hist(
    mee,
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
# thh, thbc, _ = hist.hist(
#     mee2_f,
#     bins=hlbins,
#     histtype='step',
#     weights=weights2_f*nevents/weights2_f.sum(),
#     label="Theory"
# )
hist.set_ylabel('Number of events')
hist.legend()
hist.set_xlim(20, 600)
hist.set_ylim(9e-1, 1e5)
hist.set_xscale('log')
hist.set_yscale('log')
# plt.show()

plt.figure(figsize=(5, 2))
error.plot(
    0.5*(hlbins[:-1] + hlbins[1:]),
    np.abs((mgh - mlh)/mgh),
    label="Madgraph - this work"
)
# error.plot(
#     0.5*(hlbins[:-1] + hlbins[1:]),
#     np.abs((thh - mlh)/thh),
#     label="Theory - this work"
# )
# error.plot(
#     0.5*(hlbins[:-1] + hlbins[1:]),
#     np.abs((thh - mgh)/thh),
#     label="Theory - Madgraph"
# )
error.legend(loc='upper center', ncol=2)
error.set_xlabel('$m_{ee}$ [GeV]')
error.set_ylabel('error')
error.set_xlim(20, 600)
error.set_ylim(8e-4, 10)
error.set_xscale('log')
error.set_yscale('log')
# plt.xscale('log')
# plt.yscale('log')

# plt.savefig("events.pdf")

# %%

cthbins = np.linspace(-1, 1, 51)

plt.hist(
    cth, bins=cthbins, histtype='step', density=True,
    label='ML: $P_t < 10$',
)
plt.hist(
    cthmg, bins=cthbins, histtype='step', density=True,
    label='MG: $P_t < 10$',
)
plt.hist(
    cthmg, bins=cthbins, histtype='step', density=True,
    label='MG: $P_t < 20$',
)
plt.xlabel(r'$\cos\theta$')
plt.legend()
plt.show()

plt.hist(
    cth1, bins=cthbins, density=True, histtype='step',
    label='ML: $P_t < 10$',
)
plt.hist(
    cth1mg, bins=cthbins, density=True, histtype='step',
    label='MG: $P_t < 10$',
)
plt.hist(
    cth1mg[fltrPt20], bins=cthbins, density=True, histtype='step',
    label='MG: $P_t < 20$',
)
plt.xlabel(r'$\cos\theta_1$')
plt.legend()
plt.show()

plt.hist(
    cth2, bins=cthbins, histtype='step', density=True,
    label='ML: $P_t < 10$',
)
plt.hist(
    cth2mg, bins=cthbins, histtype='step', density=True,
    label='MG: $P_t < 10$',
)
plt.hist(
    cth2mg[fltrPt20], bins=cthbins, histtype='step', density=True,
    label='MG: $P_t < 20$',
)
plt.xlabel(r'$\cos\theta_2$')
plt.legend()
plt.show()

# %%

plt.hist(
    cth1com,
    bins=50,
    histtype='step',
    density=True
)
plt.hist(
    cth1com,
    bins=50,
    histtype='step',
    density=True
)
plt.hist(
    cth1com_mg,
    bins=50,
    histtype='step',
    density=True
)
plt.hist(
    cth1com_mg,
    bins=50,
    histtype='step',
    density=True
)
plt.hist(
    cth1com_mg[fltrPt20],
    bins=50,
    histtype='step',
    density=True
)

plt.hist(
    cth1com_mg,
    bins=50,
    histtype='step',
    density=True
)
plt.show()
