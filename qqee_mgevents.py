import uproot as ur
import numpy as np
from functions import lbins

hlbins = lbins(20, 600, 50)

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

mgh, mgb = np.histogram(meemg, bins=hlbins)
