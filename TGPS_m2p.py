import ROOT
from array import array
import numpy as np
import matrix2py
import lhapdf
import subprocess as sp

ENERGY = 13e3

# Masses form Cards/param_card.dat
MB = 4.7  # Bottom
MT = 173.0  # Top
MTA = 1.777  # Tau
MW = 80.419  # W

masses4t = array('d', [MT]*4)
massesWb = array('d', [MW, MB])  # Use for t -> w b
masses0 = array('d', [0, 0])  # Use for w -> u d, s c, ve e, vm mu
masses4u4d4b = array('d', [0.0, 0.0, MB]*4)

p23 = lhapdf.mkPDF("NNPDF23_lo_as_0130_qed", 0)

ROOT.gInterpreter.ProcessLine('#include "JustGenPhaseSpace.h"')

# %%


# TODO By default, apply no cuts: cutptl=0, cutetal=infinity???
def qqee_gen_ph_spc(energy=ENERGY, npts=int(1e5), cutptl=10, cutetal=2.5):
    n = npts
    # matrix2py (madgraph) only takes initial and final 4-momentum
    thedata4mg = np.empty((n, 4, 4))
    weights = np.empty((n))

    qqee = ROOT.TGenPhaseSpace()

    if type(energy) in (float, int):
        energy1 = float(energy)/2.0
        energy2 = float(energy)/2.0
    elif len(energy) == 2:
        energy1 = energy[0]
        energy2 = energy[1]
    else:
        print("Type of parameter 'energy' not recognized:", type(energy))
        print("Setting both quarks 'energy' as half of:", energy)
        energy1 = energy/2.0
        energy2 = energy/2.0

    j = 0
    cutpts = 0
    while j < n:
        # ===============================================================
        # TODO What about using a condition that if enbeam1 or enbeam2 go past
        # energy1 or energy2, count it as part of the cut?
        enbeam1 = np.random.rand()*energy1
        # enbeam2_min = np.max([0, 4*MT - enbeam1])
        # enbeam2 = np.random.rand()*(energy/2.0 - enbeam2_min) + enbeam2_min
        enbeam2 = np.random.rand()*energy2
        # if enbeam2 < 0:
        #     print("NEGATIVE ENERGY!", enbeam1, enbeam2_min, enbeam2)
        beam1 = ROOT.TLorentzVector(0.0, 0.0, enbeam1, enbeam1)
        beam2 = ROOT.TLorentzVector(0.0, 0.0, -enbeam2, enbeam2)
        # beam1MG = [beam1[3], beam1[0], beam1[1], beam1[2]]
        # beam2MG = [beam2[3], beam2[0], beam2[1], beam2[2]]
        beamtot = beam1 + beam2

        qqee.SetDecay(beamtot, 2, masses0)
        # ===============================================================

        # Generate event
        weights[j] = qqee.Generate()

        # Get momenta of decay products
        pe = qqee.GetDecay(0)
        pp = qqee.GetDecay(1)

        # Reorder 4-momenta for using in matrix2py
        thedata4mg[j] = np.array(
            [
                [p[3], p[0], p[1], p[2]] for p in [
                    beam1, beam2,
                    pe, pp
                ]
            ]
        )

        petl = np.sqrt(pe[0]**2 + pe[1]**2)
        pptl = np.sqrt(pp[0]**2 + pp[1]**2)

        cthe = pe[2]/np.sqrt(pe[0]**2 + pe[1]**2 + pe[2]**2)
        the = np.arccos(cthe)
        etae = -np.log(np.tan(the/2))
        cthp = pp[2]/np.sqrt(pp[0]**2 + pp[1]**2 + pp[2]**2)
        thp = np.arccos(cthp)
        etap = -np.log(np.tan(thp/2))

        # Check for NaN and apply cuts
        if weights[j] == weights[j] and petl > cutptl and pptl > cutptl and np.abs(etae) < cutetal and np.abs(etap) < cutetal:
            j += 1
        else:
            cutpts += 1

    return thedata4mg, weights, cutpts


# TODO This would be better as an instantiable class or wrapped in another def
def qqee_gen_ph_spc_fast(energy=ENERGY, npts=int(1e5), cutptl=10, cutetal=2.5):
    if type(energy) in (float, int):
        energy1 = float(energy)/2.0
        energy2 = float(energy)/2.0
    elif len(energy) == 2:
        energy1 = energy[0]
        energy2 = energy[1]
    else:
        print("Type of parameter 'energy' not recognized:", type(energy))
        print("Setting both quarks 'energy' as half of:", energy)
        energy1 = energy/2.0
        energy2 = energy/2.0

    rseed = np.random.randint(1, 100000000)

    jgps = ROOT.gen_qqee_space(energy1, energy2, npts, cutptl, cutetal, rseed)

    weights = np.asarray(jgps.weight)

    enbeam1 = np.asarray(jgps.Ebeam1)
    enbeam2 = np.asarray(jgps.Ebeam2)
    pex = np.asarray(jgps.pex)
    pey = np.asarray(jgps.pey)
    pez = np.asarray(jgps.pez)

    ppz = enbeam1 - enbeam2 - pez
    pem = np.sqrt(pex**2 + pey**2 + pez**2)
    ppm = np.sqrt(pex**2 + pey**2 + ppz**2)

    zeros = np.zeros((npts))

    events = np.array([
        enbeam1, zeros, zeros, enbeam1,
        enbeam2, zeros, zeros, -enbeam2,
        pem, pex, pey, pez,
        ppm, -pex, -pey, ppz,
     ]).T.reshape((npts, 4, 4))

    return events, weights, int(jgps.cutpts)


def gg4u4d4b_gen_ph_spc_fast(energy=ENERGY, npts=int(1e5), cutptjet=20.0, cutptb=0.0, cutetajet=5.0, cutetab=-1):
    if type(energy) in (float, int):
        energy1 = float(energy)/2.0
        energy2 = float(energy)/2.0
    elif len(energy) == 2:
        energy1 = energy[0]
        energy2 = energy[1]
    else:
        print("Type of parameter 'energy' not recognized:", type(energy))
        print("Setting both quarks 'energy' as half of:", energy)
        energy1 = energy/2.0
        energy2 = energy/2.0

    rseed = np.random.randint(1, 100000000)

    jgps = ROOT.gen_gg4u4d4b_space(energy1, energy2, npts, cutptjet, cutptb, cutetajet, cutetab, rseed)

    weights = np.asarray(jgps.weight)

    enbeam1 = np.asarray(jgps.Ebeam1)
    enbeam2 = np.asarray(jgps.Ebeam2)

    pu1x, pu1y, pu1z = np.asarray(jgps.pu1x), np.asarray(jgps.pu1y), np.asarray(jgps.pu1z)
    pd1x, pd1y, pd1z = np.asarray(jgps.pd1x), np.asarray(jgps.pd1y), np.asarray(jgps.pd1z)
    pb1x, pb1y, pb1z = np.asarray(jgps.pb1x), np.asarray(jgps.pb1y), np.asarray(jgps.pb1z)

    pu2x, pu2y, pu2z = np.asarray(jgps.pu2x), np.asarray(jgps.pu2y), np.asarray(jgps.pu2z)
    pd2x, pd2y, pd2z = np.asarray(jgps.pd2x), np.asarray(jgps.pd2y), np.asarray(jgps.pd2z)
    pb2x, pb2y, pb2z = np.asarray(jgps.pb2x), np.asarray(jgps.pb2y), np.asarray(jgps.pb2z)

    pu3x, pu3y, pu3z = np.asarray(jgps.pu3x), np.asarray(jgps.pu3y), np.asarray(jgps.pu3z)
    pd3x, pd3y, pd3z = np.asarray(jgps.pd3x), np.asarray(jgps.pd3y), np.asarray(jgps.pd3z)
    pb3x, pb3y, pb3z = np.asarray(jgps.pb3x), np.asarray(jgps.pb3y), np.asarray(jgps.pb3z)

    pu4x, pu4y, pu4z = np.asarray(jgps.pu4x), np.asarray(jgps.pu4y), np.asarray(jgps.pu4z)
    pd4x, pd4y, pd4z = np.asarray(jgps.pd4x), np.asarray(jgps.pd4y), np.asarray(jgps.pd4z)
    pb4x = -(
        pu1x + pd1x + pb1x
        + pu2x + pd2x + pb2x
        + pu3x + pd3x + pb3x
        + pu4x + pd4x
    )
    pb4y = -(
        pu1y + pd1y + pb1y
        + pu2y + pd2y + pb2y
        + pu3y + pd3y + pb3y
        + pu4y + pd4y
    )
    pb4y = -(
        pu1y + pd1y + pb1y
        + pu2y + pd2y + pb2y
        + pu3y + pd3y + pb3y
        + pu4y + pd4y
    )
    pb4z = enbeam1 - enbeam2 - (
        pu1z + pd1z + pb1z
        + pu2z + pd2z + pb2z
        + pu3z + pd3z + pb3z
        + pu4z + pd4z
    )

    Eu1 = np.sqrt(pu1x**2 + pu1y**2 + pu1z**2)
    Ed1 = np.sqrt(pd1x**2 + pd1y**2 + pd1z**2)
    Eb1 = np.sqrt(pb1x**2 + pb1y**2 + pb1z**2 + MB**2)
    Eu2 = np.sqrt(pu2x**2 + pu2y**2 + pu2z**2)
    Ed2 = np.sqrt(pd2x**2 + pd2y**2 + pd2z**2)
    Eb2 = np.sqrt(pb2x**2 + pb2y**2 + pb2z**2 + MB**2)
    Eu3 = np.sqrt(pu3x**2 + pu3y**2 + pu3z**2)
    Ed3 = np.sqrt(pd3x**2 + pd3y**2 + pd3z**2)
    Eb3 = np.sqrt(pb3x**2 + pb3y**2 + pb3z**2 + MB**2)
    Eu4 = np.sqrt(pu4x**2 + pu4y**2 + pu4z**2)
    Ed4 = np.sqrt(pd4x**2 + pd4y**2 + pd4z**2)
    Eb4 = np.sqrt(pb4x**2 + pb4y**2 + pb4z**2 + MB**2)

    zeros = np.zeros((npts))

    events = np.array([
        enbeam1, zeros, zeros, enbeam1,
        enbeam2, zeros, zeros, -enbeam2,
        Eu1, pu1x, pu1y, pu1z,
        Ed1, pd1x, pd1y, pd1z,
        Eb1, pb1x, pb1y, pb1z,
        Eu2, pu2x, pu2y, pu2z,
        Ed2, pd2x, pd2y, pd2z,
        Eb2, pb2x, pb2y, pb2z,
        Eu3, pu3x, pu3y, pu3z,
        Ed3, pd3x, pd3y, pd3z,
        Eb3, pb3x, pb3y, pb3z,
        Eu4, pu4x, pu4y, pu4z,
        Ed4, pd4x, pd4y, pd4z,
        Eb4, pb4x, pb4y, pb4z,
    ]).T.reshape((npts, 14, 4))

    # Remove NANs
    events = events[weights == weights]
    weights = weights[weights == weights]

    return events, weights, int(jgps.cutpts) + (weights != weights).sum()


def gg4u4d4b_gen_ph_spc(energy=ENERGY, npts=int(1e5)):
    if type(energy) in (float, int):
        energy1 = float(energy)/2.0
        energy2 = float(energy)/2.0
    elif len(energy) == 2:
        energy1 = energy[0]
        energy2 = energy[1]
    else:
        print("Type of parameter 'energy' not recognized:", type(energy))
        print("Setting both quarks 'energy' as half of:", energy)
        energy1 = energy/2.0
        energy2 = energy/2.0

    n = npts
    # matrix2py (madgraph) only takes initial and final 4-momentum
    thedata4mg = np.empty((n, 14, 4))
    weights = np.empty((n))

    gg4u4d4b = ROOT.TGenPhaseSpace()

    j = 0
    # TODO Add cut
    ncut = 0
    while j < n:
        # ===============================================================
        enbeam1 = np.random.rand()*energy1
        # enbeam2_min = np.max([0, 4*MT - enbeam1])
        # enbeam2 = np.random.rand()*(energy/2.0 - enbeam2_min) + enbeam2_min
        enbeam2 = np.random.rand()*energy2
        # if enbeam2 < 0:
        #     print("NEGATIVE ENERGY!", enbeam1, enbeam2_min, enbeam2)
        beam1 = ROOT.TLorentzVector(0.0, 0.0, enbeam1, enbeam1)
        beam2 = ROOT.TLorentzVector(0.0, 0.0, -enbeam2, enbeam2)
        # beam1MG = [beam1[3], beam1[0], beam1[1], beam1[2]]
        # beam2MG = [beam2[3], beam2[0], beam2[1], beam2[2]]
        beamtot = beam1 + beam2

        gg4u4d4b.SetDecay(beamtot, 12, masses4u4d4b)
        # ===============================================================

        # Generate event
        weights[j] = gg4u4d4b.Generate()

        # Get momenta of decay products
        pu1 = gg4u4d4b.GetDecay(0)
        pd1 = gg4u4d4b.GetDecay(1)
        pb1 = gg4u4d4b.GetDecay(2)
        pu2 = gg4u4d4b.GetDecay(3)
        pd2 = gg4u4d4b.GetDecay(4)
        pb2 = gg4u4d4b.GetDecay(5)
        pu3 = gg4u4d4b.GetDecay(6)
        pd3 = gg4u4d4b.GetDecay(7)
        pb3 = gg4u4d4b.GetDecay(8)
        pu4 = gg4u4d4b.GetDecay(9)
        pd4 = gg4u4d4b.GetDecay(10)
        pb4 = gg4u4d4b.GetDecay(11)

        # Use the ordering that is used in madgraph: [3, 0, 1, 2] of
        # TGenPhaseSpace output
        #
        # I used this output format for the cpp script
        # thedata4mg[j] = np.array([
        #     pt1[3], pt1[0], pt1[1], pt1[2],
        #     pt2[3], pt2[0], pt2[1], pt2[2],
        #     pt3[3], pt3[0], pt3[1], pt3[2],
        #     pt4[3], pt4[0], pt4[1], pt4[2]
        # ])
        # Output format for using matrix2py
        thedata4mg[j] = np.array(
            [
                [p[3], p[0], p[1], p[2]] for p in [
                    beam1, beam2,
                    pu1, pd1, pb1,
                    pu2, pd2, pb2,
                    pu3, pd3, pb3,
                    pu4, pd4, pb4
                ]
            ]
        )
        # Check for NaN
        if weights[j] == weights[j]:
            j += 1
    return thedata4mg, weights, ncut


def ft_gen_ph_spc(energy=ENERGY, npts=int(1e5)):
    n = npts
    # matrix2py (madgraph) only takes initial and final 4-momentum
    thedata4mg = np.empty((n, 14, 4))
    # Make another array to store all of them
    thedataALL = np.empty((n, 18, 4))
    weights = np.empty((n))
    weights1 = np.empty((n))
    weightsT1 = np.empty((n))
    weightsT2 = np.empty((n))
    weightsT3 = np.empty((n))
    weightsT4 = np.empty((n))
    weightsW1 = np.empty((n))
    weightsW2 = np.empty((n))
    weightsW3 = np.empty((n))
    weightsW4 = np.empty((n))

    ftops = ROOT.TGenPhaseSpace()

    j = 0
    while j < n:
        # ===============================================================
        enbeam1 = np.random.rand()*energy/2.0
        # enbeam2_min = np.max([0, 4*MT - enbeam1])
        # enbeam2 = np.random.rand()*(energy/2.0 - enbeam2_min) + enbeam2_min
        enbeam2 = np.random.rand()*energy/2.0
        # if enbeam2 < 0:
        #     print("NEGATIVE ENERGY!", enbeam1, enbeam2_min, enbeam2)
        beam1 = ROOT.TLorentzVector(0.0, 0.0, enbeam1, enbeam1)
        beam2 = ROOT.TLorentzVector(0.0, 0.0, -enbeam2, enbeam2)
        # beam1MG = [beam1[3], beam1[0], beam1[1], beam1[2]]
        # beam2MG = [beam2[3], beam2[0], beam2[1], beam2[2]]
        beamtot = beam1 + beam2

        ftops.SetDecay(beamtot, 4, masses4t)
        # ===============================================================

        # Generate event
        weights1[j] = ftops.Generate()

        # Get momenta of decay products
        pt1 = ftops.GetDecay(0)
        pt2 = ftops.GetDecay(1)
        pt3 = ftops.GetDecay(2)
        pt4 = ftops.GetDecay(3)

        # 1. Instances for each top
        t1dec = ROOT.TGenPhaseSpace()
        t2dec = ROOT.TGenPhaseSpace()
        t3dec = ROOT.TGenPhaseSpace()
        t4dec = ROOT.TGenPhaseSpace()
        # 2. Set decays
        t1dec.SetDecay(pt1, 2, massesWb)
        t2dec.SetDecay(pt2, 2, massesWb)
        t3dec.SetDecay(pt3, 2, massesWb)
        t4dec.SetDecay(pt4, 2, massesWb)

        # 3. Generate event
        weightsT1[j] = t1dec.Generate()
        weightsT2[j] = t2dec.Generate()
        weightsT3[j] = t3dec.Generate()
        weightsT4[j] = t4dec.Generate()
        weightTAll = weightsT2[j]*weightsT2[j]*weightsT3[j]*weightsT4[j]

        # 4. Get momenta for decay products
        pw1 = t1dec.GetDecay(0)
        pb1 = t1dec.GetDecay(1)
        pw2 = t2dec.GetDecay(0)
        pb2 = t2dec.GetDecay(1)
        pw3 = t3dec.GetDecay(0)
        pb3 = t3dec.GetDecay(1)
        pw4 = t4dec.GetDecay(0)
        pb4 = t4dec.GetDecay(1)

        # 1. Instances for each W
        w1dec = ROOT.TGenPhaseSpace()
        w2dec = ROOT.TGenPhaseSpace()
        w3dec = ROOT.TGenPhaseSpace()
        w4dec = ROOT.TGenPhaseSpace()
        # 2. Set decays
        # Check the folder for the matrix element corresponding to matrix2py
        w1dec.SetDecay(pw1, 2, masses0)
        w2dec.SetDecay(pw2, 2, masses0)
        w3dec.SetDecay(pw3, 2, masses0)
        w4dec.SetDecay(pw4, 2, masses0)

        # 3. Generate event
        weightsW1[j] = w1dec.Generate()
        weightsW2[j] = w2dec.Generate()
        weightsW3[j] = w3dec.Generate()
        weightsW4[j] = w4dec.Generate()
        weightWAll = weightsW1[j]*weightsW2[j]*weightsW3[j]*weightsW4[j]

        # 4. Get momenta for decay products
        pw1to1 = w1dec.GetDecay(0)
        pw1to2 = w1dec.GetDecay(1)
        pw2to1 = w2dec.GetDecay(0)
        pw2to2 = w2dec.GetDecay(1)
        pw3to1 = w3dec.GetDecay(0)
        pw3to2 = w3dec.GetDecay(1)
        pw4to1 = w4dec.GetDecay(0)
        pw4to2 = w4dec.GetDecay(1)

        # Use the ordering that is used in madgraph: [3, 0, 1, 2] of
        # TGenPhaseSpace output
        #
        # I used this output format for the cpp script
        # thedata4mg[j] = np.array([
        #     pt1[3], pt1[0], pt1[1], pt1[2],
        #     pt2[3], pt2[0], pt2[1], pt2[2],
        #     pt3[3], pt3[0], pt3[1], pt3[2],
        #     pt4[3], pt4[0], pt4[1], pt4[2]
        # ])
        # Output format for using matrix2py
        thedata4mg[j] = np.array(
            [
                [p[3], p[0], p[1], p[2]] for p in [
                    beam1, beam2,
                    pw1to1, pw1to2, pb1,
                    pw2to1, pw2to2, pb2,
                    pw3to1, pw3to2, pb3,
                    pw4to1, pw4to2, pb4
                ]
            ]
        )
        thedataALL[j] = np.array(
            [
                [p[3], p[0], p[1], p[2]] for p in [
                    beam1, beam2,
                    pt1, pt2, pt3, pt4,
                    pw1to1, pw1to2, pb1,
                    pw2to1, pw2to2, pb2,
                    pw3to1, pw3to2, pb3,
                    pw4to1, pw4to2, pb4
                ]
            ]
        )
        weights[j] = weights1[j]*weightTAll*weightWAll
        if weights[j] == weights[j]:
            j += 1
    return thedata4mg, weights, thedataALL


def invert_momenta(p):
    # See https://cp3.irmp.ucl.ac.be/projects/madgraph/wiki/FAQ-General-4
    """ fortran/C-python do not order table in the same order"""
    new_p = []
    for i in range(len(p[0])):  new_p.append([0]*len(p))
    for i, onep in enumerate(p):
        for j, x in enumerate(onep):
            new_p[j][i] = x
    return new_p


def ft_get_mg5weights(momenta, alphas):
    # See https://cp3.irmp.ucl.ac.be/projects/madgraph/wiki/FAQ-General-4
    n = momenta.shape[0]
    matrix2py.initialisemodel('../../Cards/param_card.dat')

    mgweights = np.empty((n))
    for j in range(n):
        p = [*momenta[j]]

        P = invert_momenta(p)
        nhel = -1  # means sum over all helicity
        mgweights[j] = matrix2py.get_value(P, alphas[j], nhel)
    return mgweights
