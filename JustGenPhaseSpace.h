struct qqee_space {
	std::vector<Double_t> pex;
	std::vector<Double_t> pey;
	std::vector<Double_t> pez;
	std::vector<Double_t> Ebeam1;
	std::vector<Double_t> Ebeam2;
	std::vector<Double_t> weight;
	int cutpts;
};

struct gg4u4d4b_space {
	std::vector<Double_t> pu1x;
	std::vector<Double_t> pu1y;
	std::vector<Double_t> pu1z;
	std::vector<Double_t> pd1x;
	std::vector<Double_t> pd1y;
	std::vector<Double_t> pd1z;
	std::vector<Double_t> pb1x;
	std::vector<Double_t> pb1y;
	std::vector<Double_t> pb1z;

	std::vector<Double_t> pu2x;
	std::vector<Double_t> pu2y;
	std::vector<Double_t> pu2z;
	std::vector<Double_t> pd2x;
	std::vector<Double_t> pd2y;
	std::vector<Double_t> pd2z;
	std::vector<Double_t> pb2x;
	std::vector<Double_t> pb2y;
	std::vector<Double_t> pb2z;

	std::vector<Double_t> pu3x;
	std::vector<Double_t> pu3y;
	std::vector<Double_t> pu3z;
	std::vector<Double_t> pd3x;
	std::vector<Double_t> pd3y;
	std::vector<Double_t> pd3z;
	std::vector<Double_t> pb3x;
	std::vector<Double_t> pb3y;
	std::vector<Double_t> pb3z;

	std::vector<Double_t> pu4x;
	std::vector<Double_t> pu4y;
	std::vector<Double_t> pu4z;
	std::vector<Double_t> pd4x;
	std::vector<Double_t> pd4y;
	std::vector<Double_t> pd4z;

	std::vector<Double_t> Ebeam1;
	std::vector<Double_t> Ebeam2;
	std::vector<Double_t> weight;
	int cutpts;
};

qqee_space gen_qqee_space(
		const Double_t energy1, const Double_t energy2,
		const int npts,
		const Double_t cutptl,
		const Double_t cutetal,
		const int seed
		) {
	qqee_space mypspace;

	mypspace.weight.resize(npts);
	mypspace.Ebeam1.resize(npts);
	mypspace.Ebeam2.resize(npts);
	mypspace.pex.resize(npts);
	mypspace.pey.resize(npts);
	mypspace.pez.resize(npts);

	TRandom3 r;
	r.SetSeed(seed);

	Double_t masses0[2] = {0.0, 0.0};
	TGenPhaseSpace qqee;

	int n = 0;
	int cutpts = 0;
	while (n < npts)
	{
		Double_t enbeam1 = r.Rndm()*energy1;
		Double_t enbeam2 = r.Rndm()*energy2;

		TLorentzVector beam1(0.0, 0.0, enbeam1, enbeam1);
		TLorentzVector beam2(0.0, 0.0, -enbeam2, enbeam2);

		TLorentzVector beamtot = beam1 + beam2;

		qqee.SetDecay(beamtot, 2, masses0);

		Double_t weight = qqee.Generate();
		TLorentzVector *pEle = qqee.GetDecay(0);
		TLorentzVector *pPos = qqee.GetDecay(1);

		Double_t pex = pEle->Px();
		Double_t pey = pEle->Py();
		Double_t pez = pEle->Pz();
		Double_t ppx = pPos->Px();
		Double_t ppy = pPos->Py();
		Double_t ppz = pPos->Pz();

		Double_t petl = sqrt(pex*pex + pey*pey);
		Double_t pptl = sqrt(ppx*ppx + ppy*ppy);

		Double_t cthe = pez/sqrt(pex*pex + pey*pey + pez*pez);
		Double_t the = acos(cthe);
		Double_t etae = -log(tan(the/2.0));

		Double_t cthp = ppz/sqrt(ppx*ppx + ppy*ppy + ppz*ppz);
		Double_t thp = acos(cthp);
		Double_t etap = -log(tan(thp/2.0));

		if (petl > cutptl && pptl > cutptl && abs(etae) < cutetal && abs(etap) < cutetal)
		{
			mypspace.weight[n] = weight;
			mypspace.Ebeam1[n] = enbeam1;
			mypspace.Ebeam2[n] = enbeam2;
			mypspace.pex[n] = pex;
			mypspace.pey[n] = pey;
			mypspace.pez[n] = pez;
			n++;
		}
		else
		{
			cutpts++;
		}

	}

	mypspace.cutpts = cutpts;

	return mypspace;
}

gg4u4d4b_space gen_gg4u4d4b_space(
		const Double_t energy1, const Double_t energy2,
		const int npts,
		const Double_t cutptjet,
		const Double_t cutptb,
		const Double_t cutetajet,
		const Double_t cutetab,
		const int seed
		) {
	gg4u4d4b_space mypspace;

	mypspace.weight.resize(npts);
	mypspace.Ebeam1.resize(npts);
	mypspace.Ebeam2.resize(npts);
	mypspace.pu1x.resize(npts); mypspace.pu1y.resize(npts); mypspace.pu1z.resize(npts);
	mypspace.pd1x.resize(npts); mypspace.pd1y.resize(npts); mypspace.pd1z.resize(npts);
	mypspace.pb1x.resize(npts); mypspace.pb1y.resize(npts); mypspace.pb1z.resize(npts);

	mypspace.pu2x.resize(npts); mypspace.pu2y.resize(npts); mypspace.pu2z.resize(npts);
	mypspace.pd2x.resize(npts); mypspace.pd2y.resize(npts); mypspace.pd2z.resize(npts);
	mypspace.pb2x.resize(npts); mypspace.pb2y.resize(npts); mypspace.pb2z.resize(npts);

	mypspace.pu3x.resize(npts); mypspace.pu3y.resize(npts); mypspace.pu3z.resize(npts);
	mypspace.pd3x.resize(npts); mypspace.pd3y.resize(npts); mypspace.pd3z.resize(npts);
	mypspace.pb3x.resize(npts); mypspace.pb3y.resize(npts); mypspace.pb3z.resize(npts);

	mypspace.pu4x.resize(npts); mypspace.pu4y.resize(npts); mypspace.pu4z.resize(npts);
	mypspace.pd4x.resize(npts); mypspace.pd4y.resize(npts); mypspace.pd4z.resize(npts);

	TRandom3 r;
	r.SetSeed(seed);

	Double_t masses0[12] = {0.0, 0.0, 4.7, 0.0, 0.0, 4.7, 0.0, 0.0, 4.7, 0.0, 0.0, 4.7};
	TGenPhaseSpace gg4u4d4b;

	int n = 0;
	int cutpts = 0;
	while (n < npts)
	{
		Double_t enbeam1 = r.Rndm()*energy1;
		Double_t enbeam2 = r.Rndm()*energy2;

		if (enbeam1 + enbeam2 < 4.*173.0)
		{
			cutpts++;
			continue;
		}

		TLorentzVector beam1(0.0, 0.0, enbeam1, enbeam1);
		TLorentzVector beam2(0.0, 0.0, -enbeam2, enbeam2);

		TLorentzVector beamtot = beam1 + beam2;

		gg4u4d4b.SetDecay(beamtot, 12, masses0);

		Double_t weight = gg4u4d4b.Generate();
		TLorentzVector *pu1 = gg4u4d4b.GetDecay(0);
		TLorentzVector *pd1 = gg4u4d4b.GetDecay(1);
		TLorentzVector *pb1 = gg4u4d4b.GetDecay(2);
		TLorentzVector *pu2 = gg4u4d4b.GetDecay(3);
		TLorentzVector *pd2 = gg4u4d4b.GetDecay(4);
		TLorentzVector *pb2 = gg4u4d4b.GetDecay(5);
		TLorentzVector *pu3 = gg4u4d4b.GetDecay(6);
		TLorentzVector *pd3 = gg4u4d4b.GetDecay(7);
		TLorentzVector *pb3 = gg4u4d4b.GetDecay(8);
		TLorentzVector *pu4 = gg4u4d4b.GetDecay(9);
		TLorentzVector *pd4 = gg4u4d4b.GetDecay(10);
		TLorentzVector *pb4 = gg4u4d4b.GetDecay(11);

		Double_t pu1x = pu1->Px();
		Double_t pu1y = pu1->Py();
		Double_t pu1z = pu1->Pz();
		Double_t pd1x = pd1->Px();
		Double_t pd1y = pd1->Py();
		Double_t pd1z = pd1->Pz();
		Double_t pb1x = pb1->Px();
		Double_t pb1y = pb1->Py();
		Double_t pb1z = pb1->Pz();

		Double_t pu2x = pu2->Px();
		Double_t pu2y = pu2->Py();
		Double_t pu2z = pu2->Pz();
		Double_t pd2x = pd2->Px();
		Double_t pd2y = pd2->Py();
		Double_t pd2z = pd2->Pz();
		Double_t pb2x = pb2->Px();
		Double_t pb2y = pb2->Py();
		Double_t pb2z = pb2->Pz();

		Double_t pu3x = pu3->Px();
		Double_t pu3y = pu3->Py();
		Double_t pu3z = pu3->Pz();
		Double_t pd3x = pd3->Px();
		Double_t pd3y = pd3->Py();
		Double_t pd3z = pd3->Pz();
		Double_t pb3x = pb3->Px();
		Double_t pb3y = pb3->Py();
		Double_t pb3z = pb3->Pz();

		Double_t pu4x = pu4->Px();
		Double_t pu4y = pu4->Py();
		Double_t pu4z = pu4->Pz();
		Double_t pd4x = pd4->Px();
		Double_t pd4y = pd4->Py();
		Double_t pd4z = pd4->Pz();
		Double_t pb4x = pb4->Px();
		Double_t pb4y = pb4->Py();
		Double_t pb4z = pb4->Pz();

		if (cutptjet > 0.0)
		{
			Double_t pu1t = sqrt(pu1x*pu1x + pu1y*pu1y);
			Double_t pd1t = sqrt(pd1x*pd1x + pd1y*pd1y);

			Double_t pu2t = sqrt(pu2x*pu2x + pu2y*pu2y);
			Double_t pd2t = sqrt(pd2x*pd2x + pd2y*pd2y);

			Double_t pu3t = sqrt(pu3x*pu3x + pu3y*pu3y);
			Double_t pd3t = sqrt(pd3x*pd3x + pd3y*pd3y);

			Double_t pu4t = sqrt(pu4x*pu4x + pu4y*pu4y);
			Double_t pd4t = sqrt(pd4x*pd4x + pd4y*pd4y);

			if (
					pu1t < cutptjet || pd1t < cutptjet ||
					pu2t < cutptjet || pd2t < cutptjet ||
					pu3t < cutptjet || pd3t < cutptjet ||
					pu4t < cutptjet || pd4t < cutptjet
			   )
			{
				cutpts++;
				continue;
			}
		}

		if (cutptb > 0.0)
		{
			Double_t pb1t = sqrt(pb1x*pb1x + pb1y*pb1y);
			Double_t pb2t = sqrt(pb2x*pb2x + pb2y*pb2y);
			Double_t pb3t = sqrt(pb3x*pb3x + pb3y*pb3y);
			Double_t pb4t = sqrt(pb4x*pb4x + pb4y*pb4y);

			if (pb1t < cutptb || pb2t < cutptb || pb3t < cutptb || pb4t < cutptb)
			{
				cutpts++;
				continue;
			}
		}

		if (cutetajet > 0.0)
		{
			Double_t cthu1 = pu1z/sqrt(pu1x*pu1x + pu1y*pu1y + pu1z*pu1z);
			Double_t thu1 = acos(cthu1);
			Double_t etau1 = -log(tan(thu1/2.0));
			Double_t cthd1 = pd1z/sqrt(pd1x*pd1x + pd1y*pd1y + pd1z*pd1z);
			Double_t thd1 = acos(cthd1);
			Double_t etad1 = -log(tan(thd1/2.0));

			Double_t cthu2 = pu2z/sqrt(pu2x*pu2x + pu2y*pu2y + pu2z*pu2z);
			Double_t thu2 = acos(cthu2);
			Double_t etau2 = -log(tan(thu2/2.0));
			Double_t cthd2 = pd2z/sqrt(pd2x*pd2x + pd2y*pd2y + pd2z*pd2z);
			Double_t thd2 = acos(cthd2);
			Double_t etad2 = -log(tan(thd2/2.0));

			Double_t cthu3 = pu3z/sqrt(pu3x*pu3x + pu3y*pu3y + pu3z*pu3z);
			Double_t thu3 = acos(cthu3);
			Double_t etau3 = -log(tan(thu3/2.0));
			Double_t cthd3 = pd3z/sqrt(pd3x*pd3x + pd3y*pd3y + pd3z*pd3z);
			Double_t thd3 = acos(cthd3);
			Double_t etad3 = -log(tan(thd3/2.0));

			Double_t cthu4 = pu4z/sqrt(pu4x*pu4x + pu4y*pu4y + pu4z*pu4z);
			Double_t thu4 = acos(cthu4);
			Double_t etau4 = -log(tan(thu4/2.0));
			Double_t cthd4 = pd4z/sqrt(pd4x*pd4x + pd4y*pd4y + pd4z*pd4z);
			Double_t thd4 = acos(cthd4);
			Double_t etad4 = -log(tan(thd4/2.0));

			if (
					etau1 > cutetajet  || etad1 > cutetajet || etau2 > cutetajet || etad2 > cutetajet ||
					etau3 > cutetajet  || etad3 > cutetajet || etau4 > cutetajet || etad4 > cutetajet
			   )
			{
				cutpts++;
				continue;
			}
		}

		if (cutetab > 0.0)
		{
			Double_t cthb1 = pb1z/sqrt(pb1x*pb1x + pb1y*pb1y + pb1z*pb1z);
			Double_t thb1 = acos(cthb1);
			Double_t etab1 = -log(tan(thb1/2.0));
			Double_t cthb2 = pb2z/sqrt(pb2x*pb2x + pb2y*pb2y + pb2z*pb2z);
			Double_t thb2 = acos(cthb2);
			Double_t etab2 = -log(tan(thb2/2.0));
			Double_t cthb3 = pb3z/sqrt(pb3x*pb3x + pb3y*pb3y + pb3z*pb3z);
			Double_t thb3 = acos(cthb3);
			Double_t etab3 = -log(tan(thb3/2.0));
			Double_t cthb4 = pb4z/sqrt(pb4x*pb4x + pb4y*pb4y + pb4z*pb4z);
			Double_t thb4 = acos(cthb4);
			Double_t etab4 = -log(tan(thb4/2.0));

			if (etab1 > cutetab  || etab2 > cutetab || etab3 > cutetab || etab4 > cutetab)
			{
				cutpts++;
				continue;
			}
		}

		mypspace.weight[n] = weight;
		mypspace.Ebeam1[n] = enbeam1;
		mypspace.Ebeam2[n] = enbeam2;

		mypspace.pu1x[n] = pu1x;
		mypspace.pu1y[n] = pu1y;
		mypspace.pu1z[n] = pu1z;
		mypspace.pd1x[n] = pd1x;
		mypspace.pd1y[n] = pd1y;
		mypspace.pd1z[n] = pd1z;
		mypspace.pb1x[n] = pb1x;
		mypspace.pb1y[n] = pb1y;
		mypspace.pb1z[n] = pb1z;

		mypspace.pu2x[n] = pu2x;
		mypspace.pu2y[n] = pu2y;
		mypspace.pu2z[n] = pu2z;
		mypspace.pd2x[n] = pd2x;
		mypspace.pd2y[n] = pd2y;
		mypspace.pd2z[n] = pd2z;
		mypspace.pb2x[n] = pb2x;
		mypspace.pb2y[n] = pb2y;
		mypspace.pb2z[n] = pb2z;

		mypspace.pu3x[n] = pu3x;
		mypspace.pu3y[n] = pu3y;
		mypspace.pu3z[n] = pu3z;
		mypspace.pd3x[n] = pd3x;
		mypspace.pd3y[n] = pd3y;
		mypspace.pd3z[n] = pd3z;
		mypspace.pb3x[n] = pb3x;
		mypspace.pb3y[n] = pb3y;
		mypspace.pb3z[n] = pb3z;

		mypspace.pu4x[n] = pu4x;
		mypspace.pu4y[n] = pu4y;
		mypspace.pu4z[n] = pu4z;
		mypspace.pd4x[n] = pd4x;
		mypspace.pd4y[n] = pd4y;
		mypspace.pd4z[n] = pd4z;

		n++;
	}

	mypspace.cutpts = cutpts;

	return mypspace;
}
