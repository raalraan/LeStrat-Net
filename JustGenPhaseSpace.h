struct qqee_space {
	std::vector<Double_t> pex;
	std::vector<Double_t> pey;
	std::vector<Double_t> pez;
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
			mypspace.weight.push_back(weight);
			mypspace.Ebeam1.push_back(enbeam1);
			mypspace.Ebeam2.push_back(enbeam2);
			mypspace.pex.push_back(pex);
			mypspace.pey.push_back(pey);
			mypspace.pez.push_back(pez);
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
