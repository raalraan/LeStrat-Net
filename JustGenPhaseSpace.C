void JustGenPhaseSpace(
		const Double_t energy1, const Double_t energy2,
		const int npts,
		const Double_t cutptl,
		const Double_t cutetal,
		const int seed
		) {
cout.precision(17);

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

	/* test[n][0] = pEle->E(); */
	/* test[n][2] = pEle->Px(); */
	/* test[n][3] = pEle->Py(); */
	/* test[n][4] = pEle->Pz(); */
	/* test[n][4] = pPos->E(); */
	/* test[n][5] = pPos->Px(); */
	/* test[n][6] = pPos->Py(); */
	/* test[n][7] = pPos->Pz(); */
	/* test[n][0] = enbeam1; */
	/* test[n][1] = enbeam2; */

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
		n++;
		cout << fixed << weight << " "
			<< enbeam1 << " "
			<< enbeam2 << " "
			<< pex << " "
			<< pey << " "
			<< pez << " ";

	}
	else
	{
		cutpts++;
	}
}

cout << endl << cutpts << endl;


}
