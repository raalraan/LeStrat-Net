# Selection of regions of importance with machine learning

This is the code used for the examples.
At the moment is the code for the $u\bar{u} \to e^+ e^-$ example.

## Generating `matrix2py` using MadGraph

First, get `MadGraph` from [this link](https://launchpad.net/mg5amcnlo) and
follow one of many tutorial for its installation.
To generate `matrix2py`, first generate some process using `MadGraph` command
line.

	mg5> generate u u~ > e+ e-

Other necessary packages are `ExRootAnalysis` and `lhapdf6`.
Both can be easily installed with `MadGraph`:

	mg5> install ExRootAnalysis
	mg5> install lhapdf6

