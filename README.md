# Selection of regions of importance with machine learning

This is the code used for the examples.
At the moment is the code for the $u\bar{u} \to e^+ e^-$ example.


## Note about installing requirements

First of all, `lhapdf6` requires an older version of python.
Therefore this code has been tested with `python` 3.9.18.
The easiest way to install most of the required components with full compatibility
is by using `conda` and installing everything in a conda environment with

	$> conda create -y -c conda-forge --name py39-mlmc python=3.9 root matplotlib

where the environment `py39-mlmc` is used as a suggestion but can be any other word.
Add `jupyter` in case you are interested in using jupyter kernel and/or notebooks.
Activate the newly created environment with `conda activate py39-mlmc`.

For some reason TensorFlow works better when installed with pip than when installed with conda.
Additionally, in the [install page](https://www.tensorflow.org/install) for
TensorFlow documentation, the `pip` method is recommended.
To install TensorFlow with `pip` run

	(py39-mlmc) $> pip install tensorflow

If you want you can try installing TensorFlow with conda and if it works this
instructions could be shortened.

TODO: use `environment.yml` if we do not mind enforcing conda.


## Generating `matrix2py` using MadGraph

First, get `MadGraph` from [this link](https://launchpad.net/mg5amcnlo) and
follow one of many tutorial for its installation.
To generate `matrix2py`, first generate some process using `MadGraph` command
line.

	MG5_aMC> generate u u~ > e+ e-
	MG5_aMC> output standalone my_uu_to_ee
	MG5_aMC> exit

The string `my_uu_to_ee` is the name of the folder where the process code has been saved.
Now we need to change to the directory for the subprocess and build the `matrix2py` module

	$> cd my_uu_to_ee/SubProcesses/P1_uux_epem
	$> make matrix2py.so

See this [FAQ](https://cp3.irmp.ucl.ac.be/projects/madgraph/wiki/FAQ-General-4) for more information about `matrix2py`.

Other necessary packages are `ExRootAnalysis` and `lhapdf6`.
Both can be easily installed with `MadGraph`:

	MG5_aMC> install ExRootAnalysis
	MG5_aMC> install lhapdf6

`ExRootAnalysis` is used to convert events to `ROOT` format,
convenient since we are already using `ROOT` for phase space generation.
In the case of `lhapdf6`, we use the python module to calculate
PDFs for initial partons.
