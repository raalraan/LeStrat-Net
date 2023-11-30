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
To install tensorflow with `pip` run

	(py39-mlmc) $> pip install tensorflow

If you want you can try installing tensorflow with conda and if it works this
instructions could be shortened.

TODO: use `environment.yml` if we do not mind enforcing conda usage.

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

