# dydx
![Tests](https://github.com/akinwilson/dydx/actions/workflows/tests.yml/badge.svg)

# Todo 23 Dec 2024

- [x] Factor XOR-gate modelling into its own example
- [x] Implemement dataset set and layer seed initialisation via passing seeds into layer classes and dataset object
- [x] implement the grad method and zero_grad method at the array level to be able to perform SVD
- [] write tests for Scalar object 
- [] convert private to public repository



![alt text](img/autodydx.jpg "Automatic differentiation")
## Overview
dydx is a library implementing [automatic differentiation](https://en.wikipedia.org/wiki/Automatic_differentiation); the gradient-based optimisation technqiue and [linear algebra](https://en.wikipedia.org/wiki/Linear_algebra) routines from scratch. I.e. using **only** python's built-in libraries and avoiding others such as [numpy](https://numpy.org/), [pytorch](https://pytorch.org/), [tensorflow](https://www.tensorflow.org/), [scipy](https://scipy.org/), [pandas](https://pandas.pydata.org/) etc.

To demonstrate purposes, it is applied to various problems in [numerical linear algebra](https://en.wikipedia.org/wiki/Numerical_linear_algebra) and [machine learning](https://en.wikipedia.org/wiki/Machine_learning), especially [deep learning](https://en.wikipedia.org/wiki/Deep_learning). The library is applied to three problems in particular in the `examples/` folder; [singular value decomposition](https://en.wikipedia.org/wiki/Singular_value_decomposition) of a random non-square matrix, modelling the [non-linearities](https://en.wikipedia.org/wiki/Linear_separability) of the [XOR gate](https://en.wikipedia.org/wiki/XOR_gate) function and finally, an industry [supervised](https://en.wikipedia.org/wiki/Supervised_learning) dataset of insurance claims. 

## Installation

Create a vritual environment, clone the repository and install the package via running: 

```bash
pip install .
```
from within the root of the cloned repository. You can also install the package **without** cloning this repository via running:
```bash
pip install git+https://github.com/akinwilson/dydx
```

## Usage
Check out the `examples/` folder to see how the library is used. You can from the root of this repository run the examples via:

```bash
python examples/xor_gate.py
```
**Note**: you may alter fitting parameters from the command line like: 

```bash 
python examples/xor_gate.py --epochs 250 --learning-rate 0.01 --layer-seeds 636915800,29155285,01355285
```
The rest of the examples can be run via:
```bash
python examples/singular_value_decomposition.py
```
and 
```bash
python examples/insurance_claims.py
```

The preset values for the arguments of each optimisation problem shown in the examples should yield desirable results. 

## Running tests

To run tests locally, install developer requirements

 ```
 pip install -r requirements_dev.txt
 ```
then, with 
```
python -m pytest
```
you can run the tests locally.


## Further improvements
### Hardware optimisation 
To further improve upon the speed/efficency of the library, if available, utilisation of a GPU's computational parallelism properties is paramount. With the spirit of doing everything from *scratch*, I have looked at python's bindings to [cuda](https://github.com/NVIDIA/cuda-python) which is what libraries such as [numba](https://numba.pydata.org/) use under the hood. 


## Citation

Jorge Nocedal, Stephen Wright. [Numercial Optimisationn](https://www.amazon.co.uk/Numerical-Optimization-Operations-Financial-Engineering/dp/1493937111/ref=asc_df_1493937111?mcid=5c9ad06c6e3937ce97423f4c7092ee47&th=1&psc=1&tag=googshopuk-21&linkCode=df0&hvadid=697265600136&hvpos=&hvnetw=g&hvrand=9286832652685731556&hvpone=&hvptwo=&hvqmt=&hvdev=c&hvdvcmdl=&hvlocint=&hvlocphy=9045844&hvtargid=pla-582150399259&psc=1&gad_source=1). In Numerical Optimization (Springer Series in Operations Research and Financial Engineering), 2009 Springer, pp. 204-221. Springer, 2009.
```tex
@inproceedings{wrightNumericalOptimisation09,
  title={Numercial Optimisationn},
  author={Jorge Nocedal, Stephen Wright},
  booktitle={Numerical Optimization (Springer Series in Operations Research and Financial Engineering)},
  pages={204--221},
  year={2009},
  organization={Springer}
}
```
