# dydx
dydx is a library implementing [automatic differentiation](https://en.wikipedia.org/wiki/Automatic_differentiation) in python and applying it to various problems in linear algebra and machine learning from **scratch**; that is avoiding all popular libraries; e.g. [numpy](https://numpy.org/) or [pandas](https://pandas.pydata.org/) for any functionality. 

A greatr resource is [Numerical Optimization](https://www.amazon.co.uk/Numerical-Optimization-Operations-Financial-Engineering/dp/1493937111/ref=asc_df_1493937111?mcid=5c9ad06c6e3937ce97423f4c7092ee47&th=1&psc=1&tag=googshopuk-21&linkCode=df0&hvadid=697265600136&hvpos=&hvnetw=g&hvrand=9286832652685731556&hvpone=&hvptwo=&hvqmt=&hvdev=c&hvdvcmdl=&hvlocint=&hvlocphy=9045844&hvtargid=pla-582150399259&psc=1&gad_source=1) which among other inputs helped me understand and build this library.


## Hardware optimisatrion 
To further improve upon this library utilisation of a GPU's computational parallelism properties. With the spirit of doing everyone form *scratch*, I have looked at [python cuda](https://github.com/NVIDIA/cuda-python) which is what libraries such as [numba](https://numba.pydata.org/) use under the hood. 

