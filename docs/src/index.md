# AdaptEllipticalSliceSampler.jl

This package contains a Julia implementation of adaptive generalized elliptical slice sampling. 
Adaptive generalized elliptical sliced sampling (AGESS) facilitates Bayesian computation on a wide 
variety of (lower semi-continuous) target distributions. Specifically, we have illustrated 
the utility of AGESS across target distributions that are non-differentiable, non-elliptical,
multi-modal, high-dimensional, and\or are constrained to an open subset of $\mathbb{R}^{P}$. 
Using AGESS to sample from the posterior of your own models is relatively simple and can be 
broken down into the following steps:
1. Install Julia and the `AdaptEllipticalSliceSampler.jl` package (See **Installation** page)
2. Write a Julia function that efficiently evaluates the posterior log pdf (See **Performance Tips** and **Tutorials** pages)
3. Call the `AGESS` function (See **Tutorials** pages)

### References

If you found this package useful in your own work and want to cite it in a paper, please consider
using the following suggested citation: 

**N. Marco and S. T. Tokdar. Adaptive Generalized Elliptical Slice Sampling.**