# AdaptEllipticalSliceSampler

[![Build Status](https://github.com/ndmarco/AdaptEllipticalSliceSampler.jl/actions/workflows/ci.yml/badge.svg?branch=main)](https://github.com/ndmarco/AdaptEllipticalSliceSampler.jl/actions/workflows/ci.yml?query=branch%3Amain)
[![](https://img.shields.io/badge/docs-dev-blue.svg)](https://ndmarco.github.io/AdaptEllipticalSliceSampler.jl/dev)
[![arXiv](https://img.shields.io/badge/arXiv2605.21659-b31b1b.svg)](https://arxiv.org/abs/2605.21659)


This package contains a Julia implementation of adaptive generalized elliptical slice sampling. 
Adaptive generalized elliptical sliced sampling (AGESS) facilitates Bayesian computation on a wide 
variety of (lower semi-continuous) target distributions. Specifically, we have illustrated 
the utility of AGESS across target distributions that are non-differentiable, non-elliptical,
multi-modal, high-dimensional, and\or are constrained to an open subset of $\mathbb{R}^{P}$. 
Using AGESS to sample from the posterior of your own models is relatively simple and can be 
broken down into the following steps:
1. Install Julia and the `AdaptEllipticalSliceSampler.jl` package (See [Installation](https://ndmarco.github.io/AdaptEllipticalSliceSampler.jl/dev/assets/install/) page)
2. Write a Julia function that efficiently evaluates the log posterior density (See [Performance Tips](https://ndmarco.github.io/AdaptEllipticalSliceSampler.jl/dev/assets/Performance/) and [Tutorials](https://ndmarco.github.io/AdaptEllipticalSliceSampler.jl/dev/assets/Regression/) pages)
3. Call the `AGESS` function (See [Tutorials](https://ndmarco.github.io/AdaptEllipticalSliceSampler.jl/dev/assets/Regression/) pages)

### References

If you found this package useful in your own work and want to cite it in a paper, please consider
using the following suggested citation: 

**N. Marco and S. T. Tokdar. Adaptive generalized elliptical slice sampling. arXiv preprint arXiv:2605.21659, 2026.**

Link to [Paper](https://arxiv.org/abs/2605.21659).
