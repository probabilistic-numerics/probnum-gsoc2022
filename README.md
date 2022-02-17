
<div align="center">
    <a href="https://probnum.readthedocs.io"><img align="center" src="https://raw.githubusercontent.com/probabilistic-numerics/probnum/main/docs/source/assets/img/logo/probnum_logo_dark_txtright.svg" alt="probabilistic numerics" width="600" style="padding-right: 10px; padding left: 10px;" title="Probabilistic Numerics in Python"/>
    </a>
</div>
<br>

# ProbNum &#10005; GSoC 2022

Materials for Google Summer of Code 2022 participants interested in developing for [ProbNum](http://probnum.org).


- [**Code Repository**](https://github.com/probabilistic-numerics/probnum)
- [**Tutorials**](https://probnum.readthedocs.io/en/latest/tutorials.html)
- [**Documentation**](https://probnum.readthedocs.io/en/latest/api.html)
- [**Contribution Guides**](https://probnum.readthedocs.io/en/latest/development.html)


# Project 1: Differentiable Solvers for Ordinary Differential Equations

## Description

Ordinary differential equations (ODEs) are an important model in the natural sciences to describe dynamical processes such as Newton's laws of motion. In practice ODEs are solved using numerical methods based on discretization of the problem. ProbNum offers [probabilistic solvers](https://probnum.readthedocs.io/en/latest/api/automod/probnum.diffeq.probsolve_ivp.html#probnum.diffeq.probsolve_ivp) for ODEs with a given initial value, which quantify the uncertainty incurred by discretization. See [ProbNum's tutorials](https://probnum.readthedocs.io/en/latest/tutorials.html#ordinary-differential-equation-solvers) on ODE solvers for an example.

Multiple desirable functionalities such as improved uncertainty quantification, parameter inference and step size selection only become possible when derivatives of the functions defining the ODE are known. Automatic differentiation frameworks, such as JAX, enable a user to simply define a function in code and then compute gradients with respect to its parameters.

The goal of this project is to implement the probabilistic ODE solver in ProbNum's automatic differentiation backend to enable the described functionality above without having to manually define derivatives.

### Stretch Goal
Implement a boundary value problem solver based on [this paper](https://arxiv.org/abs/2106.07761) in analogy to [`scipy.solve_bvp`](https://docs.scipy.org/doc/scipy/reference/generated/scipy.integrate.solve_bvp.html).

### Requirements
- good Python coding skills
- familiarity with an automatic differentiation framework (ideally JAX)
- basic interest or knowledge in numerical analysis

### Project Size
175 / 350 hours

### Mentor(s)
- Primary: Nicholas Krämer
- Secondary: Jonathan Schmidt, Jonathan Wenger

### Contact
nicholas.kraemer(at)uni-tuebingen.de


# Project 2: Batched Random Variables for a Differentiable Random Process Implementation

Random variables and processes are the fundamental objects in ProbNum representing uncertainty over values and functions, respectively. Every probabilistic numerical method outputs either a random variable or process describing the quantity of interest, such as the solution to a differential equation. In practice, often one needs a batch of random variables which are assumed or known to be independent. This can speed up and simplify implementations considerably (e.g. when computing marginal distributions or evaluating random processes).

The goal of this project is to extend the current implementation of `RandomVariable` and `RandomProcess` to allow for batched `RandomVariable`s. This should be done using ProbNum's automatic differentiation backend (with a particular focus on JAX) to allow parameters of distributions to be fitted when conditioning on data. An example of this is hyperparameter optimization in Gaussian processes.


### Requirements
- good Python coding skills
- familiarity with an automatic differentiation framework (ideally JAX)
- basic interest or knowledge in probability theory

### Project Size
175 / 350 hours

### Mentor(s)
- Primary: Marvin Pförtner
- Secondary: Jonathan Wenger, Nicholas Krämer

### Contact
marvin.pfoertner(at)uni-tuebingen.de


# Ideas

## Batched Random Variables and Sampling in Random Processes
have batched RVs implemented, add them to the GP’s `__call__` method, and, if there’s time, have the intern implement path sampling (maybe via https://arxiv.org/pdf/2002.09309.pdf)

## Differentiable, backend-independent ODE solvers. 
Filters/smoothers as well, ideally. (self-explanatory, I guess.) The only issue I could see with this project is that if the backend is done soon, I might have a crack at much of that myself already. But it should be a reasonable project.

## BVP solvers and state-space optimisers. 
Basically, implement https://arxiv.org/abs/2106.07761. This would make a counterpart of https://docs.scipy.org/doc/scipy/reference/generated/scipy.integrate.solve_bvp.html. The very basics should be pretty simple to build (all parts are there), but getting it right will take the student a bit, I guess. Perhaps easy to combine with https://github.com/probabilistic-numerics/probnum/issues/248.

## Implementing perturbed-state solvers
and include some fun goodies for the ODE world, like PI(D)-control, fancier event handling, etc..

## Linear Algebra / Galerkin methods
All of the below should be done in the autodiff backend:

- implement commonly used preconditioners
  - Jacobi
  - incomplete Cholesky
  - SPAI
  - for GPs (Nyström, RFF, QFF, NNGP?)
- implement sparse linear operators
- benchmark the probabilistic linear solver and implement a lazy version