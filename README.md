
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


# Project 1: Differentiable Probabilistic Solvers for Differential Equations

## Description

Ordinary differential equations (ODEs) are an important model in the natural sciences to describe dynamical processes such as Newton's laws of motion. In practice, ODEs are solved using numerical methods based on discretization of the problem. ProbNum offers [probabilistic solvers](https://probnum.readthedocs.io/en/latest/api/automod/probnum.diffeq.html#probnum.diffeq.probsolve_ivp) for ODEs, which quantify the uncertainty associated with the discretization error. See [ProbNum's tutorials](https://probnum.readthedocs.io/en/latest/tutorials.html#ordinary-differential-equation-solvers) on ODE solvers for an example.

Multiple desirable functionalities, for example, model selection through ODE parameter inference, require derivatives of ODE solutions.
Automatic differentiation frameworks, such as JAX, enable a user to straightforwardly compute gradients of ODE solutions with respect to ODE parameters. The goal of this project is to port the probabilistic ODE solver to ProbNum's automatic differentiation backend.

### Stretch Goal(s)
Many physical applications not only involve temporal dynamics but also spatial dynamics, for example heat flow. Such phenomena are typically modelled by partial differential equations (PDEs). One can solve such problems via discretization, resulting in (typically very large) linear systems. Computing solutions to such discretized PDEs with calibrated uncertainty requires differentiable linear solvers. The stretch goal of this project is the implementation of an efficient and scalable version of such a solver.

### Requirements
- good Python coding skills
- familiarity with an automatic differentiation framework (ideally JAX)
- basic interest in or knowledge of numerical analysis and/or Kalman filtering

### Project Size
175 / 350 hours

### Mentor(s)
- Primary: [Nicholas Krämer](https://github.com/pnkraemer/)
- Secondary: [Jonathan Wenger](https://github.com/JonathanWenger/), [Jonathan Schmidt](https://github.com/schmidtjonathan/)

### Contact
nicholas.kraemer(at)uni-tuebingen.de


# Project 2: Batched Random Variables for a Differentiable Random Process Implementation

Random variables and processes are the fundamental objects in ProbNum representing uncertainty over values and functions, respectively. Every probabilistic numerical method outputs either a random variable or process describing the quantity of interest, such as the solution to a differential equation. In practice, often one needs a batch of random variables which are assumed or known to be independent. This can speed up and simplify implementations considerably (e.g. when computing marginal distributions or evaluating random processes).

The goal of this project is to extend the current implementation of `RandomVariable` and `RandomProcess` to allow for batched `RandomVariable`s. This should be done using ProbNum's automatic differentiation backend (with a particular focus on JAX) to allow parameters of distributions to be fitted when conditioning on data. An example of this is hyperparameter optimization in Gaussian processes.

### Stretch Goal(s)
Implement path-wise sampling for Gaussian processes via https://arxiv.org/pdf/2002.09309.pdf.

### Requirements
- good Python coding skills
- in-depth understanding of NumPy‘s broadcasting and vectorization mechanisms
- familiarity with an automatic differentiation framework (ideally JAX)
- basic interest or knowledge in probability theory

### Project Size
175 / 350 hours

### Mentor(s)
- Primary: [Marvin Pförtner](https://github.com/marvinpfoertner)
- Secondary: [Jonathan Wenger](https://github.com/JonathanWenger/), [Nicholas Krämer](https://github.com/pnkraemer/)

### Contact
marvin.pfoertner(at)uni-tuebingen.de