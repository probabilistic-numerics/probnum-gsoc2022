
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


# Project(s)

Primary mentor: 
Secondary mentors: 

## Batched Random Variables and Sampling in Random Processes
have batched RVs implemented, add them to the GP’s __call__ method, and, if there’s time, have the intern implement path sampling (maybe via https://arxiv.org/pdf/2002.09309.pdf)

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