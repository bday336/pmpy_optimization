# Numerical Toolkit to Analysis Optimal Control Problem of Push-Me-Pull-You (PMPY) Microswimmer System

Source code of numerical optimization toolkit for systems possessing a configuration space with a fiber bundle structure. This is an implementation of the process described by Koon and Marsden in [Optimal Control for Holonomic and Nonholonomic Mechanical Systems with Symmetry and Lagrangian Reduction](http://www.cds.caltech.edu/~koon/papers/optimalKM.pdf). Here the question of optimal control is adapted to a variational approach which resembles solving a constrained dynamics problem.

This instance of the toolkit is designed for investigating the particular deformable control system characterizing the process of Low Reynolds Number locomotion performed by microorganisms like the Euglena. This involves cyclic body elongations and expansions known as metaboly.

The PMPY model was developed by Avron et al. [Pushmepullyou: An efficient micro-swimmer](https://arxiv.org/pdf/math-ph/0501049.pdf). We use the energy expenditure to deform as the cost function for our optimization. Given that the PMPY can only move along the path through its symmetry axis, it symmetry group describing its possible motion is abelian and thus approaches like gradient descent are amenable. We utilize both methods here to investigate the question of optimal control with respect to energy expenditure, but aim to highlight the utility of the variational approach.