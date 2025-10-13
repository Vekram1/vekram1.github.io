---
title: "Portfolio Optimization using a Reinforcement Learning based Iterative Solver – FGMRES"
subtitle: "My attempt to explain, replicate, and evaluate \"Accelerated Portfolio Optimization And Option Pricing With Reinforcement Learning\" by Hadi Keramati and Samaneh Jazayeri"
author: "Vikram Oddiraju"
date: 2025-08-24
url: /portfolio_optimization_using_rl_based_iterative_solver/
summary: "A deep dive into using FGMRES, an iterative solver enhanced with reinforcement learning, for efficient portfolio optimization."
tags: ["reinforcement learning", "iterative solvers", "portfolio optimization", "GMRES", "FGMRES"]
categories: ["Iterative Solvers", "Reinforcement Learning", "Portfolio Optimization"]
cover:
  image: ""
  alt: "Disaggregated Price Screener Illustration"
  relative: true
editPost:
    URL: "https://substack.com/home/post/p-171657494"
    Text: "Substack Version"
draft: false

---

## Motivation

Something as trivial as solving **x in Ax = b** helps portfolio managers, economists, and traders execute optimized portfolios, test hypothesized financial models, and get real-time option pricing information. As one could imagine, solving for *x* has far-reaching uses in computer graphics, machine learning, fluid dynamics, and even traffic flow.

Just so we’re on the same page — when I write **Ax = b**, I don’t mean solving a single linear equation like 3x = 6. Instead, I am interested in solving a *system of linear equations* where:

- **A** is an *n × n* matrix  
- **b** is an *n × 1* vector  
- We solve for **x**, which has the same shape as **b**

Inverting A or using direct methods (like Gaussian elimination) gives precise results but is computationally expensive — especially as *n* gets large (imagine portfolios with thousands of assets).

Instead, we rely on **iterative methods**, which reduce the residual **r = Ax – b** over multiple steps.

Common iterative methods include the Jacobi method, Gauss-Seidel, and gradient descent.  
This piece focuses on the **Generalized Minimal Residual Method (GMRES)** and its variant, **Flexible GMRES (FGMRES)**.

---

## Mean-Variance Portfolio Optimization Problem Setup

Below represents the constraint-based optimization that we are trying to solve:

\[
\begin{align}
&\min_{x} \frac{1}{2}x^{T} \Sigma x \\
\text{s.t } &\mu^{T}x = R_{target} \\
&e^{T}x =1
\end{align}
\]

We are trying to choose a vector **x**, representing our asset allocation, that minimizes the variance of the portfolio.

Σ is an n by n covariance matrix of asset returns. The expression ½ xᵀ Σ x represents the variance of the portfolio as a whole, and we would like to minimize that.

e is a vector of all ones, so the condition eᵀ x = 1 just means that the weights we assign to the assets in our portfolio must add up to 1.

μ is a vector of expected returns, so the condition μᵀ x = R_target says that the dot product of the expected returns with our chosen portfolio weights must equal our portfolio’s target return.

To solve this optimization problem, we usually use the Lagrangian method. I won’t go through all the steps here since they’re not so important in the big picture, but the result looks like a system of equations that we can write as A y = b, where:
\[
A = 
\begin{pmatrix} 
\Sigma & e & \mu \\ 
e^T & 0 & 0 \\ 
\mu^{T} & 0 & 0
\end{pmatrix}, \quad
y = 
\begin{pmatrix} 
x \\ 
\lambda_1 \\ 
\lambda_2 
\end{pmatrix}, \quad
b = 
\begin{pmatrix} 
0 \\ 
1 \\ 
R_{target} 
\end{pmatrix}
\]

In our portfolio optimization, using an iterative solver, our goal is to solve for y, which will also yield us our correct portfolio allocation x.

---

## Understanding GMRES

Before diving into FGMRES, we need to understand **GMRES** — the foundation it builds upon.

### Goal

We want to solve the linear system **A x = b**, where:

- A is an *m×m* invertible matrix.
- b ∈ Rᵐ, with ‖b‖ = 1 (for simplicity).

### Step-by-Step Overview

#### 1. Initial Guess and Residual
Start with an initial guess **x₀ ≠ 0**  
Compute the initial residual:  
**r₀ = b – A x₀**

#### 2. Construct the Krylov Subspace
Iteratively build the Krylov subspace:

\[
K_n(A, r_0) = \text{span}\{r_0, Ar_0, A^2r_0, \ldots, A^{n-1}r_0\}
\]

The solution lies in:
\[
x_n \in x_0 + K_n
\]

#### 3. Issue: Near Linear Dependence
The vectors {r₀, Ar₀, …, Aⁿ⁻¹r₀} can become nearly linearly dependent, degrading the subspace quality.

#### 4. Solution: Arnoldi Process
The **Arnoldi iteration** constructs an orthonormal basis Qₙ = [q₁, q₂, …, qₙ] and an upper Hessenberg matrix Ĥₙ:

\[
A Q_n = Q_{n+1} \tilde{H}_n
\]

#### 5. Rewriting the Solution Form
Since xₙ ∈ x₀ + Kₙ:

\[
x_n = x_0 + Q_n y_n
\]

Minimize residual ‖b – A xₙ‖, equivalent to:

\[
\min_{y_n} \| \tilde{H}_n y_n - \beta e_1 \|_2
\]

Once solved for yₙ, substitute back into **xₙ = x₀ + Qₙ yₙ** to approximate the solution.

GMRES effectively reduces a large *m×m* system into smaller least squares problems of size *(n+1)×n*, where *n* ≪ *m*.

---

## FGMRES – The Iterative Method Our RL Agent Will Use

FGMRES (Flexible GMRES) extends GMRES by allowing **different preconditioners** at each iteration.

### What is a Preconditioner?

A preconditioner **M** is a matrix that approximates **A** but is easier to invert.  
We solve **M⁻¹A x = M⁻¹b**, improving the conditioning of A and speeding convergence.

- **M = I** → trivial, no improvement  
- **M = A** → perfect convergence, but as costly as solving A⁻¹ directly  

The art lies in finding a **cheap but effective** M.

---

### QR as a Preconditioner

Using **QR decomposition**, the system becomes:

\[
Qᵀ A x = Qᵀ b \Rightarrow R x = Qᵀ b
\]

QR is robust to ill-conditioning and efficient on smaller blocks, though full QR of A is impractical.

---

### GMRES with Preconditioning

Without preconditioning:
\[
K_n(A,r_0) = \text{span}\{r_0, Ar_0, A^2r_0, \ldots, A^{n-1}r_0 \}
\]

With preconditioning:
\[
C := M^{-1}A, \quad z_0 := M^{-1}r_0
\]
\[
K_n(C,z_0) = \text{span}\{z_0, Cz_0, C^2z_0, \ldots, C^{n-1}z_0 \}
\]

#### Pseudocode
In psuedocode, FGMRES looks like this:

```python
Input: Matrix A ∈ R^(m×m), vector b ∈ R^m, restart parameter k
Output: Approximate solution x to A x = b

1. Choose initial guess x₀
2. r₀ = b – A x₀
3. β = ||r₀||
4. v₁ = r₀ / β
5. V = [v₁]                  # Orthonormal basis vectors
6. H = zeros((k+1, k))       # Upper Hessenberg matrix

7. For j = 1 to k:
       w = A vⱼ
       For i = 1 to j:        # Arnoldi orthogonalization
            hᵢⱼ = ⟨w, vᵢ⟩     # Take dot product
            w = w – hᵢⱼ vᵢ    # Orthonormalize against existing basis
       End
       hⱼ₊₁,ⱼ = ||w||         
       vⱼ₊₁ = w / hⱼ₊₁,ⱼ
       Append vⱼ₊₁ to V

8. Solve the least squares problem:
       minimize || β e₁ – H y ||
   for y ∈ R^k

9. Compute solution update:
       x = x₀ + V y    
```

Instead, if we apply a preconditioner, the Arnoldi process builds this Krylov subspace as such, where C and z_0 are:

