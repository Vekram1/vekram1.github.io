---
title: "Multilevel Monte Carlo for the Ornstein–Uhlenbeck SDE"
subtitle: "Estimating expectations of SDEs efficiently using Multilevel Monte Carlo, approximate random variables, and mixed floating-point precision"
author: "Vikram Oddiraju"
date: 2025-10-21
url: /multilevel_monte_carlo_ou/
summary: "An exploration of Multilevel Monte Carlo (MLMC) for estimating expectations of stochastic differential equations, using the Ornstein–Uhlenbeck process as a test case. I investigate how approximate random variables and mixed floating-point precision influence computational efficiency and accuracy."
draft: false
---

# Multilevel Monte Carlo For Estimating the Expected Value of Stochastic Differential Equations

## Motivation

A few days ago, I replied to a tweet about stochastic differential equations (SDEs).  
*(Insert screenshot of tweet + reply)*

Quant Beckman’s response was a bit snarky—especially considering it was his own post—but it did get me thinking seriously about the computational challenges behind SDEs. Problems like computing expectations, option payoffs, or tail probabilities can easily become expensive, especially when the SDE has no closed-form solution.

But *computationally expensive* doesn’t mean *hopeless*.  
It just means we need better algorithms.

This led me down the path of **Multilevel Monte Carlo (MLMC)**, a method designed to dramatically speed up the estimation of expectations of SDEs by combining simulations at different time resolutions. Along the way, I also examined how approximate random variables, single vs double precision, and compensated summation (Kahan) affect accuracy.
---

# Background: Multilevel Monte Carlo (MLMC)

## 1. The Problem

Consider the Ornstein–Uhlenbeck (OU) SDE:

$$dx_t = \theta(\mu - x_t)\,dt + \sigma\,dW_t$$

What we can say from this equation is that if we simulated the SDE from time $0$ to $T$, the value $X_T$ is the random state of the process at the terminal time $T$. Due to the stochastic nature of stochastic differential equations, $X_T$ is non determinstic, therefore it is oftentimes a hard problem to figure out how to estimate the expectation of the distribution


$$\mathbb{E}[P(X_T)]$$

where $P(X_T)$ for us is just the distribution of $X_T$. (Oftentimes, for instance, in options pricing $P(x) = max(x-K,0)$.

For many SDEs (including OU with general payoffs), computing this expectation analytically is difficult or impossible. So we turn to **numerical simulation**.

A simple and widely used scheme is the **Euler–Maruyama method**.

---

## 1.1 Discretizing the OU SDE with Euler–Maruyama

Given


$$dx_t = \theta(\mu - x_t)dt + \sigma\,dW_t, \qquad x(0)=x_0$$

integrate over the interval \([t_n, t_{n+1}]\) with step size \(\Delta t\):

$$
x_{n+1} - x_n
= \int_{t_n}^{t_{n+1}} \theta(\mu - x_s)\,ds
+ \int_{t_n}^{t_{n+1}} \sigma\,dW_s.
$$

Euler–Maruyama approximates these integrals using:

- drift evaluated at the left endpoint:  
  $$\theta(\mu - x_n)\Delta t\$$

- Brownian increment:  
  $$\Delta W_n \sim \mathcal{N}(0,\Delta t)$$

giving the update

$$
\boxed{
x_{n+1} = x_n + \theta(\mu - x_n)\Delta t
        + \sigma\sqrt{\Delta t}\,Z_n,
\qquad Z_n \sim \mathcal{N}(0,1).
}
$$

This produces a discrete approximation $P_\ell$ with step size $\Delta t_\ell = T / 2^\ell$.

But refining the time step improves accuracy only at the cost of drastically more computation.

This is where MLMC enters.

---

## 2. The MLMC Idea

Instead of running a huge Monte Carlo simulation at the finest step size, MLMC decomposes the expectation using a **telescoping sum**:

$$
\mathbb{E}[P_L]
=
\mathbb{E}[P_0]
+
\sum_{\ell=1}^{L}
\mathbb{E}[P_\ell - P_{\ell-1}],
$$

where:

- $(P_\ell)$ is the payoff computed using time step $(\Delta t_\ell = T / 2^\ell)$,
- $(P_\ell - P_{\ell-1})$ is a **level correction**, computed by coupling coarse and fine simulations using the same random normal increments.

The key insight:

> Coarse levels are cheap but inaccurate.  
> Fine levels are expensive but accurate.  
> MLMC mixes many cheap coarse samples with few expensive fine samples—  
> **reducing total variance at minimal cost.**

---

## 3. What Quantities MLMC Needs

To carry out the MLMC optimization, we need just three quantities per level $(\ell)$:

1. **Variance**  
   $$V_\ell = \mathrm{Var}(P_\ell - P_{\ell-1})$$
   → Determines how many samples we should take at each level.

2. **Cost**  
   $$C_\ell \propto 2^\ell$$
   → Number of time steps per sample.

3. **Bias estimate**  
   
   $$B_L = |\mathbb{E}[P_L - P_{L-1}]|$$
   → Determines how deep the hierarchy must extend.

These three numbers feed directly into the MLMC formula for the **optimal number of samples per level**.

