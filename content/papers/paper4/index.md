---
title: "Multilevel Monte Carlo Ornstein-Uhlenbeck for estimation of expected values in SDEs"
subtitle: "How can we obtain the expectation of a stochastic differential equation efficiently using Multilevel Monte Carlo methods along with using approximate random variables and different floating point precisions?"
author: "Vikram Oddiraju"
date: 2025-10-21
url: /multi_level_monte_carlo_ornstein_uhlenbeck/
summary: "A deep dive into using Multilevel Monte Carlo methods for estimating expected values in Stochastic Differential Equations (SDEs), focusing on the Ornstein-Uhlenbeck process. This article explores the efficiency of combining approximate random variables and varying floating point precisions to enhance computational performance."
tags: [""]
categories: [""]
cover:
  image: ""
  alt: ""
  relative: true
editPost:
    URL: ""
    Text: ""
draft: true



---

<!-- Link to research paper: **Accelerated Portfolio Optimization And Option Pricing With Reinforcement Learning by Hadi Keramati and Samaneh Jazayeri:**    [https://arxiv.org/pdf/2507.01972v1](https://arxiv.org/pdf/2507.01972v1)

Link to my code: [https://github.com/Vekram1/ill_conditioned_ppo_RL_matrix_solver](https://github.com/Vekram1/ill_conditioned_ppo_RL_matrix_solver) -->

## Motivation

The other day, I posted a reply to a tweet about stochastic differential equations (SDEs). 
<Image of tweet and reply>
Quant Beckman's reply, I thought, was a bit snarky given the fact that is was his post, but it definenitely made me think about the problems associated with using SDEs in any field. What I found was that operations like trying to find the expectation or finding tail probabilities of SDEs can be computationally expensive, particularly on analytically intractable SDEs. However, just becasue something is computationally expensive, doesn't mean we can't find ways to make it more efficient, and that we should just leave it under the rug. 