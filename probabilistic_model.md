# Probabilistic Model & Maximum Likelihood Estimation

This document outlines a **probabilistic model** to estimate the spoilage probability of mushrooms based on storage temperature, followed by a description of the **Maximum Likelihood Estimation (MLE)** approach for parameter fitting. The model uses a **logistic regression** structure with **binomial observations**.

---

## 1.1 Probabilistic Model

We observe data for $N = 4$ temperature levels. For each level $i$, we have:

* **Storage temperature** $x_i$ (in °C)
* **Number of mushrooms** $n_i$
* **Number of spoiled mushrooms** $y_i$

| Level ID | Storage Temperature $x$ (°C) | Total Mushrooms $n$ | Spoiled Mushrooms $y$ |
|----------|-------------------------------|----------------------|-----------------------|
| 1        | 2                             | 30                   | 2                     |
| 2        | 8                             | 25                   | 4                     |
| 3        | 15                            | 20                   | 5                     |
| 4        | 25                            | 30                   | 20                    |

### Latent Spoilage Probability

Each mushroom in group $i$ spoils independently with probability $p_i$. This probability is modeled using the **sigmoid (or logistic) function**:

$$
p_i = \sigma(\alpha + \beta x_i),
\quad \text{where} \quad
\sigma(z) = \frac{1}{1 + e^{-z}}.
$$

Here, $\theta = (\alpha, \beta)$ are the unknown **parameters** to be estimated.

### Likelihood for Each Group

Given the parameters $\theta$, the observed number of spoiled mushrooms $y_i$ follows a **Binomial distribution**:

$$
y_i \mid \theta \sim \text{Binomial}(n_i, p_i)
$$

The probability mass function for a single group is:

$$
P(y_i \mid \theta)
= \binom{n_i}{y_i}\, p_i^{\,y_i} \, (1 - p_i)^{\,n_i - y_i}.
$$

### Independence Across Groups

Assuming **independence** across all $N$ groups (given $\theta$), the **full-data likelihood** is the product of the individual likelihoods:

$$
P(y \mid \theta)
= \prod_{i=1}^{N}
\binom{n_i}{y_i}
p_i^{\,y_i} (1 - p_i)^{\,n_i - y_i},
\quad
\text{where} \quad
p_i = \sigma(\alpha + \beta x_i).
$$

### Prior on Parameters

In a Bayesian setting, we place **Gaussian priors** on the parameters $\alpha$ and $\beta$:

$$
\alpha \sim \mathcal{N}(0, 4),
\qquad
\beta \sim \mathcal{N}(0, 1).
$$

Assuming independence of $\alpha$ and $\beta$, the joint prior probability is:

$$
P(\theta)
= \mathcal{N}(\alpha; 0, 4)\;\mathcal{N}(\beta; 0, 1).
$$

### Full Generative Model

Putting the likelihood and prior together defines the **full generative model**:

$$
P(y, \theta)
= P(\theta)
\prod_{i=1}^{N}
\binom{n_i}{y_i}
[\sigma(\alpha + \beta x_i)]^{\,y_i}
[1 - \sigma(\alpha + \beta x_i)]^{\,n_i - y_i}.
$$