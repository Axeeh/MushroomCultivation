# Probabilistic Model & Maximum Likelihood Estimation

This document presents a **probabilistic model** that estimates the probability of spoilage in mushrooms given a storage temperature and describes how **Maximum Likelihood Estimation (MLE)** can be used for fitting the parameters. The model uses a logistic regression structure with binomial observations.

---

## 1.1 Probabilistic Model

We are presented with data from \(N = 4\) levels of temperature. At each level \(i\), we have:

* **Storage temperature** \(x_i\) (in °C)
* **Number of mushrooms** \(n_i\)
* **Number of spoiled mushrooms** \(y_i\)

### Latent Spoilage Probability

Every mushroom in group \(i\) spoils independently with probability \(p_i\). This probability is modeled with the **sigmoid (or logistic) function**:

\[
p_i = \sigma(\alpha + \beta x_i), \quad \text{where} \quad \sigma(z) = \frac{1}{1 + e^{-z}}.
\]

Here, \(\theta = (\alpha, \beta)\) are the unknown **parameters** to be estimated.

### Likelihood for Each Group

Given the parameters \(\theta\), the observed number of spoiled mushrooms \(y_i\) follows a **Binomial distribution**:

\[
y_i \mid \theta \sim \text{Binomial}(n_i, p_i)
\]

The probability mass function for one group is:

\[
P(y_i \mid \theta) = \binom{n_i}{y_i}\, p_i^{\,y_i} \, (1 - p_i)^{\,n_i - y_i}.
\]

### Independence Across Groups

Assuming **independence** across all \(N\) groups (given \(\theta\)), the **full-data likelihood** is the product of the individual likelihoods:

\[
P(y \mid \theta) = \prod_{i=1}^{N} \binom{n_i}{y_i}\, p_i^{\,y_i} (1 - p_i)^{\,n_i - y_i}, \quad \text{where} \quad p_i = \sigma(\alpha + \beta x_i).
\]

### Prior on Parameters

We place **Gaussian priors** on the parameters \(\alpha\) and \(\beta\) in a Bayesian framework:

\[
\alpha \sim \mathcal{N}(0, 4), \quad \beta \sim \mathcal{N}(0, 1).
\]

Assuming that \(\alpha\) and \(\beta\) are independent, the joint prior probability is:

\[
P(\theta) = \mathcal{N}(\alpha; 0, 4)\,\mathcal{N}(\beta; 0, 1).
\]

### Full Generative Model

Putting the likelihood and prior together defines the **full generative model**:

\[
P(y, \theta) = P(\theta) \prod_{i=1}^{N} \binom{n_i}{y_i} [\sigma(\alpha + \beta x_i)]^{\,y_i} [1 - \sigma(\alpha + \beta x_i)]^{\,n_i - y_i}.
\]