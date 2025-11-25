# Probabilistic Model & Maximum Likelihood Estimation

## 1.1 Probabilistic Model

We observe data for \(N = 4\) temperature levels. For each level \(i\), we have:

- Storage temperature \(x_i\) (in °C)  
- Number of mushrooms \(n_i\)  
- Number of spoiled mushrooms \(y_i\)

### Latent Spoilage Probability

Each mushroom in group \(i\) spoils independently with probability:

\[
p_i = \sigma(\alpha + \beta x_i),
\qquad
\sigma(z) = \frac{1}{1 + e^{-z}}.
\]

Here, \(\theta = (\alpha, \beta)\) are parameters.

### Likelihood for Each Group

Given \(\theta\), \(y_i\) (the number of spoiled mushrooms in group \(i\)) follows:

\[
y_i \mid \theta \sim \text{Binomial}(n_i, p_i)
\]

so that

\[
P(y_i \mid \theta)
= \binom{n_i}{y_i}\, p_i^{\,y_i} \, (1 - p_i)^{\,n_i - y_i}.
\]

### Independence Across Groups

Assuming independence across groups (given \(\theta\)), the full-data likelihood is:

\[
P(y \mid \theta)
= \prod_{i=1}^{N}
\binom{n_i}{y_i}
\cdot p_i^{\,y_i} \cdot (1 - p_i)^{\,n_i - y_i},
\quad
p_i = \sigma(\alpha + \beta x_i).
\]

### Prior on Parameters

We place Gaussian priors on \(\alpha\) and \(\beta\):

\[
\alpha \sim \mathcal{N}(0, 4),
\qquad
\beta \sim \mathcal{N}(0, 1).
\]

Assuming independence:

\[
P(\theta)
= \mathcal{N}(\alpha; 0, 4)\;\mathcal{N}(\beta; 0, 1).
\]

### Full Generative Model

Putting everything together:

\[
P(y, \theta)
= P(\theta)
\cdot \prod_{i=1}^{N}
\binom{n_i}{y_i}
\; [\sigma(\alpha + \beta x_i)]^{\,y_i}
\; [1 - \sigma(\alpha + \beta x_i)]^{\,n_i - y_i}.
\]

---

## 1.2 Maximum Likelihood Estimation

### Likelihood Function

Ignoring the prior, the likelihood function is:

\[
\mathcal{L}(\theta)
= \prod_{i=1}^{N}
\binom{n_i}{y_i}
\; p_i^{\,y_i} \; (1 - p_i)^{\,n_i - y_i},
\quad
p_i = \sigma(\alpha + \beta x_i).
\]

### Log-Likelihood

Taking logs gives the log-likelihood:

\[
\ell(\theta)
= \sum_{i=1}^{N}
\left[
y_i \cdot \log \sigma(\alpha + \beta x_i)
+
(n_i - y_i) \cdot \log\bigl(1 - \sigma(\alpha + \beta x_i)\bigr)
\right]
+ \text{constant}.
\]

This is equivalent to the log-likelihood in logistic regression with binomial observations.

---

