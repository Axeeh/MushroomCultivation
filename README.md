# Mushroom Cultivation Analysis  
**Probabilistic Modeling, Inference, and Bayesian Estimation**

This project analyzes mushroom spoilage as a function of storage temperature using a **probabilistic logistic regression model** with **binomial observations**. Multiple inference techniques are implemented and compared, including **Maximum Likelihood (ML)**, **Maximum A Posteriori (MAP)**, **grid-based Bayesian inference**, and **Metropolis MCMC sampling**.

The goal is to estimate how temperature affects spoilage probability and to validate inference consistency across deterministic and stochastic methods.

---

## 📊 Dataset

We observe mushroom spoilage at four storage temperatures:

| Temperature (°C) | Total Mushrooms | Spoiled |
|------------------|-----------------|---------|
| 2                | 30              | 2       |
| 8                | 25              | 4       |
| 15               | 20              | 5       |
| 25               | 30              | 20      |

---

## 🧠 Probabilistic Model

Each mushroom spoils independently with probability:

$$
p_i = \sigma(\alpha + \beta x_i)
\quad \text{where} \quad
\sigma(z) = \frac{1}{1 + e^{-z}}
$$

### Likelihood
$$
y_i \mid \theta \sim \text{Binomial}(n_i, p_i)
$$

### Prior (Bayesian setting)
- $\alpha \sim \mathcal{N}(0, 4)$ 
- $\beta \sim \mathcal{N}(0, 1)$

The full model combines a binomial likelihood with Gaussian priors on parameters.

---

## 📁 Project Structure

```text
.
├── functions.py              # Likelihood, prior, posterior utilities
├── notebook.ipynb / .pdf     # Full analysis, plots, and results
├── probabilistic_model.pdf   # Mathematical model derivation
├── README.md                 # Project documentation