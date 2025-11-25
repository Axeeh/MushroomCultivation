# 1.1 Probabilistic model

We observe data from\
- (N = 4) temperature levels\
- For each level (i):\
- storage temperature (x_i)\
- number of mushrooms (n_i)\
- number spoiled (y_i)

## Latent spoilage probability

Each mushroom in group (i) spoils independently with probability

\[ p_i = `\sigma`{=tex}(`\alpha `{=tex}+ `\beta `{=tex}x_i),
`\qquad `{=tex} `\sigma`{=tex}(z) = `\frac{1}{1 + e^{-z}}`{=tex}. \]

Thus (p_i) is not observed, but determined by parameters
(`\theta `{=tex}= (`\alpha`{=tex}, `\beta`{=tex})).

## Likelihood for each group

\[ P(y_i `\mid `{=tex}`\theta`{=tex}) = `\binom{n_i}{y_i}`{=tex}
p_i\^{y_i} (1 - p_i)\^{n_i - y_i}. \]

## Independence across groups

\[ P(y `\mid `{=tex}`\theta`{=tex}) = `\prod`{=tex}\_{i=1}\^{N}
`\binom{n_i}{y_i}`{=tex} p_i\^{,y_i} (1-p_i)\^{,n_i - y_i},
`\quad`{=tex} p_i = `\sigma`{=tex}(`\alpha `{=tex}+ `\beta `{=tex}x_i).
\]

## Prior on parameters

\[ `\alpha `{=tex}`\sim `{=tex}`\mathcal{N}`{=tex}(0,4), `\qquad`{=tex}
`\beta `{=tex}`\sim `{=tex}`\mathcal{N}`{=tex}(0,1). \]

## Full probabilistic model

\[ P(y, `\theta`{=tex}) = P(`\theta`{=tex}) `\prod`{=tex}\_{i=1}\^{N}
`\binom{n_i}{y_i}`{=tex} \[`\sigma`{=tex}(`\alpha `{=tex}+
`\beta `{=tex}x_i)\]\^{y_i} \[1 - `\sigma`{=tex}(`\alpha `{=tex}+
`\beta `{=tex}x_i)\]\^{n_i - y_i}. \]

# 1.2 Maximum Likelihood Estimation

## Likelihood

\[ `\mathcal{L}`{=tex}(`\theta`{=tex}) = `\prod`{=tex}\_{i=1}\^{N}
`\binom{n_i}{y_i}`{=tex} p_i\^{y_i} (1-p_i)\^{n_i - y_i}. \]

## Log-likelihood

\[ `\ell`{=tex}(`\theta`{=tex}) = `\sum`{=tex}\_{i=1}\^{N} \[ y_i
`\log `{=tex}`\sigma`{=tex}(`\alpha`{=tex}+`\beta `{=tex}x_i) + (n_i -
y_i)`\log`{=tex}(1 -
`\sigma`{=tex}(`\alpha`{=tex}+`\beta `{=tex}x_i))\] +
`\text{const}`{=tex}. \]
