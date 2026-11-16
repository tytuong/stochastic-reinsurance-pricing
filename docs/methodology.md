# Methodology and derivations

This document sets out the mathematics behind the model, including the closed-form results used to validate the simulation. Notation follows the [README](../README.md): $`\lambda`$ is the Poisson frequency, $`X`$ an individual loss, $`R`$ the retention, $`L`$ the limit and $`u = R + L`$ the exhaustion point. $`\Phi`$ is the standard normal CDF and $`x \wedge d = \min(x, d)`$.

---

## 1. Lognormal severity

If $`\ln X \sim N(\mu, \sigma^2)`$, the moments of $`X`$ follow from the normal moment generating function $`\mathbb{E}[e^{tZ}] = e^{t\mu + t^2\sigma^2/2}`$:

```math
\mathbb{E}[X^k] = e^{k\mu + k^2\sigma^2/2}
```

so

```math
\mathbb{E}[X] = e^{\mu + \sigma^2/2}, \qquad \text{CV}^2 = \frac{\text{Var}[X]}{\mathbb{E}[X]^2} = e^{\sigma^2} - 1.
```

Inverting gives the parameterisation used in `config.py`:

```math
\sigma^2 = \ln(1 + \text{CV}^2), \qquad \mu = \ln \mathbb{E}[X] - \tfrac{1}{2}\sigma^2.
```

For the base case (mean £600,000, CV 2.5): $`\sigma = 1.407`$, $`\mu = 12.314`$, and the median loss is $`e^{\mu} \approx £222{,}000`$, well below the mean, which shows how skewed the distribution is.

---

## 2. Compound Poisson aggregate loss

Let $`S = X_1 + \dots + X_N`$ with $`N \sim \text{Poisson}(\lambda)`$ independent of the i.i.d. $`X_i`$. Conditioning on $`N`$:

```math
\mathbb{E}[S] = \mathbb{E}\big[\mathbb{E}[S \mid N]\big] = \mathbb{E}[N]\,\mathbb{E}[X] = \lambda\,\mathbb{E}[X]
```

```math
\text{Var}[S] = \mathbb{E}\big[\text{Var}[S \mid N]\big] + \text{Var}\big[\mathbb{E}[S \mid N]\big]
= \lambda\,\text{Var}[X] + \lambda\,\mathbb{E}[X]^2 = \lambda\,\mathbb{E}[X^2].
```

The same results hold for any function of the individual losses, in particular the per-claim layer loss $`Y`$, which gives $`\mathbb{E}[Z] = \lambda\,\mathbb{E}[Y]`$ and $`\text{Var}[Z] = \lambda\,\mathbb{E}[Y^2]`$ when there is no aggregate limit.

---

## 3. The layer as a difference of limited losses

The per-claim recovery $`Y = \min(\max(X - R, 0), L)`$ can be written as

```math
Y = (X \wedge u) - (X \wedge R).
```

Checking the three cases: if $`X \le R`$ both terms equal $`X`$ and $`Y = 0`$; if $`R < X \le u`$ then $`Y = X - R`$; if $`X > u`$ then $`Y = u - R = L`$. Taking expectations:

```math
\mathbb{E}[Y] = \mathbb{E}[X \wedge u] - \mathbb{E}[X \wedge R].
```

So the expected layer loss only needs the **limited expected value** (LEV) function $`\mathbb{E}[X \wedge d]`$.

---

## 4. Limited expected value of the Lognormal

```math
\mathbb{E}[X \wedge d] = \int_0^d x f(x)\,dx + d\,P(X > d).
```

For the first term, substitute $`x = e^{y}`$ with $`y \sim N(\mu, \sigma^2)`$ and complete the square: the partial first moment of a Lognormal is $`e^{\mu + \sigma^2/2}\,\Phi\big((\ln d - \mu - \sigma^2)/\sigma\big)`$. Hence

```math
\mathbb{E}[X \wedge d] = e^{\mu + \sigma^2/2}\,\Phi\!\left(\frac{\ln d - \mu - \sigma^2}{\sigma}\right) + d\left[1 - \Phi\!\left(\frac{\ln d - \mu}{\sigma}\right)\right].
```

The second limited moment works the same way with $`k = 2`$:

```math
\mathbb{E}[(X \wedge d)^2] = e^{2\mu + 2\sigma^2}\,\Phi\!\left(\frac{\ln d - \mu - 2\sigma^2}{\sigma}\right) + d^2\left[1 - \Phi\!\left(\frac{\ln d - \mu}{\sigma}\right)\right].
```

Both formulas are checked in the tests against direct numerical integration of $`\int_0^d S(x)\,dx`$ and $`\int_0^d 2x\,S(x)\,dx`$.

---

## 5. Second moment of the layer loss

From $`Y = (X \wedge u) - (X \wedge R)`$:

```math
\mathbb{E}[Y^2] = \mathbb{E}[(X \wedge u)^2] - \mathbb{E}[(X \wedge R)^2] - 2R\,\mathbb{E}[Y].
```

Again by cases: if $`X \le R`$ the right side is $`X^2 - X^2 - 0 = 0`$; if $`R < X \le u`$ it is $`X^2 - R^2 - 2R(X - R) = (X - R)^2`$; if $`X > u`$ it is $`u^2 - R^2 - 2R(u - R) = (u - R)^2 = L^2`$. Each matches $`Y^2`$.

---

## 6. Probability the layer is hit

Each loss independently exceeds $`R`$ with probability $`p = P(X > R)`$. Thinning a Poisson process keeps it Poisson, so the number of losses that reach the layer is $`\text{Poisson}(\lambda p)`$ and

```math
P(Z > 0) = 1 - e^{-\lambda p}, \qquad \mathbb{E}[\text{number of layer losses}] = \lambda p.
```

The same argument gives the occurrence exceedance probability: the largest loss in a year exceeds $`x`$ with probability $`1 - e^{-\lambda P(X > x)}`$, which is the OEP curve in the README.

---

## 7. Monte Carlo error

With $`n`$ independent simulated years, the estimate $`\bar Z`$ of $`\mathbb{E}[Z]`$ has standard error $`s_Z / \sqrt{n}`$ and an approximate 95% confidence interval $`\bar Z \pm 1.96\,s_Z/\sqrt{n}`$ by the Central Limit Theorem. In the base case $`s_Z \approx £1.82\text{m}`$, so with $`n = 100{,}000`$ the standard error is about £5,800, or 0.7% of the pure premium.

Tail quantities converge more slowly: the 99.5% VaR depends on only the worst 500 of 100,000 years.

---

## 8. Risk measures and capital

```math
\text{VaR}_q(Z) = \inf\{z : P(Z \le z) \ge q\}, \qquad \text{TVaR}_q(Z) = \mathbb{E}[Z \mid Z \ge \text{VaR}_q(Z)].
```

Capital is set at $`\text{VaR}_{99.5\%}(Z) - \mathbb{E}[Z]`$: the expected loss is already funded by the pure premium, so capital only needs to cover the *unexpected* loss up to the 1-in-200 year level. The reinsurer needs a return of $`\text{CoC}`$ on that capital each year, which gives the risk load $`\text{CoC} \times C`$.

TVaR is reported alongside VaR because it is coherent (sub-additive) and looks at the severity of outcomes beyond the quantile, not only the quantile itself. Switching the capital measure to TVaR would be a one-line change in `pricing.py`.

---

## 9. Reinstatement premium

With $`k`$ reinstatements at rate $`c`$ (pro rata as to amount), each £1 of limit used in the first $`k`$ limits is restored for $`c \times P / L`$, where $`P`$ is the upfront premium. The amount reinstated in a year is $`\min(Z^{\text{uncapped}}, kL)`$, so the expected total premium is

```math
P\left(1 + c\,\frac{\mathbb{E}[\min(Z^{\text{uncapped}}, kL)]}{L}\right).
```

Setting this equal to the gross technical premium and solving for $`P`$ gives the upfront premium. This treats the risk load as spread proportionally across upfront and reinstatement premium, which is the usual simplification.

---

## 10. Common random numbers in the sensitivity analysis

Every sensitivity scenario uses the same seed. Changing $`\mu`$ or $`\sigma`$ then rescales the *same* underlying normal draws, so the difference between scenarios reflects the assumption change rather than independent sampling noise. Without this, the noise in each premium (around ±0.7%) would blur the smaller sensitivities.
