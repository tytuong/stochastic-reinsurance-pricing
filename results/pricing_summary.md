## Pricing summary: GBP 5.0m xs GBP 5.0m (1 reinstatement(s) @ 100%)

Portfolio: Poisson(λ = 25) frequency, Lognormal severity (mean £600,000, CV 2.5). 100,000 simulated years, seed 2026.

### Loss statistics

| Metric | Value |
|---|---:|
| Expected gross annual loss | £14,970,823 |
| Expected annual layer loss (pure premium) | £868,479 |
| Monte Carlo standard error | £5,759 |
| 95% confidence interval | £857,192 to £879,767 |
| Expected number of losses hitting the layer | 0.34 |
| Probability the layer is hit in a year | 28.8% |
| Probability a single loss exhausts the layer | 8.3% |
| Standard deviation of layer loss | £1,821,131 |
| 99.5% VaR of layer loss | £9,488,697 |
| 99.5% TVaR of layer loss | £9,959,371 |

### Premium build-up

| Step | Value |
|---|---:|
| Expected layer loss | £868,479 |
| Capital (99.5% VaR − mean) | £8,620,218 |
| Risk load (10% cost of capital) | £862,022 |
| **Risk-adjusted premium** | **£1,730,501** |
| Gross technical premium (÷ (1 − 5% expenses − 10% brokerage)) | £2,035,883 |
| Expected reinstatement factor | 1.1618 |
| **Upfront premium** | **£1,752,394** |
| Rate on line | 35.05% |
| Payback period | 2.9 years |
| Expected loss ratio | 42.7% |

### Validation against closed-form results (before aggregate limit)

| Quantity | Simulated | Closed form | Difference |
|---|---:|---:|---:|
| Expected layer loss | £871,361 | £868,222 | +0.36% |
| Std dev of layer loss | £1,838,128 | £1,834,752 | +0.18% |
| P(layer is hit) | 28.80% | 28.73% | +0.25% |
| Expected losses hitting layer | 0.3387 | 0.3387 | +0.00% |
| Expected gross loss | £14,970,823 | £15,000,000 | -0.19% |

Expected layer loss differs from the exact value by +0.54 standard errors.

Simulated 100,000 years (2,494,323 individual losses) in 0.10s.

### Layer tower

| Layer | Expected loss | P(hit) | Risk-adjusted premium | Rate on line | Loss ratio |
|---|---:|---:|---:|---:|---:|
| £2.5m xs £2.5m | £1,457,245 | 65.7% | £1,811,521 | 58.2% | 68.4% |
| £5m xs £5m | £868,479 | 28.8% | £1,730,501 | 35.0% | 42.7% |
| £10m xs £10m | £403,010 | 8.3% | £1,362,709 | 15.4% | 25.1% |
| £20m xs £20m | £146,067 | 1.7% | £1,335,287 | 7.8% | 9.3% |

### Sensitivity (±20% on one assumption at a time)

| Driver | Shock | Risk-adjusted premium | Change vs base |
|---|---:|---:|---:|
| Claim frequency (λ) | -20% | £1,443,614 | -16.6% |
| Claim frequency (λ) | +20% | £1,937,528 | +12.0% |
| Severity mean | -20% | £1,209,694 | -30.1% |
| Severity mean | +20% | £2,103,247 | +21.5% |
| Severity CV (tail heaviness) | -20% | £1,307,116 | -24.5% |
| Severity CV (tail heaviness) | +20% | £1,945,309 | +12.4% |
| Cost of capital | -20% | £1,558,096 | -10.0% |
| Cost of capital | +20% | £1,902,905 | +10.0% |
| Retention | -20% | £2,062,540 | +19.2% |
| Retention | +20% | £1,416,011 | -18.2% |
