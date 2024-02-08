# Probabilistic losses

This repository contains PyTorch implementations of important losses and scores used in probabilistic forecasting.
The implemented losses (or scores) are the following:

- Quantile score
- Interval score
- Continuously ranked probability score (CRPS)
  - Normal distribution
  - Truncated normal distribution
  - Log-normal distribution
- Energy Score
- Variogram Score (of order p)

## Details
In the following, the different scores are presented in detail:

### Quantile score
Suppose that a forecaster aims at predicting the quantile $\alpha \in (0,1)$. If the prediction is given by $q_\alpha$ and the value $x$ materializes, then the quantile score (Gneiting and Raftery (2007)) is given by:
$$S(q_\alpha;x)=(x-q_\alpha)(\mathbb{1} \\{ x \leq q_\alpha \\} - \alpha).$$ Minimizing this score leads to the true quantile at level $\alpha$. Neural networks employing this score are usually referred to as Quantile Regression Networks (QRN).

### Interval score
Instead of a predictive quantile, consider a predictive $(1-\alpha)$ forecasting interval. Let $l_\alpha, r_\alpha$ denote the left and right endpoints of the predicted interval. If the value $x$ materializes, the interval score (Gneiting and Raftery (2007)) is given by: $$S(l_\alpha, r_\alpha; x) = (r_\alpha - l_\alpha) + \frac{2}{\alpha}(l_\alpha - x) \mathbb{1} \\{ x < l_\alpha \\} + \frac{2}{\alpha}(x-r_\alpha)\mathbb{1} \\{ x > r_\alpha \\}.$$ Minimizing this score leads to the correct interval endpoints, e.g. the quantiles at the levels $\frac{\alpha}{2}, 1 - \frac{\alpha}{2}$ respectively.

### Continuously ranked probability score 
The continuously ranked probability score (CRPS, Gneiting and Raftery (2007)) is a score that measures the discrepancy between a predictive density and a realized outcome. Consider a distribution function $F$ of some probability distribution on $\mathbb{R}$ and a realized outcome $x$. Then the CRPS is given by $$\mathrm{CRPS}(F,x) = \int_{-\infty}^{\infty} (F(y)-\mathbb{1} \\{y \geq x \\})^2 dy.$$ Intuitively, this score measures the discrepancy between a predictive distribution and a realized outcome. The CRPS is minimized, if $x \sim F$ (e.g. if the true distribution is predicted).

For several parametric distributions closed form expressions of the CRPS are available and can be used to train a neural network, then called distributional neural network (DRN, Rasp and Lerch (2018)). The following are implemented here:
#### Normal distribution (Gneiting et. al. (2005)):
$$\mathrm{CRPS}\big(\mathcal{N}(\mu,\sigma^2),y\big) = \sigma \bigg\\{ \frac{y-\mu}{\sigma} \bigg[2 \Phi \bigg( \frac{y-\mu}{\sigma} \bigg)-1\bigg] + 2 \varphi \bigg(\frac{y-\mu}{\sigma}\bigg)- \frac{1}{\sqrt{\pi}} \bigg \\},$$ where $\Phi, \varphi$ are the CDF and PDF of a standard normal distribution.

#### Truncated normal distribution (Gneiting and Thorarinsdottir (2010)):
$$
\begin{align}
\mathrm{CRPS}\big(\mathcal{N}^0(\mu,\sigma^2),y\big) & = \sigma \Phi \Big( \frac{\mu}{\sigma} \Big)^{-2} \bigg[ \frac{y-\mu}{\sigma} \Phi \Big( \frac{\mu}{\sigma} \Big) \bigg \\{ 2 \Phi \Big( \frac{y-\mu}{\sigma} \Big) + \Phi \Big( \frac{\mu}{\sigma} \Big) - 2 \bigg \\} \\
& + 2 \varphi \Big( \frac{y-\mu}{\sigma} \Big) \Phi \Big( \frac{\mu}{\sigma} \Big) - \frac{1}{\sqrt{\pi}} \Phi \Big( \sqrt{2} \frac{\mu}{\sigma} \Big) \bigg ].
\end{align}$$

#### Log-normal distribution (Baran and Lerch (2015)):

$$
\begin{align}
\mathrm{CRPS}\big(\log\mathcal{N}^0(\mu,\sigma^2),y\big) & = y \bigg[ 2 \Phi \Big( (\log(x)-\mu)/ \sigma \Big) - 1 \bigg ] \\
& - 2e^{\mu + \sigma^2/2} \bigg[ \Phi \Big( (\log(y)-\mu)/ \sigma - \sigma \Big) + \Phi \Big( \sigma / \sqrt{2} \Big) - 1 \bigg].
\end{align}$$

Closed forms for more distributions are available, compare Zamo and Naveau (2018). Applications that include these scores as loss functions for neural networks include temperature (normal distribution), windspeed (log-normal distribution) or precipitation (truncated normal distribution).

## Energy Score
The energy score (Gneiting and Raftery (2007)) can be thought of as a multivariate extension of the CRPS.
For some probability distribution $\mathbb{P}$ and a realized outcome $\boldsymbol{y}$, the energy score is given as:
$$\mathrm{ES}(\mathbb{P}, \boldsymbol{y}) =  \mathbb{E} \\| \boldsymbol{X} - \boldsymbol{y} \\| - \frac{1}{2} \mathbb{E} \\| \boldsymbol{X} - \boldsymbol{X'} \\|,$$
where $\boldsymbol{X}, \boldsymbol{X'}$ are independent copies of a random vector with distribution $\mathbb{P}$. The energy score has for example been used to forecast multivariate temperature fields.

## Variogram Score
The variogram score (Scheuerer and Hamill (2015)) is also a scoring rule used for multivariate data. For some probability distribution $\mathbb{P}$ and a realized outcome $\boldsymbol{y} \in \mathbb{R}^d$, the Variogram score of order $p$ is defined as:
$$\mathrm{VS}(\mathbb{P},\boldsymbol{y}) =  \sum_{i,j=1}^d w_{ij} (|y_i - y_j|^p - \mathbb{E} | X_i - X_j |^p )^2,$$
where $X_i, X_j$ are the *i*th and the *j* th component of a random vector $\boldsymbol{X}$ with distribution $\mathbb{P}$ and $w_{ij}$ are nonnegative weights.

## References
- Gneiting, T. and A. E. Raftery (2007). Strictly proper scoring rules, prediction, and estimation. Journal of the American Statistical Association 102(477), 359-378.
- -Rasp, S., and S. Lerch, 2018: Neural Networks for Postprocessing Ensemble Weather Forecasts. Mon. Wea. Rev., 146, 3885–3900.
- Gneiting, T. et al., 2005: Calibrated Probabilistic Forecasting Using Ensemble Model Output Statistics and Minimum CRPS Estimation. Mon. Wea. Rev., 133, 1098–1118.
- Gneiting, T., Thorarinsdottir, T. 2010: Probabilistic Forecasts of Wind Speed: Ensemble Model Output Statistics by using Heteroscedastic Censored Regression. Journal of the Royal Statistical Society Series A: Statistics in Society, Volume 173, Issue 2, Pages 371–388.
- Baran S, Lerch S (2015): Log-normal distribution based ensemble model output statistics models for probabilistic wind-speed forecasting. Q J R Meteorol Soc 141:2289–2299.
- Zamo, M., Naveau, P. Estimation of the Continuous Ranked Probability Score with Limited Information and Applications to Ensemble Weather Forecasts. Math Geosci 50, 209–234 (2018).
- Scheuerer, M. and Hamill, T. M. (2015). Variogram-based proper scoring rules for probabilistic forecasts of multivariate quantities. Monthly Weather Review, 143, 1321–1334.
