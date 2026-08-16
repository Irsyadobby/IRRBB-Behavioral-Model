# Methodology — Statistical Framework for Behavioral Rate Modeling

This document provides the full statistical specification underlying the IRRBB 
behavioral rate model: the ARIMA/ARIMAX estimation framework, the data 
transformations applied to both the target and predictor variables, and the 
diagnostic (assumption) tests used to validate each candidate model prior to 
selection.

---

## 1. ARIMA / ARIMAX Methodology

### 1.1 The ARIMA(p, d, q) Model

Let $y_t$ denote the behavioral rate series (e.g. term deposit early redemption 
rate) observed at monthly frequency. The **Autoregressive Integrated Moving 
Average** model, $\text{ARIMA}(p,d,q)$, is defined using the backshift (lag) 
operator $B$, where $B^k y_t = y_{t-k}$.

**Differencing (Integration, order $d$):**

$$
\nabla^d y_t = (1-B)^d y_t
$$

Differencing is applied when $y_t$ is non-stationary in mean (e.g. exhibits a 
trend), so that the AR/MA structure is fitted on a stationary series 
$w_t = \nabla^d y_t$.

**Autoregressive component (order $p$):**

$$
\phi(B) = 1 - \phi_1 B - \phi_2 B^2 - \dots - \phi_p B^p
$$

**Moving Average component (order $q$):**

$$
\theta(B) = 1 + \theta_1 B + \theta_2 B^2 + \dots + \theta_q B^q
$$

**Full ARIMA(p,d,q) specification:**

$$
\phi(B)\,(1-B)^d y_t = c + \theta(B)\,\varepsilon_t, \qquad \varepsilon_t \sim \text{i.i.d.}\,(0,\sigma^2)
$$

where $c$ is a constant (trend term, controlled by the `intercept` parameter in 
the library — `trend='c'` vs `trend='n'`), and $\varepsilon_t$ is white noise.

### 1.2 ARIMAX — Adding Exogenous Macroeconomic Regressors

Since the objective is to explain behavioral rate movements using macroeconomic 
drivers (interest rate, inflation, FX, etc.), the model is extended to 
**ARIMAX(p,d,q)** by adding a linear regression component with exogenous 
regressors $X_t = (x_{1,t}, \dots, x_{k,t})$:

$$
\phi(B)\,(1-B)^d \Big( y_t - \beta^\top X_t \Big) = c + \theta(B)\,\varepsilon_t
$$

Equivalently, this is the **regression-with-ARMA-errors** formulation used by 
`statsmodels.SARIMAX`:

$$
y_t = \beta^\top X_t + \eta_t, \qquad \phi(B)(1-B)^d \eta_t = \theta(B)\varepsilon_t
$$

Here $\beta = (\beta_1, \dots, \beta_k)$ are the macro-sensitivity coefficients 
of direct economic interest (e.g. sensitivity of the mortgage prepayment rate to 
a 100bps change in the benchmark rate), while the ARMA$(p,q)$ part absorbs the 
serial dependence remaining in the regression residual $\eta_t$ — this is what 
distinguishes ARIMAX from an OLS/static regression: **the error term itself is 
allowed to be autocorrelated and is explicitly modeled**, rather than assumed 
i.i.d. as in classical OLS.

### 1.3 Estimation

The model is estimated via **Maximum Likelihood Estimation (MLE)** using the 
state-space (Kalman filter) representation of the SARIMAX model. The 
log-likelihood of the Gaussian innovations is maximized:

$$
\hat{\theta} = \arg\max_{\theta} \; \ell(\theta) = \arg\max_{\theta} \sum_{t=1}^{n} \log f(\varepsilon_t \mid \theta)
$$

where $\theta = (\phi, \theta_{MA}, \beta, \sigma^2)$ is the full parameter 
vector. In the library, this optimization is attempted sequentially across 
multiple numerical optimizers — `lbfgs`, `bfgs`, `powell`, `nm` (Nelder–Mead) — 
falling back to the next optimizer if convergence fails, since likelihood 
surfaces for ARIMA models with several exogenous regressors are not always 
well-behaved (multiple local optima, flat regions).

### 1.4 Order & Variable Search Strategy

Two sources of specification uncertainty exist: **(i)** which macro variables 
belong in $X_t$, and **(ii)** which $(p,d,q)$ order best fits the residual 
dynamics. The library resolves both simultaneously via **grid search**:

1. **Variable subset search** — enumerate all $\binom{|X|}{n}$ combinations of 
   $n$ macro variables (`x_num`) from the candidate pool using 
   `itertools.combinations`, optionally appending the anomaly dummy $X_{AO}$.
2. **Order search** — for each variable subset, fit SARIMAX across the order 
   grid $(p,d,q) \in \{0,1,2\}^3$ (27 combinations by default).
3. **Selection within a subset** — for a given $X$-subset, the $(p,d,q)$ 
   yielding the lowest **out-of-sample MAPE** (Section 5.1) on the held-out test 
   window is retained.
4. **Selection across subsets** — the resulting leaderboard of best models 
   (one per $X$-subset) is ranked by test MAPE, and further filtered through the 
   diagnostic battery in Section 4 before a final model is chosen.

This two-stage search (statistical fit *and* forecast accuracy *and* residual 
diagnostics) is deliberately more conservative than selecting purely on 
in-sample AIC, which tends to favor over-parameterized models that overfit 
historical noise — a materially risk-relevant concern for behavioral models 
that feed directly into hedge notional sizing.

---

## 2. Data Transformation

### 2.1 Target Transformation

All five target series are **rates bounded in $(0,1)$**. Fitting a linear 
Gaussian model (ARIMAX) directly on a bounded series risks two problems: 
(i) forecasts/confidence intervals can fall outside $[0,1]$ (economically 
meaningless), and (ii) the variance of a rate/proportion is typically not 
constant across its range — it tends to be **larger near the boundaries** 
$y \to 0$ or $y \to 1$ (as in a Beta-distributed process), violating the 
homoskedasticity assumption of the Gaussian innovation term $\varepsilon_t$.

**Logit transformation** (primary transform used, `y_transform='logit'`):

$$
y_t^{\*} = \operatorname{logit}(y_t) = \ln\!\left(\frac{y_t}{1-y_t}\right)
$$

*Derivation / rationale:* the logit is the log-odds of the rate. As 
$y_t \to 0^{+}$, $y_t^{\*} \to -\infty$; as $y_t \to 1^{-}$, $y_t^{\*} \to +\infty$. 
This stretches the domain from the bounded interval $(0,1)$ to the unbounded 
real line $(-\infty, \infty)$, which is the natural domain for a Gaussian-error 
ARIMAX model. Because the transformation is steeper near the boundaries, a 
fixed absolute change in $y_t^{\*}$ corresponds to a smaller absolute change in 
$y_t$ near the boundaries than in the middle of the range — this compresses 
the boundary variance in the transformed space, partially correcting the 
heteroskedasticity described above.

*Inverse (back-transformation), used to convert forecasts back to a rate:*

$$
y_t = \operatorname{logit}^{-1}(y_t^{\*}) = \frac{1}{1+e^{-y_t^{\*}}} = \frac{e^{y_t^{\*}}}{1+e^{y_t^{\*}}}
$$

This is the standard **logistic (sigmoid) function** — by construction, its 
range is always $(0,1)$ regardless of the value of $y_t^{\*}$, which guarantees 
that back-transformed forecasts (and forecast confidence bounds) remain valid 
rates.

**Log transformation** (`y_transform='log'`, for strictly positive, unbounded-
above series):

$$
y_t^{\*} = \ln(y_t), \qquad y_t = e^{y_t^{\*}}
$$

Used when the series is positive but not bounded above by 1 (e.g. raw volume 
series rather than a rate), primarily to stabilize multiplicative/exponential 
growth patterns into an additive, linear-in-parameters form.

**log1p transformation** (`y_transform='log1p'`, for non-negative series that 
may include zero):

$$
y_t^{\*} = \ln(1+y_t), \qquad y_t = e^{y_t^{\*}} - 1
$$

Standard $\ln(y_t)$ is undefined at $y_t=0$; `log1p` avoids this by shifting the 
argument, which matters for behavioral series that can have zero-prepayment 
months (e.g. a small/illiquid cohort with no observed redemption in a given 
month).

### 2.2 Predictor Standardization

Exogenous macro regressors are standardized (z-scored) prior to estimation, 
using **train-window statistics only** (to avoid look-ahead bias into the test 
window):

$$
x_{j,t}^{\*} = \frac{x_{j,t} - \mu_j}{\sigma_j}, \qquad 
\mu_j = \frac{1}{n_{\text{train}}}\sum_{t \in \text{train}} x_{j,t}, \qquad
\sigma_j = \sqrt{\frac{1}{n_{\text{train}}}\sum_{t \in \text{train}} (x_{j,t}-\mu_j)^2}
$$

Binary/dummy columns (e.g. $X_{AO}$) are excluded from scaling, since 
standardizing a 0/1 indicator removes its direct interpretability as an 
on/off shift. Standardization does not change the model's forecast (it is a 
linear reparameterization, absorbed into $\hat\beta$), but materially improves 
**numerical convergence** of the MLE optimizer when regressors are on very 
different scales (e.g. an exchange rate in the thousands vs. a rate in decimals).

### 2.3 Lag Transformation

$$
x_{j,t}^{(L)} = x_{j,t-L}
$$

Lagged macro variables are used for two purposes:
1. **Economic realism** — behavioral responses to a macro shock (e.g. a rate 
   hike) are rarely instantaneous; borrowers/depositors react with a delay.
2. **In the ML benchmark (`run_hgbr`)** — lagged values of the target itself, 
   $y_{t-L}^{\*}$ (`y_lags`), are included as autoregressive features, since 
   gradient boosting has no native notion of a time index and must be given 
   explicit lag features to capture serial dependence (unlike ARIMAX, where 
   the AR/MA polynomial handles this implicitly).

### 2.4 Exponential Moving Average (EMA) Smoothing

$$
\text{EMA}_t^{(\text{span})} = \alpha\, x_t + (1-\alpha)\,\text{EMA}_{t-1}^{(\text{span})}, 
\qquad \alpha = \frac{2}{\text{span}+1}
$$

*Derivation:* unrolling the recursion gives an exponentially-decaying weighted 
average of all past observations:

$$
\text{EMA}_t = \alpha \sum_{i=0}^{t} (1-\alpha)^{i}\, x_{t-i}
$$

so that more recent observations receive geometrically larger weight, and the 
influence of older observations decays at rate $(1-\alpha)$ per period. Three 
spans are used — $\text{span} \in \{3, 6, 12\}$ months — corresponding to 
$\alpha \approx \{0.50, 0.286, 0.154\}$, capturing short-, medium-, and 
long-term macro trend persistence respectively. EMA smoothing is applied to 
reduce the influence of month-to-month noise in raw macro releases (which 
are also often subject to revision), so the model responds to the underlying 
trend rather than transient noise — at the cost of introducing a small lag in 
responsiveness to genuine turning points.

---

## 3. Feature Screening (Pre-Modeling)

### 3.1 Lag-Optimal Correlation Screening

For each candidate macro variable $x_j$, the Pearson correlation with the 
target is computed across a grid of candidate lags $L \in \{0,1,2,3,\dots\}$:

$$
\rho_j(L) = \text{corr}\big(y_t,\, x_{j,t-L}\big) 
= \frac{\sum_t (y_t-\bar y)(x_{j,t-L}-\bar x_j)}{\sqrt{\sum_t (y_t-\bar y)^2}\sqrt{\sum_t (x_{j,t-L}-\bar x_j)^2}}
$$

The lag $L_j^{\*} = \arg\max_L |\rho_j(L)|$ maximizing absolute correlation is 
selected as the "optimal lag" for variable $j$, and variables are shortlisted 
if $|\rho_j(L_j^{\*})|$ exceeds a threshold (default $0.5$). This is a fast, 
model-free screening step to reduce the combinatorial search space in Section 
1.4 before committing to full ARIMAX estimation on every possible subset.

### 3.2 Multicollinearity — Variance Inflation Factor (VIF)

For each shortlisted regressor $x_j$, an auxiliary OLS regression of $x_j$ on 
all remaining regressors is run, and:

$$
\text{VIF}_j = \frac{1}{1-R_j^2}
$$

where $R_j^2$ is the coefficient of determination of that auxiliary regression. 
*Interpretation:* $\text{VIF}_j$ measures how much the variance of $\hat\beta_j$ 
is inflated due to linear dependence with the other regressors, relative to a 
hypothetical scenario where $x_j$ is orthogonal to them 
($R_j^2=0 \Rightarrow \text{VIF}_j=1$). As $R_j^2 \to 1$ (near-perfect 
collinearity), $\text{VIF}_j \to \infty$. A common rule of thumb flags 
$\text{VIF}_j > 5$ (moderate) or $> 10$ (severe) concern — relevant here because 
several macroeconomic series (e.g. policy rate and inflation) are naturally 
correlated, which can otherwise produce unstable or sign-flipping $\hat\beta$ 
estimates.

### 3.3 Dimensionality Reduction — PCA (optional)

When collinearity cannot be resolved by variable selection alone, Principal 
Component Analysis is applied to the standardized macro matrix $X^{\*}$:

$$
X^{\*} = U\Sigma V^\top, \qquad \text{PC}_k = X^{\*} v_k
$$

where $v_k$ is the $k$-th eigenvector of the covariance matrix 
$\Sigma_X = \frac{1}{n}X^{\*\top}X^{\*}$, ranked by explained variance ratio 
$\text{EVR}_k = \lambda_k / \sum_i \lambda_i$ ($\lambda_k$ = eigenvalue). The 
number of retained components is chosen either explicitly or via a cumulative 
variance threshold (default 80%).

---

## 4. Diagnostic (Assumption) Tests

Every candidate model is validated against the standard ARIMA residual 
assumptions before being considered for final selection: residuals should be 
**(i)** serially uncorrelated, **(ii)** homoskedastic, and **(iii)** 
approximately normally distributed, and **(iv)** each included regressor 
should be statistically significant.

### 4.1 Ljung–Box Test — Residual Autocorrelation

**Hypotheses:**
- $H_0$: residuals are independently distributed (no autocorrelation up to lag $m$)
- $H_1$: residuals exhibit autocorrelation at one or more lags $\le m$

**Statistic:**

$$
Q = n(n+2)\sum_{k=1}^{m} \frac{\hat\rho_k^2}{n-k}
$$

where $n$ is the number of residuals, $m$ is the maximum lag tested, and:

$$
\hat\rho_k = \frac{\sum_{t=k+1}^{n} e_t\, e_{t-k}}{\sum_{t=1}^{n} e_t^2}
$$

is the sample autocorrelation of the residuals $e_t$ at lag $k$.

*Derivation intuition:* under $H_0$, each $\hat\rho_k$ is asymptotically 
$N(0, 1/n)$, so $n\,\hat\rho_k^2$ is asymptotically $\chi^2_1$; the 
Ljung–Box statistic sums a finite-sample-corrected version of these (the 
$n(n+2)/(n-k)$ weighting improves the chi-square approximation in small 
samples relative to the earlier Box–Pierce statistic, which uses simply 
$n\sum \hat\rho_k^2$).

**Distribution & decision:**

$$
Q \;\dot\sim\; \chi^2_{m-p-q}
$$

(degrees of freedom reduced by the number of estimated ARMA parameters). 
Reject $H_0$ (residual autocorrelation present → model misspecified) if the 
$p$-value $< \alpha$ (default $\alpha=0.05$).

### 4.2 Heteroskedasticity Test (H-test)

**Hypotheses:**
- $H_0$: residual variance is constant over time (homoskedastic)
- $H_1$: residual variance changes over time

**Statistic** (Goldfeld–Quandt-type ratio of squared residuals, as implemented 
in `statsmodels` SARIMAX diagnostics): the residual series is split into two 
equal halves of length $h = \lfloor n/2 \rfloor$, and:

$$
H = \frac{\displaystyle\sum_{t=n-h+1}^{n} e_t^2}{\displaystyle\sum_{t=1}^{h} e_t^2}
$$

*Derivation intuition:* under $H_0$ and Gaussian residuals, each sum of squared 
residuals scaled by $\sigma^2$ is $\chi^2_h$-distributed, so the ratio $H$ is 
the ratio of two independent (approximately, given ARMA whitening) 
chi-square variables divided by their degrees of freedom — precisely the 
construction of an $F$-statistic:

$$
H \;\dot\sim\; F_{h,\,h}
$$

**Decision:** since deviation in *either* direction ($H$ far above or far 
below 1) indicates heteroskedasticity, a two-sided $p$-value is used:

$$
p = 2 \min\big(F_{h,h}(H),\; 1-F_{h,h}(H)\big)
$$

Reject $H_0$ if $p < \alpha$.

### 4.3 Jarque–Bera Test — Residual Normality

**Hypotheses:**
- $H_0$: residuals are normally distributed
- $H_1$: residuals are not normally distributed

**Statistic**, based on sample skewness $S$ and excess kurtosis $K$ of the 
residuals:

$$
S = \frac{\hat m_3}{\hat m_2^{3/2}}, \qquad K = \frac{\hat m_4}{\hat m_2^{2}}, 
\qquad \hat m_j = \frac{1}{n}\sum_{t=1}^n (e_t - \bar e)^j
$$

$$
JB = \frac{n}{6}\left(S^2 + \frac{(K-3)^2}{4}\right)
$$

*Derivation intuition:* for a normal distribution, skewness $S=0$ and kurtosis 
$K=3$. The JB statistic is constructed so that, asymptotically, 
$\sqrt{n/6}\,S \sim N(0,1)$ and $\sqrt{n/24}\,(K-3) \sim N(0,1)$ under 
normality; squaring and summing these two independent asymptotically-normal 
quantities yields a sum of two squared standard normals:

$$
JB \;\dot\sim\; \chi^2_2
$$

**Decision:** reject $H_0$ (residuals non-normal — a concern for the validity 
of the Gaussian MLE and forecast confidence intervals) if $p < \alpha$.

### 4.4 Coefficient Significance (Wald / t-test)

For each estimated parameter $\hat\beta_j$ (ARMA coefficients and exogenous 
macro coefficients), the standard Wald test is applied:

$$
t_j = \frac{\hat\beta_j}{\text{SE}(\hat\beta_j)}, \qquad 
\text{SE}(\hat\beta_j) = \sqrt{\big[\hat{\mathcal{I}}(\theta)^{-1}\big]_{jj}}
$$

where $\hat{\mathcal I}(\theta)$ is the observed Fisher information matrix from 
the MLE fit. Under $H_0: \beta_j = 0$, $t_j$ is asymptotically standard normal 
(or Student-$t$ with $n-k$ degrees of freedom in finite samples), and the 
associated $p$-value is compared against $\alpha$. A regressor is flagged 
`Signif` if $p < \alpha$ — used both to prune weak macro drivers and, in 
aggregate across the leaderboard (`evaluate_selected_models`), as a model-
quality filter (models where the exogenous macro variables are *not* 
statistically significant are deprioritized even if their MAPE is low, since 
this suggests the fit is being driven by the ARMA structure alone rather than 
genuine macro sensitivity — undesirable for a *behavioral* model intended to 
explain rate movements economically).

---

## 5. Model Comparison Metrics

### 5.1 Mean Absolute Percentage Error (MAPE)

$$
\text{MAPE} = \frac{100\%}{n}\sum_{t=1}^{n} \left|\frac{y_t - \hat y_t}{y_t}\right|
$$

Computed **out-of-sample** on the held-out test window (`test_size`, default 
12 months) on the back-transformed (original rate) scale — i.e. accuracy is 
always evaluated in economically interpretable units, not in logit-space. 
A small $\varepsilon$ floor is applied to $y_t$ in the denominator 
(`mape_eps`) to avoid division-by-zero in months with a near-zero observed 
rate.

### 5.2 Akaike Information Criterion (AIC) — tracked, not primary selection criterion

$$
\text{AIC} = 2k - 2\ln(\hat L)
$$

where $k$ is the number of estimated parameters and $\hat L$ is the maximized 
likelihood. AIC is recorded for every fitted model as a reference for in-sample 
parsimony, but **final model selection is driven by out-of-sample MAPE**, 
since AIC (an in-sample fit measure) does not penalize a model that overfits 
the training window at the expense of genuine forecast accuracy.

---

## 6. Benchmark Model — HistGradientBoostingRegressor (HGBR)

As an independent, non-linear benchmark, a gradient-boosted trees model is 
fit on the same logit-transformed target, using lagged macro variables and 
autoregressive lags of the target as features. Hyperparameters 
(`max_depth`, `learning_rate`, `l2_regularization`, `min_samples_leaf`, etc.) 
are tuned via **time-series cross-validation** — either a **rolling window** 
(fixed-size training window sliding forward) or **expanding window** — rather 
than standard random $k$-fold CV, which would leak future observations into 
training folds and produce an over-optimistic validation score for a 
time-dependent process.

Model interpretability is assessed via **permutation importance**: for each 
feature $x_j$, its values are randomly shuffled in the evaluation set, and the 
resulting increase in MAPE is recorded:

$$
\text{Importance}_j = \text{MAPE}(y, \hat f(X_{\text{perm}(j)})) - \text{MAPE}(y, \hat f(X))
$$

averaged over multiple repetitions (default 30) to reduce noise from a single 
random permutation. A large positive value indicates the model relies heavily 
on that feature; a value near zero indicates the feature contributes little 
predictive information. This is cross-checked against the ARIMAX coefficient 
significance table (Section 4.4) as a robustness check on which macro drivers 
are genuinely explanatory versus incidental.

---

## References
- Box, G.E.P., Jenkins, G.M., Reinsel, G.C. (2015). *Time Series Analysis: 
  Forecasting and Control*, 5th ed. Wiley.
- Ljung, G.M., Box, G.E.P. (1978). On a Measure of Lack of Fit in Time Series 
  Models. *Biometrika*, 65(2), 297–303.
- Jarque, C.M., Bera, A.K. (1980). Efficient Tests for Normality, 
  Homoscedasticity and Serial Independence of Regression Residuals. 
  *Economics Letters*, 6(3), 255–259.
- Goldfeld, S.M., Quandt, R.E. (1965). Some Tests for Homoscedasticity. 
  *Journal of the American Statistical Association*, 60(310), 539–547.
- Adam, A. (2007). *Handbook of Asset and Liability Management*. Wiley.
- Lubinska, B. (2021). *Interest Rate Risk in the Banking Book: A Best 
  Practice Guide to Management and Hedging*. Wiley.
- Basel Committee on Banking Supervision (BCBS), *Interest Rate Risk in the 
  Banking Book*, Standards, April 2016.
