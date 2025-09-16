## Introduction (copied directly from the qgcomp package in R, basic vignette):

`Qgcomp.jl` is a module to implement g-computation for analyzing the effects of exposure
mixtures. Quantile g-computation yields estimates of the effect of increasing
all exposures by one quantile, simultaneously. This, it estimates a "mixture
effect" useful in the study of exposure mixtures such as air pollution, diet,
and water contamination.

Using terminology from methods developed for causal effect estimation, quantile 
g-computation estimates the parameters of a marginal structural model that 
characterizes the change in the expected potential outcome given a joint intervention
on all exposures, possibly conditional on confounders. Under the assumptions of
exchangeability, causal consistency, positivity, no interference, and correct
model specification, this model yields a causal effect for an intervention
on the mixture as a whole. While these assumptions may not be met exactly, they
provide a useful road map for how to interpret the results of a qgcomp fit, and
where efforts should be spent in terms of ensuring accurate model specification
and selection of exposures that are sufficient to control co-pollutant confounding.

### The model
Say we have an outcome $Y$, some exposures $\mathbb{X}$ and possibly some other covariates (e.g. potential confounders) denoted by $\mathbb{Z}$.

The basic model of quantile g-computation is a joint marginal structural model given by

$$
\mathbb{E}(Y^{\mathbf{X}_q} | \mathbf{Z,\psi,\eta}) = g(\psi_0 + \psi_1 S_q +  \mathbf{\eta Z})
$$

where $g(\cdot)$ is a link function in a generalized linear model (e.g. the inverse logit function in the case of a logistic model for the probability that $Y=1$), $\psi_0$ is the model intercept, $\mathbf{\eta}$ is a set of model coefficients for the covariates and $S_q$ is an "index" that represents a joint value of exposures. Quantile g-computation (by default) transforms all exposures $\mathbf{X}$ into $\mathbf{X}_q$, which are "scores" taking on discrete values 0,1,2,etc. representing a categorical "bin" of exposure. By default, there are four bins with evenly spaced quantile cutpoints for each exposure, so ${X}_q=0$ means that $X$ was below the observed 25th percentile for that exposure. The index $S_q$ represents all exposures being set to the same value (again, by default, discrete values 0,1,2,3). Thus, *the parameter $\psi_1$ quantifies the expected change in the outcome, given a one quantile increase in all exposures simultaneously,* possibly adjusted for $\mathbf{Z}$. 

There are nuances to this particular model form that are available in the `qgcomp` package which will be explored below. There exists one special case of quantile g-computation that leads to fast fitting: linear/additive exposure effects. Here we simulate "pre-quantized" data where the exposures $X_1, X_2, X_3$ can only take on values of 0,1,2,3 in equal proportions. The model underlying the outcomes is given by the linear regression:

$$
\mathbb{E}(Y | \mathbf{X,\beta}) = \beta_0 + \beta_1 X_1 + \beta_2 X_2  + \beta_3 X_3 
$$

with the true values of $\beta_0=0, \beta_1 =0.25, \beta_2 =-0.1, \beta_3=0.05$, and $X_1$ is strongly positively correlated with $X_2$ ($\rho=0.95$) and negatively correlated with $X_3$ ($\rho=-0.3$). In this simple setting, the parameter $\psi_1$ will equal the sum of the $\beta$ coefficients (0.2). Here we see that qgcomp estimates a value very close to 0.2 (as we increase sample size, the estimated value will be expected to become increasingly close to 0.2).

```julia
using Qgcomp, DataFrames, Random, StatsBase, StatsModels

# generate some data under a linear model with "quantized" exposures
    rng = Xoshiro(1232)
    n = 1000
    x = rand(rng, n, 3)
    xq, _ = Qgcomp.get_xq(x, 4)
    props = (.95, 0.3)
    ns = floor.(Int, n .* props)
    xq[:,2] = vcat(xq[1:ns[1],1], sample(rng, xq[ns[1]+1:end,1], n-ns[1]))
    xq[:,3] = 3 .- vcat(xq[1:ns[2],1], sample(rng, xq[ns[2]+1:end,1], n-ns[2]))
    y = randn(n) + xq * [0.25, -0.1, 0.05]
    lindata = DataFrame(hcat(y, xq), [:y, :x1, :x2, :x3])


    cor(xq)
    qgcomp_glm_noboot(@formula(y~x1+x2+x3), lindata, ["x1", "x2", "x3"], 4, Normal())

```

```output

> cor(xq)
3×3 Matrix{Float64}:
  1.0        0.949536  -0.28404
  0.949536   1.0       -0.293409
 -0.28404   -0.293409   1.0

> qgcomp_glm_noboot(@formula(y~x1+x2+x3), lindata, ["x1", "x2", "x3"], 4, Normal())
Negative weights
1×4 DataFrame
 Row │ exposure  coef       ψ_partial  weight  
     │ String    Float64    Float64    Float64 
─────┼─────────────────────────────────────────
   1 │ x2        -0.081113  -0.081113      1.0
Positive weights
2×4 DataFrame
 Row │ exposure  coef      ψ_partial  weight   
     │ String    Float64   Float64    Float64  
─────┼─────────────────────────────────────────
   1 │ x1        0.253556   0.304781  0.831926
   2 │ x3        0.051226   0.304781  0.168074
─────────────────────────────────────────────────────────────────────────
                 Coef.  Std. Error      z  Pr(>|z|)  Lower 95%  Upper 95%
─────────────────────────────────────────────────────────────────────────
(Intercept)  -0.048126   0.0824814  -0.58    0.5596  -0.209787   0.113535
ψ             0.223669   0.0566225   3.95    <1e-04   0.11269    0.334647
─────────────────────────────────────────────────────────────────────────
```



