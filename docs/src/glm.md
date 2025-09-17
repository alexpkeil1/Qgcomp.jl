# Introduction to Qgcomp.jl 

Note: this is (copied directly from the qgcomp package in R, basic vignette).

`Qgcomp.jl` is a module to implement g-computation for analyzing the effects of exposure mixtures. Quantile g-computation yields estimates of the effect of increasing all exposures by one quantile, simultaneously. This, it estimates a "mixture effect" useful in the study of exposure mixtures such as air pollution, diet, and water contamination.

Using terminology from methods developed for causal effect estimation, quantile  g-computation estimates the parameters of a marginal structural model that  characterizes the change in the expected potential outcome given a joint intervention on all exposures, possibly conditional on confounders. Under the assumptions of exchangeability, causal consistency, positivity, no interference, and correct model specification, this model yields a causal effect for an intervention on the mixture as a whole. While these assumptions may not be met exactly, they provide a useful road map for how to interpret the results of a qgcomp fit, and where efforts should be spent in terms of ensuring accurate model specification and selection of exposures that are sufficient to control co-pollutant confounding.

### The model
Say we have an outcome $Y$, some exposures $\mathbb{X}$ and possibly some other covariates (e.g. potential confounders) denoted by $\mathbb{Z}$.

The basic model of quantile g-computation is a joint marginal structural model given by

$\mathbb{E}(Y^{\mathbf{X}_q} | \mathbf{Z,\psi,\eta}) = g(\psi_0 + \psi_1 S_q +  \mathbf{\eta Z})$

where $g(\cdot)$ is a link function in a generalized linear model (e.g. the inverse logit function in the case of a logistic model for the probability that $Y=1$), $\psi_0$ is the model intercept, $\mathbf{\eta}$ is a set of model coefficients for the covariates and $S_q$ is an "index" that represents a joint value of exposures. Quantile g-computation (by default) transforms all exposures $\mathbf{X}$ into $\mathbf{X}_q$, which are "scores" taking on discrete values 0,1,2,etc. representing a categorical "bin" of exposure. By default, there are four bins with evenly spaced quantile cutpoints for each exposure, so ${X}_q=0$ means that $X$ was below the observed 25th percentile for that exposure. The index $S_q$ represents all exposures being set to the same value (again, by default, discrete values 0,1,2,3). Thus, *the parameter $\psi_1$ quantifies the expected change in the outcome, given a one quantile increase in all exposures simultaneously,* possibly adjusted for $\mathbf{Z}$. 

There are nuances to this particular model form that are available in the `qgcomp` package which will be explored below. There exists one special case of quantile g-computation that leads to fast fitting: linear/additive exposure effects. Here we simulate "pre-quantized" data where the exposures $X_1, X_2, X_3$ can only take on values of 0,1,2,3 in equal proportions. The model underlying the outcomes is given by the linear regression:

$\mathbb{E}(Y | \mathbf{X,\beta}) = \beta_0 + \beta_1 X_1 + \beta_2 X_2 + \beta_3 X_3$

with the true values of $\beta_0=0, \beta_1 =0.25, \beta_2 =-0.1, \beta_3=0.05$, and $X_1$ is strongly positively correlated with $X_2$ ($\rho=0.95$) and negatively correlated with $X_3$ ($\rho=-0.3$). In this simple setting, the parameter $\psi_1$ will equal the sum of the $\beta$ coefficients (0.2). Here we see that qgcomp estimates a value very close to 0.2 (as we increase sample size, the estimated value will be expected to become increasingly close to 0.2).

```julia
using Qgcomp, DataFrames, Random, StatsBase, GLM

# a function to generate quantized data with a specific correlation structure
function genxq(rng, n, corr=(0.95, -0.3), q=4)
    x = rand(rng, n, length(corr)+1)
    props = abs.(corr)
    ns = floor.(Int, n .* props)
    nidx = [setdiff(1:n, sample(rng, 1:n, nsi, replace=false)) for nsi in ns]
    for (j,c) in enumerate(corr)
        x[:,j+1] .= x[:,1]
        x[nidx[j],j+1] .= sample(rng, x[nidx[j],1], length(nidx[j]), replace=false)
        if c < 0
            x[:,j+1] .= 1.0 .- x[:,j+1]
        end
    end
    xq, _ = Qgcomp.get_xq(x, q)
    x, xq
end

# generate some data under a linear model with "quantized" exposures
    rng = Xoshiro(321)
    n = 1000
    X, Xq = genxq(rng, 1000, (0.95, -0.3))
    y = randn(rng, n) + Xq * [0.25, -0.1, 0.05]
    lindata = DataFrame(hcat(y, X), [:y, :x1, :x2, :x3])

    # check correlations
    cor(Xq)

    # fit model
    qgcomp_glm_noboot(@formula(y~x1+x2+x3), lindata, ["x1", "x2", "x3"], 4, Normal())

```

```output

julia> cor(Xq)
       # fit model
3×3 Matrix{Float64}:
  1.0      0.9504  -0.3184
  0.9504   1.0     -0.332
 -0.3184  -0.332    1.0

julia> qgcomp_glm_noboot(@formula(y~x1+x2+x3), lindata, ["x1", "x2", "x3"], 4, Normal())
Negative weights
1×4 DataFrame
 Row │ exposure  coef       ψ_partial  weight  
     │ String    Float64    Float64    Float64 
─────┼─────────────────────────────────────────
   1 │ x2        -0.166811  -0.166811      1.0
Positive weights
2×4 DataFrame
 Row │ exposure  coef       ψ_partial  weight    
     │ String    Float64    Float64    Float64   
─────┼───────────────────────────────────────────
   1 │ x1        0.330746    0.351734  0.940331
   2 │ x3        0.0209875   0.351734  0.0596686
─────────────────────────────────────────────────────────────────────────
                 Coef.  Std. Error     z  Pr(>|z|)   Lower 95%  Upper 95%
─────────────────────────────────────────────────────────────────────────
(Intercept)  0.0292843   0.0784494  0.37    0.7089  -0.124474    0.183042
ψ            0.184923    0.0480701  3.85    0.0001   0.0907069   0.279138
─────────────────────────────────────────────────────────────────────────
```



## How to use the `Qgcomp` module<a name="howto"></a>
Here we use a running example from the `metals` dataset (part of the `qgcomp` package in R) to demonstrate some features of the package and method. 

Namely, the examples below demonstrate use of the package for:
1. Fast estimation of exposure effects under a linear model for quantized exposures for continuous (normal) outcomes
2. Estimating conditional and marginal odds/risk ratios of a mixture effect for binary outcomes
3. Adjusting for non-exposure covariates when estimating effects of the mixture
4. Allowing non-linear and non-homogeneous effects of individual exposures and the mixture as a whole by including product terms
5. Using qgcomp to fit a time-to-event model to estimate conditional and marginal hazard ratios for the exposure mixture

For analogous approaches to estimating exposure mixture effects, illustrative examples can be seen in the `gQWS` package help files, which implements
weighted quantile sum (WQS) regression, and at https://jenfb.github.io/bkmr/overview.html, which describes Bayesian kernel machine regression.

The `metals` dataset from the package `qgcomp`, comprises a set of simulated well water exposures and two health outcomes (one continuous, one binary/time-to-event). The exposures are transformed to have mean = 0.0, standard deviation = 1.0. The data are used throughout to demonstrate usage and features of the `qgcomp` package.


```{julia metals data}
using RData
tf = tempname() * ".RData"
download("https://github.com/alexpkeil1/qgcomp/raw/refs/heads/main/data/metals.RData", tf)
metals = load(tf)["metals"]
metals[1:10,:] |> display
```

```output
julia> metals[1:10,:] |> display
10×26 DataFrame
 Row │ arsenic     barium      cadmium    calcium     chloride    chromium    copper      iron       lead         magn ⋯
     │ Float64     Float64     Float64    Float64     Float64     Float64     Float64     Float64    Float64      Floa ⋯
─────┼──────────────────────────────────────────────────────────────────────────────────────────────────────────────────
   1 │  0.0910017   0.0816636  15.0739    -0.774666   -0.154083   -0.055891    1.99438    19.1153    21.0726      -0.5 ⋯
   2 │  0.170183   -0.0359883  -0.712649  -0.685789   -0.196055   -0.0326849  -0.0249017  -0.203943  -0.0103784   -0.1
   3 │  0.133369    0.0993401   0.644199  -0.152523   -0.175118   -0.011611    0.257008   -0.196458  -0.0633759    0.9
   4 │ -0.525707   -0.766163   -0.861026   1.44727     0.025524   -0.0517329   0.754771   -0.231779  -0.00284799   2.5
   5 │  0.434205    0.406299    0.057089   0.410368   -0.241874   -0.0893182  -0.0991992  -0.169862  -0.0352763   -0.5 ⋯
   6 │  0.718327    0.195596   -0.682344  -0.89317    -0.0391994  -0.0738941  -0.0562228  -0.21293   -0.118461    -1.0
   7 │ -0.455829   13.8508      0.528584  -0.0340197  -0.176737   -0.0864338   0.161591   -0.146263  -0.114199     1.5
   8 │  0.446117   -0.92034    -0.49031   -0.0932714  -0.245691   -0.0424988   0.241774   -0.227692  -0.0606198    0.5
   9 │  0.200723   -0.296663    1.11029   -0.300652   -0.191157   -0.0391699  -0.0031943  -0.132103   0.151159    -0.1 ⋯
  10 │  0.0930527  -0.788987   -0.965073   0.617749    1.35235    -0.0628262  -0.207031    0.991215   0.00413885   8.0
```

### Example 1: linear model<a name="ex-linear"></a>
```julia linear model and runtime
# we save the names of the mixture variables in the variable "Xnm"
Xnm = [
    "arsenic","barium","cadmium","calcium","chromium","copper",
    "iron","lead","magnesium","manganese","mercury","selenium","silver",
    "sodium","zinc"

]

covars = ["nitrate","nitrite","sulfate","ph", "total_alkalinity","total_hardness"]



# Example 1: linear model
# Run the model and save the results "qc.fit"
f = @formula(y~1+a+b)
ff = FormulaTerm(f.lhs, (GLM.Term.(Symbol.(Xnm))...,))
@time qc_fit = qgcomp_glm_noboot(ff, metals[:,vcat(Xnm, "y")], Xnm, 4, Normal())
#  0.001750 seconds (49.98 k allocations: 2.679 MiB)
# contrasting other methods with computational speed
# WQS regression (v3.0.1 of gWQS package in R)
#system.time(wqs.fit = gWQS::gwqs(y~wqs,mix_name=Xnm, data=metals[:,vcat(Xnm, "y")], Normal(), 4))
#   user  system elapsed 
# 35.775   0.124  36.114 

# Bayesian kernel machine regression (note that the number of iterations here would 
#  need to be >5,000, at minimum, so this underestimates the run time by a factor
#  of 50+
#system.time(bkmr.fit = kmbayes(y=metals$y, Z=metals[,Xnm], family="gaussian", iter=100))
#   user  system elapsed 
# 81.644   4.194  86.520 
```

First note that qgcomp can be very fast relative to competing methods (with their example times given from single runs from a laptop). 

One advantage of quantile g-computation over other methods that estimate  "mixture effects" (the effect of changing all exposures at once), is that it  is very computationally efficient. Contrasting methods such as WQS (`gWQS`  package) and Bayesian Kernel Machine regression (`bkmr` package),  quantile g-computation can provide results many orders of magnitude faster. For example, the example above ran 3000X faster for quantile g-computation versus WQS regression, and we estimate the speedup would be several hundred thousand times versus Bayesian kernel machine regression. 

The speed relies on an efficient method to fit qgcomp when exposures are added additively to the model. When exposures are added using non-linear terms or non-additive terms (see below for examples), then qgcomp will be slower but often still faster than competetive approaches.

Quantile g-computation yields fixed weights in the estimation procedure, similar to WQS regression. However, note that the weights from `qgcomp_glm_noboot`  can be negative or positive. When all effects are linear and in the same  direction ("directional homogeneity"), quantile g-computation is equivalent to  weighted quantile sum regression in large samples.

The overall mixture effect from quantile g-computation ($\psi$1) is interpreted as  the effect on the outcome of increasing every exposure by one quantile, possibly conditional on covariates. Given the overall exposure effect, the weights are considered fixed and so do not have confidence intervals or p-values.

```output
# View results: scaled coefficients/weights and statistical inference about
# mixture effect
julia> @time qc_fit = qgcomp_glm_noboot(ff, metals[:,vcat(Xnm, "y")], Xnm, 4, Normal())
  0.001750 seconds (49.98 k allocations: 2.679 MiB)
Negative weights
5×4 DataFrame
 Row │ exposure   coef          weight       ψ_partial 
     │ String     Float64       Float64      Float64   
─────┼─────────────────────────────────────────────────
   1 │ selenium   -0.000105864  0.000856712   -0.12357
   2 │ manganese  -0.00788716   0.0638276     -0.12357
   3 │ lead       -0.00914645   0.0740185     -0.12357
   4 │ copper     -0.0476113    0.385299      -0.12357
   5 │ magnesium  -0.058819     0.475999      -0.12357
Positive weights
10×4 DataFrame
 Row │ exposure  coef        weight      ψ_partial 
     │ String    Float64     Float64     Float64   
─────┼─────────────────────────────────────────────
   1 │ zinc      0.00271249  0.00695574   0.389964
   2 │ cadmium   0.00517726  0.0132763    0.389964
   3 │ chromium  0.00802308  0.0205739    0.389964
   4 │ sodium    0.00843016  0.0216178    0.389964
   5 │ mercury   0.0095572   0.0245079    0.389964
   6 │ arsenic   0.013444    0.0344749    0.389964
   7 │ silver    0.0136815   0.0350839    0.389964
   8 │ barium    0.0231929   0.0594745    0.389964
   9 │ iron      0.0241278   0.061872     0.389964
  10 │ calcium   0.281618    0.722163     0.389964
─────────────────────────────────────────────────────────────────────────
                 Coef.  Std. Error      z  Pr(>|z|)  Lower 95%  Upper 95%
─────────────────────────────────────────────────────────────────────────
(Intercept)  -0.35667    0.107878   -3.31    0.0009  -0.568107  -0.145233
ψ             0.266394   0.0710247   3.75    0.0002   0.127188   0.4056
─────────────────────────────────────────────────────────────────────────
```

Now let"s take a brief look under the hood. `qgcomp` works in steps. First, the exposure variables are "quantized" or turned into score variables based on the total number of quantiles from the parameter `q`. You can access these via the `qx` object from the `qgcomp` fit object.

```r linear model and runtime
# quantized data
qc_fit.data[1:10,:]
```

You can re-fit a linear model using these quantized exposures. This is the "underlying model" of a qgcomp fit.
```julia
# regression with quantized data
newfit = lm(@formula(y ~ arsenic + barium + cadmium + calcium + chromium + copper + 
    iron + lead + magnesium + manganese + mercury + selenium + 
    silver + sodium + zinc), qc_fit.data)
newfit
```

```output
julia> newfit = lm(@formula(y ~ arsenic + barium + cadmium + calcium + chromium + copper + 
           iron + lead + magnesium + manganese + mercury + selenium + 
           silver + sodium + zinc), qc_fit.data)

StatsModels.TableRegressionModel{LinearModel{GLM.LmResp{Vector{Float64}}, GLM.DensePredChol{Float64, LinearAlgebra.CholeskyPivoted{Float64, Matrix{Float64}, Vector{Int64}}}}, Matrix{Float64}}

y ~ 1 + arsenic + barium + cadmium + calcium + chromium + copper + iron + lead + magnesium + manganese + mercury + selenium + silver + sodium + zinc

Coefficients:
──────────────────────────────────────────────────────────────────────────────
                    Coef.  Std. Error      t  Pr(>|t|)   Lower 95%   Upper 95%
──────────────────────────────────────────────────────────────────────────────
(Intercept)  -0.35667       0.107878   -3.31    0.0010  -0.568696   -0.144644
arsenic       0.013444      0.0184371   0.73    0.4663  -0.0227926   0.0496805
barium        0.0231929     0.0183474   1.26    0.2069  -0.0128675   0.0592533
cadmium       0.00517726    0.0181048   0.29    0.7750  -0.0304063   0.0407608
calcium       0.281618      0.0221639  12.71    <1e-30   0.238056    0.325179
chromium      0.00802308    0.0183908   0.44    0.6629  -0.0281225   0.0441687
copper       -0.0476113     0.0184657  -2.58    0.0103  -0.0839042  -0.0113184
iron          0.0241278     0.0201366   1.20    0.2315  -0.015449    0.0637047
lead         -0.00914645    0.0183426  -0.50    0.6183  -0.0451973   0.0269044
magnesium    -0.058819      0.0195185  -3.01    0.0027  -0.0971811  -0.0204569
manganese    -0.00788716    0.0210958  -0.37    0.7087  -0.0493493   0.033575
mercury       0.0095572     0.018297    0.52    0.6017  -0.0264041   0.0455185
selenium     -0.000105864   0.0183615  -0.01    0.9954  -0.036194    0.0359823
silver        0.0136815     0.0183149   0.75    0.4555  -0.022315    0.049678
sodium        0.00843016    0.0194193   0.43    0.6644  -0.0297368   0.0465971
zinc          0.00271249    0.0188122   0.14    0.8854  -0.0342614   0.0396864
──────────────────────────────────────────────────────────────────────────────
```

Here you can see that, *for a GLM in which all quantized exposures enter linearly and additively into the underlying model*, the overall effect from `qgcomp` is simply the sum of the adjusted coefficients from the underlying model. 

```julia
sum(coef(newfit)[2:end]) # sum of all coefficients excluding intercept and confounders, if any
qc_fit.fit[1][2] # overall effect and intercept from qgcomp fit
```

```output
julia> sum(coef(newfit)[2:end]) # sum of all coefficients excluding intercept and confounders, if any
0.2663941776757066

julia> qc_fit.fit[1][2]
0.26639417767570855
```

This equality is why we can fit qgcomp so efficiently under such a model. This is a specific case, and `qgcomp` also allows deviations from linear/additive approaches via Monte-Carlo (here generally referred to as bootstrapping methods) and estimating-equation-based methods, which require a 2-stage approach. `qgcomp` can allow for non-linearity and non-additivity in the underlying model, as well as non-linearity in the overall model. These extensions are described in some of the following examples.

### Example 2: conditional odds ratio, marginal odds ratio in a logistic model<a name="ex-logistic"></a>

This example introduces the use of a binary outcome in `qgcomp` via the 
`qgcomp_glm_noboot` function, which yields a conditional odds ratio or the
`qgcomp_glm_boot`, which yields a marginal odds ratio or risk/prevalence ratio. These
will not equal each other when there are non-exposure covariates (e.g. 
confounders) included in the model because the odds ratio is not collapsible (both
are still valid). Marginal parameters will yield estimates of the population
average exposure effect, which is often of more interest due to better 
interpretability over conditional odds ratios. Further, odds ratios are not
generally of interest when risk ratios can be validly estimated, so `qgcomp_glm_boot`
will estimate the risk ratio by default for binary data (set rr=FALSE to 
allow estimation of ORs when using `qgcomp_glm_boot`).

```julia
f = @formula(disease_state~1+a+b)
ff = FormulaTerm(f.lhs, (GLM.Term.(Symbol.(Xnm))...,))

# conditional odds ratio
qc_fit2 = qgcomp_glm_noboot(ff, metals[:,vcat(Xnm, "disease_state")], Xnm, 4, Binomial())
# marginal odds ratio
qcboot_fit2 = qgcomp_glm_boot(ff, metals[:,vcat(Xnm, "disease_state")], Xnm, 4, Binomial(), B=10)
# marginal risk ratio
qcboot_fit2b = qgcomp_glm_boot(ff, metals[:,vcat(Xnm, "disease_state")], Xnm, 4, Binomial(), B=10, msmlink=LogLink())
```

```output
julia> # conditional odds ratio
       qc_fit2 = qgcomp_glm_noboot(ff, metals[:,vcat(Xnm, "disease_state")], Xnm, 4, Binomial())
       # marginal odds ratio
Negative weights
9×4 DataFrame
 Row │ exposure   coef        weight     ψ_partial 
     │ String     Float64     Float64    Float64   
─────┼─────────────────────────────────────────────
   1 │ iron       -0.0215195  0.0309112  -0.696173
   2 │ lead       -0.0299151  0.0429708  -0.696173
   3 │ mercury    -0.033734   0.0484564  -0.696173
   4 │ cadmium    -0.0447591  0.0642931  -0.696173
   5 │ manganese  -0.0718927  0.103268   -0.696173
   6 │ calcium    -0.085805   0.123252   -0.696173
   7 │ arsenic    -0.0885288  0.127165   -0.696173
   8 │ copper     -0.113293   0.162736   -0.696173
   9 │ selenium   -0.206726   0.296946   -0.696173
Positive weights
6×4 DataFrame
 Row │ exposure   coef       weight     ψ_partial 
     │ String     Float64    Float64    Float64   
─────┼────────────────────────────────────────────
   1 │ sodium     0.0252791  0.064486    0.392009
   2 │ silver     0.0367468  0.0937398   0.392009
   3 │ magnesium  0.0506531  0.129214    0.392009
   4 │ chromium   0.0628293  0.160275    0.392009
   5 │ zinc       0.0784991  0.200248    0.392009
   6 │ barium     0.138002   0.352037    0.392009
─────────────────────────────────────────────────────────────────────────
                 Coef.  Std. Error      z  Pr(>|z|)  Lower 95%  Upper 95%
─────────────────────────────────────────────────────────────────────────
(Intercept)   0.26362     0.516085   0.51    0.6095  -0.747888   1.27513
ψ            -0.304163    0.340134  -0.89    0.3712  -0.970813   0.362486
─────────────────────────────────────────────────────────────────────────


julia> qcboot_fit2 = qgcomp_glm_boot(ff, metals[:,vcat(Xnm, "disease_state")], Xnm, 4, Binomial(), B=10)
       # marginal risk ratio
Underlying fit
────────────────────────────────────────────────────────────────────────────
                  Coef.  Std. Error      z  Pr(>|z|)   Lower 95%   Upper 95%
────────────────────────────────────────────────────────────────────────────
(Intercept)   0.26362     0.516085    0.51    0.6095  -0.747888    1.27513
arsenic      -0.0885288   0.0882484  -1.00    0.3158  -0.261492    0.0844348
barium        0.138002    0.0881834   1.56    0.1176  -0.0348345   0.310838
cadmium      -0.0447591   0.0868225  -0.52    0.6062  -0.214928    0.12541
calcium      -0.085805    0.106409   -0.81    0.4200  -0.294363    0.122753
chromium      0.0628293   0.0881578   0.71    0.4760  -0.109957    0.235615
copper       -0.113293    0.0886233  -1.28    0.2011  -0.286991    0.0604059
iron         -0.0215195   0.0966865  -0.22    0.8239  -0.211022    0.167983
lead         -0.0299151   0.0881026  -0.34    0.7342  -0.202593    0.142763
magnesium     0.0506531   0.0937246   0.54    0.5889  -0.133044    0.23435
manganese    -0.0718927   0.10069    -0.71    0.4752  -0.269242    0.125457
mercury      -0.033734    0.0878275  -0.38    0.7009  -0.205873    0.138405
selenium     -0.206726    0.0884817  -2.34    0.0195  -0.380147   -0.0333048
silver        0.0367468   0.0879218   0.42    0.6760  -0.135577    0.20907
sodium        0.0252791   0.0928861   0.27    0.7855  -0.156774    0.207332
zinc          0.0784991   0.0900685   0.87    0.3835  -0.098032    0.25503
────────────────────────────────────────────────────────────────────────────

MSM
Exposure specific weights not estimated in this type of model
─────────────────────────────────────────────────────────────────────────
                 Coef.  Std. Error      z  Pr(>|z|)  Lower 95%  Upper 95%
─────────────────────────────────────────────────────────────────────────
(Intercept)   0.26362     0.437867   0.60    0.5471  -0.594583   1.12182
mixture      -0.304163    0.285558  -1.07    0.2868  -0.863846   0.255519
─────────────────────────────────────────────────────────────────────────

```

Compare a qgcomp_glm_noboot fit:
```output
Negative weights
9×4 DataFrame
 Row │ exposure   coef        weight     ψ_partial 
     │ String     Float64     Float64    Float64   
─────┼─────────────────────────────────────────────
   1 │ iron       -0.0215195  0.0309112  -0.696173
   2 │ lead       -0.0299151  0.0429708  -0.696173
   3 │ mercury    -0.033734   0.0484564  -0.696173
   4 │ cadmium    -0.0447591  0.0642931  -0.696173
   5 │ manganese  -0.0718927  0.103268   -0.696173
   6 │ calcium    -0.085805   0.123252   -0.696173
   7 │ arsenic    -0.0885288  0.127165   -0.696173
   8 │ copper     -0.113293   0.162736   -0.696173
   9 │ selenium   -0.206726   0.296946   -0.696173
Positive weights
6×4 DataFrame
 Row │ exposure   coef       weight     ψ_partial 
     │ String     Float64    Float64    Float64   
─────┼────────────────────────────────────────────
   1 │ sodium     0.0252791  0.064486    0.392009
   2 │ silver     0.0367468  0.0937398   0.392009
   3 │ magnesium  0.0506531  0.129214    0.392009
   4 │ chromium   0.0628293  0.160275    0.392009
   5 │ zinc       0.0784991  0.200248    0.392009
   6 │ barium     0.138002   0.352037    0.392009
─────────────────────────────────────────────────────────────────────────
                 Coef.  Std. Error      z  Pr(>|z|)  Lower 95%  Upper 95%
─────────────────────────────────────────────────────────────────────────
(Intercept)   0.26362     0.516085   0.51    0.6095  -0.747888   1.27513
ψ            -0.304163    0.340134  -0.89    0.3712  -0.970813   0.362486
─────────────────────────────────────────────────────────────────────────
```

with a qgcomp_glm_boot fit:
```output
julia> qcboot_fit2 = qgcomp_glm_boot(ff, metals[:,vcat(Xnm, "disease_state")], Xnm, 4, Binomial(), B=10)
       # marginal risk ratio
Underlying fit
────────────────────────────────────────────────────────────────────────────
                  Coef.  Std. Error      z  Pr(>|z|)   Lower 95%   Upper 95%
────────────────────────────────────────────────────────────────────────────
(Intercept)   0.26362     0.516085    0.51    0.6095  -0.747888    1.27513
arsenic      -0.0885288   0.0882484  -1.00    0.3158  -0.261492    0.0844348
barium        0.138002    0.0881834   1.56    0.1176  -0.0348345   0.310838
cadmium      -0.0447591   0.0868225  -0.52    0.6062  -0.214928    0.12541
calcium      -0.085805    0.106409   -0.81    0.4200  -0.294363    0.122753
chromium      0.0628293   0.0881578   0.71    0.4760  -0.109957    0.235615
copper       -0.113293    0.0886233  -1.28    0.2011  -0.286991    0.0604059
iron         -0.0215195   0.0966865  -0.22    0.8239  -0.211022    0.167983
lead         -0.0299151   0.0881026  -0.34    0.7342  -0.202593    0.142763
magnesium     0.0506531   0.0937246   0.54    0.5889  -0.133044    0.23435
manganese    -0.0718927   0.10069    -0.71    0.4752  -0.269242    0.125457
mercury      -0.033734    0.0878275  -0.38    0.7009  -0.205873    0.138405
selenium     -0.206726    0.0884817  -2.34    0.0195  -0.380147   -0.0333048
silver        0.0367468   0.0879218   0.42    0.6760  -0.135577    0.20907
sodium        0.0252791   0.0928861   0.27    0.7855  -0.156774    0.207332
zinc          0.0784991   0.0900685   0.87    0.3835  -0.098032    0.25503
────────────────────────────────────────────────────────────────────────────

MSM
Exposure specific weights not estimated in this type of model
─────────────────────────────────────────────────────────────────────────
                 Coef.  Std. Error      z  Pr(>|z|)  Lower 95%  Upper 95%
─────────────────────────────────────────────────────────────────────────
(Intercept)   0.26362     0.437867   0.60    0.5471  -0.594583   1.12182
mixture      -0.304163    0.285558  -1.07    0.2868  -0.863846   0.255519
─────────────────────────────────────────────────────────────────────────
```

with a qgcomp_glm_boot fit, where the risk/prevalence ratio is estimated,  rather than the odds ratio:
```output
julia> qcboot_fit2b = qgcomp_glm_boot(ff, metals[:,vcat(Xnm, "disease_state")], Xnm, 4, Binomial(), B=10, msmlink=LogLink())
Underlying fit
────────────────────────────────────────────────────────────────────────────
                  Coef.  Std. Error      z  Pr(>|z|)   Lower 95%   Upper 95%
────────────────────────────────────────────────────────────────────────────
(Intercept)   0.26362     0.516085    0.51    0.6095  -0.747888    1.27513
arsenic      -0.0885288   0.0882484  -1.00    0.3158  -0.261492    0.0844348
barium        0.138002    0.0881834   1.56    0.1176  -0.0348345   0.310838
cadmium      -0.0447591   0.0868225  -0.52    0.6062  -0.214928    0.12541
calcium      -0.085805    0.106409   -0.81    0.4200  -0.294363    0.122753
chromium      0.0628293   0.0881578   0.71    0.4760  -0.109957    0.235615
copper       -0.113293    0.0886233  -1.28    0.2011  -0.286991    0.0604059
iron         -0.0215195   0.0966865  -0.22    0.8239  -0.211022    0.167983
lead         -0.0299151   0.0881026  -0.34    0.7342  -0.202593    0.142763
magnesium     0.0506531   0.0937246   0.54    0.5889  -0.133044    0.23435
manganese    -0.0718927   0.10069    -0.71    0.4752  -0.269242    0.125457
mercury      -0.033734    0.0878275  -0.38    0.7009  -0.205873    0.138405
selenium     -0.206726    0.0884817  -2.34    0.0195  -0.380147   -0.0333048
silver        0.0367468   0.0879218   0.42    0.6760  -0.135577    0.20907
sodium        0.0252791   0.0928861   0.27    0.7855  -0.156774    0.207332
zinc          0.0784991   0.0900685   0.87    0.3835  -0.098032    0.25503
────────────────────────────────────────────────────────────────────────────

MSM
Exposure specific weights not estimated in this type of model
──────────────────────────────────────────────────────────────────────────
                 Coef.  Std. Error      z  Pr(>|z|)  Lower 95%   Upper 95%
──────────────────────────────────────────────────────────────────────────
(Intercept)  -0.56237     0.241308  -2.33    0.0198  -1.03533   -0.0894143
mixture      -0.163725    0.180543  -0.91    0.3645  -0.517583   0.190132
──────────────────────────────────────────────────────────────────────────
```

### The remainder of this document IN PROGRESS
### Example 3: adjusting for covariates, plotting estimates

In the following code we run a maternal age-adjusted linear model with `qgcomp` (`family = Normal()`). Further, we plot both the weights, as well as the mixture slope which yields overall model confidence bounds, representing the bounds that, for each value of the joint exposure are expected to contain the true regression line over 95% of trials (so-called 95% "pointwise" bounds for the regression line). The pointwise comparison bounds, denoted by error bars on the plot, represent comparisons of the expected difference in outcomes at each quantile, with reference  to a specific quantile (which can be specified by the user, as below). These pointwise bounds are similar to the bounds created in the bkmr package when plotting the overall effect of all exposures. The pointwise bounds can be obtained via the pointwisebound.boot function. To avoid confusion between "pointwise regression" and "pointwise comparison" bounds, the pointwise regression bounds are denoted as the "model confidence band" in the plots, since they yield estimates of the same type of bounds as the `predict` function in R when applied to linear model fits.

Note that the underlying regression model is on the exposure quantile "scores", which take on integer values 0, 1, ..., q-1. For plotting purposes (when plotting regression line results from qgcomp_glm_boot), 
the quantile score is translated into a quantile (range = [0-1]). This is not a perfect correspondence, 
because the quantile g-computation model treats the  quantile score as a continuous variable, but the each quantile category spans a range of quantiles. For visualization, we fix the ends of the plot at the mid-points of the first and last quantile cut-point, so the range of the plot will change slightly if "q" is changed.

```julia

qc_fit3 = qgcomp_glm_noboot(@formula(y ~ mage35 + arsenic + barium + cadmium + calcium + chloride + 
                           chromium + copper + iron + lead + magnesium + manganese + 
                           mercury + selenium + silver + sodium + zinc),
                         metals, Xnm, 4, Normal())
qc_fit3
# not yet implemented in Julia
#plot(qc_fit3)

```

From the first plot we see weights from `qgcomp_glm_noboot` function, which include both
positive and negative effect directions. When the weights are all on a single side of the null,
these plots are easy to in interpret since the weight corresponds to the proportion of the
overall effect from each exposure. WQS uses a constraint in the model to force
all of the weights to be in the same direction - unfortunately such constraints
lead to biased effect estimates. The `qgcomp` package takes a different approach
and allows that "weights" might go in either direction, indicating that some exposures
may beneficial, and some harmful, or there may be sampling variation due to using
small or moderate sample sizes (or, more often, systematic bias such as unmeasured
confounding). The "weights" in `qgcomp` correspond to the proportion of the overall effect
when all of the exposures have effects in the same direction, but otherwise they
correspond to the proportion of the effect *in a particular direction*, which
may be small (or large) compared to the overall "mixture" effect. NOTE: the left 
and right sides of the  plot should not be compared with each other because the 
length of the bars corresponds to the effect size only relative to other effects 
in the same direction. The darkness of the bars corresponds to the overall effect 
size - in this case the bars on the right (positive) side of the plot are darker 
because the overall "mixture" effect is positive. Thus, the shading allows one
to make informal comparisons across the left and right sides: a large, darkly
shaded bar indicates a larger independent effect than a large, lightly shaded bar.

```julia
qcboot_fit3 = qgcomp_glm_boot(@formula(y ~ mage35 + arsenic + barium + cadmium + calcium + chloride + 
                           chromium + copper + iron + lead + magnesium + manganese + 
                           mercury + selenium + silver + sodium + zinc), metals, Xnm, 4, Normal(), 
                           B=10)# B should be 200-500+ in practice
qcboot_fit3
qcee_fit3 = qgcomp_glm_ee(@formula(y ~ mage35 + arsenic + barium + cadmium + calcium + chloride + 
                           chromium + copper + iron + lead + magnesium + manganese + 
                           mercury + selenium + silver + sodium + zinc), metals, Xnm, 4, Normal())
qcee_fit3
```

We can change the referent category for pointwise comparisons via the `pointwiseref` parameter:
```julia
# not yet implemented in Julia
#qgcomp:::modelbound.ee(qcee_fit3)
#plot(qcee_fit3, pointwiseref = 3, flexfit = FALSE, modelbound=TRUE)
#plot(qcboot_fit3, pointwiseref = 3, flexfit = FALSE)
```


Using `qgcomp_glm_boot` also allows us to assess
linearity of the total exposure effect (the second plot). Similar output is available
for WQS (`gWQS` package), though WQS results will generally be less interpretable
when exposure effects are non-linear (see below how to do this with `qgcomp_glm_boot` and `qgcomp_glm_ee`). 

The plot for the `qcboot_fit3` object (using g-computation with bootstrap variance) 
gives predictions at the joint intervention levels of exposure. It also displays
a smoothed (graphical) fit. 

Note that the uncertainty intervals given in the plot are directly accessible via the `pointwisebound` (pointwise comparison confidence intervals) and `modelbound` functions (confidence interval for the regression line):

```julia
# not yet implemented in julia
#pointwisebound.boot(qcboot_fit3, pointwiseref=3)
#qgcomp:::modelbound.boot(qcboot_fit3)
```

Because qgcomp estimates a joint effect of multiple exposures, we cannot, in general, assess model fit by overlaying predictions from the plots above with the data. Hence, it is useful to explore non-linearity by fitting models that
allow for non-linear effects, as in the next example.


### Example 4: non-linearity (and non-homogeneity)<a name="ex-nonlin"></a>

`qgcomp` (specifically `qgcomp.*.boot` and `qgcomp.*.ee` methods) addresses non-linearity in a way similar to standard parametric regression models, which lends itself to being able to leverage R language features for n-lin parametric models (or, more precisely, parametric models that deviate from a purely additive, linear function on the link function basis via the use of basis function representation of non-linear functions). 
Here is an example where we use a feature of the R language for fitting models
with interaction terms. We use `y~. + .^2` as the model formula, which fits a model
that allows for quadratic term for every predictor in the model. 



#### Aside: some details on qgcomp methods for non-linearity

Note that both `qgcomp.*.boot` (bootstrap) and `qgcomp.*.ee` (estimating equations) use standard methods for g-computation, whereas the `qgcomp.*.noboot` methods use a fast algorithm that works under the assumption of linearity and additivity of exposures (as described in the original paper on quantile-based g-computation). The "standard" method of g-computation with time-fixed exposures involves first fitting conditional models for the outcome, making predictions from those models under set exposure values, and then summarizing the predicted outcome distribution, possibly by fitting a second (marginal structural) model. `qgcomp.*.boot` follows this three-step process, while `qgcomp.*.ee` leverages estimating equations (sometimes: M-estimation) to estimate the parameters of the conditional and marginal structural model simultaneously. `qgcomp.*.ee` uses a sandwich variance estimator, which is similar to GEE (generalized estimating equation) approaches, and thus, when used correctly, can yield inference for longitudinal data in the same way that GEE does. The bootstrapping approach can also do this, but it takes longer. The extension to longitudinal data is representative of the broader concept that `qgcomp.*.boot` and `qgcomp.*.ee` can be used in a broader number of settings than `qgcomp.*.noboot` algorithms, but if one assumes linearity and additivity with no clustering of observations, and conditional parameters are of interest, then they are just a slower way to get equivalent results to `qgcomp.*.noboot`.


Below, we demonstrate a non-linear conditional fit (with a linear MSM) using the bootstrap approach. Similar approaches could be used to include interaction terms between exposures, as well as between exposures and covariates. Note this example is purposefully done incorrectly, as explained below.

```julia
f = @formula(y~1+a^2)
# create squared terms for all exposures
main_terms = GLM.Term.(Symbol.(Xnm))
squared_terms = [FunctionTerm(^, [Term(Symbol(xnmi)),ConstantTerm(2)], Expr(:call, :^, Symbol(xnmi), 2)) for xnmi in Xnm]

#ff = FormulaTerm(f.lhs, (main_terms...,)..., (squared_terms...,)...)



qcboot_fit4 = qgcomp(y~. + .^2,
                         expnms=Xnm,
                         metals[,c(Xnm, "y")], family=gaussian(), q=4, B=10, seed=125)
plot(qcboot_fit4)
```

Note that allowing for a non-linear effect of all exposures induces an apparent 
non-linear trend in the overall exposure effect. The smoothed regression line is 
still well within the confidence bands of the marginal linear model 
(by default, the overall effect of joint exposure is assumed linear, 
though this assumption can be relaxed via the "degree" parameter in qgcomp_glm_boot or qgcomp_glm_ee, 
as follows:

```{r ovrl-n-lin, results="markup", fig.show="hold", fig.height=5, fig.width=7.5, cache=FALSE}

qcboot_fit5 = qgcomp(y~. + .^2,
                         expnms=Xnm,
                         metals[,c(Xnm, "y")], family=gaussian(), q=4, degree=2, 
                      B=10, rr=FALSE, seed=125)
plot(qcboot_fit5)
qcee_fit5b = qgcomp_glm_ee(y~. + .^2,
                         expnms=Xnm,
                         metals[,c(Xnm, "y")], family=gaussian(), q=4, degree=2, 
                         rr=FALSE)
plot(qcee_fit5b)
```

Note that some features are not availble to ` qgcomp.* .ee` methods, which use estimating equations, rather than maximum likelihood methods. Briefly, these allow assessment of uncertainty under n-lin (and other) scenarios where the ` qgcomp.* .noboot` functions cannot, since they rely on the additivity and linearity assumptions to achieve speed. The ` qgcomp.* .ee` methods will generally be faster than a bootstrapped version, but they are not used extensively here because they are the newest additions to the qgcomp package, and the bootstrapped versions can be made fast (but not accurate) by reducing the number of bootstraps. Where available, the ` qgcomp.* .ee` will be preferred to the ` qgcomp.* .boot` versions for more stable and faster analyses when bootstrapping would otherwise be necessary.

Once again, we can access numerical estimates of uncertainty (answers differ between the `qgcomp.*.boot` and `qgcomp.*.ee` fits due to the small number of bootstrap samples):
```{r ovrl-n-linb, results="markup", fig.show="hold", fig.height=5, fig.width=7.5, cache=FALSE}
modelbound.boot(qcboot_fit5)
pointwisebound.boot(qcboot_fit5)
pointwisebound.noboot(qcee_fit5b)
```

Ideally, the smooth fit will look very similar to the model prediction regression 
line.

#### Interpretation of model parameters

As the output below shows, setting "degree=2" yields a second parameter in the model fit ($\psi_2$). The output of qgcomp now corresponds to estimates of the marginal structural model given by 

$\mathbb{E}\left(Y^{\mathbf{X}_q}\right) = g(\psi_0 + \psi_1 S_q + \psi_2 S_q^2)$

```julia
qcboot_fit5
```


so that $\psi_2$ can be interpreted similar to quadratic terms that might appear in a generalized linear model. $\psi_2$ estimates the change in the outcome for an additional unit of squared joint exposure, over-and-above the linear effect given by $\psi_1$. Informally, this is a way of assessing specific types of non-linearity in the joint exposure-response curves, and there are many other (slightly incorrect but intuitively useful) ways of interpreting parameters for squared terms in regressions (beyond the scope of this document). Intuition from generalized linear models (i.e. regarding interpretation of coefficients) applies directly to the models fit by quantile g-computation.


### Example 5: comparing model fits and further exploring non-linearity<a name="ex-nonlin2"></a>
Exploring a non-linear fit in settings with multiple exposures is challenging. One way to explore non-linearity, as demonstrated above, is to to include all 2-way interaction terms (including quadratic terms, or "self-interactions"). Sometimes this approach is not desired, either because the number of terms in the model can become very large, or because some sort of model selection procedure is required, which risks inducing over-fit (biased estimates and standard errors that are too small). Short of having a set of a priori non-linear terms to include, we find it best to take a default approach (e.g. taking all second order terms) that doesn"t rely on statistical significance, or to simply be honest that the search for a non-linear model is exploratory and shouldn"t be relied upon for robust inference. Methods such as kernel machine regression may be good alternatives, or supplementary approaches to exploring non-linearity.

NOTE: qgcomp necessarily fits a regression model with exposures that have a small number of possible values, based on the quantile chosen. By package default, this is `q=4`, but it is difficult to fully examine non-linear fits using only four points, so we recommend exploring larger values of `q`, which will change effect estimates (i.e. the model coefficient implies a smaller change in exposures, so the expected change in the outcome will also decrease).

Here, we examine a one strategy for default and exploratory approaches to mixtures that can be implemented in qgcomp using a smaller subset of exposures (iron, lead, cadmium), which we choose via the correlation matrix. High correlations between exposures may result from a common source, so small subsets of the mixture may be useful for examining hypotheses that relate to interventions on a common environmental source or set of behaviors. We can still adjust for the measured exposures, even though only 3 our exposures of interest are considered as the mixture of interest. This next example will require a new R package to help in exploring non-linearity: `splines`. Note that `qgcomp_glm_boot` must be used in order to produce the graphics below, as `qgcomp_glm_noboot` does not calculate the necessary quantities.

#### Graphical approach to explore non-linearity in a correlated subset of exposures using splines
```{r graf-n-lin-1, results="markup", fig.show="hold", fig.height=3, fig.width=7.5, cache=FALSE}
library(splines)
# find all correlations > 0.6 (this is an arbitrary choice)
cormat = cor(metals[,Xnm])
idx = which(cormat>0.6 & cormat <1.0, arr.ind = TRUE)
newXnm = unique(rownames(idx)) # iron, lead, and cadmium


qc.fit6lin = qgcomp_glm_boot(y ~ iron + lead + cadmium + 
                         mage35 + arsenic + magnesium + manganese + mercury + 
                         selenium + silver + sodium + zinc,
                         expnms=newXnm,
                         metals, family=gaussian(), q=8, B=10)

qc.fit6nonlin = qgcomp_glm_boot(y ~ bs(iron) + bs(cadmium) + bs(lead) +
                         mage35 + arsenic + magnesium + manganese + mercury + 
                         selenium + silver + sodium + zinc,
                         expnms=newXnm,
                         metals, family=gaussian(), q=8, B=10, degree=2)

qc.fit6nonhom = qgcomp_glm_boot(y ~ bs(iron)*bs(lead) + bs(iron)*bs(cadmium) + bs(lead)*bs(cadmium) +
                         mage35 + arsenic + magnesium + manganese + mercury + 
                         selenium + silver + sodium + zinc,
                         expnms=newXnm,
                         metals, family=gaussian(), q=8, B=10, degree=3)
```


It helps to place the plots on a common y-axis, which is easy due to dependence of the qgcomp plotting functions on ggplot. Here"s the linear fit :
```{r graf-n-lin-1b, results="markup", fig.show="hold", fig.height=3, fig.width=7.5, cache=FALSE}
pl_fit6lin = plot(qc.fit6lin, suppressprint = TRUE, pointwiseref = 4)
pl_fit6lin + coord_cartesian(ylim=c(-0.75, .75)) + 
  ggtitle("Linear fit: mixture of iron, lead, and cadmium")
```

Here"s the non-linear fit :
```{r graf-n-lin-2, results="markup", fig.show="hold", fig.height=3, fig.width=7.5, cache=FALSE}
pl_fit6nonlin = plot(qc.fit6nonlin, suppressprint = TRUE, pointwiseref = 4)
pl_fit6nonlin + coord_cartesian(ylim=c(-0.75, .75)) + 
  ggtitle("Non-linear fit: mixture of iron, lead, and cadmium")
```

And here"s the non-linear fit with statistical interaction between exposures (recalling that this will lead to non-linearity in the overall effect):
```{r graf-n-lin 3, results="markup", fig.show="hold", fig.height=3, fig.width=7.5, cache=FALSE}
pl_fit6nonhom = plot(qc.fit6nonhom, suppressprint = TRUE, pointwiseref = 4)
pl_fit6nonhom + coord_cartesian(ylim=c(-0.75, .75)) + 
  ggtitle("Non-linear, non-homogeneous fit: mixture of iron, lead, and cadmium")
```

#### Caution about graphical approaches
The underlying conditional model fit can be made extremely flexible, and the graphical representation of this (via the 
smooth conditional fit) can look extremely flexible. Simply matching the overall (MSM) fit to this line is not
a viable strategy for identifying parsimonious models because that would ignore potential for overfit. Thus,
caution should be used when judging the accuracy of a fit when comparing the "smooth conditional fit" to the 
"MSM fit." 
```{r grafwarn, results="markup", fig.show="hold", fig.height=3, fig.width=7.5, cache=FALSE}
qc.overfit = qgcomp_glm_boot(y ~ bs(iron) + bs(cadmium) + bs(lead) +
                         mage35 + bs(arsenic) + bs(magnesium) + bs(manganese) + bs(mercury) + 
                         bs(selenium) + bs(silver) + bs(sodium) + bs(zinc),
                         expnms=Xnm,
                         metals, family=gaussian(), q=8, B=10, degree=1)
qc.overfit
plot(qc.overfit, pointwiseref = 5)
```

Here, there is little statistical evidence for even a linear trend, which makes the 
smoothed conditional fit appear to be overfit. The smooth conditional fit can be turned off, as below.
```{r grafwarn-2, results="markup", fig.show="hold", fig.height=3, fig.width=7.5, cache=FALSE}
plot(qc.overfit, flexfit = FALSE, pointwiseref = 5)
```

### Example 6: miscellaneous other ways to allow non-linearity<a name="ex-nonlin3"></a>
Note that these are included as examples of *how* to include non-linearities, and are not intended as 
a demonstration of appropriate model selection. In fact, qc.fit7b is generally a bad idea in small
to moderate sample sizes due to large numbers of parameters. 

#### using indicator terms for each quantile
```{r n-lin-exs, results="markup", fig.show="hold", fig.height=3, fig.width=7.5, cache=FALSE}
qc.fit7a = qgcomp_glm_boot(y ~ factor(iron) + lead + cadmium + 
                         mage35 + arsenic + magnesium + manganese + mercury + 
                         selenium + silver + sodium + zinc,
                         expnms=newXnm,
                         metals, family=gaussian(), q=8, B=20, deg=2)
# underlying fit
summary(qc.fit7a$fit)$coefficients
plot(qc.fit7a)
```

#### interactions between indicator terms
```{r n-lin-exs-2, results="markup", fig.show="hold", fig.height=3, fig.width=7.5, cache=FALSE}
qc.fit7b = qgcomp_glm_boot(y ~ factor(iron)*factor(lead) + cadmium + 
                         mage35 + arsenic + magnesium + manganese + mercury + 
                         selenium + silver + sodium + zinc,
                         expnms=newXnm,
                         metals, family=gaussian(), q=8, B=10, deg=3)
# underlying fit
#summary(qc.fit7b$fit)$coefficients
plot(qc.fit7b)
```

#### breaks at specific quantiles (these breaks act on the quantized basis)
```{r n-lin-exs-3, results="markup", fig.show="hold", fig.height=3, fig.width=7.5, cache=FALSE}
qc.fit7c = qgcomp_glm_boot(y ~ I(iron>4)*I(lead>4) + cadmium + 
                         mage35 + arsenic + magnesium + manganese + mercury + 
                         selenium + silver + sodium + zinc,
                         expnms=newXnm,
                         metals, family=gaussian(), q=8, B=10, deg=2)
# underlying fit
summary(qc.fit7c$fit)$coefficients
plot(qc.fit7c)
```

Note one restriction on exploring non-linearity: while we can use flexible functions such as splines for individual exposures, the overall fit is limited via the `degree` parameter to polynomial functions (here a quadratic polynomial fits the non-linear model well, and a cubic polynomial fits the non-linear/non-homogeneous model well - though this is an informal argument and does not account for the wide confidence intervals). We note here that only 10 bootstrap iterations are used to calculate confidence intervals (to increase computational speed for the example), which is far too low.

#### Statistical approach explore non-linearity in a correlated subset of exposures using splines

The graphical approaches don"t give a clear picture of which model might be preferred, but we can compare the model fits using AIC, or BIC (information criterion that weigh model fit with over-parameterization). Both of these criterion suggest the linear model fits best (lowest AIC and BIC), which suggests that the apparently non-linear fits observed in the graphical approaches don"t improve prediction of the health outcome, relative to the linear fit, due to the increase in variance associated with including more parameters.
```{r splines, results="markup", fig.show="hold", fig.height=5, fig.width=7.5, cache=FALSE}
AIC(qc.fit6lin$fit)
AIC(qc.fit6nonlin$fit)
AIC(qc.fit6nonhom$fit)

BIC(qc.fit6lin$fit)
BIC(qc.fit6nonlin$fit)
BIC(qc.fit6nonhom$fit)
```

More examples on advanced topics can be viewed in the other package vignette.



## FAQ<a name="faq"></a>
### Why don"t I get weights/scaled effects from the `boot` or `ee` functions? (and other questions about the weights/scaled effect sizes)
Users often use the `qgcomp.*.boot` or `qgcomp.*.ee` functions because they want to marginalize over confounders or fit a non-linear joint exposure function. In both cases, the overall exposure response will no longer correspond to a simple weighted average of model coefficients, so none of the `qgcomp.*.boot` or `qgcomp.*.ee` functions will calculate weights. In most use cases, the weights would vary according to which level of joint exposure you"re at, so it is not a straightforward proposition to calculate them (and you may not wish to report 4 sets of weights if you use the default `q=4`). That is, the contribution of each exposure to the overall effect will change across levels of exposure if there is any non-linearity, which makes the weights not useful as simple inferential tools (and, at best, an approximation). If you wish to have an approximation, then use a "noboot" method and report the weights from that along with a caveat that they are not directly applicable to the results in which non-linearity/non-additivity/marginalization-over-covariates is performed (using boot methods). If you fit an otherwise linear model, you can get weights from a `qgcomp.*.noboot` which will be very close to the weights you might get from a linear model fit via `qgcomp.*.boot` functions, but be explicit that the weights come from a different model than the inference about joint exposure effects. 

It should be emphasized here that the weights are not a stable or entirely useful quantity for many research goals. Qgcomp addresses the mixtures problem of variance inflation by focusing on a parameter that is less susceptible to variance inflation than independent effects (the psi parameters, or overall effects of a mixture). The weights are a form of independent effect and will always be sensitive to this issue, regardless of the statistical method that is used. Some statistical approaches to improving estimation of independent effects (e.g. setting `bayes=TRUE`) is accessible in many qgcomp functions, but these approaches universally introduce bias in exchange for reducing variance and shouldn"t be used without a good understanding of what shrinkage and penalization methods actually accomplish. Principled and rigorous integration of these statistical approaches with qgcomp is in progress, but that work is inherently more bespoke and likely will not be available in this R package. The shrinkage and penalization literature is large and outside the scope of this software and documentation, so no other guidance is given here. In any case, the calculated weights are only interpretable as proportional effect sizes in a setting in which there is linearity, additivity, and collapsibility, and so the package makes no efforts to try to introduce weights into other settings in which those assumptions may not be met. Outside of those narrow settings, the weights would have a dubious interpretation and the programming underlying the qgcomp package errs on the side of preventing the reporting of results that are mutually inconsistent. If you are using the `boot` versions of a qgcomp function in a setting in which you know that the weights are valid, it is very likely that you do not actually need to be using the `boot` versions of the functions.

### Do I need to model non-linearity and non-additivity of exposures?
Maybe. The inferential object of qgcomp is the set of $\psi$ parameters that correspond to a joint exposure response. As it turns out, with correlated exposures non-linearity can disguise itself as non-additivity (Belzak and Bauer [2019] Addictive Behaviors). If we were inferring independent effects, this distinction would be crucial, but for joint effects it may turn out that it doesn"t matter much if you model non-linearity in the joint response function through non-additivity or non-linearity of individual exposures in a given study. Models fit in qgcomp still make the crucial assumption that you are able to model the joint exposure response via parametric models, so that assumption should not be forgotten in an effort to try to disentagle non-linearity (e.g. quadratic terms of exposures) from non-additivity (e.g. product terms between exposures). The important part to note about parametric modeling is that we have to explicitly tell the model to be non-linear, and no adaptation to non-linear settings will happen automatically. Exploring non-linearity is not a trivial endeavor.

### Do I have to use quantiles?
No. You can turn off "quantization" by setting `q=NULL` or you can supply your own categorization cutpoints via the "breaks" argument. It is up to the user to interpret the results if either of these options is taken. Frequently, `q=NULL` is used in concert with standardizing exposure variables by dividing them by their interquartile ranges (IQR). The joint exposure response can then be interpreted as the effect of an IQR change in all exposures. Using IQR/2 (with or without a log transformation before hand) will yield results that are most (roughly) compatible with the package defaults (`q=4`) but that does not require quantization. Quantized variables have nice properties: they prevent extrapolation and reduce influence of outliers, but the choice of how to include exposures in the model should be a deliberate and well-informed one. There are examples of setting `q=NULL` in the help files for qgcomp_glm_boot and qgcomp_glm_ee, but this approach is available for any of the qgcomp methods (and is accomplished nearly 100% outside of the package functions, aside from setting `q=NULL`).

### Can I cite this document?
Probably not in a scientific manuscript. If you find an idea here that is not published anywhere else and wish to develop it into a full manuscript, feel free! (But probably check with alex.keil@nih.gov to ask if a paper is already in development or is, perhaps, already published.)

### Where else can I get help?
The vignettes of the package and the help files of the functions give many, many examples of usage. Additionally, some edge case or interesting applicationsare available in the form of github gists at <https://gist.github.com/alexpkeil1>. If you come up with an interesting problem that you think could be solved in this package, but is currently not, feel free to submit an issue on the R package github page <https://github.com/alexpkeil1/qgcomp/issues>. Several additions to the package have already come about through that avenue (though not always quickly).

