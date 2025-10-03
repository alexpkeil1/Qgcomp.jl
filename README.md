# Qgcomp.jl
Quantile g-computation for Julia

Documentation [here (stable version)](https://alexpkeil1.github.io/Qgcomp.jl/stable/)
and [here (dev version)](https://alexpkeil1.github.io/Qgcomp.jl/dev/)

### Developer's note
This package is mostly for internal usage, so documentation is not extensive. Below is a simple simulation example to get you started if you wish to use this for your own work. Questions about functionality submitted as bug reports will be closed without comment. Functionality questions can be addressed for the R package `qgcomp`.

### Continuous outcome
```julia
using Qgcomp, DataFrames, StatsModels

x1 = rand(100, 3)
x = rand(100, 3)
z = rand(100, 3)
xq, _ = Qgcomp.get_xq(x, 4)
y = randn(100) + xq * [.1, 0.05, 0]+ (xq .* xq) * [-.1, 0.05, 0]
data = DataFrame(hcat(y,x,z), [:y, :x1, :x2, :x3, :z1, :z2, :z3])
form = @formula(y~x1+x2+x3+x1^2+x2^2+x3^2+z1+z2+z3)
expnms = [:x1, :x2, :x3]

# note the top fit is incorrect
m0 = qgcomp_glm_noboot(form, data, expnms, 4, Normal())

# three ways to specify non-linear fits
m = qgcomp_glm_boot(form, data, expnms, 4, Normal(), B=2000, msmformula=@formula(y~mixture+mixture^2))
mb = qgcomp_glm_boot(form, data, expnms, 4, Normal(), B=2000, degree=2)
m2 = qgcomp_glm_ee(form, data, expnms, 4, Normal(), degree=2)
isfitted(m)
fitted(m)
aic(m)
aicc(m)
bic(m)
loglikelihood(m)
```

### Binary outcome
```julia
using Qgcomp, DataFrames, StatsModels

x1 = rand(100, 3)
x = rand(100, 3)
z = rand(100, 3)
xq, _ = Qgcomp.get_xq(x, 4)
y = randn(100) + xq * [.1, 0.05, 0]+ (xq .* xq) * [-.1, 0.05, 0]
y = Int.(y .> 0)
data = DataFrame(hcat(y,x,z), [:y, :x1, :x2, :x3, :z1, :z2, :z3])
form = @formula(y~x1+x2+x3+x1^2+x2^2+x3^2+z1+z2+z3)
expnms = [:x1, :x2, :x3]

# note the top fit is incorrect
m0 = qgcomp_glm_noboot(form, data, expnms, 4, Binomial())

# three ways to specify non-linear fits
m = qgcomp_glm_boot(form, data, expnms, 4, Binomial(), B=2000, msmformula=@formula(y~mixture+mixture^2))
mb = qgcomp_glm_boot(form, data, expnms, 4, Binomial(), B=2000, degree=2)
m2 = qgcomp_glm_ee(form, data, expnms, 4, Binomial(), degree=2)
isfitted(m)
fitted(m)
aic(m)
aicc(m)
bic(m)
loglikelihood(m)
```

### Survival analysis
```julia
using Qgcomp
# using Pkg; Pkg.add("https://github.com/alexpkeil1/LSurvival.jl")
using LSurvival, DataFrames, Random
id, int, out, data = LSurvival.dgm(MersenneTwister(1212), 100, 20);

data[:, 1] = round.(data[:, 1], digits = 3);
d, X = data[:, 4], data[:, 1:3];
wt = ones(length(d)) # random weights just to demonstrate usage
tab = ( in = int, out = out, d=d, x=X[:,1], z1=X[:,2], z2=X[:,3]) ;
df = DataFrame(tab)

coxph(@formula(Surv(in, out, d)~x+z1+z2), tab, ties = "efron", wts = wt) |> display
qgcomp_cox_noboot(@formula(Surv(in, out, d)~x+z1+z2), df, ["z1", "z2"], 4) |> display
```
