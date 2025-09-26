#bounds.jl 

## pointwise effect size bounds



function bounds(m, intvals = m.intvals, refval = intvals[1]; level = 0.95, types = ["pointwise", "model"])
    pwnames = [:mixture, :linpred, :diff, :ll_diff, :ul_diff, :se_diff]
    mnames = [:mixture, :linpred, :ll_simul, :ul_simul]
    ret = Dict(
        :pointwise => DataFrame(pointwise_bounds(m, intvals, refval; level = level), pwnames),
        :model => DataFrame(modelwise_bounds_msm(m, intvals, refval; level = level), mnames),
    )
    ret
end


function pointwise_bounds(m, intvals = m.intvals, refval = intvals[1]; level = 0.95)
    zcrit = quantile(Normal(), 1/2 + level/2)
    if isnothing(m.msm)
        ret = pointwise_bounds_nomsm(m, intvals, refval)
    else
        ret = pointwise_bounds_msm(m, intvals, refval)
    end
    ul, ll = ret[:, 3] + zcrit .* ret[:, 4], ret[:, 3] - zcrit .* ret[:, 4]
    hcat(ret[:, 1:3], ll, ul, ret[:, 4])
end

function modelwise_bounds(m, intvals = m.intvals, refval = intvals[1]; level = 0.95)
    if isnothing(m.msm)
        ret = modelwise_bounds_nomsm(m, intvals, refval)
    else
        ret = modelwise_bounds_msm(m, intvals, refval)
    end
    ret
end



#=
```julia
using Qgcomp, DataFrames, StatsModels

x1 = rand(100, 3)
x = rand(100, 3)
z = rand(100, 3)
xq, _ = Qgcomp.get_xq(x, 4)
y = randn(100) + xq * [.1, 0.05, 0]
data = DataFrame(hcat(y,x,z), [:y, :x1, :x2, :x3, :z1, :z2, :z3])
form = @formula(y~x1+x2+x3+z1+z2+z3)
form_noint = @formula(y~-1+x1+x2+x3+z1+z2+z3)
expnms = [:x1, :x2, :x3]

m = qgcomp_glm_noboot(form, data, expnms, 4, Normal())
m = qgcomp_glm_noboot(form_noint, data, expnms, 4, Normal())


refval = 2
Qgcomp.pointwise_bounds_nomsm(m, 0:.1:3, 0.8)
```
=#
function pointwise_bounds_nomsm(m, intvals = m.intvals, refval = intvals[1])
    coef = m.fit[1]
    vcov = m.fit[2]
    p, _ = size(vcov)

    sch = schema(m.formula, m.data, m.contrasts)
    f = apply_schema(m.formula, sch, typeof(m))
    fhasintercept = StatsModels.hasintercept(f)# any([typeof(t)<:InterceptTerm for t in f.rhs.terms])
    rownms = coefnames(m.ulfit)
    nonpsi = setdiff(rownms, m.expnms)
    #if !hasintercept(m.formula) 
    #end
    if fhasintercept
        rownms = popfirst!(nonpsi)
    else
        rownms = []
    end
    rownms = vcat(rownms, "ψ")
    rownms = vcat(rownms, nonpsi)
    gradient = zeros(p)

    refindex = findfirst(intvals .== refval)
    psiidx = fhasintercept ? 2 : 1
    interceptval = fhasintercept ? coef[1] : 0.0
    # e.g. 
    linpred = fill(interceptval, length(intvals)) # zeros(length(m.intvals))
    #sepred = zeros(length(m.intvals))
    diff = zeros(length(intvals))
    sediff = zeros(length(intvals))
    for (idx, iv) in enumerate(intvals)
        gradient[psiidx] = iv - refval
        sediff[idx] = sqrt(vccomb(vcov, gradient))
        linpred[idx] += coef[psiidx]*iv # linear predictor at referent level of all covariates
        diff[idx] = coef[psiidx]*(iv - refval)
    end
    hcat(intvals, linpred, diff, sediff)
end


"""
Create a dataset used in a marginal structural model will all values of 
mixture set to a specific value and covariates (if any) set to reference
values. 
"""
function apply_mixture_msm_baseline(f, mfd::M, val) where {M<:NamedTuple}
    kk = fieldnames(typeof(mfd))
    for k in kk
        if k == :mixture
            mfd[k] .= val
        else
            mfd[k] .= 0.0 # baseline values of all other variables
        end
    end
    _, Xint = modelcols(f, mfd)
    Xint
end

function apply_mixture_msm_baseline(f, mfd::M, val) where {M<:DataFrame}
    kk = Symbol.(names(mfd))
    for k in kk
        if k == :mixture
            mfd[:, k] .= val
        else
            mfd[:, k] .= 0.0 # baseline values of all other variables
        end
    end
    _, Xint = modelcols(f, mfd)
    Xint
end

function apply_mixture_msmfast_baseline(f, expnms, mfd::M, val) where {M<:DataFrame}
    kk = String.(names(mfd))
    for k in kk
        if k ∈ expnms
            mfd[:, k] .= val
        else
            mfd[:, k] .= 0.0 # baseline values of all other variables
        end
    end
    _, Xint = modelcols(f, mfd)
    Xint
end


#=
still need:
  to be able to re-create the design matrix - this is broken under re-sampling except
  perhaps under bootstrap methods with a saved MSM


  ```julia
using Qgcomp, DataFrames, StatsModels

x1 = rand(100, 3)
x = rand(100, 3)
z = rand(100, 3)
xq, _ = Qgcomp.get_xq(x, 4)
y = randn(100) + xq * [.1, 0.05, 0]
data = DataFrame(hcat(y,x,z), [:y, :x1, :x2, :x3, :z1, :z2, :z3])
form = @formula(y~x1+x2+x1^2+x3+z1+z2+z3)
form_noint = @formula(y~-1+x1+x1^2+x2+x3+z1+z2+z3)
expnms = [:x1, :x2, :x3]

m = qgcomp_glm_boot(form, data, expnms, 4, Normal(), msmformula=@formula(y~mixture+mixture^2 + z1));
m2 = qgcomp_glm_ee(form, data, expnms, 4, Normal(), msmformula=@formula(y~mixture+mixture^2 + z1));


refval = 2
Qgcomp.pointwise_bounds_msm(m)
Qgcomp.pointwise_bounds_msm(m2, 0:.1:3)


# testing
sublhs = Qgcomp.sublhs
Qgcomp_EEmsm = Qgcomp.Qgcomp_EEmsm
apply_mixture_msm_baseline = Qgcomp.apply_mixture_msm_baseline
 ```
=#
function pointwise_bounds_msm(m, intvals = m.intvals, refval = intvals[1])
    ms = m.msm
    coef = ms.parms
    #
    if typeof(ms.msmfit) <: Qgcomp_EEmsm
        oneobs = ms.msmfit.MSMdf[1:1, :]
        ff = ms.msmformula
        vcov = ms.msmfit.vcov
    else
        oneobs = map(x -> x[1:1], ms.msmfit.mf.data)
        ff = sublhs(ms.msmformula, :ypred)
        vcov = ms.bootparm_vcov
    end
    sch = schema(ff, oneobs, m.contrasts)
    f = apply_schema(ff, sch, typeof(m))
    p, _ = size(vcov)
    #refindex = findfirst(intvals .== refval)
    refD = apply_mixture_msm_baseline(f, oneobs, refval)
    linpred = zeros(length(intvals))
    diff = zeros(length(intvals))
    sediff = zeros(length(intvals))
    for (idx, iv) in enumerate(intvals)
        D = apply_mixture_msm_baseline(f, oneobs, iv)
        gradient = D .- refD
        sediff[idx] = sqrt(vccomb(vcov, gradient[:]))
        linpred[idx] = (D*coef)[1]  # linear predictor at referent level of all covariates
        diff[idx] = ((D .- refD)*coef)[1]
    end
    hcat(intvals, linpred, diff, sediff)
end


## model fit bounds
#=
still need:
  to be able to re-create the design matrix - this is broken under re-sampling except
  perhaps under bootstrap methods with a saved MSM


```julia
using Qgcomp, DataFrames, StatsModels

x1 = rand(100, 3)
x = rand(100, 3)
z = rand(100, 3)
xq, _ = Qgcomp.get_xq(x, 4)
y = randn(100) + xq * [.1, 0.05, 0]
data = DataFrame(hcat(y,x,z), [:y, :x1, :x2, :x3, :z1, :z2, :z3])
form = @formula(y~x1+x2+x3+z1+z2+z3)
expnms = [:x1, :x2, :x3]

m = qgcomp_glm_noboot(form, data, expnms, 4, Normal());
intvals=m.intvals
refval=intvals[1]

refval = 2
Qgcomp.pointwise_bounds_msm(m)
Qgcomp.pointwise_bounds_msm(m2, 0:.1:3)


# testing
sublhs = Qgcomp.sublhs
Qgcomp_EEmsm = Qgcomp.Qgcomp_EEmsm
apply_mixture_msm_baseline = Qgcomp.apply_mixture_msm_baseline
pointwise_bounds = Qgcomp.pointwise_bounds
pointwise_bounds_msm = Qgcomp.pointwise_bounds_msm
```
=#
function modelwise_bounds_nomsm(m, intvals = m.intvals, refval = intvals[1]; level = 0.95)
    pwdiff = pointwise_bounds(m, intvals, refval)
    ms = m.msm
    parms = coef(m.ulfit)
    # bootstrap
    vcov = GLM.vcov(m.ulfit)
    parmdraws = rand(MvNormal(parms, Hermitian(vcov)), 5000)'
    oneobs = m.data[1:1, :]
    ff = m.formula
    sch = schema(ff, oneobs, m.contrasts)
    f = apply_schema(ff, sch, typeof(m))

    resample_err = parms .- parmdraws'
    iV = inv(qr(vcov))
    #nresamples = size(boot_err,2)
    chi_samples = [col' * iV * col for col in eachcol(resample_err)]
    chichrit = quantile(Chisq(length(parms)), level)
    chidx = findall(chi_samples .> chichrit)
    c_set = parmdraws[chidx, :]


    lims = reduce(
        vcat,
        map(
            x -> [x[1] x[2]],
            [extrema(apply_mixture_msmfast_baseline(f, m.expnms, oneobs, val)*c_set') for val in intvals],
        ),
    )

    hcat(pwdiff[:, 1:2], lims)
end




## model fit bounds
#=
still need:
  to be able to re-create the design matrix - this is broken under re-sampling except
  perhaps under bootstrap methods with a saved MSM


```julia
using Qgcomp, DataFrames, StatsModels

x1 = rand(100, 3)
x = rand(100, 3)
z = rand(100, 3)
xq, _ = Qgcomp.get_xq(x, 4)
y = randn(100) + xq * [.1, 0.05, 0]
data = DataFrame(hcat(y,x,z), [:y, :x1, :x2, :x3, :z1, :z2, :z3])
form = @formula(y~x1+x2+x1^2+x3+z1+z2+z3)
form_noint = @formula(y~-1+x1+x1^2+x2+x3+z1+z2+z3)
expnms = [:x1, :x2, :x3]

m = qgcomp_glm_boot(form, data, expnms, 4, Normal(), msmformula=@formula(y~mixture+mixture^2 + z1));
m = qgcomp_glm_ee(form, data, expnms, 4, Normal(), msmformula=@formula(y~mixture+mixture^2 + z1));


refval = 2
Qgcomp.pointwise_bounds_msm(m)
Qgcomp.pointwise_bounds_msm(m2, 0:.1:3)


# testing
sublhs = Qgcomp.sublhs
Qgcomp_EEmsm = Qgcomp.Qgcomp_EEmsm
apply_mixture_msm_baseline = Qgcomp.apply_mixture_msm_baseline
pointwise_bounds = Qgcomp.pointwise_bounds
pointwise_bounds_msm = Qgcomp.pointwise_bounds_msm
```
=#
function modelwise_bounds_msm(m, intvals = m.intvals, refval = intvals[1]; level = 0.95)
    pwdiff = pointwise_bounds(m, intvals, refval)
    ms = m.msm
    parms = ms.parms[:, :]
    # bootstrap
    if typeof(ms.msmfit) <: Qgcomp_EEmsm
        parmdraws = rand(MvNormal(ms.parms, Hermitian(ms.msmfit.vcov)), 5000)'
        oneobs = ms.msmfit.MSMdf[1:1, :]
        vcov = ms.msmfit.vcov
        ff = ms.msmformula
    else
        parmdraws = ms.bootparm_draws
        oneobs = map(x -> x[1:1], ms.msmfit.mf.data)
        vcov = ms.bootparm_vcov
        ff = sublhs(ms.msmformula, :ypred)
    end
    sch = schema(ff, oneobs, m.contrasts)
    f = apply_schema(ff, sch, typeof(m))

    resample_err = parms .- parmdraws'
    iV = inv(qr(vcov))
    #nresamples = size(boot_err,2)
    chi_samples = [col' * iV * col for col in eachcol(resample_err)]
    chichrit = quantile(Chisq(length(ms.parms)), level)
    chidx = findall(chi_samples .> chichrit)
    c_set = parmdraws[chidx, :]


    lims = reduce(
        vcat,
        map(x -> [x[1] x[2]], [extrema(apply_mixture_msm_baseline(f, oneobs, val)*c_set') for val in intvals]),
    )

    hcat(pwdiff[:, 1:2], lims)
end
