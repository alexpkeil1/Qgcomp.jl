#model_glm.jl

#= todo: 
1. finish bootstrapping
1. implement "degree" for polynomial, or perhaps msm formula?
=#
####################################################################################
#
# GLM based methods
#
####################################################################################


function QGcomp_glm(formula, data, expnms, q, family, link, breaks)
    QGcomp_glm(
        formula,
        data,
        String.(expnms),
        q,
        family,
        link,
        breaks,
        nothing,
        nothing,
        false,
        ID.(collect(1:size(data, 1))),
        nothing,
        QGcomp_weights()
    )
end

function QGcomp_glm(formula, data, expnms, q, family, link)
    qdata, breaks, _ = get_dfq(data, expnms, q)
    QGcomp_glm(formula, qdata, expnms, q, family, link, breaks)
end


function QGcomp_glm(formula, data, expnms, q, family)
    qdata, breaks, _ = get_dfq(data, expnms, q)
    QGcomp_glm(formula, qdata, expnms, q, family, canonicallink(family))
end


function Qgcomp_MSM(msmfit, ypred, msmexposure)
    Qgcomp_MSM(
        msmfit,
        ypred,
        msmexposure,
        Array{Float64,2}(undef, 0, 0),
        Array{Float64,2}(undef, 0, 0),
        Array{Float64,2}(undef, 0, 0),
        Array{Float64,2}(undef, 0, 0),
    )
end

function Qgcomp_MSM(msmfit, ypred)
    Qgcomp_MSM(
        msmfit,
        ypred,
        Array{Float64,1}(undef, 0),
    )
end

#=

using GLM
using Qgcomp, DataFrames, StatsModels, Random
using StatsBase
rng = Xoshiro()

xq, _ = Qgcomp.get_xq(x, 4)
y = randn(100) + xq * [.1, .05, 0] + xq .* xq * [.1, 0, 0]
data = DataFrame(hcat(y,x,z), [:y, :x1, :x2, :x3, :z1, :z2, :z3])
formula = @formula(y~x1^2+x2+x3+z1+z2+z3+z3^2)
expnms = ["x"*string(i) for i in 1:3]
m = Qgcomp.QGcomp_glm(formula, data, expnms, 4, Normal())
m.id = Qgcomp.ID.(collect(1:size(data, 1)))

fit_noboot! = Qgcomp.fit_noboot!
_fit_msm = Qgcomp._fit_msm
Qgcomp_MSM = Qgcomp.Qgcomp_MSM
kwargs = Dict(:contrasts => Dict{Symbol,Any}(), :B => 200)

contrasts::Dict{Symbol,<:Any}=Dict{Symbol,Any}()
msmformula = @formula(y~1+mixture + mixture^2 + z1)
fit!(m, bootstrap=true, msmformula = @formula(y~1+mixture + mixture^2 + z1), mcsize=300, B=1000)
m
=#

function fit_boot!(rng, m::QGcomp_glm; B::Int64=200, contrasts::Dict{Symbol,<:Any}=Dict{Symbol,Any}(), kwargs...)
    if !issubset(keys(kwargs), (:mcsize, :iters, :intvals, :msmformula, :degree, :msmlink, :msmfamily))
        throw(ArgumentError("Qgcomp error: unsupported keyword argument in: $(kwargs...)"))
    end
    ################
    # specifying overall effect definition
    ################

    intvals = :intvals ∈ keys(kwargs) ? kwargs[:intvals] : (1:m.q) .- 1.0
    ################
    # underlying model
    ################
    fit_noboot!(m)
    m.fitted = false

    ################
    # msm 
    ################
    if (:msmformula ∉ keys(kwargs)) && (:degree ∉ keys(kwargs))
        msmformula = m.formula.lhs ~ ConstantTerm(1) + Term(:mixture)
    elseif (:msmformula ∉ keys(kwargs)) && (:degree ∈ keys(kwargs))
        msmformula = m.formula.lhs ~ degreebuilder("mixture", kwargs[:degree])
    elseif (:msmformula ∈ keys(kwargs)) && (:degree ∉ keys(kwargs))
        msmformula = kwargs[:msmformula]
    elseif (:msmformula ∈ keys(kwargs)) && (:degree ∈ keys(kwargs))
        @warn "`msmformula` and `degree` are both specified, but these options both control the same thing. Overriding `degree` and using `msmformula.`"
        msmformula = kwargs[:msmformula]
    end

    if (:msmlink ∈ keys(kwargs)) 
        msmlink = kwargs[:msmlink]
    else
        msmlink = m.link
    end
    if (:msmfamily ∈ keys(kwargs)) 
        msmfamily = kwargs[:msmfamily]
    else
        msmfamily = m.family
    end


    # sub-sample or over-sample for msm (or just use sample if mcsize is not specified or equals the sample size)
    uid = unique(m.id)
    mcsize = :mcsize ∈ keys(kwargs) ? kwargs[:mcsize] : length(uid)
    msm, ypred, ypredmean =
        _fit_msm(rng, msmformula, m.data, m.ulfit, msmfamily, msmlink, contrasts, m.expnms, intvals, m.id, mcsize)
    m.msm = Qgcomp_MSM(msm, ypred, msm.mf.data.mixture)
    ################
    # bootstrapping
    ################
    m.msm.bootparm_draws = Array{Float64,2}(undef, B, length(coef(m.msm.msmfit)))
    m.msm.meanypred_draws = Array{Float64,2}(undef, B, length(ypredmean))

    for iter = 1:B
        bootmsmcoef, bootypredmean = doboot(
            rng, m.id, m.formula, msmformula, m.data, m.family, m.link, msmfamily, msmlink, m.expnms, intvals; contrasts=contrasts,
            kwargs...,
        )
        m.msm.bootparm_draws[iter, :] = bootmsmcoef
        m.msm.meanypred_draws[iter, :] = bootypredmean
    end
    ################
    # summarizing bootstrap
    ################

    m.msm.bootparm_vcov = cov(m.msm.bootparm_draws)
    m.msm.meanypred_vcov = cov(m.msm.meanypred_draws)

    psi, vc_psi = coef(m.msm.msmfit), m.msm.bootparm_vcov
    m.fit = (psi, vc_psi)
    m.fitted = true
end

fit_boot!(m::QGcomp_glm; kwargs...) = fit_boot!(Xoshiro(), m; kwargs...)


#=
using Random, StatsBase, Qgcomp
rng = Xoshiro()
ID = Qgcomp.ID
doboot = Qgcomp.doboot
mcsample = Qgcomp.mcsample
idvals = sort(sample(1:10, 30))

id = ID.(idvals)
id = m.id
=#
function doboot(
    rng,
    id::Vector{ID},
    formula,
    msmformula,
    data,
    family,
    link,
    msmfamily,
    msmlink,
    expnms,
    intvals;
    contrasts::Dict{Symbol,<:Any}=Dict{Symbol,Any}(),
    kwargs...,
)
    uid = unique(id)
    idx = collect(1:length(id))
    #idmap = [idx[findall(values(id) .== vid)] for vid in values(uid)] #observations for each id
    # individual level ids
    selectedids = sort(sample(rng, id, length(uid), replace=true))

    # rows of data/ids to pull
    bootidmap = [idx[findall(values(id) .== bid)] for bid in values(selectedids)]
    bootidx = reduce(vcat, bootidmap)
    # new individual IDs based on bootstrap copies
    ubootid = ID.(1:length(uid))
    bootid = reduce(vcat, [fill(ubootid[bi], length(bootobs)) for (bi, bootobs) in enumerate(bootidmap)])
    bootdf = data[bootidx, :]
    #bootids = id[bootidx]
    mcsize = :mcsize ∈ keys(kwargs) ? kwargs[:mcsize] : length(uid)
    # underlying fit
    ulfit = fit(GeneralizedLinearModel, formula, bootdf, family, link) # this form needed for proper use of predict function

    # msmfit
    msm, _, ymeanpred = _fit_msm(rng, msmformula, bootdf, ulfit, msmfamily, msmlink, contrasts, expnms, intvals, bootid, mcsize)
    # output: parameter draws
    coef(msm), ymeanpred
end

# check whether the predictions are constant over covariates (e.g. MSM only includes exposure)
function checkyprevalid(msmformula)
    isvalid = true
    for t in msmformula.rhs
        if hasfield(typeof(t), :sym)
            isvalid = t.sym == :mixture
        elseif hasfield(typeof(t), :args)
            isvalid = t.args[1].sym == :mixture
        end
        !isvalid && break
    end
    isvalid
end


function gen_msmdata(data, idx, intvals, expnms)
    md = [data[idx, :] for _ in intvals]
    for (wq, m) in enumerate(md)
        m[:, expnms] .*= 0.0
        m[:, expnms] .+= intvals[wq]
    end
    reduce(vcat, md)
end



function _fit_msm(rng, msmformula, data, ulfit, family, link, contrasts, expnms, intvals, id, mcsize)
    # modify msm formula to use original exposure
    ff = subformula(msmformula, ["mixture"])
    ff = sublhs(ff, :ypred)
    # sampled observations to keep (sub or super-sampling sometimes used for large samples or noisy estimators)
    uid = mcsample(rng, unique(id), mcsize)
    msmidx = reduce(vcat, [findall(values(id) .== sid) for sid in values(uid)])
    msmdata = gen_msmdata(data, msmidx, intvals, expnms) # expanded dataframe used for predictions from underlying model
    ytemp = DataFrame(ypred=zeros(size(data, 1)))
    # create MSM by calling the first exposure the "mixture"
    data.mixture = data[:, expnms[1]]
    msmdata.mixture = msmdata[:, expnms[1]]
    sch = schema(ff, hcat(ytemp, data), contrasts)
    f = apply_schema(ff, sch, GeneralizedLinearModel)
    msmdata.ypred = GLM.predict(ulfit, msmdata) # prediction from original fit with a dataframe works if the original fit was from a formula
    ypred, X = modelcols(f, StatsModels.columntable(msmdata))
    msm = glm(ff, msmdata, family, link)
    msm, ypred, [mean(ypred[findall(msmdata.mixture .== iv)]) for iv in intvals]
end


function fit_noboot!(m::QGcomp_glm)
    m.ulfit, m.fit = _fit_noboot(m.formula, m.data, m.family, m.link, m.expnms)
    m.fitted = true
    nothing
end


function _fit_noboot(formula, data, family, link, expnms)
    ulfit = fit(GeneralizedLinearModel, formula, data, family, link)
    coefs = coef(ulfit)
    vcnames = coefnames(ulfit)
    vc = vcov(ulfit)
    #psi =  hasintercept(formula) ? psicomb(coefs, vcnames, expnms) : psicomb_noint(coefs, vcnames, expnms)
    #vc_psi = hasintercept(formula) ? vccomb(vc, vcnames, expnms) : vccomb_noint(vc, vcnames, expnms)
    psi =  psicomb(coefs, vcnames, expnms)
    vc_psi = vccomb(vc, vcnames, expnms)
    ulfit, (psi, vc_psi)
end



#=
QGcomp_glm = Qgcomp.QGcomp_glm
fit! = Qgcomp.fit!
=#
"""
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
fitted(m)
aic(m)
aicc(m)
bic(m)
loglikelihood(m)
```
"""
function qgcomp_glm_noboot(formula, data, expnms, q, family)
    m = QGcomp_glm(formula, data, expnms, q, family)
    fit!(m, bootstrap=false)
    m.qgcweights.isvalid = checklinearadditive(m.formula.rhs, expnms)
    if m.qgcweights.isvalid
        m.qgcweights.neg, m.qgcweights.pos = getweights(coef(m.ulfit), coefnames(m.ulfit), String.(m.expnms))
    end
    m
end


"""
```julia
using Qgcomp, DataFrames, StatsModels, StatsBase

x1 = rand(100, 3)
x = rand(100, 3)
z = rand(100, 3)
xq, _ = Qgcomp.get_xq(x, 4)
y = randn(100) + xq * [.1, 0.05, 0]+ (xq .* xq) * [-.1, 0.05, 0]
ybin = Int.(y .> median(y))
data = DataFrame(hcat(ybin,y,x,z), [:ybin, :y, :x1, :x2, :x3, :z1, :z2, :z3])
form = @formula(y~x1+x2+x3+x1^2+x2^2+x3^2+z1+z2+z3)
formbin = @formula(ybin~x1+x2+x3+x1^2+x2^2+x3^2+z1+z2+z3)
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

# binary outcome
# note the top fit is incorrect
m0 = qgcomp_glm_noboot(formbin, data, expnms, 4, Bernoulli()) 

# three ways to specify non-linear fits
m = qgcomp_glm_boot(formbin, data, expnms, 4, Binomial(), B=2000, msmformula=@formula(y~mixture+mixture^2)) 
mb = qgcomp_glm_boot(formbin, data, expnms, 4, Binomial(), B=2000, degree=2) 
m2 = qgcomp_glm_ee(formbin, data, expnms2, 4, Binomial(), degree=2) 


```
"""
function qgcomp_glm_boot(formula, data, expnms, q, family; id=nothing, kwargs...)
    m = QGcomp_glm(formula, data, expnms, q, family)
    if isnothing(id)
        m.id = ID.(collect(1:size(data, 1)))
    else
        m.id = id
    end
    fit!(m, bootstrap=true; kwargs...)
    m
end


#= #######################################

Re-exports from Base

=# ######################################

function Base.show(io::IO, m::QGcomp_glm)
    if isfitted(m)
        if !isnothing(m.msm)
            println(io, "Underlying fit")
            println(io, coeftable(m.ulfit))
            println(io, "\nMSM")
        end
        show(io, m.qgcweights)
        println(io, coeftable(m))
    else
        println(io, "Not fitted")
    end
end


Base.show(m::QGcomp_glm) = Base.show(stdout, m)


;