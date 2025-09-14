#cox.jl

####################################################################################
#
# Cox model based methods
#
####################################################################################



function QGcomp_cox(formula, data, expnms::Vector{Symbol}, q, family, breaks, ulfit, fit, fitted, bootstrap, id)
    QGcomp_cox(formula, data, String.(expnms), q, family, breaks, nothing, nothing, false, nothing, nothing)
end

function QGcomp_cox(formula, data, expnms, q)
    qdata, breaks, _ = get_dfq(data, expnms, q)
    QGcomp_cox(formula, qdata, expnms, q, "Cox", breaks, nothing, nothing, false, nothing, nothing)
end



function fit!(m::QGcomp_cox;kwargs...)
    m.ulfit = fit(PHModel, m.formula, m.data;kwargs...)
    coefs = coef(m.ulfit)
    vcnames = coefnames(m.ulfit)
    vc = vcov(m.ulfit)
    vc_psi = vccomb(vc, vcnames, m.expnms)
    psi = psicomb(coefs, vcnames, m.expnms)
    m.fit = (psi, vc_psi)
    m.fitted = true
    nothing
end



"""
```julia

using Qgcomp
using LSurvival, DataFrames, Random
rng = Xoshiro(1232)
id, int, out, data = LSurvival.dgm(MersenneTwister(1232), 100, 1);
out = out.*rand(rng, length(out))

data[:, 1] = round.(data[:, 1], digits = 3);
d, X = data[:, 4], data[:, 1:3];
tab = ( in = int, out = out, d=d, x=X[:,1], z1=X[:,2], z2=X[:,3]) ;
df = DataFrame(tab)

formula = @formula(Surv(out, d)~z1+z2)
msmformula = @formula(Surv(out, d)~mixture)
expnms = ["z1", "z2"]
data = df
q = 4
id = ID.(id)
QGcomp_cox = Qgcomp.QGcomp_cox
subformula = Qgcomp.subformula
sublhs_surv = Qgcomp.sublhs_surv
mcsample = Qgcomp.mcsample
gen_msmdata = Qgcomp.gen_msmdata

mcsize = 1000
contrasts = Dict{Symbol,Any}()
m = QGcomp_cox(formula, data, expnms, q)
intvals = 1:m.q

ulfitg = glm(@formula(d~z1+z2), df, Bernoulli())
nbfitg = qgcomp_glm_noboot(@formula(d~z1+z2), df, ["z1", "z2"], 4, Bernoulli())
ulfit = coxph(formula, df, ties = "efron")
nbfit = qgcomp_cox_noboot(formula, df, ["z1", "z2"], 4)
_fit_msm_cox(Xoshiro(), msmformula, df, ulfit, contrasts, expnms, intvals, id, mcsize)
```
"""
function _fit_msm_cox(rng, msmformula, data, ulfit, contrasts, expnms, intvals, id, mcsize)
    # modify msm formula to use original exposure
    ff = subformula(msmformula, ["mixture"])
    ff = sublhs_surv(ff, :tpred, :dpred)
    # sampled observations to keep (sub or super-sampling sometimes used for large samples or noisy estimators)
    uid = mcsample(rng, unique(id), mcsize)
    msmidx = reduce(vcat, [findall(values(id) .== sid) for sid in values(uid)])
    data.____in = ulfit.R.enter
    data.____out = ulfit.R.exit
    msmdata = gen_msmdata(data, msmidx, intvals, expnms) # expanded dataframe used for predictions from underlying model
    dtemp = DataFrame(dpred=zeros(size(data, 1)), tpred=rand(size(data, 1)))
    # create MSM by calling the first exposure the "mixture"
    data.mixture = data[:, expnms[1]]
    msmdata.mixture = msmdata[:, expnms[1]]
    sch = schema(ff, hcat(dtemp, data), contrasts)
    f = apply_schema(ff, sch, PHModel)
    msmdata.dpred = zeros(length(msmdata.mixture)) # prediction from original fit with a dataframe works if the original fit was from a formula
    _, X = modelcols(ulfit.formula, StatsModels.columntable(msmdata))

    Ft, times = LSurvival.predict(ulfit, X)
    alltimes = vcat(times, maximum(ulfit.R.exit))
    allFt = vcat(Ft, Ft[end:end,:])
    ft = vcat(Ft[1:1,:], diff(allFt, dims=1))
    allds = rand.(Bernoulli.(ft))
    
    newds = allds .* 0 
    eventtimes = [findfirst(allds[:,c] .== 1) for c in 1:size(allds,2)]
    for (ii, tt) in enumerate(eventtimes)
        if !isnothing(tt)
            newds[tt,ii] = 1
        end
    end
    #extrema(sum(newds, dims=1))
    #sum(newds)/size(newds, 2)
    msmdata.tpred = [alltimes[argmax(newdscol)] for newdscol in eachcol(newds)]
    msmdata.dpred = sum(newds, dims=1)[:]
    

    dpred, X = modelcols(f, StatsModels.columntable(msmdata))
    msm = coxph(ff, msmdata)
    msm, dpred
end


function fit_boot!(rng, m::QGcomp_cox; B::Int64=200, contrasts::Dict{Symbol,<:Any}=Dict{Symbol,Any}(), kwargs...)
    if !issubset(keys(kwargs), (:mcsize, :iters, :intvals, :msmformula, :degree))
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
        msmformula = m.formula.lhs ~ ConstantTerm(1) + Term(mixture)
    elseif (:msmformula ∉ keys(kwargs)) && (:degree ∈ keys(kwargs))
        msmformula = m.formula.lhs ~ degreebuilder("mixture", kwargs[:degree])
    elseif (:msmformula ∈ keys(kwargs)) && (:degree ∉ keys(kwargs))
        msmformula = kwargs[:msmformula]
    elseif (:msmformula ∈ keys(kwargs)) && (:degree ∈ keys(kwargs))
        @warn "`msmformula` and `degree` are both specified, but these options both control the same thing. Overriding `degree` and using `msmformula.`"
        msmformula = kwargs[:msmformula]
    end
    # sub-sample or over-sample for msm (or just use sample if mcsize is not specified or equals the sample size)
    uid = unique(m.id)
    mcsize = :mcsize ∈ keys(kwargs) ? kwargs[:mcsize] : length(uid)

    #### identical to glm to this point
    msm, ypred, ypredmean =
        _fit_msm_cox(rng, msmformula, m.data, m.ulfit, m.family, contrasts, m.expnms, intvals, m.id, mcsize)
    m.msm = QgcMSM(msm, ypred)
    ################
    # bootstrapping
    ################
    m.msm.bootparm_draws = Array{Float64,2}(undef, B, length(coef(m.msm.msmfit)))
    m.msm.meanypred_draws = Array{Float64,2}(undef, B, length(ypredmean))

    for iter = 1:B
        bootmsmcoef, bootypredmean = doboot(
            rng,
            m.id,
            m.formula,
            msmformula,
            m.data,
            m.family,
            m.expnms,
            intvals;
            contrasts=contrasts,
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


function Base.show(io::IO, m::QGcomp_cox)
    if m.fitted 
        println(io, coeftable(m))
    else
        println(io, "Not fitted")
    end

end

Base.show(m::QGcomp_cox) = Base.show(stdout, m)

"""
```julia
using Qgcomp
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
"""
function qgcomp_cox_noboot(formula, data, expnms, q;kwargs...)
    m = QGcomp_cox(formula, data, expnms, q)
    fit!(m;kwargs...)
    m
end
;
