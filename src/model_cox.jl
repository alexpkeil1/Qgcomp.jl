#cox.jl

####################################################################################
#
# Cox model based methods
#
####################################################################################



function QGcomp_cox(formula, data, expnms, q, family, breaks, ulfit, fit, fitted)
    QGcomp_cox(formula, data, String.(expnms), q, family, breaks, nothing, nothing, false, ID.(collect(1:size(data, 1))),nothing)
end

function QGcomp_cox(formula, data, expnms, q)
    qdata, breaks, _ = get_dfq(data, expnms, q)
    QGcomp_cox(formula, qdata, expnms, q, "Cox", breaks, nothing, nothing, false)
end

function QgcMSM(msmfit, ypred, msmexposure) 
    QgcMSM(
        msmfit,
        ypred,
        msmexposure,
        Array{Float64,2}(undef, 0, 0),
        Array{Float64,2}(undef, 0, 0),
        Array{Float64,2}(undef, 0, 0),
        Array{Float64,2}(undef, 0, 0),
    )
end




function fit_noboot!(m::QGcomp_cox;kwargs...)
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
    dtemp = DataFrame(dpred=zeros(size(data, 1)), tpred=zeros(size(data, 1)))
    # create MSM by calling the first exposure the "mixture"
    data.mixture = data[:, expnms[1]]
    msmdata.mixture = msmdata[:, expnms[1]]
    sch = schema(ff, hcat(dtemp, data), contrasts)
    f = apply_schema(ff, sch, PHModel)
    _, X = modelcols(ulfit.formula, StatsModels.columntable(msmdata))

    # predicting outcomes under the model using the cumulative incidence function
    Ft, times = LSurvival.predict(ulfit, X)
    maxt = maximum(ulfit.R.exit)
    alltimes = vcat(times, maxt)
    allFt = vcat(Ft, Ft[end:end,:])
    nuniquetimes = size(allFt,1)
    ft = vcat(Ft[1:1,:], diff(allFt, dims=1))
    allds = vcat(rand.(rng, Bernoulli.(Ft)), zeros(Int, 1, size(Ft,2)))
    
    msmdata.dpred .= 0.0
    msmdata.tpred .= maxt

    newds = allds .* 0 
    eventtimes = [findfirst(allds[:,c] .== 1) for c in 1:size(allds,2)]
    for (ii, tt) in enumerate(eventtimes)
        if !isnothing(tt)
            msmdata.dpred[ii] = 1.0
            msmdata.tpred[ii] = alltimes[tt]
        end
    end

    #extrema(sum(newds, dims=1))
    #sum(newds)/size(newds, 2)
    ncopies = length(intvals)
    endidx = [j for j in (mcsize .* (1:ncopies))]
    begidx = vcat(1, endidx[1:end-1] .+ 1)
    dpredmean = [StatsBase.mean(msmdata.dpred[begidx[idx]:endidx[idx]]) for idx in 1:ncopies]

    dpred, _ = modelcols(f, StatsModels.columntable(msmdata)) # output full time
    msm = coxph(ff, msmdata)
    msm, dpred, dpredmean
end


function fit_boot!(rng, m::QGcomp_cox; B::Int64=200, contrasts::Dict{Symbol,<:Any}=Dict{Symbol,Any}(), kwargs...)
    if !issubset(keys(kwargs), (:mcsize, :iters, :intvals, :msmformula, :degree, :id))
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
    msmcoxcheck(m.ulfit.R)

    m.fitted = false

    ################
    # msm 
    ################
    if (:msmformula ∉ keys(kwargs)) && (:degree ∉ keys(kwargs))
        msmformula = deepcopy(m.formula.lhs) ~ Term(:mixture)
    elseif (:msmformula ∉ keys(kwargs)) && (:degree ∈ keys(kwargs))
        msmformula = deepcopy(m.formula.lhs) ~ degreebuilder("mixture", kwargs[:degree])
    elseif (:msmformula ∈ keys(kwargs)) && (:degree ∉ keys(kwargs))
        msmformula = kwargs[:msmformula]
    elseif (:msmformula ∈ keys(kwargs)) && (:degree ∈ keys(kwargs))
        @warn "`msmformula` and `degree` are both specified, but these options both control the same thing. Overriding `degree` and using `msmformula.`"
        msmformula = kwargs[:msmformula]
    end
    if (:id ∈ keys(kwargs))
        m.id = kwargs.id
    end

    # sub-sample or over-sample for msm (or just use sample if mcsize is not specified or equals the sample size)
    uid = unique(m.id)
    mcsize = :mcsize ∈ keys(kwargs) ? kwargs[:mcsize] : length(uid)

    #### identical to glm up to this point
    msm, dpred, dpredmean =
        _fit_msm_cox(rng, msmformula, m.data, m.ulfit, contrasts, m.expnms, intvals, m.id, mcsize)

    m.msm = QgcMSM(msm, dpred, reduce(vcat, [fill(iv, mcsize) for iv in intvals]))
    ################
    # bootstrapping
    ################
    m.msm.bootparm_draws = Array{Float64,2}(undef, B, length(coef(m.msm.msmfit)))
    m.msm.meanypred_draws = Array{Float64,2}(undef, B, length(intvals))

    for iter = 1:B
        bootmsmcoef, _, bootypredmean = dobootcox(
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
    ################################
    # summarizing bootstraps
    ################################
    m.msm.bootparm_vcov = cov(m.msm.bootparm_draws)
    m.msm.meanypred_vcov = cov(m.msm.meanypred_draws)
    #output
    psi, vc_psi = coef(m.msm.msmfit), m.msm.bootparm_vcov
    m.fit = (psi, vc_psi)
    m.fitted = true
end

function dobootcox(
    rng,
    id::Vector{ID},
    formula,
    msmformula,
    data,
    family,
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
    ulfit = fit(LSurvival.PHModel, formula, bootdf) # this form needed for proper use of predict function

    # msmfit
    msm, dpred, dpredmean = _fit_msm_cox(rng, msmformula, bootdf, ulfit, contrasts, expnms, intvals, bootid, mcsize)
    #endidx = [j for j in (mcsize .* (1:m.q))]
    #begidx = vcat(1, endidx[1:end-1] .+ 1)
    #dpredmean = [StatsBase.mean([yp.y for yp in ypred[begidx[idx]:endidx[idx]]]) for idx in 1:m.q]
    # output: parameter draws
    coef(msm), dpred, dpredmean
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
    fit_noboot!(m;kwargs...)
    m
end


"""
```julia
using Qgcomp
using LSurvival, DataFrames, Random
rng = MersenneTwister()
# expected effect size in qgcomp for X1, X2 ≈ (truebeta[1] + truebeta[2])/4 = 0.5
truebeta = [4.0, -2.0, 1.0, -1.0, 1.0]
approxpsi = (truebeta[1] + truebeta[2])/4
X, t, d = LSurvival.dgm_phmodel(300; λ=1.25,β=truebeta)
survdata = hcat(DataFrame(X, [:x1, :x2, :z1, :z2, :z3]), DataFrame(hcat(t,d),[:t,:d]))

rng = Xoshiro(1232)

# conditional MSM with fast estimator
qgcomp_cox_noboot(rng, @formula(Surv(t, d)~x1+x2+z1+z2+z3), survdata, ["x1", "x2"], 4)

# conditional MSM with traditional g-computation estimator (conditional on covariates - should look a lot like the "noboot" version)
qgcomp_cox_boot(rng, @formula(Surv(t, d)~x1+x2+z1+z2+z3), survdata, ["x1", "x2"], 4, msmformula=@formula(Surv(t, d)~mixture+z1+z2+z3))

# population MSM with traditional g-computation estimator (necessary, in this case)
qgcomp_cox_boot(rng, @formula(Surv(t, d)~x1+x2+z1+z2+z3), survdata, ["x1", "x2"], 4, msmformula=@formula(Surv(t, d)~mixture+z1+z2+z3))

# non-linear MSM with traditional g-computation estimator (necessary, in this case)
qgcomp_cox_boot(rng, @formula(Surv(t, d)~x1+x2+x1*x2+z1+z2+z3), survdata, ["x1", "x2"], 4, msmformula=@formula(Surv(t, d)~mixture+mixture^2))
```
"""
function qgcomp_cox_boot(rng, formula, data, expnms, q;kwargs...)
    m = QGcomp_cox(formula, data, expnms, q)
    fit_boot!(rng, m;kwargs...)
    m
end

qgcomp_cox_boot(formula, data, expnms, q;kwargs...) = qgcomp_cox_boot(Xoshiro(), formula, data, expnms, q;kwargs...)
;
