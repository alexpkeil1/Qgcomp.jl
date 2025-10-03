# statsbase.jl: re-exports of StatsBase functions

StatsBase.isfitted(m::M) where {M<:QGcomp_model} = m.fitted

StatsBase.coef(ms::Qgcomp_EEmsm) = ms.coef
StatsBase.vcov(ms::Qgcomp_EEmsm) = ms.vcov


function StatsBase.coef(m::M) where {M<:QGcomp_model}
    m.fit[1]
end

function StatsBase.vcov(m::M) where {M<:QGcomp_model}
    m.fit[2]
end


"""
```julia
using Qgcomp, DataFrames, StatsModels

x = rand(100, 3)
z = rand(100, 3)
xq, _ = Qgcomp.get_xq(x, 4)
y = randn(100) + xq * [.1, .05, 0]
data = DataFrame(hcat(y,x,z), [:y, :x1, :x2, :x3, :z1, :z2, :z3])
formula = @formula(y~x1+x2+x3+z1+z2+z3)
expnms = ["x"*string(i) for i in 1:3]
m = Qgcomp.QGcomp_glm(formula, data, expnms, 4, Normal())
m 
fit!(m)
m 
```
"""
function StatsBase.fit!(rng, m::M; bootstrap = false, kwargs...) where {M<:Union{QGcomp_glm,QGcomp_cox}}
    if bootstrap
        fit_boot!(rng, m; kwargs...)
    else
        fit_noboot!(m; kwargs...)
    end
    nothing
end

StatsBase.fit!(
    m::M;
    #contrasts::Dict{Symbol,<:Any} = Dict{Symbol,Any}(), 
    bootstrap = false,
    kwargs...,
) where {M<:Union{QGcomp_glm,QGcomp_cox}} = StatsBase.fit!(Xoshiro(), m; bootstrap = bootstrap, kwargs...)


"""
```julia
n=300
x = rand(n, 3)
z = rand(n, 3)
xq, _ = Qgcomp.get_xq(x, 4)
y = randn(n) + xq * [.1, .05, 0]
data = DataFrame(hcat(y,x,z), [:y, :x1, :x2, :x3, :z1, :z2, :z3])
form = @formula(y~x1+x2+x3+x3+z1+z2+z3)
form2 = @formula(y~x1+x2+x3+x1^2+x2^2+x3^2+x1*z1+z2+z3)
msmformula = @formula(y~mixture+mixture^2)
expnms = [:x1, :x2, :x3]
q = 4
m = QGcomp_glm(form, data, expnms, 4, Normal());
fit!(m)
m2 = QGcomp_ee(form, data, expnms, 4, Normal());
StatsBase.fit!(m2)
m = QGcomp_ee(form, data, expnms, q, Normal())
qgcomp_glm_noboot(form, data, expnms, 4, Normal())
ft = qgcomp_glm_ee(form2, data, expnms, q, Normal(),degree=2)



keys = Dict([
    :msmformula => msmformula,
])
contrasts = Dict{Symbol,Any}()
rr = false


```
"""
function StatsBase.fit!(
    m::QGcomp_ee;
    verbose::Bool = false,
    maxiter::Integer = 500,
    gtol::Float64 = 1e-8,
    start = nothing,
    kwargs...,
)
    validkwargs = (:mcsize, :iters, :intvals, :msmformula, :degree, :msmlink, :msmfamily, :rr, :breaks, :B, :contrasts)
    if !issubset(keys(kwargs), validkwargs)
        throw(ArgumentError("Qgcomp error: unsupported keyword argument in: $(kwargs...)"))
    end
    ################
    # specifying overall effect definition
    ################
    #intvals = :intvals ∈ keys(kwargs) ? kwargs[:intvals] : (1:m.q) .- 1.0
    if :intvals ∈ keys(kwargs)
        m.intvals = kwargs[:intvals]
    end
    #intvals = m.intvals

    ################
    # model form arguments via contrasts
    ################
    if (:contrasts ∈ keys(kwargs))
        m.contrasts = kwargs[:contrasts]
    else
        m.contrasts = Dict{Symbol,Any}()
    end
    # msm formula
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
    # msm glm family/link
    rr = false
    if (:msmlink ∈ keys(kwargs))
        msmlink = kwargs[:msmlink]
    else
        if (:rr ∈ keys(kwargs))
            rr = kwargs[:rr]
        end
        msmlink = rr ? LogLink() : m.link
    end

    if (:msmfamily ∈ keys(kwargs))
        msmfamily = kwargs[:msmfamily]
    else
        msmfamily = m.family
    end
    # design matrix for underlying model
    sch = schema(m.formula, m.data, m.contrasts)
    f = apply_schema(m.formula, sch, typeof(m))
    # need a few obs in case there are splines
    Y, X = modelcols(f, m.data[1:min(end, 5), :])
    pcond = size(X, 2)

    pmsm = size(fake_design(msmformula, m.contrasts, typeof(m)), 2)

    inits = zeros(pcond+pmsm)

    if (m.family==Binomial()) && rr
        inits[(pcond+1):(pcond+pmsm)] .= -1.0
    end
    res = nlsolve(
        x -> qgcomp_eedf( x, m.data, m.expnms, m.intvals, f, msmformula, pcond, pmsm, m.family, m.link, msmfamily, msmlink, m.contrasts, ),
        inits,
        autodiff = :forward,
    )

    A = ForwardDiff.jacobian(
        x -> qgcomp_eedf( x, m.data, m.expnms, m.intvals, f, msmformula, pcond, pmsm, m.family, m.link, msmfamily, msmlink, m.contrasts, ),
        res.zero,
    )   # d/dbeta psi(y,x,beta)
    # TODO: add id to model: currently uid is a proxy for "row"
    uid = 1:size(m.data, 1)
    psis = [
        qgcomp_eedf( res.zero, DataFrame(m.data[id, :]), m.expnms, m.intvals, f, msmformula, pcond, pmsm, m.family, m.link, msmfamily, msmlink, m.contrasts, ) for id in uid
    ]
    Bi = [psii * psii' for psii in psis];                # psi_i(y,x,beta)psi_i(y,x,beta)'
    B = reduce(+, Bi)
    iA = inv(A)
    covmat = iA * B * iA'  # sandwich variance estimate    
    vc_psi = covmat[(pcond+1):(pcond+pmsm), (pcond+1):(pcond+pmsm)]
    psi = res.zero[(pcond+1):(pcond+pmsm)]
    m.fullfit = (res.zero, covmat)
    m.fit = (psi, vc_psi)
    _, Yp, _, _, Ypmsm, Xmsm, Xmsmdf = qgcomp_eeprep( res.zero, m.data, m.expnms, m.intvals, f, msmformula, pcond, pmsm, m.family, m.link, msmfamily, msmlink, m.contrasts, )
    msmformula.rhs
    m.msm = Qgcomp_MSM( Qgcomp_EEmsm(m.fit, psi, vc_psi, Xmsmdf), psi, zeros(length(m.intvals)), Yp, float.(Xmsmdf.mixture), msmfamily, msmlink, m.contrasts, msmformula, )
    m.msm.ypredmsm = Ypmsm
    #mnolin_ee.msm.ypredmsm
    m.ulfit = coeftable_eeul(m, limitcond = false)
    m.fitted = true
    nothing
end

for f in (:loglikelihood, :aic, :aicc, :bic, :fitted)
    @eval begin
        StatsBase.$f(m::QGcomp_glm) = StatsBase.$f(m.ulfit)
    end
end


function gencoeftab(m, level)
    β = m.fit[1]
    se = sqrt.(diag(m.fit[2]))
    z = β ./ se
    p = 2 * cdf.(Normal(), .-abs.(z))
    crit = quantile.(Normal(), [(1.0 - level) / 2.0, 1.0 .- (1.0 - level) / 2.0])[:, :]'
    ci = β .+ se[:, :] * crit
    hcat(β, se, z, p, ci)
end


function StatsBase.coeftable(m::M; level = 0.95) where {M<:Union{QGcomp_glm,QGcomp_cox}}
    #contrasts = Dict{Symbol,Any}()
    sch = schema(m.formula, m.data, m.contrasts)
    f = apply_schema(m.formula, sch, typeof(m))
    hasintercept = StatsModels.hasintercept(f)# any([typeof(t)<:InterceptTerm for t in f.rhs.terms])
    coeftab = gencoeftab(m, level)
    colnms = ["Coef.", "Std. Error", "z", "Pr(>|z|)", "Lower 95%", "Upper 95%"]
    rownms = isnothing(m.msm) ? coefnames(m.ulfit) : coefnames(m.msm.msmfit)
    if typeof(rownms) <: String
        rownms = [rownms]
    end
    if isnothing(m.msm)
        nonpsi = setdiff(rownms, String.(m.expnms))
        #if !hasintercept(m.formula) 
        #end
        if hasintercept
            rownms = popfirst!(nonpsi)
        else
            rownms = []
        end
        rownms = vcat(rownms, "mixture")
        rownms = vcat(rownms, nonpsi)
    end
    CoefTable(coeftab, colnms, rownms, 4, 3)
end


function StatsBase.coeftable(m::QGcomp_ee; level = 0.95)
    #contrasts = Dict{Symbol,Any}()
    sch = schema(m.formula, m.data, m.contrasts)
    f = apply_schema(m.formula, sch, typeof(m))

    hasintercept = StatsModels.hasintercept(f)# any([typeof(t)<:InterceptTerm for t in f.rhs.terms])

    coeftab = gencoeftab(m, level)

    colnms = ["Coef.", "Std. Error", "z", "Pr(>|z|)", "Lower 95%", "Upper 95%"]
    #rownms = ["ψ$i" for i = 1:size(coeftab, 1)]
    rownms = get_rownms_msm(m)

    CoefTable(coeftab, colnms, rownms, 4, 3)
end
