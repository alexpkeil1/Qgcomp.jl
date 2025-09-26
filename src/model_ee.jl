#ee.jl


####################################################################################
#
# Estimating equation methods
#
####################################################################################


function QGcomp_ee(formula, data, expnms, q, family, link, breaks, intvals)
    QGcomp_ee(
        formula,
        data,
        String.(expnms),
        q,
        family,
        link,
        breaks,
        intvals,
        nothing,
        nothing,
        nothing,
        false,
        nothing,
        nothing,
        QGcomp_weights(),
        Dict{Symbol,Any}(),
    )
end

function QGcomp_ee(formula, data, expnms, q, family, link, breaks)
    qdata, breaks, intvals = get_dfq(data, expnms; breaks=breaks)
    QGcomp_ee(formula, qdata, expnms, q, family, link, breaks, intvals)
end

function QGcomp_ee(formula, data, expnms, q, family, link)
    qdata, breaks, intvals = get_dfq(data, expnms, q)
    QGcomp_ee(formula, qdata, expnms, q, family, link, breaks, intvals)
end

function QGcomp_ee(formula, data, expnms, q, family)
    qdata, breaks, intvals = get_dfq(data, expnms, q)
    QGcomp_ee(formula, qdata, expnms, q, family, canonicallink(family))
end

function Qgcomp_EEmsm(tuplefit)
    Qgcomp_EEmsm(tuplefit, tuplefit[1], tuplefit[2])
end



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
    m.intvals = :intvals ∈ keys(kwargs) ? kwargs[:intvals] : (1:m.q) .- 1.0

    ################
    # model form arguments via contrasts
    ################
    if (:contrasts ∈ keys(kwargs))
        contrasts = kwargs[:contrasts]
    else
        contrasts = Dict{Symbol,Any}()
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
    sch = schema(m.formula, m.data, contrasts)
    f = apply_schema(m.formula, sch, typeof(m))
    Y, X = modelcols(f, m.data[1:1, :])
    pcond = size(X, 2)

    pmsm = size(fake_design(msmformula, contrasts, typeof(m)), 2)

    inits = zeros(pcond+pmsm)

    if (m.family==Binomial()) && rr
        inits[(pcond+1):(pcond+pmsm)] .= -1.0
    end

    #res = nlsolve(x -> linreg_ee(Y,X,p,x), inits, autodiff = :forward)

    res = nlsolve(
        x -> qgcomp_eedf( x, m.data, m.expnms, m.intvals, f, msmformula, pcond, pmsm, m.family, m.link, msmfamily, msmlink, contrasts ),
        inits,
        autodiff = :forward,
    )

    A = ForwardDiff.jacobian(
        x -> qgcomp_eedf( x, m.data, m.expnms, m.intvals, f, msmformula, pcond, pmsm, m.family, m.link, msmfamily, msmlink, contrasts ),
        res.zero,
    )   # d/dbeta psi(y,x,beta)
    # TODO: add id to model: currently uid is a proxy for "row"
    uid = 1:size(m.data, 1)
    psis = [ qgcomp_eedf( res.zero, DataFrame(m.data[id, :]), m.expnms, m.intvals, f, msmformula, pcond, pmsm, m.family, m.link, msmfamily, msmlink, contrasts ) for id in uid ]
    Bi = [psii * psii' for psii in psis];                # psi_i(y,x,beta)psi_i(y,x,beta)'
    B = reduce(+, Bi)
    iA = inv(A)
    covmat = iA * B * iA'  # sandwich variance estimate    
    vc_psi = covmat[(pcond+1):(pcond+pmsm), (pcond+1):(pcond+pmsm)]
    psi = res.zero[(pcond+1):(pcond+pmsm)]
    m.fullfit = (res.zero, covmat)
    m.fit = (psi, vc_psi)
    _, Yp, _, _, Ypmsm, Xmsm, Xmsmdf = qgcomp_eeprep( res.zero, m.data, m.expnms, m.intvals, f, msmformula, pcond, pmsm, m.family, m.link, msmfamily, msmlink, contrasts )
    msmformula.rhs

    m.msm = Qgcomp_MSM( Qgcomp_EEmsm(m.fit, psi, vc_psi, Xmsmdf), psi, zeros(length(m.intvals)), Yp, float.(Xmsmdf.mixture), msmfamily, msmlink, contrasts, msmformula)
    m.msm.ypredmsm = Ypmsm
    #mnolin_ee.msm.ypredmsm
    m.ulfit = coeftable_eeul(m, limitcond = false)
    m.fitted = true
    nothing
end

expit(mu) = inv(1.0 + exp(-mu))

function glm_canonical_ee(Y, X, p, theta, linkfun)
    # can replace Poisson, Linreg, LogitReg
    if (length(Y)==0)
        return (fill(0, length(theta)))
    end
    mu = X * theta
    reduce(vcat, [(Y .- GLM.linkinv.(linkfun, mu))' * X[:, c:c] for c = 1:size(X, 2)])
end


function linreg_ee(Y, X, p, theta)
    if (length(Y)==0)
        return (fill(0, length(theta)))
    end
    mu = X * theta
    reduce(vcat, [(Y .- mu)' * X[:, c:c] for c = 1:size(X, 2)])
end

function logitreg_ee(Y, X, p, theta)
    if (length(Y)==0)
        return (fill(0, length(theta)))
    end
    mu = X * theta
    reduce(vcat, [(Y .- expit.(mu))' * X[:, c:c] for c = 1:size(X, 2)])
end

function poissonreg_ee(Y, X, p, theta)
    if (length(Y)==0)
        return (fill(0, length(theta)))
    end
    mu = X * theta
    reduce(vcat, [(Y .- exp.(mu))' * X[:, c:c] for c = 1:size(X, 2)])
end

function logbinreg_ee(Y, X, p, theta)
    if (length(Y)==0)
        return (fill(0, length(theta)))
    end
    mu = X * theta
    reduce(vcat, [((Y .- exp.(mu)) ./ (1.0 .- exp.(mu)))' * X[:, c:c] for c = 1:size(X, 2)])
end

function dfint(df, expnms, int)
    df2 = deepcopy(df)
    df2[:, expnms] .= int
    df2
end

#=
using DataFrames, Qgcomp
x = rand(100, 3)
z = rand(100, 3)
xq, _ = Qgcomp.get_xq(x, 4)
y = randn(100) + xq * [.1, .05, 0]
data = DataFrame(hcat(y,x,z), [:y, :x1, :x2, :x3, :z1, :z2, :z3])
df = data
formula = @formula(y~1+x1+x2+x3+z1+z2+z3)
msmformula = @formula(y~mixture + mixture^2)
f = formula
y,X = modelcols(formula, data)
m = QGcomp_glm(formula, data, expnms, 4, Normal())
fit!(m)
expnms = ["x1", "x2", "x3"]
Qgcomp.dfint(data, expnms, 1)
dfint = Qgcomp.dfint
intvals = 0:3
theta = rand(9)
QGcomp_model = Qgcomp.QGcomp_model
=#

function qgcomp_eeprep(theta, df, expnms, intvals, f, msmformula, p1, p2, family, link, msmfamily, msmlink, contrasts)    
    # original data
    sch = schema(f, df, contrasts)
    ff = apply_schema(f, sch, QGcomp_model)
    Y,X = modelcols(ff, df)
    # prediction data
    Xintl = [dfint(df, expnms, int) for int in intvals]
    Xintdf = reduce(vcat, Xintl) # full copy of data with exposures modified
    sch = schema(f, Xintdf, contrasts)
    ff1 = apply_schema(f, sch, QGcomp_model)
    Xint = modelcols(ff1, Xintdf)[2]
    # MSM data
    Xintdf.mixture = Xintdf[:,expnms[1]]
    Xintdf = Xintdf[:,[n for n in names(Xintdf) if n ∉ expnms]]
    sch = schema(msmformula, Xintdf, contrasts)
    ff2 = apply_schema(msmformula, sch, QGcomp_model)
    Xmsm = modelcols(ff2, Xintdf)[2]
    #Xmsmdf = ModelFrame(ff2, Xintdf, sch, QGcomp_model)

    if (family ∈ [Normal(), Poisson(), Binomial()])
        Yp = GLM.linkinv.(link, Xint * theta[1:p1]) # ypred
    else
        stop("Link/family unsupported")
    end
    if (family ∈ [Normal(), Poisson(), Binomial()])
        Ypmsm = GLM.linkinv.(msmlink, Xmsm * theta[(p1+1):(p1+p2)]) # ypred 
    end
    Y, Yp, X, Xint, Ypmsm, Xmsm, Xintdf
end




function qgcomp_eedf(theta, df, expnms, intvals, f, msmformula, p1, p2, family, link, msmfamily, msmlink, contrasts)
    Y, Yp, X, Xint, Ypmsm, Xmsm, _ =
        qgcomp_eeprep(theta, df, expnms, intvals, f, msmformula, p1, p2, family, link, msmfamily, msmlink, contrasts)

    # TODO: look for edge cases here
    if (family == msmfamily) &&
       (link == msmlink) &&
       (family ∈ [Normal(), Poisson(), Binomial()]) &&
       (link == canonicallink(family))
        # canonical link GLMs all have the same estimating equation
        Yp = GLM.linkinv.(link, Xint * theta[1:p1]) # ypred
        vcat(
            glm_canonical_ee(Y, X, p1, theta[1:p1], link),
            glm_canonical_ee(Yp, Xmsm, p2, theta[(p1+1):(p1+p2)], msmlink),
        )
    elseif (family == Binomial()) && (msmfamily == Binomial()) && (link == LogitLink()) && (msmlink == LogLink())
        Yp = expit.(Xint * theta[1:p1])
        vcat(
            #logitreg_ee(Y, X, p1, theta[1:p1]), 
            glm_canonical_ee(Y, X, p1, theta[1:p1], link),
            logbinreg_ee(Yp, Xmsm, p2, theta[(p1+1):(p1+p2)]),
        )
        #=
        elseif family == Normal()
            Yp = Xint * theta[1:p1] # ypred
            vcat(linreg_ee(Y, X, p1, theta[1:p1]), linreg_ee(Yp, Xmsm, p2, theta[(p1+1):(p1+p2)]))
        elseif family == Poisson()
            Yp = exp.(Xint * theta[1:p1])
            vcat(poissonreg_ee(Y, X, p1, theta[1:p1]), poissonreg_ee(Yp, Xmsm, p2, theta[(p1+1):(p1+p2)]))
        elseif (family == Binomial()) && (rr == false)
            Yp = expit.(Xint * theta[1:p1])
            vcat(logitreg_ee(Y, X, p1, theta[1:p1]), logitreg_ee(Yp, Xmsm, p2, theta[(p1+1):(p1+p2)]))
        =#
    else
        stop("Link/family combination not currently supported by underlying qgcomp_eedf function")
    end
end


function coeftable_eeul(m::QGcomp_ee; level = 0.95, limitcond = false)
    sch = schema(m.formula, m.data, m.contrasts)
    f = apply_schema(m.formula, sch, typeof(m))
    Y, X = modelcols(f, m.data[1:1, :])
    pmsm = length(m.fit[1])
    β = m.fullfit[1]
    se = sqrt.(diag(m.fullfit[2]))
    z = β ./ se
    p = 2 * cdf.(Normal(), .-abs.(z))
    ci = β .+ se[:, :] * quantile.(Normal(), [(1.0 - level) / 2.0, 1.0 .- (1.0 - level) / 2.0])[:, :]'
    coeftab = hcat(β, se, z, p, ci)
    ptot = size(coeftab, 1)
    pcond = ptot - pmsm
    colnms = ["Coef.", "Std. Error", "z", "Pr(>|z|)", "Lower 95%", "Upper 95%"]
    rownms1 = coefnames(f)[2]#["β$i" for i in 1:pcond]
    #rownms2 = ["ψ$i" for i = 1:pmsm]
    rownms2 = get_rownms_msm(m)

    if (typeof(f.rhs.terms[1]) == InterceptTerm{true})
        #rownms1 = vcat("(Intercept)", rownms1[1:(end-1)])
        #rownms2 = vcat("(Intercept, MSM)", rownms2[1:(end-1)])
        rownms2[1] = "(Intercept, MSM)"
    end
    if limitcond
        res = CoefTable(coeftab[1:pcond, :], colnms, vcat(rownms1, rownms2)[1:pcond], 4, 3)
    else
        res = CoefTable(coeftab, colnms, vcat(rownms1, rownms2), 4, 3)
    end
    res
end


function Base.show(io::IO, m::QGcomp_ee)
    if m.fitted
        println(io, "Underlying fit (estimating equation based CI)")
        println(io, coeftable_eeul(m, limitcond = true))
        show(io, m.qgcweights)
        println(io, "\nMSM (estimating equation based CI)")
        println(io, coeftable(m))
    else
        println(io, "Not fitted")
    end
end

Base.show(m::QGcomp_ee) = Base.show(stdout, m)

"""
# binary
```julia
dat = DataFrame(y=Int.(rand(Bernoulli(0.25), 50)), x1=rand(50), x2=rand(50), z=rand(50))

# Marginal mixture OR (no confounders)
ft1 = qgcomp_glm_noboot(@formula(y ~ x1 + x2), dat,["x1", "x2"], nothing, Binomial())
ft2 = qgcomp_glm_ee(@formula(y ~ x1 + x2), dat,["x1", "x2"], nothing, Binomial())
ft3 = qgcomp_glm_ee(@formula(y ~ x1 + x2), dat,["x1", "x2"], nothing, Binomial(), rr=true)

# Conditional mixture OR
qgcomp_glm_noboot(@formula(y ~ z + x1 + x2), dat,["x1", "x2"], 4, Binomial())
# Marginal mixture OR
qgcomp_glm_ee(@formula(y ~ z + x1 + x2), dat,["x1", "x2"], 4, Binomial())
```
"""
function qgcomp_glm_ee(formula, data, expnms, q, family; kwargs...)
    if (:breaks ∈ keys(kwargs)) && (:link ∉ keys(kwargs)) 
        m = QGcomp_ee(formula, data, expnms, q, family, canonicallink(family), kwargs[:breaks])
    elseif (:breaks ∈ keys(kwargs)) && (:link ∈ keys(kwargs)) 
        m = QGcomp_ee(formula, data, expnms, q, family, kwargs[:link], kwargs[:break])
    else (:breaks ∈ keys(kwargs))
        m = QGcomp_ee(formula, data, expnms, q, family)
    end
    #m = QGcomp_ee(formula, data, expnms, q, family)
    fit!(m; kwargs...)
    m
end


