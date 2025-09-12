#ee.jl


####################################################################################
#
# Estimating equation methods
#
####################################################################################


function QGcomp_ee(formula, data, expnms::Vector{Symbol}, q, family, breaks, intvals, fullfit, ulfit, fit, fitted, bootstrap, id)
    QGcomp_ee(formula, data, String.(expnms), q, family, breaks, intvals, nothing, nothing, nothing, false, nothing, nothing)
end

function QGcomp_ee(formula, data, expnms, q, family)
    qdata, breaks, intvals = get_dfq(data, expnms, q)
    QGcomp_ee(formula, qdata, expnms, q, family, breaks, intvals, nothing, nothing, nothing, false, nothing, nothing)
end

"""
```julia
n=300
x = rand(n, 3)
z = rand(n, 3)
xq, _ = get_xq(x, 4)
y = randn(n) + xq * [.1, .05, 0]
data = DataFrame(hcat(y,x,z), [:y, :x1, :x2, :x3, :z1, :z2, :z3])
form = @formula(y~x1+x2+x3+x3+z1+z2+z3)
form2 = @formula(y~x1+x2+x3+x1^2+x2^2+x3^2+x1*z1+z2+z3)
expnms = [:x1, :x2, :x3]
q = 4
m = QGcomp_glm(form, data, expnms, 4, Normal());
fit!(m)
m2 = QGcomp_ee(form, data, expnms, 4, Normal());
StatsBase.fit!(m2)
m = QGcomp_ee(form, data, expnms, q, Normal())
qgcomp_glm_noboot(form, data, expnms, 4, Normal())
ft = qgcomp_glm_ee(form2, data, expnms, q, Normal(),degree=2)
```
"""
function StatsBase.fit!(m::QGcomp_ee;
    rr::Bool = false,
    degree::Integer=1,
    verbose::Bool = false,
    maxiter::Integer = 500,
    gtol::Float64 = 1e-8,
    start = nothing,
    contrasts = Dict{Symbol,Any}()
)
    sch = schema(m.formula, m.data, contrasts)
    f = apply_schema(m.formula, sch, typeof(m))
    Y,X =  modelcols(f, m.data[1:1,:])
    pcond = size(X,2)
    pmsm = (typeof(f.rhs.terms[1]) <: InterceptTerm) + degree
    inits = zeros(pcond+pmsm)
    if (m.family==Binomial()) && rr
        inits[(pcond+1):(pcond+pmsm)] .= -1.0
    end
    #res = nlsolve(x -> linreg_ee(Y,X,p,x), inits, autodiff = :forward)

    res = nlsolve(x -> qgcomp_eedf(x, m.data, m.expnms, m.intvals, f, pcond, pmsm, degree, m.family, rr), inits, autodiff = :forward)

    A = ForwardDiff.jacobian(x -> qgcomp_eedf(x, m.data, m.expnms, m.intvals, f, pcond, pmsm, degree, m.family,rr), res.zero)   # d/dbeta psi(y,x,beta)
    # TODO: add id to model
    uid = 1:size(m.data,1)
    psis = [qgcomp_eedf(res.zero, DataFrame(m.data[id,:]),m.expnms, m.intvals, f, pcond, pmsm, degree, m.family,rr) for id in uid]
    Bi = [psii * psii' for psii in psis];                # psi_i(y,x,beta)psi_i(y,x,beta)'
    B = reduce(+, Bi)
    iA = inv(A)
    covmat = iA * B * iA'  # sandwich variance estimate    
    vc_psi = covmat[(pcond+1):(pcond+pmsm),(pcond+1):(pcond+pmsm)]
    psi = res.zero[(pcond+1):(pcond+pmsm)]
    m.fullfit = (res.zero, covmat)
    m.fit = (psi, vc_psi)
    m.ulfit = coeftable_eeul(m, limitcond=false)
    m.fitted = true
    nothing
end

expit(mu) = inv(1.0 + exp(-mu))

function linreg_ee(Y,X,p,theta)
    if(length(Y)==0) 
        return(fill(0,length(theta)))
    end
    mu = X * theta
    reduce(vcat,[(Y .- mu)' * X[:,c:c] for c in 1:size(X,2)])
end

function logitreg_ee(Y,X,p,theta)
    if(length(Y)==0) 
        return(fill(0,length(theta)))
    end
    mu = X * theta
    reduce(vcat,[(Y .- expit.(mu))' * X[:,c:c] for c in 1:size(X,2)])
end

function logbinreg_ee(Y,X,p,theta)
    if(length(Y)==0) 
        return(fill(0,length(theta)))
    end
    mu = X * theta
    reduce(vcat,[((Y .- exp.(mu)) ./ (1.0 .- exp.(mu)))' * X[:,c:c] for c in 1:size(X,2)])
end


function poissonreg_ee(Y,X,p,theta)
    if(length(Y)==0) 
        return(fill(0,length(theta)))
    end
    mu = X * theta
    reduce(vcat,[(Y .- exp.(mu))' * X[:,c:c] for c in 1:size(X,2)])
end



function dfint(df,expnms,int)
    df2 = deepcopy(df)
    df2[:,expnms] .= int
    df2
end

#=
x = rand(100, 3)
z = rand(100, 3)
xq, _ = get_xq(x, 4)
y = randn(100) + xq * [.1, .05, 0]
data = DataFrame(hcat(y,x,z), [:y, :x1, :x2, :x3, :z1, :z2, :z3])
formula = @formula(y~1+x1+x2+x3+z1+z2+z3)
m = QGcomp_glm(formula, data, expnms, 4, Normal())
fit!(m)
expnms = ["x1", "x2", "x3"]
intvals = 0:3
theta = rand(9)
=#
function qgcomp_eedf(theta, df, expnms, intvals, f, p1, p2, degree, family, rr)
    Y, X =  modelcols(f, df)
    n,p = size(X)
    Xintl = [modelcols(f, dfint(df,expnms,int))[2] for int in intvals]
    Xint = reduce(vcat, Xintl)
    Xmsm = reduce(hcat, [reduce(vcat,[fill(int,n) for int in intvals])[:,:].^d for d in 1:degree])
    if p2 > degree
        Xmsm = hcat(ones(size(Xmsm,1)), Xmsm)
    end
    if family == Normal()
        Yp = Xint * theta[1:p1]
        vcat(
          linreg_ee(Y,X,p1,theta[1:p1]),
          linreg_ee(Yp,Xmsm,p2,theta[(p1+1):(p1+p2)])
      )
    elseif family == Poisson()
        Yp = exp.(Xint * theta[1:p1])
        vcat(
            poissonreg_ee(Y,X,p1,theta[1:p1]),
            poissonreg_ee(Yp,Xmsm,p2,theta[(p1+1):(p1+p2)])
        )
    elseif (family == Binomial()) && (rr == false)
        Yp = expit.(Xint * theta[1:p1])
        vcat(
            logitreg_ee(Y,X,p1,theta[1:p1]),
            logitreg_ee(Yp,Xmsm,p2,theta[(p1+1):(p1+p2)])
        )
    elseif (family == Binomial()) && (rr == true)
        Yp = expit.(Xint * theta[1:p1])
        vcat(
            logitreg_ee(Y,X,p1,theta[1:p1]),
            logbinreg_ee(Yp,Xmsm,p2,theta[(p1+1):(p1+p2)])
        )
    end
end



function coeftable_eeul(m::QGcomp_ee; level = 0.95, limitcond=false)
    contrasts = Dict{Symbol,Any}()
    sch = schema(m.formula, m.data, contrasts)
    f = apply_schema(m.formula, sch, typeof(m))
    Y,X =  modelcols(f, m.data[1:1,:])
    pmsm = length(m.fit[1])
    β = m.fullfit[1]
    se = sqrt.(diag(m.fullfit[2]))
    z = β ./ se
    p = 2 * cdf.(Normal(), .-abs.(z))
    ci =
        β .+
        se[:, :] *
        quantile.(Normal(), [(1.0 - level) / 2.0, 1.0 .- (1.0 - level) / 2.0])[:, :]'
    coeftab = hcat(β, se, z, p, ci)
    ptot = size(coeftab,1)
    pcond = ptot - pmsm
    colnms = ["Coef.", "Std. Error", "z", "Pr(>|z|)", "Lower 95%", "Upper 95%"]
    rownms1 = coefnames(f)[2]#["β$i" for i in 1:pcond]
    rownms2 = ["ψ$i" for i in 1:pmsm]
    if (typeof(f.rhs.terms[1]) == InterceptTerm{true})
        #rownms1 = vcat("(Intercept)", rownms1[1:(end-1)])
        rownms2 = vcat("(Intercept, MSM)", rownms2[1:(end-1)])
    end
    #nonpsi = setdiff(coefnames(m.ulfit), m.expnms)
    #rownms = popfirst!(nonpsi)
    #rownms = vcat(rownms, "ψ")
    #rownms = vcat(rownms, nonpsi)
    #CoefTable(coeftab, colnms, rownms, 4,3)
    if limitcond
       res = CoefTable(coeftab[1:pcond,:], colnms, vcat(rownms1,rownms2)[1:pcond], 4,3)
    else 
        res = CoefTable(coeftab, colnms, vcat(rownms1,rownms2), 4,3)
    end
    res
end


function Base.show(io::IO, m::QGcomp_ee)
    if m.fitted 
        println(io, "Underlying fit")
        println(io, coeftable_eeul(m, limitcond=true))
        println(io, "\nMSM")
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
    m = QGcomp_ee(formula, data, expnms, q, family)
    fit!(m;kwargs...)
    m
end


