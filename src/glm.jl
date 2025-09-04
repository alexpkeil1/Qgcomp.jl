#glm.jl

####################################################################################
#
# GLM based methods
#
####################################################################################


function QGcomp_glm(formula, data, expnms::Vector{Symbol}, q, family, breaks, ulfit, fit, fitted)
    QGcomp_glm(formula, data, String.(expnms), q, family, breaks, nothing, nothing, false)
end

function QGcomp_glm(formula, data, expnms, q, family)
    qdata, breaks, _ = get_dfq(data, expnms, q)
    QGcomp_glm(formula, qdata, expnms, q, family, breaks, nothing, nothing, false)
end


"""
using Qgcomp, DataFrames, StatsModels

x = rand(100, 3)
z = rand(100, 3)
xq, _ = Qgcomp.get_xq(x, 4)
y = randn(100) + xq * [.1, .05, 0]
data = DataFrame(hcat(y,x,z), [:y, :x1, :x2, :x3, :z1, :z2, :z3])
formula = @formula(y~x1+x2+x3+z1+z2+z3)
expnms = ["x"*string(i) for i in 1:3]
m = Qgcomp.QGcomp_glm(formula, data, expnms, 4, Normal())
m |> display
fit!(m)
m |> display
"""
function fit!(m::QGcomp_glm)
    m.ulfit = fit(GeneralizedLinearModel, m.formula, m.data, m.family)
    coefs = coef(m.ulfit)
    vcnames = coefnames(m.ulfit)
    vc = vcov(m.ulfit)
    vc_psi = vccomb(vc, vcnames, m.expnms)
    psi = psicomb(coefs, vcnames, m.expnms)
    m.fit = (psi, vc_psi)
    m.fitted = true
    nothing
end


function Base.show(io::IO, m::QGcomp_glm)
    if isfitted(m)
        println(io, coeftable(m))
    else
        println(io, "Not fitted")
    end
end


Base.show(m::QGcomp_glm) = Base.show(stdout, m)

"""
using Qgcomp, DataFrames, StatsModels

x = rand(100, 3)
z = rand(100, 3)
xq, _ = Qgcomp.get_xq(x, 4)
y = randn(100) + xq * [.1, 0.05, 0]
data = DataFrame(hcat(y,x,z), [:y, :x1, :x2, :x3, :z1, :z2, :z3])
form = @formula(y~x1+x2+x3+z1+z2+z3)
expnms = [:x1, :x2, :x3]

m = qgcomp_glm_noboot(form, data, expnms, 4, Normal())
aic(m)
aicc(m)
bic(m)
loglikelihood(m)
"""
function qgcomp_glm_noboot(formula, data, expnms, q, family)
    m = QGcomp_glm(formula, data, expnms, q, family)
    fit!(m)
    m
end


#stats base functions
StatsBase.loglikelihood(m::QGcomp_glm) = StatsBase.loglikelihood(m.ulfit)
StatsBase.aic(m::QGcomp_glm) = StatsBase.aic(m.ulfit)
StatsBase.aicc(m::QGcomp_glm) = StatsBase.aicc(m.ulfit)
StatsBase.bic(m::QGcomp_glm) = StatsBase.bic(m.ulfit)
StatsBase.isfitted(m::QGcomp_glm) = m.fitted


#TODO: bootstrapping


;