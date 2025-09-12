# statsbase.jl: re-exports of StatsBase functions

StatsBase.isfitted(m::QGcomp_glm) = m.fitted




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
function StatsBase.fit!(
    rng,
    m::QGcomp_glm;
    contrasts::Dict{Symbol,<:Any}=Dict{Symbol,Any}(),
    bootstrap=false,
    kwargs...,
)
    if bootstrap
        fit_boot!(rng, m; contrasts=contrasts, kwargs...)
    else
        fit_noboot!(m)
    end
    nothing
end

function StatsBase.fit!(m::QGcomp_glm; contrasts::Dict{Symbol,<:Any}=Dict{Symbol,Any}(), bootstrap=false, kwargs...)
    if bootstrap
        fit_boot!(m; contrasts=contrasts, kwargs...)
    else
        fit_noboot!(m)
    end
    nothing
end


for f in (:loglikelihood, :aic, :aicc, :bic, :fitted)
    @eval begin
        StatsBase.$f(m::QGcomp_glm) = StatsBase.$f(m.ulfit)
    end
end



function StatsBase.coeftable(m::M; level = 0.95) where {M<:Union{QGcomp_glm,QGcomp_cox}}
    β = m.fit[1]
    se = sqrt.(diag(m.fit[2]))
    z = β ./ se
    p = 2 * cdf.(Normal(), .-abs.(z))
    ci = β .+ se[:, :] * quantile.(Normal(), [(1.0 - level) / 2.0, 1.0 .- (1.0 - level) / 2.0])[:, :]'
    coeftab = hcat(β, se, z, p, ci)
    colnms = ["Coef.", "Std. Error", "z", "Pr(>|z|)", "Lower 95%", "Upper 95%"]
    rownms = isnothing(m.msm) ? coefnames(m.ulfit) : coefnames(m.msm.msmfit)
    if isnothing(m.msm)
        nonpsi = setdiff(rownms, m.expnms)
        #if !hasintercept(m.formula) 
        #end
        rownms = popfirst!(nonpsi)
        rownms = vcat(rownms, "ψ")
        rownms = vcat(rownms, nonpsi)
    end
    CoefTable(coeftab, colnms, rownms, 4, 3)
end


function StatsBase.coeftable(m::QGcomp_ee; level = 0.95)
    contrasts = Dict{Symbol,Any}()
    sch = schema(m.formula, m.data, contrasts)
    f = apply_schema(m.formula, sch, typeof(m))
    #Y,X =  modelcols(f, m.data[1:1,:])

     β = m.fit[1]
    se = sqrt.(diag(m.fit[2]))
    z = β ./ se
    p = 2 * cdf.(Normal(), .-abs.(z))
    ci =
        β .+
        se[:, :] *
        quantile.(Normal(), [(1.0 - level) / 2.0, 1.0 .- (1.0 - level) / 2.0])[:, :]'
    coeftab = hcat(β, se, z, p, ci)
    colnms = ["Coef.", "Std. Error", "z", "Pr(>|z|)", "Lower 95%", "Upper 95%"]
    rownms = ["ψ$i" for i in 1:size(coeftab,1)]
    if (typeof(f.rhs.terms[1]) == InterceptTerm{true})
        rownms = vcat("(Intercept)", rownms[1:(end-1)])
    end
    CoefTable(coeftab, colnms, rownms, 4,3)
end