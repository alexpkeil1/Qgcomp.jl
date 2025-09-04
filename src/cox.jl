#cox.jl

####################################################################################
#
# Cox model based methods
#
####################################################################################



function QGcomp_cox(formula, data, expnms::Vector{Symbol}, q, family, breaks, ulfit, fit, fitted)
    QGcomp_cox(formula, data, String.(expnms), q, family, breaks, nothing, nothing, false)
end

function QGcomp_cox(formula, data, expnms, q)
    qdata, breaks, _ = get_dfq(data, expnms, q)
    QGcomp_cox(formula, qdata, expnms, q, "Cox", breaks, nothing, nothing, false)
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


function Base.show(io::IO, m::QGcomp_cox)
    if m.fitted 
        println(io, coeftable(m))
    else
        println(io, "Not fitted")
    end

end

Base.show(m::QGcomp_cox) = Base.show(stdout, m)

"""
using Qgcomp
using LSurvival, DataFrames, Random
id, int, out, data = LSurvival.dgm(MersenneTwister(1212), 100 20);

data[:, 1] = round.(data[:, 1], digits = 3);
d, X = data[:, 4], data[:, 1:3];
wt = ones(length(d)) # random weights just to demonstrate usage
tab = ( in = int, out = out, d=d, x=X[:,1], z1=X[:,2], z2=X[:,3]) ;
df = DataFrame(tab)

coxph(@formula(Surv(in, out, d)~x+z1+z2), tab, ties = "efron", wts = wt) |> display
qgcomp_cox_noboot(@formula(Surv(in, out, d)~x+z1+z2), df, ["z1", "z2"], 4) |> display

"""
function qgcomp_cox_noboot(formula, data, expnms, q;kwargs...)
    m = QGcomp_cox(formula, data, expnms, q)
    fit!(m;kwargs...)
    m
end
;
