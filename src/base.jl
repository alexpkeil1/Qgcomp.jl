# base.jl: core helper functions from quantile g-computation


function QGcomp_weights()
    QGcomp_weights(DataFrame(), DataFrame(), false)
end


function Base.show(io::IO, w::QGcomp_weights)
    if !isvalid(w)
        println(io, "Exposure specific weights not estimated in this type of model")
    else
        println(io, "Negative weights")
        println(io, size(w.neg, 1) > 0 ? w.neg : "(none)")
        println(io, "Positive weights")
        println(io, size(w.pos, 1) > 0 ? w.pos : "(none)\n")
    end
end

function Base.show(w::Q) where {Q<:QGcomp_weights}
    show(stdout, w)
end



function Base.values(x::Vector{I}) where {I<:AbstractID}
    [xi.value for xi in x]
end

function Base.show(io::IO, x::I) where {I<:AbstractID}
    show(io, x.value)
end

function Base.show(x::I) where {I<:AbstractID}
    show(stdout, x)
end

function Base.isless(x::I, y::I) where {I<:AbstractID}
    Base.isless(x.value, y.value)
end

function Base.length(x::I) where {I<:AbstractID}
    Base.length(x.value)
end


#=
"""
```julia
using Qgcomp
x = rand(100)
q=4
breaks = Qgcomp.getbreaks(x,q)
```
"""
=#
function getbreaks(x, q)
    qs = [i / (q) for i = 0:q]
    qq = quantile(x, qs)
    qq[[1, end]] = [-floatmax(), floatmax()]
    qq
end

#=
"""
```julia
using Qgcomp
x = rand(100)
q=4
breaks = Qgcomp.getbreaks(x,q)
Qgcomp.breakscore(x, breaks)
```
"""
=#
function breakscore(x, breaks)
    xq = [findlast(breaks .< xi) - 1 for xi in x]
    xq
end

#=
"""
```julia
x = rand(100, 3)
q=4
nexp = size(x,2)
```
"""
=#
function quantize(x, q)
    breaks = getbreaks(x, q)
    xq = breakscore(x, breaks)
    xq, breaks  # can turn into struct
end

function get_xq(x, q)
    nexp = size(x, 2)
    xqset = [quantize(x[:, j], q) for j = 1:nexp]
    breaks = reduce(hcat, [z[2] for z in xqset])
    xq = reduce(hcat, [z[1] for z in xqset])
    xq, breaks
end

#=
"""
```julia
expnms = [:x1, :x2, :x3]
expnms = ["x1", "x2", "x3"]
df = DataFrame(rand(100,3), expnms)
get_dfq(df, expnms, 4)
get_dfq(df, expnms, nothing)
```
"""
=#
function get_dfq(data, expnms, q)
    qdata = deepcopy(data)
    if isnothing(q)
        intvals = quantile(Matrix(qdata[:, expnms]), [q for q = 0.1:0.1:0.9])
        return ((qdata, q, intvals))
    end
    x = data[:, expnms]
    xq, breaks = get_xq(x, q)
    intvals = sort(unique(quantize(rand(1000), q)[1]))
    qdata[:, expnms] = xq
    qdata, breaks, intvals
end


function psicomb(coefs, coefnames_, expnms)
    weightvec = zeros(length(coefnames_))
    expidx = findall([vci ∈ expnms for vci in coefnames_])
    weightvec[expidx] .= 1.0
    nonpsi = deepcopy(coefs[findall(weightvec .== 0)])
    # may need to have a better way to find intercept
    psi = "(Intercept)" ∈ coefnames_ ? popfirst!(nonpsi) : float.([])
    psi = vcat(psi, weightvec' * coefs, nonpsi)
    psi
end

#=

using Qgcomp, Random, DataFrames, StatsModels
using GLM, LSurvival

    x1 = rand(100, 3)
    x = rand(100, 3)
    z = rand(100, 3)

    xq, _ = Qgcomp.get_xq(x, 4)
    y = randn(100) + xq * [.1, 0.05, 0]
    lindata = DataFrame(hcat(y,x,z), [:y, :x1, :x2, :x3, :z1, :z2, :z3])

    expnms = ["x1", "x2", "x3"]

    form = @formula(y~x1+x2+x3+z1+z2+z3)
    m = qgcomp_glm_noboot(form, lindata, expnms, 4, Normal())

    coefs = coef(m.ulfit)
    coefnames_ = coefnames(m.ulfit)
    expnms = String.(expnms)
    negw, posw = getweights(coefs, coefnames_, expnms)
=#
function getweights(coefs, coefnames_, expnms)
    expidx = reduce(vcat, [findall(coefnames_ .== expnm) for expnm in expnms])
    posidx = intersect(findall(coefs .>= 0.0), expidx)
    negidx = intersect(findall(coefs .< 0.0), expidx)
    partials = map(i -> sum(coefs[i]), (negidx, posidx))
    partialnames = map(i -> coefnames_[i], (negidx, posidx))
    outcoefs = map(i -> coefs[i], (negidx, posidx))
    weights = map(i -> outcoefs[i] ./ partials[i], (1, 2))
    map(i -> begin
        df = DataFrame(exposure = partialnames[i], coef = outcoefs[i], weight = weights[i])
        df.ψ_partial .= partials[i]
        df[:, ["exposure", "coef", "ψ_partial", "weight"]]
    end, 1:2)
end


function vccomb(vc, vcnames, expnms)
    weightvec = zeros(length(vcnames))
    expidx = findall([vci ∈ expnms for vci in vcnames])
    weightvec[expidx] .= 1.0
    nonexpcols = findall(weightvec .== 0.0)
    nonexpstart = findfirst(weightvec .== 0.0)
    expstart = findfirst(weightvec .> 0.0)
    vcov_psi = zeros(length(nonexpcols) + 1, length(nonexpcols) + 1)
    # diagonal term: mixture X mixture
    vcov_psi[expstart, expstart] = weightvec' * vc * weightvec # psi
    if length(nonexpcols)>0
        expdestination = setdiff(1:(length(nonexpcols) .+ 1), expstart)
        # diagonal terms: covariate X covariate
        vcov_psi[expdestination, expdestination] = vc[nonexpcols, nonexpcols]
        # off-diagonal terms: mixture X covariate
        wv = zeros(length(vcnames)) # temporary weight vector for each non-exposure column
        for (cov_index, cov_col) in enumerate(nonexpcols)
            j = expdestination[cov_index]
            wv .*= 0.0
            wv[cov_col] = 1.0
            vcov_psi[expstart, j] = vcov_psi[j, expstart] = weightvec' * vc * wv
        end
    end
    vcov_psi
end



function isvalid(w::QGcomp_weights)
    w.isvalid
end
