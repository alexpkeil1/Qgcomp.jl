# base.jl: core helper functions from quantile g-computation


function QGcomp_weights()
    QGcomp_weights(DataFrame(), DataFrame(), false)
end


function Base.show(io::IO, w::QGcomp_weights)
    if !isvalid(w)
        println(io, "Exposure specific weights not estimated in this type of model")
    else
        println(io, "Scaled effect size (negative direction)")
        println(io, size(w.neg, 1) > 0 ? w.neg : "(none)")
        println(io, "Scaled effect size (positive direction)")
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

function quantize(x::Vector; breaks = sort(vcat([-floatmax(), floatmax()]..., quantile(x, [.25, .5, .75]))))
    xq = breakscore(x, breaks)
    xq, breaks  # can turn into struct
end



"""
```julia
x = rand(100, 3)
q=4
nexp = size(x,2)

xq, breaks = quantize(x, q)

```
"""
function quantize(x::Vector, q)
    breaks = getbreaks(x, q)
    xq = breakscore(x, breaks)
    xq, breaks  # can turn into struct
end

function get_xq(x::D; breaks= reduce(hcat, [sort(vcat([-floatmax(), floatmax()]..., quantile(xc, [.25, .5, .75]))) for xc in eachcol(x)])) where (D<:Union{DataFrame,Matrix})
    nexp = size(x, 2)
    xqset = [quantize(x[:, j]; breaks=breaks[:,j]) for j = 1:nexp]
    xq = reduce(hcat, [z[1] for z in xqset])
    xq, breaks
end

function get_xq(x::T; breaks= reduce(hcat, [sort(vcat([-floatmax(), floatmax()]..., quantile(xc, [.25, .5, .75]))) for xc in eachcol(x)])) where {T<:NamedTuple}
    nexp = length(x)
    xqset = [quantize(getindex(x, j); breaks=breaks[:,j]) for j = 1:nexp]
    qkeys = keys(x)
    xq = Tuple([z[1] for z in xqset])
    (; (qkeys .=> xq)...), breaks
end

function get_xq(x::D, q) where (D<:Union{DataFrame,Matrix})
    nexp = size(x, 2)
    xqset = [quantize(x[:, j], q) for j = 1:nexp]
    breaks = reduce(hcat, [z[2] for z in xqset])
    xq = reduce(hcat, [z[1] for z in xqset])
    xq, breaks
end

function get_xq(x::T, q) where {T<:NamedTuple}
    nexp = length(x)
    xqset = [quantize(getindex(x, j), q) for j = 1:nexp] 
    breaks = reduce(hcat, [z[2] for z in xqset])
    qkeys = keys(x)
    xq = Tuple([z[1] for z in xqset])
    (; (qkeys .=> xq)...), breaks
end

#=
"""
```julia
using DataFrames
expnms = [:x1, :x2, :x3]
expnms = ["x1", "x2", "x3"]
df = DataFrame(rand(100,3), expnms)

    x = df[:, expnms]
    xq, breaks = get_xq(x, 4)


get_dfq = Qgcomp.get_dfq
get_xq = Qgcomp.get_xq
get_dfq(df, expnms, 4)
get_dfq(df, expnms, nothing)
```
"""
=#
function get_dfq(data::D, expnms, q) where {D<:DataFrame}
    qdata = deepcopy(data)
    if isnothing(q)
        @warn "q is nothing: defaulting to deciles of combined exposure data (unless `intvals`` specified directly)"
        intvals = quantile(Matrix(qdata[:, expnms]), [q for q = 0.1:0.1:0.9])
        return ((qdata, nothing, intvals))
    end
    x = data[:, expnms]
    #x = getindex(data, expnms)
    xq, breaks = get_xq(x, q)
    intvals = sort(unique(xq))
    qdata[:, expnms] = xq
    qdata, breaks, float.(intvals)
end

@generated function setindex(x::NamedTuple,y,v::Val)
    k = first(v.parameters)
    k ∉ x.names ? :x : :( (x..., $k=y) )
  end

function get_dfq(data::T, expnms, q) where {T<:NamedTuple}
    qdata = deepcopy(data)
    if isnothing(q)
        @warn "q is nothing: defaulting to deciles of combined exposure data (unless `intvals`` specified directly)"
        intvals = quantile(Matrix(qdata[:, expnms]), [q for q = 0.1:0.1:0.9])
        return ((qdata, nothing, intvals))
    end
    x = getindex(data, Symbol.(expnms))
    xq, breaks = get_xq(x, q)
    intvals = sort( unique(reduce(vcat, values(xq))))
    for xx in Symbol.(expnms)
        qdata[xx] .= xq[xx]
    end
    qdata, breaks, float.(intvals)
end



function get_dfq(data::D, expnms; breaks=reduce(hcat, [sort(vcat([-floatmax(), floatmax()]..., quantile(xc, [.25, .5, .75]))) for xc in eachcol(x)])) where {D<:DataFrame}
    qdata = deepcopy(data)
    x = data[:, expnms]
    xq, breaks = get_xq(x; breaks=breaks)
    intvals = sort(unique(xq))
    qdata[:, expnms] = xq
    qdata, breaks, float.(intvals)
end

function get_dfq(data::T, expnms; breaks=reduce(hcat, [sort(vcat([-floatmax(), floatmax()]..., quantile(xc, [.25, .5, .75]))) for xc in eachcol(x)])) where {T<:NamedTuple}
    qdata = deepcopy(data)
    x = getindex(data, expnms)
    xq, breaks = get_xq(x; breaks=breaks)
    intvals = sort(unique(xq))
    qdata[:, expnms] = xq
    qdata, breaks, float.(intvals)
end


function psicomb(coefs, coefnames_::Vector{S1}, expnms::Vector{S2}) where {S1<:String, S2<:String}
    weightvec = zeros(length(coefnames_))
    expidx = findall([vci ∈ expnms for vci in coefnames_])
    weightvec[expidx] .= 1.0
    nonpsi = deepcopy(coefs[findall(weightvec .== 0)])
    # may need to have a better way to find intercept
    psi = "(Intercept)" ∈ coefnames_ ? popfirst!(nonpsi) : float.([])
    psi = vcat(psi, weightvec' * coefs, nonpsi)
    psi
end

function psicomb(coefs, coefnames_::Vector{S1}, expnms::Vector{S2}) where {S1<:Symbol, S2<:String}
    psicomb(coefs, String.(coefnames_), expnms) 
end
function psicomb(coefs, coefnames_::Vector{S1}, expnms::Vector{S2}) where {S1<:String, S2<:Symbol}
    psicomb(coefs, coefnames_, String.(expnms))
end
function psicomb(coefs, coefnames_::Vector{S1}, expnms::Vector{S2}) where {S1<:Symbol, S2<:Symbol}
    psicomb(coefs, String.(coefnames_), String.(expnms))
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
        sort!(df, :weight)
    end, 1:2)
end


function getweights(coefs, coefnames_::Vector{S1}, expnms::Vector{S2}) where {S1<:Symbol, S2<:String}
    getweights(coefs, String.(coefnames_), expnms) 
end
function getweights(coefs, coefnames_::Vector{S1}, expnms::Vector{S2}) where {S1<:String, S2<:Symbol}
    getweights(coefs, coefnames_, String.(expnms))
end
function getweights(coefs, coefnames_::Vector{S1}, expnms::Vector{S2}) where {S1<:Symbol, S2<:Symbol}
    getweights(coefs, String.(coefnames_), String.(expnms))
end


"""
Covariance between linear combinations of terms

e.g. for a linear regression 
β0 + β1*x + β2*z + β3*w + β4*r

with coefficient covariance matrix V, 

the variance for ψ = β1 + β2 (e.g. the simultaneous effect of increasing x and z by one unit)
    is calculated as vccomb(V, [0,1,1,0,0], [0,1,1,0,0])

The covariance term COV(ψ, β0) 
    is calculated as vccomb(V, [0,1,1,0,0], [1,0,0,0,0])

This function is useful for qgcomp methods when ψ is just a sum of β coefficients, because it allows
    straightforward calculation of the full covariance matrix. The underlying calculations are very simple,
    but the function provides 

"""
function vccomb(vc::Matrix{R0}, weightvec1::Vector{R1}, weightvec2::Vector{R2}) where {R0<:Real, R1<:Real, R2<:Real}
    weightvec1' * vc * weightvec2
end

function vccomb(vc::Matrix{R0}, weightvec1::Vector{R1}) where {R0<:Real, R1<:Real}
    weightvec1' * vc * weightvec1
end

secomb(vc,w1,w2) = sqrt.(vccomb(vc,w1,w2))
secomb(vc,w) = sqrt.(vccomb(vc,w))

"""
One version of the function (string vector arguments) is very specifically tuned toward a covariance matrix for qgcomp_glm_boot, 
    given the covariance matrix column labels and expnms. It assumes a univariate ψ parameter.
"""
function vccomb(vc::Matrix{R0}, vcnames::Vector{S1}, expnms::Vector{S2}) where {R0<:Real, S1<:String, S2<:String}
    weightvec = zeros(length(vcnames))
    expidx = findall([vci ∈ expnms for vci in vcnames])
    weightvec[expidx] .= 1.0
    nonexpcols = findall(weightvec .== 0.0)
    #nonexpstart = findfirst(weightvec .== 0.0)
    expstart = findfirst(weightvec .> 0.0)
    vcov_psi = zeros(length(nonexpcols) + 1, length(nonexpcols) + 1)
    # diagonal term: mixture X mixture
    vcov_psi[expstart, expstart] = vccomb(vc, weightvec) # psi
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
            vcov_psi[expstart, j] = vcov_psi[j, expstart] = vccomb(vc, weightvec, wv)
        end
    end
    vcov_psi
end

function vccomb(vc::Matrix{R0}, vcnames::Vector{S1}, expnms::Vector{S2}) where {R0<:Real, S1<:Symbol, S2<:String}
    vccomb(vc, String.(vcnames), expnms)
end

function vccomb(vc::Matrix{R0}, vcnames::Vector{S1}, expnms::Vector{S2}) where {R0<:Real, S1<:String, S2<:Symbol}
    vccomb(vc, (vcnames), String.(expnms))
end

function vccomb(vc::Matrix{R0}, vcnames::Vector{S1}, expnms::Vector{S2}) where {R0<:Real, S1<:Symbol, S2<:Symbol}
    vccomb(vc, String.(vcnames), String.(expnms))
end

function isvalid(w::QGcomp_weights)
    w.isvalid
end
