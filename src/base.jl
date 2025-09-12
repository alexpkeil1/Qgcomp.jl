# base.jl: core helper functions from quantile g-computation


function Base.values(x::Vector{I}) where {I<:AbstractID}
    [xi.value for xi in x]
end

function Base.show(io::IO, x::I) where {I<:AbstractID}
    show(io, x.value)
end
Base.show(x::I) where {I<:AbstractID} = Base.show(stdout, x::I)

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
    psi = "(Intercept)" ∈ coefnames_ ? popfirst!(nonpsi) : float.([])
    psi = vcat(psi, weightvec' * coefs, nonpsi)
    psi
end

#TODO: re-work for non-intercept models

function vccomb(vc, vcnames, expnms)
    weightvec = zeros(length(vcnames))
    expidx = findall([vci ∈ expnms for vci in vcnames])
    weightvec[expidx] .= 1.0
    bcols = findall(weightvec .== 0.0)
    psicols = collect(1:length(bcols)) .+ 1
    psicols[1] = 1
    vcov_psi = zeros(length(bcols) + 1, length(bcols) + 1)
    vcov_psi[psicols, psicols] = vc[bcols, bcols]
    vcov_psi[1, 1] = vc[1, 1] # intercept
    vcov_psi[2, 2] = weightvec' * vc * weightvec
    wv = zeros(length(vcnames))
    for (psicol, bcol) in enumerate(bcols)
        j = psicol > 1 ? psicol + 1 : 1
        wv .*= 0.0
        wv[bcol] = 1.0
        vcov_psi[2, j] = vcov_psi[j, 2] = weightvec' * vc * wv
        #sum(vc[findall(weightvec .> 0.0),bcol])
    end
    vcov_psi
end

