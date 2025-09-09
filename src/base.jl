# base.jl


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



"""
```julia
using Qgcomp
x = rand(100)
q=4
breaks = Qgcomp.getbreaks(x,q)
```
"""
function getbreaks(x, q)
    qs = [i / (q) for i = 0:q]
    qq = quantile(x, qs)
    qq[[1, end]] = [-floatmax(), floatmax()]
    qq
end

"""
```julia
using Qgcomp
x = rand(100)
q=4
breaks = Qgcomp.getbreaks(x,q)
Qgcomp.breakscore(x, breaks)
```
"""
function breakscore(x, breaks)
    xq = [findlast(breaks .< xi) - 1 for xi in x]
    xq
end

"""
```julia
x = rand(100, 3)
q=4
nexp = size(x,2)
```
"""
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

"""
```julia
expnms = [:x1, :x2, :x3]
expnms = ["x1", "x2", "x3"]
df = DataFrame(rand(100,3), expnms)
get_dfq(df, expnms, 4)
get_dfq(df, expnms, nothing)
```
"""
function get_dfq(data, expnms, q)
    qdata = deepcopy(data)
    if isnothing(q)
        intvals = quantile(Matrix(qdata[:,expnms]), [q for q in .1:.1:.9])
        return((qdata, q, intvals))
    end
    x = data[:, expnms]
    xq, breaks = get_xq(x, q)
    intvals = sort(unique(quantize(rand(1000), q)[1]))
    qdata[:, expnms] = xq
    qdata, breaks, intvals
end

function psicomb(coefs, coefnames, expnms)
    weightvec = zeros(length(coefnames))
    expidx = findall([vci ∈ expnms for vci in coefnames])
    weightvec[expidx] .= 1.0
    nonpsi = deepcopy(coefs[findall(weightvec .== 0)])
    psi = popfirst!(nonpsi)
    psi = vcat(psi, weightvec' * coefs, nonpsi)
    psi
end


function vccomb(vc, vcnames, expnms)
    weightvec = zeros(length(vcnames))
    expidx = findall([vci ∈ expnms for vci in vcnames])
    weightvec[expidx] .= 1.0
    bcols = findall(weightvec .== 0.0)
    psicols = collect(1:length(bcols)) .+ 1
    psicols[1] = 1
    vcov_psi = zeros(length(bcols) + 1, length(bcols) + 1)
    vcov_psi[psicols, psicols] = vc[bcols, bcols]
    vcov_psi[1, 1] = vc[1, 1]
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

function StatsBase.coeftable(m::M; level = 0.95) where M <: Union{QGcomp_glm,QGcomp_cox}
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
    rownms = isnothing(m.msm) ? coefnames(m.ulfit) : coefnames(m.msm.msmfit)
    if isnothing(m.msm)
        nonpsi = setdiff(rownms, m.expnms)
        rownms = popfirst!(nonpsi)
        rownms = vcat(rownms, "ψ")
        rownms = vcat(rownms, nonpsi)
    end
    CoefTable(coeftab, colnms, rownms, 4,3)
end