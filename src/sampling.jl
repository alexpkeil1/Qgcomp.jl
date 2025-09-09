#bootstrap.jl

"""
if n > length(x), then subsample (without replacement), and otherwise sample with replacement
"""
function mcsample(rng, x, n)
    if length(x) != n
        r = sample(rng, x, n, replace = (n > length(x)))
    else
        r = x
    end
    r
end



function bootsample(rng::T, V::Vector{I}) where {T<:AbstractRNG, I<:AbstractID}
    uid = unique(V)
    bootid = sort(rand(rng, uid))
    idxl = [findall(getfield.(V, :value) .== bootidi.value) for bootidi in bootid]
    idx = reduce(vcat, idxl)
    nid = ID.(reduce(vcat, [fill(i, length(idxl[i])) for i in eachindex(idxl)]))
    idx, nid
end
;

"""
```julia
using Qgcomp, Random
ids = [Qgcomp.ID(i) for i in 1:100]
Qgcomp.bootstrap(Xoshiro(), ids)
```
"""
function bootstrap(rng::T, V::Vector{I}) where {T<:AbstractRNG, I<:AbstractID}
    bootsample(rng, V)
end
;

