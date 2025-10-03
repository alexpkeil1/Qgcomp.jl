# splines.jl
# regression splines using restricted spline functions and b-splines
# adapted from https://juliastats.org/StatsModels.jl/stable/internals/#extending

# type of model where syntax applies: here this applies to any model type
const SPLINE_CONTEXT = Any
const BSPLINE_DEGREES = 2

function Base.merge(d::D, nt::NT) where{D<:DataFrame, NT<:NamedTuple}
  merge(Tables.columntable(d), nt)
end

# struct for behavior
struct RCSTerm{T,D} <: AbstractTerm
    term::T
    knots::D
end

struct RQSTerm{T,D} <: AbstractTerm
    term::T
    knots::D
end

struct BSTerm{T,D} <: AbstractTerm
    term::T
    knots::D
end

################################################
#=  
Restricted cubic/quadratic splines and B-splines:
   model building functions
=#
################################################

Base.show(io::IO, p::T) where{T<:Union{RCSTerm}} = print(io, "rcs($(p.term), $(p.knots))")
Base.show(io::IO, p::T) where{T<:Union{RQSTerm}} = print(io, "rqs($(p.term), $(p.knots))")
Base.show(io::IO, p::T) where{T<:Union{BSTerm}} = print(io, "bs($(p.term), $(p.knots))")
#Base.show(io::IO, p::RQSTerm) = print(io, "rcs($(p.term), $(p.knots))")

# for `rcs` use at run-time (outside @formula), return a schema-less RCSTerm
rcs(t::Vector, k::Vector) = hcat(t, rsplineBasis(t, k, 3))
rcs(t::Symbol, k::Vector) = RCSTerm(term(t), term.(k))
rcs(t::Symbol, k::Symbol) = RCSTerm(term(t), term.(k))

rqs(t::Vector, k::Vector) = hcat(t, rsplineBasis(t, k, 2))
rqs(t::Symbol, k::Vector) = RQSTerm(term(t), term.(k))
rqs(t::Symbol, k::Symbol) = RQSTerm(term(t), term.(k))

bs(t::Vector, k::Vector) = bsplineBasis(t, k, BSPLINE_DEGREES)[:,2:end]
bs(t::Symbol, k::Vector) = BSTerm(term(t), term.(k))
bs(t::Symbol, k::Symbol) = BSTerm(term(t), term.(k))



#to be able to construct a model matrix from them we need to have a way to apply a schema
# for `rcs` use inside @formula: create a schemaless RCSTerm and then apply_schema
# when rcs is used inside a formula, it creates a function term
function StatsModels.apply_schema(t::FunctionTerm{typeof(rcs)}, sch::StatsModels.Schema, Mod::Type{<:SPLINE_CONTEXT})
	# schema that applies to RCSTerms with term, knots as arguments
    apply_schema(RCSTerm(t.args...), sch, Mod)
end

function StatsModels.apply_schema(t::FunctionTerm{typeof(rqs)}, sch::StatsModels.Schema, Mod::Type{<:SPLINE_CONTEXT})
    apply_schema(RQSTerm(t.args...), sch, Mod)
end

function StatsModels.apply_schema(t::FunctionTerm{typeof(bs)}, sch::StatsModels.Schema, Mod::Type{<:SPLINE_CONTEXT})
    apply_schema(BSTerm(t.args...), sch, Mod)
end


function extract_knots(knot_terms::Vector{ConstantTerm{F}}) where{F<:Number}
    [kn.n for kn in knot_terms]
end

function extract_knots(knot_terms::Vector{F}) where{F<:Number}
    knot_terms
end


# apply_schema to internal Terms and check for proper types
function StatsModels.apply_schema(t::RCSTerm, sch::StatsModels.Schema, Mod::Type{<:SPLINE_CONTEXT}) 
    term = apply_schema(t.term, sch, Mod)
    isa(term, ContinuousTerm) || throw(ArgumentError("$(T) only works with continuous terms (got $term)"))
    knot_terms = apply_schema(t.knots, sch, Mod)
    isa(knot_terms, Vector) ||
        throw(ArgumentError("RCSTerm knots must be a vector (got $knot_terms)"))
    RCSTerm(term, extract_knots(knot_terms))
end

function StatsModels.apply_schema(t::RQSTerm, sch::StatsModels.Schema, Mod::Type{<:SPLINE_CONTEXT})
    term = apply_schema(t.term, sch, Mod)
    isa(term, ContinuousTerm) || throw(ArgumentError("$(T) only works with continuous terms (got $term)"))
    knot_terms = apply_schema(t.knots, sch, Mod)
    isa(knot_terms, Vector) ||
        throw(ArgumentError("RQSTerm knots must be a vector (got $knot_terms)"))
    RQSTerm(term, extract_knots(knot_terms))
end

function StatsModels.apply_schema(t::BSTerm, sch::StatsModels.Schema, Mod::Type{<:SPLINE_CONTEXT}) 
    term = apply_schema(t.term, sch, Mod)
    isa(term, ContinuousTerm) || throw(ArgumentError("$(T) only works with continuous terms (got $term)"))
    knot_terms = apply_schema(t.knots, sch, Mod)
    isa(knot_terms, Vector) ||
        throw(ArgumentError("BSTerm knots must be a vector (got $knot_terms)"))
    BSTerm(term, extract_knots(knot_terms))
end

function StatsModels.modelcols(v::Vector, d::NamedTuple)
    v
end

function StatsModels.modelcols(p::RCSTerm, d::NamedTuple)
    col = modelcols(p.term, d)
    knots = modelcols(p.knots, d)
    hcat(col, rsplineBasis(col, knots, 3))
end

function StatsModels.modelcols(p::RQSTerm, d::NamedTuple)
    col = modelcols(p.term, d)
    knots = modelcols(p.knots, d)
    hcat(col, rsplineBasis(col, knots, 2))
end

function StatsModels.modelcols(p::BSTerm, d::NamedTuple)
    col = modelcols(p.term, d)
    knots = modelcols(p.knots, d)
    bsplineBasis(col, knots, BSPLINE_DEGREES)[:,2:end] # sum of b-splines = 0
end


# the basic terms contained within a RCSTerm (for schema extraction)
StatsModels.terms(p::T) where {T<:Union{RQSTerm, RCSTerm, BSTerm}} = terms(p.term)
# names variables from the data that a RCSTerm relies on
StatsModels.termvars(p::T) where {T<:Union{RQSTerm, RCSTerm, BSTerm}} = StatsModels.termvars(p.term)
# number of columns in the matrix this term produces


StatsModels.length(p::RCSTerm) = length(p.knots) - 1
StatsModels.width(p::RCSTerm) = length(p.knots) - 1

StatsModels.length(p::RQSTerm) = length(p.knots) - 0
StatsModels.width(p::RQSTerm) = length(p.knots) - 0

StatsModels.length(p::BSTerm) = length(p.knots) - BSPLINE_DEGREES
StatsModels.width(p::BSTerm) = length(p.knots) - BSPLINE_DEGREES


StatsAPI.coefnames(p::T) where {T<:Union{RCSTerm, RQSTerm}} = vcat(coefnames(p.term), [coefnames(p.term) * "_sp$basisnum"  for basisnum in 1:(width(p)-1)])
StatsAPI.coefnames(p::T) where {T<:Union{BSTerm}} = vcat([coefnames(p.term) * "_sp$basisnum"  for basisnum in 1:(width(p))])




################################################
#=  
Restricted cubic/quadratic splines, underlying functions
=#
################################################

#=
 Restricted quadratic spline (based on rcspline.eval from R Hmisc package, normalized version)
 usage: rcsBasis(x::Number,knts,numKnots,numVar)

=#
function rcsBasis(x::Number, knts, numKnots, numVar)
    # from frank harrel macro
    [
        ((x > knts[i]) * (x - knts[i]) / ((knts[numKnots] - knts[1])^(2 / 3)))^3 +
        (
            (knts[numKnots-1] - knts[i]) *
            ((x > knts[numKnots]) * (x - knts[numKnots]) / ((knts[numKnots] - knts[1])^(2 / 3)))^3 -
            (knts[numKnots] - knts[i]) *
            ((x > knts[numKnots-1]) * (x - knts[numKnots-1]) / ((knts[numKnots] - knts[1])^(2 / 3)))^3
        ) / (knts[numKnots] - knts[numKnots-1]) for i = 1:numVar
    ]
end

function rqsBasis(x::Number, knts, numKnots, numVar)
    [
        ((x > knts[i]) * ((x - knts[i]) / knts[1])^2) - ((x > knts[numKnots]) * ((x - knts[numKnots]))^2) for
        i = 1:numVar
    ]
end

#=
Create matrix representing restricted spline bases, given a set of spline knots
  usage: rsplineBasis(x,knts,degree)
  x = vector of numbers
  knts = vector of knots for restricted spline
  degree = degres of spline (2 or 3 only)
=#
function rsplineBasis(x, knts, degree)
    numKnots = length(knts)
    numVar = numKnots - degree + 1
    if numVar < 1
        throw("number of knots must be at least 3")
    end
    if degree == 3
        mat = vcat([rcsBasis(xi, knts, numKnots, numVar) for xi in x]'...)
    elseif degree == 2
        mat = vcat([rqsBasis(xi, knts, numKnots, numVar) for xi in x]'...)
    end
    mat
end


"""
 Create a vector of restricted spline knots (based on rcspline.eval from R Hmisc package, normalized version)
 usage: rsplineknots(x,nk)
 x = vector of numbers
 nk = number of (interior) knots
 output: a nk-length vector of (interior) knots



 ```julia
 using Qgcomp, DataFrames, Distributions, StatsModels
 dat = DataFrame(y=Int.(rand(Bernoulli(0.25), 50)), x1=rand(50), x2=rand(50), z=rand(50))
 
 knts = rsplineknots(dat.x1, 5)

 kform = term(:y)~ term(1) + rcs(:x1,knts) + term(:x2)
 kform2 = term(:y)~ term(1) + rqs(:x1,knts) + term(:x2)

 ft = qgcomp_glm_ee(kform, dat,["x1", "x2"], nothing, Binomial())
 ft = qgcomp_glm_ee(kform2, dat,["x1", "x2"], nothing, Binomial())
 

 ```
"""
function rsplineknots(x, nk)
    #println("Creating new knots")
    #knot placement taken from sas %DASPLINE macro
    # similar to rcs in R: set "outer" first and then get even sequence of percentiles between the outer knots
    if nk == 3
        p = [5, 50, 95]
    elseif nk == 4
        p = [5, 35, 65, 95]
    elseif nk == 5
        p = [5, 27.5, 50, 72.5, 95]
    elseif nk == 6
        p = [5, 23, 41, 59, 77, 95]
    elseif nk == 7
        p = [2.5, 18.3333, 34.1667, 50, 65.8333, 81.6667, 97.5]
    elseif nk == 8
        p = [1, 15, 29, 43, 57, 71, 85, 99]
    elseif nk == 9
        p = [1, 13.25, 25.5, 37.75, 50, 62.25, 74.5, 86.75, 99]
    elseif nk == 10
        p = [1, 11.88889, 22.77778, 33.66667, 44.55556, 55.44444, 66.33333, 77.22222, 88.11111, 99]
    elseif nk == 11
        p = [1.0, 10.8, 20.6, 30.4, 40.2, 50.0, 59.8, 69.6, 79.4, 89.2, 99.0]
    elseif nk > 11
        halfpctl = [floor(100.0 * 1.9^(j) / 1.9^(nk)) for j = ceil(nk/2):nk]
        p =
            iseven(nk) ? [halfpctl[1:(end-1)]; 100.0 .- halfpctl[(end-1):-1:1]] :
            [halfpctl[1:(end-1)]; 50; 100.0 .- halfpctl[(end-1):-1:1]]
    end
    knts = percentile(x, p)
    knts
end
rsplineknots(x, nk, degree) = rsplineknots(x, nk)


#=
Create matrix representing restricted spline bases, given a set of spline knots
  usage: rsplineBasis(x,knts,degree)
  x = vector of numbers
  nk = number of knots, or nothing if supplying knots
  degree = degres of spline (2 or 3 only)
  knts = [optional] vector of knots

  output: a matrix of spline bases with (nk - degree+1) columns and length(x) rows
=#
function rspline(x, nk; degree = 2, knts = nothing)
    #restricted cubic or quadratic spline 
    if !isnothing(nk)
        knts = rsplineknots(x, nk)
    end
    mat = rsplineBasis(x, knts, degree)
end


################################################
#=  

B-splines

=#
################################################
function bsplineIntkntsUniform(xmin, xmax, nk)
    # following SAS defaults (BSPLINE from IML)
    intknts = [xmin + (xmax - xmin) * j / (nk + 1.0) for j = 1:nk]
    intknts
end

function bsplineIntkntsPercentileU(x, nk)
    # setting internal knots based on evenly based percentiles
    halfpctl = [100 * (j - nk / 2) / (nk + 1 / 2) for j = ceil(nk/2):nk]
    p =
        iseven(nk) ? [halfpctl[1:(end-1)]; 100.0 .- halfpctl[(end-1):-1:1]] :
        [halfpctl[1:(end-1)]; 50; 100.0 .- halfpctl[(end-1):-1:1]]
    intknts = StatsBase.percentile(x, p)
    intknts
end

function bsplineIntkntsPercentile(x, nk)
    # setting internal knots based on tail focused percentiles
    halfpctl = [floor(100.0 * 1.9^(j) / 1.9^(nk)) for j = ceil(nk/2):nk]
    p =
        iseven(nk) ? [halfpctl[1:(end-1)]; 100.0 .- halfpctl[(end-1):-1:1]] :
        [halfpctl[1:(end-1)]; 50; 100.0 .- halfpctl[(end-1):-1:1]]
    intknts = StatsBase.percentile(x, p)
    intknts
end


function bsplineIntknts(xmin, xmax, nk)
    # following SAS defaults (BSPLINE from IML)
    intknts = [xmin + (xmax - xmin) * j / (nk + 1.0) for j = 1:nk]
    #vcat(xmin, intknts)
    intknts
end

#=
  bsplineExtkntsEven(0,5, 8)
=#
function bsplineExtkntsEven(xmin, xmax, degree; gap = 1.0)
    # following SAS defaults (BSPLINE from IML)
    epsilon = 1e-12
    lowknts = degree == 1 ? [xmin - epsilon] : collect(range(xmin - epsilon - degree - gap, xmin - epsilon, degree)) # returns Int64[] if degree = 0
    highknts = degree <= 1 ? [xmax + epsilon] : collect(range(xmax + epsilon, xmax + epsilon + degree + gap, degree))
    lowknts, highknts
end

#=
  bsplineExtkntsEvenExpand(0,5, 8)
=#
function bsplineExtkntsEvenExpand(xmin, xmax, degree; gap = 1.0, mult = 2.0)
    # moves first external knots slightly outward to increase the support of the spline for prediction
    epsilon = 1e-12
    lowknts =
        degree == 1 ? [xmin - gap * mult - epsilon] :
        collect(range(xmin - epsilon - degree - gap, xmin - epsilon, degree)) # returns Int64[] if degree = 0
    highknts =
        degree <= 1 ? [xmax + gap * mult + epsilon] :
        collect(range(xmax + epsilon, xmax + epsilon + degree + gap, degree))
    lowknts, highknts
end


"""
  bsplineknots(0,5, 8)
  extdist = determines the distance between external knots. Set to 0 for classic "repeated" tail knots.
            if positive, a constant gap between exterior knots starting at max value + epsilonilon
            if negative, a constant multiplier of the final gap size between interior knots

```julia
using Qgcomp, DataFrames, Distributions, StatsModels

n = 200
dat = DataFrame(y=Int.(rand(Bernoulli(0.25), n)), x1=rand(n), x2=rand(n), z=rand(n))

# 3 internal knots
deg = 2
knts = bsplineknots(dat.x1, 5, deg; ptype = "uniform", extdist = 1)
nkn = length(knts)
nkn - deg 
size(bs(dat.x1,knts), 2)


kform = term(:y) ~ term(1) + bs(:x1,knts) + term(:x2)
ft = qgcomp_glm_ee(kform, dat,[:x1, :x2], nothing, Normal())
ft = qgcomp_glm_boot(kform, dat,[:x1, :x2], nothing, Normal())


using GLM
glm(kform, dat, Binomial())



```                
"""
function bsplineknots(x, nk, degree; ptype = "uniform", extdist = 1)
    # following SAS calculations
    xmax, xmin = maximum(x), minimum(x)
    ptype = lowercase(ptype[1:2])
    if (ptype == "un") || (ptype == "ev")
        intknts = bsplineIntkntsUniform(xmin, xmax, nk)
    elseif ptype == "pe"
        intknts = bsplineIntkntsPercentile(x, nk)
    end
    if extdist < 0
        gap = length(intknts) == 1 ? xmax - intknts[1] : diff(intknts)[end]
        gap *= -extdist
    else
        gap = extdist
    end
    lowknts, highknts = bsplineExtkntsEven(xmin, xmax, degree, gap = gap)
    knts = vcat(lowknts, intknts, highknts)
    knts
end


"""
  bsplineZeroBasis([0.0,0.3,3.0],0.1, 2.0)
"""
function bsplineZeroBasis(x::Number, knts::Vector{<:Number})
    # following SAS calculations
    # dimensions: length(x) X (nk + degree + 1)
    #ki = findfirst(x .< knts)
    #B = zeros(Float64, length(knts))
    #B[ki] = 1.0
    B = [Float64((x >= knts[j]) & (x < knts[j+1])) for j = 1:length(knts[1:(end-1)])]
    B
end

function bsplineZeroBasis(x::Vector{<:Number}, knts::Vector{<:Number})
    B = vcat([bsplineZeroBasis(xi, knts) for xi in x]'...)
end



"""
  bsplineBasis([0.0,0.3,3.0],0.1, 2.0)
"""
#function bsplineBasis(x::Number,knts,degree,ki,iki)
function bsplineBasis(x::Number, knts, degree)
    # following SAS calculations
    ki = findfirst(knts .> x)  # index of first knot that exceeds x
    kki = ki - degree - 1      # index of first internal knot (or max knot) that exceeds x
    nkn = length(knts) # nkn = nk + 2*degree 
    nb = nkn - degree + 1 #  nkn - degree = degree + nk
    B = zeros(Float64, nb)
    B[1+kki] = 1.0
    if degree == 0
        return B[1:(nb-1)] # nb is wrong for zero degree b-spline
    end
    #oldB = Array{Float64, 1}(UndefInitializer(), length(B))
    W = Array{Float64,1}(UndefInitializer(), 2 * degree)
    # modification of SAS macro
    for j = 1:degree
        #oldB .= B
        W[degree+j] = knts[ki+j-1] - x
        W[j] = x - knts[ki-j]
        s = 0
        for i = 1:j # should grow by one at each step
            #t = W[degree + i] + W[j + 1 - i]
            t = (W[degree+i] == -W[j+1-i]) ? 0.0 : B[i+kki] / (W[degree+i] + W[j+1-i])
            B[i+kki] = s + W[degree+i] * t
            s = W[j+1-i] * t
        end
        B[j+1+kki] = s
    end
    B
end

function bsplineBasis(x::Vector{<:Number}, knts, degree)
    vcat([bsplineBasis(xi, knts, degree) for xi in x]'...)
end

"""
  bsplineIntknts(0,5, 8)
"""
function bspline(x, nk; degree = 1, knts = nothing)
    #https://go.documentation.sas.com/doc/en/pgmsascdc/v_013/imlug/imlug_langref_sect072.htm
    if (isnothing(knts))
        knts = bsplineknots(x, nk, degree)
        #println("knts: $knts")
    end
    bsplineBasis(x, knts, degree)
end
