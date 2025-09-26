#Types.jl


abstract type AbstractID end
abstract type AbstractQgcomp_MSM end
abstract type QGcomp_model <: RegressionModel end
abstract type QGcomp_abstractweights end

struct ID <: AbstractID
    value::T where {T<:Union{Number,String}}
end


mutable struct QGcomp_weights{N<:DataFrame,P<:DataFrame,V<:Bool} <: QGcomp_abstractweights
    neg::N
    pos::P
    isvalid::V
end


mutable struct Qgcomp_EEmsm <: RegressionModel
    coef_vcov::Tuple{Vector,Matrix}
    coef::Vector
    vcov::Matrix
    MSMdf::Union{D,M} where{D<:DataFrame, M<:Matrix}
end

mutable struct Qgcomp_MSM{
    F<:RegressionModel,
    D<:Array{Float64,2},
    V<:Array{Float64,2},
    D2<:Array{Float64,2},
    V2<:Array{Float64,2},
    D3<:Array{Float64,2},
    V3<:Array{Float64,2},
    Di<:Distribution,
    L<:Link,
    F2<:FormulaTerm,
} <: AbstractQgcomp_MSM
    msmfit::F
    parms::Vector
    meanypred::Vector
    ypred::Vector
    intvector::Vector{Float64}
    bootparm_draws::D
    bootparm_vcov::V
    meanypred_draws::D2
    meanypred_vcov::V2
    #
    meanypredmsm::Vector
    ypredmsm::Vector
    meanypredmsm_draws::D3
    meanypredmsm_vcov::V3
    msmfamily::Di
    msmlink::L
    contrasts::Dict{Symbol,<:Any}
    msmformula::F2
end



mutable struct QGcomp_glm{
    F<:FormulaTerm,
    E<:Vector{<: String},
    Q<:Union{Int,Nothing},
    I<:Union{Vector{<: AbstractID},Nothing},
    W<:Union{QGcomp_weights,Nothing},
    D<:Distribution,
    L<:Link,
} <: QGcomp_model
    formula::F
    data::Any
    expnms::E
    q::Q
    family::D
    link::L
    breaks::Any
    intvals::Any
    ulfit::Any
    fit::Any
    fitted::Bool
    id::I
    msm::Any
    qgcweights::W
    contrasts::Dict{Symbol,<:Any}
end


mutable struct QGcomp_cox{
    F<:FormulaTerm,
    E<:Vector{<: String},
    Q<:Union{Int,Nothing},
    I<:Union{ID,Nothing},
    W<:Union{QGcomp_weights,Nothing},
    D<:Distribution,
    L<:Link,
} <: QGcomp_model
    formula::F
    data::Any
    expnms::E
    q::Q
    family::D
    link::L
    breaks::Any
    intvals::Any
    ulfit::Any
    fit::Any
    fitted::Bool
    id::Vector{I}
    msm::Any
    qgcweights::W
    contrasts::Dict{Symbol,<:Any}
end

mutable struct QGcomp_ee{
    F<:FormulaTerm,
    E<:Vector{<: String},
    Q<:Union{Int,Nothing},
    I<:Union{ID,Nothing},
    W<:Union{QGcomp_weights,Nothing},
    D<:Distribution,
    L<:Link,
} <: QGcomp_model
    formula::F
    data::Any
    expnms::E
    q::Q
    family::D
    link::L
    breaks::Any
    intvals::Any
    fullfit::Any
    ulfit::Any
    fit::Any
    fitted::Bool
    id::I
    msm::Any
    qgcweights::W
    contrasts::Dict{Symbol,<:Any}
end


