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


mutable struct Qgcomp_MSM{
    F<:RegressionModel,
    D<:Array{Float64,2},
    V<:Array{Float64,2},
    D2<:Array{Float64,2},
    V2<:Array{Float64,2},
} <: AbstractQgcomp_MSM
    msmfit::F
    ypred::Vector
    intval::Vector{Float64}
    bootparm_draws::D
    bootparm_vcov::V
    meanypred_draws::D2
    meanypred_vcov::V2
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
    ulfit::Any
    fit::Any
    fitted::Bool
    id::I
    msm::Any
    qgcweights::W
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
    ulfit::Any
    fit::Any
    fitted::Bool
    id::Vector{I}
    msm::Any
    qgcweights::W
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
end


