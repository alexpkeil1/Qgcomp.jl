#Types.jl


abstract type AbstractID end
abstract type AbstractQgcMSM end
abstract type QGcomp_model <: RegressionModel end

struct ID <: AbstractID
    value::T where {T<:Union{Number,String}}
end

mutable struct QgcMSM{
    F <: RegressionModel,
    D <: Array{Float64, 2},
    V <: Array{Float64, 2},
    D2 <: Array{Float64, 2},
    V2 <: Array{Float64, 2}
    } <: AbstractQgcMSM 
    msmfit::F
    ypred::Vector{Float64}
    intval::Vector{Float64}
    bootparm_draws::D
    bootparm_vcov::V
    meanypred_draws::D2
    meanypred_vcov::V2
end



mutable struct QGcomp_glm{
    F <: FormulaTerm,
    E <: Vector{<: String},
    Q <: Union{Int, Nothing},
    I <: Union{Vector{ <: AbstractID}, Nothing}
} <: QGcomp_model
    formula::F
    data::Any
    expnms::E
    q::Q
    family::Any
    breaks::Any
    ulfit::Any
    fit::Any
    fitted::Bool
    id::I
    msm::Any
end


mutable struct QGcomp_cox{
    F <: FormulaTerm,
    E <: Vector{<: String},
    Q <: Union{Int, Nothing},    
    I <: Union{ID, Nothing}
} <: QGcomp_model
    formula::F
    data::Any
    expnms::E
    q::Q
    family::Any
    breaks::Any
    ulfit::Any
    fit::Any
    fitted::Bool
    id::I
    msm::Any
end

mutable struct QGcomp_ee{
    F <: FormulaTerm,
    E <: Vector{<: String},
    Q <: Union{Int, Nothing},    
    I <: Union{ID, Nothing}
} <: QGcomp_model
    formula::F
    data::Any
    expnms::E
    q::Q
    family::Any
    breaks::Any
    intvals::Any
    fullfit::Any
    ulfit::Any
    fit::Any
    fitted::Bool
    id::I
    msm::Any
end
