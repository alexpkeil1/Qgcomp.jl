#Types.jl


abstract type QGcomp_model <: RegressionModel end

mutable struct QGcomp_glm{
    F <: FormulaTerm,
    E <: Vector{<: String},
    Q <: Union{Int, Nothing}
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
end


mutable struct QGcomp_cox{
    F <: FormulaTerm,
    E <: Vector{<: String},
    Q <: Union{Int, Nothing}
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
end

mutable struct QGcomp_ee{
    F <: FormulaTerm,
    E <: Vector{<: String},
    Q <: Union{Int, Nothing}
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
end
