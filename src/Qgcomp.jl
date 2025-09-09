module Qgcomp
using StatsBase
using GLM
using DataFrames
using LinearAlgebra
using Distributions: Normal, cdf, quantile
using NLsolve
using ForwardDiff
using LSurvival
import Random: AbstractRNG, Xoshiro, MersenneTwister

# imports
import StatsBase:
    aic,
    aicc,
    bic,
    coef,
    coeftable,
    coefnames,
    confint,
    deviance,
    nulldeviance, #, dof_residual,
    dof,
    fitted,
    fit,
    fit!,
    isfitted,
    loglikelihood,
    #lrtest,
    modelmatrix,
    model_response,
    nullloglikelihood,
    nobs,
    PValue,
    stderror,
    residuals,
    predict,
    predict!,
    response,
    score,
    var,
    vcov,
    weights

#types
export QGcomp_model


# original functions
export qgcomp_glm_noboot, qgcomp_glm_boot, qgcomp_cox_noboot, qgcomp_glm_ee

#expanded functions from imports
export fit!, aic, aicc, bic, loglikelihood, fitted, isfitted

#re-exports
export Normal, Bernoulli, Poisson


# Abstract types
include("Types.jl")

# utility functions
include("base.jl")
include("sampling.jl")

# models
include("glm.jl")
include("cox.jl")
include("ee.jl")


end # module Qgcomp
