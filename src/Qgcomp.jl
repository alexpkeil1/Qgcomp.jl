module Qgcomp
using StatsBase
using GLM
using DataFrames
using LinearAlgebra
using Distributions: Normal, cdf, quantile, Distribution, ContinuousUnivariateDistribution
using NLsolve
using ForwardDiff
using LSurvival
import Random: AbstractRNG, Xoshiro, MersenneTwister

# imports
import StatsModels: hasintercept, drop_intercept

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
export QGcomp_model, QGcomp_weights, Qgcomp_MSM, QGcomp_glm, QGcomp_ee, QGcomp_cox

# original functions
export qgcomp_glm_noboot, qgcomp_glm_boot, qgcomp_cox_noboot, qgcomp_cox_boot, qgcomp_glm_ee

# original utility functions
export ID, genxq

#expanded functions from imports
export fit!, aic, aicc, bic, loglikelihood, fitted, isfitted

#re-exports
export Distribution, Normal, Bernoulli, Poisson, Binomial, LogitLink, IdentityLink, NegativeBinomialLink, ProbitLink, 
        @formula, FormulaTerm, Term, InteractionTerm, FunctionTerm


# Abstract types
include("Types.jl")

# utility functions
include("base.jl")
include("utility.jl")
include("sampling.jl")
include("simulation_helpers.jl")

# re-exports
include("statsbase.jl")
include("statsmodels.jl")

# models
include("model_glm.jl")
include("model_cox.jl")
include("model_ee.jl")


end # module Qgcomp
