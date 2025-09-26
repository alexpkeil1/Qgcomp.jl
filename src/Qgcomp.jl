
module Qgcomp
# utility
using DataFrames
# general stat/math
using StatsBase
using LinearAlgebra
# models
using GLM
using LSurvival
# numerical algorithms
using NLsolve
using ForwardDiff
# plotting
using RecipesBase
using Loess

using Distributions: Normal, cdf, quantile, Distribution, ContinuousUnivariateDistribution, Chisq, MvNormal

# imports
import Random: AbstractRNG, Xoshiro, MersenneTwister
import Measures: mm

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
    ordinalrank,
    predict,
    predict!,
    response,
    score,
    var,
    vcov,
    weights

#types
export QGcomp_model, QGcomp_weights, Qgcomp_MSM, QGcomp_glm, QGcomp_ee, QGcomp_cox

# original fitting functions
export qgcomp_glm_noboot, qgcomp_glm_boot, qgcomp_cox_noboot, qgcomp_cox_boot, qgcomp_glm_ee

# original summary functions
export bounds, printbounds

# original utility functions
export ID, genxq, vccomb

#expanded functions from imports
export fit!, aic, aicc, bic, loglikelihood, fitted, isfitted, coef, vcov

#re-exports, functions
export diag, ordinalrank, DummyCoding, HelmertCoding, EffectsCoding, HypothesisCoding, SeqDiffCoding

#re-exports, types
export Distribution,
    Normal,
    Bernoulli,
    Poisson,
    Binomial,
    LogitLink,
    IdentityLink,
    NegativeBinomialLink,
    ProbitLink,
    @formula,
    FormulaTerm,
    Term,
    InteractionTerm,
    FunctionTerm


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

# summary functions
include("bounds.jl")
include("plot_recipes.jl")

end # module Qgcomp
