module Qgcomp
using StatsBase
using GLM
using DataFrames
using LinearAlgebra
using Distributions: Normal, cdf, quantile
using NLsolve
using ForwardDiff
using LSurvival

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
export qgcomp_glm_noboot, qgcomp_ee_noboot, qgcomp_cox_noboot

#expanded functions from imports
export fit!, aic, aicc, bic, loglikelihood, isfitted

#re-exports
export Normal, Bernoulli, Poisson



include("Types.jl")
include("base.jl")
include("glm.jl")
include("cox.jl")
include("ee.jl")


end # module Qgcomp
