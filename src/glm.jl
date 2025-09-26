#glm.jl: re-exports from GLM package

for f in (
    :deviance,
    :nulldeviance,
    :dof,
    :dof_residual,
    :nullloglikelihood,
    :nobs,
    :residuals,
    :predict,
    :predict!,
    :model_response,
    :response,
    :modelmatrix,
    :hasintercept,
    :canonicallink,
)
    @eval begin
        GLM.$f(m::QGcomp_glm) = GLM.$f(m.ulfit)
    end
end


