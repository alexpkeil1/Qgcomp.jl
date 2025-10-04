# Some underlying mathematics of Qgcomp

Here are some of the mathematical building blocks of the package. None of these are new, but they are helpful for understanding how the approach works.

## Rules of variance:
### Variance of a sum of random variables
$Var(X + Z) = var(X) + Var(Z) + Cov(X,Z) + Cov(Z,X)$
$Var(X - Z) = Var(X + -Z)$
$= Var(X) + Var(-Z) + Cov(X,-Z) + Cov(-Z,X)$
$= Var(X) + Var(Z) - Cov(X,Z) - Cov(Z,X)$

### Variance of a sum of sums of random variables
$A=X+Z, B=Y+W$
$Var(A)= Var(X) + Var(Z) + 2*Cov(X,Z)$
$Var(B)= Var(Y) + Var(W) + 2*Cov(Y,W)$
$Var(A+B)= Var(A) + Var(B) + 2*Cov(A,B)$

### Variance of a product of a random variable and a constant
$Var(X*c) = E(X*c-E(X*c))^2$
$= E((X*c-cE(X))^2)$
$= E((X*c-cE(X))^2)$
$= E((c(X-E(X))^2)$
$= c^2 * E((X-E(X))^2)$
$= c^2 * Var(X)$

### Covariance of a random variable and a product of a random variable and a constant
$Cov(X*c, Z) = E[(X*c-E(X*c))(Z-E(X)]$
$= c*E[(X-E(X))(Z-E(X)]$
$= c * Cov(X, Z)$


# Generalized linear model characteristics

$E(g(y)|x) = β0 + β1*x1 + β2*x2$


## Standard error of a linear combination 

$RD = E(y|x=1) - E(y|x=0)$
$= β0 + β1*1 + β2*1 - (β0 + β1*0 + β2*0)$
$=β1 + β2$
$Var(RD) = Var(β1 + β2)$
$= Var(β1) + Var(β2) + 2*Cov(β1,β2)$


## Variance of a prediction
Define a prediction for a two exposure linear model as
$E(y|x) = β0 + β1*x1 + β2*x2$

$Var(E(y|x)) = Var(β0 + β1*x1 + β2*x2)$
$= Var(β0 + β1*x1 + β2*x2)$
$= Var(β0) + Var(β1*x1) + Var(β2*x2) + 2*Cov(β0,β1*x1) + 2*Cov(β0,β2*x2) + 2*Cov(β1*x1,β2*x2)$
$= Var(β0) + x1^2*Var(β1) + x2^2*Var(β2) + 2*Cov(β0,β1)*x1 + 2*Cov(β0,β2)*x2 + 2*x1*Cov(β1,β2)*x2$

## Variance/covariance matrix for a vector of predictions
$\mathbf{1} = (1, 1, ..., 1)'$
$X1 = (x1_1, x1_2, ..., x1_n)'$
$X2 = (x2_1, x2_2, ..., x2_n)'$
$\mathbf{X} = (\mathbf{1}, X1, X2)$
$\mathbf{β} = (β0, β1, β2)'$
$Vcov(E(y|\mathbf{X})) = Vcov(β0 + β1*X1 + β2*X2)$
$Vcov(E(y|\mathbf{X})) = Vcov(\mathbf{Xβ})$
$= (\mathbf{Xβ} - E\mathbf{Xβ})(\mathbf{Xβ} - E\mathbf{Xβ})'$
$= (\mathbf{Xβ} - E\mathbf{Xβ})(\mathbf{β'X} - E\mathbf{β'X})$
$= \mathbf{Xββ'X} - E(\mathbf{Xβ})\mathbf{β'X}  - \mathbf{Xβ}E(\mathbf{X'β}) + E(\mathbf{Xβ})*E(\mathbf{β'X})$






```@example math
using GLM, Random, LinearAlgebra

x = rand(100, 2)
y = 0.1 .+ x * [1.0, 0.3] + randn(100)
X = hcat(ones(length(y)), x)
ft = fit(LinearModel, X, y)
V = GLM.vcov(ft)
diag(X * V * X')

println("Coefficient estimate from Qgcomp")
grad = zeros(length(coef(ft)))
grad[2:3] .= 1 # second and third coefficients are exposures from the mixture
println(sum(coef(ft)[findall(grad .== 1)]))
println("Coefficient standard error from Qgcomp")
println(sqrt(grad' * V * grad))
```

```@example math
using Qgcomp, DataFrames
qgcomp_glm_noboot(@formula(y~x1+x2), DataFrame(y=y, x1=x[:,1], x2=x[:,2]), [:x1, :x2], nothing, Normal())

```

