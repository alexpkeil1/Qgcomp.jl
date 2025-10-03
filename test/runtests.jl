using Test
using Qgcomp
using Random, GLM, DataFrames, LSurvival, StatsBase
#using DataFrames
#using BenchmarkTools # add during testing

expit(mu) = inv(1.0 + exp(-mu))

println("Creating test datasets")
# Linear
rng = Xoshiro(1212)
n = 200
x1 = rand(n, 3)
x = rand(n, 3)
z = rand(n, 3)
xq, _ = Qgcomp.get_xq(x, 4)
y = randn(n) + xq * [0.1, 0.05, 0]
ybin = Int.(expit.(-2.0 .+ xq * [0.1, 0.05, 0]) .> rand(n))
lindata = DataFrame(hcat(y, x, z), [:y, :x1, :x2, :x3, :z1, :z2, :z3])
logitdata = DataFrame(hcat(ybin, x, z), [:y, :x1, :x2, :x3, :z1, :z2, :z3])

# Cox
id, int, out, survdata1 = LSurvival.dgm(MersenneTwister(1212), 100, 20);
d, X = survdata1[:, 4], survdata1[:, 1:3];
wt = ones(length(d)) # random weights just to demonstrate usage
tab = (
    in = int .* 3.1,
    out = out .* 3.1,
    d = d,
    x1 = X[:, 1],
    x2 = rand(rng, size(X, 1)),
    x3 = rand(rng, size(X, 1)),
    z1 = X[:, 2],
    z2 = X[:, 3],
    z3 = rand(rng, size(X, 1)),
);
survdata = DataFrame(tab)


@testset "Continouous outcomes" begin
    form = @formula(y ~ x1 + x2 + x3 + z1 + z2 + z3)
    form_nonlin = @formula(y ~ x1 + x1^2 + x2 + x3 + z1 + z2 + z3)
    form_noint = @formula(y ~ -1 + x1 + x2 + x3 + z1 + z2 + z3)
    form_noint_nonlin = @formula(y ~ -1 + x1 + x1^2 + x2 + x3 + z1 + z2 + z3)
    expnms = [:x1, :x3]
    mcsize=2000
    newintvals = [0, 2, 4]

    # test 1: does specifying breaks give the same answer as leaving unspecified when the expected answer is the same?
    mint = qgcomp_glm_noboot(form, lindata, expnms, 4, Normal())
    brks = [sort(vcat([-floatmax(), floatmax()], quantile(lindata[:, xnm], [0.25, 0.5, 0.75]))) for xnm in expnms]
    mint_equiv = qgcomp_glm_noboot(form, lindata, expnms, nothing, Normal(), breaks = reduce(hcat, brks))
    @test coef(mint) == coef(mint_equiv)
    @test vcov(mint) == vcov(mint_equiv)

    # test 2: does specifying breaks over-ride q? 
    brks2 = [sort(vcat([-floatmax(), floatmax()], quantile(lindata[:, xnm], [0.2, 0.4, 0.6, 0.8]))) for xnm in expnms] # now quintiles
    mint_nonequiv = qgcomp_glm_noboot(form, lindata, expnms, 4, Normal(), breaks = reduce(hcat, brks2))
    @test coef(mint) != coef(mint_nonequiv)
    # test: do we get the correct number of quantiles even if q disagrees with breaks
    @test sort(unique(mint_nonequiv.data[:, expnms[1]])) == collect(0:4)

    # test 3: does bootstrap point estimate equal the non-bootstrapped point estimate when the MSM is the same as the underlying model?
    mint_boot = qgcomp_glm_boot(form, lindata, expnms, 4, Normal(), msmformula = @formula(y~mixture+x2+z1+z2+z3))
    mint_ee = qgcomp_glm_ee(form, lindata, expnms, 4, Normal(), msmformula = @formula(y~mixture+x2+z1+z2+z3))
    @test isapprox(sort(coef(mint_boot)), sort(coef(mint)), atol = 0.001)
    @test isapprox(sort(coef(mint_ee)), sort(coef(mint)), atol = 0.001)

    # test 1c (ee model): does specifying breaks give the same answer as leaving unspecified when the expected answer is the same?
    mint_equiv_ee = qgcomp_glm_ee( form, lindata, expnms, 4, Normal(), msmformula = @formula(y~mixture+x2+z1+z2+z3), breaks = reduce(hcat, brks), )
    @test isapprox(sort(coef(mint_ee)), sort(coef(mint_equiv_ee)), atol = 0.001)

    # test 4: does underspecifying the intvals give expected results in MSM?
    mint_intvals = qgcomp_glm_noboot(form, lindata, expnms, 4, Normal(), intvals = newintvals)
    mint_intvals_boot = qgcomp_glm_boot( form, lindata, expnms, 4, Normal(), intvals = newintvals, msmformula = @formula(y~mixture+x2+z1+z2+z3) )
    mint_intvals_ee = qgcomp_glm_ee( form, lindata, expnms, 4, Normal(), intvals = newintvals, msmformula = @formula(y~mixture+x2+z1+z2+z3) )
    @test isapprox(sort(coef(mint_intvals)), sort(coef(mint_intvals_boot)), atol = 0.001)
    @test isapprox(sort(coef(mint_intvals)), sort(coef(mint_intvals_ee)), atol = 0.001)
    @test sort(unique(mint_intvals.data[:, expnms[1]])) != newintvals
    @test sort(unique(mint_boot.msm.intvector)) != newintvals
    @test sort(unique(mint_intvals_boot.msm.intvector)) == newintvals
    @test sort(unique(mint_intvals_ee.msm.intvector)) == newintvals

    println("Testing display: linear")
    println(mint_intvals)
    println(mint_intvals_boot)
    println(mint_intvals_ee)

    # test 5: non-linear, quadratic
    # (true quadrtatic term will equal single quadratic term)
    mint_nonlin_boot = qgcomp_glm_boot( form_nonlin, lindata, expnms, 4, Normal(), intvals = newintvals, msmformula = @formula(y~mixture + mixture^2 + x2 + z1 + z2 + z3) )
    mint_nonlin_ee = qgcomp_glm_ee( form_nonlin, lindata, expnms, 4, Normal(), intvals = newintvals, msmformula = @formula(y~mixture + mixture^2 + x2 + z1 + z2 + z3) )
    @test isapprox(coef(mint_nonlin_boot)[3], coef(mint_nonlin_boot.ulfit)[3], atol = 0.0001) &&
          !isapprox(coef(mint_nonlin_boot)[2], coef(mint_nonlin_boot.ulfit)[2], atol = 0.0001) &&
          isapprox( coef(mint_nonlin_boot)[2], coef(mint_nonlin_boot.ulfit)[2]+coef(mint_nonlin_boot.ulfit)[5], atol = 0.0001, )
    @test isapprox(coef(mint_nonlin_ee)[3], mint_nonlin_ee.ulfit.cols[1][3], atol = 0.0001) &&
          !isapprox(coef(mint_nonlin_ee)[2], mint_nonlin_ee.ulfit.cols[1][2], atol = 0.0001) &&
          isapprox( coef(mint_nonlin_ee)[2], mint_nonlin_ee.ulfit.cols[1][2]+mint_nonlin_ee.ulfit.cols[1][5], atol = 0.0001, )

    println("Testing display: non linear (polynomial)")
    println(mint_nonlin_boot)
    println(mint_nonlin_ee)
      
    # test 5b: non-linear, spline
    knts = rsplineknots(lindata.x1, 5)
    knts2 = rsplineknots(lindata.x2, 5)
    form_spline = term(:y) ~ term(1) + rcs(:x1, knts) + sum([term(Symbol("x$i")) for i in 2:3]) + sum([term(Symbol("z$i")) for i in 1:3])
    form_spline_int = term(:y) ~ term(1) + rcs(:x1, knts)*rcs(:x2, knts2) + sum([term(Symbol("x$i")) for i in 2:3]) + sum([term(Symbol("z$i")) for i in 1:3])

    mint_spline_boot = qgcomp_glm_boot( form_spline, lindata, expnms, nothing, Normal(), intvals = newintvals, msmformula = @formula(y~mixture + mixture^2 + x2 + z1 + z2 + z3) )
    mint_spline_ee = qgcomp_glm_ee( form_spline, lindata, expnms, nothing, Normal(), intvals = newintvals, msmformula = @formula(y~mixture + mixture^2 + x2 + z1 + z2 + z3) )
    mint_spline_ee2 = qgcomp_glm_ee( form_spline_int, lindata, expnms, nothing, Normal(), intvals = newintvals, msmformula = @formula(y~mixture + mixture^2 + x2 + z1 + z2 + z3) )

    println("Testing display: non linear (spline)")
    println(mint_spline_boot)
    println(mint_spline_ee)
    println(mint_spline_ee2)

    # test 6: non-linear, interaction
    # note that ordering of terms in underlying fit vary. To fix?
    # (true quadrtatic term will equal single quadratic term), main term will not but will equal sum of terms
    form_nonlinx = @formula(y ~ x1 + x1*x3 + x2 + x3 + z1 + z2 + z3)
    mint_nonlinx_boot = qgcomp_glm_boot( form_nonlinx, lindata, expnms, 4, Normal(), intvals = newintvals, msmformula = @formula(y~mixture + mixture^2 + x2 + z1 + z2 + z3) )
    mint_nonlinx_ee = qgcomp_glm_ee( form_nonlinx, lindata, expnms, 4, Normal(), intvals = newintvals, msmformula = @formula(y~mixture + mixture^2 + x2 + z1 + z2 + z3) )
    @test isapprox(coef(mint_nonlinx_boot)[3], coef(mint_nonlinx_boot.ulfit)[end], atol = 0.0001) &&
          !isapprox(coef(mint_nonlinx_boot)[2], coef(mint_nonlinx_boot.ulfit)[2], atol = 0.0001) &&
          isapprox( coef(mint_nonlinx_boot)[2], coef(mint_nonlinx_boot.ulfit)[2]+coef(mint_nonlinx_boot.ulfit)[3], atol = 0.0001, )
    @test isapprox(coef(mint_nonlinx_ee)[3], mint_nonlinx_ee.ulfit.cols[1][8], atol = 0.0001) &&
          !isapprox(coef(mint_nonlinx_ee)[2], mint_nonlinx_ee.ulfit.cols[1][2], atol = 0.0001) &&
          isapprox( coef(mint_nonlinx_ee)[2], mint_nonlinx_ee.ulfit.cols[1][2]+mint_nonlinx_ee.ulfit.cols[1][3], atol = 0.0001, )

    # test 7: no intercept
    mnoint = qgcomp_glm_noboot(form_noint, lindata, expnms, 4, Normal())
    @test length(coef(mnoint)) == length(coef(mint)) - 1

    println("Testing display: no intercept")
    println(mnoint)

    # test 8: leaving q=nothing with standardized data
    mint_noq = qgcomp_glm_noboot(form, lindata, expnms, nothing, Normal())
    glm_noq = glm(form, lindata, Normal())
    mint_nonlin_noq = qgcomp_glm_noboot(form_nonlin, lindata, expnms, nothing, Normal())
    glm_nonlin_noq = glm(form_nonlin, lindata, Normal())
    @test coef(mint_noq.ulfit) == coef(glm_noq)
    @test coef(mint_nonlin_noq.ulfit) == coef(glm_nonlin_noq)
end



@testset "Survival outcomes" begin

    form = @formula(Surv(in, out, d) ~ x1 + x2 + x3 + z1 + z2 + z3)
    form_nole = @formula(Surv(out, d) ~ x1 + x2 + x3 + z1 + z2 + z3)
    form_nonlin = @formula(Surv(in, out, d) ~ x1 + x1^2 + x2 + x3 + z1 + z2 + z3)
    form_noint = @formula(Surv(in, out, d) ~ -1 + x1 + x2 + x3 + z1 + z2 + z3)
    form_noint_nonlin = @formula(Surv(in, out, d) ~ -1 + x1 + x1^2 + x2 + x3 + z1 + z2 + z3)
    form_simple = @formula(Surv(in, out, d) ~ x1 + x2)
    expnms = [:x1, :x3]
    mcsize = 2000

    # test 1: does specifying breaks give the same answer as leaving unspecified when the expected answer is the same?
    mint = qgcomp_cox_noboot(form, survdata, expnms, 4)
    brks = [sort(vcat([-floatmax(), floatmax()], quantile(survdata[:, xnm], [0.25, 0.5, 0.75]))) for xnm in expnms]
    mint_equiv = qgcomp_cox_noboot(form, survdata, expnms, nothing, breaks = reduce(hcat, brks))
    @test coef(mint) == coef(mint_equiv)
    @test vcov(mint) == vcov(mint_equiv)

    println("Testing display: Cox model")
    println(mint)

    # test 2: does specifying breaks over-ride q? 
    brks2 = [sort(vcat([-floatmax(), floatmax()], quantile(survdata[:, xnm], [0.2, 0.4, 0.6, 0.8]))) for xnm in expnms] # now quintiles
    mint_nonequiv = qgcomp_cox_noboot(form, survdata, expnms, 4, breaks = reduce(hcat, brks2))
    @test coef(mint) != coef(mint_nonequiv)
    # test: do we get the correct number of quantiles even if q disagrees with breaks
    @test sort(unique(mint_nonequiv.data[:, expnms[1]])) == collect(0:4)

    # test 3: does bootstrap point estimate equal the non-bootstrapped point estimate when the MSM is the same as the underlying model?
    mint_nole = qgcomp_cox_noboot(form_nole, survdata, expnms, 3)
    mint_nole_boot = qgcomp_cox_boot( Xoshiro(1232), form_nole, survdata, expnms, 3, msmformula = @formula(Surv(out, d)~mixture+x2+z1+z2+z3), B = 1, mcsize = 20000, )
    @test isapprox(sort(coef(mint_nole)), sort(coef(mint_nole_boot)), atol = 0.05)

    # test 4: does underspecifying the intvals give expected results in MSM?
    newintvals = [0.0, 2.0, 4.0]
    mint_intvals = qgcomp_cox_noboot(form_nole, survdata, expnms, 4, intvals = newintvals)
    mint_intvals_boot = qgcomp_cox_boot( form_nole, survdata, expnms, 4, intvals = newintvals, msmformula = @formula(Surv(out, d)~mixture+x2+z1+z2+z3), B = 2, mcsize = 20000, )
    @test isapprox(sort(coef(mint_intvals)), sort(coef(mint_intvals_boot)), atol = 0.05)
    @test sort(unique(mint_intvals.data[:, expnms[1]])) != newintvals
    @test sort(unique(mint_nole_boot.msm.intvector)) != newintvals
    @test sort(unique(mint_intvals_boot.msm.intvector)) == newintvals

    println("Testing display: Cox model")
    println(mint_intvals_boot)

    # test 5: non-linear, quadratic
    # (true quadrtatic term will equal single quadratic term)
    # very time intensive, skipped

    # test 5b: non-linear, spline
    knts = rsplineknots(survdata.x1, 5)
    knts2 = rsplineknots(survdata.x2, 5)
    ff = @formula(Surv(out, d)~1)
    form_spline = ff.lhs ~ rcs(:x1, knts) + sum([term(Symbol("x$i")) for i in 2:3]) + sum([term(Symbol("z$i")) for i in 1:3])
    newintvals = [0.0, 2.0, 4.0]
    mint_spline = qgcomp_cox_boot(form_spline, survdata, expnms, nothing, intvals = newintvals, msmformula=@formula(Surv(out, d)~mixture + mixture^2))
    println("Testing display: Cox model with spline")
    println(mint_spline)

    # test 6: non-linear, interaction
    # (true quadrtatic term will equal single quadratic term), main term will not but will equal sum of terms
    # very time intensive, skipped

    # test 7: no intercept (just check for error here)
    msg = try
        qgcomp_cox_noboot(form_noint, survdata, expnms, 4)
    catch e
        e
    end
    @test typeof(msg) == MethodError

    # test 8: leaving q=nothing with standardized data
    mint_noq = qgcomp_cox_noboot(form, survdata, expnms, nothing)
    coxph_noq = coxph(form, survdata)
    mint_nonlin_noq = qgcomp_cox_noboot(form_nonlin, survdata, expnms, nothing)
    coxph_nonlin_noq = coxph(form_nonlin, survdata)
    @test coef(mint_noq.ulfit) == coef(coxph_noq)
    @test sort(coef(mint_nonlin_noq.ulfit)) == sort(coef(coxph_nonlin_noq))

    # test 5: does bootstrapping appropriately catch issues with late entry
    #mcoxul = coxph(@formula(Surv(in, out, d)~x+z1+z2), tab, ties = "efron", wts = wt)
    mcox = qgcomp_cox_noboot(form, survdata, expnms, 4)
    e = try
        qgcomp_cox_boot(form, survdata, expnms, 4, mcsize = 2000)
    catch e
        e
    end
    @test typeof(e)<:AssertionError
end


@testset "Binary outcomes" begin
    form = @formula(y ~ x1 + x2 + x3 + z1 + z2 + z3)
    form_nonlin = @formula(y ~ x1 + x1^2 + x2 + x3 + z1 + z2 + z3)
    form_noint = @formula(y ~ -1 + x1 + x2 + x3 + z1 + z2 + z3)
    form_noint_nonlin = @formula(y ~ -1 + x1 + x1^2 + x2 + x3 + z1 + z2 + z3)
    expnms = [:x1, :x3]
    mcsize=2000

    # test 1: does specifying breaks give the same answer as leaving unspecified when the expected answer is the same?
    mint = qgcomp_glm_noboot(form, logitdata, expnms, 4, Binomial())
    brks = [sort(vcat([-floatmax(), floatmax()], quantile(logitdata[:, xnm], [0.25, 0.5, 0.75]))) for xnm in expnms]
    mint_equiv = qgcomp_glm_noboot(form, logitdata, expnms, nothing, Binomial(), breaks = reduce(hcat, brks))
    @test coef(mint) == coef(mint_equiv)
    @test vcov(mint) == vcov(mint_equiv)

    # test 2: does specifying breaks over-ride q? 
    brks2 = [sort(vcat([-floatmax(), floatmax()], quantile(logitdata[:, xnm], [0.2, 0.4, 0.6, 0.8]))) for xnm in expnms] # now quintiles
    mint_nonequiv = qgcomp_glm_noboot(form, logitdata, expnms, 4, Binomial(), breaks = reduce(hcat, brks2))
    @test coef(mint) != coef(mint_nonequiv)
    # test: do we get the correct number of quantiles even if q disagrees with breaks
    @test sort(unique(mint_nonequiv.data[:, expnms[1]])) == collect(0:4)

    # test 3: does bootstrap point estimate equal the non-bootstrapped point estimate when the MSM is the same as the underlying model?
    mint_boot = qgcomp_glm_boot(form, logitdata, expnms, 4, Binomial(), msmformula = @formula(y~mixture+x2+z1+z2+z3))
    mint_ee = qgcomp_glm_ee(form, logitdata, expnms, 4, Binomial(), msmformula = @formula(y~mixture+x2+z1+z2+z3))
    @test isapprox(sort(coef(mint_boot)), sort(coef(mint)), atol = 0.001)
    @test isapprox(sort(coef(mint_ee)), sort(coef(mint)), atol = 0.001)

    # test 1c (ee model): does specifying breaks give the same answer as leaving unspecified when the expected answer is the same?
    mint_equiv_ee = qgcomp_glm_ee(
        form,
        logitdata,
        expnms,
        4,
        Binomial(),
        msmformula = @formula(y~mixture+x2+z1+z2+z3),
        breaks = reduce(hcat, brks),
    )
    @test isapprox(sort(coef(mint_ee)), sort(coef(mint_equiv_ee)), atol = 0.001)

    # test 4: does underspecifying the intvals give expected results in MSM?
    newintvals = [0, 2, 4]
    mint_intvals = qgcomp_glm_noboot(form, logitdata, expnms, 4, Binomial(), intvals = newintvals)
    mint_intvals_boot = qgcomp_glm_boot( form, logitdata, expnms, 4, Binomial(), intvals = newintvals, msmformula = @formula(y~mixture+x2+z1+z2+z3) )
    mint_intvals_ee = qgcomp_glm_ee( form, logitdata, expnms, 4, Binomial(), intvals = newintvals, msmformula = @formula(y~mixture+x2+z1+z2+z3) )
    @test isapprox(sort(coef(mint_intvals)), sort(coef(mint_intvals_boot)), atol = 0.001)
    @test isapprox(sort(coef(mint_intvals)), sort(coef(mint_intvals_ee)), atol = 0.001)
    @test sort(unique(mint_intvals.data[:, expnms[1]])) != newintvals
    @test sort(unique(mint_boot.msm.intvector)) != newintvals
    @test sort(unique(mint_intvals_boot.msm.intvector)) == newintvals
    @test sort(unique(mint_intvals_ee.msm.intvector)) == newintvals

    # test 5: non-linear, quadratic
    # (true quadrtatic term will equal single quadratic term)
    mint_nonlin_boot = qgcomp_glm_boot( form_nonlin, logitdata, expnms, 4, Binomial(), intvals = newintvals, msmformula = @formula(y~mixture + mixture^2 + x2 + z1 + z2 + z3) )
    mint_nonlin_ee = qgcomp_glm_ee( form_nonlin, logitdata, expnms, 4, Binomial(), intvals = newintvals, msmformula = @formula(y~mixture + mixture^2 + x2 + z1 + z2 + z3) )
    @test isapprox(coef(mint_nonlin_boot)[3], coef(mint_nonlin_boot.ulfit)[3], atol = 0.0001) &&
          !isapprox(coef(mint_nonlin_boot)[2], coef(mint_nonlin_boot.ulfit)[2], atol = 0.0001) &&
          isapprox( coef(mint_nonlin_boot)[2], coef(mint_nonlin_boot.ulfit)[2]+coef(mint_nonlin_boot.ulfit)[5], atol = 0.0001, )
    @test isapprox(coef(mint_nonlin_ee)[3], mint_nonlin_ee.ulfit.cols[1][3], atol = 0.0001) &&
          !isapprox(coef(mint_nonlin_ee)[2], mint_nonlin_ee.ulfit.cols[1][2], atol = 0.0001) &&
          isapprox( coef(mint_nonlin_ee)[2], mint_nonlin_ee.ulfit.cols[1][2]+mint_nonlin_ee.ulfit.cols[1][5], atol = 0.0001, )

    # test 5b: non-linear, spline
    knts = rsplineknots(logitdata.x1, 5)
    knts2 = rsplineknots(logitdata.x2, 5)
    form_spline = term(:y) ~ term(1) + rcs(:x1, knts) + sum([term(Symbol("x$i")) for i in 2:3]) + sum([term(Symbol("z$i")) for i in 1:3])
    form_spline_int = term(:y) ~ term(1) + rcs(:x1, knts)*rcs(:x2, knts2) + sum([term(Symbol("x$i")) for i in 2:3]) + sum([term(Symbol("z$i")) for i in 1:3])

    mint_spline_boot = qgcomp_glm_boot( form_spline, logitdata, expnms, nothing, Normal(), intvals = newintvals, msmformula = @formula(y~mixture + mixture^2 + x2 + z1 + z2 + z3) )
    mint_spline_ee = qgcomp_glm_ee( form_spline, logitdata, expnms, nothing, Normal(), intvals = newintvals, msmformula = @formula(y~mixture + mixture^2 + x2 + z1 + z2 + z3) )
    mint_spline_ee2 = qgcomp_glm_ee( form_spline_int, logitdata, expnms, nothing, Normal(), intvals = newintvals, msmformula = @formula(y~mixture + mixture^2 + x2 + z1 + z2 + z3) )

    println("Testing display: non linear (spline, logistic)")
    println(mint_spline_boot)
    println(mint_spline_ee)
    println(mint_spline_ee2)

    # test 6: non-linear, interaction
    # (true quadrtatic term will equal single quadratic term), main term will not but will equal sum of terms
    form_nonlinx = @formula(y ~ x1 + x1*x3 + x2 + x3 + z1 + z2 + z3)
    mint_nonlinx_boot = qgcomp_glm_boot( form_nonlinx, logitdata, expnms, 4, Binomial(), intvals = newintvals, msmformula = @formula(y~mixture + mixture^2 + x2 + z1 + z2 + z3) )
    mint_nonlinx_ee = qgcomp_glm_ee( form_nonlinx, logitdata, expnms, 4, Binomial(), intvals = newintvals, msmformula = @formula(y~mixture + mixture^2 + x2 + z1 + z2 + z3) )
    @test isapprox(coef(mint_nonlinx_boot)[3], coef(mint_nonlinx_boot.ulfit)[end], atol = 0.0001) &&
          !isapprox(coef(mint_nonlinx_boot)[2], coef(mint_nonlinx_boot.ulfit)[2], atol = 0.0001) &&
          isapprox( coef(mint_nonlinx_boot)[2], coef(mint_nonlinx_boot.ulfit)[2]+coef(mint_nonlinx_boot.ulfit)[3], atol = 0.0001, )
    @test isapprox(coef(mint_nonlinx_ee)[3], mint_nonlinx_ee.ulfit.cols[1][8], atol = 0.0001) &&
          !isapprox(coef(mint_nonlinx_ee)[2], mint_nonlinx_ee.ulfit.cols[1][2], atol = 0.0001) &&
          isapprox( coef(mint_nonlinx_ee)[2], mint_nonlinx_ee.ulfit.cols[1][2]+mint_nonlinx_ee.ulfit.cols[1][3], atol = 0.0001, )

    # test 7: no intercept
    mnoint = qgcomp_glm_noboot(form_noint, logitdata, expnms, 4, Binomial())
    @test length(coef(mnoint)) == length(coef(mint)) - 1

    # test 8: leaving q=nothing with standardized data
    mint_noq = qgcomp_glm_noboot(form, logitdata, expnms, nothing, Binomial())
    glm_noq = glm(form, logitdata, Binomial())
    mint_nonlin_noq = qgcomp_glm_noboot(form_nonlin, logitdata, expnms, nothing, Binomial())
    glm_nonlin_noq = glm(form_nonlin, logitdata, Binomial())
    @test coef(mint_noq.ulfit) == coef(glm_noq)
    @test coef(mint_nonlin_noq.ulfit) == coef(glm_nonlin_noq)

    # misc uncategorized function calls

    # these are expected to converge to zero in large samples
    #=
    logit_mnointnolin = qgcomp_glm_boot(form_noint_nonlin, logitdata, expnms, 4, Binomial())
    logitlog_mnointnolin = qgcomp_glm_boot(form_noint_nonlin, logitdata, expnms, 4, Binomial(), msmlink = LogLink())
    #logitlog_mnointnolin_alt =  qgcomp_glm_boot(form_noint_nonlin, logitdata, expnms, 4, Binomial(), rr=true)
    logitident_mnointnolin = qgcomp_glm_boot( form_noint_nonlin, logitdata, expnms, 4, Binomial(), msmfamily = Normal(), msmlink = IdentityLink(), )
    logit_mnointnolin = qgcomp_glm_boot(form_noint_nonlin, logitdata, expnms, 4, Binomial(), degree = 2)
    logit_mnointnolin2 = qgcomp_glm_boot( form_noint_nonlin, logitdata, expnms, 4, Binomial(), msmformula = @formula(y~-1 + mixture + mixture^2 + x2 + z1 + z2 + z3) )

    # misc 
    logit_mnoint = qgcomp_glm_noboot(form_noint, logitdata, expnms, 4, Bernoulli())
    =#
end



@testset "Count outcomes" begin
    form = @formula(y ~ x1 + x2 + x3 + z1 + z2 + z3)
    form_nonlin = @formula(y ~ x1 + x1^2 + x2 + x3 + z1 + z2 + z3)
    form_noint = @formula(y ~ -1 + x1 + x2 + x3 + z1 + z2 + z3)
    form_noint_nonlin = @formula(y ~ -1 + x1 + x1^2 + x2 + x3 + z1 + z2 + z3)
    expnms = [:x1, :x3]
    mcsize=2000

end



