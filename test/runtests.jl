using Test
using Qgcomp
using Random, GLM, DataFrames, LSurvival
#using DataFrames
#using BenchmarkTools # add during testing

@testset "Qgcomp.jl" begin

    println("Creating test datasets")
    # Linear
    x1 = rand(100, 3)
    x = rand(100, 3)
    z = rand(100, 3)
    xq, _ = Qgcomp.get_xq(x, 4)
    y = randn(100) + xq * [.1, 0.05, 0]
    lindata = DataFrame(hcat(y,x,z), [:y, :x1, :x2, :x3, :z1, :z2, :z3])


    f = @formula(Surv(exp(y))~x1+x2+x3+z1+z2+z3)
    data = lindata
    expnms = ["x1", "x3"]
    mcsize=2000
    


    # Cox
    id, int, out, survdata1 = LSurvival.dgm(MersenneTwister(1212), 100, 20);
    d, X = survdata1[:, 4], survdata1[:, 1:3];
    wt = ones(length(d)) # random weights just to demonstrate usage
    tab = ( in = int.*3.1, out = out.*3.1, d=d, x=X[:,1], z1=X[:,2], z2=X[:,3]) ;
    survdata = DataFrame(tab)


    form = @formula(y~x1+x2+x3+z1+z2+z3)
    form_noint = @formula(y~-1+x1+x2+x3+z1+z2+z3)
    form_noint_nonlin = @formula(y~-1+x1+x1^2+x2+x3+z1+z2+z3)
    expnms = [:x1, :x3]

    mint = qgcomp_glm_noboot(form, lindata, expnms, 4, Normal())
    mnoint = qgcomp_glm_noboot(form_noint, lindata, expnms, 4, Normal())
    
    
    mnointnolin = qgcomp_glm_boot(form_noint_nonlin, lindata, expnms, 4, Normal(), degree=2)
    mnointnolin2 = qgcomp_glm_boot(form_noint_nonlin, lindata, expnms, 4, Normal(), msmformula=@formula(y~-1+mixture + mixture^2 + x2+z1+z2+z3))





    #mcoxul = coxph(@formula(Surv(in, out, d)~x+z1+z2), tab, ties = "efron", wts = wt)
    mcox = qgcomp_cox_noboot(@formula(Surv(in, out, d)~x+z1+z2), survdata, ["z1", "z2"], 4)
    try 
        qgcomp_cox_boot(@formula(Surv(in, out, d)~x+z1+z2), survdata, ["z1", "z2"], 4, mcsize=2000)
    catch e
        @assert(typeof(e)<:AssertionError)
    end


    mcox2 = qgcomp_cox_noboot(@formula(Surv(out, d)~z1+z2), survdata, ["z1", "z2"], 4)
    mcox2boot = qgcomp_cox_boot(@formula(Surv(out, d)~z1+z2), survdata, ["z1", "z2"], 4, B=2, mcsize=8000)
    mcox2_alt = qgcomp_cox_noboot(@formula(Surv(out, d)~z1+z2+x), survdata, ["z1", "z2"], 4)
    mcox2boot_alt = qgcomp_cox_boot(@formula(Surv(out, d)~z1+z2+x), survdata, ["z1", "z2"], 4, B=2, mcsize=8000, msmformula=@formula(Surv(out, d)~mixture+x))


    println(mint)
    println(mint.ulfit)

    println(mnoint)
    println(mnoint.ulfit)

    println(mnointnolin)
    println(mnointnolin.ulfit)

    println(mnointnolin2)
    println(mnointnolin2.ulfit)

    println(mcox)
    println(mcox.ulfit)

    println(mcox2)   
    println(mcox2boot)
    println(mcox2boot.ulfit)

    println(mcox2_alt)   
    println(mcox2boot_alt)
    println(mcox2boot_alt.ulfit)

end