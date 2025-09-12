using Test
using Qgcomp
using Random, GLM, DataFrames, LSurvival
#using DataFrames
#using BenchmarkTools # add during testing

@testset "Qgcomp.jl" begin

    println("Creating test datasets")
    x1 = rand(100, 3)
    x = rand(100, 3)
    z = rand(100, 3)
    xq, _ = Qgcomp.get_xq(x, 4)
    y = randn(100) + xq * [.1, 0.05, 0]
    lindata = DataFrame(hcat(y,x,z), [:y, :x1, :x2, :x3, :z1, :z2, :z3])


    form = @formula(y~x1+x2+x3+z1+z2+z3)
    form_noint = @formula(y~-1+x1+x2+x3+z1+z2+z3)
    expnms = [:x1, :x3]

    m = qgcomp_glm_noboot(form, lindata, expnms, 4, Normal())
    mnoint = qgcomp_glm_noboot(form_noint, lindata, expnms, 4, Normal())

    println(m)
    println(m.ulfit)

    println(mnoint)



    id, int, out, survdata1 = LSurvival.dgm(MersenneTwister(1212), 100, 20);

    d, X = survdata1[:, 4], survdata1[:, 1:3];
    wt = ones(length(d)) # random weights just to demonstrate usage
    tab = ( in = int, out = out, d=d, x=X[:,1], z1=X[:,2], z2=X[:,3]) ;
    survdata = DataFrame(tab)

    mcoxul = coxph(@formula(Surv(in, out, d)~x+z1+z2), tab, ties = "efron", wts = wt) |> display
    mcox = qgcomp_cox_noboot(@formula(Surv(in, out, d)~x+z1+z2), survdata, ["z1", "z2"], 4) |> display

    println(mcoxul)
    println(mcox)





end