#plot_recipes.jl
# work in progress


#= weight plot
using Qgcomp
using Random, GLM, DataFrames, LSurvival
using Plots

    x1 = rand(100, 3)
    x = rand(100, 3)
    z = rand(100, 3)
    xq, _ = Qgcomp.get_xq(x, 4)
    y = randn(100) + xq * [0.1, 0.05, 0]
    ybin =  Int.(expit.(xq * [0.1, 0.05, 0]) .> rand(100))
    lindata = DataFrame(hcat(y, x, z), [:y, :x1, :x2, :x3, :z1, :z2, :z3])
    logitdata = DataFrame(hcat(ybin, x, z), [:y, :x1, :x2, :x3, :z1, :z2, :z3])

    mint = qgcomp_glm_noboot(@formula(y~x1+x2+x3+z1+z2+z3), lindata, ["x1", "x2", "x3", "z1", "z2", "z3"], 4, Normal())

    mint.qgcweights.pos

    # butterfly plot candidates
    p = plot()
    plot!(p, mint.qgcweights.pos.exposure, mint.qgcweights.pos.weight, st=:bar)
    
    p = plot()
    plot!(p, mint.qgcweights.pos.exposure, mint.qgcweights.pos.weight, st=:bar, permute=(:x, :y), ylim=(0,1))

    cw = vcat(
        mint.qgcweights.neg,
        mint.qgcweights.pos
    )
    cw.poswt = cw.weight .* (cw.coef .>= 0)
    cw.negwt = cw.weight .* (cw.coef .< 0)
    srtrd = DataFrame(exposure=coefnames(mint.ulfit), ord=1:length(coefnames(mint.ulfit)))
    cw = leftjoin(cw, srtrd, on=:exposure)
    cw.colwt = abs(cw.ψ_partial)


    # promising
    # need to figure out middle label spacing
    p1 = plot(grid=false, yticks=false, xlim=(-1,0), ymirror=true, ytick_direction=:none, xtick_direction=:out)
    plot!(p1, cw.exposure, -cw.negwt, st=:bar, permute=(:x, :y), label="", color=RGBA(0,0,0,0.5))

    p2 = plot(grid=false, yticks=true, xlim=(0,1), ytick_direction=:none, xtick_direction=:out)
    plot!(p2, cw.exposure, cw.poswt, st=:bar, permute=(:x, :y), label="", color=RGBA(0,0,0,0.5))

    plot(p1, p2, link=:y, size=(600,400))    
=#

#= regression curve plot for msm
    # note: may need more for this plot
using Qgcomp
using Random, GLM, DataFrames, LSurvival
using Plots

    x1 = rand(100, 3)
    x = rand(100, 3)
    z = rand(100, 3)
    xq, _ = Qgcomp.get_xq(x, 4)
    y = randn(100) + xq * [0.1, 0.05, 0]
    ybin =  Int.(expit.(xq * [0.1, 0.05, 0]) .> rand(100))
    lindata = DataFrame(hcat(y, x, z), [:y, :x1, :x2, :x3, :z1, :z2, :z3])
    logitdata = DataFrame(hcat(ybin, x, z), [:y, :x1, :x2, :x3, :z1, :z2, :z3])

    m = qgcomp_glm_boot(@formula(y~x1+x2+x3+z1+z2+z3), lindata, ["x1", "x2", "x3", "z1", "z2", "z3"], 4, Normal())

    m.msm.intval
    m.msm.ypred
    m.msm.meanypred_draws

    # line plot candidates
    p = plot()
    plot!(p, m.msm.intval, m.msm.ypred, st=:line)
    plot!(p, m.msm.intval, m.msm.ypred, st=:scatter)
    
=#



#=
@recipe function f(c::PHSurv)
    # convenience munging
    maxT = maximum(vcat(c.fitlist[1].R.exit, c.fitlist[2].R.exit))
    minT = min(c.fitlist[1].R.origin, c.fitlist[2].R.origin)
    # plot settings
    xlabel --> "Time" # --> sets default
    ylabel --> "Risk"
    label --> ""       # := over-rides user choices
    grid --> false
    xlim --> (minT, maxT)
    ylim --> (0, maximum(c.risk))
    for j in eachindex(c.eventtypes)
        @series begin
            seriestype := :step
            markershape := :none
            label := c.eventtypes[j]
            vcat(minT, c.times, maxT), vcat(0, c.risk[:, j], maximum(c.risk[:, j]))
        end
    end
end
=#