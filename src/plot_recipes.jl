#plot_recipes.jl
# work in progress

@userplot WeightPlot
"""
Plotting weights
```julia
using Qgcomp
using Random, GLM, DataFrames, LSurvival
using Plots

x1 = rand(100, 3)
x = rand(100, 3)
z = rand(100, 3)
xq, _ = Qgcomp.get_xq(x, 4)
y = randn(100) + xq * [0.1, 0.05, 0]
lindata = DataFrame(hcat(y, x, z), [:y, :x1, :x2, :x3, :z1, :z2, :z3])

mint = qgcomp_glm_noboot(@formula(y~x1+x2+x3+z1+z2+z3), lindata, ["x1", "x2", "x3", "z1", "z2", "z3"], 4, Normal())
weightplot(mint)
```

"""
@recipe function f(h::WeightPlot)
    m = h.args[1]
    w = m.qgcweights;
    cw = vcat(w.neg, w.pos)
    cw.poswt = cw.weight .* (cw.coef .>= 0)
    cw.negwt = cw.weight .* (cw.coef .< 0)
    cw.ord = ordinalrank(abs.(cw.coef))
    sort!(cw, :ord)
    colwt = 100 .- floor.(Int, abs.(100.0 .* sort(unique(cw.ψ_partial))))

    # global, 
    # := unchangeable option
    # --> overridable default
    #xticks := false
    label := ""
    grid --> false
    xtick_direction --> :none
    ytick_direction --> :out
    link := :y
    layout := @layout [negweights posweights; lab{0.05h}]
    @series begin
        seriestype := :bar
        ylim := (-1, 0)
        xmirror := true
        xticks := false
        color := Symbol("gray$(colwt[1])")
        permute := (:x, :y)
        bottom_margin := 0mm
        right_margin := -2mm
        cw.exposure, -cw.negwt
    end
    @series begin
        permute := (:x, :y)
        seriestype := :bar
        ylim := (0, 1)
        color := Symbol("gray$(colwt[2])")
        cw.exposure, cw.poswt
    end
    @series begin
        seriestype := :scatter
        annotation := ([0], [0], "Weight")
        xlim := (-1, 1)
        axis := false
        framestyle := :none
        markersize := 0
        markerstrokewidth := 0
        color := :white
        margin := 0mm
        [0], [0]
    end
end

#= #######

Predictions from MSM

=# #######

function loesspred(x,y)
    ll = loess(x,y)
    px = range(extrema(x)..., length=101)
    ux = predict(ll, px)
    px,ux
end

function pointwise(ms; level = 0.95)
    x = unique(ms.intvector)
    y = ms.meanypred
    ylower = sqrt.(diag(ms.meanypred_vcov)) .* quantile(Normal(), 1-(1-level)/2)
    yupper = ylower#sqrt.(diag(ms.meanypred_vcov)) .* quantile(Normal(), 1-(1-level)/2)
    x,y,ylower,yupper
end

function msmwise(ms; level = 0.95)
    x = unique(ms.intvector)
    y = ms.meanypredmsm
    ylower = sqrt.(diag(ms.meanypredmsm_vcov)) .* quantile(Normal(), 1-(1-level)/2)
    yupper = ylower#sqrt.(diag(ms.meanypred_vcov)) .* quantile(Normal(), 1-(1-level)/2)
    x,y,ylower,yupper
end

@userplot ResponsePlot
"""
regression curve plot for msm

```julia
using Qgcomp
using Random, GLM, DataFrames, LSurvival
using Plots

#using LinearAlgebra, StatsBase
pointwise = Qgcomp.pointwise
#loesspred = Qgcomp.loesspred
smoothpred = Qgcomp.loesspred

x1 = rand(100, 3)
x = rand(100, 3)
z = rand(100, 3)
xq, _ = Qgcomp.get_xq(x, 4)
y = randn(100) + xq .*  xq * [0.1, 0.05, 0]
lindata = DataFrame(hcat(y, x, z), [:y, :x1, :x2, :x3, :z1, :z2, :z3])

m = qgcomp_glm_boot(@formula(y~x1+x1^2+x2+x2^2+x3+x3^2+z1+z2+z3), lindata, ["x1", "x2", "x3", "z1"], 6, Normal(), msmformula=@formula(y~mixture+mixture^2))
#x,y,ylower,yupper = pointwise(m.msm)

responseplot(m)

# plots based on these bounds, except the smooth function
bnds = bounds(m)
mw = bnds[:model]
pw = bnds[:pointwise]


```
"""
@recipe function f(h::ResponsePlot; referentindex=1, method="loess", plots=["smooth", "pointwise", "model"])
    m = h.args[1]
    ms = m.msm;
    plottypes = map(x -> x[1:min(1,end)], plots)
    # global, 
    # := unchangeable option
    # --> overridable default
    function smoothpred(x,y,method="loess") 
        if method == "loess"
            return(loesspred(x,y))
        else
            @error("Smoothing method not defined (or misspelled)")
        end
    end
    bnds = bounds(m,m.intvals,m.intvals[referentindex]; level=0.95, types = ["pointwise", "model"]);

    label := ""
    grid --> false
    xtick_direction --> :none
    ytick_direction --> :out
    if "s" ∈ plottypes
        #println("Smooth")
        @series begin
            label := "Flexible function of underlying fit"
            seriestype := :path
            smoothpred(ms.intvector, float.(ms.ypred))
        end
    end
    if "p" ∈ plottypes
        #println("Pointwise")
        pw = bnds[:pointwise]
        refidx = findfirst(pw.diff .≈ 0)
        baselin = pw.linpred[refidx]
        @series begin
            label := "Pointwise contrasts"
            seriestype := :scatter
            #x,y,ylower,yupper = pw.mixture, pw.diff .+ pw.linpred, pw.ll_diff .+ pw.linpred, pw.ul_diff .+ pw.linpred
            yerror := (abs.(pw.ll_diff .- pw.diff), pw.ul_diff .- pw.diff)
            pw.mixture, pw.diff .+ baselin
        end
    end
    if "m" ∈ plottypes
        #println("Model-wise, simultaneous 95% CI")
        mw = bnds[:model]
        @series begin
            label := "MSM fit"
            seriestype := :path
            #x,y,ylower,yupper = mw.mixture, mw.linpred, mw.ll_simul, mw.ul_simul
            mw.mixture, mw.linpred
        end
        @series begin
            label := "MSM fit"
            seriestype := :path
            #x,y,ylower,yupper = mw.mixture, mw.linpred, mw.ll_simul, mw.ul_simul
            #ribbon := (mw.ll_simul, mw.ul_simul)
            #mw.mixture, mw.linpred
            ribbon := (abs.(mw.ll_simul .- mw.linpred), mw.ul_simul .- mw.linpred)
            mw.mixture, mw.linpred
        end
    end
end




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