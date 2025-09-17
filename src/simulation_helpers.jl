#simulation_helpers.jl


function genxq(rng, n, corr = (0.95, -0.3), q = 4)
    x = rand(rng, n, length(corr)+1)
    props = abs.(corr)
    ns = floor.(Int, n .* props)
    nidx = [setdiff(1:n, sample(rng, 1:n, nsi, replace = false)) for nsi in ns]
    for (j, c) in enumerate(corr)
        x[:, j+1] .= x[:, 1]
        x[nidx[j], j+1] .= sample(rng, x[nidx[j], 1], length(nidx[j]), replace = false)
        if c < 0
            x[:, j+1] .= 1.0 .- x[:, j+1]
        end
    end
    xq, _ = Qgcomp.get_xq(x, q)
    x, xq
end