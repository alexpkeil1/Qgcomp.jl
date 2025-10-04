#! cd /Users/keilap/repo/Qgcomp_julia/Qgcomp.jl/docs; /usr/bin/env julia --project '/Users/keilap/repo/Qgcomp_julia/Qgcomp.jl/docs/make.jl'

# Root of the repository
const repo_root = dirname(@__DIR__)

using Pkg
Pkg.Registry.add(url="https://github.com/alexpkeil1/EpiRegistry.git")

using Documenter, Qgcomp
using Documenter.Remotes: GitHub
using Weave

println(repo_root)

jmdfiles = ["underlying_math.md"]

for jmdf in jmdfiles
#    weave(joinpath(repo_root, "docs/src/", jmdf))
end

#DocMeta.setdocmeta!(LSurvival, :DocTestSetup, :(using LSurvival); recursive = true)
# Note, use dev .. (rather than add ...) from docs directory to load a version of Qgcomp to create docs that doesn't require github commit
# You may need to manually install LSurvival first
push!(LOAD_PATH, "../src/")

makedocs(;
    format = Documenter.HTML(; size_threshold=1_000_000),
    modules = [Qgcomp],
    sitename = "Qgcomp: quantile g-computation in Julia",
    pages = [
        "Help" => "index.md",
        "Details" =>[
            "Generalized Linear Models" => "glm.md", 
            "Time-to-event and advanced topics" => "cox.md", 
            ],
        "Math" =>["Underlying math" => "underlying_math.md"],
        #=
        "Examples" => [
            "Non-parametric survival analysis" => "nonparametric.md",
            "Semi-parametric survival analysis with Cox models" => "coxmodel.md",
            "Parametric survival analysis with AFT models" => "parametric.md",
            ],
            =#
    ],
    debug = true,
    checkdocs=:exports,
    doctest = true,
    source = "src",
    build = "build",
    highlightsig = true,
    repo = GitHub("alexpkeil1", "Qgcomp.jl"),
    remotes = nothing
)

deploydocs(;
    repo = "github.com/alexpkeil1/Qgcomp.jl.git",
    branch = "gh-pages",
    target = "build",
    devbranch = "main",
    devurl = "dev",
    versions = ["stable" => "v^", "v#.#", "dev" => "dev"],
)
