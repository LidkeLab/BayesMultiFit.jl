# push!(LOAD_PATH,"../src/")

using Documenter, BayesMultiFit

makedocs(sitename="BayesMultiFit.jl Documentation",
modules=[BayesMultiFit]
)

deploydocs(;
    repo="github.com/LidkeLab/BayesMultiFit.jl.git"
)