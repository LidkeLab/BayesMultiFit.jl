using Documenter, BAMF 

makedocs(sitename="My Documentation",
modules=[BAMF]
)

deploydocs(;
    repo="github.com/LidkeLab/BAMF.jl.git"
)