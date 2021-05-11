# push!(LOAD_PATH,"../src/")

using Documenter, BAMF, BAMF.RJMCMC 
import BAMF: RJMCMC

makedocs(sitename="BAMF.jl Documentation",
modules=[BAMF,RJMCMC]

)

deploydocs(;
    repo="github.com/LidkeLab/BAMF.jl.git"
)