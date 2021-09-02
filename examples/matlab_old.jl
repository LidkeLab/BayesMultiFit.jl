## Testing the matlab interface
out=BAMF.matlab_DD_FlatBG(data.data,"gauss",1.3f0,θ_start,θ_step,len,mypdf,Int32(burnin),Int32(iterations),xystd,istd,split_std,bndpixels)

# mex interface

args=[MATLAB.mxarray(data.data),
MATLAB.mxarray("gauss"),MATLAB.mxarray(1.3f0),MATLAB.mxarray(θ_start),MATLAB.mxarray(θ_step),
MATLAB.mxarray(len),MATLAB.mxarray(mypdf),MATLAB.mxarray(Int32(burnin)),MATLAB.mxarray(Int32(iterations)),
MATLAB.mxarray(xystd),MATLAB.mxarray(istd),MATLAB.mxarray(split_std),MATLAB.mxarray(bndpixels)
]

BAMF.mextypes(args)
BAMF.mextest(args)

mapn=BAMF.matlab_DD_FlatBG_mex(args)

@time mapn=BAMF.matlab_DD_FlatBG_mex_lite(args);

