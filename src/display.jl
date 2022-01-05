## Analysis and Display

function circleShape(h, k, r)
    θ = LinRange(0, 2 * π, 500)
    h .+ r * sin.(θ), k .+ r * cos.(θ)
end




function histogram2D(states::Vector{Any}, sz::Int, zoom::Int)
    x,y=getxy(states)
    xbins = range(1, sz;step=1 / zoom)
    ybins = range(1, sz;step=1 / zoom)
    histplot = histogram2d(x, y, nbins=xbins, ybins, aspect_ratio=:equal, show_empty_bins=true) 
    return histplot
end

function histogram2D(states::Vector{Any}, sz::Int, zoom::Int, d::ArrayDD, truestate::StateFlatBg)
    x,y=getxy(states)
    xbins = range(1, sz;step=1 / zoom)
    ybins = range(1, sz;step=1 / zoom)
    fig = heatmap(d.data, color=:greys)
    histogram2d!(fig, x, y, nbins=xbins, ybins, aspect_ratio=:equal, show_empty_bins=false) 
    dcircle = 0.5
    for nn = 1:truestate.n
        plot!(fig, circleShape(truestate.x[nn], truestate.y[nn], dcircle), color=:blue)  
    end
    return fig
end

function histogram2D(states::Vector{Any}, sz::Int, zoom::Int, truestate::StateFlatBg)
    x,y=getxy(states)
    xbins = range(1, sz;step=1 / zoom)
    ybins = range(1, sz;step=1 / zoom)
    fig=histogram2d(x, y, nbins=xbins, ybins, aspect_ratio=:equal, show_empty_bins=false) 
    dcircle = 0.5
    for nn = 1:truestate.n
        plot!(fig, circleShape(truestate.x[nn], truestate.y[nn], dcircle), color=:blue)  
    end
    histogram2d!(fig, x, y, nbins=xbins, ybins, aspect_ratio=:equal, show_empty_bins=false,yflip=true) 
    # scatter!(fig, x, y) 
    return fig
end

#"Show model and data vs chain step"
function showoverlay(states::Vector{Any},data::BAMFData,psf::MicroscopePSFs.PSF)
    len=length(states)
    tmp=deepcopy(data)
    sz=data.sz
    d=Array{Float32}(undef,(sz,sz*size(tmp.data,3),len))
    m=Array{Float32}(undef,(sz,sz*size(tmp.data,3),len))
    println(size(d))
    dataim=reshape(data.data,(sz,sz*size(data.data,3)))

    println(states[1])
    for nn=1:len
        d[:,:,nn]=dataim     
        genmodel!(states[nn],psf,tmp)
        m[:,:,nn]=reshape(tmp.data,(sz,sz*size(data.data,3)))
    end

    globmax=maximum((maximum(m),maximum(d)))
    d=d./globmax
    m=m./globmax
    out=cat(dims=1,d,m,d-m)
    println(size(out))
    return out
end

function plotstate(truestate::StateFlatBg,foundstate::StateFlatBg)
    fig=plot()
    dcircle = 0.5
    for nn = 1:truestate.n
        plot!(fig, circleShape(truestate.x[nn], truestate.y[nn], dcircle), color=:blue,yflip=true)  
    end

    for nn = 1:foundstate.n
        plot!(fig, circleShape(foundstate.x[nn], foundstate.y[nn], dcircle), color=:red,yflip=true)  
    end
    
    return fig
end

function plotstate(truestate::StateFlatBg,foundstate::StateFlatBg_Results; fig::Plots.Plot{Plots.GRBackend}=plot())
    #fig=plot()
    dcircle = 0.5
    for nn = 1:truestate.n
        plot!(fig, circleShape(truestate.x[nn], truestate.y[nn], dcircle), color=:blue,yflip=true)  
    end

    for nn = 1:foundstate.n
        σ=(foundstate.σ_x[nn]+foundstate.σ_x[nn])/2f0
        plot!(fig, circleShape(foundstate.x[nn], foundstate.y[nn], σ), color=:red,yflip=true)  
    end
    
    return fig
end



