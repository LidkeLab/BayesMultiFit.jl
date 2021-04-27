## Analysis and Display

function circleShape(h, k, r)
    θ = LinRange(0, 2 * π, 500)
    h .+ r * sin.(θ), k .+ r * cos.(θ)
end


function getxy(states::Vector{Any})
    # count number of emitters 
    nemitters = 0;
    for nn = 1:length(states) 
        nemitters += states[nn].n
    end
    
    x = Vector{Float32}(undef, nemitters)
    y = Vector{Float32}(undef, nemitters)
 
    cnt = 0;
    for ss = 1:length(states) 
        for nn = 1:states[ss].n
            cnt += 1  
            x[cnt] = states[ss].x[nn]
            y[cnt] = states[ss].y[nn]
        end
    end   

    return x,y 

end

function histogram2D(states::Vector{Any}, sz::Int32, zoom::Int32)
    x,y=getxy(states)
    xbins = range(1, sz;step=1 / zoom)
    ybins = range(1, sz;step=1 / zoom)
    histplot = histogram2d(x, y, nbins=xbins, ybins, aspect_ratio=:equal, show_empty_bins=true) 
    return histplot
end

function histogram2D(states::Vector{Any}, sz::Int32, zoom::Int32, d::ArrayDD, truestate::StateFlatBg)
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

function histogram2D(states::Vector{Any}, sz::Int32, zoom::Int32, truestate::StateFlatBg)
    x,y=getxy(states)
    xbins = range(1, sz;step=1 / zoom)
    ybins = range(1, sz;step=1 / zoom)
    fig=histogram2d(x, y, nbins=xbins, ybins, aspect_ratio=:equal, show_empty_bins=false) 
    dcircle = 0.5
    for nn = 1:truestate.n
        plot!(fig, circleShape(truestate.x[nn], truestate.y[nn], dcircle), color=:blue)  
    end
    # histogram2d!(fig, x, y, nbins=xbins, ybins, aspect_ratio=:equal, show_empty_bins=false) 
    scatter!(fig, x, y) 
    return fig
end

