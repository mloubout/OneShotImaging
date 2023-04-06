module OneShotImaging

    using LinearAlgebra, Random
    using PyPlot, SlimPlotting, Zygote, JUDI, DrWatson, UNet, Flux

    import UNet: padz_unet

    include("training.jl")
    include("data_utils.jl")
    include("judi_utils.jl")
    include("plotting.jl")

    # extras for Unet to backpropagate a judiVector    
    padz_unet(x::judiVector, P) = padz_unet(reshape(x.data[1], size(x.data[1])..., 1, 1), P)
end
