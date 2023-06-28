# Compare acoustic data and imaging on MIDA model vs models derived from FASTMRI
using DrWatson

#ENV["DEVITO_LOGGING"] = "ERROR"

using OneShotImaging
using JLD2, JUDI
using Statistics, LinearAlgebra, Random, Printf, SegyIO
using Flux, Zygote
using BSON
using UNet
using CUDA
using ProgressMeter
using Images

import Flux: update!

sim_name = "bg-marine-sup"
save_path = "/slimdata/mlouboutin3/OneShot"
_dict = @strdict 
plot_path = "$(save_path)/plots/$(sim_name)"
save_path = "$(save_path)/data/$(sim_name)"

# Check if can run on gpu
if CUDA.has_cuda()
    device = gpu
    CUDA.allowscalar(false)
    @info "Training on GPU"
else
    device = cpu
    @info "Training on CPU"
end

# Data
compass = "/slimdata/SharedData/Compass/Marine2DLine/"
all_vels = readdir("$(compass)/data")
all_vels = filter(x->startswith(x, "2d"), all_vels)
nslice = length(all_vels)

# Wavelet
@load "/slimdata/SharedData/Compass/Marine2DLine/data/wavelet.jld2"

mis = Vector{Model}(undef, nslice)
dis = Vector{judiVector}(undef, nslice)
qis = Vector{judiVector}(undef, nslice)
dm = Vector{Matrix}(undef, nslice)

wb = nothing

for (i, v) in enumerate(all_vels)
    
    # Data
    if isfile("$(compass)/data/$(v)/$(v)-segyio.jld2")
        @load "$(compass)/data/$(v)/$(v)-segyio.jld2" container
    else
        container = segy_scan("$(compass)/data/$(v)", "shot", ["GroupX", "GroupY", "RecGroupElevation", "SourceSurfaceElevation", "dt"])
        @save "$(compass)/data/$(v)/$(v)-segyio.jld2" container
    end
    ix = sortperm(get_header(container, "SourceX")[:, 1])
    dis[i] = judiVector(container; segy_depth_key = "SourceSurfaceElevation")[ix]
    # source
    src_geometry = Geometry(container; key = "source", dt=dtS, t=TD)
    qis[i] = judiVector(src_geometry, wavelet)[ix]
    # model
    @load "$(compass)/slices/$v.jld2" d slice
    m = (1f-3 .* slice) .^(-1)
    isnothing(wb) && (global wb = find_water_bottom(m)[1])
    # Vary smoothing to have better robustness to background
    k1, k2 = rand(1f0:25f0, 2)
    m[:, wb+1:end] .= imfilter(m[:, wb+1:end],  Kernel.gaussian((k1, k2)))
    m = m.^2 
    dm[i] = m .- (1f-3 .* slice) .^(-2)
    mis[i] = Model(d, (0, 0), m)
end

####################################################################################################
n_sim_src = 63 # about 1 src every ten withing one single shot 15km offset, taken 15 on each side
batch_size = 1 # To be increased
M_build = ones(Float32, batch_size, n_sim_src)
opt = Options(free_surface=true, IC="as", limit_m=true, buffer_size=0f0, subsampling_factor=12, dt_comp=1f0)

# Setup operators
F0 = [judiModeling(mis[i], qis[i].geometry, dis[i].geometry; options=opt) for i=1:nslice]
J = [judiJacobian(F0[i], qis[i]) for i=1:nslice]

#############################################################################################################################################
# Train

# Setup neural networks
net, ps = make_model(J[1], 5, 3; supervised=true, device=device, nsim=n_sim_src)

# Train
n_epochs = 20

# We have 45 slices, keep 20% for testing

n_test = div(nslice, 5)
n_train = nslice - n_test

idx_train = [(s, i) for s in n_test+1:nslice for i=1:dis[s].nsrc]
idx_test =  [(s, i) for s in 1:n_test for i=1:dis[s].nsrc]
n_samples_train = length(idx_train)

plot_every = 200
save_every = 10000
test_every = 500

lr = 1f-6
opt = Flux.ADAM(lr, (0.9, 0.999))
train_loss_h = Vector{Float32}()
test_loss_h = Vector{Float32}()

p = isinteractive() ? Progress(n_epochs*n_samples_train, color=:red) : nothing

estart=1
kstart=1

# Check if already partially traianed
lastsave = sort(readdir(save_path))

if !isempty(lastsave)
    kstart = parse(Int, match(r"k=(.*)_lr", lastsave[end])[1])
    estart = parse(Int, match(r"e=(.*)_k", lastsave[end])[1])
    bson_file = BSON.load("$(save_path)/$(lastsave[end])")
    net = bson_file["netc"] |> device
end


for e in estart:n_epochs
    idxtrain = shuffle(idx_train)
    for (k, (si, idx)) in enumerate(idxtrain)
        if e == estart && k <= kstart
	   continue
	end
        GC.gc(true)
        CUDA.reclaim()
        # Training
        t0 = @elapsed begin
	    dk, dks, mk, dmk, Jkl, Jks = get_shots(idx, mis[si], dm[si], dis[si], J[si]; nsim=n_sim_src)
        end
	println("Setup time: $(t0) sec")
        Base.flush(stdout)
        t1 = @elapsed begin
            # Compute gradient and update parameters
            local loss_log
            grads = Flux.gradient(ps) do
                loss_log = net(dmk, dk, dks, Jkl, Jks)[1]
                return loss_log
            end
            update!(opt, ps, grads)
            push!(train_loss_h, loss_log)
            GC.gc(true)
        end
        mod(k-1, plot_every) == 0 && plot_prediction(net, Jkl, Jks, mk, dmk, dk, dks, k, e, plot_path;lr=1, n_epochs=1, name="train")
        
	# Testing
        if mod(k-1, test_every) == 0
	    st, idxt = rand(idx_test)
            dkt, dkts, mkt, dmkt, Jktl, Jkts = get_shots(idxt, mis[st], dm[st], dis[st], J[st]; nsim=n_sim_src)
            loss_log = net(dmkt, dkt, dkts, Jktl, Jkts)[1]
            GC.gc(true)
	    push!(test_loss_h, loss_log)
            mod(k-1, plot_every) == 0 && plot_prediction(net, Jktl, Jkts, mkt, dmkt, dkt, dkts, k, e, plot_path;lr=1, n_epochs=1, name="test")
        end

        mod(k-1, plot_every) == 0 && plot_losses(train_loss_h, test_loss_h, k, e, plot_path; lr=1, n_epochs=1)
        
	# Save network and print progress
        if mod(k-1, save_every) == 0
            mname = @strdict k e n_epochs lr
            netc = cpu(net)
            safesave(joinpath(save_path, savename(mname; digits=6)*"train.bson"), @strdict netc train_loss_h);
        end
        
	if isinteractive()
            ProgressMeter.next!(p; showvalues=[(:loss_train, train_loss_h[end]), (:epoch, e), (:iter, k), (:t, t1)], valuecolor=:blue)
        else
            @printf("epoch %d/%d, iter %d/%d, loss=%2.4e, t=%1.3f sec \n", e, n_epochs, k, n_samples_train, train_loss_h[end], t1)
        end
    end
end