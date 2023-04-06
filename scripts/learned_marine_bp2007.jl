# Compare acoustic data and imaging on MIDA model vs models derived from FASTMRI
using DrWatson
@quickactivate :OneShotImaging

ENV["DEVITO_LOGGING"] = "ERROR"

using JLD2, JUDI
using Statistics, LinearAlgebra, Random, Printf, SegyIO
using Flux, Zygote
using UNet
using CUDA
using ProgressMeter

import Flux: update!

sim_name = "bp-marine"
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
segy_path = "/slimdata/SharedData/BP2004"
container = segy_scan(segy_path, "shots", ["GroupX", "GroupY", "RecGroupElevation", "SourceSurfaceElevation", "dt"])
d_obs = judiVector(container; segy_depth_key = "SourceDepth")

# Source, only for ref, not used
src_geometry = Geometry(container; key = "source")
wavelet = ricker_wavelet(src_geometry.t[1], src_geometry.dt[1], 0.020)
q = judiVector(src_geometry, wavelet)

# Set up model structure
all_vels = ["vel_z6.25m_x12.5m_exact.segy", "vel_z6.25m_x12.5m_lw.segy", "vel_z6.25m_x12.5m_nosalt.segy"]
d = (12.5, 12.5)
o = (0., 0.)
mis = Vector{Model}(undef, 3)
# Pad left with frist trace as per the readme 2004_Benchmark_READMES.pdf
padleft(vp) = ones(Float32, 1195)*vp[1:1, :]
for (i, v) in enumerate(all_vels)
    vp = Matrix{Float32}(segy_read("$(segy_path)/$(v)").data)[:, 1:2:end]' ./ 1f3
    vp = cat(padleft(vp), vp, dims=1)
    mis[i] = Model(d, o, vp.^(-2))
end

####################################################################################################
n_sim_src = 30 # about 1 src every ten withing one single shot 15km offset, taken 15 on each side
batch_size = 1 # To be increased
M_build = ones(Float32, batch_size, n_sim_src)
opt = Options(free_surface=true, IC="isic", limit_m=true, buffer_size=250f0, subsampling_factor=15)
# Setup operators
F0 = judiModeling(mis[1], src_geometry, d_obs.geometry; options=opt)
J = judiJacobian(F0, q)

#############################################################################################################################################
# Train

# Setup neural networks
net, ps = make_model(J, 5, 3; supervised=false, device=device, nsim=n_sim_src)

# Train
n_epochs = 50
# We have 3 model and d_obs.nsrc, remove a chunk of data in the midle for testing
n_test = div(d_obs.nsrc, 10)
idx_test = 1:n_test
idx_train = n_test+1:d_obs.nsrc

d_test = d_obs[idx_test]
d_train = d_obs[idx_train]
n_samples_train = d_train.nsrc*3
n_samples_test = d_test.nsrc*3

plot_every = 500
save_every = 500
test_every = 100

lr = 1f-5
opt = Flux.ADAM(lr, (0.9, 0.999))
train_loss_h = Vector{Float32}()
test_loss_h = Vector{Float32}()

p = isinteractive() ? Progress(n_epochs*n_samples_train, color=:red) : nothing

for e in 1:n_epochs
    idxtrain = randperm(n_samples_train)
    for (k, idx) in enumerate(idxtrain)
        # Training
        dk, mk, Jk = get_shots(idx, mis, d_train, J; nsim=n_sim_src)
        Base.flush(stdout)
        t1 = @elapsed begin
            # Compute gradient and update parameters
            local loss_log
            grads = Flux.gradient(ps) do
                loss_log = net(nothing, dk, Jk)[1]
                return loss_log
            end
            update!(opt, ps, grads)
            push!(train_loss_h, loss_log)
            GC.gc(true)
        end
        mod(k-1, plot_every) == 0 && plot_prediction(net, Ji, m0j, dmj, d_obs, k, e, plot_path;lr=1, n_epochs=1, name="train")
        # Testing
        if mod(k-1, test_every) == 0
            dkt, mkt, Jkt = get_shots(idx, mis, d_test, J; nsim=n_sim_src)
            loss_log = net(nothing, dkt, Jkt)[1]
            push!(test_loss_h, loss_log)
            mod(k-1, plot_every) == 0 && plot_prediction(net, Jit, m0t, dmjt, d_obst, k, e, plot_path;lr=1, n_epochs=1, name="test")
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
            @printf("epoch %d/%d, iter %d/%d, loss=%2.4e, t=%1.3f sec \n", e, n_epochs, k, ntrain, train_loss_h[end], t1)
        end
    end
end
