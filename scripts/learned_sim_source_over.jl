# Compare acoustic data and imaging on MIDA model vs models derived from FASTMRI
using DrWatson
@quickactivate :OneShotImaging
import Pkg; Pkg.instantiate()

ENV["DEVITO_LOGGING"] = "ERROR"

using JLD2, JUDI
using Statistics, LinearAlgebra, Random, Printf
using Flux, Zygote
using UNet
using CUDA
using ProgressMeter

import Flux: update!

sim_name = "over-subset-7src"
_dict = @strdict 
plot_path = "/localdata/mlouboutin3/learned-sim-source/plots/$(sim_name)"
save_path = "/localdata/mlouboutin3/learned-sim-source/data/$(sim_name)"
data_path = "/localdata/mlouboutin3/learned-sim-source/data"
shot_path = "/localdata/mlouboutin3/learned-sim-source/data/over-shots"


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
Slices = load("$(data_path)/overthrust_images_train.jld")
m = Slices["m"][3]
m0 = Slices["m0"][3]
dm = vec(Slices["dm"][3])
ntrain = length(m)

# Set up model structure
n = size(m0)
d = (25., 25.)
o = (0., 0.)

# Setup info and model structure
nsrc = 21
sub_fact = 3
n_sim_src = div(nsrc, sub_fact)
model0 = Model(n, d, o, m0)

# Set up receiver geometry
nxrec = 256
xrec = range(400f0, stop=9600f0, length=nxrec)
dxrec = xrec[2] - xrec[1]
yrec = 0f0
zrec = range(250f0, stop=250f0, length=nxrec)

# receiver sampling and recording time
time = 2040f0   # receiver recording time [ms]
dt = 8f0    # receiver sampling interval [ms]

# Set up receiver structure
recGeometry = Geometry(xrec, yrec, zrec; dt=dt, t=time, nsrc=nsrc)
recGeometrysim = Geometry(xrec, yrec, zrec; dt=dt, t=time, nsrc=1)

# Set up source geometry (cell array with source locations for each shot)
xsrc = range(500f0, stop=9500f0, length=nsrc)
dxsrc = xsrc[2] - xsrc[1]
ysrc = range(0f0, stop=0f0, length=nsrc)
zsrc = range(20f0, stop=20f0, length=nsrc)

# Set up source structure
f0 = 0.015f0
srcGeometry = Geometry(convertToCell(xsrc), convertToCell(ysrc), convertToCell(zsrc); dt=dt, t=time)
q = judiVector(srcGeometry, ricker_wavelet(time, dt, f0))

####################################################################################################

# Setup options
opt = Options(isic=true, return_array=true)

# Setup operators
F0 = judiModeling(model0, srcGeometry, recGeometry; options=opt)
J = judiJacobian(F0, q)
Jsim = setup_operators(J)

#############################################################################################################################################
# Train

# Setup neural networks
net, ps = make_model(J, 4, 3; supervised=true, device=device)

# Train
n_epochs = 20
n_samples_train = 1600
n_samples_test = 400

plot_every = 100
save_every = 100
test_every = 20

lr = 1f-4
opt = Flux.ADAM(lr, (0.9, 0.999))
train_loss_h = Vector{Float32}()
test_loss_h = Vector{Float32}()

# Split 80/20
inds = randperm(2000)
indices_train = repeat(inds[1:n_samples_train], sub_fact)
indices_test = inds[n_samples_train+1:end]
ntest = length(indices_test)
ntrain = length(indices_train)

iname = @strdict indices_train indices_test
safesave(joinpath(save_path, savename(iname; digits=6)*"train-split-indices.jld2"), @strdict indices_train indices_test);

p = isinteractive() ? Progress(n_epochs*n_samples_train, color=:red) : nothing

for e in 1:n_epochs
    idx_train = indices_train[randperm(ntrain)]
    for (k, idx) in enumerate(idx_train)
        # Training
        dmj = Slices["dm"][idx]
        m0j = Slices["m0"][idx]
        d_obs, Ji = get_shots(idx, J, shot_path, dmj, m0j; nsim=n_sim_src)
        Base.flush(stdout)
        t1 = @elapsed begin
            # Compute gradient and update parameters
            local loss_log
            grads = Flux.gradient(ps) do
                loss_log = net(dmj, d_obs, m0j)[1]
                return loss_log
            end
            update!(opt, ps, grads)
            push!(train_loss_h, loss_log)
            GC.gc(true)
        end
        mod(k-1, plot_every) == 0 && plot_prediction(net, Ji, m0j, dmj, d_obs, k, e, plot_path;lr=1, n_epochs=1, name="train")
        # Testing
        if mod(k-1, test_every) == 0
            idxt = indices_test[rand(1:ntest)]
            dmjt = Slices["dm"][idxt]
            m0t = Slices["m0"][idxt]
            d_obst, Jit = get_shots(idxt, J, shot_path, dmjt, m0t)
            loss_log = net(dmjt, d_obst, m0t)[1]
            push!(test_loss_h, loss_log)

            mod(k-1, plot_every) == 0 && plot_prediction(net, Jit, m0t, dmjt, d_obst, k, e, plot_path;lr=1, n_epochs=1, name="test")
        end

        mod(k-1, plot_every) == 0 && plot_losses(train_loss_h, test_loss_h, k, e, plot_path; lr=1, n_epochs=1)
        # Save network and print progress
        if mod(k-1, save_every) == 0
            mname = @strdict k e n_epochs lr
            safesave(joinpath(save_path, savename(mname; digits=6)*"train.jld2"), @strdict ps train_loss_h);
        end
        if isinteractive()
            ProgressMeter.next!(p; showvalues=[(:loss_train, train_loss_h[end]), (:epoch, e), (:iter, k), (:t, t1)], valuecolor=:blue)
        else
            @printf("epoch %d/%d, iter %d/%d, loss=%2.4e, t=%1.3f sec \n", e, n_epochs, k, ntrain, train_loss_h[end], t1)
        end
    end
end
