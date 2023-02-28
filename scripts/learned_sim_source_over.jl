# Compare acoustic data and imaging on MIDA model vs models derived from FASTMRI
#nohup julia scripts/learned_sim_source_over.jl &
using DrWatson
@quickactivate :OneShotImaging
import Pkg; Pkg.instantiate()

#using Pkg;Pkg.add(PackageSpec(url="https://github.com/mloubout/UNet.jl.git", rev="remove-bad-layes")) 
ENV["DEVITO_LOGGING"] = "ERROR"

using JLD2, JUDI
using Statistics, LinearAlgebra, Random, Printf
using Flux, Zygote
using UNet
using CUDA
using ProgressMeter
using Images

import Flux: update!

function gaussian(x, magnitude, mean, variance)
    magnitude .* exp.(.-(x .- mean).^2f0 ./ (2f0 .* variance));
end

function tone_burst(sample_freq, signal_freq, num_cycles; signal_length=nothing)
    # calculate the temporal spacing
    dt = 1f0 / sample_freq;

    tone_length = num_cycles / (signal_freq);
    tone_t = 0f0:dt:tone_length;
    tone_burst = sin.(2f0 * pi * signal_freq * tone_t);

    #apply the envelope
    x_lim = 3f0;
    window_x = (-x_lim):( 2f0 * x_lim / (length(tone_burst) - 1f0) ):x_lim;
    window = gaussian(window_x, 1f0, 0f0, 1f0);
    tone_burst = tone_burst .* window;

    if ~isnothing(signal_length)
        signal_full = zeros(Float32, signal_length)
    else 
        signal_full = zeros(Float32, length(tone_burst))
    end

    signal_full[1:length(tone_burst)] = tone_burst
    signal_full / maximum(signal_full)
end

function circle_geom(h, k, r, numpoints)
       #h and k are the center coordes
       #r is the radius
       theta = LinRange(0f0, 2*pi, numpoints+1)[1:end-1]
       Float32.(h .- r*cos.(theta)), Float32.(k .+ r*sin.(theta)), theta
end

sim_name = "train-clean"
plot_path = plotsdir(sim_name)
save_path = datadir()
shot_path = datadir()

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
mkpath(datadir("training-data"))
#data_path = "../MultiSourceSummary.jl/data/training_data/fastmri_training_data_1-2750_512_mo_rho0.jld2"
# data_path = datadir("training-data","brain_mo_dimp_256.jld2")
# if isfile(data_path) == false
#     println("Downloading data...");
#     download("https://www.dropbox.com/s/4ydd0etzk7xxbu8/brain_mo_dimp_256.jld2?dl=0", data_path)
# end

# Slices = load(data_path)
# nx = 256 #getting nans when doing original size of 512

mkpath(datadir("training-data"))
#data_path = "../MultiSourceSummary.jl/data/training_data/fastmri_training_data_1-2750_512_mo_rho0.jld2"
data_path = datadir("training-data","fastmri_training_data_49-49_512_m_rho_mo_rho0.jld2")

Slices = load(data_path)
nx = 256 #getting nans when doing original size of 512

m   = Slices["m_train"][:,:,1,1]'
m0   = Slices["m0_train"][:,:,1,1]'
rho0   = Slices["rho0_train"][:,:,1,1]'
rho   = Slices["rho_train"][:,:,1,1]'


using Images 
m = imresize(m,(nx,nx))
m0 = imresize(m0,(nx,nx))
rho0 = imresize(rho0,(nx,nx))
rho = imresize(rho,(nx,nx))
dm = m0-m

# Set up model structure
n = size(m0)
dx=1
#d = (0.5,0.5) # for 512 resolution
d = (dx,dx)
o = (0., 0.)

# Setup info and model structure
nsrc = 4
model0 = Model(n, d, o, m0)

# Setup circumference of receivers 
nxrec = 256
domain_x = (n[1] - 1)*d[1]
domain_z = (n[2] - 1)*d[2]
rad = .95*domain_x / 2
xrec, zrec, theta = circle_geom(domain_x / 2, domain_z / 2, rad, nxrec)
yrec = 0f0 #2d so always 0


# Set up source geometry (cell array with source locations for each shot)
step_num = Int(nxrec/nsrc)
xsrc  = xrec[1:step_num:end]
ysrc  = range(0f0, stop=0f0, length=nsrc)
zsrc  = zrec[1:step_num:end]

# receiver sampling and recording time
f0 = .4f0    # Central frequency in MHz
time = 204.6f0   # receiver recording time [ms]
dt = .4f0    # receiver sampling interval [ms] 
#nt NEEDS TO BE POWER OF TWO

sampling_rate = 1f0 / dt
cycles_wavelet = 3  # number of sine cycles
nt = Int(div(time,dt)) + 1 #NEEDS TO BE POWER OF TWO
wavelet = tone_burst(sampling_rate, f0, cycles_wavelet; signal_length=nt);
wavelet = reshape(wavelet, length(wavelet), 1)

# Set up source structure
srcGeometry = Geometry(convertToCell(xsrc), convertToCell(ysrc), convertToCell(zsrc); dt=dt, t=time)
q = judiVector(srcGeometry, wavelet)

# Set up receiver structure
recGeometry = Geometry(xrec, yrec, zrec; dt=dt, t=time, nsrc=nsrc)
####################################################################################################

# Setup options
opt = Options(isic=false, return_array=true)

# Setup operators
F0 = judiModeling(model0, srcGeometry, recGeometry; options=opt)
J = judiJacobian(F0, q)

#############################################################################################################################################
# Train

# Setup neural networks
using JOLI 
# radius = 110;
# mask = [(i - model0.n[1] / 2)^2 + (j - model0.n[2] / 2)^2 - radius^2 for i = 1:model0.n[1], j = 1:model0.n[2]];
# mask[mask .> 0] .= 0f0;
# mask[mask .< 0] .= 1f0;

mask = zeros(Float32,size(dm))
mask[findall(x -> x != 0,dm)] .= 1f0

#Mr = joEye(prod(model0.n);DDT=Float32,RDT=Float32)
Mr = joDiag(vec(mask );  RDT=Float32, DDT=Float32)  
supervised=true
net, ps = make_model(J, 4, 3; supervised=supervised, device=device, M=Mr, precon=true);

# Train
n_epochs = 6000
n_samples_train = 2500
n_samples_test = 250

plot_every = 200
save_every = 100
test_every = 25

fac = 100f0
lr = 8f-5
lr_clip = 10f0
opt = Flux.Optimiser(ClipNorm(lr_clip),Flux.ADAM(lr, (0.9, 0.999)))
train_loss_h = Vector{Float32}()
test_loss_h = Vector{Float32}()

# Split 80/20
Random.seed!(123)
inds = randperm(n_samples_train+n_samples_test)
indices_train = inds[1:n_samples_train]
indices_test = inds[n_samples_train+1:end]

p = isinteractive() ? Progress(n_epochs*n_samples_train, color=:red) : nothing



idx_plot = 2
d_obs = get_shots(idx_plot, J, shot_path, dm, m0;fac=fac)
rtms, rtm = make_rtms(J, d_obs, m0;precon=false)
rtm = reshape(Mr * vec(rtm), size(rtm))
to_train = rtm



for e in 1:n_epochs
    idx_train = [1]#
    #idx_train = indices_train[randperm(length(indices_train))]
    if mod(e-1, plot_every) == 0
        dmj = to_train#dimp_train[:,:,idx_plot]
        m0j = m0#m0_train[:,:,idx_plot]
        d_obs = get_shots(idx_plot, J, shot_path, dmj, m0j;fac=fac)
        fig_name = @strdict e n_epochs lr fac supervised nt dx lr_clip
        plot_prediction(net, J, m0j, dmj, d_obs, 1, e, plot_path,fig_name;lr=lr, n_epochs=n_epochs, name="train")
        plot_losses(train_loss_h, train_loss_h, 1, e, plot_path; lr=lr, n_epochs=n_epochs)
     end

    for (k, idx) in enumerate(idx_train)
        println("training on ind $(idx)")

        # Training
        dmj = to_train#dimp_train[:,:,idx]
        m0j = m0#m0_train[:,:,idx]
        d_obs = get_shots(idx, J, shot_path, dmj, m0j;fac=fac)
   
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
        # if mod(k-1, plot_every) == 0
        #     idx_plot_train = indices_train[3]
        #     dmj = dimp_train[:,:,idx_plot_train]
        #     m0j = m0_train[:,:,idx_plot_train]
        #     d_obs = get_shots(idx_plot_train, J, shot_path, dmj, m0j;fac=fac)
        #     plot_prediction(net, J, m0j, dmj, d_obs, k, e, plot_path;lr=lr, n_epochs=n_epochs, name="train")
        #     plot_losses(train_loss_h, test_loss_h, k, e, plot_path; lr=lr, n_epochs=n_epochs)
        # end
        # #Testing
        # if mod(k-1, test_every) == 0
        #     idxt = indices_test[1]
        #     dmjt = dimp_train[:,:,idxt]
        #     m0t = m0_train[:,:,idxt]
        #     d_obst = get_shots(idxt, J, shot_path, dmjt, m0t;fac=fac)
        #     loss_log = net(dmjt, d_obst, m0t)[1]
        #     push!(test_loss_h, loss_log)

        #     mod(k-1, plot_every) == 0 && plot_prediction(net, J, m0t, dmjt, d_obst, k, e, plot_path;lr=lr, n_epochs=n_epochs, name="test")
        # end

        # #mod(k-1, plot_every) == 0 && plot_losses(train_loss_h, train_loss_h, k, e, plot_path; lr=lr, n_epochs=n_epochs)
        # #mod(k-1, plot_every) == 0 && plot_losses(train_loss_h, test_loss_h, k, e, plot_path; lr=lr, n_epochs=n_epochs)
        
        # #Save network and print progress
        # if mod(k-1, save_every) == 0
        #     mname = @strdict k e n_epochs lr
        #     safesave(joinpath(save_path, savename(mname; digits=6)*"train.jld2"), @strdict ps train_loss_h);
        # end
        if isinteractive()
            ProgressMeter.next!(p; showvalues=[(:loss_train, train_loss_h[end]), (:epoch, e), (:iter, k), (:t, t1)], valuecolor=:blue)
        else
            @printf("epoch %d/%d, iter %d/%d, loss=%2.4e, t=%1.3f sec \n", e, n_epochs, k, n_samples_train, train_loss_h[end], t1)
        end
    end
end
