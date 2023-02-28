# Compare acoustic data and imaging on MIDA model vs models derived from FASTMRI
#nohup julia scripts/learned_sim_source_update.jl &
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
using BSON

import Flux: update!

using PyPlot 
using SlimPlotting
function create_multi_color_map(intervals::Vector{Tuple{T, T}}, cmap_types::Vector{ColorMap}; out_of_interval_color=[0.,0.,0.,1.]) where T

    @assert length(intervals) == length(cmap_types) "number of intervals and number of colormaps do not match"
    
    # ensure at least 100 points in each interval
    colorrange = intervals[1][1]:minimum([intervals[i][2]-intervals[i][1] for i = 1:length(intervals)])/1f2:intervals[end][2]
             
    # number of points in each colormap
    colorlength = [Int.(round.((intervals[i][2]-intervals[i][1])./(colorrange[2]-colorrange[1]))) for i = 1:length(intervals)]
    # create each colormap
    cmap_sections = [cmap_types[i](range(0f0, stop=1f0, length=colorlength[i])) for i = 1:length(cmap_types)]

    # fill in out-of-interval by out_of_interval_color
    between_interval_sections = [reshape(repeat(out_of_interval_color, inner=Int(round((intervals[i][1]-intervals[i-1][2])/(colorrange[2]-colorrange[1])))), :, 4) for i = 2:length(intervals)]
    
    # concatenate all the sections to get the entire colormap
    total_sections = vcat(cmap_sections[1], vcat([vcat(between_interval_sections[i], cmap_sections[i+1]) for i = 1:length(cmap_sections)-1]...))

    # construct the colormaps
    cmaps = matplotlib.colors.ListedColormap(total_sections)
    
    return cmaps
end

min_vel_soft  = 1.48
max_vel_soft  = 1.5782

#use colorcet bone or gray cet_CET_D1 cet_CET_D2
#USE RAINBOW4 FOR UQ ERROR PLOTS 

cmap_type_1 = ColorMap("cet_CET_L2")
cmap_types = [cmap_type_1,matplotlib.cm.Reds]
intervals = [(min_vel_soft,max_vel_soft),(max_vel_soft,2.8)] #in slowness
cmap = create_multi_color_map(intervals, cmap_types)
vmin_brain = intervals[1][1]
vmax_brain = intervals[end][2]

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

sim_name = "train-update"
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
dx=0.5
#d = (0.5,0.5) # for 512 resolution
d = (dx,dx)
o = (0., 0.)

# Setup info and model structure
nsrc = 4
model0 = Model(n, d, o, m0)
model = Model(n, d, o, m)

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
time = 204.6f0/2f0   # receiver recording time [ms]
dt = .2f0    # receiver sampling interval [ms] 
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
M = joDiag(vec(mask );  RDT=Float32, DDT=Float32)  
supervised=true
misfit = "sum"
Random.seed!(123)
depth_2 = 3
net, ps, h1, h2 = make_model(J, 4, depth_2; supervised=supervised, device=device, M=M, precon=true);

# Train
n_epochs = 4000
n_samples_train = 2500
n_samples_test = 250


test_every = 25

#fac = 1000f0
fac = 5f5

fac_rtm = 1f0
lr = 1f-4
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


# idx_plot = 2
# d_obs = get_shots(idx_plot, J, shot_path, dm, m0;fac=fac)
# rtms, rtm = make_rtms(J, d_obs, m0;precon=false)
# rtm = reshape(Mr * vec(rtm), size(rtm))
# to_train = rtm

Pr = judiProjection(recGeometry)
A_inv = judiModeling(model; )
Ps = judiProjection(q.geometry)

# non-linear data
d_obs = Pr*A_inv*Ps'*q #F'*q
d_back = F0(m0)*q
d_res =  (d_back - vec(d_obs))

#rtm = reshape(J'd_res,size(m0))
#rtm = reshape(M * vec(rtm), size(rtm))
to_train = dm#fac_rtm .* rtm

d_res = reshape(vec(d_res), nt, nxrec, nsrc, 1)
snorms = mapslices(x->norm(x, Inf), d_res; dims=[1,2,4])
#d_res ./= snorms
d_res = fac .* d_res #|> device

#norm(100000 .* d_res)
#julia> norm(d_res)
#22414.004f0


num_iters = 3
m0_all = zeros(Float32,size(m)...,num_iters)
d_res_all = zeros(Float32,nt, nxrec, nsrc,num_iters)

m0_all[:,:,1] = m0
d_res_all[:,:,:,1] = d_res

for i in 2:num_iters
    m0_prev =  m0_all[:,:,i-1];
    d_res_prev = d_res_all[:,:,:,i-1:i-1]; 
    dm_pred   = net(dm, d_res_prev, m0_prev)[4];

    m0_curr = m0_prev - dm_pred
    d_res_curr = (F0(m0_curr)*q - vec(d_obs));
    d_res_curr = reshape(vec(d_res_curr), nt, nxrec, nsrc, 1);
    snorms = mapslices(x->norm(x, Inf), d_res_curr; dims=[1,2,4]);
    #d_res_curr ./= snorms;
    d_res_curr = fac .* d_res_curr; #|> device
    d_res_all[:,:,:,i] = d_res_curr
    m0_all[:,:,i] = m0_curr
end

#update_every = 10
#plot_every = 20

update_every = 100
plot_every = 100
save_every = n_epochs

dxrec = dxsrc = abs(diff(J.rInterpolation.geometry.xloc[1])[1])
dtrec = J.rInterpolation.geometry.dt[1]
# for i in 1:num_iters
#     fig = figure(figsize=(8,8))
#     subplot(1,3,1)
#     plot_sdata(d_res_all[:,:,1,i:i], (dtrec, dxrec); cmap="seismic", name=L"observed non-linear", new_fig=false, cbar=true)

#     tight_layout()
#     fig_name = @strdict  i
#     safesave(joinpath(plot_path,savename(fig_name; digits=6)*"_data.png"), fig); close(fig)
# end

for e in 1:n_epochs
    idx_train = [1,2,3]#
    #idx_train = indices_train[randperm(length(indices_train))]
    if mod(e, plot_every) == 0
   
        plot_losses(train_loss_h ./ train_loss_h[1], train_loss_h ./ train_loss_h[1], 1, e, plot_path; lr=lr, n_epochs=n_epochs)
        
        mses = []
        m0_plot = zeros(Float32,size(m)...,num_iters+1)
        m0_plot[:,:,1] = m0

        for i in 2:(num_iters+1)
            m0_curr =  m0_plot[:,:,i-1];
            d_res_curr = fac .* (F0(m0_curr)*q - vec(d_obs));
            d_res_curr = reshape(vec(d_res_curr), nt, nxrec, nsrc, 1);
            _,dp,qp,dm_pred = net(dm,  d_res_curr, m0_curr);

            m0_plot[:,:,i] = m0_curr - dm_pred
            append!(mses,mean((m0_plot[:,:,i]-m).^2))

            fig = figure(figsize=(8,8))
            subplot(1,3,1)
            plot_sdata(d_res_curr[:,:,1,1], (dtrec, dxrec); cmap="seismic", name=L"residual d_o", new_fig=false, cbar=true)
            subplot(1,3,2)
            plot_sdata(dp[:, :, 1, 1], (dtrec, dxrec); cmap="seismic", name=L"d_p = h_1(d_o)", new_fig=false, cbar=true)
            subplot(1,3,3)
            plot_sdata(qp[:, :, 1, 1], (dtrec, dxsrc); cmap="seismic", name=L"q_p = h_2(h_1(d_o))", new_fig=false, cbar=true)
            tight_layout()
            fig_name = @strdict e n_epochs lr fac fac_rtm supervised nt dx lr_clip misfit depth_2 dx num_iters update_every i
            safesave(joinpath(plot_path,savename(fig_name; digits=6)*"_data.png"), fig); close(fig)
        end

        fig = figure(figsize=(16, 7))
        subplot(1,5,1); title("m0")
        imshow(sqrt.(1f0 ./ m0_plot[:,:,1]'),vmin=vmin_brain,vmax=vmax_brain,cmap=cmap);colorbar(fraction=0.046, pad=0.04)
        subplot(1,5,2); title("m1 MSE=$(mses[1])")
        imshow(sqrt.(1f0 ./ m0_plot[:,:,2]'),vmin=vmin_brain,vmax=vmax_brain,cmap=cmap); colorbar(fraction=0.046, pad=0.04)
        subplot(1,5,3); title("m2 MSE=$(mses[2])")
        imshow(sqrt.(1f0 ./ m0_plot[:,:,3]'),vmin=vmin_brain,vmax=vmax_brain,cmap=cmap); colorbar(fraction=0.046, pad=0.04)
        subplot(1,5,4); title("m3 MSE=$(mses[3])")
        imshow(sqrt.(1f0 ./ m0_plot[:,:,4]'),vmin=vmin_brain,vmax=vmax_brain,cmap=cmap); colorbar(fraction=0.046, pad=0.04)
        subplot(1,5,5); title("ground truth m")
        imshow(sqrt.(1f0 ./ m'),vmin=vmin_brain,vmax=vmax_brain,cmap=cmap);

        tight_layout()
         fig_name = @strdict e n_epochs lr fac fac_rtm supervised nt dx lr_clip misfit depth_2 dx num_iters update_every 
        safesave(joinpath(plot_path,savename(fig_name; digits=6)*"_update.png"), fig); close(fig)


        # for i in 1:num_iters
        #     m0_prev = m0_all[:,:,i]
        #     _,dp,qp,dm_pred = net(dm,  d_res_all[:,:,:,i:i], m0_prev);

        #     m0_plot[:,:,i] = m0_prev - dm_pred
        #     append!(mses,mean((m0_plot[:,:,i]-m).^2))

        #     fig = figure(figsize=(8,8))
        #     subplot(1,3,1)
        #     plot_sdata(d_res_all[:,:,1,i:i], (dtrec, dxrec); cmap="seismic", name=L"residual d_o", new_fig=false, cbar=true)
        #     subplot(1,3,2)
        #     plot_sdata(dp[:, :, 1, 1], (dtrec, dxrec); cmap="seismic", name=L"d_p = h_1(d_o)", new_fig=false, cbar=true)
        #     subplot(1,3,3)
        #     plot_sdata(qp[:, :, 1, 1], (dtrec, dxsrc); cmap="seismic", name=L"q_p = h_2(h_1(d_o))", new_fig=false, cbar=true)
        #     tight_layout()
        #     fig_name = @strdict e n_epochs lr fac fac_rtm supervised nt dx lr_clip misfit depth_2 dx num_iters update_every i
        #     safesave(joinpath(plot_path,savename(fig_name; digits=6)*"_data.png"), fig); close(fig)
        # end
 
        # fig = figure(figsize=(16, 7))
        # subplot(1,5,1); title("m0")
        # imshow(sqrt.(1f0 ./ m0'),vmin=vmin_brain,vmax=vmax_brain,cmap=cmap);colorbar(fraction=0.046, pad=0.04)
        # subplot(1,5,2); title("m1 MSE=$(mses[1])")
        # imshow(sqrt.(1f0 ./ m0_plot[:,:,1]'),vmin=vmin_brain,vmax=vmax_brain,cmap=cmap); colorbar(fraction=0.046, pad=0.04)
        # subplot(1,5,3); title("m2 MSE=$(mses[2])")
        # imshow(sqrt.(1f0 ./ m0_plot[:,:,2]'),vmin=vmin_brain,vmax=vmax_brain,cmap=cmap); colorbar(fraction=0.046, pad=0.04)
        # subplot(1,5,4); title("m3 MSE=$(mses[3])")
        # imshow(sqrt.(1f0 ./ m0_plot[:,:,3]'),vmin=vmin_brain,vmax=vmax_brain,cmap=cmap); colorbar(fraction=0.046, pad=0.04)
        # subplot(1,5,5); title("ground truth m")
        # imshow(sqrt.(1f0 ./ m'),vmin=vmin_brain,vmax=vmax_brain,cmap=cmap);

        # tight_layout()
        #  fig_name = @strdict e n_epochs lr fac fac_rtm supervised nt dx lr_clip misfit depth_2 dx num_iters update_every 
        # safesave(joinpath(plot_path,savename(fig_name; digits=6)*"_update.png"), fig); close(fig)


        

     end

    if mod(e, update_every) == 0

        #m0_prev = m0_all[:,:,1];
        #d_res_prev = d_res_all[:,:,:,1:1] 
        #m0_all[:,:,2] = m0_all[:,:,1] - dm_pred

# fig = figure(figsize=(8,8))
# subplot(2,1,1)
# imshow(m0')
# subplot(2,1,2)
# imshow(m0_curr[:,:,1,1]')
# fig_name = @strdict e n_epochs lr fac fac_rtm supervised nt dx lr_clip misfit depth_2 dx num_iters update_every i
# safesave(joinpath(plot_path,savename(fig_name; digits=6)*"_data.png"), fig); close(fig)


        for i in 2:num_iters
            m0_prev =  m0_all[:,:,i-1];
            d_res_prev = d_res_all[:,:,:,i-1:i-1]; 
            dm_pred   = net(dm, d_res_prev, m0_prev)[4];

            m0_curr = m0_prev - dm_pred
            d_res_curr = (F0(m0_curr)*q - vec(d_obs));
            d_res_curr = reshape(vec(d_res_curr), nt, nxrec, nsrc, 1);
            #snorms = mapslices(x->norm(x, Inf), d_res_curr; dims=[1,2,4]);
            #d_res_curr ./= snorms;
            d_res_curr = fac .* d_res_curr; #|> device
            d_res_all[:,:,:,i] = d_res_curr
            m0_all[:,:,i] = m0_curr
        end
    end
    if mod(e, save_every) == 0
         h1_cpu = cpu(h1);
         h2_cpu = cpu(h2);
         save_dict = @strdict update_every fac h1_cpu h2_cpu train_loss_h  e n_epochs lr nsrc;
         safesave(joinpath(datadir(), savename(save_dict, "bson"; digits=6)),save_dict;);

    end

    for (k, idx) in enumerate(idx_train)
        println("training on ind $(idx)")

        # Training
        m0j = m0_all[:,:,idx]
        dmj = m0j-m
        d_resj = d_res_all[:,:,:,idx:idx] 
   
        Base.flush(stdout)
        t1 = @elapsed begin
            # Compute gradient and update parameters
            local loss_log
            grads = Flux.gradient(ps) do
                loss_log = net(dmj, d_resj, m0j)[1]
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
        #     d_resj = get_shots(idx_plot_train, J, shot_path, dmj, m0j;fac=fac)
        #     plot_prediction(net, J, m0j, dmj, d_resj, k, e, plot_path;lr=lr, n_epochs=n_epochs, name="train")
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
        
       
        if isinteractive()
            ProgressMeter.next!(p; showvalues=[(:loss_train, train_loss_h[end]), (:epoch, e), (:iter, k), (:t, t1)], valuecolor=:blue)
        else
            @printf("epoch %d/%d, iter %d/%d, loss=%2.4e, t=%1.3f sec \n", e, n_epochs, k, n_samples_train, train_loss_h[end], t1)
        end
    end
end
