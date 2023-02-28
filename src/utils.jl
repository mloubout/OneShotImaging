export setup_refs, make_model, get_shots, loss_func
using Statistics 

function make_data(J, dm, idx, shot_path; perm=false)
    nsrc = get_nsrc(J.rInterpolation.geometry)
    name = @strdict J dm idx nsrc
    sname = joinpath(shot_path, savename(name; digits=6)*"data.jld2")
    if isfile(sname)
        dobs = load(sname)["dobs"]
    else
        dobs = J*dm
        safesave(sname, @strdict dobs);
    end
    nt = J.rInterpolation.geometry.nt[1]
    
    nxrec = J.rInterpolation.geometry.nrec[1]
    dobs = reshape(vec(dobs), nt, nxrec, nsrc, 1)
    if perm
        dobs = dobs[:, :, randperm(nsrc), :]
    end
    snorms = mapslices(x->norm(x, Inf), dobs; dims=[1,2,4])
    dobs ./= snorms
    return dobs
end

function get_shots(idx, J, shot_path, dmj, m0;fac=1f0)
    # Observed data
    J.model.m .= m0
    d_obs = make_data(J, dmj, idx, shot_path)
    dmj = reshape(dmj, J.model.n..., 1, 1)
    return fac .* d_obs
end

# Forward pass on neural networks
function loss_unet_sup(h1, h2, Js, dmj, d_obs, m0; misfit=Flux.Losses.mse, device=cpu)
    Zygote.ignore() do
        set_m0(Js, m0)
    end
    dp = h1(device(d_obs))
    qp = cpu(h2(dp))
    dmpred = vec(Js.Jsim'(qp)*cpu(dp))
    dmpred = reshape(Js.M * dmpred, size(dmj))

	loss = misfit(dmpred, dmj; agg=sum)
    return loss, cpu(dp), qp, reshape(dmpred, Js.J.model.n..., 1, 1)
end

# Forward pass on neural networks
function loss_unet_unsup(h1, h2, Js, dmj, d_obs, m0; misfit=Flux.Losses.mse, device=cpu)
    simd = Zygote.ignore() do
        q_sim, qw = make_sim_source(Js.J.q)
        simd = make_super_shot(d_obs, qw;J=Js.J)
        set_m0(Js, m0)
        Js.Jsq.q.data .= q_sim
        return simd
    end
    dp = h1(device(d_obs))
    qp = cpu(h2(dp))
    dmpred = vec(Js.Jsim'(qp)*cpu(dp))
    dmpred = reshape(Js.M * dmpred, size(dmj))

    predict = reshape(Js.Jsq*dmpred, size(simd))

	loss = misfit(predict, simd; agg=sum)
    return loss, cpu(dp), qp, reshape(dmpred, Js.J.model.n..., 1, 1)
end

loss_func(sup::Bool) = sup ? loss_unet_sup : loss_unet_unsup

function make_model(J, depth1, depth2;M=nothing, supervised=true, device=cpu, precon)
    #h1 = Chain(x->sum(x; dims=3), Unet(1, 1, depth1)) |> device
    nsrc = get_nsrc(J.rInterpolation.geometry)
    h1 = Unet(nsrc, 1, depth1) |> device
 
    h2 = Unet(1, 1, depth2) |> device
    ps = Flux.params(h1, h2)
    Js = make_Js(J;M=M, precon=precon)
    net(dm, dobs, m0) = loss_func(supervised)(h1, h2, Js, dm, dobs, m0; device=device)
    return net, ps, h1, h2
end
