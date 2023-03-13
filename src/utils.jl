export setup_refs, make_model, get_shots

function make_data(J, dm, idx, shot_path; perm=false, nsim::Integer=get_nsrc(J.rInterpolation))
    name = @strdict J dm idx
    sname = joinpath(shot_path, savename(name; digits=6)*"data.jld2")
    if isfile(sname)
        dobs = load(sname)["dobs"]
    else
        dobs = J*dm
        safesave(sname, @strdict dobs);
    end
    nt = J.rInterpolation.geometry.nt[1]
    nsrc = get_nsrc(J.rInterpolation.geometry)

    inds = nsim < nsrc ? randperm(nsrc)[1:nsim] : 1:nsrc

    nxrec = J.rInterpolation.geometry.nrec[1]
    dobs = reshape(vec(dobs), nt, nxrec, nsrc, 1)[:, :, inds, :]

    snorms = mapslices(x->norm(x, Inf), dobs; dims=[1,2,4])
    dobs ./= snorms
    return dobs, J[inds]
end

function get_shots(idx, J, shot_path, dmj, m0; nsim::Integer=get_nsrc(J.rInterpolation))
    # Observed data
    J.model.m .= m0
    d_obs = make_data(J, dmj, idx, shot_path; nsim=nsim)
    dmj = reshape(dmj, J.model.n..., 1, 1)
    return d_obs
end

# Forward pass on neural networks
function loss_unet_sup(h1, h2, Js, dmj, d_obs::Array{T, 4}, m0; misfit=Flux.Losses.mae, device=cpu) where T
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
function loss_unet_unsup(h1, h2, Js, dmj, d_obs::Array{T, 4}, m0; misfit=Flux.Losses.mae, device=cpu) where T
    simd = Zygote.ignore() do
        q_sim, qw = make_sim_source(Js.J.q)
        simd = make_super_shot(d_obs, qw)
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

function make_model(J, depth1, depth2; supervised=true, device=cpu, precon=true, nsim::Integer=get_nsrc(J.rInterpolation))
    h1 = Chain(Conv((1,1), nsim=>nsim, identity; bias=false), Unet(nsim, 1, depth1)) |> device
    h2 = Unet(1, 1, depth2) |> device
    ps = Flux.params(h1, h2)
    Js = make_Js(J; precon=precon)
    net(dm, dobs, m0) = loss_func(supervised)(h1, h2, Js, dm, dobs, m0; device=device)
    return net, ps
end
