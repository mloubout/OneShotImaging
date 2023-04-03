
# Forward pass on neural networks
function loss_sup(h1, h2, J, dmj, d_obs::judiVector{T, AT}; misfit=Flux.Losses.mae, device=cpu) where {T, AT}
    M, Js, d_ch_m = Zygote.ignore() do
        M = judiTopmute(J.model)
        Js = sim_rec_J(J, d_obs)
        d_ch_m = make_unet_input(d_obs, J.model.m)
        return M, Js, d_ch_m
    end

    dp = h1(device(d_ch_m))
    qp = cpu(h2(dp))
    dmpred = vec(Js'(qp)*cpu(dp))
    dmpred = reshape(M * dmpred, size(dmj))

	loss = misfit(dmpred, dmj; agg=sum)
    return loss, cpu(dp), qp, reshape(dmpred, Js.J.model.n..., 1, 1)
end

# Forward pass on neural networks
function loss_unsup(h1, h2, J, ::Nothing, d_obs::judiVector{T, AT}; misfit=Flux.Losses.mae, device=cpu) where {T, AT}
    simd, M, Jsq, Js, d_ch_m = Zygote.ignore() do
        qw = randn(Float32, 1, d_obs.nsrc)
        simd = qw * d_obs
        Jsq = qw * J
        M = judiTopmute(J.model)
        Js = sim_rec_J(J, d_obs)
        d_ch_m = make_unet_input(d_obs, J.model.m)
        return simd, M, Jsq, Js, d_ch_m
    end
    dp = h1(device(d_ch_m))
    qp = cpu(h2(dp))
    dmpred = vec(Js'(qp)*cpu(dp))
    dmpred = reshape(M * dmpred, Js.model.m)

    predict = reshape(Jsq*dmpred, J.model.n)

	loss = misfit(predict, simd; agg=sum)
    return loss, cpu(dp), qp, reshape(dmpred, J.model.n..., 1, 1)
end

loss_func(sup::Bool) = sup ? loss_sup : loss_unsup

function make_model(J, depth1, depth2; supervised=true, device=cpu, precon=true, nsim::Integer=get_nsrc(J.rInterpolation))
    h1 = Unet(nsim+1, 1, depth1) |> device
    h2 = Unet(1, 1, depth2) |> device
    ps = Flux.params(h1, h2)
    net(dm, dobs, J) = loss_func(supervised)(h1, h2, J, dm, dobs; device=device)
    return net, ps
end


function make_unet_input(d_obs::judiVector, m::PhysicalParameter)
    sdata = reshape(cat(d_obs.data..., dims=3), d_obs.geometry.nt[1], d_obs.geometry.nrec[1], d_obs.nsrc, 1)
    @show size(sdata)
    m1 = JUDI.SincInterpolation(m.data, range(0f0, stop=1f0, length=m.n[1]), range(0f0, stop=1f0, length=size(sdata, 1)))
    m1 = JUDI.SincInterpolation(PermutedDimsArray(m1, (2,1)), range(0f0, stop=1f0, length=m.n[2]), range(0f0, stop=1f0, length=size(sdata, 2)))'
    return cat(sdata, reshape(m1, size(m1)..., 1, 1), dims=3)
end