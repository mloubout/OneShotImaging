
# Forward pass on neural networks
function loss_sup(h1, h2, Jl, Js, dmj, ::judiVector, d_obs::AbstractArray{T, 4}; misfit=Flux.Losses.mae, device=cpu) where {T}
    M = Zygote.ignore() do
        M = judiTopmute(Jl.model)
        return M
    end

    dp = h1(device(d_obs))
    qp = cpu(h2(dp))
    dmpred = Jl'(qp)*cpu(dp)
    dmpred = reshape(M * dmpred, size(dmj))
    
    loss = misfit(dmpred, dmj; agg=sum)
    return loss, cpu(dp), qp, reshape(dmpred, Jl.model.n..., 1, 1)
end

# Forward pass on neural networks
function loss_unsup(h1, h2, Jl, Js, dm, simd::judiVector, d_obs::AbstractArray{T, 4}; misfit=Flux.Losses.mae, device=cpu) where T
    M = Zygote.ignore() do
        M = judiTopmute(Jl.model)
        return M
    end
    dp = h1(device(d_obs))
    qp = cpu(h2(dp))
    dmpred = Jl'(qp)*cpu(dp)
    dmpred = reshape(M * dmpred, Js.model.m)

    predict = Js*dmpred

    loss = .5f0*norm(predict - simd)^2
    return loss, cpu(dp), qp, reshape(dmpred, Js.model.n..., 1, 1)
end

loss_func(sup::Bool) = sup ? loss_sup : loss_unsup

function make_model(J, depth1, depth2; supervised=true, device=cpu, precon=true, nsim::Integer=get_nsrc(J.rInterpolation))
    h1 = Unet(nsim+1, 1, depth1) |> device
    h2 = Unet(1, 1, depth2) |> device
    ps = Flux.params(h1, h2)
    net(dm, dobs, simd, Jl, Js) = loss_func(supervised)(h1, h2, Jl, Js, dm, simd, dobs; device=device)
    return net, ps
end


