export setup_refs, make_model, get_shots


#### JLD@ setup

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


### SEGY setup, prefered
function get_shots(idx::Integer, models::Vector{Model}, d_obs::judiVector, J::judiJacobian; nsim::Integer=get_nsrc(J.rInterpolation), batch_size=1, buffer=250f0)
    mid_shot = randperm(d_obs.nsrc)[1]
    is, ie = max(1, mid_shot-150), min(d_obs.nsrc, mid_shot+150)
    sidx = randperm((ie-is+1))[1:nsim] .+ is
    sdata = d_obs[sidx]
    # Pad data zero per channels
    sw = ones(Float32, batch_size, sdata.nsrc)
    sdata = simsource(sw, sdata; reduction=nothing)
    # Get model
    m = models[(idx%3) + 1]
    m, _ = limit_model_to_receiver_area(sdata.geometry, sdata.geometry, m, buffer)
    return sdata, m, J[sidx]
end
