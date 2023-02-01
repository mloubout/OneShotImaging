export setup_operators
using Random 
_vec_cell(x) = [vcat(x...)]

function setup_operators(J::judiJacobian)
    sim_G_rec = simult_rec_geom(J.rInterpolation.geometry)
    # Sim rec-as-src Jacobian
    simD = randn(Float32, sim_G_rec.nt[1], sim_G_rec.nrec[1])
    Jsr = judiJacobian(judiModeling(deepcopy(J.model), sim_G_rec, sim_G_rec; options=J.options), judiVector(sim_G_rec, [simD]))
    return Jsr
end

function sim_src_imaging(J::judiJacobian, d_obs, m0)
    q_sim, qw = make_sim_source(J.q)
    sim_shot = make_super_shot(d_obs, qw;J=J)
    Jsim = judiJacobian(judiModeling(deepcopy(J.model), q_sim.geometry, sim_shot.geometry; options=J.options), q_sim)
    return Jsim(;m=m0), sim_shot
end

simult_src_geom(G::Geometry) = Geometry(_vec_cell(G.xloc), _vec_cell(G.yloc), _vec_cell(G.zloc); dt=G.dt[1], t=G.t[1])
simult_rec_geom(G::Geometry) = G[1]

function make_sim_source(q::judiVector) 
    sim_G = simult_src_geom(q.geometry)
    Random.seed!(123)
    qw = randn(Float32, 1, q.nsrc)
    sim_D = hcat(q.data...) .* qw
    return judiVector(sim_G, sim_D), qw
end

function make_super_shot(d::judiVector, qw::Array{Float32};J=nothing)
    sim_G = simult_rec_geom(d.geometry)
    sim_D = sum(d.data .* qw)
    return judiVector(sim_G, sim_D)
end

function make_super_shot(d_obs::Array{Float32, 4}, qw::Array{Float32};J=nothing)
    sim_G = simult_rec_geom(J.rInterpolation.geometry)
    nsrc = size(d_obs, 3)
    d = judiVector(J.rInterpolation.geometry, [d_obs[:, :, s, 1] for s=1:nsrc])
    sim_D = sum(d.data .* qw)
    return sim_D
end

make_super_shot(d::judiVector) = make_super_shot(d, randn(Float32, d.nsrc))

make_precon(::Any, ::Val{false}) = LinearAlgebra.I
make_precon(J::judiJacobian, ::Val{true}) = judiTopmute(J.model.n, 20, 10) #inv(judiIllumination(J; mode="uv", recompute=false))

function make_rtms(J, d_obs::Array{Float32, 4}, m0; precon=false)
    nsrc = size(d_obs, 3)
    d_obs = judiVector(J.rInterpolation.geometry, [d_obs[:, :, s, 1] for s=1:nsrc])
    return make_rtms(J, d_obs, m0; precon=precon)
end


function make_rtms(J, d_obs::judiVector, m0; precon=false)
    Jsim, simshots = sim_src_imaging(J, d_obs, m0)
    J.model.m .= m0
    rtms = reshape(make_precon(Jsim, Val(precon))*vec(Jsim' * simshots), J.model.n...)
    rtm = reshape(make_precon(J, Val(precon))*vec(J' * d_obs), J.model.n...)
    return rtms, rtm
end


struct Jacobians
    J
    Jsim
    Jsq
    M
end

function make_Js(J::judiJacobian; M=nothing, precon=true)
    # Shot record as source
    Jsim = setup_operators(J)
    #M = make_precon(Jsim, Val(precon))

    # Standard SimSource
    q_sim, _ = make_sim_source(J.q)
    sim_G_rec = simult_rec_geom(J.rInterpolation.geometry)
    Jsq = judiJacobian(judiModeling(J.model, q_sim.geometry, sim_G_rec; options=J.options), q_sim)
    return Jacobians(J, Jsim, Jsq, M)
end

function set_m0(Js::Jacobians, m0)
    Js.J.model.m .= m0
    Js.Jsim.model.m .= m0
    Js.Jsq.model.m .= m0
end