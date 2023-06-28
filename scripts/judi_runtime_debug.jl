using LinearAlgebra
using Random
using JLD2
using JUDI

JUDI.set_verbosity(true)

function circle_geom_half(h, k, r, numpoints)
    theta = LinRange(0f0, pi, numpoints+1)[1:end-1]
    Float32.(h .- r*cos.(theta)), Float32.(k .+ r*sin.(theta)), theta
end

# get previous iterate

for ind in (550*4+1):(550*5)
    println("does file exist for $(ind)?")
    Base.flush(Base.stdout)
    if !isfile("/slimdata/rafaeldata/fastmri_brains/cond_brain_half_iter_1_grad_ind_"*string(ind)*".jld2")
        println("no then please generate  for $(ind)")
          Base.flush(Base.stdout)
        #get dobs and rho0 to calculate new gradient
        @load "/slimdata/rafaeldata/fastmri_brains/fastmri_training_data_"*string(ind)*".jld2" m0 m rho0 rho   

        ############################### simulation configurations ################################# 
        n = size(m0)
        d = (0.5,0.5)
        o = (0,0)
        # Set up rec geometry 
        nrec = 256 # no. of recs
        nsrc = 16 # no. of srcs
        tn = 320  # total simulation time in microseconds
        dt = 0.06f0 # around 10Mhz sampling rate
        nt = Int(div(tn,dt)) + 1

        # Modeling time and sampling interval
        sampling_rate = 1f0 / dt
        f0 = .4    # Central frequency in MHz
        cycles_wavelet = 3  # number of sine cycles
        nt = Int(div(tn,dt)) + 1
        wavelet = ricker_wavelet(tn, dt, f0)

        model = Model(n, d, o, m; rho=rho)
        model0 = Model(n, d, o, m0;rho=rho0)
        # Setup circumference of receivers 
        domain_x = (n[1] - 1)*d[1]
        domain_z = (n[2] - 1)*d[2]
        rad = .95*domain_x / 2
        xrec, zrec, theta = circle_geom_half(domain_x / 2, domain_z / 2, rad, nrec)
        yrec = 0f0 #2d so always 0

        # Set up source structure
        #get correct number of sources by grabbing subset of the receiver positions
        step_num = Int(nrec/nsrc)
        xsrc  = xrec[1:step_num:end]
        ysrc  = range(0f0, stop=0f0, length=nsrc)
        zsrc  = zrec[1:step_num:end]

        # Convert to cell in order to separate sources 
        src_geometry = Geometry(convertToCell(xsrc), convertToCell(ysrc), convertToCell(zsrc); dt=dt, t=tn)
        q = judiVector(src_geometry, wavelet)

        # Set up receiver structure
        rec_geometry = Geometry(xrec, yrec, zrec; dt=dt, t=tn, nsrc=nsrc);

        ###################################################################################################
        # Setup operators
        opt = Options(isic=true, dt_comp=dt)
        F = judiModeling(model, q.geometry, rec_geometry; options=opt)
        F0 = judiModeling(model0, q.geometry, rec_geometry; options=opt)
        J = judiJacobian(F0, q)
        t1 = @elapsed begin
            dobs = F*q
        end

        Y_train = zeros(Float32, n..., nsrc)
        for i in 1:nsrc
            ts = @elapsed begin
                f, g = fwi_objective(model0, q[i], dobs[i]; options=opt)
                Y_train[:,:,i] .= g.data
            end
            println("Data imaged in $(ts)sec")
        end
    end
end



