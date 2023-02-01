export plot_losses, plot_prediction

function plot_losses(loss_train::Vector{T}, loss_test::Vector{T}, k, e, plot_path; lr=1, n_epochs=1) where T
    length(loss_train) < 2 && return
    fig=figure();
    for (x, nax) in zip([loss_train, loss_test], ["train", "test"])
        xx = range(0f0, 1f0, length=length(x))
        semilogy(xx, x, label=nax)
    end
    ylabel("Objective f")
    xlabel("Normalize iterations")
    legend()
    title("Loss")
    fig_name = @strdict k e n_epochs lr 
    safesave(joinpath(plot_path,savename(fig_name; digits=6)*"_log.png"), fig)
    close(fig)
end


function plot_prediction(net, J, m0, dm, shots, k::Integer, e::Integer, plot_path;
                         lr=1, n_epochs=1, name="train")
    _, dp, qp, rtmp = net(dm, shots, m0)
    nsrc = size(shots, 3)
    dxrec = dxsrc = abs(diff(J.rInterpolation.geometry.xloc[1])[1])
    dtrec = J.rInterpolation.geometry.dt[1]
    model0 = J.model
    # Learned data and source
    fig = figure(figsize=(16,16))
    subplot(1,3,1)
    plot_sdata(shots[:, :, rand(1:nsrc), 1], (dtrec, dxrec); cmap="seismic", name=L"Single shot", new_fig=false, cbar=true)
    subplot(1,3,2)
    plot_sdata(dp[:, :, 1, 1], (dtrec, dxrec); cmap="seismic", name=L"d_p = h_1(d_o)", new_fig=false, cbar=true)
    subplot(1,3,3)
    plot_sdata(qp[:, :, 1, 1], (dtrec, dxsrc); cmap="seismic", name=L"q_p = h_2(h_1(d_o))", new_fig=false, cbar=true)

    tight_layout()
    fig_name = @strdict k e n_epochs lr  nsrc
    safesave(joinpath(plot_path, savename(fig_name; digits=6)*"_$(name)_data.png"), fig); close(fig)
    close(fig)

    # Rtms
    rtms, rtm = make_rtms(J, shots, m0;precon=false)

    fig = figure(figsize=(16, 16))
    subplot(2,2,1);
    im1 = plot_simage(reshape(rtmp, model0.n)', model0.d; cmap="gray", interp="none", name=L"J(q_p)'d_p", new_fig=false, cbar=true)
    subplot(2,2,2);
    im1 = plot_simage(reshape(dm, model0.n)', model0.d; cmap="gray", interp="none", name="True dm", new_fig=false, cbar=true)
    subplot(2,2,3);
    im1 = plot_simage(rtms', model0.d; cmap="gray", interp="none", name="SimSource RTM", new_fig=false, cbar=true)
    subplot(2,2,4);
    im1 = plot_simage(rtm', model0.d; cmap="gray", interp="none", name="Standard RTM", new_fig=false, cbar=true)
    tight_layout()

    fig_name = @strdict k e n_epochs lr  
    safesave(joinpath(plot_path, savename(fig_name; digits=6)*"_$(name)_image.png"), fig);
    close(fig)
end