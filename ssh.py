#!/usr/bin/env python
# coding: utf-8
import meep as mp
import numpy as np
import matplotlib.pyplot as plt

for N in range(16,18):
    resolution = 20   # pixels/um
    eps = 13          # dielectric constant of waveguide
    w = 1.2           # width of waveguide
    r = 0.36          # radius of holes
    t_1 = 1
    t_2 = 2          # defect spacing (ordinary spacing = 1)
    sy = 5      # size of cell in y direction (perpendicular to wvg.)
    pad = 2           # padding between last hole and PML edge
    dpml = 1          # PML thickness
    
    sx = 2*(pad+dpml)+N*2*r + N//2 * t_1 + N%2 * t_2
    #sy = 2*(pad+dpml+N)+d-1 # size of cell in X direction
    sy = 10 # size of cell in Y direction
    cell = mp.Vector3(sx, sy, 0)

    dpml = 1.0
    pml_layers = [mp.PML(dpml)]

    
    w = 1  # width of waveguide

    wvg_xcen = 0.5 * sx  # x center of horiz. wvg
    wvg_ycen = -0.5 * sy  # y center of vert. wvg

    geometry = [mp.Block(size=mp.Vector3(mp.inf, w, mp.inf),material=mp.Medium(epsilon=eps),)]

    def generate_waveguide(N, intra_pair_spacing, inter_pair_spacing, radius):
        if N < 1:
            return []
        
        # Initialize the list with the first value
        result = [0]
        
        # Use a loop to generate the rest of the numbers
        for i in range(1, N):
            if i % 2 == 1:  # If it's an odd index, use intra-pair spacing
                next_value = result[-1] + intra_pair_spacing
            else:  # For even indices, use inter-pair spacing
                next_value = result[-1] + inter_pair_spacing - intra_pair_spacing
            result.append(next_value)
        
        # Find the center value
        if N % 2 == 1:
            center_value = result[N // 2]
        else:
            center_value = (result[(N // 2) - 1] + result[N // 2]) / 2
        
        # Shift the list so the center value is at 0
        shifted_result = np.array(result) - center_value
    
        return shifted_result
    
    for i in generate_waveguide(N, t_1, t_2, r):
        geometry.append(mp.Cylinder(r, center=mp.Vector3(i)))
    
    fcen = 0.15  # pulse center frequency
    df = 0.1  # pulse width (in frequency)
    sources = [mp.Source(mp.GaussianSource(fcen, fwidth=df),component=mp.Ez,center=mp.Vector3(-0.5 * sx + dpml, wvg_ycen, 0),size=mp.Vector3(0, w, 0),)]
    #sym = [mp.Mirror(mp.Y, phase=-1)]
    sim = mp.Simulation(cell_size=cell,boundary_layers=pml_layers,geometry=geometry,sources=sources,resolution=resolution)#,symmetries=sym)

    sim.run(until=1000)

    eps_data = sim.get_array(center=mp.Vector3(), size=cell, component=mp.Dielectric)
    ez_data = sim.get_array(center=mp.Vector3(), size=cell, component=mp.Ez)
    plt.figure()
    plt.imshow(eps_data.transpose(), interpolation='spline36', cmap='binary')
    plt.imshow(ez_data.transpose(), interpolation='spline36', cmap='RdBu', alpha=0.9)
    plt.axis('off')
    plt.savefig(f'N={N}.png')
    plt.close()
    sim.reset_meep()
    
    sim = mp.Simulation(cell_size=cell,boundary_layers=pml_layers,geometry=geometry,sources=sources,resolution=resolution)#,symmetries=sym)

    nfreq = 500  # number of frequencies at which to compute flux

    # reflected flux
    refl_fr = mp.FluxRegion(center=mp.Vector3(-0.5 * sx + dpml + 0.5, wvg_ycen, 0), size=mp.Vector3(0, 2 * w, 0))
    refl = sim.add_flux(fcen, df, nfreq, refl_fr)

    # transmitted flux
    tran_fr = mp.FluxRegion(center=mp.Vector3(0.5 * sx - dpml, wvg_ycen, 0), size=mp.Vector3(0, 2 * w, 0))
    tran = sim.add_flux(fcen, df, nfreq, tran_fr)

    plt.figure(dpi=150)
    sim.plot2D()
    plt.savefig(f"struct_N{N}.png")
    plt.close()

    pt = mp.Vector3(0.5 * sx - dpml - 0.5, wvg_ycen)
    sim.run(until_after_sources=mp.stop_when_fields_decayed(50, mp.Ez, pt, 1e-3))
    # for normalization run, save flux fields data for reflection plane
    straight_refl_data = sim.get_flux_data(refl)

    # save incident power for transmission plane
    straight_tran_flux = mp.get_fluxes(tran)

    sim.reset_meep()

    sim = mp.Simulation(cell_size=cell,boundary_layers=pml_layers,geometry=geometry,sources=sources,resolution=resolution,)

    # reflected flux
    refl = sim.add_flux(fcen, df, nfreq, refl_fr)

    tran_fr = mp.FluxRegion(center=mp.Vector3(wvg_xcen, 0.5 * sy - dpml - 0.5, 0), size=mp.Vector3(2 * w, 0, 0))
    tran = sim.add_flux(fcen, df, nfreq, tran_fr)

    # for normal run, load negated fields to subtract incident from refl. fields
    sim.load_minus_flux_data(refl, straight_refl_data)

    pt = mp.Vector3(wvg_xcen, 0.5 * sy - dpml - 0.5)

    sim.run(until_after_sources=mp.stop_when_fields_decayed(50, mp.Ez, pt, 1e-3))

    bend_refl_flux = mp.get_fluxes(refl)
    bend_tran_flux = mp.get_fluxes(tran)

    flux_freqs = mp.get_flux_freqs(refl)

    wl = []
    Rs = []
    Ts = []
    for i in range(nfreq):
        wl = np.append(wl, 1 / flux_freqs[i])
        Rs = np.append(Rs, -bend_refl_flux[i] / straight_tran_flux[i])
        Ts = np.append(Ts, bend_tran_flux[i] / straight_tran_flux[i])

    if mp.am_master():
        plt.figure(dpi=150)
        plt.plot(wl, Rs, "b-", label="reflectance")
        plt.plot(wl, Ts, "r-", label="transmittance")
        plt.plot(wl, 1 - Rs - Ts, "g-", label="loss")
        #plt.axis([5.0, 10.0, 0, 1])
        plt.xlabel("wavelength (μm)")
        plt.legend(loc="upper right")
        plt.savefig(f"trans_ref_N{N}.png")
        plt.close()