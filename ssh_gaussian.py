#!/usr/bin/env python
# coding: utf-8
import meep as mp
import numpy as np
import matplotlib.pyplot as plt
from scipy.constants import c

def generate_waveguide(N, intra_pair_spacing, inter_pair_spacing, radius):
        # Adjust spacings to include the diameters of the circles
        adjusted_intra_pair_spacing = intra_pair_spacing + 2 * radius
        adjusted_inter_pair_spacing = inter_pair_spacing + 2 * radius
        
        # Generate the list with adjusted spacings
        result = [0]
        for i in range(1, N):
            if i % 2 == 1:  # If it's an odd index, use adjusted intra-pair spacing
                next_value = result[-1] + adjusted_intra_pair_spacing
            else:  # For even indices, use adjusted inter-pair spacing
                next_value = result[-1] + adjusted_inter_pair_spacing
            result.append(next_value)
        
        # Find the center value
        if N % 2 == 1:
            center_value = result[len(result) // 2]
        else:
            center_value = (result[(len(result) // 2) - 1] + result[len(result) // 2]) / 2
        
        # Shift the list so the center value is at 0
        shifted_result = [x - center_value for x in result]
        
        return shifted_result

def wave_propagate_plot(cell, sim, comp, N, freq):
    if comp == "z":
        field=mp.Ez
    elif comp == "y":
        field=mp.Ey
    elif comp == "x":
        field=mp.Ex
    eps_data = sim.get_array(center=mp.Vector3(), size=cell, component=mp.Dielectric)
    ez_data = sim.get_array(center=mp.Vector3(), size=cell, component=field)
    plt.figure(figsize=(16,9) )
    plt.title(f"Wave propagation with {N} holes {comp} component frequency {freq} Hz")
    plt.imshow(eps_data.transpose(), interpolation='spline36', cmap='binary')
    plt.imshow(ez_data.transpose(), interpolation='spline36', cmap='RdBu', alpha=0.9)
    plt.axis('off')
    plt.savefig(f"wave_propag_N{N}_C{comp}_f{freq}.png")

def waveguide_visualizer(sim, N):
    plt.figure(dpi=150)
    plt.title(f"Waveguide with {N} holes")
    sim.plot2D()
    plt.savefig(f"waveguide_N{N}.png")

def trans_refl_spectra(frequency, trans, refl, N, comp, freq):
    plt.figure(figsize=(10, 6))
    plt.plot(frequency*c/(1e-6), trans, label="Transmission")
    plt.plot(frequency*c/(1e-6), refl, label="Reflection")
    plt.plot(frequency*c/(1e-6), 1 - refl - trans, label="Loss")
    plt.xlabel("Frequency Hz")
    plt.ylabel("Normalized Flux")
    plt.legend()
    plt.title(f"Transmission and Reflection Spectra N {N} at center frequency {freq} Hz")
    plt.grid(True)
    plt.savefig(f"Spectra_N{N}_C{comp}_f{freq}.png")

def simulation(cell,pml_layers, geometry_noholes,geometry_holes, resolution, sx, dpml, fcen, df, nfreq, w, comp, N, pad, sy):
    if comp == "z":
        field = mp.Ez
    elif comp == "y":
        field = mp.Ey
    elif comp == "x":
        field = mp.Ex

    wvg_xcen =  0.5*(sx-w-2*pad)  # x center of horiz. wvg
    wvg_ycen = -0.5*(sy-w-2*pad)  # y center of vert. wvg

    # Source
    source = [mp.Source(mp.GaussianSource(frequency=fcen, fwidth=df),component=field,center=mp.Vector3(-0.5 * sx + dpml + 0.5, 0),size=mp.Vector3(0, w, 0))]
    
    # Normalization run simulation
    norm_sim = mp.Simulation(cell_size=cell,
                            boundary_layers=pml_layers,
                            geometry=geometry_noholes,  # No holes for normalization
                            sources=source,
                            resolution=resolution)
    # Flux monitors
    refl_fr = mp.FluxRegion(center=mp.Vector3(-0.5 * sx + dpml + 0.1), size=mp.Vector3(0, 2*w, 0))
    tran_fr = mp.FluxRegion(center=mp.Vector3(0.5 * sx - dpml - 0.1), size=mp.Vector3(0, 2*w, 0))

    # Add flux monitors
    norm_refl = norm_sim.add_flux(fcen, df, nfreq, refl_fr)
    norm_tran = norm_sim.add_flux(fcen, df, nfreq, tran_fr)

    waveguide_visualizer(norm_sim, 0)
    norm_sim.run(until_after_sources=mp.stop_when_fields_decayed(50, field, mp.Vector3(0.5 * sx - dpml - 0.5), 1e-3), )

    wave_propagate_plot(cell, norm_sim, comp, 0, fcen)

    # Save flux for normalization
    norm_refl_data = norm_sim.get_flux_data(norm_refl)
    norm_tran_data = norm_sim.get_flux_data(norm_tran)

    # Calculate the incident power for normalization
    norm_refl_flux = np.array(mp.get_fluxes(norm_refl))
    norm_tran_flux = np.array(mp.get_fluxes(norm_tran))
    norm_incident_flux = np.array(norm_refl_flux) + np.array(norm_tran_flux)
    
    flux_freqs = np.array(mp.get_flux_freqs(norm_refl))

    trans_refl_spectra(flux_freqs,norm_tran_flux/norm_incident_flux , norm_refl_flux/norm_incident_flux , 0, comp, fcen)
    # Reset simulation for main run with holes
    norm_sim.reset_meep()

    sim = mp.Simulation(cell_size=cell,boundary_layers=pml_layers,geometry=geometry_holes, sources=source,resolution=resolution)

    # Add flux monitors
    refl = sim.add_flux(fcen, df, nfreq, refl_fr)
    tran = sim.add_flux(fcen, df, nfreq, tran_fr)

    # Load the saved flux data for normalization
    sim.load_minus_flux_data(refl, norm_refl_data)
    sim.load_minus_flux_data(tran, norm_tran_data)

    # Run simulation with holes
    waveguide_visualizer(sim, N)
    sim.run(until_after_sources=mp.stop_when_fields_decayed(50, field, mp.Vector3(0.5 * sx - dpml - 0.5), 1e-3))
    
    wave_propagate_plot(cell, sim, comp, N, fcen)

    # Get the reflected and transmitted flux
    reflected_flux = mp.get_fluxes(refl)
    transmitted_flux = mp.get_fluxes(tran)

    # Normalize the transmission and reflection
    transmission = np.array(transmitted_flux) / norm_incident_flux
    reflection = np.array(reflected_flux) / norm_incident_flux
    freqs = np.linspace(fcen - df/2, fcen + df/2, nfreq)
    
    trans_refl_spectra(flux_freqs, transmission, reflection, N, comp, fcen)

if __name__ == "__main__":

    # Material definitions
    n_SiO2 = 1.45  # Refractive index of SiO2
    eps_SiO2 = n_SiO2**2  # Dielectric constant of SiO2

    resolution = 20  # pixels/um
    w = 2  # width of waveguide in um
    r = 0.5  # radius of holes in um
    t_1 = 1  # intra-pair spacing
    t_2 = 2  # inter-pair spacing (defect spacing, ordinary spacing = 1)
    sy = 10  # size of cell in y direction (perpendicular to waveguide), in um
    pad = 7  # padding between last hole and PML edge, in um
    dpml = 1  # PML thickness, in um
    N = 16  # number of holes

    # Calculate the size of the cell in the x direction
    if N % 2 == 0:
        sx = 2 * (pad + dpml + N // 2 + 2 * N // 2) - 20
    else:
        sx = 2 * (pad + dpml + (N + 1) // 2 + 2 * (N + 1) // 2) - 20

    wvg_xcen =  0.5*(sx-w-2*pad)  # x center of horiz. wvg
    wvg_ycen = -0.5*(sy-w-2*pad)  # y center of vert. wvg

    cell = mp.Vector3(sx, sy, 0)
    pml_layers = [mp.PML(dpml)]

    # Define the waveguide material
    SiO2_material = mp.Medium(epsilon=eps_SiO2)

    geometry_noholes = [mp.Block(size=mp.Vector3(mp.inf, w, mp.inf),material=SiO2_material)]

    # Main simulation with holes
    geometry_holes = [mp.Block(size=mp.Vector3(mp.inf, w, mp.inf), material=SiO2_material)]
    hole_positions = generate_waveguide(N, t_1, t_2, r)

    for x_position in hole_positions:
        geometry_holes.append(mp.Cylinder(r, height=mp.inf, center=mp.Vector3(x_position, 0, 0), material=mp.air))

    # Wavelength range from 400 nm (0.4 microns) to 1 micron
    # Convert this range to frequency in Meep units (1/λ where λ is in microns)
    min_wl = 1  # in microns
    max_wl = 1.5  # in microns
    min_f = 1/max_wl  # Minimum frequency (1/λ)
    max_f = 1/min_wl  # Maximum frequency (1/λ)

    # Center frequency and frequency width
    fcen = (min_f + max_f) / 2
    df = 0.6#(max_f - min_f)


    # Number of frequency points
    nfreq = 500

    #simulation(cell,pml_layers, geometry_noholes,geometry_holes, resolution, sx, dpml, fcen, df, nfreq, w, "z", N, pad, sy)

    #simulation(cell,pml_layers, geometry_noholes,geometry_holes, resolution, sx, dpml, fcen, df, nfreq, w, "y", N, pad, sy)
    simulation(cell,pml_layers, geometry_noholes,geometry_holes, resolution, sx, dpml, fcen, df, nfreq, w, "x", N, pad, sy)