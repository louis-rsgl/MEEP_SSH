import meep as mp
import argparse
import numpy as np
from meep.materials import SiO2

def main(args):
    resolution = 20   # pixels/um

    eps = 13          # dielectric constant of waveguide
    w = 1.2           # width of waveguide
    r = 0.36 
    t_1 = 1.1
    t_2 = 0.7
    N = args.N        # number of holes on either side of defect

    sy = args.sy      # size of cell in y direction (perpendicular to wvg.)
    pad = 2           # padding between last hole and PML edge
    dpml = 1          # PML thickness
    if N % 2 == 0:
        sx = 2*(pad+dpml+N//2 + 2*N//2) - 20 # size of cell in x direction
    else:
        sx = 2*(pad+dpml+(N-1)//2 + 2*(N-1)//2) - 20 # size of cell in x direction
    print(sx)
    cell = mp.Vector3(sx,sy,0)
    blk = mp.Block(size=mp.Vector3(mp.inf,w,mp.inf), material=mp.Medium(epsilon=eps))#SiO2)#mp.Medium(epsilon=eps))
    geometry = [blk]

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

    
    for i in generate_waveguide(N, t_1, t_2, r):
        geometry.append(mp.Cylinder(r, center=mp.Vector3(i)))
    
    pml_layers = [mp.PML(1.0)]

    fcen = args.fcen   # pulse center frequency
    df = args.df       # pulse frequency width

    src = [mp.Source(mp.GaussianSource(fcen, fwidth=df),
	                 component=mp.Ey,
			         center=mp.Vector3(-0.5*sx+dpml),
			         size=mp.Vector3(0,w))]
    
    sym = [mp.Mirror(mp.Y, phase=-1)]

    sim = mp.Simulation(cell_size=cell,
	                    geometry=geometry,
			            boundary_layers=pml_layers,
				        sources=src,
					    symmetries=sym,
					    resolution=resolution)
    
    freg = mp.FluxRegion(center=mp.Vector3(0.5*sx-dpml-0.5),
	                     size=mp.Vector3(0,2*w))

    nfreq = 500 # number of frequencies at which to compute flux

    # transmitted flux
    trans = sim.add_flux(fcen, df, nfreq, freg)

    vol = mp.Volume(mp.Vector3(0), size=mp.Vector3(sx))

    sim.run(mp.at_beginning(mp.output_epsilon),
                mp.during_sources(mp.in_volume(vol, mp.to_appended("hz-slice", mp.at_every(0.4, mp.output_hfield_z)))),
                until_after_sources=mp.stop_when_fields_decayed(50, mp.Ey, mp.Vector3(0.5*sx-dpml-0.5), 1e-3))

    sim.display_fluxes(trans)  # print out the flux spectrum

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-N', type=int, default=17, help='number of holes on either side of defect')
    parser.add_argument('-sy', type=int, default=2.5, help='size of cell in y direction (perpendicular to wvg.)')
    parser.add_argument('-fcen', type=float, default=0.25, help='pulse center frequency')
    parser.add_argument('-df', type=float, default=0.2, help='pulse frequency width')
    args = parser.parse_args()
    main(args)