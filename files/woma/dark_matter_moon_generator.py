#################################################################
#								#
#								#
#		Would a Mighty Smack Tilt Uranus?		#
#								#
#		Louis Eddershaw					#
#								#
#		2023/24						#
#								#
#								#
#################################################################

import woma
import h5py
import os
import numpy as np

from woma.misc import glob_vars as gv
from woma.misc.utils import SI_to_SI, SI_to_cgs


import womaplotting

R_earth = 6.371e6   # m
M_earth = 5.9722e24  # kg

R_moon = 1.740e6     # m
M_moon = 7.35e22     # kg

G = 6.67430e-11  # m^3 kg^-1 s^-2


A2_pos = np.asarray([[0.0, 0.0, 0.0]])
A2_vel = np.asarray([[0.0, 0.0, 0.0]])
A1_m = np.asarray([M_moon])
A1_rho = np.asarray([M_moon / (4 * np.pi * R_moon**3)])
A1_empty_array = np.asarray([[0.0]])

filename = "{0}.hdf5".format(input("Name of particle file: "))

## Save the particle configuration for this planet
with h5py.File(filename, "w") as f:
	woma.save_particle_data(
	f,
	A2_pos,
	A2_vel,
	A1_m,
	A1_empty_array,
        A1_rho,
	A1_empty_array,
	A1_empty_array,
	A1_empty_array,
	boxsize=15 * R_earth,
	dark_matter=True
	)
print("Done!")

quit()

## Define the planet
moon = woma.Planet( 
    name            = "Single Particle Moon", 
    A1_mat_layer    = ["ANEOS_forsterite"], 
    A1_T_rho_type   = ["power=2"],
    rho_s           = 3300,
    T_s             = 273,
    M               = M_moon
    #R               = R_moon
)
## Converge on a solution with these planet parameters
moon.gen_prof_L1_find_R_given_M(R_max = 1.1*R_moon)



output_directory = "planetesimal_files/{0}/".format(moon.name.replace(" ", "_"))

if os.path.exists(output_directory) == False:
	os.makedirs(output_directory)

## Plot profiles for sanity checking
womaplotting.plot_profiles(moon, output_directory)

resolution_input = input("Particle resolution: ")

if resolution_input != "":
	resolution = int(resolution_input)
else:
	resolution = 1

## Place particles, at the required resolution, onto the planet model
particles = woma.ParticlePlanet(moon, resolution, verbosity=0)
womaplotting.plot_particles(particles, moon, "{0}-cross-section-{1:.0e}".format(moon.name.replace(" ", "-"), resolution), output_directory)

filename = "{0}{1}.hdf5".format(output_directory, input("Name of particle file: "))

## Save the particle configuration for this planet
with h5py.File(filename, "w") as f:
	woma.save_dark_matter_particle_data(
	f,
	particles.A2_pos,
	particles.A2_vel,
	particles.A1_m,
        particles.A1_h,
        particles.A1_rho,
        particles.A1_P,
        particles.A1_u,
        particles.A1_mat_id,
	boxsize=15 * R_earth
	)
print("Done!")


