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
import sys

import womaplotting

R_earth = 6.371e6   # m
M_earth = 5.9722e24  # kg
G = 6.67430e-11  # m^3 kg^-1 s^-2

## Define the planet
mass_scale = float(sys.argv[1])

#planet = woma.Planet(
#	name		= f"Earth {mass_scale}M",
#	A1_mat_layer	= ["ANEOS_iron", "ANEOS_forsterite"],
#	A1_T_rho_type	= ["power=2", "power=2"],
#	rho_s		= 3300,
#	T_s		= 2000,
#	A1_M_layer	= [0.3 * mass_scale * M_earth, 0.7 * mass_scale * M_earth],
#	M		= mass_scale * M_earth
#)

# Generate the profiles
#planet.gen_prof_L2_find_R_R1_given_M1_M2(0.99 * R_earth, 1.055 * R_earth, num_attempt=40)

planet = woma.Planet(
	name		= f"Impactor {sys.argv[1]}M",
	A1_mat_layer	= ["ANEOS_iron", "ANEOS_forsterite"],
	A1_T_rho_type	= ["power=2", "power=2"],
	rho_s		= 3300,
	T_s		= 2000,
	A1_M_layer	= [0.3 * mass_scale * M_earth, 0.7 * mass_scale * M_earth],
	M		= mass_scale * M_earth
)

# Generate the profiles
planet.gen_prof_L2_find_R_R1_given_M1_M2(float(sys.argv[2]) * R_earth, float(sys.argv[3]) * R_earth, num_attempt=40)



output_directory = "planetesimal_files/{0}/".format(planet.name.replace(" ", "_"))

if os.path.exists(output_directory) == False:
	os.makedirs(output_directory)

## Plot profiles for sanity checking
womaplotting.plot_profiles(planet, output_directory)

maximum_expected_mass = 1.1 * M_earth
maximum_desired_particles = 6e5
particles_per_mass = maximum_desired_particles / maximum_expected_mass
print(particles_per_mass * mass_scale * M_earth)

#resolution_input = input("Particle resolution: ")

if False:#resolution_input != "":
	resolution = int(resolution_input)
else:
	resolution = particles_per_mass * mass_scale * M_earth

## Place particles, at the required resolution, onto the planet model
particles = woma.ParticlePlanet(planet, resolution, verbosity=0)
womaplotting.plot_particles(particles, planet, "{0}-cross-section-{1:.0e}".format(planet.name.replace(" ", "-"), resolution), output_directory)

filename = "{0}{1}.hdf5".format(output_directory, sys.argv[1] + "_impactor")

## Save the particle configuration for this planet
with h5py.File(filename, "w") as f:
	woma.save_particle_data(
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


