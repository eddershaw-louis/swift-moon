import womaplotting
import woma
import h5py
import numpy as np
import os
import glob
import swiftsimio as sw
import unyt
import sys
import pandas as pd

R_earth = 6.371e6   # m
M_earth = 5.9724e24  # kg

R_moon = 1.740e6     # m
M_moon = 7.35e22     # kg

G = 6.67408e-11  # m^3 kg^-1 s^-2

sys.path.append("..")
sys.path.append("../..")
sys.path.append("../custom_functions")

from custom_functions import custom_woma
from custom_functions import write_gadget as write

from sampled_impact import sample_distributions

def sample_distributions(distributions):
	samples = []
	for distribution in distributions:
		histogram = distribution[0]
		bin_edges = distribution[1]

		probabilities = histogram / histogram.sum()
		bin_centres = (bin_edges[:-1] + bin_edges[1:]) / 2
		samples.append(np.random.choice(bin_centres, size=1, p=probabilities)[0])

	return tuple(samples)

def main(parameters):

	print(parameters)

	if len(parameters) > 0:
		target_file = parameters[0]
	else:
		raise Exception("Error: No target hdf5 file provided!")

	if ('-M' in parameters):
		M_moonlet = float(parameters[parameters.index('-M') + 1])
	else:
		M_moonlet = 0

	if ('-o' in parameters):
		output_directory = parameters[parameters.index('-o') + 1]
	else:
		output_directory = "impact_files"

	if ('-c' in parameters):
		chain = True
	else:
		chain = False
	

	if False:
		sample_size = 10000

		mass_histogram_df = pd.read_excel("../scripts/mass_histogram.xlsx")
		mass_histogram = (mass_histogram_df['Counts'].values, np.append(mass_histogram_df['Bin Start'].values, mass_histogram_df['Bin End'].values[-1]))

		vel_histogram_df = pd.read_excel("../scripts/velocity_histogram.xlsx")
		vel_histogram = (vel_histogram_df['Counts'].values, np.append(vel_histogram_df['Bin Start'].values, vel_histogram_df['Bin End'].values[-1]))

		## Impact ANGLE (in units of degrees)
		B_histogram, B_bin_edges = np.histogram(np.random.default_rng().normal(45, 35, size=sample_size), bins=150)

		## Lunar orbital distance (in units of R_earth)
		#lunar_orbit_histogram, lunar_orbit_bin_edges = np.histogram(np.random.random(size=sample_size) * 100 + 50, bins=150)

		## Lunar starting angle (in units of degrees, angle measured from the x axis initially in the positive y direction)
		theta_histogram, theta_bin_edges = np.histogram(np.random.random(size=sample_size) * 360, bins=150)

		distributions = [[mass_histogram, mass_bin_edges], [vel_histogram, vel_bin_edges], [B_histogram, B_bin_edges], [theta_histogram, theta_bin_edges]]

		impact_parameters = sample_distributions(distributions)
	else:
		impact_parameters = (0.013, 2.3, 41.0, 43.0, 172.0)

	print(impact_parameters)

	impactor_mass = impact_parameters[0]
	impact_velocity = impact_parameters[1]
	impact_angle = impact_parameters[2]
	lunar_orbit = 16	## R_earth

	## Consider a Moon or not (setting Moon mass to 0 will ignore Moon)
	moon_flag = M_moonlet != 0


	impact_approach_angle = impact_parameters[4]

	boxsize = 500 * R_earth

	#help(woma.Conversions(M_earth, R_earth, 1))
	unit_converter = woma.Conversions(M_earth, R_earth, 1)

	## Reads ROTATING Earth particle file
	#earth_file = "rotating_earth_7.5hr.hdf5"
	#if "rotating" in earth_file: rotating = True

	with h5py.File(target_file, "r") as f:
		m_t = f["PartType0/Masses"][:] * unit_converter.m
		pos_t = f["PartType0/Coordinates"][:] * unit_converter.l
		vel_t = f["PartType0/Velocities"][:] * unit_converter.v
		matid_t = f["PartType0/MaterialIDs"][:]
		h_t = f["PartType0/SmoothingLengths"][:] * unit_converter.l
		rho_t = f["PartType0/Densities"][:] * unit_converter.rho
		P_t = f["PartType0/Pressures"][:] * unit_converter.P
		u_t = f["PartType0/InternalEnergies"][:] * unit_converter.u

	M_t = np.sum(m_t)
	R_t = (np.max(pos_t) - np.min(pos_t)) / 2
	print(M_t, R_t)

	## Adjust centre of mass to origin
	pos_t -= np.sum(m_t[:, np.newaxis] * pos_t, axis=0) / M_t


	## Select the impactor with the closest mass to the one sampled
	impactor_files = glob.glob("impact_files/impactors/*_impactor.hdf5")
	impactor_masses = []
	for impactor_file in impactor_files:
		impactor_masses.append(float(impactor_file[23:-14]))
	impactor_masses.sort()

	closest_impactor_mass = min(impactor_masses, key=lambda x: abs(x - impactor_mass))

	with h5py.File(f"impact_files/impactors/{closest_impactor_mass}_impactor.hdf5", "r") as f:
		m_i = f["PartType0/Masses"][:] * unit_converter.m
		pos_i = f["PartType0/Coordinates"][:] * unit_converter.l
		vel_i = f["PartType0/Velocities"][:] * unit_converter.v
		matid_i = f["PartType0/MaterialIDs"][:]
		h_i = f["PartType0/SmoothingLengths"][:] * unit_converter.l
		rho_i = f["PartType0/Densities"][:] * unit_converter.rho
		P_i = f["PartType0/Pressures"][:] * unit_converter.P
		u_i = f["PartType0/InternalEnergies"][:] * unit_converter.u

	M_i = np.sum(m_i)
	R_i = (np.max(pos_i) - np.min(pos_i)) / 2
	print(M_i, R_i)

	## Adjust centre of mass to origin
	pos_i -= np.sum(m_i[:, np.newaxis] * pos_i, axis=0) / M_i

	## Calculate the mutual escape velocity of the impactor and target
	v_esc = np.sqrt(2 * G * (M_t + M_i) / (R_t + R_i)) 

	## Velocity at contact
	v_c = v_esc * impact_velocity


	impact_pos_t = np.array([0., 0., 0.])
	impact_vel_t = np.array([0., 0., 0.])

	## Compute the initial configuration that satisfies the impact parameters
	impact_pos_i, impact_vel_i = woma.impact_pos_vel_b_v_c_t(
		b       = np.sin(impact_angle * np.pi/180), 
		v_c     = v_c,
		t       = 1.5 * 3600,  
		R_t     = R_t, 
		R_i     = R_i, 
		M_t     = M_t, 
		M_i     = M_i,
	)

	if moon_flag:
		M_m = M_moonlet
		impact_pos_m = np.asarray([np.cos(impact_approach_angle * np.pi/180) * lunar_orbit * R_earth, np.sin(impact_approach_angle * np.pi/180) * lunar_orbit * R_earth, 0.0])

		orbital_speed = np.sqrt(G * M_t / np.linalg.norm(impact_pos_m - impact_pos_t))
		impact_vel_m = np.array([np.cos(impact_approach_angle * np.pi/180 + np.pi/2) * orbital_speed, np.sin(impact_approach_angle * np.pi/180 + np.pi/2) * orbital_speed, 0.0])

	# Centre of mass and momentum
	impact_flag = True
	if impact_flag:
		if moon_flag:
			impact_pos_com = (M_t * impact_pos_t + M_i * impact_pos_i + M_m * impact_pos_m) / (M_t + M_i + M_m)
			impact_vel_com = (M_t * impact_vel_t + M_i * impact_vel_i + M_m * impact_vel_m) / (M_t + M_i + M_m)
		else:
			impact_pos_com = (M_t * impact_pos_t + M_i * impact_pos_i) / (M_t + M_i)
			impact_vel_com = (M_t * impact_vel_t + M_i * impact_vel_i) / (M_t + M_i)
	else:
		if moon_flag:
			impact_pos_com = (M_t * impact_pos_t + M_m * impact_pos_m) / (M_t + M_m)
			impact_vel_com = (M_t * impact_vel_t + M_m * impact_vel_m) / (M_t + M_m)
		else:
			impact_pos_com = impact_pos_t * 0
			impact_vel_com = impact_pos_t * 0


	## Offset bodies such that the origin is the centre of mass
	impact_pos_t -= impact_pos_com
	if impact_flag: impact_pos_i -= impact_pos_com
	if moon_flag: impact_pos_m -= impact_pos_com


	## Offset the velocities of the bodies such that the origin is the centre of momentum
	impact_vel_t -= impact_vel_com
	if impact_flag: impact_vel_i -= impact_vel_com
	if moon_flag: impact_vel_m -= impact_vel_com

	print("Centre of momentum", impact_vel_com, "m/s")
	print(np.linalg.norm(impact_vel_com), "m/s\n")


	print("New Target Position:")
	print(impact_pos_t / R_earth, "R_earth")
	print("New Target Velocity:")
	print(impact_vel_t, "m/s")

	if impact_flag:
		print("\nNew Impactor Position:")
		print(impact_pos_i / R_earth, "R_earth")
		print("New Impactor Velocity:")
		print(impact_vel_i, "m/s")
		print(np.linalg.norm(impact_vel_i), "m/s\n")


		print("Impactor Approach Velocity Relative to Target:")
		print(np.linalg.norm(impact_vel_i) - np.linalg.norm(impact_vel_com), "m/s\n\n")
	
	if moon_flag:
		print("\nMoon Position:")
		print(impact_pos_m / R_earth, "R_earth")
		print("Moon Velocity:")
		print(impact_vel_m, "m/s")



	## Offset each bodies' particles' position and velocities to the correct values
	pos_t += impact_pos_t
	vel_t[:] += impact_vel_t

	if impact_flag:
		pos_i += impact_pos_i
		vel_i[:] += impact_vel_i

	if moon_flag:
		pos_m = np.array([impact_pos_m])
		pos_m += boxsize / 2.0

		vel_m = np.array([impact_vel_m])
	print()


	if impact_flag:
		## Save the impact configuration
		filename = "impact"
		#if rotating: filename += "_7.5hr"
		filename += "_{0:.3f}M_{1:.3f}M_b{3:.1f}_v{2:.1f}".format(M_t / M_earth, M_i / M_earth, impact_velocity, impact_angle)
		if moon_flag: filename += "_moon_{0:.3f}LM_d{1:.1f}_a{2:.1f}".format(M_moonlet / M_moon, lunar_orbit, impact_approach_angle)
	elif moon_flag:
		filename = "orbit"
		#if rotating: filename += "_7.5hr"
		filename += "_{0:.3f}M".format(M_t / M_earth)
		if moon_flag: filename += "_moon_{0:.3f}LM_d{1:.1f}_a{2:.1f}".format(M_moonlet / M_moon, lunar_orbit, impact_approach_angle)
	else:
		print("No moon or impact, nothing to write!")
		quit()

	with open(f"{output_directory}/chain_impact_info.txt", "w") as writer:
		writer.write("Chained Impact Info:")
		writer.write("\nTarget Mass: {M_t / M_earth} M_earth")
		if impact_flag:
			writer.write("\nImpactor Mass: {M_i / M_earth} M_earth")
			writer.write("\nImpact Angle: {impact_angle} degrees")
			writer.write("\nImpact Velocity: {impact_velocity} v_esc")
		if moon_flag:
			writer.write("\nMoonlet Mass: {M_moonlet / M_earth} M_earth")
			writer.write("\nMoonlet Distance: {lunar_orbit} R_earth")
			writer.write("\nMoonlet Approach Angle: {impact_approach_angle} degrees")
		
	if chain: filename = "ICs"

	with h5py.File(f"{output_directory}/{filename}.hdf5", "w") as f:
		if impact_flag:	
			custom_woma.save_particle_data(
				f,
				np.append(pos_t, pos_i, axis=0),
				np.append(vel_t, vel_i, axis=0),
				np.append(m_t, m_i),
				np.append(h_t, h_i),
				np.append(rho_t, rho_i),
				np.append(P_t, P_i),
				np.append(u_t, u_i),
				np.append(matid_t, matid_i),
				boxsize=boxsize,
				file_to_SI=woma.Conversions(M_earth, R_earth, 1),
				dark_matter=moon_flag,
			)
		else:
			custom_woma.save_particle_data(
				f,
				pos_t,
				vel_t,
				m_t,
				h_t,
				rho_t,
				P_t,
				u_t,
				matid_t,
				boxsize=boxsize,
				file_to_SI=woma.Conversions(M_earth, R_earth, 1),
				dark_matter=moon_flag,
			)

		if moon_flag:
			# Add in the dark matter moon
			id = np.array([len(pos_t) + len(pos_i)])  # Shouldn't clash...
			m = np.array([M_moonlet])
			#u = np.array([7e-8])  # Really not sure what this should be. Copied rough value from particles.
			#smoothing = np.array([0.03])  # Really not sure what this should be. Copied rough value from particles.
		
			Di_hdf5_particle_label = {  # Type
			"pos": "Coordinates",  # d
			"vel": "Velocities",  # f
			"m": "Masses",  # f
			"h": "SmoothingLengths",  # f
			"u": "InternalEnergies",  # f
			"rho": "Densities",  # f
			"P": "Pressures",  # f
			"s": "Entropies",  # f
			"id": "ParticleIDs",  # L
			"mat_id": "MaterialIDs",  # i
			"phi": "Potentials",  # f
			"T": "Temperatures",  # f
			}

			# Particles
			grp = f.create_group("/PartType1")
			grp.create_dataset(Di_hdf5_particle_label["pos"], data=pos_m / R_earth, dtype="d")
			grp.create_dataset(Di_hdf5_particle_label["vel"], data=vel_m / R_earth, dtype="f")
			grp.create_dataset(Di_hdf5_particle_label["m"], data=m / M_earth, dtype="f")
			grp.create_dataset(Di_hdf5_particle_label["id"], data=id, dtype="L")

if __name__ == "__main__":
	main(sys.argv)