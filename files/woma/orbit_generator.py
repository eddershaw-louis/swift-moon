import womaplotting
import woma
import h5py
import numpy as np
import os
import glob
import swiftsimio as sw
import unyt
import sys

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




sample_size = 10000

## Impactor mass distribution (in units of M_earth)
mass_histogram, mass_bin_edges = np.histogram(np.random.default_rng().normal(0.0, 1.0, size=sample_size), bins=150)

## Impactor velocity distribution (in units of v_esc)
vel_histogram, vel_bin_edges = np.histogram(np.random.default_rng().exponential(scale=1.0, size=sample_size), bins=150)

## Impact ANGLE (in units of degrees)
B_histogram, B_bin_edges = np.histogram(np.random.default_rng().normal(0, 90, size=sample_size), bins=150)

## Lunar orbital distance (in units of R_earth)
lunar_orbit_histogram, lunar_orbit_bin_edges = np.histogram(np.random.random(size=sample_size) * 100 + 50, bins=150)

## Lunar starting angle (in units of degrees, angle measured from the x axis initially in the positive y direction)
theta_histogram, theta_bin_edges = np.histogram(np.random.random(size=sample_size) * 360, bins=150)

distributions = [[mass_histogram, mass_bin_edges], [vel_histogram, vel_bin_edges], [B_histogram, B_bin_edges], [lunar_orbit_histogram, lunar_orbit_bin_edges], [theta_histogram, theta_bin_edges]]

		    ## Test parameters		    ## Sample the parameter distributions
impact_parameters = (0.013, 2.3, 41.0, 43.0, 172.0) #sample_distributions(distributions)
print(impact_parameters)

impactor_mass = impact_parameters[0]
impact_velocity = impact_parameters[1]
impact_angle = impact_parameters[2]
lunar_orbit = impact_parameters[3]

lunar_mass = 1
## Consider a Moon or not (setting Moon mass to 0 will ignore Moon)
moon_flag = lunar_mass != 0


impact_approach_angle = impact_parameters[4]

boxsize = 500 * R_earth

#help(woma.Conversions(M_earth, R_earth, 1))
unit_converter = woma.Conversions(M_earth, R_earth, 1)

## Reads ROTATING Earth particle file
earth_file = "rotating_earth_7.5hr.hdf5"
if "rotating" in earth_file: rotating = True

with h5py.File(f"impact_files/earths/{earth_file}", "r") as f:
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
	M_m = lunar_mass * M_moon
	impact_pos_m = np.asarray([np.cos(impact_approach_angle * np.pi/180) * lunar_orbit * R_earth, np.sin(impact_approach_angle * np.pi/180) * lunar_orbit * R_earth, 0.0])

	orbital_speed = np.sqrt(G * M_t / np.linalg.norm(impact_pos_m - impact_pos_t))
	impact_vel_m = np.array([np.cos(impact_approach_angle * np.pi/180 + np.pi/2) * orbital_speed, np.sin(impact_approach_angle * np.pi/180 + np.pi/2) * orbital_speed, 0.0])

# Centre of mass
if moon_flag:
	impact_pos_com = (M_t * impact_pos_t + M_i * impact_pos_i + M_m * impact_pos_m) / (M_t + M_i + M_m)
else:
	impact_pos_com = (M_t * impact_pos_t + M_i * impact_pos_i) / (M_t + M_i)


## Offset bodies such that the origin is the centre of mass
impact_pos_t -= impact_pos_com
impact_pos_i -= impact_pos_com
if moon_flag: impact_pos_m -= impact_pos_com


# Centre of momentum
if moon_flag:
	impact_vel_com = (M_t * impact_vel_t + M_i * impact_vel_i + M_m * impact_vel_m) / (M_t + M_i + M_m)
else:
	impact_vel_com = (M_t * impact_vel_t + M_i * impact_vel_i) / (M_t + M_i)

## Offset the velocities of the bodies such that the origin is the centre of momentum
impact_vel_t -= impact_vel_com
impact_vel_i -= impact_vel_com
if moon_flag: impact_vel_m -= impact_vel_com


print("New Target Position:")
print(impact_pos_t / R_earth, "R_earth")
print("New Target Velocity:")
print(impact_vel_t, "m/s")

print("\nNew Impactor Position:")
print(impact_pos_i / R_earth, "R_earth")
print("New Impactor Velocity:")
print(impact_vel_i, "m/s")
print(np.linalg.norm(impact_vel_i), "m/s\n")

if moon_flag:
	print("\nMoon Position:")
	print(impact_pos_m / R_earth, "R_earth")
	print("Moon Velocity:")
	print(impact_vel_m, "m/s")

print("Centre of momentum", impact_vel_com, "m/s")
print(np.linalg.norm(impact_vel_com), "m/s\n")


print("Impactor Approach Velocity Relative to Target:")
print(np.linalg.norm(impact_vel_i) - np.linalg.norm(impact_vel_com), "m/s\n\n")

## Offset each bodies' particles' position and velocities to the correct values
pos_t += impact_pos_t
vel_t[:] += impact_vel_t

pos_i += impact_pos_i
vel_i[:] += impact_vel_i

if moon_flag:
	pos_m = np.array([[np.cos(impact_approach_angle * np.pi/180) * lunar_orbit * R_earth, np.sin(impact_approach_angle * np.pi/180) * lunar_orbit * R_earth, 0.0]])
	pos_m += boxsize / 2.0

	vel_m = np.array([impact_vel_m])
print()


## Save the impact configuration
filename = "impact"
if rotating: filename += "_7.5hr"
filename += "_{0:.3f}M_{1:.3f}M_b{3:.1f}_v{2:.1f}".format(M_t / M_earth, M_i / M_earth, impact_velocity, impact_angle)
if moon_flag: filename += "_moon_{0:.3f}LM_d{1:.1f}_a{2:.1f}".format(lunar_mass, lunar_orbit, impact_approach_angle)

with h5py.File(f"impact_files/{filename}.hdf5", "w") as f:	
	woma.save_particle_data(
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

	if moon_flag:
		# Add in the dark matter moon
		id = np.array([len(pos_t) + len(pos_i)])  # Shouldn't clash...
		m = np.array([lunar_mass * M_moon])
		u = np.array([7e-8])  # Really not sure what this should be. Copied rough value from particles.
		smoothing = np.array([0.03])  # Really not sure what this should be. Copied rough value from particles.
		
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