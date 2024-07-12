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

import womaplotting
import woma
import h5py
import numpy as np
import os
import swiftsimio as sw
import unyt
import sys
import math

sys.path.append("..")
sys.path.append("../..")

from custom_functions import custom_woma


R_earth = 6.371e6   # m
M_earth = 5.9724e24  # kg
G = 6.67408e-11  # m^3 kg^-1 s^-2


def check_ang_momentum(pos, vel, m, axis_just_rotated_around=False):
	##
	## Calculate the angular momentum (in the same way as angular_momentum.py in analysis) and return the angle of the vector from each of the z and y axes, as well as the magnitude 
	##


	L = np.sum(np.cross(pos, vel) * m[:, np.newaxis], axis=0)
	L_mag = np.sqrt(L[0]**2 + L[1]**2 + L[2]**2)


	L_ang_from_z = math.acos(np.sum(np.asarray([L[0], 0, L[2]]) * np.asarray([0, 0, 1])) / L_mag) * 180 / math.pi
	if (L[0] < 0):
		L_ang_from_z *= -1
		
	L_ang_from_y = math.acos(np.sum(L * np.asarray([0, 1, 0])) / L_mag) * 180 / math.pi

	if axis_just_rotated_around:
		print("\nAfter rotation about {0}:".format(axis_just_rotated_around))	
	else:
		print("\nPre-rotation angular momentum:".format(axis_just_rotated_around))	
	print(L, "kg m^2 s^-1")
	
	L_norm = L / L_mag
	print(L_norm)
		

	print(L_ang_from_z, "degrees from z axis as projected into the x,z plane (-ve angle means pointing towards -ve X)")
	print(L_ang_from_y, "degrees from y axis\n")


	if axis_just_rotated_around == False:
		## Calculate and print the angular frequency in radians per second, and the period in hours
		## I = mr^2
		print("\nAngular Frequency and Period")

		r2 = np.sum(pos**2, axis=1)	## This converts [[x_0, y_0, z_0], [x_1, y_1, z_1], ...] to [r_0^2, r_1^2, ...]
		I = np.sum(m * r2)	# The total moment of inertia of an N body system is given by the sum of the moments of inertia of each of the particles
		
		## The angular frequency is defined via the definition of the angular momentum (scalar), L = Iw
		angular_speed = L_mag / I

		period = 2 * math.pi / (angular_speed * 3600)

		print(angular_speed, "rad s^-1")
		print(period, "hours")

		#periods = []
		#distances = []
		#masses = []
		#for particle_index in range(len(pos)):
		#	test_ang_mom = m[particle_index] * np.cross(pos[particle_index], vel[particle_index])
		#	test_moi = m[particle_index] * r2[particle_index]
		#	test_ang_vel = np.sum(test_ang_mom / test_moi)
		#	test_period = 2 * math.pi / (test_ang_vel * 3600)
		#	periods.append(test_period)
		#	distances.append(np.sqrt(r2[particle_index]) / R_earth)
		#	masses.append(m[particle_index])
		
		#distances = np.asarray(distances)
		#periods = np.asarray(periods)

		#print(distances)
		#sort = np.argsort(distances)	
		#print(distances)
		#print(distances[sort])
		#print(periods[sort])

		#print(np.sum(masses * periods) / (len(periods)))

		
			
	return L_ang_from_z, L_ang_from_y, L_norm


def main(parameters):

	## Read in hdf5 filename
	if (parameters[0][-3:] == ".py" and len(parameters) > 1):
		target_filepath = parameters[1]
	elif len(parameters) > 0:
		target_filepath = parameters[0]
		print(target_filepath)
	else:
		target_filepath = input("Enter filepath to hdf5 file to spin up: ")

	target_filename = ""
	for i in range(1, len(target_filepath)):
		if target_filepath[-i] == "/":
			target_filename = target_filepath[-i + 1:-5]
			break

	if target_filename == "":
		raise Exception("Unable to find file!")

	if ('-o' in parameters):
		output_path = parameters[parameters.index('-o') + 1]
	else:
		output_path = None

	if ('-ow' in parameters):
		overwrite_output = True	
	else:
		overwrite_output = False

	## Read in requested period for spin up if given
	if ('-P' in parameters):
		spin_up = True
		rotation_period = float(parameters[parameters.index('-P') + 1])
		#increments = float(parameters[parameters.index('-P') + 2])
	else:
		spin_up = False
		#rotation_period = float(input("Enter period of rotation of planet (in hours): "))
		##
		##
		##
		##
		##
		## Omega_max for Earth is equivalent to period of 1.484543011 hr (estimated from Canup 2017)
		##
		##
		##
		##
		##

	## Read in requested rotation transformation angle if given
	if ('-R' in parameters):
		rotate = True
		rotation_angle = float(parameters[parameters.index('-R') + 1])
	else:
		rotate = False

	## Read in whether or not to correct the angular velocity vector of the entire system to be along the z axis
	if ('-C' in parameters):
		correct = True
	else:
		correct = False

	if not (spin_up or rotate or correction):
		raise Exception("Please provide -R or -P tag to rotate or spin up.")

	## Read in particles
	pos, vel, h, m, rho, p, u, mat_id, R = custom_woma.load_to_woma(
	    target_filepath
	)

	L_norm = None
	
	if correct:

		correction_z, correction_y, L_norm = check_ang_momentum(pos, vel, m)
	

		## Rotate about the y axis by the OPPOSITE of the angle provided, i.e. undo the rotation in the x,z plane
	
		correction_z = (-correction_z) * math.pi / 180

		if (correction_z < 0):
			direction = "clockwise"
		else:
			direction = "anti-clockwise"

		print("\nCorrecting {0} about the y axis by {1} degrees (axis pointing towards observer)".format(direction, abs(correction_z * 180 / math.pi)))
	
		R_y = np.asarray([	[	math.cos(correction_z),		0,	math.sin(correction_z)	],
					[	0,	 			1, 	0			],
					[	-math.sin(correction_z),	0,	math.cos(correction_z)	]
			])

		pos = (R_y @ pos.T).T
		vel = (R_y @ vel.T).T


		# By getting the correction angles again (technically correction_z is useless, we don't use it again), we get a better correction in y after the first rotation
		correction_z, correction_y, L_norm = check_ang_momentum(pos, vel, m, "y")





		## Rotate about the x axis by the OPPOSITE of the angle provided, i.e. push the rotation to 90 degrees from the y axis, such that the result is pointing along z

		correction_y = (90 - correction_y) * math.pi / 180

		if (correction_y < 0):
			direction = "clockwise"
		else:
			direction = "anti-clockwise"

		print("\nCorrecting {0} about the x axis by {1} degrees (axis pointing towards observer)".format(direction, abs(correction_y * 180 / math.pi)))

		R_x = np.asarray([	[	1,	0,			0			],
					[	0,	math.cos(correction_y), -math.sin(correction_y)	],
					[	0,	math.sin(correction_y),	math.cos(correction_y)	]
			])

		pos = (R_x @ pos.T).T
		vel = (R_x @ vel.T).T

		check_ang_momentum(pos, vel, m, "x")


	if spin_up:
		## We don't want to do hemispherical offsetting anymore so "if False" will make sure it is never run, but it's useful to see how it was done
		if False:
			## Hemispherical offsetting method

			mask_above_x_axis = np.where(pos[:, 1] >= 0, True, False)
			mask_below_x_axis = ~mask_above_x_axis

			## Calculate angular velocity of requested spin up
			ang_vel_scalar = 2 * math.pi / (rotation_period * 3600) 	#s^-1

			## Equatorial speed of a rigid body with this angular velocity
			rim_speed = ang_vel_scalar * R
			## Factor this down if incremental velocity kicks are desired
			hemisphere_speed_offset = rim_speed / increments

			## Give the velocity kick to the two hemispheres
			vel[mask_above_x_axis, 0] -= hemisphere_speed_offset
			vel[mask_below_x_axis, 0] += hemisphere_speed_offset
	
			print("Added a single velocity kick for a rotation of {0} hours in {1} increment(s)".format(rotation_period, increments))
		else:
			## Rigid-body-like rotation

			## Adjust the positions such that Uranus is centred at the origin

				## Calculate radial positions of the particles
			pos_r = np.sqrt(np.sum(pos**2, axis=1))

				## Compute centre of mass for particles within 25 Earth radii from origin
			com_truncate_mask = np.where(pos_r <= 25 * R_earth)[0]
			temp_pos = pos[com_truncate_mask]
			temp_m = m[com_truncate_mask]
			com = np.sum(temp_m[:, np.newaxis] * temp_pos, axis=0) / (np.sum(temp_m))
			com[-1] = 0

				## Offset particles by the centre of mass, so that the centre of mass is situated at the origin
			pos -= com


			## Calculate angular velocity of requested spin up
			period_min_earth = 1.484543011	#hr
			ang_vel_scalar = 2 * math.pi / (rotation_period * 3600) 	#s^-1


			## Calculate how far away each particle is from the z axis (i.e. the axis the spin up will be around)
			pos_vectors_projected_into_xy = np.copy(pos)
			pos_vectors_projected_into_xy[:, 2] = 0		
			pos_r_from_z = np.sqrt(np.sum(pos_vectors_projected_into_xy**2, axis=1))
		
			## The speed of each particle as it rotates is proportional to its distance from the z axis
			rotation_speeds = ang_vel_scalar * pos_r_from_z

			## Normalise the position vector, this is essentially returning the radial basis vector from the z axis of each particle
			normalised_projected_pos_vectors = pos_vectors_projected_into_xy / pos_r_from_z[:, np.newaxis]
			#print(normalised_projected_pos_vectors)
			#print()
		
			## Define a 90 degree rotation matrix around the z axis
			R_z_90 = np.asarray([	[0,	-1,	0],
						[1,	0,	0],
						[0,	0,	0]
				])

			## The azimuthal basis vector (parallel to the desired velocity vector of each particle in rigid body motion) is a 90 degree rotation of the radial basis vector of each particle
			new_velocity_unit_vectors = (R_z_90 @ normalised_projected_pos_vectors.T).T

			## The desired velocity vector of each particle is the azimuthal basis vector scaled to the required speed
			rotating_velocities = new_velocity_unit_vectors * rotation_speeds[:, np.newaxis]

			#print(new_velocity_unit_vectors)
			#print(rotating_velocities)
		
			## Give every particle their corresponding velocity they need to be in rotational motion
			vel += rotating_velocities

			## Print the angular momentum to sanity check
			check_ang_momentum(pos, vel, m)

		
	if rotate:
		## The rotation is going to be from the z axis towards the y axis, about x. This is a clockwise rotation, so negative angle for positive meaning anti-clockwise
		_rotation_angle = (-rotation_angle) * math.pi / 180

		print("\nRotating clockwise about the x axis by {0} degrees (axis pointing towards observer)".format(rotation_angle))

		## Define the rotation matrix around x by the required angle
		R_x = np.asarray([	[	1,	0,				0				],
					[	0,	math.cos(_rotation_angle),	-math.sin(_rotation_angle)	],
					[	0,	math.sin(_rotation_angle),	math.cos(_rotation_angle)	]
			])
	
		## Apply the matrix transformation to both position and velocity
		pos = (R_x @ pos.T).T
		vel = (R_x @ vel.T).T

		_, __, L_norm = check_ang_momentum(pos, vel, m, "x")

	

	


	suffix = ""
	if correct:
		suffix = "{0}-corrected".format(suffix)
	if spin_up:
		suffix = "{0}-spun_up-{1}".format(suffix, rotation_period)
	if rotate:
		suffix = "{0}-rotated-{1}".format(suffix, rotation_angle)

	

	if output_path == None:
		output_path = "post_spin_up_files"
	output_path += f"/{target_filename}{suffix}.hdf5"

	if overwrite_output:
		output_path = f"{target_filepath}"

	## Save newly rotating particles to an hdf5 file
	with h5py.File(output_path, "w") as f:
		custom_woma.save_particle_data(
			f,
			pos,
			vel,
			m,
			h,
			rho,
			p,
			u,
			mat_id,
			boxsize=500 * R_earth,
			file_to_SI=woma.Conversions(M_earth, R_earth, 1),
    	)

	## Print the calculated properties of this spin up to a text file for archival purposes (very, very useful)
	with open(f"{output_path[:-5]}_info.txt", "w") as writer:
		if correct:
			writer.write("Corrected.\n\n")
		if spin_up:
			writer.write("Spun up.\n")
			writer.write("Requested: {0} hours\n".format(rotation_period))
			#writer.write("In {0} increments\n\n".format(increments))
		if rotate:
			writer.write("Rotated.\n")
			writer.write("Rotated by {0} degrees\n".format(rotation_angle))
			writer.write("Angular momentum vector, normalised:\n")
			for i in L_norm:
				writer.write(str(i) + " ")
	print(f"Saved {output_path[:-5]}_info.txt!")

if __name__ == "__main__":
	main(sys.argv)