import matplotlib
import matplotlib.pyplot as plt
import numpy as np

import glob
import sys
import os
import math
import h5py

from PIL import Image

from custom_functions import material_colour_map
from custom_functions import snapshot

font_size = 24
params = {
    "axes.labelsize": 0.85 * font_size,
    "font.size": font_size,
    "xtick.labelsize": 0.7 * font_size,
    "ytick.labelsize": 0.7 * font_size,
    "font.family": "serif",
}
plt.style.use('dark_background')
matplotlib.rcParams.update(params)

R_earth = 6.371e6   # m
M_earth = 5.9724e24  # kg

def concatenate_images(front=False, side=False, all_around=False):
	## Open each image to be combined
	top_down_image = Image.open("moon_simulations/{0}/xy/{0}_{1}.png".format(simulation_name, str(snapshots[snapshot_id][-9:-5]).zfill(snapshot_number_sig_fig)))
	
	if front or all_around:
		front_on_image = Image.open("moon_simulations/{0}/xz/{0}_{1}.png".format(simulation_name, str(snapshots[snapshot_id][-9:-5]).zfill(snapshot_number_sig_fig)))		
	if side or all_around:
		side_on_image = Image.open("moon_simulations/{0}/yz/{0}_{1}.png".format(simulation_name, str(snapshots[snapshot_id][-9:-5]).zfill(snapshot_number_sig_fig)))	
	

	top_down_image_size = top_down_image.size
	
	if front or all_around:
		front_on_image_size = front_on_image.size
		## Resize the front image to be the same as the top image
		front_on_image = front_on_image.resize((top_down_image_size[0], top_down_image_size[1]))
	if side or all_around:
		side_on_image_size = side_on_image.size
		## Resize the side image to be the same as the top image
		side_on_image = side_on_image.resize((top_down_image_size[0], top_down_image_size[1]))

	
	
	if front or side:
		width_factor = 2
	if all_around:
		width_factor = 3

	## Create a new image and add the two images to it
	combined_image = Image.new('RGB',(width_factor * top_down_image_size[0], top_down_image_size[1]), (250, 250, 250))
	combined_image.paste(top_down_image, (0, 0))
	if front or all_around:
		combined_image.paste(front_on_image, (top_down_image_size[0], 0))
	if side:
		combined_image.paste(side_on_image, (top_down_image_size[0], 0))
	if all_around:
		combined_image.paste(side_on_image, (2 * top_down_image_size[0], 0))


	## Save the combined image
	if front:
		return
		#combined_output_name = "moon_simulations/{0}/xy_xz/{0}_{1}.png".format(simulation_name, str(snapshots[snapshot_id][-9:-5]).zfill(snapshot_number_sig_fig))
	if side:
		combined_output_name = "moon_simulations/{0}/xy_yz/{0}_{1}.png".format(simulation_name, str(snapshots[snapshot_id][-9:-5]).zfill(snapshot_number_sig_fig))
	if all_around:
		combined_output_name = "moon_simulations/{0}/xy_xz_yz/{0}_{1}.png".format(simulation_name, str(snapshots[snapshot_id][-9:-5]).zfill(snapshot_number_sig_fig))


	combined_image.save(combined_output_name, "PNG")

	print("Saved", combined_output_name)

##
##
## Get the name of the simulation
if (len(sys.argv) > 1):
	simulation_name = sys.argv[1]
else:
	simulation_name = input("Enter the name of the simulation: ")
##
##
##
##
##
## Clean up the simulation name if the user exploited the Linux filename autocomplete thing to enter the simulation name
if "simulations/" in simulation_name:
	simulation_name = simulation_name[12:]
	if simulation_name[-1] == "/":
		simulation_name = simulation_name[:-1]
##
##
##
##
## Get the axis limits to plot the snapshots with
## The user can also request auto limits which will ask the snapshot reading function to determine a suitable range
assumed_ax_lim = False
if ('-a' in sys.argv):
	ax_lim = float(sys.argv[sys.argv.index('-a') + 1])
else:
	ax_lim_input = ""#input("Axis limit (R_e): ")
	if ax_lim_input != "":
		ax_lim = float(ax_lim_input)
	else:
		ax_lim = -1
		assumed_ax_lim = True
##
##
##
##
##
## Get the number of particles that are in the target body, which is used to colour the two bodies differently
if ('-n' in sys.argv):
	num_target_particles = int(sys.argv[sys.argv.index('-n') + 1])
else:
	num_target_particles = -1
	#num_target_particles_input = input("Number of target particles: ")
	#if num_target_particles_input != "":
	#	num_target_particles = int(num_target_particles_input)
	#else:
	#	num_target_particles = -1
##
##
##
##
##
## The user can ask the program to find snapshots that haven't been plotted yet rather than explicitly working out which snapshot to start with
find_recent = False
if ('-r' in sys.argv):
	find_recent = True
	sys.argv.remove('-r')
##
##
##
##
##
## The user can specify a specific snapshot number to start with; skipping those before it
if ('-s' in sys.argv):
	start_index = int(sys.argv[sys.argv.index('-s') + 1])
else:
	start_index = 0
##
##
##
##
##
## The user can specify a specific snapshot number to end before; skipping those after it
if ('-e' in sys.argv):
	end_index = int(sys.argv[sys.argv.index('-e') + 1])
else:
	end_index = -1
##
##
##
##
##
## The user can specify the timestep between snapshots, which is used to display timing information on the plots
if ('-t' in sys.argv):
	time_step = float(sys.argv[sys.argv.index('-t') + 1])
else:
	time_step = -1
##
##
##
##
##
## The user can specify a time of impact, which is then used to offset times displayed on the plots to match
if ('-T0' in sys.argv):
	impact_time_shift = float(sys.argv[sys.argv.index('-T0') + 1])
else:
	impact_time_shift = 0.0
##
##
##
##
##
## If the simulation is actually a continuation of another one, the time doesn't start at 0 but rather at some specified time
if ('-start' in sys.argv):
	simulation_time_shift = float(sys.argv[sys.argv.index('-start') + 1])
else:
	simulation_time_shift = 0.0
##
##
##
##
##
## The user can request the program to generate final report "worthy" images 
if ('-f' in sys.argv):
	final_output = True
else:
	final_output = False
##
##
##
##
##
## The user can request for the x axis labels and tick marks to be hidden which is useful when combining images later on
if ('-cleanX' in sys.argv):
	x_axes = False
else:
	x_axes = True
##
##
##
##
##
## The user can request for the y axis labels and tick marks to be hidden which is useful when combining images later on
if ('-cleanY' in sys.argv):
	y_axes = False
else:
	y_axes = True
##
##
##
##
## The user can request for the z axis labels and tick marks to be hidden which is useful when combining images later on
## This was implemented when 3D plotting was attempted, but this was later removed in favour of multiple view 2D images instead (as of 07.02.2024)
if ('-cleanZ' in sys.argv):
	z_axes = False
else:
	z_axes = True
##
##
##
##
##
## The user can request the xy view of the simulation
if ('-top' in sys.argv):
	top = True
else:
	top = False
##
##
##
##
##
## The user can request the xz view of the simulation as well as the normal xy view
if ('-front' in sys.argv):
	front = True
else:
	front = False
##
##
##
##
##
## The user can request the yz view of the simulation as well as the normal xy view
if ('-side' in sys.argv):
	side = True
else:
	side = False
if not (front or side): top = True
##
##
##
##
## The user can request that a single particle be tracked across frames. Here, the index of a particle in the position array is specified since the user might not know what the particle IDs are yet
## The program will then tell the user what particle ID the particle at this index has, which is useful for consistent tracking later on
if ('-tindex' in sys.argv):
	tracked_index = int(sys.argv[sys.argv.index('-tindex') + 1])
else:
	tracked_index = -1
##
##
##
##
##
## The user can request a single particle be tracked across frames. Here, the particle ID is known by the user and so they can provide it.
if ('-tid' in sys.argv):
	tracked_id = int(sys.argv[sys.argv.index('-tid') + 1])
else:
	tracked_id = -1
##
##
##
##
##
## The user can request a specific dpi quality of the output images if they so desire
if ('-dpi' in sys.argv):
	dpi = int(sys.argv[sys.argv.index('-dpi') + 1])
else:
	dpi = 125
##
##
##
##
##
## The user can request a freezeframe to be generated at the start (usually to show the initial angular momentum vector which should be provided)
if ('-ffs' in sys.argv):
	freezeframe_start = True
	freezeframe_start_index = sys.argv.index('-ffs')
	#freezeframe_start_origin_vector = np.asarray(sys.argv[freezeframe_start_index + 1: freezeframe_start_index + 4], dtype=float)
	freezeframe_start_vector = np.asarray(sys.argv[freezeframe_start_index + 1: freezeframe_start_index + 4], dtype=float)
else:
	freezeframe_start = False
##
##
##
##
##
## The user can request a freezeframe to be generated at the end (ususally to show the final angular momentum vector which should be provided)
if ('-ffe' in sys.argv):
	freezeframe_end = True
	freezeframe_end_index = sys.argv.index('-ffe')
	#freezeframe_end_origin_vector = np.asarray(sys.argv[freezeframe_end_index + 1: freezeframe_end_index + 4], dtype=float)
	freezeframe_end_vector = np.asarray(sys.argv[freezeframe_end_index + 1: freezeframe_end_index + 4], dtype=float)
else:
	freezeframe_end = False
##
##
##
##
##
##g
if ('-ref' in sys.argv):
	reference_vector = True
	reference_vector_index = sys.argv.index('-ref')
	
	reference_vectors = np.asarray(sys.argv[reference_vector_index + 1: reference_vector_index + 4], dtype=float)
else:
	reference_vector = False
##
##
##
##
##
## The user can request to not see the cross section and instead just view the "full" planet from each angle they would like
if ('-full' in sys.argv):
	full = True
else:
	full = False 
##
##
##
#if ('-ndm' in sys.argv):
	#dark_matter = 
#else:
dark_matter = False
##################################################################################################



## Where the simulation files are located
snapshot_path = "/data/cluster4/hd20558/moon_simulations/{0}/output/{0}_*.hdf5".format(simulation_name)

## Before generating the output directory, if the user is wanting final report worthy images we might want to save these in their own folder
if final_output:
	extra_directory = "/final"
else:
	extra_directory = ""

## Generate the path to the image output directory
output_path = "moon_simulations/{0}{1}".format(simulation_name, extra_directory)

## Get the list of snapshots that exist for this simulation
snapshots = glob.glob(snapshot_path)
snapshots.sort()
#print(snapshots)

## Work out how many files there are and work out a good length for the snapshot number added to each image
## (Linux orders numbers alphabetically so 0100 is actually "less than" 10, and this should help fix that)
num_of_snapshots = len(snapshots)
oom = int(math.floor(math.log10(num_of_snapshots)))
snapshot_number_sig_fig = max(5, oom + 2)


## Generate the folders needed in the output directory
if not os.path.exists(output_path):
	os.makedirs(output_path)
if not os.path.exists("{0}{1}".format(output_path, "/xy")):
	os.makedirs("{0}{1}".format(output_path, "/xy"))
if front:
	if not os.path.exists("{0}{1}".format(output_path, "/xz")):
		os.makedirs("{0}{1}".format(output_path, "/xz"))
	if not os.path.exists("{0}{1}".format(output_path, "/xy_xz")):
		os.makedirs("{0}{1}".format(output_path, "/xy_xz"))
if side:
	if not os.path.exists("{0}{1}".format(output_path, "/yz")):
		os.makedirs("{0}{1}".format(output_path, "/yz"))
	if not os.path.exists("{0}{1}".format(output_path, "/xy_yz")):
		os.makedirs("{0}{1}".format(output_path, "/xy_yz"))
if front and side:
	if not os.path.exists("{0}{1}".format(output_path, "/xy_xz_yz")):
		os.makedirs("{0}{1}".format(output_path, "/xy_xz_yz"))
#if not os.path.exists("{0}{1}".format(output_path, "/3d")):
#	os.makedirs("{0}{1}".format(output_path, "/3d"))
if not os.path.exists("analysis/simulations/{0}".format(simulation_name)):
	os.makedirs("analysis/simulations/{0}".format(simulation_name))

## If the user wants to start from the first snapshot still yet to be plotted then work out how many have been drawn so far and start from there
if find_recent:
	if front:
		images = glob.glob("{0}/xz/*.png".format(output_path))
	elif side:
		images = glob.glob("{0}/yz/*.png".format(output_path))
	else:
		images = glob.glob("{0}/xy/*.png".format(output_path))
	start_index = len(images)

## If the user hasn't specified at what index to end then we'll just keep going to the end
if end_index == -1: end_index = num_of_snapshots

#print(start_index, end_index)
## Start looping over all snapshots and plotting them
for snapshot_id in range(start_index, end_index):

	## Calculate the time stamp of this snapshot based off the time between snapshots, the snapshot number, the impact time, and starting time
	## If some of these aren't provided then it won't matter
	if time_step != -1:
		length_of_filepath_to_snapshot_number = len("/data/cluster4/hd20558/moon_simulations/{0}/output/{0}_".format(simulation_name))
		snapshot_number = int(snapshots[snapshot_id][length_of_filepath_to_snapshot_number:-5])
		#print(snapshot_number)
		adjusted_time_step = (time_step * snapshot_number) - impact_time_shift + simulation_time_shift
	else:
		adjusted_time_step = "undefined"

	## If this is a report worthy image then make that clear in the filename
	## (It'll already be in a separate folder but still its nice to make it clear)
	if final_output:
		additional_text = "_final"
	else:
		additional_text = ""

	## Read the snapshot data
	#print(snapshots[snapshot_id])
	with h5py.File(snapshots[snapshot_id], "r") as snapshot_file:
		##
		##
		##
		## QUANTITIES ARE GIVEN IN UNITS OF R_EARTH, M_EARTH, ETC.
		##
		##
		##

		boxsize = snapshot_file["Header"].attrs["BoxSize"][0]

		master_pos = snapshot_file["PartType0/Coordinates"][:]
		master_pos -= boxsize / 2.0

		master_vel = snapshot_file["PartType0/Velocities"][:]

		master_m = snapshot_file["PartType0/Masses"][:]

		master_mat_id = snapshot_file["PartType0/MaterialIDs"][:]
		master_id = snapshot_file["PartType0/ParticleIDs"][:]

			
		if snapshot_file["Header"].attrs["NumPart_Total"][1] != 0:
			dark_matter = True
			print("Reading DM particle data")
			master_dm_pos = snapshot_file["PartType1/Coordinates"][0]
			master_dm_pos -= boxsize / 2.0
			
			master_dm_vel = snapshot_file["PartType1/Velocities"][0]

			master_dm_m = snapshot_file["PartType1/Masses"][0]
			master_dm_id = snapshot_file["PartType1/ParticleIDs"][0]

 
		#master_A1_h_Earth = snapshot_file["PartType0/SmoothingLengths"][:]
		#master_A1_rho_Earth = snapshot_file["PartType0/Densities"][:]
		#master_A1_P_Earth = snapshot_file["PartType0/Pressures"][:]
		#master_A1_u_Earth = snapshot_file["PartType0/InternalEnergies"][:]

	## If the user wanted auto limits then tell them what the limits are so they can object if need be!
	if assumed_ax_lim == True:
		if dark_matter:
			maximum = max([master_pos.max(), master_dm_pos.max()])
			minimum = min([master_pos.min(), master_dm_pos.min()])
		else:
			maximum = master_pos.max()
			minimum = master_pos.min()

		ax_lim = max([abs(maximum), abs(minimum)]) * 1.05
		print("Assuming axis limits of +- {0} R_e".format(ax_lim))
		assumed_ax_lim = False
		

	## Adjust the positions such that Earth is centred at the origin
	pos_r = np.sqrt(np.sum(master_pos**2, axis=1))
	#print(pos_r)
	com_truncate_mask = np.where(pos_r <= 25)[0]
	temp_pos = master_pos[com_truncate_mask]
	#print(np.max(temp_pos))
	temp_m = master_m[com_truncate_mask]
	com = np.sum(temp_m[:, np.newaxis] * temp_pos, axis=0) / (np.sum(temp_m))
	master_pos -= com

	if tracked_id == -1 and tracked_index != -1:
		tracked_id = master_id[tracked_index]
		print(f"Tracked particle at index {tracked_index} has id {tracked_id}")

	if top:
		mask = np.where(master_pos[:, 2] < 0)[0]
		pos = master_pos[mask]
		vel = master_vel[mask]
		m = master_m[mask]
		mat_id = master_mat_id[mask]
		id = master_id[mask]

		sort = np.argsort(pos[:, 2])
		pos = pos[sort]
		vel = vel[sort]
		m = m[sort]
		id = id[sort]
		mat_id = mat_id[sort]



		## plot the xy frame of this snapshot
		snapshot.plot_pos_mat_id(pos, mat_id, id, ax_lim, simulation_name, snapshot_id, adjusted_time_step, final_output, x_axes, y_axes, tracked_id)

		if dark_matter:
			plt.scatter(master_dm_pos[0], master_dm_pos[1], c="seagreen", edgecolors='black', marker=".", linewidth=0.0, s=30, alpha=0.75)
		
		## Save the image to the correct place
		output_name = "{0}/xy/{1}_{2}{3}.png".format(output_path, simulation_name, str(snapshots[snapshot_id][-9:-5]).zfill(snapshot_number_sig_fig), additional_text)
		plt.savefig(output_name, dpi=dpi)
		plt.close()	
		print("Saved", output_name)


		## If the user was tracking a particle by its index then tell them what the ID of that particle is so that they can use it elsewhere
		## We only need to tell them this on the first go round though
		## (Like if they want to have multiple instances of this code running concurrently i.e. in parallel)
		#if snapshot_id == start_index:
		#	if tracked_index != -1:
		#		print("Tracked particle with index {0} has ID {1}".format(tracked_index, tracked_id))
		#		with open("{0}/{1}_tracked_info.txt".format(output_path, simulation_name), "w") as writer:
		#			writer.write("Tracked index: {0}\nTracked ID: {1}".format(tracked_index, tracked_id))
		#		tracked_index = -1


	## If the user also wanted the front (xz) view then we need to rinse and repeat for that too
	if front:
		
		#mask = np.where(master_pos[:, 1] > 0)[0]
		pos = master_pos#[mask]
		vel = master_vel#[mask]
		m = master_m#[mask]
		mat_id = master_mat_id#[mask]
		id = master_id#[mask]

		sort = np.argsort(pos[:, 1])
		pos = pos[sort]
		vel = vel[sort]
		m = m[sort]
		id = id[sort]
		mat_id = mat_id[sort]


		if False:
			if snapshot_id == 0 and freezeframe_start:
				temp_ff_origin_vector = [-float(com[0]), -float(com[1]), -float(com[2])]

				## Plot and save the side (xz) view
				snapshot.plot_pos_mat_id(pos, mat_id, id, ax_lim, simulation_name, snapshot_id, adjusted_time_step, final_output, x_axes, y_axes, tracked_id, side="front", ff_origin_vector=temp_ff_origin_vector, ff_vector=freezeframe_start_vector, ff_colour="red")
				
			elif snapshot_id == num_of_snapshots - 1 and freezeframe_end:
				temp_ff_origin_vector = [-float(com[0]), -float(com[1]), -float(com[2])]

				## Plot and save the side (xz) view
				snapshot.plot_pos_mat_id(pos, mat_id, id, ax_lim, simulation_name, snapshot_id, adjusted_time_step, final_output, x_axes, y_axes, tracked_id, side="front", ff_origin_vector=temp_ff_origin_vector, ff_vector=freezeframe_end_vector, ff_colour="red")
			else:
				## Plot and save the side (yz) view
				snapshot.plot_pos_mat_id(pos, mat_id, id, ax_lim, simulation_name, snapshot_id, adjusted_time_step, final_output, x_axes, y_axes, tracked_id, side="front")
		else:
			## Plot and save the front (xz) view
			snapshot.plot_pos_mat_id(pos, mat_id, id, ax_lim, simulation_name, snapshot_id, adjusted_time_step, final_output, x_axes, y_axes, tracked_id, side="front")
		
		if dark_matter:
			plt.scatter(master_dm_pos[0], master_dm_pos[2], c="seagreen", edgecolors='black', marker=".", linewidth=0.0, s=30, alpha=0.75)
		

		## Save the image to the correct place
		output_name = "{0}/xz/{1}_{2}{3}.png".format(output_path, simulation_name, str(snapshots[snapshot_id][-9:-5]).zfill(snapshot_number_sig_fig), additional_text)
		plt.savefig(output_name, dpi=dpi)
		plt.close()	
		print("Saved", output_name)


		## If the user was tracking a particle by its index then tell them what the ID of that particle is so that they can use it elsewhere
		## We only need to tell them this on the first go round though
		## (Like if they want to have multiple instances of this code running concurrently i.e. in parallel)
		#if snapshot_id == start_index:
		#	if tracked_index != -1:
		#		print("Tracked particle with index {0} has ID {1}".format(tracked_index, tracked_id))
		#		with open("{0}/{1}_tracked_info.txt".format(output_path, simulation_name), "w") as writer:
		#			writer.write("Tracked index: {0}\nTracked ID: {1}".format(tracked_index, tracked_id))
		#		tracked_index = -1

		

	## If the user also wanted the side (yz) view then we need to rinse and repeat for that too
	if side:
		#mask = np.where(master_pos[:, 0] > 0)[0]
		pos = master_pos#[mask]
		vel = master_vel#[mask]
		m = master_m#[mask]
		mat_id = master_mat_id#[mask]
		id = master_id#[mask]

		sort = np.argsort(pos[:, 0])
		pos = pos[sort]
		vel = vel[sort]
		m = m[sort]
		id = id[sort]
		mat_id = mat_id[sort]


		if snapshot_id == 0 and freezeframe_start:
			## Plot and save the side (yz) view
			snapshot.plot_pos_mat_id(pos, mat_id, id, ax_lim, simulation_name, snapshot_id, adjusted_time_step, final_output, x_axes, y_axes, tracked_id, side="side", ff_origin_vector=com, ff_vector=freezeframe_start_vector, ff_colour="red")
			
		elif snapshot_id == num_of_snapshots - 1 and freezeframe_end:
			## Plot and save the side (yz) view
			snapshot.plot_pos_mat_id(pos, mat_id, id, ax_lim, simulation_name, snapshot_id, adjusted_time_step, final_output, x_axes, y_axes, tracked_id, side="side", ff_origin_vector=com, ff_vector=freezeframe_end_vector, ff_colour="red")
		else:
			if False:
				pos_r = np.sqrt(np.sum(pos**2, axis=1)) / R_earth
				mask = np.where((mat_id == 304) | (mat_id == 200000304))[0]
				L = np.sum(np.cross(pos[mask], vel[mask]) * m[mask, np.newaxis], axis=0)
				L_norm = L / np.linalg.norm(L)

				ref_vecs = np.concatenate((reference_vectors, L_norm)) #reference_vectors

				if True:
					mask = np.where((mat_id == 200))[0]
					L = np.sum(np.cross(pos[mask], vel[mask]) * m[mask, np.newaxis], axis=0)
					L_norm = L / np.linalg.norm(L)

					ref_vecs = np.concatenate((ref_vecs, L_norm)) #reference_vectors

					mask = np.where((mat_id == 400) | (mat_id == 200000400))[0]
					L = np.sum(np.cross(pos[mask], vel[mask]) * m[mask, np.newaxis], axis=0)
					L_norm = L / np.linalg.norm(L)

					ref_vecs = np.concatenate((ref_vecs, L_norm)) #reference_vectors


			else:
				pass
				#ref_vecs = reference_vectors
			

			## Plot and save the side (yz) view
			snapshot.plot_pos_mat_id(pos, mat_id, id, ax_lim, simulation_name, snapshot_id, adjusted_time_step, final_output, x_axes, y_axes, tracked_id, side="side")#, reference_vectors=ref_vecs, origin_vector=com)

		if dark_matter:
			plt.scatter(master_dm_pos[2], master_dm_pos[1], c="seagreen", edgecolors='black', marker=".", linewidth=0.0, s=30, alpha=0.75)
		

		## Save the image to the correct place
		output_name = "{0}/yz/{1}_{2}{3}.png".format(output_path, simulation_name, str(snapshots[snapshot_id][-9:-5]).zfill(snapshot_number_sig_fig), additional_text)
		plt.savefig(output_name, dpi=dpi)
		plt.close()	
		print("Saved", output_name)

		## If the user was tracking a particle by its index then tell them what the ID of that particle is so that they can use it elsewhere
		## We only need to tell them this on the first go round though
		## (Like if they want to have multiple instances of this code running concurrently i.e. in parallel)
		#if snapshot_id == start_index:
		#	if tracked_index != -1:
		#		print("Tracked particle with index {0} has ID {1}".format(tracked_index, tracked_id))
		#		with open("{0}/{1}_tracked_info.txt".format(output_path, simulation_name), "w") as writer:
		#			writer.write("Tracked index: {0}\nTracked ID: {1}".format(tracked_index, tracked_id))
		#		tracked_index = -1

	## If the user wanted a front (xz) view then they also probably want to see an image that combines the top (xy) and front view
	if front:
		#pass
		#concatenate_images(front=True)
		front_exists = True

	## If they didn't want a front (xz) view, but these images exist, then we might want to update the combined triple view (if the side ones exist too) to reflect the new images
	elif len(glob.glob('moon_simulations/{0}/xz/*.png'.format(simulation_name))) > 0:
		front_exists = True
	else:
		front_exists = False


	## If the user wanted a side (yz) view then they also probably want to see an image that combines the top (xy) and side view
	if side:
		concatenate_images(side=True)
		side_exists = True

	## If they didn't want a side (yz) view, but these images exist, then we might want to update the combined triple view (if the front ones exist too) to reflect the new images
	elif len(glob.glob('moon_simulations/{0}/yz/*.png'.format(simulation_name))) > 0:
		side_exists = True
	else:
		side_exists = False


	## Update the triple view if they all exist
	## (Top view will always be plotted so that will always be true if we get here)
	if front_exists and side_exists:
		concatenate_images(all_around=True)
		pass
	print()



