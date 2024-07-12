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

import swiftsimio as sw
import unyt
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
from custom_functions import material_colour_map

import math

R_e = 6.371e6 #m

def get_pos_vel_m_mat_id(snapshot_path, ax_lim, num_target_particles, mask_window_length=-1, tracked_index=-1, tracked_id=-1):
	""" Select and load the particles to plot. """

	# Only load data with the axis limits (if provided) and below z=0
	mask = sw.mask(snapshot_path)
	
	box_length = mask.metadata.boxsize[0].to(unyt.Rearth)
	box_mid = box_length / 2
	if ax_lim == -1: ax_lim = 1.5 * float(box_mid)

	if mask_window_length == -1:
		mask_min = box_mid - ax_lim * unyt.Rearth
		mask_max = box_mid + ax_lim * unyt.Rearth
	else:
		mask_min = box_mid - mask_window_length * unyt.Rearth
		mask_max = box_mid + mask_window_length * unyt.Rearth

	load_region = [[mask_min, mask_max], [mask_min, mask_max], [mask_min, box_mid]]
	mask.constrain_spatial(load_region)

	
	snapshot_data = sw.load(snapshot_path, mask=mask)
	pos = snapshot_data.gas.coordinates.to(unyt.Rearth) - box_mid
	snapshot_data.gas.velocities.convert_to_mks()
	vel = np.array(snapshot_data.gas.velocities) / unyt.Rearth
	snapshot_data.gas.masses.convert_to_mks()
	m = snapshot_data.gas.masses
	id = snapshot_data.gas.particle_ids
	mat_id = snapshot_data.gas.material_ids.value


	# Restrict to z < 0, plus selected particle to track
	position_mask = np.where(pos[:, 2] < 0)[0]
	if tracked_index != -1:
		tracked_mask = np.where(pos[:, 2] == pos[tracked_index, 2])[0]
		tracked_id = id[tracked_index]
		sel = np.append(position_mask, tracked_mask)
	elif tracked_id != -1:
		tracked_mask = np.where(id == tracked_id)
		sel = np.append(position_mask, tracked_mask)
	else:
		sel = position_mask
	pos = pos[sel]
	vel = vel[sel]
	m = m[sel]
	id = id[sel]
	mat_id = mat_id[sel]


	# Sort in z order so higher particles are plotted on top
	sort = np.argsort(pos[:, 2])
	pos = pos[sort]
	vel = vel[sort]
	m = m[sort]
	id = id[sort]
	mat_id = mat_id[sort]

	if num_target_particles != -1:
		# Edit material IDs for particles in the impactor
		mat_id[num_target_particles <= id] += material_colour_map.id_body

	return pos, vel, m, mat_id, ax_lim, id, tracked_id

def get_pos_vel_m_mat_id_side(snapshot_path, ax_lim, num_target_particles, side, mask_window_length=-1, tracked_index=-1, tracked_id=-1, full=False):
	""" Select and load the particles to plot. """

	# Only load data with the axis limits (if provided) and below z=0
	mask = sw.mask(snapshot_path)
	
	box_length = mask.metadata.boxsize[0].to(unyt.Rearth)
	box_mid = box_length / 2
	if ax_lim == -1: ax_lim = 1.5 * float(box_mid)

	if mask_window_length == -1:
		mask_min = box_mid - ax_lim * unyt.Rearth
		mask_max = box_mid + ax_lim * unyt.Rearth
	else:
		mask_min = box_mid - mask_window_length * unyt.Rearth
		mask_max = box_mid + mask_window_length * unyt.Rearth

	load_region = [[mask_min, mask_max], [mask_min, mask_max], [mask_min, box_mid]]
	mask.constrain_spatial(load_region)

	snapshot_data = sw.load(snapshot_path, mask=mask)
	pos = snapshot_data.gas.coordinates.to(unyt.Rearth) - box_mid
	snapshot_data.gas.velocities.convert_to_mks()
	vel = np.array(snapshot_data.gas.velocities) / unyt.Rearth
	snapshot_data.gas.masses.convert_to_mks()
	m = snapshot_data.gas.masses
	id = snapshot_data.gas.particle_ids
	mat_id = snapshot_data.gas.material_ids.value


	if side == "front":
		if full:
			# Restrict to y > 0
			#position_mask = np.where((pos[:, 1] > -1.5))[0]
			position_mask = np.where((pos[:, 1] == pos[:, 1]))[0]

		else:
			# Restrict to z < 0, y > 0
			######position_mask = np.where((pos[:, 2] < 0) & (pos[:, 1] > 0))[0]

			# Restrict to z < 0
			position_mask = np.where((pos[:, 2] < 0))[0]

	elif side == "side":
		if full:
			# Restrict to x > 0
			#position_mask = np.where((pos[:, 0] > -1.5))[0]
			position_mask = np.where((pos[:, 0] == pos[:, 0]))[0]
		else:
			# Restrict to z < 0, x < 0
			#######position_mask = np.where((pos[:, 2] < 0) & (pos[:, 0] < 0))[0]

			# Restrict to z < 0
			position_mask = np.where((pos[:, 2] < 0))[0]


	if tracked_index != -1:
		tracked_mask = np.where(pos[:, 2] == pos[tracked_index, 2])[0]
		tracked_id = id[tracked_index]
		sel = np.append(position_mask, tracked_mask)
	elif tracked_id != -1:
		tracked_mask = np.where(id == tracked_id)
		sel = np.append(position_mask, tracked_mask)
	else:
		sel = position_mask
	pos = pos[sel]
	m = m[sel]
	id = id[sel]
	mat_id = mat_id[sel]


	if side == "front":
		# Sort in y order so lower particles are plotted on top
		sort = np.argsort(-pos[:, 1])
	elif side == "side":
		# Sort in x order so higher particles are plotted on top
		sort = np.argsort(pos[:, 0])
	pos = pos[sort]
	m = m[sort]
	id = id[sort]
	mat_id = mat_id[sort]

	if num_target_particles != -1:
		# Edit material IDs for particles in the impactor
		mat_id[num_target_particles <= id] += material_colour_map.id_body

	return pos, vel, m, mat_id, ax_lim, id, tracked_id



def plot_pos_mat_id(pos, mat_id, id, ax_lim, simulation_name, snapshot_id, time_step="undefined", final_output=False, x_axes="undefined", y_axes="undefined", tracked_id=-1, side=False, origin_vector="undefined", ff_vector="undefined", ff_colour="undefined", reference_vectors=[]):
	""" Plot the particles, coloured by their material. """
	if final_output:
		final_factor = 2
	else:
		final_factor = 1
	plt.figure(figsize=(6 * final_factor, 6 * final_factor))
	ax = plt.gca()
	ax.set_aspect("equal")

	colour = np.empty(len(pos), dtype=object)
	for id_c, c in material_colour_map.ID_COLOUR_MAP.items():
		colour[mat_id == id_c] = c


	if side == "front":
		plot_x = pos[:, 0]
		plot_y = pos[:, 2]
	elif side == "side":
		plot_x = pos[:, 2]
		plot_y = pos[:, 1]
	else:
		plot_x = pos[:, 0]
		plot_y = pos[:, 1]

	ax.scatter(plot_x, plot_y, c=colour, edgecolors='black', marker=".", linewidth=0.0, s=10, alpha=0.3)

	if tracked_id != -1:
		tracked_mask = id == tracked_id
		if side == "front":
			tracked_x = pos[tracked_mask, 0]
			tracked_y = pos[tracked_mask, 2]
		elif side == "side":
			tracked_x = pos[tracked_mask, 2]
			tracked_y = pos[tracked_mask, 1]
		else:
			tracked_x = pos[tracked_mask, 0]
			tracked_y = pos[tracked_mask, 1]

		ax.scatter(tracked_x, tracked_y, c="red", edgecolors="none", marker=".", s=20, alpha=1)


	vector_scaling = ax_lim
	if ff_vector != "undefined":
		
		if side == "front":
			plotting_ff_origin_x = origin_vector[0]
			plotting_ff_origin_y = origin_vector[2]

			plotting_ff_dx = ff_vector[0] * vector_scaling
			plotting_ff_dy = ff_vector[2] * vector_scaling

			#plotting_ff = plotting_ff_origin + (np.asarray([ff_vector[0], ff_vector[2]]) * vector_scaling)
		elif side == "side":
			plotting_ff_origin_x = origin_vector[2]
			plotting_ff_origin_y = origin_vector[1]

			plotting_ff_dx = ff_vector[2] * vector_scaling
			plotting_ff_dy = ff_vector[1] * vector_scaling

			#plotting_ff = plotting_ff_origin + (np.asarray([ff_vector[2], ff_vector[1]]) * vector_scaling)
		

		print(plotting_ff_origin_x, plotting_ff_origin_y)
		print(plotting_ff_dx, plotting_ff_dy)

		plt.arrow(plotting_ff_origin_x, plotting_ff_origin_y, plotting_ff_dx, plotting_ff_dy, color=ff_colour)
		
	elif len(reference_vectors) > 0:
		#print(reference_vectors)
		num_reference_vectors = len(reference_vectors) // 3

		plotting_reference_vector_origin = [origin_vector[2], origin_vector[1]]
		colours = ["white", "red", "lightblue", "slategray"]

		for vec_num in range(num_reference_vectors):
			plotting_reference_vector = [reference_vectors[vec_num * 3 + 2] * vector_scaling, reference_vectors[vec_num * 3 + 1] * vector_scaling]
			
			
			plt.arrow(plotting_reference_vector_origin[0], plotting_reference_vector_origin[1], plotting_reference_vector[0], plotting_reference_vector[1], color=colours[vec_num], alpha=0.75, width=0.2, linewidth=0, length_includes_head=True)




	font_size = matplotlib.rcParams['font.size'] * final_factor

	if ax_lim > 70:
		ax_res = 50
	elif ax_lim > 50:
		ax_res = 25
	elif ax_lim > 25:
		ax_res = 20
	elif ax_lim > 7.5:
		ax_res = 5
	else:
		ax_res = 2
	tick_values = [i for i in range(-math.ceil(ax_lim), math.ceil(ax_lim) + 1) if abs(i) % ax_res == 0 and abs(i) <= ax_lim]
	#print(ax_res, tick_values)

	ax.set_xlim(-ax_lim, ax_lim)
	if side == "side":
		ax.invert_xaxis()

	if x_axes != True:
		ax.xaxis.label.set_color('black')
		ax.tick_params(axis='x', colors='black')
	else:
		if side == "side":
			ax.set_xlabel(r"z Position [R$_\oplus$]")
		else:
			ax.set_xlabel(r"x Position [R$_\oplus$]")
		
		
		ax.xaxis.set_major_locator(ticker.FixedLocator(tick_values))

		#ax.xaxis.label.set_fontsize(font_size * 1.5)
		for label in (ax.get_xticklabels()):
			label.set_fontsize(font_size)



	ax.set_ylim(-ax_lim, ax_lim)

	if y_axes != True:
		ax.yaxis.label.set_color('black')
		ax.tick_params(axis='y', colors='black')
	else:
		if side == "front":
			ax.set_ylabel(r"z Position [R$_\oplus$]")
		else:
			ax.set_ylabel(r"y Position [R$_\oplus$]")
	

		ax.yaxis.set_major_locator(ticker.FixedLocator(tick_values))

		#ax.set_yticks(ax.get_xticks())
		#ax.yaxis.label.set_fontsize(font_size * 1.5)
		for label in (ax.get_yticklabels()):
			label.set_fontsize(font_size)
	
	



	if time_step != "undefined":
		hours = int(abs(time_step) // 3600)
		and_minutes = int((abs(time_step) - (hours * 3600)) // 60)
		time_text = "{0}h {1}m".format(hours, str(and_minutes).zfill(2))
		if time_step > 0:
			time_text = "+{0}".format(time_text)
		elif time_step < 0:
			time_text = "-{0}".format(time_text)

		time_text = "{0}".format(time_text)

		plt.scatter(2 * ax_lim, 2* ax_lim, c="black", label=time_text)
		plt.legend(handlelength=0, handletextpad=0, loc='upper right', prop={'size': 20}, frameon=False, framealpha=0)
		
	else:
		time_text = ""


	title_text = ""

	if not final_output:
		#title_text = "{0}\n".format(simulation_name)

		title_text += "Snapshot {0}".format(snapshot_id)

	title_text = "{0}{1}".format(title_text, time_text)
	#plt.title(title_text, fontsize=font_size)

	
	plt.tight_layout()


def plot_pos_mat_id_3d(pos, mat_id, id, R, ax_lim, simulation_name, snapshot_id, time_step="undefined", final_output=False, x_axes="undefined", y_axes="undefined", z_axes="undefined", tracked_id=-1):
	#
	#
	#
	#	3D
	#
	#
	#
	""" Plot the particles, coloured by their material. IN 3D"""
	fig = plt.figure(figsize=(7, 7))
	ax = fig.add_subplot(projection='3d',computed_zorder=False)
	#plt.figure(figsize=(7, 7))
	#ax = plt.gca()
	ax.set_aspect("equal")

	colour = np.empty(len(pos), dtype=object)
	for id_c, c in material_colour_map.ID_COLOUR_MAP.items():
		colour[mat_id == id_c] = c

	ax.scatter(pos[:, 0], pos[:, 1], pos[:, 2], c=colour, edgecolors="none", marker=".", s=10, alpha=0.5, zorder=1)

	if tracked_id != -1:
		tracked_mask = id == tracked_id
		tracked_x = pos[tracked_mask, 0][0]
		tracked_y = pos[tracked_mask, 1][0]
		tracked_z = pos[tracked_mask, 2][0]

		edge_x = np.sqrt((R**2 - tracked_z**2) - tracked_y**2)
		edge_y = -np.sqrt((R**2 - tracked_z**2) - tracked_x**2)

		ax.scatter(tracked_x, tracked_y, tracked_z, c="red", edgecolors="none", marker=".", s=50, alpha=1)

		ax.plot([edge_x, ax_lim], [tracked_y, tracked_y], [tracked_z, tracked_z], c="red", alpha = 0.5, zorder=2)
		ax.plot([tracked_x, tracked_x], [-ax_lim, edge_y], [tracked_z, tracked_z], c="red", alpha = 0.5, zorder=2)
		ax.plot([tracked_x, tracked_x], [tracked_y, tracked_y], [0, ax_lim], c="red", alpha = 0.5, zorder=2)


	if final_output:
		font_size = 60
	else:
		font_size = matplotlib.rcParams['font.size']

	ax.set_xlim(-ax_lim, ax_lim)
	ax.set_ylim(-ax_lim, ax_lim)
	ax.set_zlim(-ax_lim, ax_lim)
	ax.set_yticks(ax.get_xticks())
	ax.set_zticks(ax.get_xticks())

	ax.set_xlabel(r"x Position ($R_\oplus$)")
	
	ax.xaxis.label.set_fontsize(font_size/2)
	for label in (ax.get_xticklabels()):
		label.set_fontsize(font_size/3)

	if x_axes != True:
		ax.xaxis.label.set_color('black')
		ax.tick_params(axis='x', colors='black')


	ax.set_ylabel(r"y Position ($R_\oplus$)")

	ax.yaxis.label.set_fontsize(font_size/2)
	for label in (ax.get_yticklabels()):
		label.set_fontsize(font_size/3)

	if y_axes != True:
		ax.yaxis.label.set_color('black')
		ax.tick_params(axis='y', colors='black')

	ax.set_zlabel(r"z Position ($R_\oplus$)")

	ax.zaxis.label.set_fontsize(font_size/2)
	for label in (ax.get_zticklabels()):
		label.set_fontsize(font_size/3)

	if z_axes != True:
		ax.zaxis.label.set_color('black')
		ax.tick_params(axis='z', colors='black')



	if time_step != "undefined":
		hours = int(abs(time_step) // 3600)
		and_minutes = int((abs(time_step) - (hours * 3600)) // 60)
		time_text = "{0}h {1}m".format(hours, and_minutes)
		if time_step > 0:
			time_text = "+{0}".format(time_text)
		elif time_step < 0:
			time_text = "-{0}".format(time_text)

		time_text = "{0}".format(time_text)
		
	else:
		time_text = ""


	title_text = ""

	if not final_output:
		title_text = "{0}\nSnapshot {1}\n".format(simulation_name, snapshot_id)

	title_text = "{0}{1}".format(title_text, time_text)
	plt.title(title_text, fontsize=font_size)

	#plt.tight_layout()
