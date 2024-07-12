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
import numpy as np
import unyt

from custom_functions import material_colour_map
from woma.misc import glob_vars as gv
from woma.misc.utils import SI_to_SI, SI_to_cgs

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




def load_to_woma(snapshot, num_target_particles=-1):
	snapshot_data = sw.load(snapshot)
	box_mid = 0.5 * snapshot_data.metadata.boxsize[0].to(unyt.m)
	snapshot_data.gas.coordinates.convert_to_mks()
	pos = np.array(snapshot_data.gas.coordinates - box_mid)
	snapshot_data.gas.velocities.convert_to_mks()
	vel = np.array(snapshot_data.gas.velocities)
	snapshot_data.gas.smoothing_lengths.convert_to_mks()
	h = np.array(snapshot_data.gas.smoothing_lengths)
	snapshot_data.gas.masses.convert_to_mks()
	m = np.array(snapshot_data.gas.masses)
	snapshot_data.gas.densities.convert_to_mks()
	rho = np.array(snapshot_data.gas.densities)
	snapshot_data.gas.pressures.convert_to_mks()
	p = np.array(snapshot_data.gas.pressures)
	snapshot_data.gas.internal_energies.convert_to_mks()
	u = np.array(snapshot_data.gas.internal_energies)
	mat_id = np.array(snapshot_data.gas.material_ids)
	id = np.array(snapshot_data.gas.particle_ids)
    
	pos_centerM = np.sum(pos * m[:, np.newaxis], axis=0) / np.sum(m)
	vel_centerM = np.sum(vel * m[:, np.newaxis], axis=0) / np.sum(m)

	pos -= pos_centerM
	vel -= vel_centerM

	xy = np.hypot(pos[:, 0], pos[:, 1])
	r = np.hypot(xy, pos[:, 2])
	r = np.sort(r)
	R = np.mean(r[-1000:])

	if num_target_particles != -1:
		# Edit material IDs for particles in the impactor
		mat_id[num_target_particles <= id] += material_colour_map.id_body

	return pos, vel, h, m, rho, p, u, mat_id, R

def bound_load_to_woma(snapshot, remnant_id):
	pos, vel, h, m, rho, p, u, mat_id, R = load_to_woma(snapshot)
	
	snapshot_data = sw.load(snapshot)
	bound_id = np.array(snapshot_data.gas.bound_ids, dtype=np.int32)
	mask = bound_id == remnant_id
	
	return pos[mask], vel[mask], h[mask], m[mask], rho[mask], p[mask], u[mask], mat_id[mask], R
	
	
def save_particle_data(
    f,
    A2_pos,
    A2_vel,
    A1_m,
    A1_h,
    A1_rho,
    A1_P,
    A1_u,
    A1_mat_id,
    A1_id=None,
    A1_s=None,
    boxsize=0,
    file_to_SI=SI_to_SI,
    verbosity=1,
    dark_matter=False
):
    """Save particle data to an hdf5 file.

    Uses the same format as the SWIFT simulation code (www.swiftsim.com).

    Parameters
    ----------
    f : h5py File
        The opened hdf5 data file (with "w").

    A2_pos, A2_vel, A1_m, A1_h, A1_rho, A1_P, A1_u, A1_mat_id
        : [float] or [int]
        The particle data arrays. See Di_hdf5_particle_label for details.

    A1_id : [int] (opt.)
        The particle IDs. Defaults to the order in which they're provided.

    A1_s : [float] (opt.)
        The particle specific entropies.

    boxsize : float (opt.)
        The simulation box side length (m). If provided, then the origin will be
        shifted to the centre of the box.

    file_to_SI : woma.Conversions (opt.)
        Simple unit conversion object from the file's units to SI. Defaults to
        staying in SI. See Conversions in misc/utils.py for more details.
    """
    num_particle = len(A1_m)
    if A1_id is None:
        A1_id = np.arange(num_particle)

    # Convert to file units
    SI_to_file = file_to_SI.inv()
    boxsize *= SI_to_file.l
    A2_pos *= SI_to_file.l
    A2_vel *= SI_to_file.v
    A1_m *= SI_to_file.m
    A1_h *= SI_to_file.l
    A1_rho *= SI_to_file.rho
    A1_P *= SI_to_file.P
    A1_u *= SI_to_file.u
    if A1_s is not None:
        A1_s *= SI_to_file.s

    # Shift to box coordinates
    A2_pos += boxsize / 2.0

    # Print info
    if verbosity >= 1:
        print("")
        print("num_particle = %d" % num_particle)
        print("boxsize      = %.2g" % boxsize)
        print("mat_id       = ", end="")
        for mat_id in np.unique(A1_mat_id):
            print("%d " % mat_id, end="")
        print("\n")
        print("Unit mass    = %.5e g" % (file_to_SI.m * SI_to_cgs.m))
        print("Unit length  = %.5e cm" % (file_to_SI.l * SI_to_cgs.l))
        print("Unit time    = %.5e s" % file_to_SI.t)
        print("")
        print("Min, max values (file units):")
        print(
            "  pos = [%.5g, %.5g,    %.5g, %.5g,    %.5g, %.5g]"
            % (
                np.amin(A2_pos[:, 0]),
                np.amax(A2_pos[:, 0]),
                np.amin(A2_pos[:, 1]),
                np.amax(A2_pos[:, 1]),
                np.amin(A2_pos[:, 2]),
                np.amax(A2_pos[:, 2]),
            )
        )
        print(
            "  vel = [%.5g, %.5g,    %.5g, %.5g,    %.5g, %.5g]"
            % (
                np.amin(A2_vel[:, 0]),
                np.amax(A2_vel[:, 0]),
                np.amin(A2_vel[:, 1]),
                np.amax(A2_vel[:, 1]),
                np.amin(A2_vel[:, 2]),
                np.amax(A2_vel[:, 2]),
            )
        )
        for name, array in zip(
            ["m", "rho", "P", "u", "h"], [A1_m, A1_rho, A1_P, A1_u, A1_h]
        ):
            print("  %s = %.5g, %.5g" % (name, np.amin(array), np.amax(array)))
        if A1_s is not None:
            print("  s = %.5g, %.5g" % (np.amin(A1_s), np.amax(A1_s)))
        print("")

    # Save
    # Header
    grp = f.create_group("/Header")
    grp.attrs["BoxSize"] = [boxsize] * 3

    if dark_matter: 
        num_dm_particle = 1 
    else:
        num_dm_particle = 0

    grp.attrs["NumPart_Total"] = [num_particle, num_dm_particle, 0, 0, 0, 0]
    grp.attrs["NumPart_Total_HighWord"] = [0, 0, 0, 0, 0, 0]
    grp.attrs["NumPart_ThisFile"] = [num_particle, num_dm_particle, 0, 0, 0, 0]
    grp.attrs["Time"] = 0.0
    grp.attrs["NumFilesPerSnapshot"] = 1
    grp.attrs["MassTable"] = [0.0, 0.0, 0.0, 0.0, 0.0, 0.0]
    grp.attrs["Flag_Entropy_ICs"] = 0
    grp.attrs["Dimension"] = 3

    # Runtime parameters
    grp = f.create_group("/RuntimePars")
    grp.attrs["PeriodicBoundariesOn"] = 0

    # Units
    grp = f.create_group("/Units")
    grp.attrs["Unit mass in cgs (U_M)"] = file_to_SI.m * SI_to_cgs.m
    grp.attrs["Unit length in cgs (U_L)"] = file_to_SI.l * SI_to_cgs.l
    grp.attrs["Unit time in cgs (U_t)"] = file_to_SI.t
    grp.attrs["Unit current in cgs (U_I)"] = 1.0
    grp.attrs["Unit temperature in cgs (U_T)"] = 1.0

    # Particles
    grp = f.create_group("/PartType0")
    grp.create_dataset(Di_hdf5_particle_label["pos"], data=A2_pos, dtype="d")
    grp.create_dataset(Di_hdf5_particle_label["vel"], data=A2_vel, dtype="f")
    grp.create_dataset(Di_hdf5_particle_label["m"], data=A1_m, dtype="f")
    grp.create_dataset(Di_hdf5_particle_label["h"], data=A1_h, dtype="f")
    grp.create_dataset(Di_hdf5_particle_label["rho"], data=A1_rho, dtype="f")
    grp.create_dataset(Di_hdf5_particle_label["P"], data=A1_P, dtype="f")
    grp.create_dataset(Di_hdf5_particle_label["u"], data=A1_u, dtype="f")
    grp.create_dataset(Di_hdf5_particle_label["id"], data=A1_id, dtype="L")
    grp.create_dataset(Di_hdf5_particle_label["mat_id"], data=A1_mat_id, dtype="i")
    if A1_s is not None:
        grp.create_dataset(Di_hdf5_particle_label["s"], data=A1_s, dtype="f")

    if verbosity >= 1:
        print('Saved "%s"' % f.filename[-64:])

