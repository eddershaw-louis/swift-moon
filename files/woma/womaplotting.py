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

import matplotlib.pyplot as plt
import matplotlib
from matplotlib.ticker import ScalarFormatter
import woma

R_earth = 6.371e6   # m
M_earth = 5.9724e24  # kg
G = 6.67408e-11  # m^3 kg^-1 s^-2

font_size = 20
params = {
    "axes.labelsize": font_size,
    "font.size": font_size,
    "xtick.labelsize": font_size,
    "ytick.labelsize": font_size,
    "font.family": "serif",
}
matplotlib.rcParams.update(params)
plt.rcParams["mathtext.fontset"] = "cm"


def plot_profiles(planet, output_directory):
    fig, ax = plt.subplots(2, 2, figsize=(8,8))

    ax[0, 0].plot(planet.A1_r / R_earth, planet.A1_rho / 1000)
    ax[0, 0].set_xlabel(r"Radius [R$_\oplus$]")
    ax[0, 0].set_ylabel(r"Density [g cm$^{-3}$]")
    ax[0, 0].set_yscale("log")
    ax[0, 0].set_xlim(0, None)

    ax[1, 0].plot(planet.A1_r / R_earth, planet.A1_m_enc / M_earth)
    ax[1, 0].set_xlabel(r"Radius [R$_\oplus$]")
    ax[1, 0].set_ylabel(r"Enclosed Mass [M$_\oplus$]")
    ax[1, 0].set_xlim(0, None)
    ax[1, 0].set_ylim(0, None)

    ax[0, 1].plot(planet.A1_r / R_earth, planet.A1_P)
    ax[0, 1].set_xlabel(r"Radius [R$_\oplus$]")
    ax[0, 1].set_ylabel(r"Pressure [Pa]")
    ax[0, 1].set_yscale("log")
    ax[0, 1].set_xlim(0, None)

    ax[1, 1].plot(planet.A1_r / R_earth, planet.A1_T / 1000)
    ax[1, 1].set_xlabel(r"Radius [R$_\oplus$]")
    ax[1, 1].set_ylabel(r"Temperature [$10^3$ K]")
    ax[1, 1].set_xlim(0, None)
    ax[1, 1].set_ylim(0, None)

    plt.tight_layout()
    #plt.show()
    plt.savefig("{0}{1}.png".format(output_directory, planet.name.replace(" ", "-")))
    plt.close()

def plot_multiple_profiles(planets, impactors, output_path):
    all = True
    if all:
    	fig, ax = plt.subplots(2, 2, figsize=(8,8))
    else:
        fig, ax = plt.subplots(1, 2, figsize=(8,4))

    rho_scale = 1000
    P_scale = 100000000000
    T_scale = 1000

    planet_colour_cycle = ["teal", "lightseagreen", "turquoise", "aquamarine", "mediumturquoise", "darkturquoise", "paleturquoise", "mediumaquamarine", "seagreen"]
    planet_colour_index = 0
    for planet in planets:   	
        if all:
            ax[0, 0].plot(planet.A1_r / R_earth, planet.A1_rho / rho_scale, label=planet.name, color=planet_colour_cycle[planet_colour_index])

            ax[1, 0].plot(planet.A1_r / R_earth, planet.A1_m_enc / M_earth, color=planet_colour_cycle[planet_colour_index])
    	
            ax[0, 1].plot(planet.A1_r / R_earth, planet.A1_P / P_scale, color=planet_colour_cycle[planet_colour_index])
    	
            ax[1, 1].plot(planet.A1_r / R_earth, planet.A1_T / T_scale, color=planet_colour_cycle[planet_colour_index])
        else:
            ax[0].plot(planet.A1_r / R_earth, planet.A1_rho / rho_scale, label=planet.name, color=planet_colour_cycle[planet_colour_index])
            ax[1].plot(planet.A1_r / R_earth, planet.A1_T / T_scale, color=planet_colour_cycle[planet_colour_index])

        planet_colour_index += 1
    
    #ax[0, 0].plot(impactors[0].A1_r / R_earth, impactors[0].A1_rho / rho_scale, label=" ", color="white", alpha=0.0)
    impactor_colour_cycle = ["orchid", "deeppink", "darkorchid", "rebeccapurple", "mediumpurple", "thistle", "mediumvioletred", "deeppink", "crimson", "blueviolet"]
    impactor_colour_index = 0
    for impactor in impactors:   	
        if all:
            ax[0, 0].plot(impactor.A1_r / R_earth, impactor.A1_rho / rho_scale, label=impactor.name, color=impactor_colour_cycle[impactor_colour_index], linestyle="dashed")

            ax[1, 0].plot(impactor.A1_r / R_earth, impactor.A1_m_enc / M_earth, color=impactor_colour_cycle[impactor_colour_index], linestyle="dashed")
    	
            ax[0, 1].plot(impactor.A1_r / R_earth, impactor.A1_P / P_scale, color=impactor_colour_cycle[impactor_colour_index], linestyle="dashed")
    	
            ax[1, 1].plot(impactor.A1_r / R_earth, impactor.A1_T / T_scale, label=impactor.name, color=impactor_colour_cycle[impactor_colour_index], linestyle="dashed")
        else:
            ax[0].plot(impactor.A1_r / R_earth, impactor.A1_rho / rho_scale, color=impactor_colour_cycle[impactor_colour_index], linestyle="dashed")
            ax[1].plot(impactor.A1_r / R_earth, impactor.A1_T / T_scale, label=impactor.name, color=impactor_colour_cycle[impactor_colour_index], linestyle="dashed")

        impactor_colour_index += 1

    if all:
        ax[0, 0].set_xlabel(r"Radius [R$_\oplus$]")
        ax[0, 0].set_ylabel(r"Density [g cm$^{-3}$]")
        #ax[0, 0].set_yscale("log")
        ax[0, 0].set_ylim(0, None)
        ax[0, 0].set_xlim(0, None) 
        ax[0, 0].legend(loc=1, prop={'size': 11})#, ncol=2)

        ax[1, 0].set_xlabel(r"Radius [R$_\oplus$]")
        ax[1, 0].set_ylabel(r"Enclosed Mass [M$_\oplus$]")
        ax[1, 0].set_xlim(0, None)
        ax[1, 0].set_ylim(0, None)

        ax[0, 1].set_xlabel(r"Radius [R$_\oplus$]")
        ax[0, 1].set_ylabel(r"Pressure [Mbar]")
        #ax[0, 1].set_yscale("log")
        ax[0, 1].set_xlim(0, None)
        ax[0, 1].set_ylim(0, 12.51)

        ax[1, 1].set_xlabel(r"Radius [R$_\oplus$]")
        ax[1, 1].set_ylabel(r"Temperature [$10^3$ K]")
        ax[1, 1].set_xlim(0, None)
        ax[1, 1].set_ylim(0, None)
        #ax[1, 1].yaxis.set_major_formatter(ScalarFormatter(useMathText=True, useOffset=False))
        #ax[1, 1].legend(loc=1, prop={'size': 10})
    else:
        ax[0].set_xlabel(r"Radius [R$_\oplus$]")
        ax[0].set_ylabel(r"Density [g cm$^{-3}$]")
        #ax[0].set_yscale("log")
        ax[0].set_xlim(0, None) 
        ax[0].legend(loc=3, prop={'size': 10}, ncol=2)

        ax[1].set_xlabel(r"Radius [R$_\oplus$]")
        ax[1].set_ylabel(r"Temperature [$10^3$ K]")
        #ax[1].yaxis.set_major_formatter(ScalarFormatter(useMathText=True, useOffset=False))
        ax[1].set_xlim(0, None)
        ax[1].set_ylim(0, None)
        #ax[1].legend(loc=1, prop={'size': 10})

    plt.tight_layout()
    #plt.show()
    plt.savefig(output_path, dpi=300)
    plt.close()


def plot_particles(particles, planet, filename, output_directory):
    materials = list(set(particles.A1_mat_id))
    
    fig, ax = plt.subplots(1, 1, figsize=(6,6))
    
    for material in materials:
        material_indexes = [i for i in range(len(particles.A2_pos)) if particles.A1_mat_id[i] == material]
        material_pos_xs = []
        material_pos_ys = []
        for index in material_indexes:
            slice_percentage = 0.01
            if particles.A2_pos[index][2] < planet.R * slice_percentage and particles.A2_pos[index][2] > - planet.R * slice_percentage:
                material_pos_xs += [particles.A2_pos[index][0]/R_earth]
                material_pos_ys += [particles.A2_pos[index][1]/R_earth]
                
        ax.scatter(material_pos_xs, material_pos_ys, label=planet.A1_mat_layer[planet.A1_mat_id_layer.index(material)])
    
    axis_lim = planet.R * 1.05 / R_earth
    ax.set_xlim(-axis_lim, axis_lim)
    ax.set_xlabel(r"x, $[R_\oplus]$")
    ax.set_ylim(-axis_lim, axis_lim)
    ax.set_ylabel(r"y, $[R_\oplus]$")
    ax.set_aspect('equal')
    ax.legend()
    
    plt.tight_layout()
    #plt.show()
    plt.savefig("{0}{1}.png".format(output_directory, filename))
    plt.close()

    """
    fig = plt.figure(figsize=plt.figaspect(1))
    ax = fig.add_subplot(projection='3d')
    
    material_pos_xs_3D = []
    material_pos_ys_3D = []
    material_pos_zs_3D = []
    for material in materials:
        material_indexes = [i for i in range(len(particles.A2_pos)) if particles.A1_mat_id[i] == material]

        
        for index in material_indexes:
            if particles.A2_pos[index][2] > 0:
                material_pos_xs_3D += [particles.A2_pos[index][0]/R_earth]
                material_pos_ys_3D += [particles.A2_pos[index][1]/R_earth]
                material_pos_zs_3D += [particles.A2_pos[index][2]/R_earth]
        #fig = plt.figure(figsize=plt.figaspect(1))
        #ax = fig.add_subplot(projection='3d')
        ax.plot_surface(material_pos_xs_3D, material_pos_zs_3D, material_pos_ys_3D)
        #plt.savefig("{0}-3D-cross-section-{1}.png".format(planet.name.replace(" ", "-"), material))
        #plt.close()
    #plt.savefig("{0}{1}-3D-cross-section.png".format(output_directory, planet.name.replace(" ", "-")))
    plt.show()
    plt.close()
    """

