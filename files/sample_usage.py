from sampled_impact import sample_distributions
import numpy as np
import matplotlib.pyplot as plt

sample_size = 10000
mass_histogram, mass_bin_edges = np.histogram(np.random.default_rng().normal(0.0, 1.0, size=sample_size), bins=150)
vel_histogram, vel_bin_edges = np.histogram(np.random.default_rng().exponential(scale=1.0, size=sample_size), bins=150)
b_histogram, b_bin_edges = np.histogram(np.random.default_rng().normal(0.5, 0.2, size=sample_size), bins=150)
theta_histogram, theta_bin_edges = np.histogram(np.random.random(size=sample_size) * 360, bins=150)

distributions = [[mass_histogram, mass_bin_edges], [vel_histogram, vel_bin_edges], [b_histogram, b_bin_edges], [theta_histogram, theta_bin_edges]]

impact_parameters = sample_distributions(distributions)
print(impact_parameters)


width = 2 # plots
height = 2 # plots 
fig, axes = plt.subplots(width, height, figsize=(10, 12))

distribution_num = 0
for i in range(width):
	for j in range(height):
		axes[i, j].plot(distributions[distribution_num][1][:-1], distributions[distribution_num][0], color='#b21b00')
		#axes[i, j].set_yscale('log')
		#axes[i, j].set_xscale('log')
		#axes[i, j].set_title('Mass Distribution', fontsize=12)
		#axes[i, j].set_xlabel(r'Mass ($M_\oplus$)', fontsize=12)
		axes[i, j].set_ylabel('Impact count', fontsize=12)
		axes[i, j].grid(True)
		axes[i, j].tick_params(labelsize=12)
		axes[i, j].set_ylim(-0.05 * max(distributions[distribution_num][0]), None)
		distribution_num += 1

plt.tight_layout()

plt.savefig("histograms.png")