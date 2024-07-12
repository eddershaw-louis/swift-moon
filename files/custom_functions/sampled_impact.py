import numpy as np

def sample_distributions(distributions):#mass_distribution, velocity_distribution, b_distribution, theta_distribution):
	samples = []
	for distribution in distributions:
		histogram = distribution[0]
		bin_edges = distribution[1]

		probabilities = histogram / histogram.sum()
		bin_centres = (bin_edges[:-1] + bin_edges[1:]) / 2
		samples.append(np.random.choice(bin_centres, size=1, p=probabilities)[0])

	return tuple(samples)