import numpy as np
from scipy.stats import poisson

def poisson_em(data, max_iter=100, tol=1e-4):
    # Initialize parameters
    λ1, λ2 = np.random.uniform(0, 10, 2)
    π = 0.5

    for i in range(max_iter):
        # E-step
        resp1 = π * poisson.pmf(data, λ1)
        resp2 = (1 - π) * poisson.pmf(data, λ2)
        total_resp = resp1 + resp2
        gamma = resp1 / total_resp

        # M-step
        λ1_new = np.sum(gamma * data) / np.sum(gamma)
        λ2_new = np.sum((1 - gamma) * data) / np.sum(1 - gamma)
        π_new = np.mean(gamma)

        # Check convergence
        if np.abs(λ1_new - λ1) < tol and np.abs(λ2_new - λ2) < tol and np.abs(π_new - π) < tol:
            break

        λ1, λ2, π = λ1_new, λ2_new, π_new

    return λ1, λ2, π


data = np.loadtxt("em_data.txt")
mean_with_advice, mean_without_advice, proportion_with_advice = poisson_em(data)

print(f"Mean number of children in families with family planning: {mean_with_advice}")
print(f"Mean number of children in families without family planning: {mean_without_advice}")
print(f"Proportion of families with family planning: {proportion_with_advice}")
print(f"Proportion of families without family planning: {1 - proportion_with_advice}")