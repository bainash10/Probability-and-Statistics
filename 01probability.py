import numpy as np
import scipy.stats as stats
import matplotlib.pyplot as plt

class ProbabilityProgram:
    def __init__(self, data=None):
        self.data = data

    # Basic Probability
    def probability(self, event_outcomes, sample_space):
        return len(event_outcomes) / len(sample_space)
    
    # Conditional Probability
    def conditional_probability(self, A, B):
        return len(A & B) / len(B)
    
    # Bayes' Theorem
    def bayes_theorem(self, P_B_given_A, P_A, P_B):
        return (P_B_given_A * P_A) / P_B

    # Random Variables and Distributions
    def binomial_distribution(self, n, p, size=1000):
        return np.random.binomial(n, p, size)

    def normal_distribution(self, mean=0, std=1, size=1000):
        return np.random.normal(mean, std, size)

    def poisson_distribution(self, lam, size=1000):
        return np.random.poisson(lam, size)

    def exponential_distribution(self, scale=1, size=1000):
        return np.random.exponential(scale, size)

    # Expected Value
    def expected_value(self, probabilities, values):
        return np.sum(probabilities * values)

    # Variance and Standard Deviation
    def variance(self, probabilities, values):
        mean = self.expected_value(probabilities, values)
        return np.sum(probabilities * (values - mean) ** 2)

    def std_deviation(self, probabilities, values):
        return np.sqrt(self.variance(probabilities, values))

    # Data Visualization
    def plot_distribution(self, data, bins=30, title='Distribution'):
        plt.hist(data, bins=bins)
        plt.title(title)
        plt.show()

# Example usage
prob_prog = ProbabilityProgram()

# Basic Probability
sample_space = set(range(1, 7))  # Sample space for a die roll
event_outcomes = {1, 2}  # Event: rolling a 1 or 2
print("Probability of rolling a 1 or 2:", prob_prog.probability(event_outcomes, sample_space))

# Conditional Probability
A = {2, 4, 6}  # Event A: rolling an even number
B = {4, 5, 6}  # Event B: rolling a number greater than 3
print("P(A|B):", prob_prog.conditional_probability(A, B))

# Bayes' Theorem
P_B_given_A = 0.9
P_A = 0.2
P_B = 0.3
print("P(A|B) using Bayes' Theorem:", prob_prog.bayes_theorem(P_B_given_A, P_A, P_B))

# Random Variables and Distributions
binom_data = prob_prog.binomial_distribution(10, 0.5, size=1000)
norm_data = prob_prog.normal_distribution(0, 1, size=1000)
poisson_data = prob_prog.poisson_distribution(5, size=1000)
exp_data = prob_prog.exponential_distribution(1, size=1000)

# Plotting Distributions
prob_prog.plot_distribution(binom_data, title='Binomial Distribution')
prob_prog.plot_distribution(norm_data, title='Normal Distribution')
prob_prog.plot_distribution(poisson_data, title='Poisson Distribution')
prob_prog.plot_distribution(exp_data, title='Exponential Distribution')

# Expected Value, Variance, Standard Deviation
values = np.array([0, 1, 2, 3, 4, 5])
probabilities = np.array([0.1, 0.2, 0.3, 0.2, 0.1, 0.1])
print("Expected Value:", prob_prog.expected_value(probabilities, values))
print("Variance:", prob_prog.variance(probabilities, values))
print("Standard Deviation:", prob_prog.std_deviation(probabilities, values))
