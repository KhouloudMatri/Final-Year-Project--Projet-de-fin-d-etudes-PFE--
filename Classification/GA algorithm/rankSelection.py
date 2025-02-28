import numpy as np

def rank_selection(expectation, n_parents):
    """
    Perform rank-based selection to choose parents for genetic algorithms.

    Parameters:
    expectation : list or numpy array
        The expectation values for each individual in the population.
    n_parents : int
        The number of parents to select.

    Returns:
    parents : list
        Indices of the selected parents.
    """
    # Normalize the expectation values to get probabilities
    denominator = np.sum(expectation)
    ranks = expectation / denominator

    # Calculate cumulative probabilities
    cumulative_ranks = np.cumsum(ranks)

    index_parents = []

    # Select parents based on their rank probabilities
    for _ in range(n_parents):
        r = np.random.rand()  # Random number between 0 and 1
        for j, cumulative_rank in enumerate(cumulative_ranks):
            if r <= cumulative_rank:
                index_parents.append(j)
                break
    
    return index_parents

# Example usage:
# expectation = [0.2, 0.3, 0.5]
# n_parents = 2
# parents = rank_selection(expectation, n_parents)
# print(parents)
