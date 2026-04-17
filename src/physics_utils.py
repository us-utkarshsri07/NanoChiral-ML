import numpy as np

def calculate_cnt_radius(n, m):
    """Calculates CNT radius in nm based on chiral indices."""
    a = 0.142  # C-C bond distance in nm
    return (a * np.sqrt(3 * (n**2 + m**2 + n*m))) / (2 * np.pi)

def get_chirality_type(n, m):
    """Determines if the CNT is metallic or semiconducting."""
    if (n - m) % 3 == 0:
        return "Metallic"
    return "Semiconducting"