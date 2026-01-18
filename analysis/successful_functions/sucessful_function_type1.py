"""Extracted priority functions that achieved target signature."""

# Function 1
def f(v, G, n, s):
    position = [(j + 1) * (n - j) / (6 * s) for j, value in enumerate(v) if int(value) == 1]
    total_position = np.sum(position)
    degree = G.degree(v) / float(n)
    return 4 * total_position + 5 * degree


def priority_1(node, n, s, q):
