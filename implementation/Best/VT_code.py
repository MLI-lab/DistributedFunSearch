def priority(node, G, n, s):
    """
    Priority based on VT code property: 
    Assigns +inf to nodes satisfying the VT checksum condition, -inf otherwise.
    """

    def vt_checksum(node):
        """Calculate the VT checksum of a binary string (node)."""
        return sum((i + 1) * int(bit) for i, bit in enumerate(node)) % (n + 1)

    # Compute VT checksum for the current node
    checksum = vt_checksum(node)

    # Compute priority
    if checksum == 0:
        # Node satisfies VT condition
        return float('inf')  # Highest priority
    else:
        # Node does not satisfy VT condition
        return float('-inf')  # Lowest priority
