# C++ extension for brute-force decoder validation
try:
    from .bruteforce_cpp import validate_decoder_bruteforce
    __all__ = ['validate_decoder_bruteforce']
except ImportError:
    # Extension not built yet
    pass
