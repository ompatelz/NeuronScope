"""NeuronScope backend package."""

import os

# Keep one bounded experiment from monopolizing every host core. These defaults are
# established before NumPy, scikit-learn, or PyTorch are imported, while still
# allowing an operator to choose a smaller or otherwise measured deployment value.
for _thread_variable in (
    "OMP_NUM_THREADS",
    "MKL_NUM_THREADS",
    "OPENBLAS_NUM_THREADS",
    "NUMEXPR_NUM_THREADS",
):
    os.environ.setdefault(_thread_variable, "1")
