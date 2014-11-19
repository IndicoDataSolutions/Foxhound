"""
GPU memory calculations and configuration
"""
import numpy as np

def n_chunks(memory, *data):
	unit_size = 0
	for arr in data:
		unit_size += arr[0].nbytes
	return int(np.floor(memory / unit_size))
