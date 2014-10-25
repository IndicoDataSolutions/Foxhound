"""
GPU memory calculations and configuration
"""
import numpy as np

def n_chunks(memory, batch_size, data):
	return int(np.floor(memory / (data[0].nbytes * batch_size)))
