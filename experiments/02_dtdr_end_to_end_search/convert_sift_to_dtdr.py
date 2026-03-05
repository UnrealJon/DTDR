import numpy as np
from dtdr_index_io import write_dtdr_index

# Adjust path if needed
sift_path = r"G:\train_jw\datasets\sift1m\sift_base.npy"
vectors = np.load(sift_path)

print("Loaded:", vectors.shape)

write_dtdr_index("sift1m_base.dtdr", vectors, quant_bits=8)

print("DTDR file written.")