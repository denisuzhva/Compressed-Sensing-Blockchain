import numpy as np
import matplotlib.pyplot as plt
import os



def rand_signal_generator(len):
    times = np.arange(0, len)
    signal = np.sin(times) + np.random.normal(scale=0.1, size=times.size) 
    return signal


def generate_block(input_data, seed_vector, m, n):
    meas_mat = np.zeros((m, n), dtype=np.float32)
    for idx, seed in enumerate(seed_vector):
        seed_int = np.asarray(seed, dtype=np.float32).view(np.uint32)
        meas_mat[idx] = np.random.RandomState(seed_int).binomial(1, .5, n) * 2 - 1
    meas_mat /= np.sqrt(m)
    out_data = meas_mat.dot(input_data)
    return out_data, meas_mat


if __name__ == "__main__":
    dataset_path = "./datasets/extrasensory/"
    sample_names = os.listdir(dataset_path)[:]

    m = 8
    y = np.arange(0, m, dtype=np.float32)
    
    cs_blockchain = np.zeros((len(sample_names), m))
    for idx, sample_name in enumerate(sample_names):
        sample = np.loadtxt(dataset_path + sample_name)[:, 3]
        n = sample.size
        y, _ = generate_block(sample, y, m, n)
        cs_blockchain[idx] = y 

    plt.plot(cs_blockchain[:, 0])
    plt.plot(cs_blockchain[:, 1])
    plt.plot(cs_blockchain[:, 2])
    plt.show()
