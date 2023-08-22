import os
import numpy as np

import h5py

import scipy.io as scio

data_path = '../mpi_inf_3dhp/test_data'
cam_set = [0, 1, 2, 4, 5, 6, 7, 8]
# joint_set = [8, 6, 15, 16, 17, 10, 11, 12, 24, 25, 26, 19, 20, 21, 5, 4, 7]
joint_set = [7, 5, 14, 15, 16, 9, 10, 11, 23, 24, 25, 18, 19, 20, 4, 3, 6]

dic_seq={}

for root, dirs, files in os.walk(data_path):

    for file in files:
        if file.endswith("mat"):

            path = root.split("/")
            subject = path[-1][2]
            print("loading %s..."%path[-1])

            data = h5py.File(os.path.join(root, file))

            valid_frame = np.squeeze(data['valid_frame'][:])

            data_2d = np.squeeze(data['annot2'][:])
            data_3d = np.squeeze(data['univ_annot3'][:])

            dic_data = {"data_2d":data_2d,"data_3d":data_3d, "valid":valid_frame}

            dic_seq.update({path[-1]:dic_data})

np.savez_compressed('../motion3d/data_test_3dhp', data=dic_seq)
