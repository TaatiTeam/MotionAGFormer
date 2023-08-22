import argparse
import os
from glob import glob
from dataclasses import dataclass

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.animation import FuncAnimation

import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.getcwd())))

from utils.data import read_pkl
from data.const import H36M_TO_MPI
from data.reader.motion_dataset import MPI3DHP

connections = [
    (10, 9),
    (9, 8),
    (8, 11),
    (8, 14),
    (14, 15),
    (15, 16),
    (11, 12),
    (12, 13),
    (8, 7),
    (7, 0),
    (0, 4),
    (0, 1),
    (1, 2),
    (2, 3),
    (4, 5),
    (5, 6)
]

def read_h36m(args):
    cam2real = np.array([[1, 0, 0],
                         [0, 0, -1],
                         [0, 1, 0]], dtype=np.float32)
    scale_factor = 0.298

    sample_joint_seq = read_pkl('../motion3d/H36M-243/test/%08d.pkl' % args.sequence_number)['data_label']
    sample_joint_seq = sample_joint_seq.transpose(1, 0, 2)
    sample_joint_seq = (sample_joint_seq / scale_factor) @ cam2real
    return sample_joint_seq


def convert_h36m_to_mpi_connection():
    global connections
    new_connections = []
    for connection in connections:
        new_connection = (H36M_TO_MPI[connection[0]], H36M_TO_MPI[connection[1]])
        new_connections.append(new_connection)
    connections = new_connections


def read_mpi(args):
    @dataclass
    class DatasetArgs:
        data_root: str
        n_frames: int
        stride: int
        flip: bool

    dataset_args = DatasetArgs('../motion3d/', 243, 81, False)

    dataset = MPI3DHP(dataset_args, train=True)

    _, sequence_3d = dataset[args.sequence_number]
    sequence_3d = sequence_3d.cpu().numpy()

    cam2real = np.array([[1, 0, 0],
                         [0, 0, -1],
                         [0, -1, 0]], dtype=np.float32)
    sequence_3d = sequence_3d.transpose(1, 0, 2)
    sequence_3d[14, ...] = 0
    sequence_3d = sequence_3d @ cam2real
    convert_h36m_to_mpi_connection()
    return sequence_3d

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--sequence-number', type=int, default=0)
    parser.add_argument('--dataset', choices=['h36m', 'mpi'], default='h36m')
    args = parser.parse_args()

    print(f"Visualizing sequence {args.sequence_number} of {args.dataset} dataset")

    def update(frame):
        ax.clear()

        ax.set_xlim3d([min_value[0], max_value[0]])
        ax.set_ylim3d([min_value[1], max_value[1]])
        ax.set_zlim3d([min_value[2], max_value[2]])

        x = sample_joint_seq[:, frame, 0]
        y = sample_joint_seq[:, frame, 1]
        z = sample_joint_seq[:, frame, 2]

        for connection in connections:
            start = sample_joint_seq[connection[0], frame, :]
            end = sample_joint_seq[connection[1], frame, :]
            xs = [start[0], end[0]]
            ys = [start[1], end[1]]
            zs = [start[2], end[2]]
            ax.plot(xs, ys, zs, c='b')

        ax.scatter(x, y, z)

        return ax,

    dataset_reader_mapper = {
        'h36m': read_h36m,
        'mpi': read_mpi,
    }
    sample_joint_seq = dataset_reader_mapper[args.dataset](args)

    print(f"Number of frames: {sample_joint_seq.shape[1]}")

    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')

    min_value = np.min(sample_joint_seq, axis=(0, 1))
    max_value = np.max(sample_joint_seq, axis=(0, 1))

    # create the animation
    ani = FuncAnimation(fig, update, frames=sample_joint_seq.shape[1], interval=50)
    ani.save(f'../{args.dataset}_pose{args.sequence_number}.gif')


if __name__ == '__main__':
    main()
