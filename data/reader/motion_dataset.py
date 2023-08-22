import os
import random
import json
from collections import defaultdict

import numpy as np
from torch.utils.data import Dataset
import torch

from utils.data import read_pkl, flip_data, crop_scale, posetrack2h36m, split_clips, resample, normalize_screen_coordinates
from data.reader.generator_3dhp import ChunkedGenerator

class Fusion(Dataset):
    def __init__(self, opt, train=True):
        self.train = train
        pad = (opt.n_frames - 1) // 2

        self.test_aug = opt.test_augmentation
        if self.train:
            self.poses_train, self.poses_train_2d = self.prepare_data(opt.data_root, train=True)
            self.generator = ChunkedGenerator(opt.test_batch_size, None, self.poses_train,
                                              self.poses_train_2d, None, chunk_length=1, pad=pad,
                                              augment=opt.data_augmentation, reverse_aug=opt.reverse_augmentation,
                                              kps_left=self.kps_left, kps_right=self.kps_right,
                                              joints_left=self.joints_left,
                                              joints_right=self.joints_right, out_all=opt.out_all, train = True)
            print('INFO: Training on {} frames'.format(self.generator.num_frames()))
        else:
            self.poses_test, self.poses_test_2d, self.valid_frame = self.prepare_data(opt.data_root, train=False)
            self.generator = ChunkedGenerator(opt.test_batch_size, None, self.poses_test,
                                                self.poses_test_2d, self.valid_frame,
                                                pad=pad, augment=False, kps_left=self.kps_left,
                                                kps_right=self.kps_right, joints_left=self.joints_left,
                                                joints_right=self.joints_right, train = False)
        self.key_index = self.generator.saved_index
        print('INFO: Testing on {} frames'.format(self.generator.num_frames()))

    def prepare_data(self, path, train=True):
        out_poses_3d = {}
        out_poses_2d = {}
        valid_frame={}

        self.kps_left, self.kps_right = [5, 6, 7, 11, 12, 13], [2, 3, 4, 8, 9, 10]
        self.joints_left, self.joints_right = [5, 6, 7, 11, 12, 13], [2, 3, 4, 8, 9, 10]

        if train == True:
            data = np.load(os.path.join(path, "data_train_3dhp.npz"),allow_pickle=True)['data'].item()
            for seq in data.keys():
                for cam in data[seq][0].keys():
                    anim = data[seq][0][cam]

                    subject_name, seq_name = seq.split(" ")

                    data_3d = anim['data_3d']
                    data_3d[:, :14] -= data_3d[:, 14:15]
                    data_3d[:, 15:] -= data_3d[:, 14:15]
                    out_poses_3d[(subject_name, seq_name, cam)] = data_3d

                    data_2d = anim['data_2d']
                    data_2d[..., :2] = normalize_screen_coordinates(data_2d[..., :2], w=2048, h=2048)
                    # Adding 1 as confidence scores since MotionAGFormer needs (x, y, conf_score)
                    confidence_scores = np.ones((*data_2d.shape[:2], 1))
                    data_2d = np.concatenate((data_2d, confidence_scores), axis=-1)
                    
                    out_poses_2d[(subject_name, seq_name, cam)]=data_2d

            return out_poses_3d, out_poses_2d
        else:
            data = np.load(os.path.join(path, "data_test_3dhp.npz"), allow_pickle=True)['data'].item()
            for seq in data.keys():

                anim = data[seq]

                valid_frame[seq] = anim["valid"]

                data_3d = anim['data_3d']
                data_3d[:, :14] -= data_3d[:, 14:15]
                data_3d[:, 15:] -= data_3d[:, 14:15]
                out_poses_3d[seq] = data_3d

                data_2d = anim['data_2d']

                if seq == "TS5" or seq == "TS6":
                    width = 1920
                    height = 1080
                else:
                    width = 2048
                    height = 2048
                data_2d[..., :2] = normalize_screen_coordinates(data_2d[..., :2], w=width, h=height)
                # Adding 1 as confidence scores since MotionAGFormer needs (x, y, conf_score)
                confidence_scores = np.ones((*data_2d.shape[:2], 1))
                data_2d = np.concatenate((data_2d, confidence_scores), axis=-1)
                out_poses_2d[seq] = data_2d

            return out_poses_3d, out_poses_2d, valid_frame

    def __len__(self):
        return len(self.generator.pairs)

    def __getitem__(self, index):
        seq_name, start_3d, end_3d, flip, reverse = self.generator.pairs[index]

        cam, gt_3D, input_2D, seq, _, _ = self.generator.get_batch(seq_name, start_3d, end_3d, flip, reverse)

        if self.train == False and self.test_aug:
            _, _, input_2D_aug, _, _,_ = self.generator.get_batch(seq_name, start_3d, end_3d, flip=True, reverse=reverse)
            input_2D = np.concatenate((np.expand_dims(input_2D,axis=0),np.expand_dims(input_2D_aug,axis=0)),0)
            
        bb_box = np.array([0, 0, 1, 1])
        input_2D_update = input_2D

        scale = float(1.0)

        if self.train == True:
            return cam, gt_3D, input_2D_update, seq, scale, bb_box
        else:
            return cam, gt_3D, input_2D_update, seq, scale, bb_box
            

class MPI3DHP(Dataset):
    def __init__(self, args, train=True):
        self.train = train
        self.poses_3d, self.poses_2d, self.poses_3d_valid_frames, self.seq_names = self.prepare_data(args)
        self.normalized_poses3d = self.normalize_poses()
        self.flip = args.flip
        self.left_joints = [8, 9, 10, 2, 3, 4]
        self.right_joints = [11, 12, 13, 5, 6, 7]

    def normalize_poses(self):
        normalized_poses_3d = []
        if self.train:
            for pose_sequence in self.poses_3d: # pose_sequence dim is (T, J, 3)
                width = 2048
                height = 2048
                normalized_sequence = pose_sequence.copy()
                normalized_sequence[..., :2]  = normalized_sequence[..., :2] / width * 2 - [1, height / width]
                normalized_sequence[..., 2:] = normalized_sequence[..., 2:] / width * 2

                normalized_sequence = normalized_sequence - normalized_sequence[:, 14:15, :]
                
                normalized_poses_3d.append(normalized_sequence[None, ...])
        else:
            for seq_name, pose_sequence in zip(self.seq_names, self.poses_3d): # pose_sequence dim is (T, J, 3)
                if seq_name in ["TS5", "TS6"]:
                    width = 1920
                    height = 1080
                else:
                    width = 2048
                    height = 2048
                normalized_sequence = pose_sequence.copy()
                normalized_sequence[..., :2]  = normalized_sequence[..., :2] / width * 2 - [1, height / width]
                normalized_sequence[..., 2:] = normalized_sequence[..., 2:] / width * 2

                normalized_sequence = normalized_sequence - normalized_sequence[:, 14:15, :]
                
                normalized_poses_3d.append(normalized_sequence[None, ...])

        normalized_poses_3d = np.concatenate(normalized_poses_3d, axis=0)
        
        return normalized_poses_3d

    def prepare_data(self, args):
        poses_2d, poses_3d, poses_3d_valid_frames, seq_names = [], [], [], []
        data_file = "data_train_3dhp.npz" if self.train else "data_test_3dhp.npz"
        data = np.load(os.path.join(args.data_root, data_file),allow_pickle=True)['data'].item()
        n_frames, stride = args.n_frames, args.stride if self.train else args.n_frames

        for seq in data.keys():
            if self.train:
                for cam in data[seq][0].keys():
                    anim = data[seq][0][cam]

                    data_3d_partitioned, data_2d_partitioned, _ = self.extract_poses(anim, seq, n_frames, stride)
                    poses_3d.extend(data_3d_partitioned)
                    poses_2d.extend(data_2d_partitioned)
            else:
                anim = data[seq]
                valid_frames = anim['valid']

                data_3d_partitioned, data_2d_partitioned, valid_frames_partitioned = self.extract_poses(anim, seq, n_frames, stride, valid_frames)
                poses_3d.extend(data_3d_partitioned)
                poses_2d.extend(data_2d_partitioned)
                seq_names.extend([seq] * len(data_3d_partitioned))
                poses_3d_valid_frames.extend(valid_frames_partitioned)
        
        poses_3d = np.concatenate(poses_3d, axis=0)
        poses_2d = np.concatenate(poses_2d, axis=0)
        if len(poses_3d_valid_frames) > 0:
            poses_3d_valid_frames = np.concatenate(poses_3d_valid_frames, axis=0)

        return poses_3d, poses_2d, poses_3d_valid_frames, seq_names

    def __len__(self):
        return self.poses_3d.shape[0]
    
    def extract_poses(self, anim, seq, n_frames, stride, valid_frames=None):
        data_3d = anim['data_3d']
        # data_3d -= data_3d[:, 14:15]
        # data_3d[..., 2] -= data_3d[:, 14:15, 2]
        data_3d_partitioned, valid_frames_partitioned = self.partition(data_3d, clip_length=n_frames, stride=stride, valid_frames=valid_frames)

        data_2d = anim['data_2d']
        if seq in ["TS5", "TS6"]:
            width = 1920
            height = 1080
        else:
            width = 2048
            height = 2048

        data_2d[..., :2] = self.normalize_screen_coordinates(data_2d[..., :2], w=width, h=height)
        # Adding 1 as confidence scores since MotionAGFormer needs (x, y, conf_score)
        confidence_scores = np.ones((*data_2d.shape[:2], 1))
        data_2d = np.concatenate((data_2d, confidence_scores), axis=-1)
        data_2d_partitioned, _ = self.partition(data_2d, clip_length=n_frames, stride=stride)

        return data_3d_partitioned, data_2d_partitioned, valid_frames_partitioned

    @staticmethod
    def normalize_screen_coordinates(X, w, h):
        assert X.shape[-1] == 2
        return X / w * 2 - [1, h / w]
    
    def partition(self, data, clip_length=243, stride=81, valid_frames=None):
        """Partitions data (n_frames, 17, 3) into list of (clip_length, 17, 3) data with given stride"""
        data_list, valid_list = [], []
        n_frames = data.shape[0]
        for i in range(0, n_frames, stride):
            sequence = data[i:i+clip_length]
            sequence_length = sequence.shape[0]
            if sequence_length == clip_length:
                data_list.append(sequence[None, ...])
            else:
                new_indices = self.resample(sequence_length, clip_length)
                extrapolated_sequence = sequence[new_indices]
                data_list.append(extrapolated_sequence[None, ...])

        if valid_frames is not None:
            for i in range(0, n_frames, stride):
                valid_sequence = valid_frames[i:i+clip_length]
                sequence_length = valid_sequence.shape[0]
                if sequence_length == clip_length:
                    valid_list.append(valid_sequence[None, ...])
                else:
                    new_indices = self.resample(sequence_length, clip_length)
                    extrapolated_sequence = valid_sequence[new_indices]
                    valid_list.append(extrapolated_sequence[None, ...])

        return data_list, valid_list

    @staticmethod
    def resample(original_length, target_length):
        """
        Returns an array that has indices of frames. elements of array are in range (0, original_length -1) and
        we have target_len numbers (So it interpolates the frames)
        """
        even = np.linspace(0, original_length, num=target_length, endpoint=False)
        result = np.floor(even)
        result = np.clip(result, a_min=0, a_max=original_length - 1).astype(np.uint32)
        return result

    def __getitem__(self, index):
        pose_2d = self.poses_2d[index]
        pose_3d_normalized = self.normalized_poses3d[index]
        
        if not self.train:
            valid_frames = self.poses_3d_valid_frames[index]
            pose_3d = self.poses_3d[index]
            seq_name = self.seq_names[index]
            return torch.FloatTensor(pose_2d), torch.FloatTensor(pose_3d_normalized), torch.FloatTensor(pose_3d), \
                   torch.IntTensor(valid_frames), seq_name
        
        if self.flip and random.random() > 0.5:
            pose_2d = flip_data(pose_2d, self.left_joints, self.right_joints)
            pose_3d_normalized = flip_data(pose_3d_normalized, self.left_joints, self.right_joints)

        return torch.FloatTensor(pose_2d), torch.FloatTensor(pose_3d_normalized)
            

class PoseTrackDataset2D(Dataset):
    def __init__(self, flip=True, scale_range=[0.25, 1], data_root_2d='data/motion2d/', n_frames=243, data_stride=81):
        super(PoseTrackDataset2D, self).__init__()
        self.n_frames = n_frames
        self.data_stride = data_stride
        self.flip = flip
        data_root = os.path.join(data_root_2d, 'posetrack18_annotations/train/')
        file_list = sorted(os.listdir(data_root))
        all_motions = []
        all_motions_filtered = []
        self.scale_range = scale_range
        for filename in file_list:
            with open(os.path.join(data_root, filename), 'r') as file:
                json_dict = json.load(file)
                annots = json_dict['annotations']
                motions = defaultdict(list)
                for annot in annots:
                    tid = annot['track_id']
                    pose2d = np.array(annot['keypoints']).reshape(-1,3)
                    motions[tid].append(pose2d)
            all_motions += list(motions.values())
        for motion in all_motions:
            if len(motion)<30:
                continue
            motion = np.array(motion[:30])
            if np.sum(motion[:,:,2]) <= 306:  # Valid joint num threshold
                continue
            motion = crop_scale(motion, self.scale_range) 
            motion = posetrack2h36m(motion)
            motion[motion[:,:,2]==0] = 0
            if np.sum(motion[:,0,2]) < 30:
                continue                      # Root all visible (needed for framewise rootrel)
            all_motions_filtered.append(motion)

        all_motions_filtered = self.extrapolate(all_motions_filtered)
        all_motions_filtered = np.array(all_motions_filtered)
        self.motions_2d = all_motions_filtered
    
    def extrapolate(self, motions):
        extrapolated_motions = []
        for motion in motions:
            new_indices = resample(motion.shape[0], self.n_frames)
            extrapolated_motions.append(motion[new_indices])
        return extrapolated_motions

    def __len__(self):
        'Denotes the total number of samples'
        return len(self.motions_2d)

    def __getitem__(self, index):
        'Generates one sample of data'
        motion_2d = torch.FloatTensor(self.motions_2d[index])
        if self.flip and random.random()>0.5:                       
            motion_2d = flip_data(motion_2d)
        return motion_2d, motion_2d
    

class InstaVDataset2D(Dataset):
    def __init__(self, n_frames=243, data_stride=81, flip=True, valid_threshold=0.0, scale_range=[0.25, 1],
                 data_root_2d='data/motion2d/'):
        super(InstaVDataset2D, self).__init__()
        self.flip = flip
        self.scale_range = scale_range
        motion_all = np.load(os.path.join(data_root_2d, 'InstaVariety/motion_all.npy'))
        id_all = np.load(os.path.join(data_root_2d, 'InstaVariety/id_all.npy'))
        split_id = split_clips(id_all, n_frames, data_stride)  
        motions_2d = motion_all[split_id]                        # [N, T, 17, 3]
        valid_idx = (motions_2d[:,0,0,2] > valid_threshold)
        self.motions_2d = motions_2d[valid_idx]
        
    def __len__(self):
        'Denotes the total number of samples'
        return len(self.motions_2d)

    def __getitem__(self, index):
        'Generates one sample of data'
        motion_2d = self.motions_2d[index]
        motion_2d = crop_scale(motion_2d, self.scale_range) 
        motion_2d[motion_2d[:,:,2]==0] = 0
        if self.flip and random.random()>0.5:                       
            motion_2d = flip_data(motion_2d)
        motion_2d = torch.FloatTensor(motion_2d)
        return motion_2d, motion_2d
        
    
class MotionDataset3D(Dataset):
    def __init__(self, args, subset_list, data_split, return_stats=False):
        """
        :param args: Arguments from the config file
        :param subset_list: A list of datasets
        :param data_split: Either 'train' or 'test'
        """
        np.random.seed(0)
        self.data_root = args.data_root
        self.add_velocity = args.add_velocity
        self.subset_list = subset_list
        self.data_split = data_split
        self.return_stats = return_stats

        self.flip = args.flip
        self.use_proj_as_2d = args.use_proj_as_2d

        self.file_list = self._generate_file_list()

    def _generate_file_list(self):
        file_list = []
        for subset in self.subset_list:
            data_path = os.path.join(self.data_root, subset, self.data_split)
            motion_list = sorted(os.listdir(data_path))
            for i in motion_list:
                file_list.append(os.path.join(data_path, i))
        return file_list

    @staticmethod
    def _construct_motion2d_by_projection(motion_3d):
        """Constructs 2D pose sequence by projecting the 3D pose orthographically"""
        motion_2d = np.zeros(motion_3d.shape, dtype=np.float32)
        motion_2d[:, :, :2] = motion_3d[:, :, :2]  # Get x and y from the 3D pose
        motion_2d[:, :, 2] = 1  # Set confidence score as 1
        return motion_2d

    def __len__(self):
        return len(self.file_list)

    def __getitem__(self, idx):
        file_path = self.file_list[idx]
        motion_file = read_pkl(file_path)

        motion_2d = motion_file["data_input"]
        motion_3d = motion_file["data_label"]

        if motion_2d is None or self.use_proj_as_2d:
            motion_2d = self._construct_motion2d_by_projection(motion_3d)

        if self.add_velocity:
            motion_2d_coord = motion_2d[..., :2]
            velocity_motion_2d = motion_2d_coord[1:] - motion_2d_coord[:-1]
            motion_2d = motion_2d[:-1]
            motion_2d = np.concatenate((motion_2d, velocity_motion_2d), axis=-1)

            motion_3d = motion_3d[:-1]

        if self.data_split == 'train':
            if self.flip and random.random() > 0.5:
                motion_2d = flip_data(motion_2d)
                motion_3d = flip_data(motion_3d)

        if self.return_stats:
            return torch.FloatTensor(motion_2d), torch.FloatTensor(motion_3d), motion_file['mean'], motion_file['std']
        else:
            return torch.FloatTensor(motion_2d), torch.FloatTensor(motion_3d)
