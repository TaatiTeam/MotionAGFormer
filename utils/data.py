import copy
import pickle
import torch 
import numpy as np

class Augmenter2D(object):
    """
        Make 2D augmentations on the fly. PyTorch batch-processing GPU version.
        Adapted from https://github.com/Walter0807/MotionBERT/blob/main/lib/data/augmentation.py#L10
    """
    def __init__(self, args):
        self.d2c_params = read_pkl(args.d2c_params_path)
        self.noise = torch.load(args.noise_path)
        self.mask_ratio = args.mask_ratio
        self.mask_T_ratio = args.mask_T_ratio
        self.num_Kframes = 27
        self.noise_std = 0.002

    def dis2conf(self, dis, a, b, m, s):
        f = a/(dis+a)+b*dis
        shift = torch.randn(*dis.shape)*s + m
        # if torch.cuda.is_available():
        shift = shift.to(dis.device)
        return f + shift
    
    def add_noise(self, motion_2d):
        a, b, m, s = self.d2c_params["a"], self.d2c_params["b"], self.d2c_params["m"], self.d2c_params["s"]
        if "uniform_range" in self.noise.keys():
            uniform_range = self.noise["uniform_range"]
        else:
            uniform_range = 0.06
        motion_2d = motion_2d[:,:,:,:2]
        batch_size = motion_2d.shape[0]
        num_frames = motion_2d.shape[1]
        num_joints = motion_2d.shape[2]
        mean = self.noise['mean'].float()
        std = self.noise['std'].float()
        weight = self.noise['weight'][:,None].float()
        sel = torch.rand((batch_size, self.num_Kframes, num_joints, 1))
        gaussian_sample = (torch.randn(batch_size, self.num_Kframes, num_joints, 2) * std + mean) 
        uniform_sample = (torch.rand((batch_size, self.num_Kframes, num_joints, 2))-0.5) * uniform_range
        noise_mean = 0
        delta_noise = torch.randn(num_frames, num_joints, 2) * self.noise_std + noise_mean
        # if torch.cuda.is_available():
        mean = mean.to(motion_2d.device)
        std = std.to(motion_2d.device)
        weight = weight.to(motion_2d.device)
        gaussian_sample = gaussian_sample.to(motion_2d.device)
        uniform_sample = uniform_sample.to(motion_2d.device)
        sel = sel.to(motion_2d.device)
        delta_noise = delta_noise.to(motion_2d.device)
            
        delta = gaussian_sample*(sel<weight) + uniform_sample*(sel>=weight)
        delta_expand = torch.nn.functional.interpolate(delta.unsqueeze(1), [num_frames, num_joints, 2], mode='trilinear', align_corners=True)[:,0]
        delta_final = delta_expand + delta_noise      
        motion_2d = motion_2d + delta_final 
        dx = delta_final[:,:,:,0]
        dy = delta_final[:,:,:,1]
        dis2 = dx*dx+dy*dy
        dis = torch.sqrt(dis2)
        conf = self.dis2conf(dis, a, b, m, s).clip(0,1).reshape([batch_size, num_frames, num_joints, -1])
        return torch.cat((motion_2d, conf), dim=3)
        
    def add_mask(self, x):
        ''' motion_2d: (N,T,17,3)
        '''
        N,T,J,C = x.shape
        mask = torch.rand(N,T,J,1, dtype=x.dtype, device=x.device) > self.mask_ratio
        mask_T = torch.rand(1,T,1,1, dtype=x.dtype, device=x.device) > self.mask_T_ratio
        x = x * mask * mask_T
        return x
    
    def augment2D(self, motion_2d, mask=False, noise=False):     
        if noise:
            motion_2d = self.add_noise(motion_2d)
        if mask:
            motion_2d = self.add_mask(motion_2d)
        return motion_2d
    


def resample(ori_len, target_len, replay=False, randomness=True):
    """Adapted from https://github.com/Walter0807/MotionBERT/blob/main/lib/utils/utils_data.py#L68"""
    if replay:
        if ori_len > target_len:
            st = np.random.randint(ori_len - target_len)
            return range(st, st + target_len)  # Random clipping from sequence
        else:
            return np.array(range(target_len)) % ori_len  # Replay padding
    else:
        if randomness:
            even = np.linspace(0, ori_len, num=target_len, endpoint=False)
            if ori_len < target_len:
                low = np.floor(even)
                high = np.ceil(even)
                sel = np.random.randint(2, size=even.shape)
                result = np.sort(sel * low + (1 - sel) * high)
            else:
                interval = even[1] - even[0]
                result = np.random.random(even.shape) * interval + even
            result = np.clip(result, a_min=0, a_max=ori_len - 1).astype(np.uint32)
        else:
            result = np.linspace(0, ori_len, num=target_len, endpoint=False, dtype=int)
        return result


def split_clips(vid_list, n_frames, data_stride):
    """Adapted from https://github.com/Walter0807/MotionBERT/blob/main/lib/utils/utils_data.py#L91"""
    result = []
    n_clips = 0
    st = 0
    i = 0
    saved = set()
    while i < len(vid_list):
        i += 1
        if i - st == n_frames:
            result.append(range(st, i))
            saved.add(vid_list[i - 1])
            st = st + data_stride
            n_clips += 1
        if i == len(vid_list):
            break
        if vid_list[i] != vid_list[i - 1]:
            if not (vid_list[i - 1] in saved):
                resampled = resample(i - st, n_frames) + st
                result.append(resampled)
                saved.add(vid_list[i - 1])
            st = i
    return result


def read_pkl(data_url):
    file = open(data_url, 'rb')
    content = pickle.load(file)
    file.close()
    return content


def flip_data(data, left_joints=[1, 2, 3, 14, 15, 16], right_joints=[4, 5, 6, 11, 12, 13]):
    """
    data: [N, F, 17, D] or [F, 17, D]
    """
    flipped_data = copy.deepcopy(data)
    flipped_data[..., 0] *= -1  # flip x of all joints
    flipped_data[..., left_joints + right_joints, :] = flipped_data[..., right_joints + left_joints, :]  # Change orders
    return flipped_data


def crop_scale(motion, scale_range=[1, 1]):
    '''
        Motion: [(M), T, 17, 3].
        Normalize to [-1, 1]
    '''
    result = copy.deepcopy(motion)
    valid_coords = motion[motion[..., 2]!=0][:,:2]
    if len(valid_coords) < 4:
        return np.zeros(motion.shape)
    xmin = min(valid_coords[:,0])
    xmax = max(valid_coords[:,0])
    ymin = min(valid_coords[:,1])
    ymax = max(valid_coords[:,1])
    ratio = np.random.uniform(low=scale_range[0], high=scale_range[1], size=1)[0]
    scale = max(xmax-xmin, ymax-ymin) * ratio
    if scale==0:
        return np.zeros(motion.shape)
    xs = (xmin+xmax-scale) / 2
    ys = (ymin+ymax-scale) / 2
    result[...,:2] = (motion[..., :2]- [xs,ys]) / scale
    result[...,:2] = (result[..., :2] - 0.5) * 2
    result = np.clip(result, -1, 1)
    return result


def posetrack2h36m(x):
    '''
        Input: x (T x V x C)

        PoseTrack keypoints = [ 'nose',
                                'head_bottom',
                                'head_top',
                                'left_ear',
                                'right_ear',
                                'left_shoulder',
                                'right_shoulder',
                                'left_elbow',
                                'right_elbow',
                                'left_wrist',
                                'right_wrist',
                                'left_hip',
                                'right_hip',
                                'left_knee',
                                'right_knee',
                                'left_ankle',
                                'right_ankle']
        H36M:
        0: 'root',
        1: 'rhip',
        2: 'rkne',
        3: 'rank',
        4: 'lhip',
        5: 'lkne',
        6: 'lank',
        7: 'belly',
        8: 'neck',
        9: 'nose',
        10: 'head',
        11: 'lsho',
        12: 'lelb',
        13: 'lwri',
        14: 'rsho',
        15: 'relb',
        16: 'rwri'
    '''
    y = np.zeros(x.shape)
    y[:,0,:] = (x[:,11,:] + x[:,12,:]) * 0.5
    y[:,1,:] = x[:,12,:]
    y[:,2,:] = x[:,14,:]
    y[:,3,:] = x[:,16,:]
    y[:,4,:] = x[:,11,:]
    y[:,5,:] = x[:,13,:]
    y[:,6,:] = x[:,15,:]
    y[:,8,:] = x[:,1,:]
    y[:,7,:] = (y[:,0,:] + y[:,8,:]) * 0.5
    y[:,9,:] = x[:,0,:]
    y[:,10,:] = x[:,2,:]
    y[:,11,:] = x[:,5,:]
    y[:,12,:] = x[:,7,:]
    y[:,13,:] = x[:,9,:]
    y[:,14,:] = x[:,6,:]
    y[:,15,:] = x[:,8,:]
    y[:,16,:] = x[:,10,:]
    y[:,0,2] = np.minimum(x[:,11,2], x[:,12,2])
    y[:,7,2] = np.minimum(y[:,0,2], y[:,8,2])
    return y


def normalize_screen_coordinates(X, w, h):
    assert X.shape[-1] == 2
    return X / w * 2 - [1, h / w]


def denormalize(pred, seq):
    out = pred.cpu().numpy()
    for idx in range(out.shape[0]):
        if seq[idx] in ['TS5', 'TS6']:
            res_w, res_h = 1920, 1080
        else:
            res_w, res_h = 2048, 2048
        out[idx, :, :, :2] = (out[idx, :, :, :2] + np.array([1, res_h / res_w])) * res_w / 2
        out[idx, :, :, 2:] = out[idx, :, :, 2:] * res_w / 2
    out = out - out[..., 0:1, :]
    return torch.tensor(out).cuda()