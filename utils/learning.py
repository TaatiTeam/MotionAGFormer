from model.MotionAGFormer import MotionAGFormer
from torch import nn
import torch


class AverageMeter(object):
    """Computes and stores the average and current value"""

    def __init__(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count


def load_model(args):
    act_mapper = {
        "gelu": nn.GELU,
        'relu': nn.ReLU
    }

    if args.model_name == "MotionAGFormer":
        model = MotionAGFormer(n_layers=args.n_layers,
                               dim_in=args.dim_in,
                               dim_feat=args.dim_feat,
                               dim_rep=args.dim_rep,
                               dim_out=args.dim_out,
                               mlp_ratio=args.mlp_ratio,
                               act_layer=act_mapper[args.act_layer],
                               attn_drop=args.attn_drop,
                               drop=args.drop,
                               drop_path=args.drop_path,
                               use_layer_scale=args.use_layer_scale,
                               layer_scale_init_value=args.layer_scale_init_value,
                               use_adaptive_fusion=args.use_adaptive_fusion,
                               num_heads=args.num_heads,
                               qkv_bias=args.qkv_bias,
                               qkv_scale=args.qkv_scale,
                               hierarchical=args.hierarchical,
                               num_joints=args.num_joints,
                               use_temporal_similarity=args.use_temporal_similarity,
                               temporal_connection_len=args.temporal_connection_len,
                               use_tcn=args.use_tcn,
                               graph_only=args.graph_only,
                               neighbour_num=args.neighbour_num,
                               n_frames=args.n_frames)
    else:
        raise Exception("Undefined model name")

    return model


def load_pretrained_weights(model, checkpoint):
    """
    Load pretrained weights to model
    Incompatible layers (unmatched in name or size) will be ignored
    Args:
    - model (nn.Module): network model, which must not be nn.DataParallel
    - checkpoint (dict): the checkpoint
    """
    import collections
    if 'state_dict' in checkpoint:
        state_dict = checkpoint['state_dict']
    else:
        state_dict = checkpoint
    model_dict = model.state_dict()
    new_state_dict = collections.OrderedDict()
    matched_layers, discarded_layers = [], []
    for k, v in state_dict.items():
        # If the pretrained state_dict was saved as nn.DataParallel,
        # keys would contain "module.", which should be ignored.
        if k.startswith('module.'):
            k = k[7:]
        if k in model_dict:
            new_state_dict[k] = v
            matched_layers.append(k)
        else:
            discarded_layers.append(k)
    model_dict.update(new_state_dict)
    model.load_state_dict(model_dict, strict=True)
    print(f'[INFO] (load_pretrained_weights) {len(matched_layers)} layers are loaded')
    print(f'[INFO] (load_pretrained_weights) {len(discarded_layers)} layers are discared')


def accuracy(output, target, topk=(1,)):
    """Computes the accuracy over the k top predictions for the specified values of k"""
    with torch.no_grad():
        maxk = max(topk)
        batch_size = target.size(0)
        _, pred = output.topk(maxk, 1, True, True)
        pred = pred.t()
        correct = pred.eq(target.view(1, -1).expand_as(pred))
        res = []
        for k in topk:
            correct_k = correct[:k].reshape(-1).float().sum(0, keepdim=True)
            res.append(correct_k.mul_(100.0 / batch_size))
        return res


def decay_lr_exponentially(lr, lr_decay, optimizer):
    lr *= lr_decay
    for param_group in optimizer.param_groups:
        param_group['lr'] *= lr_decay
    return lr