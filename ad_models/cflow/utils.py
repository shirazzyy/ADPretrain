from typing import List
#明确函数接口的预期输入输出类型
import os
import math
import numpy as np
from PIL import Image
#图像处理库
import torch
from torch import Tensor
import torch.nn.functional as F
import torchvision.transforms as T
#图像处理库

np.random.seed(0)
_GCONST_ = -0.9189385332046727 # ln(sqrt(2*pi))
#设置随机数种子以确保结果可重现


class Score_Observer:
    def __init__(self, name):
        self.name = name
        self.max_epoch = 0
        self.max_score = 0.0
        self.last = 0.0

    def update(self, score, epoch, print_score=True):
        self.last = score
        save_weights = False
        if epoch == 0 or score > self.max_score:
            self.max_score = score
            self.max_epoch = epoch
            save_weights = True
        if print_score:
            self.print_score()
        
        return save_weights

    def print_score(self):
        print('{:s}: \t last: {:.2f} \t max: {:.2f} \t epoch_max: {:d}'.format(
            self.name, self.last, self.max_score, self.max_epoch))
        #1. 该类用于监控和记录分数变化，跟踪最大分数及其对应的训练轮次。
        #2. 提供更新分数和打印分数信息的功能，支持自动保存权重的判断逻辑。
        #3. 通过print_score方法格式化输出当前分数、最大分数和对应轮次信息。
    


def t2np(tensor):
    '''pytorch tensor -> numpy array'''
    return tensor.cpu().data.numpy() if tensor is not None else None
    #1. 该函数能够处理GPU和CPU上的PyTorch张量转换
    #2. 自动处理张量设备迁移，确保转换安全
    #3. 使用detach()方法避免梯度计算，提高转换效率
    #4. 支持任意维度的张量转换为numpy数组


def get_logp(C, z, logdet_J):
    logp = C * _GCONST_ - 0.5*torch.sum(z**2, 1) + logdet_J
    return logp
    #C: 条件向量 z: 变量 logdet_J: 雅可比行列式的对数
    #返回:logp: 对数概率
    #转对数概率避免梯度消失、保持概率分布的单调性，有利于优化过程。

def rescale(x):
    return (x - x.min()) / (x.max() - x.min())
    #将输入数据缩放到[0,1]区间

RESULT_DIR = './cflow/results'
WEIGHT_DIR = './cflow/weights'
MODEL_DIR  = './cflow/models'
#CFLOW模型的结果存储路径，分别用于保存测试结果、权重文件和模型文件

def save_results(det_roc_obs, seg_roc_obs, seg_pro_obs, model_name, class_name, run_date):
    result = '{:.2f},{:.2f},{:.2f} \t\tfor {:s}/{:s}/{:s} at epoch {:d}/{:d}/{:d} for {:s}\n'.format(
        det_roc_obs.max_score, seg_roc_obs.max_score, seg_pro_obs.max_score,
        det_roc_obs.name, seg_roc_obs.name, seg_pro_obs.name,
        det_roc_obs.max_epoch, seg_roc_obs.max_epoch, seg_pro_obs.max_epoch, class_name)
    if not os.path.exists(RESULT_DIR):
        os.makedirs(RESULT_DIR)
    fp = open(os.path.join(RESULT_DIR, '{}_{}.txt'.format(model_name, run_date)), "w")
    fp.write(result)
    fp.close()
    #1. 该函数用于将检测、分割和PRO的ROC观察结果保存到文件中。
    #2. 使用格式化字符串将结果组织成可读的格式。
    #3. 将结果追加写入以日期命名的结果文件中。

def save_weights(encoder, decoders, model_name, run_date):
    if not os.path.exists(WEIGHT_DIR):
        os.makedirs(WEIGHT_DIR)
    state = {'encoder_state_dict': encoder.state_dict(),
             'decoder_state_dict': [decoder.state_dict() for decoder in decoders]}
    filename = '{}_{}.pt'.format(model_name, run_date)
    path = os.path.join(WEIGHT_DIR, filename)
    torch.save(state, path)
    print('Saving weights to {}'.format(filename))
    #定义保存权重参数的路径

def load_weights(encoder, decoders, filename):
    path = os.path.join(filename)
    state = torch.load(path)
    encoder.load_state_dict(state['encoder_state_dict'], strict=False)
    decoders = [decoder.load_state_dict(state, strict=False) for decoder, state in zip(decoders, state['decoder_state_dict'])]
    print('Loading weights from {}'.format(filename))
    #定义加载权重参数的函数
    
def adjust_learning_rate(c, optimizer, epoch):
    lr = c['lr']
    if c['lr_cosine']:
        eta_min = lr * (c['lr_decay_rate'] ** 3)
        lr = eta_min + (lr - eta_min) * (
                1 + math.cos(math.pi * epoch / c['meta_epochs'])) / 2
    else:
        steps = np.sum(epoch >= np.asarray(c['lr_decay_epochs']))
        if steps > 0:
            lr = lr * (c['lr_decay_rate'] ** steps)

    for param_group in optimizer.param_groups:
        param_group['lr'] = lr
    #实现了两种学习率调整策略：余弦退火和阶梯式衰减
    #余弦退火策略使学习率平滑下降，有助于模型收敛
    #阶梯式衰减在特定轮次点降低学习率，适用于分阶段训练
    #支持同时更新优化器中多个参数组的学习率
    #返回调整后的学习率值，便于监控和记录


def warmup_learning_rate(c, epoch, batch_id, total_batches, optimizer):
    if c['lr_warm'] and epoch < c['lr_warm_epochs']:
        p = (batch_id + epoch * total_batches) / \
            (c['lr_warm_epochs'] * total_batches)
        lr = c['lr_warmup_from'] + p * (c['lr_warmup_to'] - c['lr_warmup_from'])
        for param_group in optimizer.param_groups:
            param_group['lr'] = lr
    #
    for param_group in optimizer.param_groups:
        lrate = param_group['lr']
    return lrate
    #1. 该函数实现学习率预热机制，在训练初期逐步增加学习率
    #2. 通过线性插值计算预热阶段的学习率值
    #3. 支持多参数组优化器的统一学习率调整
    #4. 预热完成后返回当前实际学习率值
    #在 warmup_learning_rate 函数中，参数 c 是一个配置字典（configuration dictionary）：
'''c['lr_warm']：布尔值，用于控制是否启用学习率预热机制
c['lr_warm_epochs']：整数，预热阶段的训练轮次
c['lr_warmup_from']：预热开始时的学习率值
c['lr_warmup_to']：预热结束时的目标学习率值'''

def get_matched_ref_features(features: List[Tensor], ref_features: List[Tensor]) -> List[Tensor]:
    """
    Get matched reference features for one class.
    """
    matched_ref_features = []
    for layer_id in range(len(features)):
        feature = features[layer_id]
        B, C, H, W = feature.shape
        feature = feature.permute(0, 2, 3, 1).reshape(-1, C).contiguous()  # (N1, C)
        '''将特征图从四维张量转换为二维张量:feature 原本的形状是 (B, C, H, W)，其中 B 是批次大小，C 是通道数，H 和 W 分别是高度和宽度。
        通过 permute(0, 2, 3, 1) 操作，将维度顺序从 (B, C, H, W) 调整为 (B, H, W, C)，即把通道维度移到最后。接着使用 reshape(-1, C) 
        将高度和宽度维度合并，同时保持通道维度不变，最终得到形状为 (N1, C) 的张量，其中 N1 = B × H × W。'''
        feature_n = F.normalize(feature, p=2, dim=1)
        coreset = ref_features[layer_id]  # (N2, C)
        coreset_n = F.normalize(coreset, p=2, dim=1)
        
        dist = feature_n @ coreset_n.T
        cidx = torch.argmax(dist, dim=1)
        index_feats = coreset[cidx]
        index_feats = index_feats.reshape(B, H, W, C).permute(0, 3, 1, 2)
        matched_ref_features.append(index_feats)
    
    return matched_ref_features
    #匹配参考特征与目标特征‌


def get_residual_features(features: List[Tensor], ref_features: List[Tensor], pos_flag: bool = False) -> List[Tensor]:
    residual_features = []
    for layer_id in range(len(features)):
        fi = features[layer_id]  # (B, dim, h, w)
        pi = ref_features[layer_id]  # (B, dim, h, w)
        
        if not pos_flag:
            ri = fi - pi
        else:
            ri = F.mse_loss(fi, pi, reduction='none')
        residual_features.append(ri)
    
    return residual_features


def get_random_normal_images(class_name, num_shot=1):
    if class_name in ['bottle', 'cable', 'capsule', 'carpet', 'grid',
               'hazelnut', 'leather', 'metal_nut', 'pill', 'screw',
               'tile', 'toothbrush', 'transistor', 'wood', 'zipper']:
        path = os.path.join('/data/data1/yxc/datasets/mvtec_anomaly_detection', class_name, 'train', 'good')
    elif class_name in ['candle', 'capsules', 'cashew', 'chewinggum', 'fryum',
               'macaroni1', 'macaroni2', 'pcb1', 'pcb2', 'pcb3', 'pcb4', 'pipe_fryum']:
        path = os.path.join('/data/data1/yxc/datasets/visa', class_name, 'Data', 'Images', 'Normal')
    else:
        raise ValueError('Unrecognized class_name!')
    
    filenames = os.listdir(path)
    n_idxs = np.random.randint(len(filenames), size=num_shot)
    n_idxs = n_idxs.tolist()
    normal_paths = []
    for n_idx in n_idxs:
        normal_paths.append(os.path.join(path, filenames[n_idx]))
    
    return normal_paths


def get_reference_features(encoder, class_names, device):
    images_list = []
    for class_name in class_names:
        normal_paths = get_random_normal_images(class_name)
        images = load_and_transform_vision_data(normal_paths, device)
        images_list.append(images)
    images = torch.cat(images_list, dim=0)
    with torch.no_grad():
        features = encoder.encode_image_from_tensors(images)
        for i in range(len(features)):
            b, l, c = features[i].shape
            features[i] = features[i].reshape(len(class_names), -1, c)
        
    return features


def get_mc_matched_reference_features(features, class_names, ref_features):
    matched_ref_features = [[] for _ in range(len(features))]
    for idx, _ in enumerate(class_names):  # for each image
        for layer_id in range(len(features)):  # for all layers of one image
            feature = features[layer_id][idx:idx+1]
            _, C, H, W = feature.shape
            
            feature = feature.permute(0, 2, 3, 1).reshape(-1, C).contiguous()  # (N1, C)
            feature_n = F.normalize(feature, p=2, dim=1)  # normalized features
            ref_feature = ref_features[layer_id][idx:idx+1].squeeze(0)  # (N2, C)
            ref_feature_n = F.normalize(ref_feature, p=2, dim=1)  # normalized features
            
            dist = feature_n @ ref_feature_n.T  # (N1, N2)
            cidx = torch.argmax(dist, dim=1)
            index_feats = ref_feature[cidx]
            index_feats = index_feats.permute(1, 0).reshape(C, H, W)
            matched_ref_features[layer_id].append(index_feats)
            
    matched_ref_features = [torch.stack(item, dim=0) for item in matched_ref_features]
    
    return matched_ref_features


def load_and_transform_vision_data(image_paths, device):
    if image_paths is None:
        return None

    image_ouputs = []
    for image_path in image_paths:
        data_transform = T.Compose([
                T.Resize(224, T.InterpolationMode.BICUBIC),
                T.CenterCrop(224),
                T.ToTensor(),
                T.Normalize(mean=(0.48145466, 0.4578275, 0.40821073), std=(0.26862954, 0.26130258, 0.27577711))])
        with open(image_path, "rb") as fopen:
            image = Image.open(fopen).convert("RGB")

        image = data_transform(image).to(device)
        image_ouputs.append(image)
    return torch.stack(image_ouputs, dim=0)
