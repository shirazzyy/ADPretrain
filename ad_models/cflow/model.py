import math
import torch
from torch import nn
# FrEIA (https://github.com/VLL-HD/FrEIA/)
import FrEIA.framework as Ff
import FrEIA.modules as Fm
#构建可逆神经网络


def positionalencoding2d(D, H, W):#为输入特征图添加空间位置信息
    """
    :param D: dimension of the model
    :param H: H of the positions
    :param W: W of the positions
    :return: DxHxW position matrix
    """
    if D % 4 != 0:
        raise ValueError("Cannot use sin/cos positional encoding with odd dimension (got dim={:d})".format(D))
    P = torch.zeros(D, H, W)
    # Each dimension use half of D
    D = D // 2
    div_term = torch.exp(torch.arange(0.0, D, 2) * -(math.log(1e4) / D))
    pos_w = torch.arange(0.0, W).unsqueeze(1)
    pos_h = torch.arange(0.0, H).unsqueeze(1)
    P[0:D:2, :, :]  = torch.sin(pos_w * div_term).transpose(0, 1).unsqueeze(1).repeat(1, H, 1)
    P[1:D:2, :, :]  = torch.cos(pos_w * div_term).transpose(0, 1).unsqueeze(1).repeat(1, H, 1)
    P[D::2,  :, :]  = torch.sin(pos_h * div_term).transpose(0, 1).unsqueeze(2).repeat(1, 1, W)
    P[D+1::2,:, :]  = torch.cos(pos_h * div_term).transpose(0, 1).unsqueeze(2).repeat(1, 1, W)
    return P


def subnet_fc(dims_in, dims_out):
    return nn.Sequential(nn.Linear(dims_in, 2*dims_in), nn.ReLU(), nn.Linear(2*dims_in, dims_out))
    #通过接收输入跟输出维度返回全连接：包含2个隐藏层，每层使用Relu作为激活函数，最终输出维度为 dims_out


def freia_flow_head(c, n_feat):
    coder = Ff.SequenceINN(n_feat)
    print('NF coder:', n_feat)
    for k in range(c['coupling_blocks']):
        coder.append(Fm.AllInOneBlock, subnet_constructor=subnet_fc, affine_clamping=c['clamp_alpha'],
            global_affine_type='SOFTPLUS', permute_soft=True)
    return coder
    #构建无条件可逆神经网络
    #接收配置参数 c 和特征维度 n_feat，创建了一个 SequenceINN 序列模型，并添加了指定数量的 AllInOneBlock 耦合块。
    #每个耦合块都使用 subnet_fc 作为子网络构造器，并设置了仿射夹紧、全局仿射类型和软置换等参数，以确保模型的可逆性和训练稳定性。


def freia_cflow_head(c, n_feat):
    n_cond = c['condition_vec']
    coder = Ff.SequenceINN(n_feat)
    print('CNF coder:', n_feat)
    for k in range(c['coupling_blocks']):
        coder.append(Fm.AllInOneBlock, cond=0, cond_shape=(n_cond,), subnet_constructor=subnet_fc, affine_clamping=c['clamp_alpha'],
            global_affine_type='SOFTPLUS', permute_soft=True)
    return coder
    #构建条件流模型，通过添加 cond=0 和 cond_shape=(n_cond,) 参数，使网络能够接收额外的条件向量作为输入，从而实现条件生成或条件变换。
    #跟前者使用相同的耦合块结构 Fm.AllInOneBlock 和子网络构造器 subnet_fc。


def load_decoder_arch(c, dim_in):
    if c['dec_arch'] == 'freia-flow':
        decoder = freia_flow_head(c, dim_in)
    elif c['dec_arch'] == 'freia-cflow':
        decoder = freia_cflow_head(c, dim_in)
    else:
        raise NotImplementedError('{} is not supported NF!'.format(c['dec_arch']))
    #print(decoder)
    return decoder
    #根据配置参数动态加载和构建解码器架构‌，它通常用于构建可逆神经网络的解码部分，
    #根据输入的配置参数 c 来决定使用哪种解码器架构。它会检查配置中的 decoder_type 参数，如果设置为 'flow'，
    #则调用 freia_flow_head 函数来构建无条件的流模型解码器；如果设置为 'cflow'，则调用 freia_cflow_head 函数来构建条件流模型解码器。
