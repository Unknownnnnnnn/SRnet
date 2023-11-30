from __future__ import print_function
import os
import argparse
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import numpy as np
from scipy.spatial.transform import Rotation
import math
from functools import partial
from torch.nn import init
class ParameterPredictionNet(nn.Module):
    def __init__(self, weights_dim):
        super().__init__()

        # self._logger = logging.getLogger(self.__class__.__name__)

        self.weights_dim = weights_dim

        # Pointnet
        self.prepool = nn.Sequential(
            nn.Conv1d(4, 64, 1), nn.ReLU(),
            nn.Conv1d(64, 64, 1), nn.ReLU(),
            nn.Conv1d(64, 64, 1), nn.ReLU(),
            nn.Conv1d(64, 128, 1), nn.ReLU(),
            nn.Conv1d(128, 1024, 1), nn.ReLU(),
        )
        self.pooling = nn.AdaptiveMaxPool1d(1)
        self.postpool = nn.Sequential(
            nn.Linear(1024, 512), nn.ReLU(),
            nn.Linear(512, 256), nn.ReLU(),
            nn.Linear(256, 2 + np.prod(weights_dim)),
        )

        # self._logger.info('Predicting weights with dim {}.'.format(self.weights_dim))

    def forward(self, x):
        src_padded = F.pad(x[0], (0, 1), mode='constant', value=0)
        ref_padded = F.pad(x[1], (0, 1), mode='constant', value=1)
        concatenated = torch.cat([src_padded, ref_padded], dim=1)

        prepool_feat = self.prepool(concatenated.permute(0, 2, 1))
        pooled = torch.flatten(self.pooling(prepool_feat), start_dim=-2)
        raw_weights = self.postpool(pooled)

        beta = F.softplus(raw_weights[:, 0])
        alpha = F.softplus(raw_weights[:, 1])

        return beta, alpha
class AutomaticWeightedLoss(nn.Module):
    """automatically weighted multi-task loss
    Params：
        num: int，the number of loss
        x: multi-task loss
    Examples：
        loss1=1
        loss2=2
        awl = AutomaticWeightedLoss(2)
        loss_sum = awl(loss1, loss2)
    """
    def __init__(self, num=2):
        super(AutomaticWeightedLoss, self).__init__()
        params = torch.ones(num, requires_grad=True)
        self.params = torch.nn.Parameter(params)

    def forward(self, *x):
        loss_sum = 0
        self.params.data = self.params.data.clamp(min=0.1, max=1)
        for i, loss in enumerate(x):
            loss_sum += 0.5 / (self.params[i] ** 2) * loss + torch.log(1 + self.params[i] ** 2)
        return loss_sum
    
def sinkhorn(log_alpha, num_iters: int = 5):
    zero_pad = nn.ZeroPad2d((0, 1, 0, 1))
    log_alpha_padded = zero_pad(log_alpha[:, None, :, :])

    log_alpha_padded = torch.squeeze(log_alpha_padded, dim=1)

    for i in range(num_iters):
        # Row normalization
        log_alpha_padded = torch.cat((
                log_alpha_padded[:, :-1, :] - (torch.logsumexp(log_alpha_padded[:, :-1, :], dim=2, keepdim=True)),
                log_alpha_padded[:, -1, None, :]),  # Don't normalize last row
            dim=1)

        # Column normalization
        log_alpha_padded = torch.cat((
                log_alpha_padded[:, :, :-1] - (torch.logsumexp(log_alpha_padded[:, :, :-1], dim=1, keepdim=True)),
                log_alpha_padded[:, :, -1, None]),  # Don't normalize last column
            dim=2)
    return log_alpha_padded
def square_distance(src, dst):
    dist = -2 * torch.matmul(src, dst.transpose(-1, -2)) # B * N M
    dist += torch.sum(src ** 2, dim=-1).unsqueeze(-1) # B * N 1
    dist += torch.sum(dst ** 2, dim=-1).unsqueeze(-2) # B * 1 M
    return dist
class Conv1DBNReLU(nn.Module):
    def __init__(self, in_channel, out_channel, ksize):
        super(Conv1DBNReLU, self).__init__()
        self.conv = nn.Conv1d(in_channel, out_channel, ksize, bias=False)
        self.bn = nn.BatchNorm1d(out_channel)
        self.relu = nn.ReLU()

    def forward(self, x):
        x = self.conv(x)
        x = self.bn(x)
        x = self.relu(x)
        return x
class Conv1DBlock(nn.Module):
    def __init__(self, channels, ksize):
        super(Conv1DBlock, self).__init__()
        self.conv = nn.ModuleList()
        for i in range(len(channels)-2):
            self.conv.append(Conv1DBNReLU(channels[i], channels[i+1], ksize))
        self.conv.append(nn.Conv1d(channels[-2], channels[-1], ksize))

    def forward(self, x):
        for conv in self.conv:
            x = conv(x)
        return x
import torch.nn as nn
import torch.nn.functional as F
import torch, math

class MHAtt(nn.Module):
    def __init__(self, channel):
        super(MHAtt, self).__init__()
        self.HIDDEN_SIZE = channel
        self.linear_v = nn.Conv1d(channel, channel, 1)
        self.linear_k = nn.Conv1d(channel, channel, 1)
        self.linear_q = nn.Conv1d(channel, channel, 1)
        self.linear_merge = nn.Conv1d(channel, channel, 1)
        self.HEAD = 4
        self.dropout = nn.Dropout(0.5)

    def forward(self, v, k, q, mask):
        n_batches = q.size(0)
        v = self.linear_v(v).view(
            n_batches,
            -1,
            self.HEAD,
            self.HIDDEN_SIZE
        ).transpose(1, 2)  # b, head, seq, hidden_dim

        k = self.linear_k(k).view(
            n_batches,
            -1,
            self.HEAD,
            self.HIDDEN_SIZE
        ).transpose(1, 2)

        q = self.linear_q(q).view(
            n_batches,
            -1,
            self.HEAD,
            self.HIDDEN_SIZE
        ).transpose(1, 2)

        atted = self.att(v, k, q, mask)
        atted = atted.transpose(1, 2).contiguous().view(
            n_batches,
            self.HIDDEN_SIZE,
            -1,
        )
        atted = self.linear_merge(atted)

        return atted

    def att(self, value, key, query, mask):
        d_k = query.size(-1) # hidden dim
        
#         b, head, seq, hidden_dim
        
        scores = torch.matmul(
            query, key.transpose(-2, -1)
        ) / math.sqrt(d_k) # (b, head, seq_q, hidden_dim) x (b, head, hidden_dim, seq_k) -> (b,head,seq_q,seq_k)

        if mask is not None: # mask(b, seq_q)
            scores = scores.masked_fill(mask, -1e9)  # value 中 padding部分会
        att_map = F.softmax(scores, dim=-1)  # query中每个词 在 所有value上的概率分布
        att_map = self.dropout(att_map)
        
        return torch.matmul(att_map, value) # (b,head,seq_q,seq_k) x (b, head, seq_k, hidden_)
class LayerNorm(nn.Module):
    def __init__(self, size, eps=1e-6):
        super(LayerNorm, self).__init__()
        self.eps = eps

        self.a_2 = nn.Parameter(torch.ones(size))
        self.b_2 = nn.Parameter(torch.zeros(size))

    def forward(self, x):
        x = x.transpose(2,1)
        mean = x.mean(-1, keepdim=True)
        std = x.std(-1, keepdim=True)
        res = self.a_2 * (x - mean) / (std + self.eps) + self.b_2
        
        return res.transpose(2,1)

# ---------------------------
# ---- Feed Forward Nets ----
# ---------------------------
    
class FFN(nn.Module):
    def __init__(self, channel):
        super(FFN, self).__init__()
        
        self.relu = nn.ReLU()
        self.mlp = nn.Conv1d(channel, channel, 1)
        self.mlp2 = nn.Conv1d(channel, channel, 1)
        
    def forward(self, x):
        x = self.relu(self.mlp(x))
        x = self.mlp2(x)
        return x


# ------------------------
# ---- Self Attention ----
# ------------------------
    
class SA(nn.Module):
    def __init__(self, channel):
        super(SA, self).__init__()

        self.mhatt = MHAtt(channel)
        self.ffn = FFN(channel)

        self.dropout1 = nn.Dropout(0.5)
        self.norm1 = nn.BatchNorm1d(channel)

        self.dropout2 = nn.Dropout(0.5)
        self.norm2 = nn.BatchNorm1d(channel)

    def forward(self, x, x_mask):
        x = x + self.dropout1(
            self.mhatt(x, x, x, x_mask)
        ) # (b, seq_q, hidden_dim)

        x = x + self.dropout2(
            self.ffn(x)
        )

        return x


# -------------------------------
# ---- Self Guided Attention ----
# -------------------------------
class SGA(nn.Module):
    def __init__(self, channel):
        super(SGA, self).__init__()

        self.mhatt1 = MHAtt(channel)
        self.mhatt2 = MHAtt(channel)
        self.ffn = FFN(channel)

        self.dropout1 = nn.Dropout(0.5)
        self.norm1 = nn.BatchNorm1d(channel)

        self.dropout2 = nn.Dropout(0.5)
        self.norm2 = nn.BatchNorm1d(channel)

        self.dropout3 = nn.Dropout(0.5)
        self.norm3 = nn.BatchNorm1d(channel)

    def forward(self, x, y, x_mask, y_mask):
        x = x + self.dropout1(
            self.mhatt1(x, x, x, x_mask)
        )

        x = x + self.dropout2(
            self.mhatt2(y, y, x, y_mask)
        )

        x = x + self.dropout3(
            self.ffn(x)
        )

        return x
class SGA_last(nn.Module):
    def __init__(self, channel):
        super(SGA_last, self).__init__()

        self.mhatt1 = MHAtt(channel)
        self.mhatt2 = MHAtt(channel)
        self.ffn = FFN(channel)

        self.dropout1 = nn.Dropout(0.5)
        self.norm1 = nn.BatchNorm1d(channel)

        self.dropout2 = nn.Dropout(0.5)
        self.norm2 = nn.BatchNorm1d(channel)

        self.dropout3 = nn.Dropout(0.5)
        self.norm3 = nn.BatchNorm1d(channel)

    def forward(self, x, y, x_mask, y_mask):
        x = x + self.dropout1(
            self.mhatt1(x, x, x, x_mask)
        )

        # x = self.norm2(x + self.dropout2(
        #     self.mhatt2(y, y, x, y_mask)
        # ))

        x = self.dropout2(
            self.mhatt2(x, x, y, x_mask)
        )

        x = self.dropout3(
            self.ffn(x)
        )

        return x
class MCA_ED(nn.Module):
    def __init__(self, channel):
        super(MCA_ED, self).__init__()

        self.enc_x = SA(channel)
        self.enc_y = SA(channel)
        self.dec_x = SGA(channel)
        self.dec_y = SGA(channel)
#         nn.ModuleList([SA(channel) for _ in range(1)])
    def forward(self, x, y, mask):
        # Get hidden vector
        if (mask):
            x_mask, y_mask = mask[0], mask[1]
        else:
            x_mask, y_mask = None, None
        x_ = self.enc_x(x, x_mask)
        y_ = self.enc_y(y, y_mask)

#         x = torch.cat([x[:,:4,:], x_], dim =1)
#         y = torch.cat([y[:,:4,:], y_], dim =1)
        x__ = self.dec_x(x_, y_, x_mask, y_mask)
        y__ = self.dec_y(y_, x__, y_mask, x_mask)
        
        return x__, y__
class ResidualAttention(nn.Module):

    def __init__(self, channel=512 , num_class=1000,la=0.2):
        super().__init__()
        self.la=la
        self.fc=nn.Conv2d(in_channels=channel,out_channels=num_class,kernel_size=1,stride=1,bias=False)

    def forward(self, x):
        b,c,h,w=x.shape
        y_raw=self.fc(x).flatten(2) #b,num_class,hxw
        y_avg=torch.mean(y_raw,dim=2) #b,num_class
        y_max=torch.max(y_raw,dim=2)[0] #b,num_class
        score=y_avg+self.la*y_max
        return score

class CoTAttention(nn.Module):

    def __init__(self, dim=512,kernel_size=3):
        super().__init__()
        self.dim=dim
        self.kernel_size=kernel_size

        self.key_embed=nn.Sequential(
            nn.Conv2d(dim,dim,kernel_size=kernel_size,padding=kernel_size//2,groups=4,bias=False),
            nn.BatchNorm2d(dim),
            nn.ReLU()
        )
        self.value_embed=nn.Sequential(
            nn.Conv2d(dim,dim,1,bias=False),
            nn.BatchNorm2d(dim)
        )

        factor=4
        self.attention_embed=nn.Sequential(
            nn.Conv2d(2*dim,2*dim//factor,1,bias=False),
            nn.BatchNorm2d(2*dim//factor),
            nn.ReLU(),
            nn.Conv2d(2*dim//factor,kernel_size*kernel_size*dim,1)
        )


    def forward(self, x):
        bs,c,h,w=x.shape
        k1=self.key_embed(x) #bs,c,h,w
        v=self.value_embed(x).view(bs,c,-1) #bs,c,h,w

        y=torch.cat([k1,x],dim=1) #bs,2c,h,w
        att=self.attention_embed(y) #bs,c*k*k,h,w
        att=att.reshape(bs,c,self.kernel_size*self.kernel_size,h,w)
        att=att.mean(2,keepdim=False).view(bs,c,-1) #bs,c,h*w
        k2=F.softmax(att,dim=-1)*v
        k2=k2.view(bs,c,h,w)


        return k1+k2
class ScaledDotProductAttention(nn.Module):
    '''
    Scaled dot-product attention
    '''

    def __init__(self, d_model, d_k, d_v, h,dropout=.1):
        '''
        :param d_model: Output dimensionality of the model
        :param d_k: Dimensionality of queries and keys
        :param d_v: Dimensionality of values
        :param h: Number of heads
        '''
        super(ScaledDotProductAttention, self).__init__()
        self.fc_q = nn.Linear(d_model, h * d_k)
        self.fc_k = nn.Linear(d_model, h * d_k)
        self.fc_v = nn.Linear(d_model, h * d_v)
        self.fc_o = nn.Linear(h * d_v, d_model)
        self.dropout=nn.Dropout(dropout)

        self.d_model = d_model
        self.d_k = d_k
        self.d_v = d_v
        self.h = h

        self.init_weights()


    def init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                init.kaiming_normal_(m.weight, mode='fan_out')
                if m.bias is not None:
                    init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm2d):
                init.constant_(m.weight, 1)
                init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                init.normal_(m.weight, std=0.001)
                if m.bias is not None:
                    init.constant_(m.bias, 0)

    def forward(self, queries, keys, values, attention_mask=None, attention_weights=None):
        '''
        Computes
        :param queries: Queries (b_s, nq, d_model)
        :param keys: Keys (b_s, nk, d_model)
        :param values: Values (b_s, nk, d_model)
        :param attention_mask: Mask over attention values (b_s, h, nq, nk). True indicates masking.
        :param attention_weights: Multiplicative weights for attention values (b_s, h, nq, nk).
        :return:
        '''
        b_s, nq = queries.shape[:2]
        nk = keys.shape[1]

        q = self.fc_q(queries).view(b_s, nq, self.h, self.d_k).permute(0, 2, 1, 3)  # (b_s, h, nq, d_k)
        k = self.fc_k(keys).view(b_s, nk, self.h, self.d_k).permute(0, 2, 3, 1)  # (b_s, h, d_k, nk)
        v = self.fc_v(values).view(b_s, nk, self.h, self.d_v).permute(0, 2, 1, 3)  # (b_s, h, nk, d_v)

        att = torch.matmul(q, k) / np.sqrt(self.d_k)  # (b_s, h, nq, nk)
        if attention_weights is not None:
            att = att * attention_weights
        if attention_mask is not None:
            att = att.masked_fill(attention_mask, -np.inf)
        att = torch.softmax(att, -1)
        att=self.dropout(att)

        out = torch.matmul(att, v).permute(0, 2, 1, 3).contiguous().view(b_s, nq, self.h * self.d_v)  # (b_s, nq, h*d_v)
        out = self.fc_o(out)  # (b_s, nq, d_model)
        return out
class SwishImplementation(torch.autograd.Function):
    @staticmethod
    def forward(ctx, i):
        result = i * torch.sigmoid(i)
        ctx.save_for_backward(i)
        return result

    @staticmethod
    def backward(ctx, grad_output):
        i = ctx.saved_variables[0]
        sigmoid_i = torch.sigmoid(i)
        return grad_output * (sigmoid_i * (1 + i * (1 - sigmoid_i)))

class MemoryEfficientSwish(nn.Module):
    def forward(self, x):
        return SwishImplementation.apply(x)


def drop_connect(inputs, p, training):
    """ Drop connect. """
    if not training: return inputs
    batch_size = inputs.shape[0]
    keep_prob = 1 - p
    random_tensor = keep_prob
    random_tensor += torch.rand([batch_size, 1, 1, 1], dtype=inputs.dtype, device=inputs.device)
    binary_tensor = torch.floor(random_tensor)
    output = inputs / keep_prob * binary_tensor
    return output


def get_same_padding_conv2d(image_size=None):
     return partial(Conv2dStaticSamePadding, image_size=image_size)

def get_width_and_height_from_size(x):
    """ Obtains width and height from a int or tuple """
    if isinstance(x, int): return x, x
    if isinstance(x, list) or isinstance(x, tuple): return x
    else: raise TypeError()

def calculate_output_image_size(input_image_size, stride):
    """
    计算出 Conv2dSamePadding with a stride.
    """
    if input_image_size is None: return None
    image_height, image_width = get_width_and_height_from_size(input_image_size)
    stride = stride if isinstance(stride, int) else stride[0]
    image_height = int(math.ceil(image_height / stride))
    image_width = int(math.ceil(image_width / stride))
    return [image_height, image_width]



class Conv2dStaticSamePadding(nn.Conv2d):
    """ 2D Convolutions like TensorFlow, for a fixed image size"""

    def __init__(self, in_channels, out_channels, kernel_size, image_size=None, **kwargs):
        super().__init__(in_channels, out_channels, kernel_size, **kwargs)
        self.stride = self.stride if len(self.stride) == 2 else [self.stride[0]] * 2

        # Calculate padding based on image size and save it
        assert image_size is not None
        ih, iw = (image_size, image_size) if isinstance(image_size, int) else image_size
        kh, kw = self.weight.size()[-2:]
        sh, sw = self.stride
        oh, ow = math.ceil(ih / sh), math.ceil(iw / sw)
        pad_h = max((oh - 1) * self.stride[0] + (kh - 1) * self.dilation[0] + 1 - ih, 0)
        pad_w = max((ow - 1) * self.stride[1] + (kw - 1) * self.dilation[1] + 1 - iw, 0)
        if pad_h > 0 or pad_w > 0:
            self.static_padding = nn.ZeroPad2d((pad_w // 2, pad_w - pad_w // 2, pad_h // 2, pad_h - pad_h // 2))
        else:
            self.static_padding = Identity()

    def forward(self, x):
        x = self.static_padding(x)
        x = F.conv2d(x, self.weight, self.bias, self.stride, self.padding, self.dilation, self.groups)
        return x

class Identity(nn.Module):
    def __init__(self, ):
        super(Identity, self).__init__()

    def forward(self, input):
        return 
class MBConvBlock(nn.Module):
    '''
    层 ksize3*3 输入32 输出16  conv1  stride步长1
    '''
    def __init__(self, ksize, input_filters, output_filters, expand_ratio=1, stride=1, image_size=224):
        super().__init__()
        self._bn_mom = 0.1
        self._bn_eps = 0.01
        self._se_ratio = 0.25
        self._input_filters = input_filters
        self._output_filters = output_filters
        self._expand_ratio = expand_ratio
        self._kernel_size = ksize
        self._stride = stride

        inp = self._input_filters
        oup = self._input_filters * self._expand_ratio
        if self._expand_ratio != 1:
            Conv2d = get_same_padding_conv2d(image_size=image_size)
            self._expand_conv = Conv2d(in_channels=inp, out_channels=oup, kernel_size=1, bias=False)
            self._bn0 = nn.BatchNorm2d(num_features=oup, momentum=self._bn_mom, eps=self._bn_eps)


        # Depthwise convolution
        k = self._kernel_size
        s = self._stride
        Conv2d = get_same_padding_conv2d(image_size=image_size)
        self._depthwise_conv = Conv2d(
            in_channels=oup, out_channels=oup, groups=oup,
            kernel_size=k, stride=s, bias=False)
        self._bn1 = nn.BatchNorm2d(num_features=oup, momentum=self._bn_mom, eps=self._bn_eps)
        image_size = calculate_output_image_size(image_size, s)

        # Squeeze and Excitation layer, if desired
        Conv2d = get_same_padding_conv2d(image_size=(1,1))
        num_squeezed_channels = max(1, int(self._input_filters * self._se_ratio))
        self._se_reduce = Conv2d(in_channels=oup, out_channels=num_squeezed_channels, kernel_size=1)
        self._se_expand = Conv2d(in_channels=num_squeezed_channels, out_channels=oup, kernel_size=1)

        # Output phase
        final_oup = self._output_filters
        Conv2d = get_same_padding_conv2d(image_size=image_size)
        self._project_conv = Conv2d(in_channels=oup, out_channels=final_oup, kernel_size=1, bias=False)
        self._bn2 = nn.BatchNorm2d(num_features=final_oup, momentum=self._bn_mom, eps=self._bn_eps)
        self._swish = MemoryEfficientSwish()

    def forward(self, inputs, drop_connect_rate=None):
        """
        :param inputs: input tensor
        :param drop_connect_rate: drop connect rate (float, between 0 and 1)
        :return: output of block
        """

        # Expansion and Depthwise Convolution
        x = inputs
        if self._expand_ratio != 1:
            expand = self._expand_conv(inputs)
            bn0 = self._bn0(expand)
            x = self._swish(bn0)
        depthwise = self._depthwise_conv(x)
        bn1 = self._bn1(depthwise)
        x = self._swish(bn1)

        # Squeeze and Excitation
        x_squeezed = F.adaptive_avg_pool2d(x, 1)
        x_squeezed = self._se_reduce(x_squeezed)
        x_squeezed = self._swish(x_squeezed)
        x_squeezed = self._se_expand(x_squeezed)
        x = torch.sigmoid(x_squeezed) * x

        x = self._bn2(self._project_conv(x))

        # Skip connection and drop connect
        input_filters, output_filters = self._input_filters, self._output_filters
        if self._stride == 1 and input_filters == output_filters:
            if drop_connect_rate:
                x = drop_connect(x, p=drop_connect_rate, training=self.training)
            x = x + inputs  # skip connection
        return x
class CoAtNet(nn.Module):
    def __init__(self,in_ch,image_size,out_chs=[64,96,192,384,768]):
        super().__init__()
        self.out_chs=out_chs
        self.maxpool2d=nn.MaxPool2d(kernel_size=2,stride=2)
        self.maxpool1d = nn.MaxPool1d(kernel_size=2, stride=2)

        self.s0=nn.Sequential(
            nn.Conv2d(in_ch,in_ch,kernel_size=3,padding=1),
            nn.ReLU(),
            nn.Conv2d(in_ch,in_ch,kernel_size=3,padding=1)
        )
        self.mlp0=nn.Sequential(
            nn.Conv2d(in_ch,out_chs[0],kernel_size=1),
            nn.ReLU(),
            nn.Conv2d(out_chs[0],out_chs[0],kernel_size=1)
        )
        
        self.s1=MBConvBlock(ksize=3,input_filters=out_chs[0],output_filters=out_chs[0],image_size=image_size//2)
        self.mlp1=nn.Sequential(
            nn.Conv2d(out_chs[0],out_chs[1],kernel_size=1),
            nn.ReLU(),
            nn.Conv2d(out_chs[1],out_chs[1],kernel_size=1)
        )

        self.s2=MBConvBlock(ksize=3,input_filters=out_chs[1],output_filters=out_chs[1],image_size=image_size//4)
        self.mlp2=nn.Sequential(
            nn.Conv2d(out_chs[1],out_chs[2],kernel_size=1),
            nn.ReLU(),
            nn.Conv2d(out_chs[2],out_chs[2],kernel_size=1)
        )

        self.s3=ScaledDotProductAttention(out_chs[2],out_chs[2]//8,out_chs[2]//8,8)
        self.mlp3=nn.Sequential(
            nn.Linear(out_chs[2],out_chs[3]),
            nn.ReLU(),
            nn.Linear(out_chs[3],out_chs[3])
        )

        self.s4=ScaledDotProductAttention(out_chs[3],out_chs[3]//8,out_chs[3]//8,8)
        self.mlp4=nn.Sequential(
            nn.Linear(out_chs[3],out_chs[4]),
            nn.ReLU(),
            nn.Linear(out_chs[4],out_chs[4])
        )


    def forward(self, x) :
        B,C,H,W=x.shape
        #stage0
        y=self.mlp0(self.s0(x))
        y=self.maxpool2d(y)
        #stage1
        y=self.mlp1(self.s1(y))
        y=self.maxpool2d(y)
        #stage2
        y=self.mlp2(self.s2(y))
        y=self.maxpool2d(y)
        #stage3
        y=y.reshape(B,self.out_chs[2],-1).permute(0,2,1) #B,N,C
        y=self.mlp3(self.s3(y,y,y))
        y=self.maxpool1d(y.permute(0,2,1)).permute(0,2,1)
        #stage4
        y=self.mlp4(self.s4(y,y,y))
        y=self.maxpool1d(y.permute(0,2,1))
        N=y.shape[-1]
        y=y.reshape(B,self.out_chs[4],int(sqrt(N)),int(sqrt(N)))

        return y

class AverageValueMeter(object):
    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0.0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count


def save_model(path, net):
    torch.save({'net_state_dict': net.module.state_dict()}, path)


# Part of the code is referred from: https://github.com/ClementPinard/SfmLearner-Pytorch/blob/master/inverse_warp.py

def quat2mat(quat):
    x, y, z, w = quat[:, 0], quat[:, 1], quat[:, 2], quat[:, 3]

    B = quat.size(0)

    w2, x2, y2, z2 = w.pow(2), x.pow(2), y.pow(2), z.pow(2)
    wx, wy, wz = w*x, w*y, w*z
    xy, xz, yz = x*y, x*z, y*z

    rotMat = torch.stack([w2 + x2 - y2 - z2, 2*xy - 2*wz, 2*wy + 2*xz,
                          2*wz + 2*xy, w2 - x2 + y2 - z2, 2*yz - 2*wx,
                          2*xz - 2*wy, 2*wx + 2*yz, w2 - x2 - y2 + z2], dim=1).reshape(B, 3, 3)
    return rotMat


def transform_point_cloud(point_cloud, rotation, translation):
    if len(rotation.size()) == 2:
        rot_mat = quat2mat(rotation)
    else:
        rot_mat = rotation
    return torch.matmul(rot_mat, point_cloud) + translation.unsqueeze(2)


def npmat2euler(mats, seq='zyx'):
    eulers = []
    for i in range(mats.shape[0]):
        r = Rotation.from_dcm(mats[i])
        eulers.append(r.as_euler(seq, degrees=True))
    return np.asarray(eulers, dtype='float32')


def rt_to_transformation(R, t):
	bot_row = torch.Tensor([[[0, 0, 0, 1]]]).repeat(R.shape[0], 1, 1).to(R.device)
	T = torch.cat([torch.cat([R, t], dim=2), bot_row], dim=1)
	return T


def rotation_error(R, R_gt):
	cos_theta = (torch.einsum('bij,bij->b', R, R_gt) - 1) / 2
	cos_theta = torch.clamp(cos_theta, -1, 1)
	return torch.acos(cos_theta) * 180 / math.pi

def rotation_geodesic_error(m1, m2):
	batch=m1.shape[0]
	m = torch.bmm(m1, m2.transpose(1,2)) #batch*3*3

	cos = (  m[:,0,0] + m[:,1,1] + m[:,2,2] - 1 )/2
	cos = torch.min(cos, torch.autograd.Variable(torch.ones(batch).cuda()) )
	cos = torch.max(cos, torch.autograd.Variable(torch.ones(batch).cuda())*-1 )

	theta = torch.acos(cos)

	#theta = torch.min(theta, 2*np.pi - theta)

	return theta
def translation_error(t, t_gt):
	return torch.norm(t - t_gt, dim=1)


def rmse_loss(pts, T, T_gt):
	pts_pred = pts @ T[:, :3, :3].transpose(1, 2) + T[:, :3, 3].unsqueeze(1)
	pts_gt = pts @ T_gt[:, :3, :3].transpose(1, 2) + T_gt[:, :3, 3].unsqueeze(1)
	return torch.norm(pts_pred - pts_gt, dim=2).mean(dim=1)
