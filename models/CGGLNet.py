import timm
import torch.nn.functional as F
import torch
from torch import nn
#from functools import partial
#from torch.autograd import Variable
#from einops import rearrange
#from timm.models.layers import DropPath
#import cv2
from timm.models.layers import DropPath, to_2tuple, trunc_normal_

class ConvBNAct(nn.Sequential):
    def __init__(self, in_channels, out_channels, kernel_size=3, dilation=1, stride=1,
                 norm_layer=nn.BatchNorm2d, act_layer=nn.ReLU6, bias=False, inplace=False):
        super(ConvBNAct, self).__init__(
            nn.Conv2d(in_channels, out_channels, kernel_size=kernel_size, bias=bias,
                      dilation=dilation, stride=stride, padding=((stride - 1) + dilation * (kernel_size - 1)) // 2),
            norm_layer(out_channels),
            act_layer(inplace=inplace)
        )

class ConvBNReLU(nn.Sequential):
    def __init__(self, in_channels, out_channels, kernel_size=3, dilation=1, stride=1, norm_layer=nn.BatchNorm2d, groups=1, bias=False):
        super(ConvBNReLU, self).__init__(
            nn.Conv2d(in_channels, out_channels, kernel_size=kernel_size, bias=bias,
                      dilation=dilation, stride=stride, padding=((stride - 1) + dilation * (kernel_size - 1)) // 2, groups=groups),
            norm_layer(out_channels),
            nn.ReLU6()
        )

class ConvBNReLUy(nn.Sequential):##水平
    def __init__(self, in_channels, out_channels, kernel_size=3, dilation=1, stride=1, norm_layer=nn.BatchNorm2d, bias=False):
        super(ConvBNReLUy, self).__init__(
            nn.Conv2d(in_channels, out_channels, kernel_size=(1, kernel_size), bias=bias,
                      dilation=(dilation, dilation), stride=(stride, stride),
                      padding=(0, ((stride - 1) + dilation * (kernel_size - 1)) // 2)
                      ),
            norm_layer(out_channels),
            nn.ReLU6()
        )

class ConvBNy(nn.Sequential):
    def __init__(self, in_channels, out_channels, kernel_size=3, dilation=1, stride=1, norm_layer=nn.BatchNorm2d, bias=False):
        super(ConvBNy, self).__init__(
            nn.Conv2d(in_channels, out_channels, kernel_size=(1, kernel_size), bias=bias,
                      dilation=(dilation, dilation), stride=(stride, stride),
                      padding=(0, ((stride - 1) + dilation * (kernel_size - 1)) // 2)
                      ),
            norm_layer(out_channels),
        )

class ConvBNReLUx(nn.Sequential):
    def __init__(self, in_channels, out_channels, kernel_size=3, dilation=1, stride=1, norm_layer=nn.BatchNorm2d, bias=False):
        super(ConvBNReLUx, self).__init__(
            nn.Conv2d(in_channels, out_channels, kernel_size=(kernel_size, 1), bias=bias,
                      dilation=(dilation, dilation), stride=(stride, stride),
                      padding=(((stride - 1) + dilation * (kernel_size - 1)) // 2, 0)
                      ),
            norm_layer(out_channels),
            nn.ReLU6()
        )

class ConvBNx(nn.Sequential):
    def __init__(self, in_channels, out_channels, kernel_size=3, dilation=1, stride=1, norm_layer=nn.BatchNorm2d, bias=False):
        super(ConvBNx, self).__init__(
            nn.Conv2d(in_channels, out_channels, kernel_size=(kernel_size, 1), bias=bias,
                      dilation=(dilation, dilation), stride=(stride, stride),
                      padding=(((stride - 1) + dilation * (kernel_size - 1)) // 2, 0)
                      ),
            norm_layer(out_channels),
        )

class SeparableConvBNReLU(nn.Sequential):
    def __init__(self, in_channels, out_channels, kernel_size=3, stride=1, dilation=1,
                 norm_layer=nn.BatchNorm2d):
        super(SeparableConvBNReLU, self).__init__(
            nn.Conv2d(in_channels, in_channels, kernel_size, stride=stride, dilation=dilation,
                      padding=((stride - 1) + dilation * (kernel_size - 1)) // 2,
                      groups=in_channels, bias=False),
            norm_layer(out_channels),
            nn.Conv2d(in_channels, out_channels, kernel_size=1, bias=False),
            nn.ReLU6()
        )

class SeparableConvBN(nn.Sequential):
    def __init__(self, in_channels, out_channels, kernel_size=3, stride=1, dilation=1,
                 norm_layer=nn.BatchNorm2d):
        super(SeparableConvBN, self).__init__(
            nn.Conv2d(in_channels, in_channels, kernel_size, stride=stride, dilation=dilation,
                      padding=((stride - 1) + dilation * (kernel_size - 1)) // 2,
                      groups=in_channels, bias=False),
            norm_layer(out_channels),
            nn.Conv2d(in_channels, out_channels, kernel_size=1, bias=False)
        )

class SeparableConv(nn.Sequential):
    def __init__(self, in_channels, out_channels, kernel_size=3, stride=1, dilation=1):
        super(SeparableConv, self).__init__(
            nn.Conv2d(in_channels, in_channels, kernel_size, stride=stride, dilation=dilation,
                      padding=((stride - 1) + dilation * (kernel_size - 1)) // 2,
                      groups=in_channels, bias=False),
            nn.Conv2d(in_channels, out_channels, kernel_size=1, bias=False)
        )

class Conv(nn.Sequential):
    def __init__(self, in_channels, out_channels, kernel_size=3, dilation=1, stride=1, bias=False):
        super(Conv, self).__init__(
            nn.Conv2d(in_channels, out_channels, kernel_size=kernel_size, bias=bias,
                      dilation=dilation, stride=stride, padding=((stride - 1) + dilation * (kernel_size - 1)) // 2)
        )

class ChannelPool(nn.Module):
    def forward(self, x):
        return torch.cat((torch.max(x, 1)[0].unsqueeze(1), torch.mean(x, 1).unsqueeze(1)), dim=1)

class ConvBN(nn.Sequential):
    def __init__(self, in_channels, out_channels, kernel_size=3, dilation=1, stride=1,
                 norm_layer=nn.BatchNorm2d, act_layer=nn.ReLU6, bias=False, inplace=False):
        super(ConvBN, self).__init__(
            nn.Conv2d(in_channels, out_channels, kernel_size=kernel_size, bias=bias,
                      dilation=dilation, stride=stride, padding=((stride - 1) + dilation * (kernel_size - 1)) // 2),
            norm_layer(out_channels)
        )

class OCM(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=3, stride=1):
        super().__init__()

        self.Recx = ConvBNReLUx(in_channels, out_channels, kernel_size=kernel_size, stride=stride)
        self.Recy = ConvBNReLUy(in_channels, out_channels, kernel_size=kernel_size, stride=stride)
        self.conv = ConvBNAct(in_channels, out_channels, kernel_size=kernel_size, stride=stride, dilation=1)

        self.init_weight()

    def forward(self, x):

        feats = self.Recx(x) + self.Recy(x) + self.conv(x)

        return feats

    def init_weight(self):
        for m in self.children():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, a=1)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)

class OACM(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=3,dilation=[1, 2, 4, 8]):
        super().__init__()


        self.preconv = ConvBNReLU(in_channels, out_channels, kernel_size=1, stride=1)

        self.Recx = ConvBNReLUx(out_channels, out_channels // 4, kernel_size=kernel_size)
        self.Recx2 = ConvBNReLUx(out_channels, out_channels//4, kernel_size=kernel_size, dilation=dilation[1])
        self.Recx4 = ConvBNReLUx(out_channels, out_channels//4, kernel_size=kernel_size, dilation=dilation[2])
        self.Recx8 = ConvBNReLUx(out_channels, out_channels//4, kernel_size=kernel_size, dilation=dilation[3])

        self.Recy = ConvBNReLUy(out_channels, out_channels//4, kernel_size=kernel_size)
        self.Recy2 = ConvBNReLUy(out_channels, out_channels//4, kernel_size=kernel_size, dilation=dilation[1])
        self.Recy4 = ConvBNReLUy(out_channels, out_channels//4, kernel_size=kernel_size, dilation=dilation[2])
        self.Recy8 = ConvBNReLUy(out_channels, out_channels//4, kernel_size=kernel_size, dilation=dilation[3])

        self.conv = ConvBNReLU(out_channels, out_channels//4, kernel_size=kernel_size, stride=1, dilation=1)
        self.conv2 = ConvBNReLU(out_channels, out_channels//4, kernel_size=kernel_size, stride=1, dilation=dilation[1])
        self.conv4 = ConvBNReLU(out_channels, out_channels//4, kernel_size=kernel_size, stride=1, dilation=dilation[2])
        self.conv8 = ConvBNReLU(out_channels, out_channels//4, kernel_size=kernel_size, stride=1, dilation=dilation[3])

        self.convxout = ConvBNReLU(out_channels, out_channels, stride=1)


    def forward(self, x):

        x = self.preconv(x)

        featsx = torch.cat((self.Recx(x), self.Recx2(x), self.Recx4(x), self.Recx8(x)),dim=1)
        featsy = torch.cat((self.Recy(x), self.Recy2(x), self.Recy4(x), self.Recy8(x)),dim=1)
        feats = torch.cat((self.conv(x), self.conv2(x), self.conv4(x), self.conv8(x)),dim=1)
        out = featsx + featsy + feats

        out = self.convxout(out)

        return out

    def init_weight(self):
        for m in self.children():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, a=1)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)

import numpy as np

class MultiHeadAttention(nn.Module):
    '''
    input:
        query --- [N, T_q, query_dim]
        key --- [N, T_k, key_dim]
        mask --- [N, T_k]
    output:
        out --- [N, T_q, num_units]
        scores -- [h, N, T_q, T_k]
    '''

    def __init__(self, query_dim: object, key_dim: object, num_units: object, num_heads: object, Mode: object = 'conv') -> object:
        super().__init__()
        self.num_units = num_units
        self.num_heads = num_heads
        self.key_dim = key_dim
        self.mode =Mode

        if self.mode != 'no_conv':
            self.W_query = nn.Linear(in_features=query_dim, out_features=query_dim, bias=False)

        self.W_key = nn.Linear(in_features=key_dim, out_features=key_dim, bias=False)
        self.W_value = nn.Linear(in_features=num_units, out_features=num_units, bias=False)
        self.out = nn.Linear(in_features=num_units, out_features=num_units, bias=False)

    def forward(self, query, key, mask=None):

        if self.mode=='no_conv':
            querys = query  # [N, T_q, num_units]
            keys = self.W_key(query) + query
        else:
            querys = self.W_query(query)  # [N, T_q, num_units]
            keys = self.W_key(query)  # [N, T_k, num_units]
        values = self.W_value(key)

        split_size_qk = self.key_dim // self.num_heads
        split_size = self.num_units // self.num_heads
        querys = torch.stack(torch.split(querys, split_size_qk, dim=2), dim=0)  # [h, N, T_q, num_units/h]
        keys = torch.stack(torch.split(keys, split_size_qk, dim=2), dim=0)  # [h, N, T_k, num_units/h]
        values = torch.stack(torch.split(values, split_size, dim=2), dim=0)  # [h, N, T_k, num_units/h]

        scores = torch.matmul(querys, keys.transpose(2, 3))  # [h, N, T_q, T_k]
        scores = scores / (split_size_qk ** 0.5)
        scores = F.softmax(scores, dim=3)

        out = torch.matmul(scores, values)  # [h, N, T_q, num_units/h]
        out = torch.cat(torch.split(out, 1, dim=0), dim=3).squeeze(0)  # [N, T_q, num_units]
        out=self.out(out)

        return out, scores

class ConvBlock(nn.Module):

    def __init__(self, in_channels, out_channels, padding=1, kernel_size=3, stride=1, with_nonlinearity=True):
        super().__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, padding=padding, kernel_size=kernel_size, stride=stride)
        self.bn = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU()
        self.with_nonlinearity = with_nonlinearity

    def forward(self, x):
        x = self.conv(x)
        x = self.bn(x)
        if self.with_nonlinearity:
            x = self.relu(x)
        return x

class GLSTB(nn.Module):
    def __init__(self, in_channels,nums_heads = 6,weight_ratio = 1.0):
        super(GLSTB, self).__init__()

        self.out_channels=in_channels
        self.weight_ratio=weight_ratio

        self.msa_v = MultiHeadAttention(nums_heads, nums_heads, self.out_channels, nums_heads,Mode='no_conv')
        self.local_v = AdaptiveLocalFeatureExtraction(in_channels, ratio=8, mode='v')
        self.conv_v = ConvBlock(in_channels=in_channels, out_channels=in_channels, padding=1, kernel_size=3, stride=1)

        self.msa_h = MultiHeadAttention(nums_heads, nums_heads, self.out_channels, nums_heads,Mode='no_conv')
        self.local_h = AdaptiveLocalFeatureExtraction(in_channels, ratio=8, mode='h')
        self.conv_h = ConvBlock(in_channels=in_channels, out_channels=in_channels, padding=1, kernel_size=3, stride=1)


    def forward(self, qk,x,label=None):

        b,c,h,w=x.size(0),x.size(1),x.size(2),x.size(3)
        qk = F.softmax(qk*self.weight_ratio, dim=1)

        vf = (x).clone()#[:, :, :, :].contiguous()  # b,c,h,w
        vqk_view = qk.clone().permute(0, 3, 2, 1).reshape(b * w, h, -1).contiguous()
        vf_view = vf.permute(0, 3, 2, 1).reshape(b * w, h, c).contiguous()  # b,w,h,c
        x,vscores = self.msa_v(vqk_view, vf_view)#
        x=x.reshape(b, w, h, c).permute(0, 3, 2, 1).contiguous() #+ vf

        x=x + self.local_v(vf)
        x = self.conv_v(x)

        hf = x.clone()#[:, :, :, :].contiguous()  # b,c,h,w
        hqk_view = qk.clone().permute(0, 2, 3, 1).reshape(b * h, w, -1).contiguous()  # b,h,w,c
        hf_view = hf.permute(0, 2, 3, 1).reshape(b * h, w, c).contiguous() # b,h,w,c
        x,hscores = self.msa_h(hqk_view, hf_view)
        x=x.reshape(b, h, w, c).permute(0, 3, 1, 2).contiguous() #+ hf
        x = x + self.local_h(hf)
        x = self.conv_h(x)

        return x,vscores,hscores

class Mlp_decoder(nn.Module):
    def __init__(self, in_features, hidden_features=None, out_features=None, act_layer=nn.ReLU6, drop=0.):
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        self.fc1 = nn.Conv2d(in_features, hidden_features, 1, 1, 0, bias=True)
        self.act = act_layer()
        self.fc2 = nn.Conv2d(hidden_features, out_features, 1, 1, 0, bias=True)
        self.drop = nn.Dropout(drop, inplace=True)

    def forward(self, x):
        x = self.fc1(x)
        x = self.act(x)
        x = self.drop(x)
        x = self.fc2(x)
        x = self.drop(x)
        return x

class Channel_Selection(nn.Module):
    def __init__(self, channels, ratio=8):
        super(Channel_Selection, self).__init__()

        self.avg_pooling = nn.AdaptiveAvgPool2d(1)
        self.max_pooling = nn.AdaptiveMaxPool2d(1)

        self.fc_layers = nn.Sequential(
            Conv(channels, channels // ratio, kernel_size=1),
            nn.ReLU(),
            Conv(channels // ratio, channels, kernel_size=1)
        )

        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        b, c, h, w = x.shape
        avg_x = self.avg_pooling(x).view(b, c, 1, 1)
        max_x = self.max_pooling(x).view(b, c, 1, 1)
        v = self.fc_layers(avg_x) + self.fc_layers(max_x)
        v = self.sigmoid(v).view(b, c,1,1)

        return v

class AdaptiveLocalFeatureExtraction(nn.Module):
    def __init__(self, dim, ratio=8,mode='v'):
        super(AdaptiveLocalFeatureExtraction, self).__init__()

        self.preconv = ConvBN(in_channels=dim,out_channels=dim,kernel_size=3)

        self.Channel_Selection = Channel_Selection(channels = dim, ratio=ratio)

        if mode=='v':
            self.convbase = ConvBNx(in_channels=dim,out_channels=dim,kernel_size=3)
            self.convlarge = ConvBNx(in_channels=dim,out_channels=dim,kernel_size=5)
        else:
            self.convbase = ConvBNy(in_channels=dim, out_channels=dim, kernel_size=3)
            self.convlarge = ConvBNy(in_channels=dim, out_channels=dim, kernel_size=5)


        self.post_conv = SeparableConvBNReLU(dim, dim, 3)


    def forward(self, x):

        s = self.Channel_Selection(self.preconv(x))
        x = self.post_conv(s * self.convbase(x) + (1 - s) * self.convlarge(x))

        return x

class GLSTM(nn.Module):
    def __init__(self, dim=512, num_heads=6,  mlp_ratio=4, drop=0.,weight_ratio = 1.0,
                 drop_path=0., act_layer=nn.ReLU6, norm_layer=nn.BatchNorm2d):
        super().__init__()
        self.norm1 = norm_layer(dim)

        self.attn = GLSTB(dim,nums_heads=num_heads,weight_ratio =weight_ratio)

        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()
        mlp_hidden_dim = int(dim // mlp_ratio)

        self.mlp = Mlp_decoder(in_features=dim, hidden_features=mlp_hidden_dim, out_features=dim, act_layer=act_layer,
                               drop=drop)
        self.norm2 = norm_layer(dim)

    def forward(self, aux, x):
        x, vscores, hscores = self.attn(aux,x)
        x = x + self.drop_path(x)
        x = x + self.drop_path(self.mlp(self.norm2(x)))

        return x, vscores, hscores

class GLTM(nn.Module):
    def __init__(self, dim=512, num_heads=6,  mlp_ratio=4,drop=0.,
                 drop_path=0., act_layer=nn.ReLU6, norm_layer=nn.BatchNorm2d):
        super().__init__()
        self.norm1 = norm_layer(dim)

        self.msa_v = MultiHeadAttention(dim, dim, dim, num_heads)
        self.local_v = AdaptiveLocalFeatureExtraction(dim, ratio=8,mode='v')
        self.conv_v = ConvBlock(in_channels=dim, out_channels=dim, padding=1, kernel_size=3, stride=1)

        self.msa_h = MultiHeadAttention(dim, dim, dim, num_heads)
        self.local_h = AdaptiveLocalFeatureExtraction(dim, ratio=8, mode='h')
        self.conv_h = ConvBlock(in_channels=dim, out_channels=dim, padding=1, kernel_size=3, stride=1)

        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()
        mlp_hidden_dim = int(dim // mlp_ratio)

        self.mlp = Mlp_decoder(in_features=dim, hidden_features=mlp_hidden_dim, out_features=dim, act_layer=act_layer,
                               drop=drop)
        self.norm2 = norm_layer(dim)

    def forward(self, x):

        b, c, h, w = x.size(0), x.size(1), x.size(2), x.size(3)


        vf = (x).clone()  # [:, :, :, :].contiguous()  # b,c,h,w
        vqk_view = x.clone().permute(0, 3, 2, 1).reshape(b * w, h, -1).contiguous()
        x, vscores = self.msa_v(vqk_view, vqk_view)  #
        x = x.reshape(b, w, h, c).permute(0, 3, 2, 1).contiguous()  # + vf

        x = x + self.local_v(vf)
        x = self.conv_v(x)

        hf = x.clone()  # [:, :, :, :].contiguous()  # b,c,h,w
        hqk_view = x.clone().permute(0, 2, 3, 1).reshape(b * h, w, -1).contiguous()  # b,h,w,c
        x, hscores = self.msa_h(hqk_view, hqk_view)
        x = x.reshape(b, h, w, c).permute(0, 3, 1, 2).contiguous()  # + hf
        x = x + self.local_h(hf)
        x = self.conv_h(x)

        x = x + self.drop_path(x)
        x = x + self.drop_path(self.mlp(self.norm2(x)))

        return x, vscores, hscores


class Fusion(nn.Module):
    def __init__(self, in_channsel=64,out_channels=64, eps=1e-8):
        super(Fusion, self).__init__()


        self.weights = nn.Parameter(torch.ones(2, dtype=torch.float32), requires_grad=True)
        self.eps = eps
        self.Preconv = Conv(in_channels=in_channsel,out_channels=out_channels,kernel_size=1)
        self.post_conv = SeparableConvBNReLU(out_channels, out_channels, 5)


    def forward(self, x, res):
        x = F.interpolate(x, scale_factor=2, mode='bilinear', align_corners=False)
        weights = nn.ReLU6()(self.weights)
        fuse_weights = weights / (torch.sum(weights, dim=0) + self.eps)
        x = fuse_weights[0] * res + fuse_weights[1] *self.Preconv(x)
        x = self.post_conv(x)
        return x

class FRSH(nn.Module):
    def __init__(self, dim, fc_ratio, dilation=[1, 2, 4, 8], dropout=0., num_classes=6):
        super(FRSH, self).__init__()

        self.oacm = OACM(in_channels=dim, out_channels=dim, kernel_size=3, dilation=dilation)

        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Sequential(
            nn.Conv2d(dim, dim//fc_ratio, 1, 1),
            nn.ReLU6(),
            nn.Conv2d(dim//fc_ratio, dim, 1, 1),
            nn.Sigmoid()
        )

        self.s_conv = nn.Conv2d(in_channels=2, out_channels=1, kernel_size=5, padding=2)
        self.sigmoid = nn.Sigmoid()

        self.head = nn.Sequential(SeparableConvBNReLU(dim, dim, kernel_size=3),
                                  nn.Dropout2d(p=dropout, inplace=True),
                                  Conv(dim, num_classes, kernel_size=1))

    def forward(self, x):
        u = x.clone()

        attn = self.oacm(x)
        attn = attn * u

        c_attn = self.avg_pool(x)
        c_attn = self.fc(c_attn)
        c_attn = u * c_attn

        s_max_out, _ = torch.max(x, dim=1, keepdim=True)
        s_avg_out = torch.mean(x, dim=1, keepdim=True)
        s_attn = torch.cat((s_avg_out, s_max_out), dim=1)
        s_attn = self.s_conv(s_attn)
        s_attn = self.sigmoid(s_attn)
        s_attn = u * s_attn

        out = self.head(attn + c_attn + s_attn)

        return out

class SegHead(nn.Module):

    def __init__(self, in_channels=64, num_classes=8):
        super().__init__()
        self.conv = ConvBNReLU(in_channels, in_channels)
        self.drop = nn.Dropout(0.1)
        self.conv_out = Conv(in_channels, num_classes, kernel_size=1)

        self.qkconv_out = nn.Sequential(ConvBNReLU(num_classes, num_classes),
                                        nn.Dropout(0.1),
                                        Conv(num_classes, num_classes, kernel_size=1))


    def forward(self, x, h, w):
        feat = self.conv(x)
        feat = self.drop(feat)
        feat = self.conv_out(feat)
        aux = F.interpolate(feat, size=(h, w), mode='bilinear', align_corners=False)

        return feat,aux

class Decoder(nn.Module):
    def __init__(self,
                 encode_channels=[256, 512, 1024, 2048],
                 decode_channels=[256, 512, 1024, 2048],
                 dilation = [[1, 2, 4, 8], [1, 2, 4, 8], [1, 2, 4, 8], [1, 2, 4, 8]],
                 fc_ratio=4,
                 dropout=0.1,
                 num_classes=6,
                 weight_ratio = 1.0):
        super(Decoder, self).__init__()

        self.Conv1 = ConvBNReLU(encode_channels[-1], decode_channels[-1], 1)
        self.Conv2 = ConvBNReLU(encode_channels[-2], decode_channels[-2], 1)
        self.Conv3 = ConvBNReLU(encode_channels[-3], decode_channels[-3], 1)
        self.Conv4 = ConvBNReLU(encode_channels[-4], decode_channels[-4], 1)

        self.b4 = GLTM(dim=decode_channels[-1], num_heads=num_classes, mlp_ratio=fc_ratio)

        self.p3 = Fusion(decode_channels[-1], decode_channels[-2])
        self.b3 = GLSTM(dim=decode_channels[-2], num_heads=num_classes, mlp_ratio=fc_ratio,weight_ratio = weight_ratio)


        self.p2 = Fusion(decode_channels[-2], decode_channels[-3])
        self.b2 = GLSTM(dim=decode_channels[-3], num_heads=num_classes, mlp_ratio=fc_ratio,weight_ratio = weight_ratio)

        self.p1 = Fusion(decode_channels[-3], decode_channels[-4])
        self.b1 = GLSTM(dim=decode_channels[-4], num_heads=num_classes, mlp_ratio=fc_ratio,weight_ratio = weight_ratio)

        self.Conv5 = ConvBN(decode_channels[-4], 64, 1)


        self.p = Fusion(64)
        self.seg_head = FRSH(64, fc_ratio=fc_ratio, dilation=dilation[3], dropout=dropout, num_classes=num_classes)


        #FeatureRefinementHead(encoder_channels[-4], decode_channels)

        ##
        self.up4 = nn.UpsamplingBilinear2d(scale_factor=4)
        self.up3 = nn.UpsamplingBilinear2d(scale_factor=2)
        self.aux_head4 = SegHead(decode_channels[-1], num_classes)
        self.aux_head3 = SegHead(decode_channels[-2], num_classes)
        self.aux_head2 = SegHead(decode_channels[-3], num_classes)
        self.aux_head1 = SegHead(decode_channels[-4], num_classes)
        self.init_weight()

    def forward(self, res, res1, res2, res3, res4, h, w):

        res4 = self.Conv1(res4)
        res3 = self.Conv2(res3)
        res2 = self.Conv3(res2)
        res1 = self.Conv4(res1)

        aux4_4, aux4 = self.aux_head4(res4, h, w)
        x, _, _ = self.b4(res4)

        x = self.p3(x, res3)
        aux3_3, aux3 = self.aux_head3(x, h, w)
        x, _, _ = self.b3(aux3_3, x)

        x = self.p2(x, res2)
        aux2_2, aux2 = self.aux_head2(x, h, w)
        x, _, _ = self.b2(aux2_2, x)

        x = self.p1(x, res1)
        aux1_1, aux1 = self.aux_head1(x, h, w)
        x, _, _ = self.b1(aux1_1, x)

        x = self.Conv5(x)
        x = self.p(x, res)
        x = self.seg_head(x)

        x = F.interpolate(x, size=(h, w), mode='bilinear', align_corners=False)

        return x, aux1,aux2,aux3,aux4,aux1_1,aux2_2,aux3_3,aux4_4

    def init_weight(self):
        for m in self.children():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, a=1)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)



class CGGLNet(nn.Module):
    def __init__(self,num_classes,
                 dropout=0.1,
                 fc_ratio=4,
                 decode_channels=32):
        super(CGGLNet, self).__init__()

#        self.backbone = timm.create_model('swsl_resnet50', features_only=True, output_stride=32,
#                                          out_indices=(1, 2, 3, 4), pretrained=True)
#        encoder_channels = self.backbone.feature_info.channels()

        pretrained_cfg = timm.models.create_model('swsl_resnet50', features_only=True, output_stride=32,
                                                  out_indices=(1, 2, 3, 4)).default_cfg
        pretrained_cfg[
            'file'] = r'C:\\Users\\MeloNy\\.cache\\torch\\hub\\checkpoints\\semi_weakly_supervised_resnet50-16a12f1b.pth'
        self.backbone = timm.models.swsl_resnet50(pretrained=True, pretrained_cfg=pretrained_cfg)

        encoder_channels = [info['num_chs'] for info in self.backbone.feature_info]


        self.cnn = nn.Sequential(self.backbone.conv1,
                                 self.backbone.bn1,
                                 self.backbone.act1
                                 )

        self.cnn1 = nn.Sequential(self.backbone.maxpool,self.backbone.layer1)
        self.cnn2 = self.backbone.layer2
        self.cnn3 = self.backbone.layer3
        self.cnn4 = self.backbone.layer4

        decode_channels = [decode_channels * num_classes,decode_channels * num_classes,
                           decode_channels * num_classes,decode_channels * num_classes]
        ##
        self.decoder = Decoder(encoder_channels, decode_channels=decode_channels,
                               dropout=dropout, num_classes=num_classes,weight_ratio = 1.0)


    def forward(self, x):
        h, w = x.size()[-2:]

        # Encoder ResNet50
        x_pre = self.cnn(x)    ##H/2
        res1 = self.cnn1(x_pre)##H/4
        res2 = self.cnn2(res1) ##H/8
        res3 = self.cnn3(res2) ##H/16
        res4 = self.cnn4(res3) ##H/32

        ##
        out, aux1,aux2,aux3,aux4,aux1_1,aux2_2,aux3_3,aux4_4 = self.decoder(x_pre, res1, res2, res3, res4,h, w)


        if self.training:

            return out, aux1,aux2,aux3
        else:

            return out


if __name__ == '__main__':

    num_classes = 6
    in_batch, inchannel, in_h, in_w = 1, 3, 1024, 1024
    x = torch.randn(in_batch, inchannel, in_h, in_w)
    net = CGGLNet(num_classes)
    out,aux2,aux3,aux4 = net(x)
    print(out.shape)

    total = sum([param.nelement() for param in net.parameters()])
    print("Number of parameter: %.2fM" % (total / 1e6))