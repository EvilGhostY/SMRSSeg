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
from einops import rearrange
import numbers
from pytorch_wavelets import DWTForward, DWTInverse
from einops.layers.torch import Rearrange





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

def to_3d(x):
    return rearrange(x, 'b c h w -> b (h w) c')

def to_4d(x, h, w):
    return rearrange(x, 'b (h w) c -> b c h w', h=h, w=w)

class BiasFree_LayerNorm(nn.Module):
    def __init__(self, normalized_shape):
        super(BiasFree_LayerNorm, self).__init__()
        if isinstance(normalized_shape, numbers.Integral):
            normalized_shape = (normalized_shape,)
        normalized_shape = torch.Size(normalized_shape)

        assert len(normalized_shape) == 1

        self.weight = nn.Parameter(torch.ones(normalized_shape))
        self.normalized_shape = normalized_shape

    def forward(self, x):
        sigma = x.var(-1, keepdim=True, unbiased=False)
        return x / torch.sqrt(sigma + 1e-5) * self.weight


class WithBias_LayerNorm(nn.Module):
    def __init__(self, normalized_shape):
        super(WithBias_LayerNorm, self).__init__()
        if isinstance(normalized_shape, numbers.Integral):
            normalized_shape = (normalized_shape,)
        normalized_shape = torch.Size(normalized_shape)

        assert len(normalized_shape) == 1

        self.weight = nn.Parameter(torch.ones(normalized_shape))
        self.bias = nn.Parameter(torch.zeros(normalized_shape))
        self.normalized_shape = normalized_shape

    def forward(self, x):
        mu = x.mean(-1, keepdim=True)
        sigma = x.var(-1, keepdim=True, unbiased=False)
        return (x - mu) / torch.sqrt(sigma + 1e-5) * self.weight + self.bias


class GlobalAttention(nn.Module):
    def __init__(self,
                 query_dim = 6,
                 key_dim = 6,
                 value_dim=256,
                 num_heads=6,
                 Mode = 'conv',
                 qkv_bias=False,
                 window_size=8,
                 relative_pos_embedding=True
                 ):
        super().__init__()

        self.query_dim = query_dim
        self.key_dim = key_dim
        self.value_dim = value_dim
        self.num_heads = num_heads
        self.mode = Mode

        self.num_heads = num_heads
        self.head_qdim = self.query_dim // self.num_heads
        self.head_kdim = self.key_dim // self.num_heads
        self.head_vdim = self.value_dim // self.num_heads
        self.scale = self.head_qdim ** -0.5
        self.ws = window_size


        if self.mode != 'no_conv':
            self.q = Conv(self.query_dim, self.query_dim, kernel_size=1, bias=qkv_bias)

        self.k = Conv(self.key_dim, self.key_dim, kernel_size=1, bias=qkv_bias)
        self.v = Conv(self.value_dim, self.value_dim, kernel_size=1, bias=qkv_bias)

        self.proj = SeparableConvBN(self.value_dim, self.value_dim, kernel_size=window_size)

        self.relative_pos_embedding = relative_pos_embedding

        if self.relative_pos_embedding:
            # define a parameter table of relative position bias
            self.relative_position_bias_table = nn.Parameter(
                torch.zeros((2 * window_size - 1) * (2 * window_size - 1), num_heads))  # 2*Wh-1 * 2*Ww-1, nH

            # get pair-wise relative position index for each token inside the window
            coords_h = torch.arange(self.ws)
            coords_w = torch.arange(self.ws)
            coords = torch.stack(torch.meshgrid([coords_h, coords_w]))  # 2, Wh, Ww
            coords_flatten = torch.flatten(coords, 1)  # 2, Wh*Ww
            relative_coords = coords_flatten[:, :, None] - coords_flatten[:, None, :]  # 2, Wh*Ww, Wh*Ww
            relative_coords = relative_coords.permute(1, 2, 0).contiguous()  # Wh*Ww, Wh*Ww, 2
            relative_coords[:, :, 0] += self.ws - 1  # shift to start from 0
            relative_coords[:, :, 1] += self.ws - 1
            relative_coords[:, :, 0] *= 2 * self.ws - 1
            relative_position_index = relative_coords.sum(-1)  # Wh*Ww, Wh*Ww
            self.register_buffer("relative_position_index", relative_position_index)

            trunc_normal_(self.relative_position_bias_table, std=.02)

    def pad(self, x, ps):
        _, _, H, W = x.size()
        if W % ps != 0:
            x = F.pad(x, (0, ps - W % ps), mode='reflect')
        if H % ps != 0:
            x = F.pad(x, (0, 0, 0, ps - H % ps), mode='reflect')
        return x

    def pad_out(self, x):
        x = F.pad(x, pad=(0, 1, 0, 1), mode='reflect')
        return x

    def forward(self, q, v):
        B, C, H, W = v.shape
        v = self.pad(v, self.ws)
        q = self.pad(q, self.ws)
        B, C, Hp, Wp = v.shape

        if self.mode=='no_conv':
            q = q  # [N, T_q, num_units]
            k = self.k(q) + q
        else:
            q = self.q(q)  # [N, T_q, num_units]
            k = self.k(q)  # [N, T_k, num_units]
        v = self.v(v)

        q = rearrange(q, 'b (q h d) (hh ws1) (ww ws2) -> (q b hh ww) h (ws1 ws2) d', h=self.num_heads,
                            d=self.query_dim // self.num_heads, hh=Hp // self.ws, ww=Wp // self.ws, q=1, ws1=self.ws, ws2=self.ws)
        k = rearrange(k, 'b (k h d) (hh ws1) (ww ws2) -> (k b hh ww) h (ws1 ws2) d', h=self.num_heads,
                      d=self.key_dim // self.num_heads, hh=Hp // self.ws, ww=Wp // self.ws, k=1, ws1=self.ws,
                      ws2=self.ws)
        v = rearrange(v, 'b (v h d) (hh ws1) (ww ws2) -> (v b hh ww) h (ws1 ws2) d', h=self.num_heads,
                            d=C // self.num_heads, hh=Hp // self.ws, ww=Wp // self.ws, v=1, ws1=self.ws, ws2=self.ws)


        dots = (q @ k.transpose(-2, -1)) * self.scale

        if self.relative_pos_embedding:
            relative_position_bias = self.relative_position_bias_table[self.relative_position_index.view(-1)].view(
                self.ws * self.ws, self.ws * self.ws, -1)  # Wh*Ww,Wh*Ww,nH
            relative_position_bias = relative_position_bias.permute(2, 0, 1).contiguous()  # nH, Wh*Ww, Wh*Ww
            dots += relative_position_bias.unsqueeze(0)

        attn = dots.softmax(dim=-1)
        attn = attn @ v

        attn = rearrange(attn, '(b hh ww) h (ws1 ws2) d -> b (h d) (hh ws1) (ww ws2)', h=self.num_heads,
                         d=C//self.num_heads, hh=Hp//self.ws, ww=Wp//self.ws, ws1=self.ws, ws2=self.ws)

        attn = attn[:, :, :H, :W]
        out = self.pad_out(attn)
        out = self.proj(out)
        # print(out.size())
        out = out[:, :, :H, :W]

        return out

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
            querys = query# [N, T_q, num_units]
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
        out = self.out(out)

        return out

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
            self.convbase = ConvBNx(in_channels=dim, out_channels=dim, kernel_size=3)
            self.convlarge = ConvBNx(in_channels=dim, out_channels=dim, kernel_size=5)
        elif mode=='h':
            self.convbase = ConvBNy(in_channels=dim, out_channels=dim, kernel_size=3)
            self.convlarge = ConvBNy(in_channels=dim, out_channels=dim, kernel_size=5)
        else:
            self.convbase = ConvBN(in_channels=dim, out_channels=dim, kernel_size=3)
            self.convlarge = ConvBN(in_channels=dim, out_channels=dim, kernel_size=5)


        self.post_conv = SeparableConvBNReLU(dim, dim, 3)


    def forward(self, x):

        s = self.Channel_Selection(self.preconv(x))
        x = self.post_conv(s * self.convbase(x) + (1 - s) * self.convlarge(x))

        return x

class GLSTM(nn.Module):
    def __init__(self, dim=512, num_heads=6, window_size=8, mlp_ratio=4, drop=0.,weight_ratio = 1.0,
                 drop_path=0., act_layer=nn.ReLU6, norm_layer=nn.BatchNorm2d):
        super().__init__()
        self.norm1 = norm_layer(dim)

        self.weight_ratio = weight_ratio
        self.window_size = window_size

        self.msa = GlobalAttention(query_dim=num_heads, key_dim=num_heads, value_dim=dim,
                                   window_size=window_size,
                                   num_heads=num_heads, Mode='no_conv')
        self.cmsa = MultiHeadAttention(query_dim=num_heads, key_dim=num_heads, num_units=dim, num_heads=num_heads, Mode='no_conv')
        self.local = AdaptiveLocalFeatureExtraction(dim, ratio=8, mode='hv')
        self.proj = SeparableConvBN(dim, dim, kernel_size=window_size)

        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()
        mlp_hidden_dim = int(dim // mlp_ratio)

        self.mlp = Mlp_decoder(in_features=dim, hidden_features=mlp_hidden_dim, out_features=dim, act_layer=act_layer,
                               drop=drop)
        self.norm2 = norm_layer(dim)

    def pad_out(self, x):
        x = F.pad(x, pad=(0, 1, 0, 1), mode='reflect')
        return x

    def forward(self, aux, x):
        b, c, h, w = x.size()
        x = self.norm1(x)
        vf = x.clone()
        gx = self.msa(aux, x)

        Adaptivepool = nn.AdaptiveAvgPool2d((h // self.window_size, w // self.window_size))
        waux = Adaptivepool(aux)
        wgf = Adaptivepool(x)
        b, c, hh, ww = wgf.size()
        qk_view = waux.clone().permute(0, 2, 3, 1).reshape(b, hh * ww, -1).contiguous()
        v_view = wgf.clone().permute(0, 2, 3, 1).reshape(b, hh * ww, -1).contiguous()
        wgf = self.cmsa(qk_view, v_view)  #
        wgf = wgf.reshape(b, hh, ww, c).permute(0, 3, 1, 2).contiguous()  # + vf
        wgf = F.interpolate(wgf, scale_factor=self.window_size, mode='bilinear', align_corners=False)
        gx = gx + wgf
        lx = self.local(vf)
        gx = self.pad_out(gx)
        lx = self.pad_out(lx)
        x = self.proj(gx + lx)
        x = x + self.drop_path(x)
        x = x + self.drop_path(self.mlp(self.norm2(x)))

        return x

class GLTM(nn.Module):
    def __init__(self, dim=512, num_heads=6,  mlp_ratio=4,drop=0.,window_size=8,
                 drop_path=0., act_layer=nn.ReLU6, norm_layer=nn.BatchNorm2d):
        super().__init__()
        self.norm1 = norm_layer(dim)
        self.window_size = window_size

        self.msa = GlobalAttention(query_dim = dim,key_dim = dim,value_dim=dim,window_size =window_size,num_heads=num_heads,Mode = 'conv')
        self.cmsa = MultiHeadAttention(query_dim = dim, key_dim = dim, num_units = dim, num_heads=num_heads, Mode= 'conv')
        self.local = AdaptiveLocalFeatureExtraction(dim, ratio=8, mode='hv')
        self.proj = SeparableConvBN(dim, dim, kernel_size=window_size)

        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()
        mlp_hidden_dim = int(dim // mlp_ratio)
        self.mlp = Mlp_decoder(in_features=dim, hidden_features=mlp_hidden_dim, out_features=dim, act_layer=act_layer,
                               drop=drop)
        self.norm2 = norm_layer(dim)

    def pad_out(self, x):
        x = F.pad(x, pad=(0, 1, 0, 1), mode='reflect')
        return x

    def forward(self, x):

        b,c,h,w = x.size()

        x = self.norm1(x)
        vf = x.clone()
        gx = self.msa(x, x)
        Adaptivepool = nn.AdaptiveAvgPool2d((h//self.window_size,w//self.window_size))
        wgf = Adaptivepool(x)
        b, c, hh, ww = wgf.size()
        vqk_view = wgf.clone().permute(0, 2, 3, 1).reshape(b , -1 , c).contiguous()
        wgf = self.cmsa(vqk_view, vqk_view)  #
        wgf = wgf.reshape(b, hh, ww, c).permute(0, 3, 1, 2).contiguous()  # + vf
        wgf = F.interpolate(wgf, scale_factor=self.window_size, mode='bilinear', align_corners=False)
        gx = gx + wgf
        lx = self.local(vf)
        gx = self.pad_out(gx)
        lx = self.pad_out(lx)
        x = self.proj(gx + lx)
        x = x + self.drop_path(x)
        x = x + self.drop_path(self.mlp(self.norm2(x)))

        return x

class LearnableDWT(nn.Module):
    def __init__(self, in_channels):
        super(LearnableDWT, self).__init__()
        self.in_channels = in_channels

        # 初始化为 Haar 小波核
        haar_low = torch.tensor([1 / 2 ** 0.5, 1 / 2 ** 0.5])
        haar_high = torch.tensor([1 / 2 ** 0.5, -1 / 2 ** 0.5])

        # 构造二维 separable 卷积核（行列组合）
        # low x low, low x high, high x low, high x high
        self.LL_weight = nn.Parameter(torch.einsum('i,j->ij', haar_low, haar_low)[None, None, :, :])
        self.LH_weight = nn.Parameter(torch.einsum('i,j->ij', haar_low, haar_high)[None, None, :, :])
        self.HL_weight = nn.Parameter(torch.einsum('i,j->ij', haar_high, haar_low)[None, None, :, :])
        self.HH_weight = nn.Parameter(torch.einsum('i,j->ij', haar_high, haar_high)[None, None, :, :])

        # 每个核都扩展到 in_channels 个 group 卷积
        self.LL_weight = nn.Parameter(self.LL_weight.repeat(in_channels, 1, 1, 1))
        self.LH_weight = nn.Parameter(self.LH_weight.repeat(in_channels, 1, 1, 1))
        self.HL_weight = nn.Parameter(self.HL_weight.repeat(in_channels, 1, 1, 1))
        self.HH_weight = nn.Parameter(self.HH_weight.repeat(in_channels, 1, 1, 1))

    def forward(self, x):
        B, C, H, W = x.shape

        # 确保输入为偶数尺寸
        assert H % 2 == 0 and W % 2 == 0, "输入高宽必须为偶数"

        LL = F.conv2d(x, self.LL_weight, stride=2, groups=self.in_channels)
        LH = F.conv2d(x, self.LH_weight, stride=2, groups=self.in_channels)
        HL = F.conv2d(x, self.HL_weight, stride=2, groups=self.in_channels)
        HH = F.conv2d(x, self.HH_weight, stride=2, groups=self.in_channels)

        return LL, LH, HL, HH

class LearnableIDWT(nn.Module):
    def __init__(self, in_channels):
        super(LearnableIDWT, self).__init__()
        self.in_channels = in_channels

        sqrt2_inv = 1 / 2 ** 0.5
        haar_low = torch.tensor([sqrt2_inv, sqrt2_inv])
        haar_high = torch.tensor([sqrt2_inv, -sqrt2_inv])

        self.register_parameter('low_filter', nn.Parameter(haar_low[None, None, :].repeat(in_channels, 1, 1)))
        self.register_parameter('high_filter', nn.Parameter(haar_high[None, None, :].repeat(in_channels, 1, 1)))

    def upconv(self, x, kernel, dim):
        if dim == 'vertical':
            # ConvTrans2d over height axis
            kernel = kernel.unsqueeze(3)  # shape: [C, 1, 2, 1]
            return F.conv_transpose2d(x, kernel, stride=(2, 1), groups=self.in_channels)
        else:
            # ConvTrans2d over width axis
            kernel = kernel.unsqueeze(2)  # shape: [C, 1, 1, 2]
            return F.conv_transpose2d(x, kernel, stride=(1, 2), groups=self.in_channels)

    def forward(self, LL, LH, HL, HH):
        # 先恢复列方向
        L = self.upconv(LL, self.low_filter, 'vertical') + self.upconv(LH, self.high_filter, 'vertical')
        H = self.upconv(HL, self.low_filter, 'vertical') + self.upconv(HH, self.high_filter, 'vertical')

        # 再恢复行方向
        out = self.upconv(L, self.low_filter, 'horizontal') + self.upconv(H, self.high_filter, 'horizontal')
        return out

class SpatialAttention(nn.Module):
    """空间注意力机制，通过通道压缩生成空间权重图"""
    def __init__(self):
        super().__init__()
        self.sa = nn.Sequential(
            nn.Conv2d(2, 1, 7, padding=3, padding_mode='reflect'),
            nn.Sigmoid()
        )

    def forward(self, x):
        x_avg = torch.mean(x, dim=1, keepdim=True)  # 通道维度平均
        x_max, _ = torch.max(x, dim=1, keepdim=True)  # 通道维度最大
        return self.sa(torch.cat([x_avg, x_max], dim=1))  # 双通道融合

class ChannelAttention(nn.Module):
    """通道注意力机制，采用SE模块结构"""
    def __init__(self, dim, reduction=8):
        super().__init__()
        self.gap = nn.AdaptiveAvgPool2d(1)
        self.ca = nn.Sequential(
            nn.Conv2d(dim, dim//reduction, 1),
            nn.ReLU(inplace=True),
            nn.Conv2d(dim//reduction, dim, 1),
            nn.Sigmoid()  # 添加Sigmoid确保输出在[0,1]范围
        )

    def forward(self, x):
        return self.ca(self.gap(x))  # [B,C,1,1]

class PixelAttention(nn.Module):
    """像素级注意力，融合特征图与注意力图"""
    def __init__(self, dim):
        super().__init__()
        self.pa_conv = nn.Sequential(
            nn.Conv2d(2*dim, dim, 7, padding=3, padding_mode='reflect', groups=dim),
            nn.Sigmoid()
        )

    def forward(self, x, pattn1):
        # 拼接原始特征与初级注意力图
        x_cat = torch.cat([x, pattn1], dim=1)  # [B,2C,H,W]
        return self.pa_conv(x_cat)  # [B,C,H,W]

class ConvBlock(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=3, dilation=1, stride=1,
                 norm_layer=nn.BatchNorm2d, linearity=nn.ReLU6, groups=1, bias=False, mode="square"):
        super().__init__()

        if mode == "vertical":
            self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=(kernel_size, 1), bias=bias,
                                  dilation=(dilation, dilation), stride=(stride, stride),
                                  padding=(((stride - 1) + dilation * (kernel_size - 1)) // 2, 0), groups=groups
                                  )
        elif mode == "horizontal":
            self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=(1, kernel_size), bias=bias,
                                  dilation=(dilation, dilation), stride=(stride, stride),
                                  padding=(0, ((stride - 1) + dilation * (kernel_size - 1)) // 2), groups=groups
                                  )
        else:
            self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=kernel_size, bias=bias,
                                  dilation=dilation, stride=stride,
                                  padding=((stride - 1) + dilation * (kernel_size - 1)) // 2, groups=groups)


        # If norm_layer is provided, initialize it, otherwise None
        self.with_batchnorm = norm_layer is not None
        if self.with_batchnorm:
            self.bn = norm_layer(out_channels)

        # If linearity is provided, initialize it, otherwise None
        self.with_nonlinearity = linearity is not None
        if self.with_nonlinearity:
            self.relu = linearity()

    def forward(self, x):
        x = self.conv(x)
        if self.with_batchnorm:
            x = self.bn(x)
        if self.with_nonlinearity:
            x = self.relu(x)
        return x

class CGAFusion(nn.Module):
    def __init__(self, in_channels=64, out_channels=64, eps=1e-8, reduction=8):  # 修正拼写错误：in_channsel -> in_channels
        super(CGAFusion, self).__init__()

        # 可学习的融合权重
        self.weights = nn.Parameter(torch.ones(2, dtype=torch.float32), requires_grad=True)
        self.eps = eps

        # 大尺度下采样
        self.ldwt = LearnableDWT(in_channels)
        self.LL_conv = ConvBlock(in_channels, in_channels, 3, norm_layer=nn.BatchNorm2d, mode="square")
#        self.LH_conv = ConvBlock(in_channels, in_channels, 3, norm_layer=nn.BatchNorm2d, mode="vertical")  #"horizontal"
 #       self.HL_conv = ConvBlock(in_channels, in_channels, 3, norm_layer=nn.BatchNorm2d, mode="horizontal")  #"vertical"
#        self.HH_conv = ConvBlock(in_channels, in_channels, 3, norm_layer=nn.BatchNorm2d, mode="square")

        # 注意力模块组合
        self.sa = SpatialAttention()
        self.ca = ChannelAttention(out_channels, reduction)

        # 后处理模块
        self.lidwt = LearnableIDWT(in_channels)
        self.sigmoid = nn.Sigmoid()
        self.post_conv = SeparableConvBNReLU(out_channels, out_channels, 3)

    def forward(self, x, y):
        ##大尺度下采样与分量处理
        LL, LH, HL, HH =self.ldwt(y)
   #     LL = self.LL_conv(LL)
   #     LH = self.LH_conv(LH)
   #     HL = self.HL_conv(HL)
    #    HH = self.HH_conv(HH)


        # 特征对齐与加权融合
        weights = nn.ReLU6()(self.weights)
        fuse_weights = weights / (torch.sum(weights, dim=0) + self.eps)
        initial = fuse_weights[0] * x + fuse_weights[1] *LL
        # 多级注意力协同
        cattn = self.ca(initial)  # 通道注意力 [B,C,1,1]
        sattn = self.sa(initial)  # 空间注意力 [B,1,H,W]
        pattn1 = sattn + cattn  # 空间+通道注意力融合
        # 残差融合
        result = initial + pattn1 * x + (1 - pattn1) * LL
        out = self.lidwt(result, LH, HL, HH)

        return self.post_conv(out)  # 后处理卷积

class SegHead(nn.Module):

    def __init__(self, in_channels=64, num_classes=8):
        super().__init__()
        self.conv = ConvBNReLU(in_channels, in_channels)
        self.drop = nn.Dropout(0.1)
        self.conv_out = Conv(in_channels, num_classes, kernel_size=1)

    def forward(self, x, h, w):
        feat = self.conv(x)
        feat = self.drop(feat)
        feat = self.conv_out(feat)
        aux = F.interpolate(feat, size=(h, w), mode='bilinear', align_corners=False)

        return feat,aux

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

class Decoder(nn.Module):
    def __init__(self,
                 encode_channels=[256, 512, 1024, 2048],
                 decode_channels=[256, 512, 1024, 2048],
                 dilation = [[1, 2, 4, 8], [1, 2, 4, 8], [1, 2, 4, 8], [1, 2, 4, 8]],
                 fc_ratio=4,
                 dropout=0.1,
                 num_classes=6,
                 window_size=8,
                 weight_ratio = 1.0):
        super(Decoder, self).__init__()

        self.Conv1 = ConvBNReLU(encode_channels[-1], decode_channels[-1], 1)
        self.Conv2 = ConvBNReLU(encode_channels[-2], decode_channels[-2], 1)
        self.Conv3 = ConvBNReLU(encode_channels[-3], decode_channels[-3], 1)
        self.Conv4 = ConvBNReLU(encode_channels[-4], decode_channels[-4], 1)

        self.b4 = GLTM(dim=decode_channels[-1],window_size=window_size, num_heads=num_classes, mlp_ratio=fc_ratio)

        self.p3 = CGAFusion(decode_channels[-1], decode_channels[-2])
        self.b3 = GLSTM(dim=decode_channels[-2],window_size=window_size, num_heads=num_classes, mlp_ratio=fc_ratio,weight_ratio = weight_ratio)


        self.p2 = CGAFusion(decode_channels[-1], decode_channels[-2])
        self.b2 = GLSTM(dim=decode_channels[-3],window_size=window_size, num_heads=num_classes, mlp_ratio=fc_ratio,weight_ratio = weight_ratio)

        self.p1 = CGAFusion(decode_channels[-1], decode_channels[-2])
        self.b1 = GLSTM(dim=decode_channels[-4],window_size=window_size, num_heads=num_classes, mlp_ratio=fc_ratio,weight_ratio = weight_ratio)

        self.seg_head = nn.Sequential(SeparableConvBNReLU(decode_channels[-4], decode_channels[-4], kernel_size=3),
                      nn.Dropout2d(p=dropout, inplace=True),
                      Conv(decode_channels[-4], num_classes, kernel_size=1))
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
        x = self.b4(res4)

        x = self.p3(x, res3)
        aux3_3, aux3 = self.aux_head3(x, h, w)
        x = self.b3(aux3_3, x)

        x = self.p2(x, res2)
        aux2_2, aux2 = self.aux_head2(x, h, w)
        x = self.b2(aux2_2, x)

        x = self.p1(x, res1)
        aux1_1, aux1 = self.aux_head1(x, h, w)
        x = self.b1(aux1_1, x)
        feature = x

#        x = self.Conv5(x)
#        x = self.p(x, res)
        x = self.seg_head(x)

        x = F.interpolate(x, size=(h, w), mode='bilinear', align_corners=False)

        return x, aux1,aux2,aux3,aux4,aux1_1,aux2_2,aux3_3,aux4_4,feature

    def init_weight(self):
        for m in self.children():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, a=1)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)



class CIGFormer(nn.Module):
    def __init__(self,num_classes,
                 dropout=0.1,
                 fc_ratio=4,
                 decode_channels=32):
        super(CIGFormer, self).__init__()

        self.backbone = timm.create_model('swsl_resnet50', features_only=True, output_stride=32,
                                          out_indices=(1, 2, 3, 4), pretrained=True)
#        encoder_channels = self.backbone.feature_info.channels()


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
                               dropout=dropout, num_classes=num_classes,weight_ratio =1.0)


    def forward(self, x):

        h, w = vis.size()[-2:]

        # Encoder ResNet50
        x_pre = self.cnn(x)    ##H/2
        res1 = self.cnn1(x_pre)##H/4
        res2 = self.cnn2(res1) ##H/8
        res3 = self.cnn3(res2) ##H/16
        res4 = self.cnn4(res3) ##H/32

        ##
        out, aux1,aux2,aux3,aux4,aux1_1,aux2_2,aux3_3,aux4_4, feature = self.decoder(x_pre, res1, res2, res3, res4,h, w)


        if self.training:

            return out, aux1,aux2,aux3
        else:

            return out
        
from fvcore.nn import FlopCountAnalysis
if __name__ == '__main__':

    num_classes = 6
    in_batch, inchannel, in_h, in_w = 1, 3, 1024, 1024
    vis = torch.randn(in_batch, 3, in_h, in_w)
    net = CIGFormer(num_classes)
    out,aux2,aux3,aux4 = net(vis)
    print(out.shape)

    total = sum([param.nelement() for param in net.parameters()])
    print("Number of parameter: %.2fM" % (total / 1e6))
    FLOPS = FlopCountAnalysis(net,inputs=(vis))
    print("Number of parameter: %.2fM" % (FLOPS.total() / 1e9))