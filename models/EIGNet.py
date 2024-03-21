""" Parts of the U-Net model """

import timm
import torch.nn.functional as F
import torch
from torch import nn
from thop import profile
import time

from functools import partial



nonlinearity = partial(F.relu, inplace=True)
class DecoderBlock(nn.Module):
    def __init__(self, in_channels, n_filters, eps=1e-8):
        super(DecoderBlock, self).__init__()

        self.CBR1 = nn.Sequential(nn.Conv2d(in_channels, in_channels // 4, 1),
                                  nn.BatchNorm2d(in_channels // 4),
                                  nn.ReLU6())


 #       self.deconv2 = nn.ConvTranspose2d(in_channels // 4, in_channels // 4, 3, stride=2, padding=1, output_padding=1)
        self.CBR2 = nn.Sequential(nn.BatchNorm2d(in_channels // 4),
                                  nn.ReLU6())


        self.CBR3 = nn.Sequential(nn.Conv2d(in_channels // 4, n_filters, 1),
                                  nn.BatchNorm2d(n_filters),
                                  nn.ReLU6())


    def forward(self, x):
        x = self.CBR1(x)
        x = F.interpolate(x, scale_factor=2, mode='bilinear', align_corners=False)
        x = self.CBR2(x)
        x = self.CBR3(x)

        return x


class SeparableConvBN(nn.Sequential):
    def __init__(self, in_channels, out_channels, kernel_size=3, stride=1, dilation=1,
                 norm_layer=nn.BatchNorm2d):
        super(SeparableConvBN, self).__init__(
            nn.Conv2d(in_channels, in_channels, kernel_size, stride=stride, dilation=dilation,
                      padding=((stride - 1) + dilation * (kernel_size - 1)) // 2,
                      groups=in_channels, bias=False),
            norm_layer(in_channels),
            nn.Conv2d(in_channels, out_channels, kernel_size=1, bias=False)
        )

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
    def __init__(self, in_channels, out_channels, kernel_size=3, dilation=1, stride=1, norm_layer=nn.BatchNorm2d,
                 bias=False):
        super(ConvBNReLU, self).__init__(
            nn.Conv2d(in_channels, out_channels, kernel_size=kernel_size, bias=bias,
                      dilation=dilation, stride=stride, padding=((stride - 1) + dilation * (kernel_size - 1)) // 2),
            norm_layer(out_channels),
            nn.ReLU6()
        )

class ConvBNActx(nn.Sequential):
    def __init__(self, in_channels, out_channels, kernel_size=3, dilation=1, stride=1,
                 norm_layer=nn.BatchNorm2d, act_layer=nn.ReLU6, bias=False, inplace=False):
        super(ConvBNActx, self).__init__(
            nn.Conv2d(in_channels, out_channels, kernel_size=(1, kernel_size), bias=bias,
                      dilation=(dilation, dilation), stride=(stride, stride),
                      padding=(0, ((stride - 1) + dilation * (kernel_size - 1)) // 2)
                      ),
            norm_layer(out_channels),
            act_layer(inplace=inplace)
        )

class ConvBNActy(nn.Sequential):
    def __init__(self, in_channels, out_channels, kernel_size=3, dilation=1, stride=1,
                 norm_layer=nn.BatchNorm2d, act_layer=nn.ReLU6, bias=False, inplace=False, num_classes=6):
        super(ConvBNActy, self).__init__(
            nn.Conv2d(in_channels, out_channels, kernel_size=(kernel_size, 1), bias=bias,
                      dilation=(dilation, dilation), stride=(stride, stride),
                      padding=(((stride - 1) + dilation * (kernel_size - 1)) // 2, 0)
                      ),
            norm_layer(out_channels),
            act_layer(inplace=inplace)
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
    #

class ConvBN(nn.Sequential):
    def __init__(self, in_channels, out_channels, kernel_size=3, dilation=1, stride=1,
                 norm_layer=nn.BatchNorm2d, act_layer=nn.ReLU6, bias=False, inplace=False):
        super(ConvBN, self).__init__(
            nn.Conv2d(in_channels, out_channels, kernel_size=kernel_size, bias=bias,
                      dilation=dilation, stride=stride, padding=((stride - 1) + dilation * (kernel_size - 1)) // 2),
            norm_layer(out_channels)
        )

class SEM(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=3, stride=1):
        super().__init__()

        self.convx = ConvBNAct(in_channels, in_channels, kernel_size=1, stride=1)
        self.convy = ConvBNAct(in_channels, in_channels, kernel_size=1, stride=1)
        self.Recx = ConvBNActx(in_channels, out_channels, kernel_size=kernel_size)
        self.Recy = ConvBNActy(in_channels, out_channels, kernel_size=kernel_size)
        self.conv = ConvBNAct(in_channels, out_channels, kernel_size=kernel_size, stride=1, dilation=1)
        self.outconv = ConvBNAct(out_channels, out_channels, kernel_size=kernel_size, stride=stride, dilation=1)
        self.init_weight()

    def forward(self, x):

        feats = self.outconv(self.Recx(self.convx(x)) + self.Recy(self.convy(x)) + self.conv(x))

        return feats

    def init_weight(self):
        for m in self.children():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, a=1)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)

class EDGAuX(nn.Module):

    def __init__(self, encoder_channels=(64, 128, 256, 512), num_classes=6):
        super().__init__()

        self.up2 = nn.UpsamplingBilinear2d(scale_factor=2)
        self.up4 = nn.UpsamplingBilinear2d(scale_factor=4)
        self.up8 = nn.UpsamplingBilinear2d(scale_factor=8)

        self.conv = ConvBNReLU(in_channels=encoder_channels[0], out_channels=encoder_channels[0])
        self.conv1 = ConvBNReLU(in_channels=encoder_channels[1], out_channels=encoder_channels[0])
        self.conv2 = ConvBNReLU(in_channels=encoder_channels[2], out_channels=encoder_channels[0])
        self.conv3 = ConvBNReLU(in_channels=encoder_channels[3], out_channels=encoder_channels[0])

        self.convfuse = Conv(encoder_channels[0] *3, encoder_channels[0], kernel_size=1)
        self.convfuse1 = Conv(encoder_channels[0], num_classes, kernel_size=3)
        self.drop = nn.Dropout(0.1)
        self.edgconv_out = Conv(num_classes, 1, kernel_size=1)
        self.sigmoid = nn.Sigmoid()


    def forward(self, x1, x2, x3, x4 ,h ,w):

#        e1 = self.conv(x1)
        e2 = self.conv1(x2)
        e3 = self.conv2(x3)
        e4 = self.conv3(x4)

        feat = torch.cat((self.up2(e2), self.up4(e3), self.up8(e4)),dim=1)
        feat = self.convfuse(feat)
        feat = F.interpolate(feat, size=(h, w), mode='bilinear', align_corners=False)
        feat = self.drop(feat)
        feat = self.convfuse1(feat)
        feat = self.sigmoid(feat)

        few = self.edgconv_out(feat)
        few = self.sigmoid(few)

        return feat,few

class OACM(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=3):
        super().__init__()

        self.se = nn.Sequential(nn.AdaptiveAvgPool2d(1),
                                Conv(in_channels, in_channels // 8, kernel_size=1),
                                nn.ReLU6(),
                                Conv(in_channels // 8, in_channels, kernel_size=1),
                                nn.Sigmoid())

        self.usalconv = nn.Sequential(ConvBNAct(in_channels, out_channels, kernel_size=1, stride=1),
                                      ConvBNAct(out_channels, out_channels, kernel_size=3, stride=1))

        self.convx = ConvBNAct(in_channels, out_channels, kernel_size=1, stride=1)
        self.Recx = ConvBNActx(out_channels, out_channels//4, kernel_size=kernel_size)
        self.Recx2 = ConvBNActx(out_channels, out_channels//4, kernel_size=kernel_size, dilation=2)
        self.Recx4 = ConvBNActx(out_channels, out_channels//4, kernel_size=kernel_size, dilation=4)
        self.Recx8 = ConvBNActx(out_channels, out_channels//4, kernel_size=kernel_size, dilation=8)

        self.convy = ConvBNAct(in_channels, out_channels, kernel_size=1, stride=1)
        self.Recy = ConvBNActy(out_channels, out_channels//4, kernel_size=kernel_size)
        self.Recy2 = ConvBNActy(out_channels, out_channels//4, kernel_size=kernel_size, dilation=2)
        self.Recy4 = ConvBNActy(out_channels, out_channels//4, kernel_size=kernel_size, dilation=4)
        self.Recy8 = ConvBNActy(out_channels, out_channels//4, kernel_size=kernel_size, dilation=8)

        self.preconv = ConvBNAct(in_channels, out_channels, kernel_size=1, stride=1)
        self.conv = ConvBNAct(out_channels, out_channels//4, kernel_size=kernel_size, stride=1, dilation=1)
        self.conv2 = ConvBNAct(out_channels, out_channels//4, kernel_size=kernel_size, stride=1, dilation=2)
        self.conv4 = ConvBNAct(out_channels, out_channels//4, kernel_size=kernel_size, stride=1, dilation=4)
        self.conv8 = ConvBNAct(out_channels, out_channels//4, kernel_size=kernel_size, stride=1, dilation=8)

        self.convxout = ConvBNAct(out_channels, out_channels, stride=1)


        self.init_weight()

    def forward(self, x):

        x = self.se(x) * x + x

        featsx = torch.cat((self.Recx(self.convx(x)), self.Recx2(self.convx(x)), self.Recx4(self.convx(x)), self.Recx8(self.convx(x))),dim=1)
        featsy = torch.cat((self.Recy(self.convy(x)), self.Recy2(self.convy(x)), self.Recy4(self.convy(x)), self.Recy8(self.convy(x))),dim=1)
        feats = torch.cat((self.conv(self.preconv(x)), self.conv2(self.preconv(x)), self.conv4(self.preconv(x)), self.conv8(self.preconv(x))),dim=1)
        out = featsx + featsy + feats
        out = self.convxout(out)

        return out

    def init_weight(self):
        for m in self.children():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, a=1)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)

class HIAM(nn.Module):
    def __init__(self, in_channels, out_channels, num_classes = 6, kernel_size=3):
        super().__init__()

        self.SA = SAM(drop_rate=0.1)

        self.degeweight = nn.Sequential(ConvBNAct(in_channels=num_classes, out_channels=1,kernel_size=1),
                                        nn.Sigmoid())

        self.fuseconv = ConvBNAct(in_channels=in_channels, out_channels=in_channels,
                                 kernel_size=1)

        self.outconv = ConvBNAct(in_channels=in_channels, out_channels=out_channels,
                              kernel_size=kernel_size)



        self.init_weight()

    def forward(self, x , res , fe):

        b, c, h, w = x.shape

        AvgPool = nn.AdaptiveAvgPool2d((h, w))
        few = AvgPool(fe)
        few = self.degeweight(few)

        f = self.fuseconv((x * few + x) + (res * few + res))

        w_sa = self.SA(f)

        out = self.outconv(w_sa * (f + x + res))

        return out

    def init_weight(self):
        for m in self.children():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, a=1)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)

class SAM(torch.nn.Module):
    def __init__(self,drop_rate=0.1):
        super(SAM, self).__init__()

        # spatial attention for F_l
        self.compress = ChannelPool()
        self.spatial = ConvBN(2, 1, 7)


        self.dropout = nn.Dropout2d(drop_rate)
        self.drop_rate = drop_rate
        self.sigmoid = nn.Sigmoid()


    def forward(self, x):

        x = self.compress(x)
        x = self.spatial(x)
        weight_pool = self.sigmoid(x)  # B 1 H W

        out = weight_pool

        if self.drop_rate > 0:
            return self.dropout(out)
        else:
            return out

class EIGNet(nn.Module):
    def __init__(self,num_classes):
        super(EIGNet, self).__init__()

        self.backbone = timm.create_model('swsl_resnet18', features_only=True, output_stride=32,
                                          out_indices=(1, 2, 3, 4), pretrained=True)
        encoder_channels = self.backbone.feature_info.channels()

        self.cnn = nn.Sequential(self.backbone.conv1,
                                 self.backbone.bn1,
                                 self.backbone.act1,
                                 self.backbone.maxpool
                                 )

        self.cnn1 = self.backbone.layer1
        self.cnn2 = self.backbone.layer2
        self.cnn3 = self.backbone.layer3
        self.cnn4 = self.backbone.layer4

        # 
        self.preedge = nn.Sequential(ConvBNReLU(3, encoder_channels[0], kernel_size=7, stride=2),
                                     SEM(in_channels=encoder_channels[0], out_channels=encoder_channels[0],
                                         kernel_size=3, stride=2),
                                     )
        self.OCM = SEM(in_channels=encoder_channels[0], out_channels=encoder_channels[0], kernel_size=3, stride=1)
        self.OCM1 = SEM(in_channels=encoder_channels[0], out_channels=encoder_channels[1], kernel_size=3, stride=2)
        self.OCM2 = SEM(in_channels=encoder_channels[1], out_channels=encoder_channels[2], kernel_size=3, stride=2)
        self.OCM3 = SEM(in_channels=encoder_channels[2], out_channels=encoder_channels[3], kernel_size=3, stride=2)

        ##
        self.getedge = EDGAuX(encoder_channels=encoder_channels, num_classes=num_classes)

        ##
        self.hiam = HIAM(in_channels=encoder_channels[0], out_channels=encoder_channels[0],num_classes = num_classes, kernel_size=3)
        self.hiam1 = HIAM(in_channels=encoder_channels[1], out_channels=encoder_channels[1],num_classes = num_classes,  kernel_size=3)
        self.hiam2 = HIAM(in_channels=encoder_channels[2], out_channels=encoder_channels[2],num_classes = num_classes,  kernel_size=3)
        self.hiam3 = HIAM(in_channels=encoder_channels[3], out_channels=encoder_channels[3],num_classes = num_classes,  kernel_size=3)

        ## 
        self.oacm = OACM(in_channels=encoder_channels[0], out_channels=encoder_channels[0], kernel_size=3)
        self.oacm1 = OACM(in_channels=encoder_channels[1], out_channels=encoder_channels[1], kernel_size=3)
        self.oacm2 = OACM(in_channels=encoder_channels[2], out_channels=encoder_channels[2], kernel_size=3)
        self.oacm3 = OACM(in_channels=encoder_channels[3], out_channels=encoder_channels[3], kernel_size=3)



        self.decoder4 = DecoderBlock(encoder_channels[3], encoder_channels[2])
        self.decoder3 = DecoderBlock(encoder_channels[2], encoder_channels[1])
        self.decoder2 = DecoderBlock(encoder_channels[1], encoder_channels[0])
        self.decoder1 = DecoderBlock(encoder_channels[0], encoder_channels[0])


        self.finaldeconv1 = nn.ConvTranspose2d(encoder_channels[0], 32, 4, 2, 1)
        self.finalrelu1 = nonlinearity
        self.finalconv2 = nn.Conv2d(32, 32, 3, padding=1)
        self.finalrelu2 = nonlinearity
        self.finalconv3 = nn.Conv2d(32, num_classes, 3, padding=1)



    def forward(self, x):
        h, w = x.size()[-2:]
        # Encoder ResNet18 
        x_pre = self.cnn(x)
        res1 = self.cnn1(x_pre)
        res2 = self.cnn2(res1)
        res3 = self.cnn3(res2)
        res4 = self.cnn4(res3)

        # 
        
        s_pre = self.preedge(x)
        s1 = self.OCM(s_pre)
        s2 = self.OCM1(s1)
        s3 = self.OCM2(s2)
        s4 = self.OCM3(s3)
        fe, few = self.getedge(s1, s2, s3, s4, h, w)
        
        ## 
        
        fuse1 = self.hiam(s1, res1, fe)
        fuse2 = self.hiam1(s2, res2, fe)
        fuse3 = self.hiam2(s3, res3, fe)
        fuse4 = self.hiam3(s4, res4, fe)
        
        ##

        fuse4 = self.oacm3(fuse4)
        d4 = self.decoder4(fuse4) + fuse3
        d4 = self.oacm2(d4)
        d3 = self.decoder3(d4) + fuse2
        d3 = self.oacm1(d3)
        d2 = self.decoder2(d3)+ fuse1
        d2 = self.oacm(d2)


        d1 = self.decoder1(d2)
        out = self.finaldeconv1(d1)
        out = self.finalrelu1(out)
        out = self.finalconv2(out)
        out = self.finalrelu2(out)
        out = self.finalconv3(out)

        if self.training:

            return out,few
        else:

            return out



if __name__ == '__main__':
    num_classes = 6
    in_batch, inchannel, in_h, in_w = 1, 3, 1024, 1024
    x = torch.randn(in_batch, inchannel, in_h, in_w)
    net = EIGNet(num_classes)
    out,shape1 = net(x)
    print(out.shape)
    print(net.getedge.edgconv_out)



