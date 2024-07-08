import torch
import functools
import numpy as np
import torch.nn.functional as F
from torch import nn
from torchvision.models import resnet


class ImageEncoder(nn.Module):
    def __init__(self, depth=18, dim=128):
        super().__init__()
        self.model = getattr(resnet, f'resnet{depth}')(pretrained=True)
        self.model.fc = nn.Linear(self.model.fc.weight.shape[1], dim)
        self.dim = dim

    def forward(self, x):
        image_embedding = self.encode(x)
        out = self.model.avgpool(image_embedding).flatten(1)
        out = self.model.fc(out)
        return out

    def encode(self, x):
        x = self.model.conv1(x)
        x = self.model.bn1(x)
        x = self.model.relu(x)
        x = self.model.maxpool(x)

        x = self.model.layer1(x)
        x = self.model.layer2(x)
        x = self.model.layer3(x)
        x = self.model.layer4(x)

        return x


class Conv2d(nn.Module):
    def __init__(self, cin, cout, kernel_size, stride, padding, residual=False, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.conv_block = nn.Sequential(
            nn.Conv2d(cin, cout, kernel_size, stride, padding),
            nn.BatchNorm2d(cout)
        )
        self.act = nn.ReLU()
        self.residual = residual

    def forward(self, x):
        out = self.conv_block(x)
        if self.residual:
            out += x
        return self.act(out)


class AudioEncoder(nn.Module):
    def __init__(self, dim=128, wav2lip_checkpoint='checkpoints/wav2lip.pth'):
        super(AudioEncoder, self).__init__()

        self.audio_encoder = nn.Sequential(
            Conv2d(1, 32, kernel_size=3, stride=1, padding=1),
            Conv2d(32, 32, kernel_size=3, stride=1, padding=1, residual=True),
            Conv2d(32, 32, kernel_size=3, stride=1, padding=1, residual=True),

            Conv2d(32, 64, kernel_size=3, stride=(3, 1), padding=1),
            Conv2d(64, 64, kernel_size=3, stride=1, padding=1, residual=True),
            Conv2d(64, 64, kernel_size=3, stride=1, padding=1, residual=True),

            Conv2d(64, 128, kernel_size=3, stride=3, padding=1),
            Conv2d(128, 128, kernel_size=3, stride=1,
                   padding=1, residual=True),
            Conv2d(128, 128, kernel_size=3, stride=1,
                   padding=1, residual=True),

            Conv2d(128, 256, kernel_size=3, stride=(3, 2), padding=1),
            Conv2d(256, 256, kernel_size=3, stride=1,
                   padding=1, residual=True),

            Conv2d(256, 512, kernel_size=3, stride=1, padding=0),
            Conv2d(512, 512, kernel_size=1, stride=1, padding=0),)

        # load the pre-trained audio_encoder, we do not need to load wav2lip model here.
        wav2lip_state_dict = torch.load(
            wav2lip_checkpoint, 'cpu')['state_dict']
        state_dict = self.audio_encoder.state_dict()

        for k, v in wav2lip_state_dict.items():
            if 'audio_encoder' in k:
                state_dict[k.replace('module.audio_encoder.', '')] = v
        self.audio_encoder.load_state_dict(state_dict)

        self.fc = nn.Linear(512, dim)

    def forward(self, x):
        # x (B, 1, 80, 16)
        audio_embedding = self.audio_encoder(x).flatten(1)  # B, 512
        out = self.fc(audio_embedding)
        return out


class CoeffEncoder(nn.Module):
    def __init__(self, nconv=14, dim=128, nchan=64, chin=1, downsample_freq=4):
        super().__init__()
        self.nconv = nconv
        convs = []
        for iconv in range(nconv):
            if (iconv+1) % 4 == 0:
                nchan *= 2
            if iconv % downsample_freq == 0:
                stride = 2
            else:
                stride = 1
            convs.append(nn.Sequential(*[
                nn.Conv1d(chin,nchan,3,stride=stride,padding=1,bias=False),
                nn.BatchNorm1d(nchan),
                nn.ReLU(inplace=True)
            ]))
            chin = nchan
        self.convs = nn.Sequential(*convs)
        self.fc = nn.Linear(nchan, dim)
        self.dim = dim

    def forward(self, x):
        return self.fc(self.convs(x).mean(-1))

def conv3x3(in_planes, out_planes, stride=1, groups=1, dilation=1):
    """3x3 convolution with padding"""
    return nn.Conv2d(in_planes,
                     out_planes,
                     kernel_size=3,
                     stride=stride,
                     padding=dilation,
                     groups=groups,
                     bias=False,
                     dilation=dilation)


def conv1x1(in_planes, out_planes, stride=1):
    """1x1 convolution"""
    return nn.Conv2d(in_planes,
                     out_planes,
                     kernel_size=1,
                     stride=stride,
                     bias=False)


class IBasicBlock(nn.Module):
    expansion = 1
    def __init__(self, inplanes, planes, stride=1, downsample=None,
                 groups=1, base_width=64, dilation=1):
        super(IBasicBlock, self).__init__()
        if groups != 1 or base_width != 64:
            raise ValueError('BasicBlock only supports groups=1 and base_width=64')
        if dilation > 1:
            raise NotImplementedError("Dilation > 1 not supported in BasicBlock")
        self.bn1 = nn.BatchNorm2d(inplanes, eps=1e-05,)
        self.conv1 = conv3x3(inplanes, planes)
        self.bn2 = nn.BatchNorm2d(planes, eps=1e-05,)
        self.prelu = nn.PReLU(planes)
        self.conv2 = conv3x3(planes, planes, stride)
        self.bn3 = nn.BatchNorm2d(planes, eps=1e-05,)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        identity = x
        out = self.bn1(x)
        out = self.conv1(out)
        out = self.bn2(out)
        out = self.prelu(out)
        out = self.conv2(out)
        out = self.bn3(out)
        if self.downsample is not None:
            identity = self.downsample(x)
        out += identity
        return out        


class IResNet(nn.Module):
    fc_scale = 7 * 7
    def __init__(self,
                 block, layers, dropout=0, num_features=512, zero_init_residual=False,
                 groups=1, width_per_group=64, replace_stride_with_dilation=None):
        super(IResNet, self).__init__()
        self.extra_gflops = 0.0
        self.inplanes = 64
        self.dilation = 1
        if replace_stride_with_dilation is None:
            replace_stride_with_dilation = [False, False, False]
        if len(replace_stride_with_dilation) != 3:
            raise ValueError("replace_stride_with_dilation should be None "
                             "or a 3-element tuple, got {}".format(replace_stride_with_dilation))
        self.groups = groups
        self.base_width = width_per_group
        self.conv1 = nn.Conv2d(3, self.inplanes, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(self.inplanes, eps=1e-05)
        self.prelu = nn.PReLU(self.inplanes)
        self.layer1 = self._make_layer(block, 64, layers[0], stride=2)
        self.layer2 = self._make_layer(block,
                                       128,
                                       layers[1],
                                       stride=2,
                                       dilate=replace_stride_with_dilation[0])
        self.layer3 = self._make_layer(block,
                                       256,
                                       layers[2],
                                       stride=2,
                                       dilate=replace_stride_with_dilation[1])
        self.layer4 = self._make_layer(block,
                                       512,
                                       layers[3],
                                       stride=2,
                                       dilate=replace_stride_with_dilation[2])
        self.bn2 = nn.BatchNorm2d(512 * block.expansion, eps=1e-05,)
        self.dropout = nn.Dropout(p=dropout, inplace=True)
        self.fc = nn.Linear(512 * block.expansion * self.fc_scale, num_features)
        self.features = nn.BatchNorm1d(num_features, eps=1e-05)
        nn.init.constant_(self.features.weight, 1.0)
        self.features.weight.requires_grad = False

        #self.upsample = nn.ConvTranspose2d(512, 256, 2, stride=2, padding=0)
        #self.fuse = nn.Sequential(nn.Conv2d(256, 256, 3, stride=1, padding=1, bias=False), nn.BatchNorm2d(256), nn.ReLU())

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.normal_(m.weight, 0, 0.1)
            elif isinstance(m, (nn.BatchNorm2d, nn.GroupNorm)):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

        if zero_init_residual:
            for m in self.modules():
                if isinstance(m, IBasicBlock):
                    nn.init.constant_(m.bn2.weight, 0)

    def _make_layer(self, block, planes, blocks, stride=1, dilate=False):
        downsample = None
        previous_dilation = self.dilation
        if dilate:
            self.dilation *= stride
            stride = 1
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                conv1x1(self.inplanes, planes * block.expansion, stride),
                nn.BatchNorm2d(planes * block.expansion, eps=1e-05, ),
            )
        layers = []
        layers.append(
            block(self.inplanes, planes, stride, downsample, self.groups,
                  self.base_width, previous_dilation))
        self.inplanes = planes * block.expansion
        for _ in range(1, blocks):
            layers.append(
                block(self.inplanes,
                      planes,
                      groups=self.groups,
                      base_width=self.base_width,
                      dilation=self.dilation))

        return nn.Sequential(*layers)

    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.prelu(x)
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        x = self.bn2(x)
        x = torch.flatten(x, 1)
        x = self.dropout(x)
        x = self.fc(x)
        x = self.features(x)
        return x


def IResNet18(pretrain_weight='checkpoints/iresnet18.pth', **kwargs):
    model = IResNet(IBasicBlock, [2, 2, 2, 2], **kwargs)
    state_dict = {k: v for k, v in torch.load(pretrain_weight, 'cpu').items() if not (k.startswith('fc.') or k.startswith('features.'))}
    model.load_state_dict(state_dict, strict=False)
    return model



class Generator(nn.Module):
    def __init__(self, input_nc=1, output_nc=3, image_nc=3, ngf=64, n_downsampling=3, n_blocks=9,
                 norm_layer=None, n_frames_G=1, padding_type='reflect'):
        assert(n_blocks >= 0)
        super(Generator, self).__init__()
        self.n_downsampling = n_downsampling
        self.n_frames_G = n_frames_G
        activation = nn.ReLU(True)
        if norm_layer is None:
            norm_layer = functools.partial(nn.InstanceNorm2d, affine=False, track_running_stats=True)

        #model_down_seg = [nn.ReflectionPad2d(3), nn.Conv2d(input_nc, ngf, kernel_size=7, padding=0), norm_layer(ngf), activation]
        #feat_dim = ngf*2**n_downsampling
        #model_down_seg = [nn.Conv2d(input_nc, feat_dim, 1, bias=False), nn.LayerNorm(feat_dim)] # nn.ReLU(), nn.Linear(feat_dim, feat_dim), nn.Sigmoid()
        model_down_img = [nn.ReflectionPad2d(3), nn.Conv2d(image_nc, ngf, kernel_size=7, padding=0), norm_layer(ngf), activation]
        for i in range(n_downsampling):
            mult = 2**i
            #model_down_seg += [nn.Conv2d(ngf * mult, ngf * mult * 2, kernel_size=3, stride=2, padding=1),
            #                   norm_layer(ngf * mult * 2), activation]
            model_down_img += [nn.Conv2d(ngf * mult, ngf * mult * 2, kernel_size=3, stride=2, padding=1),
                               norm_layer(ngf * mult * 2), activation]

        mult = 2**n_downsampling
        for i in range(n_blocks - n_blocks//2):
            #model_down_seg += [ResnetBlock(ngf * mult, padding_type=padding_type, activation=activation, norm_layer=norm_layer)]
            model_down_img += [ResnetBlock(ngf * mult, padding_type=padding_type, activation=activation, norm_layer=norm_layer)]

        model_res_img = []
        for i in range(n_blocks//2):
            model_res_img += [ResnetBlock(ngf * mult, padding_type=padding_type, activation=activation, norm_layer=norm_layer)]

        model_up_img = []
        for i in range(n_downsampling):
            mult = 2**(n_downsampling - i)
            model_up_img += [nn.ConvTranspose2d(ngf*mult, ngf*mult//2, kernel_size=3, stride=2, padding=1, output_padding=1),
                             norm_layer(ngf*mult//2), activation]

        model_final_img = [nn.ReflectionPad2d(3), nn.Conv2d(ngf, output_nc, kernel_size=7, padding=0), nn.Tanh()]

        #self.model_down_seg = nn.Sequential(*model_down_seg)
        self.model_down_img = nn.Sequential(*model_down_img)
        self.model_res_img = nn.Sequential(*model_res_img)
        self.model_up_img = nn.Sequential(*model_up_img)
        self.model_final_img = nn.Sequential(*model_final_img)
        self.proj1 = nn.Sequential(nn.Conv2d(ngf*2**n_downsampling, input_nc, 1, bias=False), norm_layer(input_nc))
        self.proj2 = nn.Sequential(nn.Conv2d(input_nc, ngf*2**n_downsampling, 1, bias=False), norm_layer(ngf*2**n_downsampling))

    def forward(self, input, img_prev):
        # mask = input[:,-1,:,:].unsqueeze(1)
        if self.n_frames_G > 1:
            input = torch.cat([ch for ch in torch.chunk(input, chunks=self.n_frames_G, dim=1)],1)

        seg_feat = F.normalize(input, dim=2) # B C C
        img_feat = self.model_down_img(img_prev) # B C H W

        downsample = self.proj2(torch.einsum('bcc,bchw->bchw', seg_feat, self.proj1(img_feat)))

        img_feat = self.model_up_img(self.model_res_img(img_feat + downsample))
        img_final = self.model_final_img(img_feat)

        # perform the face masking
        # img_final = img_final*mask - (1-mask)
        return img_final

# Define a resnet block
class ResnetBlock(nn.Module):
    def __init__(self, dim, padding_type, norm_layer, activation=nn.ReLU(True), use_dropout=False):
        super(ResnetBlock, self).__init__()
        self.conv_block = self.build_conv_block(dim, padding_type, norm_layer, activation, use_dropout)

    def build_conv_block(self, dim, padding_type, norm_layer, activation, use_dropout):
        conv_block = []
        p = 0
        if padding_type == 'reflect':
            conv_block += [nn.ReflectionPad2d(1)]
        elif padding_type == 'replicate':
            conv_block += [nn.ReplicationPad2d(1)]
        elif padding_type == 'zero':
            p = 1
        else:
            raise NotImplementedError('padding [%s] is not implemented' % padding_type)

        conv_block += [nn.Conv2d(dim, dim, kernel_size=3, padding=p),
                       norm_layer(dim),
                       activation]
        if use_dropout:
            conv_block += [nn.Dropout(0.5)]

        p = 0
        if padding_type == 'reflect':
            conv_block += [nn.ReflectionPad2d(1)]
        elif padding_type == 'replicate':
            conv_block += [nn.ReplicationPad2d(1)]
        elif padding_type == 'zero':
            p = 1
        else:
            raise NotImplementedError('padding [%s] is not implemented' % padding_type)
        conv_block += [nn.Conv2d(dim, dim, kernel_size=3, padding=p),
                       norm_layer(dim)]
        self.resnetblock_sequential = nn.Sequential(*conv_block)
        return self.resnetblock_sequential

    def forward(self, x):
        out = x + self.conv_block(x)
        return out

class MVal:
    def __init__(self, momentum=0.999, warmup=500):
        self.momentum = momentum
        self.warmup = warmup

        self.val = 0
        self.step = 0
    
    @torch.no_grad()
    def update(self, val):
        if self.step < self.warmup:
            momentum = np.linspace(0, self.momentum, self.warmup)[self.step]
            self.step += 1
        else:
            momentum = self.momentum
        self.val = self.val * momentum + (1 - momentum) * val
