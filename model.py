import math
import torch
import torch.nn as nn
import torchvision
import numpy as np
import torch.nn.functional as F
import numbers
from einops import rearrange


def generate_w():
    w = torch.tensor([[[1., 1.], [1., 1.]],
                      [[1., 1.], [-1., -1.]],
                      [[1., -1.], [1., -1.]],
                      [[1., -1.], [-1., 1.]]]).reshape(4, 1, 2, 2)
    return 0.5 * w


def dwt2_haar(data, w):
    coeff = []
    for i in range(data.shape[1]):
        coeff.append(F.conv2d(data[:, i:i+1, :, :], w, stride=2))
    coeffs = torch.cat(coeff).reshape(data.shape[0], -1, data.shape[2]//2, data.shape[3]//2)
    return coeffs

def idwt2_haar(data, w):
    coeff = []
    for i in range(data.shape[1] // 4):
        coeff.append(F.conv_transpose2d(data[:, 4*i:4*(i+1), :, :], w, stride=2))
    coeffs = torch.cat(coeff).reshape(data.shape[0], -1, data.shape[2] * 2, data.shape[3]*2)
    return coeffs

def cdwt2_haar(data, w):
    data_r, data_i = torch.unbind(data, -1)
    data_r = dwt2_haar(data_r, w)
    data_i = dwt2_haar(data_i, w)
    coeffs = torch.stack([data_r, data_i], -1)
    return coeffs

def cidwt2_haar(data, w):
    data_r, data_i = torch.unbind(data, -1)
    data_r = idwt2_haar(data_r, w)
    data_i = idwt2_haar(data_i, w)
    coeffs = torch.stack([data_r, data_i], -1)
    return coeffs


## Layer Norm
def to_4d(x):
    return rearrange(x, 'b c h w d -> b d (h w) c')

def to_5d(x,h,w):
    return rearrange(x, 'b d (h w) c -> b c h w d', h=h, w=w)

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
        return x / torch.sqrt(sigma+1e-5) * self.weight

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
        return (x - mu) / torch.sqrt(sigma+1e-5) * self.weight + self.bias


class LayerNorm(nn.Module):
    def __init__(self, dim, LayerNorm_type="BiasFree"):
        super(LayerNorm, self).__init__()
        if LayerNorm_type =='BiasFree':
            self.body = BiasFree_LayerNorm(dim)
        else:
            self.body = WithBias_LayerNorm(dim)

    def forward(self, x):
        h, w, d = x.shape[-3:]
        return to_5d(self.body(to_4d(x)), h, w)



class CDownsample(nn.Module):
    def __init__(self, n_feat):
        super(CDownsample, self).__init__()
        self.conv = ComplexConv(n_feat, n_feat // 2, kernel_size=3, stride=1, padding=1, bias=False)

    def forward(self, x):
        x = self.conv(x)
        x = F.pixel_unshuffle(x.permute(0, 4, 1, 2, 3), 2).permute(0, 2, 3, 4, 1)
        # print("x_down:", x.shape)
        return x

class CUpsample(nn.Module):
    def __init__(self, n_feat):
        super(CUpsample, self).__init__()
        self.conv = ComplexConv(n_feat, n_feat*2, kernel_size=3, stride=1, padding=1, bias=False)
    def forward(self, x):
        x = self.conv(x)
        x = F.pixel_shuffle(x.permute(0, 4, 1, 2, 3), 2).permute(0, 2, 3, 4, 1)
        return x


def fft_layer_complex(x, norm="ortho"):
    x_real, x_imag = torch.unbind(x, -1)
    x_fft = torch.fft.fft2(torch.complex(x_real, x_imag), norm=norm)
    x_fft = torch.stack([x_fft.real, x_fft.imag], -1)
    return x_fft

def ifft_layer_complex(x, norm="ortho"):
    x_real, x_imag = torch.unbind(x, -1)
    x_ifft = torch.fft.ifft2(torch.complex(x_real, x_imag), norm=norm)
    x_ifft = torch.stack([x_ifft.real, x_ifft.imag], -1)
    return x_ifft


class CReLU(nn.ReLU):
    def __init__(self, inplace: bool=False):
        super(CReLU, self).__init__(inplace)


class ComplexConv(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride=1, padding=0, dilation=1, groups=1, bias=False):
        super(ComplexConv, self).__init__()

        self.real = nn.Conv2d(in_channels, out_channels, kernel_size=kernel_size,
                              stride=stride, padding=padding, dilation=dilation, groups=groups, bias=bias)
        self.imag = nn.Conv2d(in_channels, out_channels, kernel_size=kernel_size,
                              stride=stride, padding=padding, dilation=dilation, groups=groups, bias=bias)

    def forward(self, inputs):
        x_r, x_i = torch.unbind(inputs, -1)
        y_real = self.real(x_r) - self.imag(x_i)
        y_imag = self.imag(x_r) + self.real(x_i)
        return torch.stack([y_real, y_imag], -1)


## Complex Transposed Self-Attention (CTA)
class Attention(nn.Module):
    def __init__(self, dim, bias=False):
        super(Attention, self).__init__()
        self.qkv = ComplexConv(dim, dim * 3, kernel_size=1, bias=bias)
        self.qkv_dwconv = ComplexConv(dim*3, dim*3, kernel_size=3, stride=1, padding=1, groups=dim*3, bias=bias)
        self.project_out = ComplexConv(dim, dim, kernel_size=1, bias=bias)

    def forward(self, x):
        b, c, w, h, d = x.shape
        qkv = self.qkv_dwconv(self.qkv(x))
        q, k, v = qkv.chunk(3, dim=1)

        q = rearrange(q, 'b c w h d -> b d c (h w)')
        k = rearrange(k, 'b c w h d -> b d c (h w)')
        v = rearrange(v, 'b c w h d -> b d c (h w)')

        q = torch.nn.functional.normalize(q, dim=-1)
        k = torch.nn.functional.normalize(k, dim=-1)
        attn =(q @ k.transpose(-2, -1)).softmax(dim=-1)
        out = (attn @ v)
        out = rearrange(out, 'b d c (h w) -> b c w h d', h=h, w=w)
        out = self.project_out(out)
        return out


#%% complex Gated-dconv Feed-Forward Network (CGFN)
class FeedForward(nn.Module):
    def __init__(self, dim, ffn_expansion_factor, bias):
        super(FeedForward, self).__init__()

        hidden_features = int(dim*ffn_expansion_factor)

        self.project_in = ComplexConv(dim, hidden_features*2, kernel_size=1, bias=bias)
        self.dwconv = ComplexConv(hidden_features*2, hidden_features*2, kernel_size=3, stride=1, padding=1,
                                  groups=hidden_features*2, bias=bias)
        self.project_out = ComplexConv(hidden_features, dim, kernel_size=1, bias=bias)

    def forward(self, x):
        x = self.project_in(x)
        x1, x2 = self.dwconv(x).chunk(2, dim=1)
        x = F.gelu(x1) * x2
        x = self.project_out(x)
        return x


class TransformerBlock(nn.Module):
    def __init__(self, dim, ffn_expansion_factor=2.66, bias=False, LayerNorm_type="BiasFree"):
        super(TransformerBlock, self).__init__()

        self.norm1 = LayerNorm(dim, LayerNorm_type)
        self.attn = Attention(dim, bias)
        self.norm2 = LayerNorm(dim, LayerNorm_type)
        self.ffn = FeedForward(dim, ffn_expansion_factor, bias)

    def forward(self, x):
        x = x + self.attn(self.norm1(x))
        x = x + self.ffn(self.norm2(x))
        return x


#%%fuse
class Fuse(nn.Module):
    def __init__(self, n_feat):
        super(Fuse, self).__init__()
        self.n_feat = n_feat
        self.att_channel = TransformerBlock(dim=n_feat * 2)

        self.conv = ComplexConv(n_feat * 2, n_feat * 2, 1, 1, 0)
        self.conv2 = ComplexConv(n_feat * 2, n_feat * 2, 1, 1, 0)

    def forward(self, enc, dnc):
        x = self.conv(torch.cat((enc, dnc), dim=1))
        x = self.att_channel(x)
        x = self.conv2(x)
        e, d = torch.split(x, [self.n_feat, self.n_feat], dim=1)
        output = e + d
        return output

class ComplexBlock(nn.Module):
    """
    double branch that including conv in image and FFT domain
    """
    def __init__(self, ch_in=1, conv_dim=16):
        super(ComplexBlock, self).__init__()

        self.img_branch = nn.Sequential(
            ComplexConv(in_channels=ch_in, out_channels=conv_dim, kernel_size=3, stride=1, padding=1),
            CReLU(),
            ComplexConv(in_channels=conv_dim, out_channels=conv_dim, kernel_size=3, stride=1, padding=1),
            CReLU()
        )
        self.fft_branch = nn.Sequential(
            ComplexConv(in_channels=ch_in, out_channels=conv_dim, kernel_size=1),
            CReLU(),
            ComplexConv(in_channels=conv_dim, out_channels=conv_dim, kernel_size=1),
            CReLU()
        )
        self.fuse = ComplexConv(conv_dim * 2, conv_dim, kernel_size=3, stride=1, padding=1)


    def forward(self, x):
        x1 = self.img_branch(x)
        x2 = ifft_layer_complex(self.fft_branch(fft_layer_complex(x)))
        out = self.fuse(torch.cat([x1, x2], 1))
        return out

class Basic_Block(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(Basic_Block, self).__init__()
        self.block = nn.Sequential(
            ComplexConv(in_channels, out_channels, kernel_size=3, stride=1, padding=1),
            CReLU()
        )
    def forward(self, x):
        return self.block(x)

class CUnet(nn.Module):
    def __init__(self, in_channels, out_channels, dim):
        super(CUnet, self).__init__()
        self.cu1 = Basic_Block(in_channels * 4, dim)
        self.encoder1 = nn.Sequential(ComplexBlock(dim, dim), TransformerBlock(dim))
        self.down1 = CDownsample(dim)
        self.encoder2_1 = nn.Sequential(ComplexBlock(dim * 2, dim * 2), TransformerBlock(dim * 2))
        self.encoder2_2 = nn.Sequential(ComplexBlock(dim * 2, dim * 2), TransformerBlock(dim * 2))
        self.down2 = CDownsample(dim * 2)
        self.latent = nn.Sequential(ComplexBlock(dim * 4, dim * 4), TransformerBlock(dim * 4))
        self.up2 = CUpsample(dim * 4)
        self.decoder2 = nn.Sequential(ComplexBlock(dim * 2, dim * 2), TransformerBlock(dim * 2))

        self.up1 = CUpsample(dim * 2)
        self.decoder1 = nn.Sequential(ComplexBlock(dim, dim), TransformerBlock(dim))
        self.convT = ComplexConv(dim, out_channels * 4, kernel_size=1)

        self.ff2 = Fuse(dim * 2)
        self.ff1 = Fuse(dim)


        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0.0, math.sqrt(0.5 / n))

            elif isinstance(m, (nn.BatchNorm2d, nn.GroupNorm)):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

    def forward(self, x):
        w = generate_w().to(x.device)
        x0 = self.cu1(cdwt2_haar(x.permute(0, 2, 3, 1).unsqueeze(1), w))
        x1 = self.encoder1(x0)
        x2 = self.encoder2_2(self.encoder2_1(self.down1(x1)))
        latent = self.latent(self.down2(x2))
        x23 = self.decoder2(self.ff2(x2, self.up2(latent)))
        out = self.decoder1(self.ff1(x1, self.up1(x23)))
        out = cidwt2_haar(self.convT(out), w)

        return x + out.squeeze(1).permute(0, 3, 1, 2)



def network_params(model):
    print('# The number of net parameters:', sum(param.numel() for param in model.parameters()) /1e6,"M")
    # return num_params


# if __name__ == '__main__':
#     device ="cpu"#"cuda:0"
#
#     net1 = CUnet(1, 1, 32).to(device)  #661,312(dim=16),2,639,744(dim=32),10,548,736(dim=64)
#     network_params(net1)
#     x = torch.randn(3, 2, 128, 128).to(device)
#     y1 = net1(x)
#     print(y1.shape)


