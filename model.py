from einops import rearrange
import numbers
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision.ops import DeformConv2d  # torchvision >= 0.10 필요


class DCMBlock(nn.Module):
    def __init__(self, in_channels=192, mid_channels=180, kernel_size=3):
        super().__init__()
        self.reduce_conv = nn.Conv2d(in_channels, mid_channels, kernel_size=1)
        self.offset_conv = nn.Conv2d(mid_channels, 2 * mid_channels, kernel_size=1)

        self.dcn_w = DeformConv2d(mid_channels, mid_channels, kernel_size=kernel_size, padding=kernel_size // 2)
        self.dcn_h = DeformConv2d(mid_channels, mid_channels, kernel_size=kernel_size, padding=kernel_size // 2)

        self.expand_conv = nn.Conv2d(mid_channels, in_channels, kernel_size=1)
        self.attention_conv = nn.Conv2d(mid_channels, 2, kernel_size=1)

    def forward(self, x):
        x_reduced = self.reduce_conv(x)
        offsets = self.offset_conv(x_reduced)
        offsets_w, offsets_h = offsets[:, :x_reduced.shape[1], :, :], offsets[:, x_reduced.shape[1]:, :, :]
        F_dc_w = self.dcn_w(x_reduced, offsets_w)
        F_dc_h = self.dcn_h(x_reduced, offsets_h)
        F_avg = (F_dc_w + F_dc_h).mean(dim=(2, 3), keepdim=True)
        attn = F.softmax(self.attention_conv(F_avg), dim=1)
        F_dcm = attn[:, 0:1, :, :] * F_dc_h + attn[:, 1:2, :, :] * F_dc_w
        out = self.expand_conv(F_dcm)
        return out


class DWConv3x3(nn.Module):
    def __init__(self, c):
        super().__init__()
        self.op = nn.Conv2d(c, c, 3, padding=1, groups=c, bias=True)
    def forward(self, x): return self.op(x)

def to_luminance(x):
    if x.size(1) >= 3:
        R, G, B = x[:,0:1], x[:,1:2], x[:,2:3]
        y = 0.299*R + 0.587*G + 0.114*B
    else:
        y = x.mean(dim=1, keepdim=True)
    return y

class GaussianBlur(nn.Module):
    def __init__(self, k=7, sigma=2.0):
        super().__init__()
        ax = torch.arange(k) - (k-1)/2
        xx, yy = torch.meshgrid(ax, ax, indexing='ij')
        kernel = torch.exp(-(xx**2 + yy**2) / (2*sigma**2))
        kernel = kernel / kernel.sum()
        self.register_buffer('w', kernel[None, None, :, :])  # [1,1,k,k]
        self.k = k
    def forward(self, x):
        return F.conv2d(x, self.w, padding=self.k//2)

class PG_SFE(nn.Module):
    def __init__(self, in_ch=3, out_ch=64, p_in=1, p_embed=16, use_ext_prior=False):
        super().__init__()
        self.use_ext_prior = use_ext_prior
        self.blur = GaussianBlur(k=7, sigma=2.0)

        self.sfe1 = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, 3, padding=1), nn.GELU(),
            nn.Conv2d(out_ch, out_ch, 1), nn.GELU(),
            DWConv3x3(out_ch), nn.GELU(),
            nn.Conv2d(out_ch, out_ch, 1)  # no act (pre-modulation)
        )
        self.res_in = nn.Conv2d(in_ch, out_ch, 1) if in_ch != out_ch else nn.Identity()

        self.p_enc = nn.Sequential(
            nn.Conv2d(p_in if use_ext_prior else 1, p_embed, 1), nn.GELU(),
            DWConv3x3(p_embed), nn.GELU()
        )

        gate_in_ch = out_ch + p_embed
        self.c_gate = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(gate_in_ch, out_ch, 1), nn.Sigmoid()
        )
        self.s_gate = nn.Sequential(
            nn.Conv2d(gate_in_ch, 1, 3, padding=1), nn.Sigmoid()
        )

        self.post = nn.Sequential(
            DWConv3x3(out_ch), nn.GELU(),
            nn.Conv2d(out_ch, out_ch, 1)
        )

    def _make_prior(self, x, prior=None):
        if self.use_ext_prior:
            P = prior
        else:
            y = to_luminance(x)
            P = self.blur(y)
        return P

    def forward(self, x, prior=None):
        base = self.sfe1(x)
        skip = self.res_in(x)
        feat = base + skip

        P = self._make_prior(x, prior)
        P_emb = self.p_enc(P)

        g_in = torch.cat([feat, P_emb], dim=1)
        g_c = self.c_gate(g_in)
        g_s = self.s_gate(g_in)

        mod = (feat * g_c) * g_s

        out = self.post(mod) + feat                 
        return out

def to_3d(x):
    return rearrange(x, 'b c h w -> b (h w) c')

def to_4d(x,h,w):
    return rearrange(x, 'b (h w) c -> b c h w',h=h,w=w)


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
    def __init__(self, dim, LayerNorm_type):
        super(LayerNorm, self).__init__()
        if LayerNorm_type =='BiasFree':
            self.body = BiasFree_LayerNorm(dim)
        else:
            self.body = WithBias_LayerNorm(dim)

    def forward(self, x):
        h, w = x.shape[-2:]
        return to_4d(self.body(to_3d(x)), h, w)


def downshuffle(var, r):
    b, c, h, w = var.size()
    out_channel = c * (r ** 2)
    out_h = h // r
    out_w = w // r
    return var.contiguous().view(b, c, out_h, r, out_w, r) \
        .permute(0, 1, 3, 5, 2, 4).contiguous().view(b, out_channel, out_h, out_w).contiguous()


class Downsample(nn.Module):
    def __init__(self, n_feat):
        super(Downsample, self).__init__()
        self.body = nn.Sequential(nn.Conv2d(n_feat, n_feat // 2, kernel_size=3, stride=1, padding=1, bias=False),)
                                  #nn.PixelUnshuffle(2))

    def forward(self, x):
        return downshuffle(self.body(x),2)


class Attention(nn.Module):
    def __init__(self, dim, num_heads, bias):
        super(Attention, self).__init__()
        self.num_heads = num_heads
        self.temperature = nn.Parameter(torch.ones(num_heads, 1, 1))
        self.pool = nn.AvgPool2d(kernel_size=8, stride=8)

        self.qkv = nn.Conv2d(dim, dim * 3, kernel_size=1, bias=bias)
        self.qkv_dwconv = nn.Conv2d(dim * 3, dim * 3, kernel_size=3, stride=1, padding=1, groups=dim * 3, bias=bias)
        self.project_out = nn.Conv2d(dim, dim, kernel_size=1, bias=bias)
        self.alpha = nn.Parameter(torch.tensor(0.5))

    def forward(self, x):
        b, c, h, w = x.shape

        qkv = self.qkv_dwconv(self.qkv(x))
        qkv_down = self.pool(qkv)

        q, k, v = qkv.chunk(3, dim=1)
        q_ds, k_ds, v_ds = qkv_down.chunk(3, dim=1)

        q = rearrange(q, 'b (head c) h w -> b head c (h w)', head=self.num_heads)
        k = rearrange(k, 'b (head c) h w -> b head c (h w)', head=self.num_heads)
        v = rearrange(v, 'b (head c) h w -> b head c (h w)', head=self.num_heads)

        q = torch.nn.functional.normalize(q, dim=-1)
        k = torch.nn.functional.normalize(k, dim=-1)

        # transposed self-attention with attention map of shape (C×C)
        attn1 = (q @ k.transpose(-2, -1)) * self.temperature
        attn1 = attn1.softmax(dim=-1)

        out1 = (attn1 @ v)

        out1 = rearrange(out1, 'b head c (h w) -> b (head c) h w', head=self.num_heads, h=h, w=w)

        q_ds = rearrange(q_ds, 'b (head c) h w -> b head c (h w)', head=self.num_heads)
        k_ds = rearrange(k_ds, 'b (head c) h w -> b head c (h w)', head=self.num_heads)

        q_ds = torch.nn.functional.normalize(q_ds, dim=-1)
        k_ds = torch.nn.functional.normalize(k_ds, dim=-1)
        attn2 = (q_ds @ k_ds.transpose(-2, -1)) * self.temperature
        attn2 = attn2.softmax(dim=-1)
        out2 = (attn2 @ v)
        out2 = rearrange(out2, 'b head c (h w) -> b (head c) h w', head=self.num_heads, h=h, w=w)

        out = self.alpha * out1 + (1 - self.alpha) * out2

        out = self.project_out(out)

        return out


class GDFN(nn.Module):
    def __init__(self, dim, ffn_expansion_factor, bias):
        super(GDFN, self).__init__()

        hidden_features = int(dim * ffn_expansion_factor)

        self.project_in = nn.Conv2d(dim, hidden_features * 2, kernel_size=1, bias=bias)

        self.dwconv = nn.Conv2d(hidden_features * 2, hidden_features * 2, kernel_size=3, stride=1, padding=1,
                                groups=hidden_features * 2, bias=bias)

        self.project_out = nn.Conv2d(hidden_features, dim, kernel_size=1, bias=bias)

    def forward(self, x):
        x = self.project_in(x)
        x1, x2 = self.dwconv(x).chunk(2, dim=1)
        x = F.gelu(x1) * x2
        x = self.project_out(x)
        return x


class TransformerBlock(nn.Module):
    def __init__(self, dim, num_heads, ffn_expansion_factor, bias, LayerNorm_type):
        super(TransformerBlock, self).__init__()
        self.norm1 = LayerNorm(dim, LayerNorm_type)
        self.attn = Attention(dim, num_heads, bias)
        self.norm2 = LayerNorm(dim, LayerNorm_type)
        self.ffn = GDFN(dim, ffn_expansion_factor, bias)

    def forward(self, x):
        x = x + self.attn(self.norm1(x))
        x = x + self.ffn(self.norm2(x))

        return x


class FAM(nn.Module):
    def __init__(self, channel):
        super(FAM, self).__init__()
        self.merge = nn.Conv2d(channel, channel, kernel_size=3, stride=1, padding=1, groups=channel)

    def forward(self, x1, x2):
        x = x1 * x2
        out = x1 + self.merge(x)
        return out

class LGFF(nn.Module):
    def __init__(self, in_dim, out_dim, ffn_expansion_factor, bias):
        super(LGFF, self).__init__()
        self.project_in = nn.Sequential(nn.Conv2d(in_dim, in_dim, kernel_size=3, padding=1, groups=in_dim),
                                        nn.Conv2d(in_dim, out_dim, kernel_size=1))
        self.norm = LayerNorm(out_dim, LayerNorm_type = 'WithBias')
        self.ffn = GDFN(out_dim, ffn_expansion_factor, bias)

    def forward(self, x):
        x = self.project_in(x)
        x = x + self.ffn(self.norm(x))
        return x


class RawFormer(nn.Module):
    def __init__(self, out_channels=3, dim=32, num_heads=[2, 4, 8], num_blocks=[2, 4, 4, 4], ffn_expansion_factor=2.66,
                 bias=True, LayerNorm_type='WithBias'):
        super(RawFormer, self).__init__()
        self.pr = nn.Conv2d(3, dim, 3, 1, 1)
        self.proj = PG_SFE(3, dim)
        self.proj1 = PG_SFE(3, dim * 2)
        self.proj2 = PG_SFE(3, dim * 4)
        self.conv_tran1 = nn.Sequential(*[TransformerBlock(dim, num_heads[0], ffn_expansion_factor, bias=bias, LayerNorm_type=LayerNorm_type)
             for i in
             range(num_blocks[0])])
        self.down1 = Downsample(dim)
        self.conv_tran2 = nn.Sequential(*[TransformerBlock(dim * 2, num_heads[1], ffn_expansion_factor, bias=bias,
                                                          LayerNorm_type=LayerNorm_type) for i in
                                         range(num_blocks[1])])
        self.down2 = Downsample(dim * 2)
        self.conv_tran3 = nn.Sequential(*[TransformerBlock(dim * 4, num_heads[2], ffn_expansion_factor, bias=bias,
                                                          LayerNorm_type=LayerNorm_type) for i in
                                         range(num_blocks[2])])

        self.conv_tran4 = nn.Sequential(*[
            TransformerBlock(dim * 4, num_heads[2], ffn_expansion_factor, bias=bias, LayerNorm_type=LayerNorm_type) for i in
            range(num_blocks[2])])
        self.channel_reduce1 = nn.Conv2d(dim * 8, dim * 4, 1, 1)

        self.up1 = nn.ConvTranspose2d(dim * 4, dim * 2, 2, stride=2)
        self.channel_reduce1 = nn.Conv2d(dim * 4, dim * 2, 1, 1)
        self.conv_tran5 = nn.Sequential(*[
            TransformerBlock(dim * 2, num_heads[1], ffn_expansion_factor, bias=bias, LayerNorm_type=LayerNorm_type) for i in
            range(num_blocks[1])])
        self.up2 = nn.ConvTranspose2d(dim * 2, dim * 1, 2, stride=2)
        self.channel_reduce2 = nn.Conv2d(dim * 2, dim * 1, 1, 1)
        self.conv_tran6 = nn.Sequential(*[
            TransformerBlock(dim * 1, num_heads[0], ffn_expansion_factor, bias=bias, LayerNorm_type=LayerNorm_type) for i in
            range(num_blocks[0])])

        self.conv_tran7 = nn.Sequential(*[
            TransformerBlock(dim * 1, num_heads[0], ffn_expansion_factor, bias=bias, LayerNorm_type=LayerNorm_type) for i in
            range(num_blocks[3])])
        self.conv_out = nn.Conv2d(dim * 1, out_channels, kernel_size=3, stride=1, padding=1)

        self.FAM = FAM(dim)
        self.FAM1 = FAM(dim * 2)
        self.FAM2 = FAM(dim * 4)

        self.AFF1 = LGFF(dim * 3, dim * 2, 1, False)
        self.AFF2 = LGFF(dim * 3, dim * 1, 1, False)
        self.DCM = DCMBlock(dim * 4)

    def forward(self, x):
        x_2 = F.interpolate(x, scale_factor=0.5)
        x_4 = F.interpolate(x_2, scale_factor=0.5)

        proj = self.pr(x)
        mean_c = x.mean(dim=1).unsqueeze(1)
        fea = self.proj(x, mean_c)
        fusion = self.FAM(proj, fea)
        conv_tran1 = self.conv_tran1(fusion)

        pool1 = self.down1(conv_tran1)
        mean_c1 = x_2.mean(dim=1).unsqueeze(1)
        fea1 = self.proj1(x_2, mean_c1)
        fusion1 = self.FAM1(pool1, fea1)
        conv_tran2 = self.conv_tran2(fusion1)

        pool2 = self.down2(conv_tran2)
        mean_c2 = x_4.mean(dim=1).unsqueeze(1)
        fea2 = self.proj2(x_4, mean_c2)
        fusion2 = self.FAM2(pool2, fea2)
        conv_tran3 = self.conv_tran3(fusion2)

        conv_tran3 = self.DCM(conv_tran3)

        conv_tran1_2 = F.interpolate(conv_tran1, scale_factor=0.5)
        conv_tran2_1 = F.interpolate(conv_tran2, scale_factor=2)

        res2 = self.AFF2(torch.cat((conv_tran1, conv_tran2_1), dim=1))
        res1 = self.AFF1(torch.cat((conv_tran1_2, conv_tran2), dim=1))

        conv_tran4 = self.conv_tran4(conv_tran3)

        up1 = self.up1(conv_tran4)
        concat1 = torch.cat([up1, res1], 1)
        concat1 = self.channel_reduce1(concat1)
        conv_tran5 = self.conv_tran5(concat1)

        up2 = self.up2(conv_tran5)
        concat2 = torch.cat([up2, res2], 1)
        concat2 = self.channel_reduce2(concat2)
        conv_tran6 = self.conv_tran6(concat2)

        conv_tran7 = self.conv_tran7(conv_tran6)
        conv_out = self.conv_out(conv_tran7)

        return conv_out + x