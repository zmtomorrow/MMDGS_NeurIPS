import torch.nn as nn
import torch.nn.functional as F
import torch


class Identity(nn.Module):
    def __init__(self):
        super().__init__()
    def forward(self,x):
        return x

class FeedFowardNet(nn.Module):
    def __init__(self,  input_dim=784, output_dim=1, h_dim=400, h_layer_num=1,act='tanh', if_bn= False):
        super().__init__()
        self.idenity=Identity()
        self.input_dim=input_dim
        self.h_layer_num=h_layer_num
        self.fc_list=nn.ModuleList([])
        self.bn_list=nn.ModuleList([])
        if act=='tanh':
            self.act=torch.tanh
        elif act=='relu':
            self.act=torch.relu
        elif act=='swish':
            self.act=lambda x: x*torch.sigmoid(x)
        elif act=='leakyrelu':
            self.act=F.leaky_relu
        
        for i in range(0,h_layer_num+1):
            if i==0:
                self.fc_list.append(nn.Linear(input_dim, h_dim))
            else:
                self.fc_list.append(nn.Linear(h_dim, h_dim))
            if if_bn:
                self.bn_list.append(nn.BatchNorm1d(h_dim))
            else:
                self.bn_list.append(self.idenity)
        self.fc_out = nn.Linear(h_dim, output_dim)

    def forward(self, x):
        x=x.view(-1,self.input_dim)
        for i in range(0,self.h_layer_num+1):
            x=self.act(self.bn_list[i](self.fc_list[i](x)))
        return self.fc_out(x)


class EnergyNet(nn.Module):
    def __init__(self,  input_dim=784, sigma=0.1, h_dim=400, h_layer_num=1,act='swish', if_bn= False):
        super().__init__()
        self.idenity=Identity()
        self.input_dim=input_dim
        self.h_layer_num=h_layer_num
        self.sigma=sigma
        self.fc_list=nn.ModuleList([])
        self.bn_list=nn.ModuleList([])
        if act=='tanh':
            self.act=torch.tanh
        elif act=='relu':
            self.act=torch.relu
        elif act=='swish':
            self.act=lambda x: x*torch.sigmoid(x)
        elif act=='leakyrelu':
            self.act=F.leaky_relu
        
        for i in range(0,h_layer_num+1):
            if i==0:
                self.fc_list.append(nn.Linear(input_dim, h_dim))
            else:
                self.fc_list.append(nn.Linear(h_dim, h_dim))
            if if_bn:
                self.bn_list.append(nn.BatchNorm1d(h_dim))
            else:
                self.bn_list.append(self.idenity)
        self.fc_out = nn.Linear(h_dim,input_dim)

    def forward(self, x):
        x_mid=x.view(-1,self.input_dim)
        for i in range(0,self.h_layer_num+1):
            x_mid=self.act(self.bn_list[i](self.fc_list[i](x_mid)))
        return torch.sum((self.fc_out(x_mid)-x)**2,-1)/(2*(self.sigma**2))






def nonlinearity(x):
    # swish
    return x*torch.sigmoid(x)


def Normalize(in_channels):
    return torch.nn.GroupNorm(num_groups=32, num_channels=in_channels, eps=1e-6, affine=True)


class Upsample(nn.Module):
    def __init__(self, in_channels, with_conv):
        super().__init__()
        self.with_conv = with_conv
        if self.with_conv:
            self.conv = torch.nn.Conv2d(in_channels,
                                        in_channels,
                                        kernel_size=3,
                                        stride=1,
                                        padding=1)

    def forward(self, x):
        x = torch.nn.functional.interpolate(
            x, scale_factor=2.0, mode="nearest")
        # print('up_pre:',x.size())
        if x.shape[-1] == x.shape[-2] == 6:
            # upsampling layer transform [3x3] to [6x6]. Manually paddding it to make [7x7]
            x = F.pad(x, (1, 0, 1, 0))
        if self.with_conv:
            x = self.conv(x)
        # print('up:',x.size())
        return x


class Downsample(nn.Module):
    def __init__(self, in_channels, with_conv):
        super().__init__()
        self.with_conv = with_conv
        if self.with_conv:
            # no asymmetric padding in torch conv, must do it ourselves
            self.conv = torch.nn.Conv2d(in_channels,
                                        in_channels,
                                        kernel_size=3,
                                        stride=2,
                                        padding=0)

    def forward(self, x):
        if self.with_conv:
            pad = (0, 1, 0, 1)
            x = torch.nn.functional.pad(x, pad, mode="constant", value=0)
            x = self.conv(x)
        else:
            x = torch.nn.functional.avg_pool2d(x, kernel_size=2, stride=2)
        return x


class ResnetBlock(nn.Module):
    def __init__(self, *, in_channels, out_channels=None, conv_shortcut=False,
                 dropout):
        super().__init__()
        self.in_channels = in_channels
        out_channels = in_channels if out_channels is None else out_channels
        self.out_channels = out_channels
        self.use_conv_shortcut = conv_shortcut

        self.norm1 = Normalize(in_channels)
        self.conv1 = torch.nn.Conv2d(in_channels,
                                     out_channels,
                                     kernel_size=3,
                                     stride=1,
                                     padding=1)
        self.norm2 = Normalize(out_channels)
        self.dropout = torch.nn.Dropout(dropout)
        self.conv2 = torch.nn.Conv2d(out_channels,
                                     out_channels,
                                     kernel_size=3,
                                     stride=1,
                                     padding=1)
        if self.in_channels != self.out_channels:
            if self.use_conv_shortcut:
                self.conv_shortcut = torch.nn.Conv2d(in_channels,
                                                     out_channels,
                                                     kernel_size=3,
                                                     stride=1,
                                                     padding=1)
            else:
                self.nin_shortcut = torch.nn.Conv2d(in_channels,
                                                    out_channels,
                                                    kernel_size=1,
                                                    stride=1,
                                                    padding=0)

    def forward(self, x):
        h = x
        h = self.norm1(h)
        h = nonlinearity(h)
        h = self.conv1(h)
        h = self.norm2(h)
        h = nonlinearity(h)
        h = self.dropout(h)
        h = self.conv2(h)

        if self.in_channels != self.out_channels:
            if self.use_conv_shortcut:
                x = self.conv_shortcut(x)
            else:
                x = self.nin_shortcut(x)

        return x+h


class AttnBlock(nn.Module):
    def __init__(self, in_channels):
        super().__init__()
        self.in_channels = in_channels

        self.norm = Normalize(in_channels)
        self.q = torch.nn.Conv2d(in_channels,
                                 in_channels,
                                 kernel_size=1,
                                 stride=1,
                                 padding=0)
        self.k = torch.nn.Conv2d(in_channels,
                                 in_channels,
                                 kernel_size=1,
                                 stride=1,
                                 padding=0)
        self.v = torch.nn.Conv2d(in_channels,
                                 in_channels,
                                 kernel_size=1,
                                 stride=1,
                                 padding=0)
        self.proj_out = torch.nn.Conv2d(in_channels,
                                        in_channels,
                                        kernel_size=1,
                                        stride=1,
                                        padding=0)

    def forward(self, x):
        h_ = x
        h_ = self.norm(h_)
        q = self.q(h_)
        k = self.k(h_)
        v = self.v(h_)

        # compute attention
        b, c, h, w = q.shape
        q = q.reshape(b, c, h*w)
        q = q.permute(0, 2, 1)   # b,hw,c
        k = k.reshape(b, c, h*w)  # b,c,hw
        w_ = torch.bmm(q, k)     # b,hw,hw    w[b,i,j]=sum_c q[b,i,c]k[b,c,j]
        w_ = w_ * (int(c)**(-0.5))
        w_ = torch.nn.functional.softmax(w_, dim=2)

        # attend to values
        v = v.reshape(b, c, h*w)
        w_ = w_.permute(0, 2, 1)   # b,hw,hw (first hw of k, second of q)
        # b, c,hw (hw of q) h_[b,c,j] = sum_i v[b,c,i] w_[b,i,j]
        h_ = torch.bmm(v, w_)
        h_ = h_.reshape(b, c, h, w)

        h_ = self.proj_out(h_)

        return x+h_


class Unet(nn.Module):
    def __init__(self, in_channels, out_channels, resolution):
        super().__init__()
        # self.config = config
        ch, out_ch, ch_mult = 128, out_channels, tuple([1, 2, 2, 2])
        num_res_blocks = 2
        attn_resolutions = [int(resolution/2), ]
        dropout = 0.1
        in_channels = in_channels
        resolution = resolution
        resamp_with_conv = True
        
        # if config.model.type == 'bayesian':
        #     self.logvar = nn.Parameter(torch.zeros(num_timesteps))
        
        self.ch = ch
        self.num_resolutions = len(ch_mult)
        self.num_res_blocks = num_res_blocks
        self.resolution = resolution
        self.in_channels = in_channels


        # downsampling
        self.conv_in = torch.nn.Conv2d(in_channels,
                                       self.ch,
                                       kernel_size=3,
                                       stride=1,
                                       padding=1)

        curr_res = resolution
        in_ch_mult = (1,)+ch_mult
        self.down = nn.ModuleList()
        block_in = None
        for i_level in range(self.num_resolutions):
            block = nn.ModuleList()
            attn = nn.ModuleList()
            block_in = ch*in_ch_mult[i_level]
            block_out = ch*ch_mult[i_level]
            for i_block in range(self.num_res_blocks):
                block.append(ResnetBlock(in_channels=block_in,
                                         out_channels=block_out,
                                         dropout=dropout))
                block_in = block_out
                if curr_res in attn_resolutions:
                    attn.append(AttnBlock(block_in))
            down = nn.Module()
            down.block = block
            down.attn = attn
            if i_level != self.num_resolutions-1:
                down.downsample = Downsample(block_in, resamp_with_conv)
                curr_res = curr_res // 2
            self.down.append(down)

        # middle
        self.mid = nn.Module()
        self.mid.block_1 = ResnetBlock(in_channels=block_in,
                                       out_channels=block_in,
                                       dropout=dropout)
        self.mid.attn_1 = AttnBlock(block_in)
        self.mid.block_2 = ResnetBlock(in_channels=block_in,
                                       out_channels=block_in,
                                       dropout=dropout)

        # upsampling
        self.up = nn.ModuleList()
        for i_level in reversed(range(self.num_resolutions)):
            block = nn.ModuleList()
            attn = nn.ModuleList()
            block_out = ch*ch_mult[i_level]
            skip_in = ch*ch_mult[i_level]
            for i_block in range(self.num_res_blocks+1):
                if i_block == self.num_res_blocks:
                    skip_in = ch*in_ch_mult[i_level]
                block.append(ResnetBlock(in_channels=block_in+skip_in,
                                         out_channels=block_out,
                                         dropout=dropout))
                block_in = block_out
                if curr_res in attn_resolutions:
                    attn.append(AttnBlock(block_in))
            up = nn.Module()
            up.block = block
            up.attn = attn
            if i_level != 0:
                up.upsample = Upsample(block_in, resamp_with_conv)
                curr_res = curr_res * 2
            self.up.insert(0, up)  # prepend to get consistent order

        # end
        self.norm_out = Normalize(block_in)
        self.conv_out = torch.nn.Conv2d(block_in,
                                        out_ch,
                                        kernel_size=3,
                                        stride=1,
                                        padding=1)

    def forward(self, x):
        assert x.shape[2] == x.shape[3] == self.resolution



        # downsampling
        hs = [self.conv_in(x)]
        for i_level in range(self.num_resolutions):
            for i_block in range(self.num_res_blocks):
                h = self.down[i_level].block[i_block](hs[-1])
                # print('here h:',h.size())
                if len(self.down[i_level].attn) > 0:
                    h = self.down[i_level].attn[i_block](h)

                hs.append(h)
            if i_level != self.num_resolutions-1:
                hs.append(self.down[i_level].downsample(hs[-1]))

        # middle
        h = hs[-1]
        h = self.mid.block_1(h)
        h = self.mid.attn_1(h)
        h = self.mid.block_2(h)

        # upsampling
        for i_level in reversed(range(self.num_resolutions)):
            for i_block in range(self.num_res_blocks+1):
                h_p=hs.pop()
                # print('h:',h.size())
                # print('h_p',h_p.size())
                h = self.up[i_level].block[i_block](
                    torch.cat([h, h_p], dim=1))
                if len(self.up[i_level].attn) > 0:
                    h = self.up[i_level].attn[i_block](h)
            if i_level != 0:
                h = self.up[i_level].upsample(h)

        # end
        h = self.norm_out(h)
        h = nonlinearity(h)
        h = self.conv_out(h)
        return h
