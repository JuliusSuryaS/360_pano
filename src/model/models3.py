from util.base import *
import model.ops as ops

class G5(nn.Module):
    def __init__(self):
        super(G5, self).__init__()
        # In 16 x 32
        self.block_in = ops.conv_norm_leak(3, 64, 3, 1, 1) # out 16 x 32
        self.block1 = ops.conv_norm_leak(64, 256, 3, 1, 1) # out 16 x 32
        self.block2 = ops.conv_norm_leak(256, 256, 4, 2, 1) # out 8 x 16
        self.block3 = ops.conv_norm_leak(256, 512, 3, 1, 1) # out 8 x 16
        self.block4 = ops.conv_norm_leak(512, 512, 4, 2, 1) # out 4 x 8
        self.block5 = ops.conv_norm_leak(512, 1024, 3, 1, 1) # out 4 x 8
        self.block6 = ops.conv_norm_leak(1024, 1024, 3, 1, 1) # out 4 x 8

        # --

        self.dblock1 = ops.convT_norm_leak(1024, 512, 3, 1, 1) # out 8 x 4
        self.dblock2 = ops.convT_norm_leak(512 + 1024, 512, 3, 1, 1) # out 8 x 4
        self.dblock3 = ops.convT_norm_leak(512 + 512,  512, 4, 2, 1) # out 16 x 8
        self.dblock4 = ops.convT_norm_leak(512 + 512, 256, 3, 1, 1) # out 16 x 8
        self.dblock5 = ops.convT_norm_leak(256 + 256, 256, 4, 2, 1) # out 32 x 16
        self.dblock6 = ops.convT_norm_leak(256 + 256, 256, 3, 1, 1) # out 32 x 16
        self.dblock_out =  ops.conv(256, 3, 1, 1, 0) # out 32 x 16

    def forward(self, x):
        enc0 = self.block_in(x)
        enc1 = self.block1(enc0)
        enc2 = self.block2(enc1)
        enc3 = self.block3(enc2)
        enc4 = self.block4(enc3)
        enc5 = self.block5(enc4)
        enc6 = self.block6(enc5)
        dec1 = self.dblock1(enc6)
        dec2 = self.dblock2(ops.dstack(dec1,enc5))
        dec3 = self.dblock3(ops.dstack(dec2,enc4))
        dec4 = self.dblock4(ops.dstack(dec3,enc3))
        dec5 = self.dblock5(ops.dstack(dec4,enc2))
        dec6 = self.dblock6(ops.dstack(dec5,enc1))
        out = self.dblock_out(dec6)
        return out


class D5(nn.Module):
    def __init__(self):
        super(D5, self).__init__()
        self.discriminator = nn.Sequential(
            ops.conv_leak(3, 64, 3, 1, 1), # out 16 x 32
            ops.conv_norm_leak(64, 128, 3, 1, 1), # out 16 x 32
            ops.conv_norm_leak(128, 256, 4, 2, 1), # out 8 x 16
            ops.conv_norm_leak(256, 256, 4, 2, 1), # out 4 x 8
            ops.conv_norm_leak(256, 512, 4, 2, 1), # out 1 x 4
            ops.conv(512, 1, (1,4), 1, 1), # out 1 x 4
        )

    def forward(self, x):
        out = self.discriminator(x)
        return out

class G4(nn.Module):
    def __init__(self):
        super(G4, self).__init__()
        # In 32 x 64
        self.block4_in = ops.conv_leak(3, 64, 3, 1, 1) # out 32 x 64
        self.block4_1 = ops.conv_norm_leak(64, 256, 3, 1, 1) # out 32 x 64
        self.block4_2 = ops.conv_norm_leak(256, 256, 4, 2, 1) # out 16 x 32
        # In 16 x 32
        self.block_in = ops.conv_leak(3, 64, 3, 1, 1) # out 16 x 32
        self.block1 = ops.conv_norm_leak(64, 256, 3, 1, 1) # out 16 x 32
        self.block2 = ops.conv_norm_leak(256, 256, 4, 2, 1) # out 8 x 16
        self.block3 = ops.conv_norm_leak(256, 512, 3, 1, 1) # out 8 x 16
        self.block4 = ops.conv_norm_leak(512, 512, 4, 2, 1) # out 4 x 8
        self.block5 = ops.conv_norm_leak(512, 1024, 3, 1, 1) # out 4 x 8
        self.block6 = ops.conv_norm_leak(1024, 1024, 3, 1, 1) # out 4 x 8

        # --
        # Out 16 x 32
        self.dblock1 = ops.convT_norm_leak(1024, 512, 3, 1, 1) # out 4 x 8
        self.dblock2 = ops.convT_norm_leak(512 + 1024, 512, 3, 1, 1) # out 4 x 8
        self.dblock3 = ops.convT_norm_leak(512 + 512,  512, 4, 2, 1) # out 8 x 16
        self.dblock4 = ops.convT_norm_leak(512 + 512, 256, 3, 1, 1) # out 8 x 16
        self.dblock5 = ops.convT_norm_leak(256 + 256, 256, 4, 2, 1) # out 16 x 32
        self.dblock6 = ops.convT_norm_leak(256 + 256, 256, 3, 1, 1) # out 16 x 32
        self.dblock_out =  ops.conv(256, 3, 1, 1, 0) # out 16 x 32
        # Out 32 x 64
        self.dblock4_0 = ops.convT_norm_leak(256, 256, 3, 1, 1) # 16 x 32
        self.dblock4_1 = ops.convT_norm_leak(256 + 256, 256, 4, 2, 1)  # 32 x 64
        self.dblock4_2 = ops.convT_norm_leak(256 + 256, 256, 3, 1, 1)  # 32 x 64
        self.dblock4_out = ops.conv(256, 3, 1, 1, 0) # out 32 x 64

    def forward(self, x):
        x4 = ops.downsample(x, 16)
        x5 = ops.downsample(x, 32)
        # G4
        enc4_0 = self.block4_in(x4) # 32 x 64
        enc4_1 = self.block4_1(enc4_0) # 32 x 64
        enc4_2 = self.block4_2(enc4_1) # 16 x 32
        # G5
        enc0 = self.block_in(x5)
        enc1 = self.block1(enc0)
        enc2 = self.block2(enc1)
        enc3 = self.block3(enc2)
        enc4 = self.block4(enc3)
        enc5 = self.block5(enc4)
        enc6 = self.block6(enc5)
        dec1 = self.dblock1(enc6)
        dec2 = self.dblock2(ops.dstack(dec1,enc5))
        dec3 = self.dblock3(ops.dstack(dec2,enc4))
        dec4 = self.dblock4(ops.dstack(dec3,enc3))
        dec5 = self.dblock5(ops.dstack(dec4,enc2))
        dec6 = self.dblock6(ops.dstack(dec5,enc1))
        out5 = self.dblock_out(dec6)
        # G4
        dec4_0 = self.dblock4_0(enc4_2 + dec6)
        dec4_1 = self.dblock4_1(ops.dstack(dec4_0, enc4_2))
        dec4_2 = self.dblock4_2(ops.dstack(dec4_1, enc4_1))
        out4 = self.dblock4_out(dec4_2)
        out = F.tanh(F.interpolate(out5, scale_factor=2, mode='bilinear') + out4)
        return out, out5

class D4(nn.Module):
    def __init__(self):
        super(D4, self).__init__()
        self.discriminator = nn.Sequential(
            ops.conv_leak(3, 64, 3, 1, 1), # out 32 x 64
            ops.conv_norm_leak(64, 128, 4, 2, 1), # out 16 x 32
            ops.conv_norm_leak(128, 256, 4, 2, 1), # out 8 x 16
            ops.conv_norm_leak(256, 256, 4, 2, 1), # out 4 x 8
            ops.conv_norm_leak(256, 512, 4, 2, 1), # out 1 x 4
            ops.conv(512, 1, (1,4), 1, 1), # out 1 x 4
        )

    def forward(self, x):
        out = self.discriminator(x)
        return out

class G3(nn.Module):
    def __init__(self):
        super(G3, self).__init__()
        # In 64 x 128
        self.block3_in = ops.conv_leak(3, 64, 3, 1, 1) # out 64 x 128
        self.block3_1 = ops.conv_norm_leak(64, 256, 3, 1, 1) # out 64 x 128
        self.block3_2 = ops.conv_norm_leak(256, 256, 4, 2, 1) # out 32 x 64
        # In 32 x 64
        self.block4_in = ops.conv_leak(3, 64, 3, 1, 1) # out 32 x 64
        self.block4_1 = ops.conv_norm_leak(64, 256, 3, 1, 1) # out 32 x 64
        self.block4_2 = ops.conv_norm_leak(256, 256, 4, 2, 1) # out 16 x 32
        # In 16 x 32
        self.block_in = ops.conv_leak(3, 64, 3, 1, 1) # out 16 x 32
        self.block1 = ops.conv_norm_leak(64, 256, 3, 1, 1) # out 16 x 32
        self.block2 = ops.conv_norm_leak(256, 256, 4, 2, 1) # out 8 x 16
        self.block3 = ops.conv_norm_leak(256, 512, 3, 1, 1) # out 8 x 16
        self.block4 = ops.conv_norm_leak(512, 512, 4, 2, 1) # out 4 x 8
        self.block5 = ops.conv_norm_leak(512, 1024, 3, 1, 1) # out 4 x 8
        self.block6 = ops.conv_norm_leak(1024, 1024, 3, 1, 1) # out 4 x 8

        # --
        # Out 16 x 32
        self.dblock1 = ops.convT_norm_leak(1024, 512, 3, 1, 1) # out 4 x 8
        self.dblock2 = ops.convT_norm_leak(512 + 1024, 512, 3, 1, 1) # out 4 x 8
        self.dblock3 = ops.convT_norm_leak(512 + 512,  512, 4, 2, 1) # out 8 x 16
        self.dblock4 = ops.convT_norm_leak(512 + 512, 256, 3, 1, 1) # out 8 x 16
        self.dblock5 = ops.convT_norm_leak(256 + 256, 256, 4, 2, 1) # out 16 x 32
        self.dblock6 = ops.convT_norm_leak(256 + 256, 256, 3, 1, 1) # out 16 x 32
        self.dblock_out =  ops.conv(256, 3, 1, 1, 0) # out 16 x 32
        # Out 32 x 64
        self.dblock4_0 = ops.convT_norm_leak(256, 256, 3, 1, 1) # 16 x 32
        self.dblock4_1 = ops.convT_norm_leak(256 + 256, 256, 4, 2, 1)  # 32 x 64
        self.dblock4_2 = ops.convT_norm_leak(256 + 256, 256, 3, 1, 1)  # 32 x 64
        self.dblock4_out = ops.conv(256, 3, 1, 1, 0) # out 32 x 64
        # Out 64 x 128
        self.dblock3_0 = ops.convT_norm_leak(256, 256, 3, 1, 1) # out 32 x 64
        self.dblock3_1 = ops.convT_norm_leak(256 + 256, 256, 4, 2, 1) # out 64 x 128
        self.dblock3_2 = ops.convT_norm_leak(256 + 256, 256, 3, 1, 1) # out 64 x 128
        self.dblock3_out = ops.conv(256, 3, 1, 1, 0) # out 64 x 128

    def forward(self, x):
        x3 = ops.downsample(x, 8)
        x4 = ops.downsample(x, 16)
        x5 = ops.downsample(x, 32)
        # G3
        enc3_0 = self.block3_in(x3) # 64 x 128
        enc3_1 = self.block3_1(enc3_0) # 64 x 128
        enc3_2 = self.block3_2(enc3_1) # 32 x 64
        # G4
        enc4_0 = self.block4_in(x4) # 32 x 64
        enc4_1 = self.block4_1(enc4_0) # 32 x 64
        enc4_2 = self.block4_2(enc4_1) # 16 x 32
        # G5
        enc0 = self.block_in(x5)
        enc1 = self.block1(enc0)
        enc2 = self.block2(enc1)
        enc3 = self.block3(enc2)
        enc4 = self.block4(enc3)
        enc5 = self.block5(enc4)
        enc6 = self.block6(enc5)
        dec1 = self.dblock1(enc6)
        dec2 = self.dblock2(ops.dstack(dec1,enc5))
        dec3 = self.dblock3(ops.dstack(dec2,enc4))
        dec4 = self.dblock4(ops.dstack(dec3,enc3))
        dec5 = self.dblock5(ops.dstack(dec4,enc2))
        dec6 = self.dblock6(ops.dstack(dec5,enc1))
        out5 = self.dblock_out(dec6)
        # G4
        dec4_0 = self.dblock4_0(enc4_2 + dec6)
        dec4_1 = self.dblock4_1(ops.dstack(dec4_0, enc4_2))
        dec4_2 = self.dblock4_2(ops.dstack(dec4_1, enc4_1))
        out4 = self.dblock4_out(dec4_2)
        out4 = (F.interpolate(out5, scale_factor=2, mode='bilinear') + out4)
        # G3
        dec3_0 = self.dblock3_0(enc3_2 + dec4_2)
        dec3_1 = self.dblock3_1(ops.dstack(dec3_0, enc3_2))
        dec3_2 = self.dblock3_2(ops.dstack(dec3_1, enc3_1))
        out3 = self.dblock3_out(dec3_2)
        out =  F.tanh(F.interpolate(out4, scale_factor=2, mode='bilinear') + out3)

        return out, out4

class D3(nn.Module):
    def __init__(self):
        super(D3, self).__init__()
        self.discirminator = nn.Sequential(
            ops.conv_leak(3, 64, 3, 1, 1), # out 64 x 128
            ops.conv_norm_leak(64, 128, 4, 2, 1), # out 32 x 64
            ops.conv_norm_leak(128, 256, 4, 2, 1), # out 16 x 32
            ops.conv_norm_leak(256, 256, 4, 2, 1), # out 8 x 16
            ops.conv_norm_leak(256, 512, 4, 2, 1), # out 4 x 8
            ops.conv(512, 1, (4,4), 2, 1), # out 2 x 4
        )

    def forward(self, x):
        out = self.discirminator(x)
        return out


class G2(nn.Module):
    def __init__(self):
        super(G2, self).__init__()
        # In 128 x 256
        self.block2_in = ops.conv_sn_relu(3, 64, 3, 1, 1) # out 128 x 256
        self.block2_1 = ops.conv_sn_relu(64, 256, 3, 1, 1) # out 128 x 256
        self.block2_2 = ops.conv_sn_relu(256, 256, 4, 2, 1) # out 64 x 128
        # In 64 x 128
        self.block3_in = ops.conv_sn_relu(3, 64, 3, 1, 1) # out 64 x 128
        self.block3_1 = ops.conv_sn_relu(64, 256, 3, 1, 1) # out 64 x 128
        self.block3_2 = ops.conv_sn_relu(256, 256, 4, 2, 1) # out 32 x 64
        # In 32 x 64
        self.block4_in = ops.conv_sn_relu(3, 64, 3, 1, 1) # out 32 x 64
        self.block4_1 = ops.conv_sn_relu(64, 256, 3, 1, 1) # out 32 x 64
        self.block4_2 = ops.conv_sn_relu(256, 256, 4, 2, 1) # out 16 x 32
        # In 16 x 32
        self.block_in = ops.conv_sn_relu(3, 64, 3, 1, 1) # out 16 x 32
        self.block1 = ops.conv_sn_relu(64, 256, 3, 1, 1) # out 16 x 32
        self.block2 = ops.conv_sn_relu(256, 256, 4, 2, 1) # out 8 x 16
        self.block3 = ops.conv_sn_relu(256, 512, 3, 1, 1) # out 8 x 16
        self.block4 = ops.conv_sn_relu(512, 512, 4, 2, 1) # out 4 x 8
        self.block5 = ops.conv_sn_relu(512, 1024, 3, 1, 1) # out 4 x 8
        self.block6 = ops.conv_sn_relu(1024, 1024, 3, 1, 1) # out 4 x 8

        # --
        # Out 16 x 32
        self.dblock1 = ops.convT_sn_leak(1024, 512, 3, 1, 1) # out 4 x 8
        self.dblock2 = ops.convT_sn_leak(512 + 1024, 512, 3, 1, 1) # out 4 x 8
        self.dblock3 = ops.convT_sn_leak(512 + 512,  512, 4, 2, 1) # out 8 x 16
        self.dblock4 = ops.convT_sn_leak(512 + 512, 256, 3, 1, 1) # out 8 x 16
        self.dblock5 = ops.convT_sn_leak(256 + 256, 256, 4, 2, 1) # out 16 x 32
        self.dblock6 = ops.convT_sn_leak(256 + 256, 256, 3, 1, 1) # out 16 x 32
        self.dblock_out =  ops.conv(256, 3, 1, 1, 0) # out 16 x 32
        # Out 32 x 64
        self.dblock4_0 = ops.convT_sn_leak(256, 256, 3, 1, 1) # 16 x 32
        self.dblock4_1 = ops.convT_sn_leak(256 + 256, 256, 4, 2, 1)  # 32 x 64
        self.dblock4_2 = ops.convT_sn_leak(256 + 256, 256, 3, 1, 1)  # 32 x 64
        self.dblock4_out = ops.conv(256, 3, 1, 1, 0) # out 32 x 64
        # Out 64 x 128
        self.dblock3_0 = ops.convT_sn_leak(256, 256, 3, 1, 1) # out 32 x 64
        self.dblock3_1 = ops.convT_sn_leak(256 + 256, 256, 4, 2, 1) # out 64 x 128
        self.dblock3_2 = ops.convT_sn_leak(256 + 256, 256, 3, 1, 1) # out 64 x 128
        self.dblock3_out = ops.conv(256, 3, 1, 1, 0) # out 64 x 128
        # Out 128 x 256
        self.dblock2_0 = ops.convT_sn_leak(256, 256, 3, 1, 1) # out 64 x 128
        self.dblock2_1 = ops.convT_sn_leak(256 + 256, 256, 4, 2, 1) # out 128 x 256
        self.dblock2_2 = ops.convT_sn_leak(256 + 256, 256, 3, 1, 1) # out 128 x 256
        self.dblock2_out = ops.conv(256, 3, 3, 1, 1) # out 128 x 256

    def forward(self, x):
        x2 = ops.downsample(x, 4)
        x3 = ops.downsample(x, 8)
        x4 = ops.downsample(x, 16)
        x5 = ops.downsample(x, 32)
        # G2
        enc2_0 = (self.block2_in(x2)) # 64 x 128
        enc2_1 = (self.block2_1(enc2_0)) # 64 x 128
        enc2_2 = (self.block2_2(enc2_1)) # 32 x 64
        # G3
        enc3_0 = (self.block3_in(x3)) # 64 x 128
        enc3_1 = (self.block3_1(enc3_0)) # 64 x 128
        enc3_2 = (self.block3_2(enc3_1)) # 32 x 64
        # G4
        enc4_0 = (self.block4_in(x4)) # 32 x 64
        enc4_1 = (self.block4_1(enc4_0)) # 32 x 64
        enc4_2 = (self.block4_2(enc4_1)) # 16 x 32
        # G5
        enc0 = (self.block_in(x5))
        enc1 = (self.block1(enc0))
        enc2 = (self.block2(enc1))
        enc3 = (self.block3(enc2))
        enc4 = (self.block4(enc3))
        enc5 = (self.block5(enc4))
        enc6 = (self.block6(enc5))
        dec1 = (self.dblock1(enc6))
        dec2 = (self.dblock2(ops.dstack(dec1,enc5)))
        dec3 = (self.dblock3(ops.dstack(dec2,enc4)))
        dec4 = (self.dblock4(ops.dstack(dec3,enc3)))
        dec5 = (self.dblock5(ops.dstack(dec4,enc2)))
        dec6 = (self.dblock6(ops.dstack(dec5,enc1)))
        # out5 = self.dblock_out(dec6)
        # G4
        dec4_0 = F.relu((self.dblock4_0(enc4_2) + dec6))
        dec4_1 = self.dblock4_1((ops.dstack(dec4_0, enc4_2)))
        dec4_2 = self.dblock4_2((ops.dstack(dec4_1, enc4_1)))
        # out4 = self.dblock4_out(dec4_2)
        # out4 = (F.interpolate(out5, scale_factor=2, mode='bilinear') + out4)
        # G3
        dec3_0 = F.relu((self.dblock3_0(enc3_2) + dec4_2))
        dec3_1 = self.dblock3_1((ops.dstack(dec3_0, enc3_2)))
        dec3_2 = self.dblock3_2((ops.dstack(dec3_1, enc3_1)))
        # out3 = self.dblock3_out(dec3_2)
        # out3 =  (F.interpolate(out4, scale_factor=2, mode='bilinear') + out3)
        # G2
        dec2_0 = F.relu((self.dblock2_0(enc2_2) + dec3_2))
        dec2_1 = self.dblock2_1((ops.dstack(dec2_0, enc2_2)))
        dec2_2 = self.dblock2_2((ops.dstack(dec2_1, enc2_1)))
        out2 = self.dblock2_out((dec2_2))
        out = torch.tanh(out2)
        # out =  torch.tanh(F.interpolate(out3, scale_factor=2, mode='bilinear') + out2)

        return out, out

class D2(nn.Module):
    def __init__(self):
        super(D2, self).__init__()
        self.discirminator = nn.Sequential(
            ops.conv_sn_leak(6, 128, 3, 1, 1), # out 128 x 256
            ops.conv_sn_leak(128, 128, 4, 2, 1), # out 64 x 128
            ops.conv_sn_leak(128, 256, 4, 2, 1), # out 32 x 64
            ops.conv_sn_leak(256, 256, 4, 2, 1), # out 16 x 32
            ops.conv_sn_leak(256, 512, 4, 2, 1), # out 8 x 16
            ops.conv_sn_leak(512, 512, 3, 1, 1), # out 8 x 16
            ops.conv_sn_leak(512, 512, 3, 1, 1), # out 8 x 16
            ops.conv_sn_leak(512, 512, 4, 2, 1), # out 4 x 8
            ops.conv_sn_leak(512, 512, 4, 2, 1), # out 2 x 4
            ops.conv_sigmoid(512, 1, (2,4), 1, 0), # out 1 x 1
        )

    def forward(self, x):
        out = self.discirminator(x)
        return out

class D2_down(nn.Module):
    def __init__(self):
        super(D2_down, self).__init__()
        self.conv1 = ops.conv_sn_leak(6, 128, 3, 1, 1)
        self.conv2 = ops.conv_sn_leak(128 + 6, 128, 4, 2, 1) # out 64 x 128
        self.conv3 = ops.conv_sn_leak(128 + 6, 256, 4, 2, 1) # out 32 x 64
        self.conv4 = ops.conv_sn_leak(256 + 6, 256, 4, 2, 1) # out 16 x 32
        self.conv5 = ops.conv_sn_leak(256 + 6, 512, 4, 2, 1) # out 8 x 16
        self.conv6 = ops.conv_sn_leak(512 + 6, 512, 3, 1, 1) # out 8 x 16
        self.conv7 = ops.conv_sn_leak(512, 512, 3, 1, 1) # out 8 x 16
        self.conv8 = ops.conv_sn_leak(512, 512, 4, 2, 1) # out 4 x 8
        self.conv9 = ops.conv_sn_leak(512, 512, 4, 2, 1) # out 2 x 4
        self.conv10 = ops.conv_sigmoid(512, 1, (2,4), 1, 0) # out 1 x 1

    def forward(self, x):
        x2 = ops.downsample(x, 2) # out 64 x 128
        x3 = ops.downsample(x, 4) # out 32 x 64
        x4 = ops.downsample(x, 8) # out 16 x 32
        x5 = ops.downsample(x, 16) # out 8 x 16
        out = self.conv1(x) # 128 x 256
        out = self.conv2(ops.dstack(out,x))
        out = self.conv3(ops.dstack(out,x2))
        out = self.conv4(ops.dstack(out,x3))
        out = self.conv5(ops.dstack(out,x4))
        out = self.conv6(ops.dstack(out,x5))
        out = self.conv7(out)
        out = self.conv8(out)
        out = self.conv9(out)
        out = self.conv10(out)
        return out

class G1(nn.Module):
    def __init__(self):
        super(G1, self).__init__()
        # In 256 x 512
        self.block1_in = ops.conv_norm_relu(3, 64, 3, 1, 1) # out 256 x 512
        self.block1_1 = ops.conv_norm_relu(64, 256, 3, 1, 1) # out 256 x 512
        self.block1_2 = ops.conv_norm_relu(256, 256, 4, 2, 1) # 128 x 256
        # In 128 x 256
        self.block2_in = ops.conv_norm_relu(3, 64, 3, 1, 1) # out 128 x 256
        self.block2_1 = ops.conv_norm_relu(64, 256, 3, 1, 1) # out 128 x 256
        self.block2_2 = ops.conv_norm_relu(256, 256, 4, 2, 1) # out 64 x 128
        # In 64 x 128
        self.block3_in = ops.conv_norm_relu(3, 64, 3, 1, 1) # out 64 x 128
        self.block3_1 = ops.conv_norm_relu(64, 256, 3, 1, 1) # out 64 x 128
        self.block3_2 = ops.conv_norm_relu(256, 256, 4, 2, 1) # out 32 x 64
        # In 32 x 64
        self.block4_in = ops.conv_norm_relu(3, 64, 3, 1, 1) # out 32 x 64
        self.block4_1 = ops.conv_norm_relu(64, 256, 3, 1, 1) # out 32 x 64
        self.block4_2 = ops.conv_norm_relu(256, 256, 4, 2, 1) # out 16 x 32
        # In 16 x 32
        self.block_in = ops.conv_norm_relu(3, 64, 3, 1, 1) # out 16 x 32
        self.block1 = ops.conv_norm_relu(64, 256, 3, 1, 1) # out 16 x 32
        self.block2 = ops.conv_norm_relu(256, 256, 4, 2, 1) # out 8 x 16
        self.block3 = ops.conv_norm_relu(256, 512, 3, 1, 1) # out 8 x 16
        self.block4 = ops.conv_norm_relu(512, 512, 4, 2, 1) # out 4 x 8
        self.block5 = ops.conv_norm_relu(512, 512, 3, 1, 1) # out 4 x 8
        self.block6 = ops.conv_norm_relu(512, 512, 3, 1, 1) # out 4 x 8

        # --
        # Out 16 x 32
        self.dblock1 = ops.convT_norm_leak(512, 512, 3, 1, 1) # out 4 x 8
        self.dblock2 = ops.convT_norm_leak(512 + 512, 512, 3, 1, 1) # out 4 x 8
        self.dblock3 = ops.convT_norm_leak(512 + 512,  512, 4, 2, 1) # out 8 x 16
        self.dblock4 = ops.convT_norm_leak(512 + 512, 256, 3, 1, 1) # out 8 x 16
        self.dblock5 = ops.convT_norm_leak(256 + 256, 256, 4, 2, 1) # out 16 x 32
        self.dblock6 = ops.convT_norm_leak(256 + 256, 256, 3, 1, 1) # out 16 x 32
        self.dblock_out =  ops.conv(256, 3, 1, 1, 0) # out 16 x 32
        # Out 32 x 64
        self.dblock4_0 = ops.convT_norm_leak(256, 256, 3, 1, 1) # 16 x 32
        self.dblock4_1 = ops.convT_norm_leak(256 + 256, 256, 4, 2, 1)  # 32 x 64
        self.dblock4_2 = ops.convT_norm_leak(256 + 256, 256, 3, 1, 1)  # 32 x 64
        self.dblock4_out = ops.conv(256, 3, 1, 1, 0) # out 32 x 64
        # Out 64 x 128
        self.dblock3_0 = ops.convT_norm_leak(256, 256, 3, 1, 1) # out 32 x 64
        self.dblock3_1 = ops.convT_norm_leak(256 + 256, 256, 4, 2, 1) # out 64 x 128
        self.dblock3_2 = ops.convT_norm_leak(256 + 256, 256, 3, 1, 1) # out 64 x 128
        self.dblock3_out = ops.conv(256, 3, 1, 1, 0) # out 64 x 128
        # Out 128 x 256
        self.dblock2_0 = ops.convT_norm_leak(256, 256, 3, 1, 1) # out 64 x 128
        self.dblock2_1 = ops.convT_norm_leak(256 + 256, 256, 4, 2, 1) # out 128 x 256
        self.dblock2_2 = ops.convT_norm_leak(256 + 256, 256, 3, 1, 1) # out 128 x 256
        self.dblock2_out = ops.conv(256, 3, 3, 1, 1) # out 128 x 256
        # Out 256 x 512
        self.dblock1_0 = ops.convT_norm_leak(256, 256, 3, 1, 1) # out 128 x 256
        self.dblock1_1 = ops.convT_norm_leak(256 + 256, 256, 4, 2, 1) # out 256 x 512
        self.dblock1_2 = ops.convT_norm_leak(256 + 256, 256, 3, 1, 1) # out 256 x 512
        self.dblock1_out = ops.conv(256, 3, 3, 1, 1) # out 256 x 512

    def forward(self, x):
        x1 = ops.downsample(x, 2)
        x2 = ops.downsample(x, 4)
        x3 = ops.downsample(x, 8)
        x4 = ops.downsample(x, 16)
        x5 = ops.downsample(x, 32)
        # G1
        enc1_0 = self.block1_in(x1)
        enc1_1 = self.block1_1(enc1_0)
        enc1_2 = self.block1_2(enc1_1)
        # G2
        enc2_0 = (self.block2_in(x2)) # 64 x 128
        enc2_1 = (self.block2_1(enc2_0)) # 64 x 128
        enc2_2 = (self.block2_2(enc2_1)) # 32 x 64
        # G3
        enc3_0 = (self.block3_in(x3)) # 64 x 128
        enc3_1 = (self.block3_1(enc3_0)) # 64 x 128
        enc3_2 = (self.block3_2(enc3_1)) # 32 x 64
        # G4
        enc4_0 = (self.block4_in(x4)) # 32 x 64
        enc4_1 = (self.block4_1(enc4_0)) # 32 x 64
        enc4_2 = (self.block4_2(enc4_1)) # 16 x 32
        # G5
        enc0 = (self.block_in(x5))
        enc1 = (self.block1(enc0))
        enc2 = (self.block2(enc1))
        enc3 = (self.block3(enc2))
        enc4 = (self.block4(enc3))
        enc5 = (self.block5(enc4))
        enc6 = (self.block6(enc5))
        dec1 = (self.dblock1(enc6))
        dec2 = (self.dblock2(ops.dstack(dec1,enc5)))
        dec3 = (self.dblock3(ops.dstack(dec2,enc4)))
        dec4 = (self.dblock4(ops.dstack(dec3,enc3)))
        dec5 = (self.dblock5(ops.dstack(dec4,enc2)))
        dec6 = (self.dblock6(ops.dstack(dec5,enc1)))
        # out5 = self.dblock_out(dec6)
        # G4
        dec4_0 = F.relu((self.dblock4_0(enc4_2) + dec6))
        dec4_1 = self.dblock4_1((ops.dstack(dec4_0, enc4_2)))
        dec4_2 = self.dblock4_2((ops.dstack(dec4_1, enc4_1)))
        # out4 = self.dblock4_out(dec4_2)
        # out4 = (F.interpolate(out5, scale_factor=2, mode='bilinear') + out4)
        # G3
        dec3_0 = F.relu((self.dblock3_0(enc3_2) + dec4_2))
        dec3_1 = self.dblock3_1((ops.dstack(dec3_0, enc3_2)))
        dec3_2 = self.dblock3_2((ops.dstack(dec3_1, enc3_1)))
        # out3 = self.dblock3_out(dec3_2)
        # out3 =  (F.interpolate(out4, scale_factor=2, mode='bilinear') + out3)
        # G2
        dec2_0 = F.relu((self.dblock2_0(enc2_2) + dec3_2))
        dec2_1 = self.dblock2_1((ops.dstack(dec2_0, enc2_2)))
        dec2_2 = self.dblock2_2((ops.dstack(dec2_1, enc2_1)))
        # out2 = self.dblock2_out((dec2_2))
        # out = torch.tanh(out2)
        # out =  torch.tanh(F.interpolate(out3, scale_factor=2, mode='bilinear') + out2)
        dec1_0 = F.relu((self.dblock1_0(enc1_2) + dec2_2))
        dec1_1 = self.dblock1_1(ops.dstack(dec1_0, enc1_2))
        dec1_2 = self.dblock1_2(ops.dstack(dec1_1, enc1_1))
        out1 = self.dblock1_out(dec1_2)
        out = torch.tanh(out1)
        # out = torch.tanh(out1 + ops.upsample(out2, 2))

        return out, out

class NDisc(nn.Module):
    def __init__(self, inch):
        super(NDisc, self).__init__()
        norm = 'instnorm'
        activ = 'leaky'
        self.conv1 = Block(inch, 64, 3, 2, 1, activ)
        self.conv2 = Block(64, 128, 3, 2, 1, activ, norm)
        self.conv3 = Block(128, 256, 3, 2, 1, activ, norm)
        self.conv4 = Block(256, 512, 3, 2, 1, activ, norm)
        self.conv5 = Block(512, 1024, 3, 2, 1, activ, norm)
        self.conv6 = Block(1024, 1, 3, 1, 1)

    def forward(self, x):
        out1 = self.conv1(x)
        out2 = self.conv2(out1)
        out3 = self.conv3(out2)
        out4 = self.conv4(out3)
        out5 = self.conv5(out4)
        out6 = self.conv6(out5)
        return [out1, out2, out3, out4, out5, out6]

class MultiD(nn.Module):
    def __init__(self, inch):
        super(MultiD, self).__init__()
        self.d1 = NDisc(inch)
        self.d2 = NDisc(inch)

    def forward(self, x):
        xs = ops.downsample(x, 2)
        out1 = self.d1(x)
        out2 = self.d2(xs)
        return [out1, out2]

class MultiDscale(nn.Module):
    def __init__(self, inch):
        super(MultiDscale, self).__init__()
        self.d1 = NDisc(inch)
        self.d2 = NDisc(inch)
        self.d3 = NDisc(inch)

    def forward(self, x1, x2, x3):
        x2 = ops.downsample(x2, 2)
        x3 = ops.downsample(x3, 4)
        out1 = self.d1(x1)
        out2 = self.d1(x2)
        out3 = self.d1(x3)
        return [out1, out2, out3]

class D1(nn.Module):
    def __init__(self, inch=6):
        super(D1, self).__init__()
        norm = 'instnorm'
        activ = 'leaky'
        modules = []

        modules += [Block(inch, 64, 3, 2, 1, activ, norm)] # out 128 x 256
        # modules += [ResBlock(64, activ, norm, 4)] # out 128 x 256
        modules += [Block(64, 128, 3, 2, 1, activ, norm)] # out 64 x 128
        # modules += [ResBlock(128, activ, norm, 4)] # out 64 x 128
        modules += [Block(128, 256, 3, 2, 1, activ, norm)] # out 32 x 64
        # modules += [ResBlock(256, activ, norm, 4)] # out 32 x 64
        modules += [Block(256, 512, 3, 2, 1, activ, norm)] # out 16 x 32
        # modules += [ResBlock(512, activ, norm, 4)] # out 16 x 32
        modules += [Block(512, 1024, 3, 2, 1, activ, norm)] # out 8 x 16
        # modules += [ResBlock(1024, activ, norm, 4)] # out 8 x 16
        modules += [Block(1024, 1, (3,3), 1, 1)]

        self.convblock1 = nn.Sequential(*modules)
        self.convblock2 = nn.Sequential(*modules)


    def forward(self, x):
        x2 = ops.downsample(x, 2)
        out1 = self.convblock1(x)
        out2 = self.convblock2(x2)
        # out1 = torch.sigmoid(out1)
        # out2 = torch.sigmoid(out2)
        return [out1, out2]

class G0(nn.Module):
    def __init__(self):
        super(G0, self).__init__()
        # Block 0
        self.block0_0 = ops.conv_norm_relu(3, 64, 3, 1, 1) # out 512 x 1024
        self.block0_1 = ops.conv_norm_relu(64, 64, 3, 1, 1) # out 512 x 1024
        self.block0_2 = ops.conv_norm_relu(64, 64, 4, 2, 1) # out 256 x 512
        # Block 1
        self.block1_0 = ops.conv_norm_relu(3, 64, 3, 1, 1) # out 256 x 512
        self.block1_1 = ops.conv_norm_relu(64, 64, 3, 1, 1) # out 256 x 512
        self.block1_2 = ops.conv_norm_relu(64, 64, 4, 2, 1) # out 128 x 256
        # Block 2
        self.block2_0 = ops.conv_norm_relu(3, 64, 3, 1, 1) # out 128 x 256
        self.block2_1 = ops.conv_norm_relu(64, 128, 3, 1, 1) # out 128 x 256
        self.block2_2 = ops.conv_norm_relu(128, 256, 4, 2, 1) # out 64 x 128
        # Block 3
        self.block3_0 = ops.conv_norm_relu(3, 64, 3, 1, 1) # out 64 x 128
        self.block3_1 = ops.conv_norm_relu(64, 128, 3, 1, 1) # out 64 x 128
        self.block3_2 = ops.conv_norm_relu(128, 256, 4, 2, 1) # out 32 x 64
        # Block 4
        self.block4_0 = ops.conv_norm_relu(3, 64, 3, 1, 1) # out 32 x 64
        self.block4_1 = ops.conv_norm_relu(64, 128, 3, 1, 1) # out 32 x 64
        self.block4_2 = ops.conv_norm_relu(128, 256, 4, 2, 1) # out 16 x 32
        self.block4_3 = ops.conv_norm_relu(256, 512, 4, 2, 1) # out 8 x 16
        self.block4_4 = ops.conv_norm_relu(512, 512, 4, 2, 1) # out 4 x 8
        self.block4_5 = ops.conv_norm_relu(512, 1024, 3, 1, 1) # out 4 x 8
        self.block4_6 = ops.conv_norm_relu(1024, 1024, 3, 1, 1) # out 4 x 8
        # Dblock 4
        self.dblock4_0 = ops.convT_norm_leak(1024, 512, 3, 1, 1) # out 4 x 8
        self.dblock4_1 = ops.convT_norm_leak(1024 + 512, 512, 3, 1, 1) # out 4 x 8
        self.dblock4_2 = ops.convT_norm_leak(512 + 512, 512, 4, 2, 1) # out 8 x 16
        self.dblock4_3 = ops.convT_norm_leak(512 + 512, 256, 4, 2, 1) # out 16 x 32
        self.dblock4_4 = ops.convT_norm_leak(256 + 256, 128, 4, 2, 1) # out 32 x 64
        # self.dblock4_1 = ops.convT_sn_leak(512, 512, 3, 1, 1) # out 4 x 8
        # self.dblock4_2 = ops.convT_sn_leak(512, 512, 4, 2, 1) # out 8 x 16
        # self.dblock4_3 = ops.convT_sn_leak(512, 256, 4, 2, 1) # out 16 x 32
        # self.dblock4_4 = ops.convT_sn_leak(256, 128, 4, 2, 1) # out 32 x 64
        self.dblock4_5 = ops.convT_norm_leak(128 + 128, 64, 3, 1, 1) # out 32 x 64
        self.dblock4_6 = ops.convT_norm_leak(64 + 64, 256, 3, 1, 1) # out 32 x 64
        # Dblock 3
        self.dblock3_0 = ops.convT_norm_leak(256, 128, 4, 2, 1) # 64 x 128
        self.dblock3_1 = ops.convT_norm_leak(128 + 128, 64, 3, 1, 1) # 64 x 128
        self.dblock3_2 = ops.convT_norm_leak(64 + 64, 256, 3, 1, 1) # 64 x 128
        # Dblock 2
        self.dblock2_0 = ops.convT_norm_leak(256, 128, 4, 2, 1) # 128 x 256
        self.dblock2_1 = ops.convT_norm_leak(128 + 128, 64, 3, 1, 1) # 128 x 256
        self.dblock2_2 = ops.convT_norm_leak(64 + 64, 64, 3, 1, 1) # 128 x 256
        # Dblock 1
        self.dblock1_0 = ops.convT_norm_leak(64, 64, 4, 2, 1) # 256 x 512
        self.dblock1_1 = ops.convT_norm_leak(64 + 64, 64, 3, 1, 1) # 256 x 512
        self.dblock1_2 = ops.convT_norm_leak(64 + 64, 64, 3, 1, 1) # 256 x 512
        # Dblock 1
        self.dblock0_0 = ops.convT_norm_leak(64, 64, 4, 2, 1) # 512 x 1024
        self.dblock0_1 = ops.convT_norm_leak(64 + 64, 64, 3, 1, 1) # 512 x 1024
        self.dblock0_2 = ops.convT(64 + 64, 3, 3, 1, 1) # 512 x 1024

    def forward(self, x):
        x1 = ops.downsample(x, 2)
        x2 = ops.downsample(x, 4)
        x3 = ops.downsample(x, 8)
        x4 = ops.downsample(x, 16)

        enc0_0 = self.block0_0(x) # 512 x 1024
        enc0_1 = self.block0_1(enc0_0) # 512 x 1024
        enc0_2 = self.block0_2(enc0_1) # 256 x 512

        enc1_0 = self.block1_0(x1) # 256 x 512
        enc1_1 = self.block1_1(enc1_0) # 256 X 512
        enc1_2 = self.block1_2(enc1_1) # 128 x 256

        enc2_0 = self.block2_0(x2) # 128 x 256
        enc2_1 = self.block2_1(enc2_0) # 128 x 256
        enc2_2 = self.block2_2(enc2_1) # 64 x 128

        enc3_0 = self.block3_0(x3) # 64 x 128
        enc3_1 = self.block3_1(enc3_0) # 64 x 128
        enc3_2 = self.block3_2(enc3_1) # 32 x 64

        enc4_0 = self.block4_0(x4) # out 32 x 64 x 64
        enc4_1 = self.block4_1(enc4_0) # out 32 x 64 x 128
        enc4_2 = self.block4_2(enc4_1) # out 16 x 32 x 256
        enc4_3 = self.block4_3(enc4_2) # out 8 x 16 x 512
        enc4_4 = self.block4_4(enc4_3) # out 4 x 8 x 512
        enc4_5 = self.block4_5(enc4_4) # out 4 x 8 x 1024
        enc4_6 = self.block4_6(enc4_5) # out 4 x 8 x 1024

        dec4_0 = self.dblock4_0(enc4_6) # out 4 x 8 x 512
        dec4_1 = self.dblock4_1(ops.dstack(dec4_0, enc4_5)) # out 4 x 8
        dec4_2 = self.dblock4_2(ops.dstack(dec4_1, enc4_4)) # out 8 x 16
        dec4_3 = self.dblock4_3(ops.dstack(dec4_2, enc4_3)) # out 16 x 32
        dec4_4 = self.dblock4_4(ops.dstack(dec4_3, enc4_2)) # out 32 x 64
        dec4_5 = self.dblock4_5(ops.dstack(dec4_4, enc4_1)) # out 32 x 64
        dec4_6 = self.dblock4_6(ops.dstack(dec4_5, enc4_0)) # out 32 x 64

        dec3_0 = self.dblock3_0(dec4_6 + enc3_2) # out 64 x 128
        dec3_1 = self.dblock3_1(ops.dstack(dec3_0, enc3_1)) # out 64 x 128
        dec3_2 = self.dblock3_2(ops.dstack(dec3_1, enc3_0)) # out 64 x 128

        dec2_0 = self.dblock2_0(dec3_2 + enc2_2) # out 128 x 256
        dec2_1 = self.dblock2_1(ops.dstack(dec2_0, enc2_1)) # out 128 x x 256
        dec2_2 = self.dblock2_2(ops.dstack(dec2_1, enc2_0)) # out 128 x 256

        dec1_0 = self.dblock1_0(dec2_2 + enc1_2) # out 256 x 512
        dec1_1 = self.dblock1_1(ops.dstack(dec1_0, enc1_1)) # out 256 x 512
        dec1_2 = self.dblock1_2(ops.dstack(dec1_1, enc1_0)) # out 256 x 512

        dec0_0 = self.dblock0_0(dec1_2 + enc0_2) # out 512 x 1024
        dec0_1 = self.dblock0_1(ops.dstack(dec0_0, enc0_1)) # out 512 x 1024
        dec0_2 = self.dblock0_2(ops.dstack(dec0_1, enc0_0)) # out 512 x 1024

        return torch.tanh(dec0_2)

class D0(nn.Module):
    def __init__(self, inch=6):
        super(D0, self).__init__()
        norm = 'instnorm'
        activ = 'leaky'
        modules = []

        modules += [Block(inch, 64, 4, 2, 1, activ, norm)]  # out 256 x 512
        modules += [ResBlock(64, activ, norm, 4)]  # out 256 x 512
        modules += [ResBlock(64, activ, norm, 4)]  # out 256 x 512
        modules += [Block(64, 128, 4, 2, 1, activ, norm)]  # out 128 x 256
        modules += [ResBlock(128, activ, norm, 4)]  # out 128 x 256
        modules += [ResBlock(128, activ, norm, 4)]  # out 128 x 256
        modules += [Block(128, 256, 4, 2, 1, activ, norm)]  # out 64 x 128
        modules += [ResBlock(256, activ, norm, 4)]  # out 64 x 128
        modules += [ResBlock(256, activ, norm, 4)]  # out 64 x 128
        modules += [Block(256, 512, 4, 2, 1, activ, norm)]  # out 32 x 64
        modules += [ResBlock(512, activ, norm, 4)]  # out 32 x 64
        modules += [ResBlock(512, activ, norm, 4)]  # out 32 x 64
        modules += [Block(512, 512, 4, 2, 1, activ, norm)]  # out 16 x 32
        modules += [ResBlock(512, activ, norm, 4)]  # out 16 x 32
        modules += [ResBlock(512, activ, norm, 4)]  # out 16 x 32
        modules += [Block(512, 1, (3, 1), 1, 1)]

        self.convblock = nn.Sequential(*modules)

    def forward(self, x):
        out = self.convblock(x)
        return torch.sigmoid(out)


class BlockTrans(nn.Module):
    def __init__(self, inf, outf, ksz, s, pad, activ=None, norm=None):
        super(BlockTrans, self).__init__()
        modules = []
        modules.append(nn.ConvTranspose2d(inf, outf, ksz, s, pad))

        if norm == 'batchnorm':
            modules.append(nn.BatchNorm2d(outf, affine=True))
        elif norm == 'instnorm':
            modules.append(nn.InstanceNorm2d(outf, affine=False))

        if activ == 'relu':
            modules.append(nn.ReLU(inplace=True))
        elif activ == 'leaky':
            modules.append(nn.LeakyReLU(0.2, inplace=True))
        elif activ == 'sigmoid':
            modules.append(nn.Sigmoid())
        elif activ == 'tanh':
            modules.append(nn.Tanh())

        self.block = nn.Sequential(*modules)

    def forward(self, x):
        out = self.block(x)
        return out

class Block(nn.Module):
    def __init__(self, inf, outf, ksz, s, pad, activ=None, norm=None):
        super(Block, self).__init__()
        modules = []
        modules.append(nn.Conv2d(inf, outf, ksz, s, pad))

        if norm == 'batchnorm':
            modules.append(nn.BatchNorm2d(outf, affine=True))
        elif norm == 'instnorm':
            modules.append(nn.InstanceNorm2d(outf, affine=False))

        if activ == 'relu':
            modules.append(nn.ReLU(inplace=True))
        elif activ == 'leaky':
            modules.append(nn.LeakyReLU(0.2, inplace=True))
        elif activ == 'sigmoid':
            modules.append(nn.Sigmoid())
        elif activ == 'tanh':
            modules.append(nn.Tanh())

        self.block = nn.Sequential(*modules)

    def forward(self, x):
        out = self.block(x)
        return out

class ResBlock(nn.Module):
    def __init__(self, inf, activ=None, norm=None, depth=2):
        super(ResBlock, self).__init__()
        modules = []

        for i in range(depth):
            modules.append(nn.Conv2d(inf, inf, 3, 1, 1))

            if norm == 'batchnorm':
                modules.append(nn.BatchNorm2d(inf, affine=True))
            elif norm == 'instnorm':
                modules.append(nn.InstanceNorm2d(inf, affine=False))

            if activ == 'relu':
                modules.append(nn.ReLU(inplace=True))
                self.activ = nn.ReLU(inplace=True)
            elif activ == 'leaky':
                modules.append(nn.LeakyReLU(0.2, inplace=True))
                self.activ = nn.LeakyReLU(0.2, inplace=True)
            elif activ == 'sigmoid':
                modules.append(nn.Sigmoid())
                self.activ = nn.Sigmoid()
            elif activ == 'tanh':
                modules.append(nn.Tanh())
                self.activ = nn.Tanh()

        modules = modules[:-1]
        self.block = nn.Sequential(*modules)

    def forward(self, x):
        out = self.activ(x + self.block(x))
        return out


class GeneratorMed(nn.Module):
    def __init__(self):
        super(GeneratorMed, self).__init__()
        norm = 'instnorm'
        self.block1 = Block(3, 32, 3, 1, 1, 'relu') # 512 x 1024
        self.block2 = Block(32, 64, 4, 2, 1, 'relu', norm) # 256 x 512
        self.block3 = Block(64, 128, 4, 2, 1, 'relu', norm) # 128 x 256
        self.block4 = Block(128, 128, 4, 2, 1, norm=norm) # 64 x 128
        self.block4r = nn.Sequential(
            ResBlock(128,'relu', norm, depth=2),
            ResBlock(128,'relu', norm, depth=2),
            ResBlock(128,'relu', norm, depth=2),
            ResBlock(128,'relu', norm, depth=2),
        ) # 64 x 128
        self.block5 = Block(128, 128, 4, 2, 1, norm=norm) # 32 x 64
        self.block5r = nn.Sequential(
            ResBlock(128, 'relu', norm, depth=2),
            ResBlock(128, 'relu', norm, depth=2),
            ResBlock(128, 'relu', norm, depth=2),
            ResBlock(128, 'relu', norm, depth=2),
        ) # 32 x 64
        self.block6 = Block(128, 128, 4, 2, 1, norm=norm) # 16 x 32
        self.block6r = nn.Sequential(
            ResBlock(128, 'relu', norm, depth=2),
            ResBlock(128, 'relu', norm, depth=2),
            ResBlock(128, 'relu', norm, depth=2),
            ResBlock(128, 'relu', norm, depth=2),
        ) # 16 x 32
        self.block7 = Block(128, 256, 4, 2, 1, norm=norm) # 8 x 16
        self.block7r = nn.Sequential(
            ResBlock(256, 'relu', norm, depth=2),
            ResBlock(256, 'relu', norm, depth=2),
            ResBlock(256, 'relu', norm, depth=2),
            ResBlock(256, 'relu', norm, depth=2),
        ) # 8 x 16
        self.block8 = Block(256, 512, 4, 2, 1, 'relu', norm) # 4 x 8

        self.dblock1 = BlockTrans(512, 256, 3, 1, 1, norm=norm) # 4 x 8
        self.dblock1r = nn.Sequential(
            ResBlock(256, 'relu', norm, depth=2),
            ResBlock(256, 'relu', norm, depth=2),
            ResBlock(256, 'relu', norm, depth=2),
            ResBlock(256, 'relu', norm, depth=2),
        ) # 4 x 8
        self.dblock2 = BlockTrans(256, 128, 4, 2, 1, norm=norm) # 8 x 16
        self.dblock2r = nn.Sequential(
            ResBlock(128, 'relu', norm, depth=2),
            ResBlock(128, 'relu', norm, depth=2),
            ResBlock(128, 'relu', norm, depth=2),
            ResBlock(128, 'relu', norm, depth=2),
        ) # 8 x 16
        self.dblock3 = BlockTrans(128, 128, 4, 2, 1, norm=norm) # 16 x 32
        self.dblock3r = nn.Sequential(
            ResBlock(128, 'relu', norm, depth=2),
            ResBlock(128, 'relu', norm, depth=2),
            ResBlock(128, 'relu', norm, depth=2),
            ResBlock(128, 'relu', norm, depth=2),
        ) # 16 x 32
        self.dblock4 = BlockTrans(128, 128, 4, 2, 1, 'relu', norm) # 32 x 64
        self.dblock4r = nn.Sequential(
            ResBlock(128, 'relu', norm, depth=2),
            ResBlock(128, 'relu', norm, depth=2),
            ResBlock(128, 'relu', norm, depth=2),
            ResBlock(128, 'relu', norm, depth=2),
        ) # 32 x 64
        self.dblock5 = BlockTrans(128, 128, 4, 2, 1, 'relu', norm) # 64 x 128
        self.dblock5r = nn.Sequential(
            ResBlock(128, 'relu', norm, depth=2),
            ResBlock(128, 'relu', norm, depth=2),
            ResBlock(128, 'relu', norm, depth=2),
            ResBlock(128, 'relu', norm, depth=2),
        ) # 64 x 128
        self.dblock6 = BlockTrans(128 + 128, 256, 4, 2, 1, 'relu', norm) # 128 x 256
        self.dblock6r = ResBlock(256, 'relu', norm, depth=2)
        self.dblock7 = BlockTrans(256 + 128, 128, 4, 2, 1, 'relu', norm) # 256 x 512
        self.dblock7r = ResBlock(128, 'relu', norm, depth=2)
        self.rgb_med = BlockTrans(128, 3, 3, 1, 1) # 256 x 512
        # self.dblock8 = BlockTrans(64, 3, 4, 2, 1, 'leaky', norm) # 512 x 1024
        # self.dblock10 = BlockTrans(512, 512, 4, 2, 1, 'leak', norm)

    def forward(self, x):
        xmed = ops.downsample(x, 2)
        enc1 = self.block1(x)
        enc2 = self.block2(enc1)
        enc3 = self.block3(enc2)
        enc4 = self.block4(enc3)
        enc4 = self.block4r(enc4)
        enc5 = self.block5(enc4)
        enc5 = self.block5r(enc5)
        enc6 = self.block6(enc5)
        enc6 = self.block6r(enc6)
        enc7 = self.block7(enc6)
        enc7 = self.block7r(enc7)
        enc8 = self.block8(enc7)

        dec1 = self.dblock1(enc8)
        dec1 = self.dblock1r(dec1)
        dec2 = self.dblock2(dec1)
        dec2 = self.dblock2r(dec2)
        dec3 = self.dblock3(dec2)
        dec3 = self.dblock3r(dec3)
        dec4 = self.dblock4(dec3)
        dec4 = self.dblock4r(dec4)
        dec5 = self.dblock5(dec4)
        dec5 = self.dblock5r(dec5)
        dec6 = self.dblock6(ops.dstack(dec5, enc4))
        dec6 = self.dblock6r(dec6)
        dec7 = self.dblock7(ops.dstack(dec6, enc3))
        dec7 = self.dblock7r(dec7)
        out = self.rgb_med(dec7)

        return out, torch.tanh(out)


class GeneratorMedBalance(nn.Module):
    def __init__(self, inch=3):
        super(GeneratorMedBalance, self).__init__()
        norm = 'instnorm'
        activ = 'relu'

        self.in_large0 = Block(inch, 64, 3, 1, 1, activ, norm) # 512 x 1024
        self.in_large1 = Block(64, 128, 4, 2, 1, norm) # 256 x 512

        self.in_med0 = Block(inch, 64, 3, 1, 1, activ, norm) # 256 x 512
        self.in_med1 = Block(64, 128, 3, 1, 1, activ, norm) # 256 x 512
        self.in_med2 = Block(128, 128, 4, 2, 1, norm) # 128 x 256

        self.in_small0 = Block(inch, 64, 3, 1, 1, activ, norm) # 128 x 256
        self.in_small1 = Block(64, 128, 3, 1, 1, activ, norm) # 128 x 256
        self.in_small2 = Block(128, 128, 4, 2, 1, activ, norm) # 64 x 128

        self.conv1 = nn.Sequential(
            Block(128, 128, 3, 1, 1, activ, norm),
            ResBlock(128, activ, norm, 4),
        ) # 64 x 128
        self.conv2 = nn.Sequential(
            Block(128, 256, 4, 2, 1, activ, norm),
            ResBlock(256, activ, norm, 4),
        ) # 32 x 64
        self.conv3 = nn.Sequential(
            Block(256, 256, 4, 2, 1, activ, norm),
            ResBlock(256, activ, norm, 4),
        ) # 16 x 32
        self.conv4 = nn.Sequential(
            Block(256, 256, 4, 2, 1, activ, norm),
            ResBlock(256, activ, norm, 4),
        ) # 8 x 16
        self.conv5 = nn.Sequential(
            Block(256, 512, 4, 2, 1, activ, norm),
            ResBlock(512, activ, norm, 4),
        ) # 4 x 8

        self.dconv5 = nn.Sequential(
            BlockTrans(512, 512, 4, 2, 1, activ, norm),
            ResBlock(512, activ, norm, 4),
        ) # 8 x 16
        self.dconv4 = nn.Sequential(
            BlockTrans(512, 256, 4, 2, 1, activ, norm),
            ResBlock(256, activ, norm, 4),
        ) # 16 x 32
        self.dconv3 = nn.Sequential(
            BlockTrans(256, 256, 4, 2, 1, activ, norm),
            ResBlock(256, activ, norm, 4),
        ) # 32 x 64
        self.dconv2 = nn.Sequential(
            BlockTrans(256, 128, 4, 2, 1, activ, norm),
            ResBlock(128, activ, norm, 4),
        ) # 64 x 128
        self.dconvs = BlockTrans(128 + 128, 128, 4, 2, 1, activ, norm) # 128 x 256
        self.dconvm = BlockTrans(128 + 128, 128, 4, 2, 1, activ, norm) # 256 x 512
        self.dconvl = BlockTrans(128 + 128, 128, 4, 2, 1, activ, norm) # 512 x 1024

        self.out_small = Block(128, 3, 3, 1, 1) # 128 x 256
        self.out_med = Block(128, 3, 3, 1, 1) # 256 x 512
        self.out_large = Block(128, 3, 3, 1, 1) # 512 x 1024

    def forward(self, x):
        xs = ops.downsample(x, 2)
        in_med0 = self.in_med0(x) # 256 x 512
        in_med1 = self.in_med1(in_med0) # 256 x 512
        in_med2 = self.in_med2(in_med1) # 128 x 256

        in_small0 = self.in_small0(xs) # 128 x 256
        in_small1 = self.in_small1(in_small0) # 128 x 256
        in_small2 = self.in_small2(in_small1 + in_med2) # 64 x 128
        enc1 = self.conv1(in_small2)
        enc2 = self.conv2(enc1)
        enc3 = self.conv3(enc2)
        enc4 = self.conv4(enc3)
        enc5 = self.conv5(enc4)

        dec5 = self.dconv5(enc5)
        dec4 = self.dconv4(dec5)
        dec3 = self.dconv3(dec4)
        dec2 = self.dconv2(dec3)
        decs = self.dconvs(dec2) + in_small2
        decm = self.dconvm(decs) + in_med2
        outm = self.out_med(decm)

        return outm, torch.tanh(outm)
        # Input
        # outl = self.in_large(xl)
        # outm = self.in_med(xm + outl)
        # outs = self.in_small(xs + outm)

        # Net
        # out = self.in_med(x)

        # Output
        # outs = self.deconvs(out)
        # small = self.out_small(outs)

        # outm = self.deconvm(outs + Up(small))
        # med = self.out_med(outm)

        # outl = self.deconvs(outm + Up(med))
        # large = self.out_large(outl)


class GeneratorLargeBalance(nn.Module):
    def __init__(self):
        super(GeneratorLargeBalance, self).__init__()
        norm = 'instnorm'
        activ = 'relu'

        self.in_large0 = Block(3, 64, 3, 1, 1, activ, norm) # 512 x 1024
        self.in_large1 = Block(64, 128, 4, 2, 1, norm) # 256 x 512

        self.in_med0 = Block(3, 64, 3, 1, 1, activ, norm) # 256 x 512
        self.in_med1 = Block(64, 128, 3, 1, 1, activ, norm) # 256 x 512
        self.in_med2 = Block(128, 128, 4, 2, 1, norm) # 128 x 256

        self.in_small0 = Block(3, 64, 3, 1, 1, activ, norm) # 128 x 256
        self.in_small1 = Block(64, 128, 3, 1, 1, activ, norm) # 128 x 256
        self.in_small2 = Block(128, 128, 4, 2, 1, activ, norm) # 64 x 128

        self.conv1 = nn.Sequential(
            Block(128, 128, 3, 1, 1, activ, norm),
            ResBlock(128, activ, norm, 4),
        ) # 64 x 128
        self.conv2 = nn.Sequential(
            Block(128, 256, 4, 2, 1, activ, norm),
            ResBlock(256, activ, norm, 4),
        ) # 32 x 64
        self.conv3 = nn.Sequential(
            Block(256, 256, 4, 2, 1, activ, norm),
            ResBlock(256, activ, norm, 4),
        ) # 16 x 32
        self.conv4 = nn.Sequential(
            Block(256, 256, 4, 2, 1, activ, norm),
            ResBlock(256, activ, norm, 4),
        ) # 8 x 16
        self.conv5 = nn.Sequential(
            Block(256, 512, 4, 2, 1, activ, norm),
            ResBlock(512, activ, norm, 4),
        ) # 4 x 8

        self.dconv5 = nn.Sequential(
            BlockTrans(512, 512, 4, 2, 1, activ, norm),
            ResBlock(512, activ, norm, 4),
        ) # 8 x 16
        self.dconv4 = nn.Sequential(
            BlockTrans(512, 256, 4, 2, 1, activ, norm),
            ResBlock(256, activ, norm, 4),
        ) # 16 x 32
        self.dconv3 = nn.Sequential(
            BlockTrans(256, 256, 4, 2, 1, activ, norm),
            ResBlock(256, activ, norm, 4),
        ) # 32 x 64
        self.dconv2 = nn.Sequential(
            BlockTrans(256, 128, 4, 2, 1, activ, norm),
            ResBlock(128, activ, norm, 4),
        ) # 64 x 128
        self.dconvs = BlockTrans(128 + 128, 128, 4, 2, 1, activ, norm) # 128 x 256
        self.dconvm = BlockTrans(128 + 128, 128, 4, 2, 1, activ, norm) # 256 x 512
        self.dconvl = BlockTrans(128 + 128, 128, 4, 2, 1, activ, norm) # 512 x 1024

        self.out_small = Block(128, 3, 3, 1, 1) # 128 x 256
        self.out_med = Block(128, 3, 3, 1, 1) # 256 x 512
        self.out_large = Block(128, 3, 3, 1, 1) # 512 x 1024

    def forward(self, x):
        xm = ops.downsample(x, 2)
        xs = ops.downsample(x, 4)

        in_large0 = self.in_large0(x)
        in_large1 = self.in_large1(in_large0)

        in_med0 = self.in_med0(xm) # 256 x 512
        in_med1 = self.in_med1(in_med0) # 256 x 512
        in_med2 = self.in_med2(in_med1 + in_large1) # 128 x 256

        in_small0 = self.in_small0(xs) # 128 x 256
        in_small1 = self.in_small1(in_small0) # 128 x 256
        in_small2 = self.in_small2(in_small1 + in_med2) # 64 x 128
        enc1 = self.conv1(in_small2)
        enc2 = self.conv2(enc1)
        enc3 = self.conv3(enc2)
        enc4 = self.conv4(enc3)
        enc5 = self.conv5(enc4)

        dec5 = self.dconv5(enc5)
        dec4 = self.dconv4(dec5)
        dec3 = self.dconv3(dec4)
        dec2 = self.dconv2(dec3)
        decs = self.dconvs(ops.dstack(dec2, in_small2))
        decm = self.dconvm(ops.dstack(decs, in_med2))
        decl = self.dconvl(ops.dstack(decm, in_large1))
        outl = self.out_med(decl)

        return torch.tanh(outl)


class GeneratorLargeBalance2(nn.Module):
    def __init__(self):
        super(GeneratorLargeBalance2, self).__init__()
        norm = 'instnorm'
        activ = 'leaky'
        self.conv1 = Block(3, 64, 3, 2, 1, activ) # 256 x 512
        self.conv2 = Block(64, 128, 3, 2, 1, activ, norm) # 128 x 256
        self.conv3 = Block(128, 256, 3, 2, 1, activ, norm) # 64 x 128
        self.conv4 = Block(256, 512, 3, 2, 1, activ, norm)
        self.conv5 = Block(512, 1024, 3, 2, 1, activ, norm)

        self.resblock = nn.Sequential(
           ResBlock(1024, activ, norm, 2),
           ResBlock(1024, activ, norm, 2),
           ResBlock(1024, activ, norm, 2),
           ResBlock(1024, activ, norm, 2),
           ResBlock(1024, activ, norm, 2),
           ResBlock(1024, activ, norm, 2),
        )
        self.dconv1 = BlockTrans(1024, 512, 4, 2, 1, activ, norm)
        self.dconv2 = BlockTrans(512, 256, 4, 2, 1, activ, norm)
        self.dconv3 = BlockTrans(256, 128, 4, 2, 1, activ, norm)
        self.dconv4 = BlockTrans(128, 64, 4, 2, 1, activ, norm)
        self.dconv5 = BlockTrans(64, 3, 4, 2, 1)

    def forward(self, x):
        enc1 = self.conv1(x)
        enc2 = self.conv2(enc1)
        enc3 = self.conv3(enc2)
        enc4 = self.conv4(enc3)
        enc5 = self.conv5(enc4)

        encr = self.resblock(enc5)

        dec1 = self.dconv1(encr)
        dec2 = self.dconv2(dec1)
        dec3 = self.dconv3(dec2)
        dec4 = self.dconv4(dec3)
        dec5 = self.dconv5(dec4)

        return torch.tanh(dec5)


# class GS(nn.Module):
#     def __init__(self):
#         super(GS, self).__init__()
#         activ = 'leaky'
#         norm = 'instnorm'
#
#         self.in_large0 = Block(3, 64, 3, 1, 1, activ, norm) # 512 x 1024
#         self.in_large1 = Block(64, 128, 4, 2, 1, norm) # 256 x 512
#
#         self.in_med0 = Block(3, 64, 3, 1, 1, activ, norm) # 256 x 512
#         self.in_med1 = Block(64, 128, 3, 1, 1, activ, norm) # 256 x 512
#         self.in_med2 = Block(128, 128, 4, 2, 1, norm) # 128 x 256
#
#         self.in_small0 = Block(3, 64, 3, 1, 1, activ, norm) # 128 x 256
#         self.in_small1 = Block(64, 128, 3, 1, 1, activ, norm) # 128 x 256
#         self.in_small2 = Block(128, 128, 4, 2, 1, activ, norm) # 64 x 128
#         self.attn_small = Attn(128)
#
#         self.conv3 = Block(128, 256, 3, 2, 1, activ, norm) # 32 x 64
#         self.conv4 = Block(256, 512, 3, 2, 1, activ, norm) # 16 x 32
#         self.conv5 = Block(512, 1024, 3, 2, 1, activ, norm) # 8 x 16
#
#         self.resblock = nn.Sequential(
#            ResBlock(1024, activ, norm, 2),
#            ResBlock(1024, activ, norm, 2),
#            ResBlock(1024, activ, norm, 2),
#            ResBlock(1024, activ, norm, 2),
#            ResBlock(1024, activ, norm, 2),
#            ResBlock(1024, activ, norm, 2),
#         )
#         self.dconv1 = BlockTrans(1024, 512, 4, 2, 1, activ, norm) # 16 x 32
#         self.dconv2 = BlockTrans(512, 256, 4, 2, 1, activ, norm) # 32 x 64
#         self.dconv3 = BlockTrans(256, 128, 4, 2, 1, activ, norm) # 64 x 128
#         self.dconvs = BlockTrans(128+ 128, 64, 4, 2, 1, activ, norm) # 128 x 256
#         self.dconvm = BlockTrans(64 + 128, 64, 4, 2, 1, activ, norm) # 256 x 512
#         self.dconvl = BlockTrans(64 + 128, 64, 4, 2, 1, activ, norm) # 512 x 1024
#
#         self.outs = BlockTrans(64, 3, 3, 1, 1)
#         self.outm = BlockTrans(64, 3, 3, 1, 1)
#         self.outl = BlockTrans(64, 3, 3, 1, 1)
#
#     def forward(self, x):
#         # xs = ops.downsample(x, 2)
#         # med0 = self.in_med0(x) # 256 x 512
#         # med1 = self.in_med1(med0) # 256 x 512
#         # med2 = self.in_med2(med1) # 128 x 256
#
#         small0 = self.in_small0(x) # 128 x 256
#         small1 = self.in_small1(small0) # 128 x 256
#         small2 = self.in_small2(small1) # 64 x 128
#         attn = self.attn_small(small2) # --> Go to decs
#
#         enc3 = self.conv3(small2)
#         enc4 = self.conv4(enc3)
#         enc5 = self.conv5(enc4)
#
#         encr = self.resblock(enc5)
#
#         dec1 = self.dconv1(encr)
#         dec2 = self.dconv2(dec1)
#         dec3 = self.dconv3(dec2)
#         decs = self.dconvs(ops.dstack(dec3, attn))
#         outs = self.outs(decs)
#
#         return torch.tanh(outs)

class GS(nn.Module):
    def __init__(self):
        super(GS, self).__init__()
        activ = 'leaky'
        norm = 'instnorm'

        self.in_large0 = Block(3, 64, 3, 1, 1, activ, norm) # 512 x 1024
        self.in_large1 = Block(64, 128, 4, 2, 1, norm) # 256 x 512

        self.in_med0 = Block(3, 64, 3, 1, 1, activ, norm) # 256 x 512
        self.in_med1 = Block(64, 128, 3, 1, 1, activ, norm) # 256 x 512
        self.in_med2 = Block(128, 128, 4, 2, 1, norm) # 128 x 256

        self.in_small0 = Block(3, 64, 3, 1, 1, activ, norm) # 128 x 256
        self.in_small1 = Block(64, 128, 3, 1, 1, activ, norm) # 128 x 256
        self.in_small2 = Block(128, 128, 4, 2, 1, activ, norm) # 64 x 128
        self.attn_small = Attn(128)

        self.conv3 = Block(128, 256, 3, 2, 1, activ, norm) # 32 x 64
        self.conv4 = Block(256, 512, 3, 2, 1, activ, norm) # 16 x 32
        self.conv5 = Block(512, 1024, 3, 2, 1, activ, norm) # 8 x 16

        self.resblock = nn.Sequential(
           ResBlock(1024, activ, norm, 2),
           ResBlock(1024, activ, norm, 2),
           ResBlock(1024, activ, norm, 2),
           ResBlock(1024, activ, norm, 2),
           ResBlock(1024, activ, norm, 2),
           ResBlock(1024, activ, norm, 2),
        )
        self.dconv1 = BlockTrans(1024, 512, 4, 2, 1, activ, norm) # 16 x 32
        self.dconv2 = BlockTrans(512, 256, 4, 2, 1, activ, norm) # 32 x 64
        self.dconv3 = BlockTrans(256, 128, 4, 2, 1, activ, norm) # 64 x 128
        self.dconvs = BlockTrans(128+ 128, 64, 4, 2, 1, activ, norm) # 128 x 256
        self.dconvm = BlockTrans(64 + 128, 64, 4, 2, 1, activ, norm) # 256 x 512
        self.dconvl = BlockTrans(64 + 128, 64, 4, 2, 1, activ, norm) # 512 x 1024

        self.outs = BlockTrans(64, 3, 3, 1, 1)
        self.outm = BlockTrans(64, 3, 3, 1, 1)
        self.outl = BlockTrans(64, 3, 3, 1, 1)

    def forward(self, x):
        # xs = ops.downsample(x, 2)
        # med0 = self.in_med0(x) # 256 x 512
        # med1 = self.in_med1(med0) # 256 x 512
        # med2 = self.in_med2(med1) # 128 x 256

        small0 = self.in_small0(x) # 128 x 256
        small1 = self.in_small1(small0) # 128 x 256
        small2 = self.in_small2(small1) # 64 x 128
        attn = self.attn_small(small2) # --> Go to decs

        enc3 = self.conv3(small2)
        enc4 = self.conv4(enc3)
        enc5 = self.conv5(enc4)

        encr = self.resblock(enc5)

        dec1 = self.dconv1(encr)
        dec2 = self.dconv2(dec1)
        dec3 = self.dconv3(dec2)
        decs = self.dconvs(ops.dstack(dec3, attn))
        outs = self.outs(decs)

        return torch.tanh(outs)

class GM(nn.Module):
    def __init__(self):
        super(GM, self).__init__()
        activ = 'leaky'
        norm = 'instnorm'

        self.in_large0 = Block(3, 64, 3, 1, 1, activ, norm) # 512 x 1024
        self.in_large1 = Block(64, 128, 4, 2, 1, norm) # 256 x 512

        self.in_med0 = Block(3, 64, 3, 1, 1, activ, norm) # 256 x 512
        self.in_med1 = Block(64, 128, 3, 1, 1, activ, norm) # 256 x 512
        self.in_med2 = Block(128, 128, 4, 2, 1, norm) # 128 x 256

        self.in_small0 = Block(3, 64, 3, 1, 1, activ, norm) # 128 x 256
        self.in_small1 = Block(64, 128, 3, 1, 1, activ, norm) # 128 x 256
        self.in_small2 = Block(128, 128, 4, 2, 1, activ, norm) # 64 x 128

        self.conv3 = Block(128, 256, 3, 2, 1, activ, norm) # 32 x 64
        self.conv4 = Block(256, 512, 3, 2, 1, activ, norm) # 16 x 32
        self.conv5 = Block(512, 1024, 3, 2, 1, activ, norm) # 8 x 16

        self.resblock = nn.Sequential(
           ResBlock(1024, activ, norm, 2),
           ResBlock(1024, activ, norm, 2),
           ResBlock(1024, activ, norm, 2),
           ResBlock(1024, activ, norm, 2),
           ResBlock(1024, activ, norm, 2),
           ResBlock(1024, activ, norm, 2),
        )
        self.dconv1 = BlockTrans(1024, 512, 4, 2, 1, activ, norm) # 16 x 32
        self.dconv2 = BlockTrans(512, 256, 4, 2, 1, activ, norm) # 32 x 64
        self.dconv3 = BlockTrans(256, 128, 4, 2, 1, activ, norm) # 64 x 128
        self.dconvs = BlockTrans(128+ 128, 64, 4, 2, 1, activ, norm) # 128 x 256
        self.dconvm = BlockTrans(64 + 128, 64, 4, 2, 1, activ, norm) # 256 x 512
        self.dconvl = BlockTrans(64 + 128, 64, 4, 2, 1, activ, norm) # 512 x 1024

        self.outs = BlockTrans(64, 3, 3, 1, 1)
        self.outm = BlockTrans(64, 3, 3, 1, 1)
        self.outl = BlockTrans(64, 3, 3, 1, 1)

    def forward(self, x):
        xs = ops.downsample(x, 2)
        med0 = self.in_med0(x) # 256 x 512
        med1 = self.in_med1(med0) # 256 x 512
        med2 = self.in_med2(med1) # 128 x 256

        small0 = self.in_small0(xs) # 128 x 256
        small1 = self.in_small1(small0) # 128 x 256
        small2 = self.in_small2(small1 + med2) # 64 x 128

        enc3 = self.conv3(small2)
        enc4 = self.conv4(enc3)
        enc5 = self.conv5(enc4)

        encr = self.resblock(enc5)

        dec1 = self.dconv1(encr)
        dec2 = self.dconv2(dec1)
        dec3 = self.dconv3(dec2)
        decs = self.dconvs(ops.dstack(dec3, small2))
        decm = self.dconvm(ops.dstack(decs, med2))
        outm = self.outm(decm)

        return outm, torch.tanh(outm)

class GL(nn.Module):
    def __init__(self):
        super(GL, self).__init__()
        activ = 'leaky'
        norm = 'instnorm'

        self.in_l0 = Block(3, 64, 3, 1, 1, activ, norm) # 512 x 1024
        # self.in_l1 = Block(64, 128, 4, 2, 1, norm) # 256 x 512
        self.in_l1 = Block(64, 128, 4, 2, 1, activ, norm) # 256 x 512

        self.in_med0 = Block(3, 64, 3, 1, 1, activ, norm) # 256 x 512
        self.in_med1 = Block(64, 128, 3, 1, 1, activ, norm) # 256 x 512
        self.in_med2 = Block(128, 128, 4, 2, 1, norm) # 128 x 256

        self.in_small0 = Block(3, 64, 3, 1, 1, activ, norm) # 128 x 256
        self.in_small1 = Block(64, 128, 3, 1, 1, activ, norm) # 128 x 256
        self.in_small2 = Block(128, 128, 4, 2, 1, activ, norm) # 64 x 128

        self.conv3 = Block(128, 256, 3, 2, 1, activ, norm) # 32 x 64
        self.conv4 = Block(256, 512, 3, 2, 1, activ, norm) # 16 x 32
        self.conv5 = Block(512, 1024, 3, 2, 1, activ, norm) # 8 x 16

        self.resblock = nn.Sequential(
           ResBlock(1024, activ, norm, 2),
           ResBlock(1024, activ, norm, 2),
           ResBlock(1024, activ, norm, 2),
           ResBlock(1024, activ, norm, 2),
           ResBlock(1024, activ, norm, 2),
           ResBlock(1024, activ, norm, 2),
        )
        self.dconv1 = BlockTrans(1024, 512, 4, 2, 1, activ, norm) # 16 x 32
        self.dconv2 = BlockTrans(512, 256, 4, 2, 1, activ, norm) # 32 x 64
        self.dconv3 = BlockTrans(256, 128, 4, 2, 1, activ, norm) # 64 x 128
        self.dconvs = BlockTrans(128 + 128, 64, 4, 2, 1, activ, norm) # 128 x 256
        self.dconvm = BlockTrans(64 + 128, 64, 4, 2, 1, activ, norm) # 256 x 512
        self.dconvl_ = BlockTrans(64 + 128, 64, 4, 2, 1, activ, norm) # 512 x 1024

        self.outs = BlockTrans(64, 3, 3, 1, 1)
        self.outm = BlockTrans(64, 3, 3, 1, 1)
        self.outl_ = BlockTrans(64, 3, 3, 1, 1)

    def forward(self, x):
        xs = ops.downsample(x, 4)
        xm = ops.downsample(x, 2)
        large0 = self.in_l0(x) # 512 x 1024
        large1 = self.in_l1(large0) # 256 x 512
        # attn_large1 = self.attn_large(large1)
        # print(attn_large1.size())

        med0 = self.in_med0(xm) # 256 x 512
        med1 = self.in_med1(med0) # 256 x 512
        med2 = self.in_med2(med1 + large1) # 128 x 256

        small0 = self.in_small0(xs) # 128 x 256
        small1 = self.in_small1(small0) # 128 x 256
        small2 = self.in_small2(small1 + med2) # 64 x 128

        enc3 = self.conv3(small2)
        enc4 = self.conv4(enc3)
        enc5 = self.conv5(enc4)

        encr = self.resblock(enc5)

        dec1 = self.dconv1(encr)
        dec2 = self.dconv2(dec1)
        dec3 = self.dconv3(dec2)
        decs = self.dconvs(ops.dstack(dec3, small2))
        decm = self.dconvm(ops.dstack(decs, med2))
        decl = self.dconvl_(ops.dstack(decm, large1))
        # outs = self.outs(decs)
        # outm = self.outm(decm) #+ ops.upsample(outs, 2)
        # outl = self.outl_(decl) + ops.upsample(outm,2)
        outl = self.outl_(decl)

        return torch.tanh(outl), torch.tanh(outl), torch.tanh(outl)


class GL2(nn.Module):
    def __init__(self):
        super(GL2, self).__init__()
        activ = 'leaky'
        norm = 'instnorm'

        self.in_l0 = Block(3, 64, 4, 2, 1, activ, norm) # 256 x 512
        self.in_l1 = Block(64, 128, 4, 2, 1, activ, norm) # 128 x 256
        self.in_l2 = Block(128, 128, 4, 2, 1, activ, norm) # 64 x 128
        self.in_l3 = Block(128, 128, 3, 1, 1) # 64 x 128

        self.in_med0 = Block(3, 64, 3, 1, 1, activ, norm) # 256 x 512
        self.in_med1 = Block(64, 128, 3, 1, 1, activ, norm) # 256 x 512
        self.in_med2 = Block(128, 128, 4, 2, 1, norm) # 128 x 256

        self.in_small0 = Block(3, 64, 3, 1, 1, activ, norm) # 128 x 256
        self.in_small1 = Block(64, 128, 3, 1, 1, activ, norm) # 128 x 256
        self.in_small2 = Block(128, 128, 4, 2, 1, activ, norm) # 64 x 128

        self.conv3 = Block(128, 256, 3, 2, 1, activ, norm) # 32 x 64
        self.conv4 = Block(256, 512, 3, 2, 1, activ, norm) # 16 x 32
        self.conv5 = Block(512, 1024, 3, 2, 1, activ, norm) # 8 x 16

        self.resblock = nn.Sequential(
           ResBlock(1024, activ, norm, 2),
           ResBlock(1024, activ, norm, 2),
           ResBlock(1024, activ, norm, 2),
           ResBlock(1024, activ, norm, 2),
           ResBlock(1024, activ, norm, 2),
           ResBlock(1024, activ, norm, 2),
        )
        self.dconv1 = BlockTrans(1024, 512, 4, 2, 1, activ, norm) # 16 x 32
        self.dconv2 = BlockTrans(512, 256, 4, 2, 1, activ, norm) # 32 x 64
        self.dconv3 = BlockTrans(256, 128, 4, 2, 1, activ, norm) # 64 x 128
        self.dconvs = BlockTrans(128 + 128, 64, 4, 2, 1, activ, norm) # 128 x 256
        self.dconvm = BlockTrans(64 + 128, 64, 4, 2, 1, activ, norm) # 256 x 512

        self.dconvl_0 = BlockTrans(128, 128, 4, 2, 1, activ, norm) # 128 x 256
        self.dconvl_1 = BlockTrans(128, 128, 4, 2, 1, activ, norm) # 256 x 512
        self.dconvl_ = BlockTrans(128 + 64, 64, 4, 2, 1, activ, norm) # 512 x 1024

        # self.outs = BlockTrans(64, 3, 3, 1, 1)
        self.outm = BlockTrans(64, 3, 3, 1, 1)
        self.outl_ = BlockTrans(64, 3, 3, 1, 1)

    def forward(self, x):
        xs = ops.downsample(x, 4)
        xm = ops.downsample(x, 2)
        large0 = self.in_l0(x) # 512 x 1024
        large1 = self.in_l1(large0) # 256 x 512
        large2 = self.in_l2(large1) # 256 x 512
        large3 = self.in_l3(large2) # 256 x 512
        # attn_large1 = self.attn_large(large1)
        # print(attn_large1.size())

        med0 = self.in_med0(xm) # 256 x 512
        med1 = self.in_med1(med0) # 256 x 512
        med2 = self.in_med2(med1) # 128 x 256

        small0 = self.in_small0(xs) # 128 x 256
        small1 = self.in_small1(small0) # 128 x 256
        small2 = self.in_small2(small1 + med2) # 64 x 128

        enc3 = self.conv3(small2)
        enc4 = self.conv4(enc3)
        enc5 = self.conv5(enc4)

        encr = self.resblock(enc5)

        dec1 = self.dconv1(encr)
        dec2 = self.dconv2(dec1)
        dec3 = self.dconv3(dec2)
        decs = self.dconvs(ops.dstack(dec3, small2))
        decm = self.dconvm(ops.dstack(decs, med2))

        decl = self.dconvl_0(dec3 + large3)
        decl = self.dconvl_1(decl)
        # decl = self.dconvl_(decl)
        decl = self.dconvl_(ops.dstack(decl, large0))
        # outs = self.outs(decs)
        outm = self.outm(decm) #+ ops.upsample(outs, 2)
        outl_ = self.outl_(decl)#
        outl = outl_  + ops.upsample(outm,2)

        return torch.tanh(outl_), torch.tanh(outm), torch.tanh(outl)

class Attn(nn.Module):
    def __init__(self, inch):
        super(Attn, self).__init__()
        self.inch = inch
        self.conv_theta = (nn.Conv2d(inch, inch//8, 1, 1, 0))
        self.conv_phi = (nn.Conv2d(inch, inch//8, 1, 1, 0))
        self.conv_g = (nn.Conv2d(inch, inch//2, 1, 1, 0))
        self.conv_attn = (nn.Conv2d(inch//2, inch, 1, 1, 0))
        self.maxpool = nn.MaxPool2d(2, 2, 0)
        self.softmax = nn.Softmax(dim=-1)
        self.sigma = nn.Parameter(torch.zeros(1))

    def forward(self, x):
        batch, ch, h, w  = x.size()

        theta = self.conv_theta(x)
        theta = theta.view(-1, ch//8, h*w)

        phi = self.conv_phi(x)
        phi = self.maxpool(phi)
        phi = phi.view(-1, ch//8, h*w//4)

        attn = torch.bmm(theta.permute(0,2,1), phi)
        attn = self.softmax(attn)

        g = self.conv_g(x)
        g = self.maxpool(g)
        g = g.view(-1, ch//2, h*w//4)

        attn_g = torch.bmm(g, attn.permute(0,2,1))
        attn_g = attn_g.view(-1, ch//2, h, w)
        attn_g = self.conv_attn(attn_g)
        out = x + self.sigma * attn_g
        return out
