from util.base import *

"""
 Functional
 ===========
    for utilities operation directly on tensor 
"""
def dstack(x,y):
    out = torch.cat((x,y), dim=1)
    return out

def downsample(x, scale=2):
    scale = 1/scale
    out = F.interpolate(x, scale_factor=scale, mode='bilinear')
    return out

def upsample(x, scale=2, mode='linear'):
    out = F.interpolate(x, scale_factor=scale)
    return out

def sobel_conv(x, device):
    kx = torch.tensor([[-1, 0, 1],[-2, 0, 2],[-1, 0, 1]])
    ky = torch.tensor([[-1, -2, -1],[0, 0, 0],[1, 2, 1]])
    kx = kx.expand((1,3,3,3)).type(torch.FloatTensor).to(device)
    ky = ky.expand((1,3,3,3)).type(torch.FloatTensor).to(device)
    outx = F.conv2d(x,kx)
    outy = F.conv2d(x,ky)
    return outx, outy

"""
 Block operations
 =================
    for nn.Module class, mostly in Sequential for better
    flow control of network
"""
def conv(prev, out, ksz=3, s=2, pad=1, bias=True):
    block = nn.Sequential(nn.Conv2d(prev, out, ksz, s, pad, bias=bias))
    return block

def conv_relu(prev, out, ksz=3, s=2, pad=1, bias=True):
    block = nn.Sequential(
        nn.Conv2d(prev, out, ksz, s, pad, bias=bias),
        nn.ReLU(True)
    )
    return block

def conv_sn_relu(prev, out, ksz=3, s=2, pad=1, bias=True):
    block = nn.Sequential(
        spectral_norm(nn.Conv2d(prev, out, ksz, s, pad, bias=bias)),
        nn.ReLU(True)
    )
    return block

def conv_relu_drop(prev, out, ksz=3, s=2, pad=1, bias=True):
    block = nn.Sequential(
        nn.Conv2d(prev, out, ksz, s, pad, bias=bias),
        nn.ReLU(True),
        nn.Dropout(0.5),
    )
    return block

def conv_leak(prev, out, ksz=3, s=2, pad=1, bias=True):
    block = nn.Sequential(
        nn.Conv2d(prev, out, ksz, s, pad, bias),
        nn.LeakyReLU(0.2, inplace=True)
    )
    return block

def conv_sn_leak(prev, out, ksz=3, s=2, pad=1, bias=True):
    block = nn.Sequential(
        spectral_norm(nn.Conv2d(prev, out, ksz, s, pad, bias)),
        nn.LeakyReLU(0.2, inplace=True)
    )
    return block

def conv_sigmoid(prev, out, ksz=3, s=2, pad=1 ):
    block = nn.Sequential(
        nn.Conv2d(prev, out, ksz, s, pad),
        nn.Sigmoid()
    )
    return block

def conv_tanh(prev, out, ksz=3, s=2, pad= 1, bias=True):
    block = nn.Sequential(
        nn.Conv2d(prev, out, ksz, s, pad, bias=bias),
        nn.Tanh()
    )
    return block

def conv_selu(prev, out, ksz=3, s=2, pad=1, bias=True):
    block = nn.Sequential(
        nn.Conv2d(prev, out, ksz, s, pad),
        nn.SELU(inplace=True),
    )
    return block

def norm_lyr(feats, norm_type='batchnorm'):
    if norm_type == 'instnorm':
        return nn.InstanceNorm2d(feats, affine=False)
    else:
        return nn.BatchNorm2d(feats, affine=True)

def conv_norm_relu(prev, out, ksz=3, s=2, pad=1, norm='instnorm', bias=True):
    block = nn.Sequential(
        nn.Conv2d(prev, out, ksz, s, pad, bias=bias),
        norm_lyr(out, norm),
        nn.ReLU(inplace=True)
    )
    return block

def conv_norm_leak(prev, out, ksz=3, s=2, pad=1, norm='instnorm', bias=True):
    block = nn.Sequential(
        nn.Conv2d(prev, out, ksz, s, pad, bias=bias),
        norm_lyr(out, norm),
        nn.LeakyReLU(0.2, inplace=True)
    )
    return block

def conv_norm_selu(prev, out, ksz=3, s=2, pad=1, norm='instnorm', bias=True):
    block = nn.Sequential(
        nn.Conv2d(prev, out, ksz, s, pad),
        norm_lyr(out, norm),
        nn.SELU(inplace=True),
    )
    return block

def convT(prev, out, ksz=3, s=2, pad=1, bias=True):
    block = nn.Sequential(nn.ConvTranspose2d(prev, out, ksz, s, pad, bias=bias))
    return block

def convT_tanh(prev, out, ksz=3, s=2, pad=1, bias=True):
    block = nn.Sequential(
        nn.ConvTranspose2d(prev, out, ksz, s, pad, bias=bias),
        nn.Tanh()
    )
    return block

def convT_relu(prev, out, ksz=3, s=2, pad=1, bias=True):
    block = nn.Sequential(
        nn.ConvTranspose2d(prev, out, ksz, s, pad, bias=bias),
        nn.ReLU(inplace=True)
    )
    return block

def convT_relu_drop(prev, out, ksz=3, s=2, pad=1, bias=True):
    block = nn.Sequential(
        nn.ConvTranspose2d(prev, out, ksz, s, pad, bias=bias),
        nn.ReLU(inplace=True),
        nn.Dropout(0.5),
    )
    return block

def convT_leak(prev, out, ksz=3, s=2, pad=1, bias=True):
    block = nn.Sequential(
        nn.ConvTranspose2d(prev, out, ksz, s, pad, bias=bias),
        nn.LeakyReLU(0.2, inplace=True)
    )
    return block

def convT_sn(prev, out, ksz=3, s=2, pad=1, bias=True):
    block = nn.Sequential(
        spectral_norm(nn.ConvTranspose2d(prev, out, ksz, s, pad, bias=bias)),
    )
    return block

def convT_sn_leak(prev, out, ksz=3, s=2, pad=1, bias=True):
    block = nn.Sequential(
        spectral_norm(nn.ConvTranspose2d(prev, out, ksz, s, pad, bias=bias)),
        nn.LeakyReLU(0.2, inplace=True)
    )
    return block

def convT_norm_leak(prev, out, ksz=3, s=2, pad=1, norm='instnorm', bias=True):
    block = nn.Sequential(
        nn.ConvTranspose2d(prev, out, ksz, s, pad, bias=bias),
        norm_lyr(out, norm),
        nn.LeakyReLU(0.2, inplace=True)
    )
    return block

def convT_norm_relu(prev, out, ksz=3, s=2, pad=1, norm='instnorm', bias=True):
    block = nn.Sequential(
        nn.ConvTranspose2d(prev, out, ksz, s, pad, bias=bias),
        norm_lyr(out, norm),
        nn.ReLU(inplace=True)
    )
    return block

def convT_selu(prev, out, ksz=3, s=2, pad=1, bias=True):
    block = nn.Sequential(
        nn.ConvTranspose2d(prev, out, ksz, s, pad, bias=bias),
        nn.SELU(True),
    )
    return block

def convT_norm_selu(prev, out, ksz=3, s=2, pad=1, norm='instnorm', bias=True):
    block = nn.Sequential(
        nn.ConvTranspose2d(prev, out, ksz, s, pad, bias=bias),
        norm_lyr(out, norm),
        nn.SELU(True),
    )
    return block

def l2normalize(v, eps=1e-4):
    return v/(v.norm() + eps)

class SpectralNorm(nn.Module):
  def __init__(self, module, name='weight', power_iterations=1):
    super(SpectralNorm, self).__init__()
    self.module = module
    self.name = name
    self.power_iterations = power_iterations
    if not self._made_params():
      self._make_params()

  def _update_u_v(self):
    u = getattr(self.module, self.name + "_u")
    v = getattr(self.module, self.name + "_v")
    w = getattr(self.module, self.name + "_bar")

    height = w.data.shape[0]
    _w = w.view(height, -1)
    for _ in range(self.power_iterations):
      v = l2normalize(torch.matmul(_w.t(), u))
      u = l2normalize(torch.matmul(_w, v))

    sigma = u.dot((_w).mv(v))
    setattr(self.module, self.name, w / sigma.expand_as(w))

  def _made_params(self):
    try:
      getattr(self.module, self.name + "_u")
      getattr(self.module, self.name + "_v")
      getattr(self.module, self.name + "_bar")
      return True
    except AttributeError:
      return False

  def _make_params(self):
    w = getattr(self.module, self.name)

    height = w.data.shape[0]
    width = w.view(height, -1).data.shape[1]

    u = Parameter(w.data.new(height).normal_(0, 1), requires_grad=False)
    v = Parameter(w.data.new(height).normal_(0, 1), requires_grad=False)
    u.data = l2normalize(u.data)
    v.data = l2normalize(v.data)
    w_bar = Parameter(w.data)

    del self.module._parameters[self.name]
    self.module.register_parameter(self.name + "_u", u)
    self.module.register_parameter(self.name + "_v", v)
    self.module.register_parameter(self.name + "_bar", w_bar)

  def forward(self, *args):
    self._update_u_v()
    return self.module.forward(*args)
