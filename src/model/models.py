from util.base import *
import model.ops as ops
from util.opt import Options

opt = Options(sys.argv[0])

"""
 VGG Network Loss
 ======================
"""
class Vgg19(torch.nn.Module):
    def __init__(self, requires_grad=False):
        super(Vgg19, self).__init__()
        vgg_pretrained_features = models.vgg19(pretrained=True).features
        self.slice1 = torch.nn.Sequential()
        self.slice2 = torch.nn.Sequential()
        self.slice3 = torch.nn.Sequential()
        self.slice4 = torch.nn.Sequential()
        self.slice5 = torch.nn.Sequential()
        for x in range(2):
            self.slice1.add_module(str(x), vgg_pretrained_features[x])
        for x in range(2, 7):
            self.slice2.add_module(str(x), vgg_pretrained_features[x])
        for x in range(7, 12):
            self.slice3.add_module(str(x), vgg_pretrained_features[x])
        for x in range(12, 21):
            self.slice4.add_module(str(x), vgg_pretrained_features[x])
        for x in range(21, 30):
            self.slice5.add_module(str(x), vgg_pretrained_features[x])
        if not requires_grad:
            for param in self.parameters():
                param.requires_grad = False

    def forward(self, X):
        h_relu1 = self.slice1(X)
        h_relu2 = self.slice2(h_relu1)
        h_relu3 = self.slice3(h_relu2)
        h_relu4 = self.slice4(h_relu3)
        h_relu5 = self.slice5(h_relu4)
        out = [h_relu1, h_relu2, h_relu3, h_relu4, h_relu5]
        return out

class VGGLoss(nn.Module):
    def __init__(self):
        super(VGGLoss, self).__init__()
        self.vgg = Vgg19().cuda()
        self.criterion = nn.L1Loss()
        self.weights = [1.0/32, 1.0/16, 1.0/8, 1.0/4, 1.0]

    def forward(self, x, y):
        x_vgg, y_vgg = self.vgg(x), self.vgg(y)
        loss = 0
        for i in range(len(x_vgg)):
            loss += self.weights[i] * self.criterion(x_vgg[i], y_vgg[i].detach())
        return loss

class VGG16(nn.Module):
    def __init__(self):
        super(VGG16, self).__init__()
        vgg_net = models.vgg16(pretrained=False).features
        weights = torch.load(opt.model_path + '/vgg16-397923af.pth')
        vgg_net.load_state_dict(weights, strict=False)
        self.feat_layers = vgg_net
        self.class_layers = nn.Sequential(
            nn.Linear(512 * 4 * 16, 4096),
            nn.ReLU(True),
            nn.Linear(4096, 2048),
            nn.ReLU(True),
            nn.Linear(2048, 128),
            # nn.Sigmoid(),
        )

    def forward(self, x):
        out = self.feat_layers(x)
        out = out.view(-1, 512 * 4 * 16)
        out = self.class_layers(out)
        return out

"""
 Weight Initialization
 ======================
"""
def init_weights(net):
    class_name = net.__class__.__name__
    if class_name.find('Conv') != -1:
        net.weight.data.normal_(0.0,0.02)
    elif class_name.find('BatchNorm2d') != -1:
        net.weight.data.normal_(1.0, 0.02)
        net.bias.data.fill_(0)
