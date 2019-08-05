from util.base import *

class Vgg(nn.Module):
    def __init__(self):
        super(Vgg, self).__init__()
        vgg_model = models.vgg19(pretrained=True).features
        self.slice1 = torch.nn.Sequential()
        self.slice2 = torch.nn.Sequential()
        # self.slice3 = torch.nn.Sequential()
        # self.slice4 = torch.nn.Sequential()
        # self.slice5 = torch.nn.Sequential()
        for x in range(2):
            self.slice1.add_module(str(x), vgg_model[x])
        for x in range(2, 7):
            self.slice2.add_module(str(x), vgg_model[x])
        # for x in range(7, 12):
        #     self.slice3.add_module(str(x), vgg_model[x])
        # for x in range(12, 21):
        #     self.slice4.add_module(str(x), vgg_model[x])
        # for x in range(21, 30):
        #     self.slice5.add_module(str(x), vgg_model[x])

        for param in self.parameters():
            param.requires_grad = False

    def forward(self, x):
        h_relu1 = self.slice1(x)
        h_relu2 = self.slice2(h_relu1)
        # h_relu3 = self.slice3(h_relu2)
        # h_relu4 = self.slice4(h_relu3)
        # h_relu5 = self.slice5(h_relu4)
        # out = [h_relu1, h_relu2, h_relu3, h_relu4, h_relu5]
        out = [h_relu1, h_relu2]
        return out
