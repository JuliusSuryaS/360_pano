from util.base import *
from model.models import *
from model.vgg import Vgg
import model.models2 as m2
import model.ops as ops
from util.opt import  Options
import util.utilities as utl

opt = Options()

class StructNet():
    def __init__(self, device):
        self.device = device
        self.net = m2.SemanticNet(3).to(device)
        self.net.apply(init_weights)
        self.loss_fn = torch.nn.CrossEntropyLoss()
        self.optimizer = optim.SGD(self.net.parameters(), lr=0.1, momentum=0.9)
        self.total_iter = 50

    def load_input_batch(self, in_img, gt_img):
        self.in_img = in_img
        self.lbl_ind = self.slic(gt_img)

    def slic(self, in_img:torch.Tensor):
        np_img = in_img.data.numpy()
        labels = ski.slic(np_img, compactness=1, n_segments=10000)
        labels = labels.reshape(np_img.shape[0] * np_img.shape[1])
        u_labels = np.unique(labels)
        l_inds = []
        for i in range(len(u_labels)):
            l_inds.append(np.where(labels == u_labels[i])[0])

    def train(self):
        for iter in range(self.total_iter):
            self.optimizer.zero_grad()
            out = self.net(self.in_img)
            out = out.permute(1,2,0).contiguous().view(-1, 100)
            ignore, target = torch.max(out, 1)
            im_target = target.data.cpu().numpy()
            nlabels = len(np.unique(im_target))

            for i in range(len(self.lbl_ind)):
                labels_per_sp = im_target[self.lbl_ind[i]]
                u_labels_per_sp = np.unique(labels_per_sp)
                hist = np.zeros(len(u_labels_per_sp))
                for j in range(len(hist)):
                    hist[j] = len(np.where(labels_per_sp == u_labels_per_sp[j])[0])
                im_target[self.lbl_ind[i]] = u_labels_per_sp[np.argmax(hist)]
            target = torch.from_numpy(im_target)
            target = target.to(self.device)

            self.loss = self.loss_fn(out, target)
            self.loss.backward()
            self.optimizer.step()
        return target

