from util.base import *
from model.models import *
from model.vgg import Vgg
import model.models3 as m3
import model.ops as ops
from util.opt import  Options
import util.utilities as utl

opt = Options()

class Networks():
    def __init__(self, device, net_type, sobel=False):
        print('Initiating network......', end='')
        if net_type == 'small':
            self.prefix = 'small'
            self.Discriminator = m3.D1(6).to(device)
            self.Generator = m3.GS().to(device)
            # load initial weights
            self.Discriminator.apply(init_weights)
            self.Generator.apply(init_weights)

        elif net_type == 'medium':
            self.prefix = 'medium'
            self.Discriminator = m3.D1(6).to(device)
            self.Generator = m3.GM().to(device)
            self.Discriminator.apply(init_weights)
            self.Generator.apply(init_weights)

        elif net_type == 'large':
            self.prefix = 'large'
            self.Discriminator = m3.D1(6).to(device)
            self.Generator = m3.GL2().to(device)
            self.Discriminator.apply(init_weights)
            self.Generator.apply(init_weights)

        if sobel == True:
            self.use_sobel = True
            self.Sobel = Sobel().to(device)
        else:
            self.use_sobel = False

        # Network data
        self.device = device
        self.in_img = None
        self.gt_img = None

        # Optimizer
        self.optim_g = optim.Adam(self.Generator.parameters(),
                                  lr=opt.learn_rate,
                                  betas=(opt.beta1, opt.beta2))
        self.optim_d = optim.Adam(self.Discriminator.parameters(),
                                  lr=opt.learn_rate,
                                  betas=(opt.beta1, opt.beta2))

        # Network parameters
        self.phase = 'train'
        self.restore = False

        # Summary and Optimizer
        self.writer = SummaryWriter(opt.train_log)

        self.loss_fn_BCE = nn.BCELoss()
        self.loss_fn_MSE = nn.MSELoss()
        self.loss_fn_L1 = nn.L1Loss()
        self.loss_fn_feat = nn.L1Loss()
        self.loss_vgg = VGGLoss()

        # Network output
        self.d_out_fake = None
        self.d_out_real = None
        self.d_out_adv = None
        self.g_out = None
        self.g_out_sobel = None
        self.gt_sobel = None

        print('completed')

    def load_input_batch(self, in_img, gt_img, fov):
        self.in_img = in_img
        self.gt_img = gt_img

        self.preprocess_input()

        self.in_img_s = ops.downsample(self.in_img, 4)
        self.gt_img_s = ops.downsample(self.gt_img, 4)
        self.in_img_m = ops.downsample(self.in_img, 2)
        self.gt_img_m = ops.downsample(self.gt_img, 2)


    def train_small(self):
        # ----------------------
        # Train discriminator
        # ----------------------
        self.Discriminator.zero_grad()
        self.Generator.zero_grad()

        self.g_out_s = self.Generator(self.in_img_s)

        self.d_out_real = self.Discriminator(ops.dstack(self.in_img_s, self.gt_img_s.detach()))
        self.d_out_fake = self.Discriminator(ops.dstack(self.in_img_s, self.g_out_s.detach()))

        # Loss
        self.loss_d_real = 0
        self.loss_d_fake = 0
        for i in range(len(self.d_out_real)):
            self.loss_d_real += self.loss_fn_MSE(self.d_out_real[i], torch.ones_like(self.d_out_real[i]))
            # self.loss_d_real += self.loss_fn_MSE(self.d_out_real[i][-1], torch.ones_like(self.d_out_real[i][-1]))
            self.loss_d_fake += self.loss_fn_MSE(self.d_out_fake[i], torch.zeros_like(self.d_out_fake[i]))
            # self.loss_d_fake += self.loss_fn_MSE(self.d_out_fake[i][-1], torch.zeros_like(self.d_out_fake[i][-1]))
        self.loss_d_total = (self.loss_d_real + self.loss_d_fake)

        # Optimize
        self.loss_d_total.backward()
        self.optim_d.step()

        # ----------------------
        # Train Generator
        # ----------------------
        self.d_out_adv = self.Discriminator(ops.dstack(self.in_img_s, self.g_out_s))

        # Loss
        self.loss_g_adv = 0
        self.loss_g_feats = 0
        for i in range(len(self.d_out_adv)):
            self.loss_g_adv += self.loss_fn_MSE(self.d_out_adv[i], torch.ones_like(self.d_out_real[i]))
            # self.loss_g_adv += self.loss_fn_MSE(self.d_out_adv[i][-1}, torch.ones_like(self.d_out_real[i][-1))
            # for j in range(3):
            #     self.loss_g_feats += 0.5 * 10 *self.loss_fn_feat(self.d_out_adv[i][j], self.d_out_real[i][j].detach())
        self.loss_g_pix = self.loss_fn_L1(self.g_out_s, self.gt_img_s)
        self.loss_g_vgg = self.loss_vgg(self.g_out_s, self.gt_img_s)
        self.loss_g_total = 1 * self.loss_g_adv + 1 * self.loss_g_pix + 10 * self.loss_g_vgg
        # self.loss_g_total = 1 * self.loss_g_adv +  1 * self.loss_g_pix + 10 * self.loss_g_vgg + self.loss_g_feats

        # Optimize
        self.loss_g_total.backward()
        self.optim_g.step()

    def train_medium(self):
        # ----------------------
        # Train discriminator
        # ----------------------
        self.Discriminator.zero_grad()
        self.Generator.zero_grad()

        self.g_out_s, self.g_out_m = self.Generator(self.in_img_m)

        self.d_out_real = self.Discriminator(ops.dstack(self.in_img_m, self.gt_img_m.detach()))
        self.d_out_fake = self.Discriminator(ops.dstack(self.in_img_m, self.g_out_m.detach()))

        # Loss
        self.loss_d_real = 0
        self.loss_d_fake = 0
        for i in range(len(self.d_out_real)):
            self.loss_d_real += self.loss_fn_MSE(self.d_out_real[i], torch.ones_like(self.d_out_real[i]))
            # self.loss_d_real += self.loss_fn_MSE(self.d_out_real[i][-1], torch.ones_like(self.d_out_real[i][-1]))
            self.loss_d_fake += self.loss_fn_MSE(self.d_out_fake[i], torch.zeros_like(self.d_out_fake[i]))
            # self.loss_d_fake += self.loss_fn_MSE(self.d_out_fake[i][-1], torch.zeros_like(self.d_out_fake[i][-1]))
        self.loss_d_total = (self.loss_d_real + self.loss_d_fake)

        # Optimize
        self.loss_d_total.backward()
        self.optim_d.step()

        # ----------------------
        # Train Generator
        # ----------------------
        self.d_out_adv = self.Discriminator(ops.dstack(self.in_img_m, self.g_out_m))

        # Loss
        self.loss_g_adv = 0
        self.loss_g_feats = 0
        for i in range(len(self.d_out_adv)):
            self.loss_g_adv += self.loss_fn_MSE(self.d_out_adv[i], torch.ones_like(self.d_out_real[i]))
            # self.loss_g_adv += self.loss_fn_MSE(self.d_out_adv[i][-1}, torch.ones_like(self.d_out_real[i][-1))
            # for j in range(3):
            #     self.loss_g_feats += 0.5 * 10 *self.loss_fn_feat(self.d_out_adv[i][j], self.d_out_real[i][j].detach())
        self.loss_g_pix = self.loss_fn_L1(self.g_out_m, self.gt_img_m)
        self.loss_g_vgg = self.loss_vgg(self.g_out_m, self.gt_img_m)
        self.loss_g_total = 1 * self.loss_g_adv +  1 * self.loss_g_pix + 10 * self.loss_g_vgg
        # self.loss_g_total = 1 * self.loss_g_adv +  1 * self.loss_g_pix + 10 * self.loss_g_vgg + self.loss_g_feats

        # Optimize
        self.loss_g_total.backward()
        self.optim_g.step()


    def train_large(self):
        # ----------------------
        # Train discriminator
        # ----------------------
        self.Discriminator.zero_grad()
        self.Generator.zero_grad()

        self.g_out_s, self.g_out_m, self.g_out_l = self.Generator(self.in_img)

        self.d_out_real = self.Discriminator(ops.dstack(self.in_img, self.gt_img.detach()))
        self.d_out_fake = self.Discriminator(ops.dstack(self.in_img, self.g_out_l.detach()))

        # Loss
        self.loss_d_real = 0
        self.loss_d_fake = 0
        for i in range(len(self.d_out_real)):
            self.loss_d_real += self.loss_fn_MSE(self.d_out_real[i], torch.ones_like(self.d_out_real[i]))
            # self.loss_d_real += self.loss_fn_MSE(self.d_out_real[i][-1], torch.ones_like(self.d_out_real[i][-1]))
            self.loss_d_fake += self.loss_fn_MSE(self.d_out_fake[i], torch.zeros_like(self.d_out_fake[i]))
            # self.loss_d_fake += self.loss_fn_MSE(self.d_out_fake[i][-1], torch.zeros_like(self.d_out_fake[i][-1]))
        self.loss_d_total = (self.loss_d_real + self.loss_d_fake)

        # Optimize
        self.loss_d_total.backward()
        self.optim_d.step()

        # ----------------------
        # Train Generator
        # ----------------------
        self.d_out_adv = self.Discriminator(ops.dstack(self.in_img, self.g_out_l))

        # Loss
        self.loss_g_adv = 0
        self.loss_g_feats = 0
        for i in range(len(self.d_out_adv)):
            self.loss_g_adv += self.loss_fn_MSE(self.d_out_adv[i], torch.ones_like(self.d_out_real[i]))
            # self.loss_g_adv += self.loss_fn_MSE(self.d_out_adv[i][-1], torch.ones_like(self.d_out_real[i][-1]))
            # for j in range(3):
            #     self.loss_g_feats += 0.5 * 10 *self.loss_fn_feat(self.d_out_adv[i][j], self.d_out_real[i][j].detach())
        self.loss_g_pix = self.loss_fn_L1(self.g_out_l, self.gt_img)
        self.loss_g_vgg = self.loss_vgg(self.g_out_l, self.gt_img)
        self.loss_g_total = 1 * self.loss_g_adv + 10 * self.loss_g_pix + 10 * self.loss_g_vgg
        # self.loss_g_total = 1 * self.loss_g_adv +  1 * self.loss_g_pix + 10 * self.loss_g_vgg + self.loss_g_feats

        # Optimize
        self.loss_g_total.backward()
        self.optim_g.step()

    """
        Saving and loading Model
        ========================
    """
    def save_model(self, step, model_name='model'):
        print('Saving model at step ' + str(step))
        model_path = os.path.join(opt.model_path, model_name + '_' + str(step))
        torch.save({'Generator': self.Generator.state_dict(),
                    'Discriminator' : self.Discriminator.state_dict()},
                   model_path + '.pt')
        print('Finished saving model')

    def load_model(self, model_name, strict=True):
        print('Loading trained model')
        model_path = os.path.join(opt.model_path, model_name + '.pt')
        model = torch.load(model_path)
        self.Discriminator.load_state_dict(model['Discriminator'], strict=strict)
        self.Generator.load_state_dict(model['Generator'], strict=strict)
        print('Finished loading trained model')

    def load_model_G(self, model_name, strict=True):
        print('Loading trained model')
        model_path = os.path.join(opt.model_path, model_name + '.pt')
        model = torch.load(model_path)
        self.Generator.load_state_dict(model['Generator'], strict=strict)
        print('Finished loading trained model')


    """
        Training Summary
        ================
    """
    def write_imgs_summary_small(self, step):
        g_out_s =self.g_out_s[0,:,:,:]
        in_img_s =self.in_img_s[0,:,:,:]
        gt_img_s =self.gt_img_s[0,:,:,:]
        self.writer.add_image('out/output_small', (g_out_s+1)/2,  step)
        self.writer.add_image('in/input_small', (in_img_s+1)/2,  step)
        self.writer.add_image('in/gt_small', (gt_img_s+1)/2,  step)

    def write_scalars(self, step):
        self.writer.add_scalars('GAN Loss', {'Dreal': self.loss_d_real,
                                             'Dfake': self.loss_d_fake,
                                             'Dadv': self.loss_g_adv}, step)

    def write_imgs_summary_medium(self, step):
        g_out_m = self.g_out_m[0,:,:,:]
        g_out_s = torch.tanh(self.g_out_s[0,:,:,:])
        in_img_m = self.in_img_m[0,:,:,:]
        gt_img_m = self.gt_img_m[0,:,:,:]
        self.writer.add_image('out/output_small', (g_out_s.squeeze(0)+1)/2,  step)
        self.writer.add_image('out/output_medium', (g_out_m.squeeze(0)+1)/2,  step)
        self.writer.add_image('in/input_medium', (in_img_m.squeeze(0)+1)/2,  step)
        self.writer.add_image('in/gt_medium', (gt_img_m.squeeze(0)+1)/2,  step)

    def write_imgs_summary_large(self, step):
        g_out_s = self.g_out_s[0,:,:,:]
        g_out_m = self.g_out_m[0,:,:,:]
        g_out_l = self.g_out_l[0,:,:,:]
        in_img = self.in_img[0,:,:,:]
        gt_img = self.gt_img[0,:,:,:]
        # Small
        self.writer.add_image('out/output_small', (g_out_s.squeeze(0)+1)/2,  step)
        # Medium
        self.writer.add_image('out/output_medium', (g_out_m.squeeze(0)+1)/2,  step)
        # Large
        self.writer.add_image('out/output_large', (g_out_l.squeeze(0)+1)/2,  step)
        self.writer.add_image('in/input_large', (in_img.squeeze(0)+1)/2,  step)
        self.writer.add_image('in/gt_large', (gt_img.squeeze(0)+1)/2,  step)


    def write_img_summary(self, input, name, step):
        self.writer.add_image(name, (input.squeeze(0)+1)/2, step)

    def write_scalar_summary(self, tag, step):
        self.writer.add_scalar(tag, self.loss_g_pix, step)

    def print_summary_small(self, epoch, step):
        real = 0
        fake = 0
        real = torch.mean(self.d_out_real[0][-1])
        fake = torch.mean(self.d_out_adv[0][-1])
        print('Epoch %d [%d | %d] > G : %.5f | D : %.5f | Real: %.2f Fake: %.2f'
              %(epoch, step, opt.train_len, self.loss_g_total, self.loss_d_total,
                real.item(), fake.item()))

    def print_summary_medium(self, epoch, step):
        real = 0
        fake = 0
        for i in range(2):
            # compute mean of pixel
            real += torch.mean(self.d_out_real[i])
            # real += torch.mean(self.d_out_real[i][-1])
            fake += torch.mean(self.d_out_adv[i])
            # fake += torch.mean(self.d_out_adv[i][-1])
        # Mean of all resolution
        real = real / 3.0
        fake = fake / 3.0
        print('Epoch %d [%d | %d] > G : %.5f | D : %.5f | Real: %.2f Fake: %.2f'
              %(epoch, step, opt.train_len, self.loss_g_total, self.loss_d_total,
                real.item(), fake.item()))

    """
        Network related function
        ========================
    """
    def print_structure(self):
        print(self.Generator)
        print(self.Discriminator)
        print('Phase : ' + self.phase)

    def clear_gradient(self):
        self.Discriminator.zero_grad()
        self.Generator.zero_grad()

    def preprocess_input(self):
        self.gt_img = self.gt_img.to(self.device)
        self.in_img = self.in_img.to(self.device)

    def set_phase(self, phase='train'):
        if phase == 'test':
            print('Network phase : Test')
            self.phase = 'test'
            self.Generator.eval()
            self.Discriminator.eval()
        else:
            print('Network phase : Train')
            self.Generator.train()
            self.Discriminator.train()

