from util.base import *
from util.opt import Options
from util.utilities import cube_to_equirect
from dataset.panodata import PanoData

opt = Options()
class Dataset():
    def __init__(self, phase, resize=False):
        print('Initiating dataset....', end='')

        # Init Transform function
        if resize == True:
            transform_fn = transforms.Compose([Resize(),ToTensor()])
        else:
            transform_fn = transforms.Compose([ToTensor()])


        # Select Phase
        if phase == 'test':
            data_len = opt.test_len
            dataset = PanoData(opt.test_path, data_len, transform=transform_fn)
            self.data_loader = DataLoader(dataset,
                                          batch_size=opt.test_batch,
                                          shuffle=False,
                                          num_workers=opt.workers)
        elif phase == 'val':
            data_len = opt.val_len
            dataset = PanoData(opt.val_path, data_len, transform=transform_fn)
            self.data_loader = dataset

        elif phase == 'train':
            data_len = opt.train_len
            dataset = PanoData(opt.train_path, data_len, transform=transform_fn)
            self.data_loader = DataLoader(dataset,
                                          batch_size=opt.train_batch,
                                          shuffle=opt.train_shuffle,
                                          num_workers=opt.workers)

        print('completed ...' + phase)

    def load_data(self):
        return self.data_loader


class ToTensor():
    def __call__(self, sample):
        gt_img, in_img, fov, sub_dir = sample['gt'], sample['input'], sample['fov'], sample['dir']

        gt = torch.from_numpy(gt_img.transpose(2,0,1)) # NCHW
        image = torch.from_numpy(in_img.transpose(2,0,1)) # NCHW
        # fov = torch.from_numpy(fov)

        gt = gt.type(torch.FloatTensor)
        image = image.type(torch.FloatTensor)
        # fov = fov.type(torch.LongTensor)

        return {'input': image, 'gt': gt, 'fov': fov, 'dir': sub_dir}

class Resize():
    def __call__(self, sample):
        scale = opt.resize_scale
        imw = int(opt.imw/scale)
        imh = int(opt.imh/scale)
        gt_img, in_img = sample['gt'], sample['input']
        gt_img = cv2.resize(gt_img,(imw, imh))
        in_img = cv2.resize(in_img,(imw, imh))

        return {'input': in_img, 'gt': gt_img}

class ToEqui():
    def __call__(self, sample):
        gt_img, in_img = sample['gt'], sample['input']
        gt_img_equi = cube_to_equirect(gt_img, opt.equi_coord)
        in_img_equi = cube_to_equirect(in_img, opt.equi_coord)

        return {'input': in_img_equi, 'gt': gt_img_equi}
