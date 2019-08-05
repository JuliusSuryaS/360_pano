from util.base import *
import glob

class PanoData(Dataset):
    def __init__(self, root_dir, data_len, transform=None, data_path=None):
        self.root_dir = root_dir
        self.data_len = data_len - 1
        self.transform = transform

    def __getitem__(self, idx):

        sub_dir = 'pano_' + str(idx + 1)

        """Standard"""
        # ----------
        in_img_cat = self._read_pano(sub_dir, prefix='trained_input_gt.jpg', scale=1.0)
        gt_img_cat = self._read_pano(sub_dir, prefix='pano_*.jpg.jpg', scale=1.0) # pano groundtruth
        in_img_cat = gt_img_cat
        fov = self._read_fov(sub_dir, prefix='fov.txt')

        in_img_cat = in_img_cat/127.5 - 1
        gt_img_cat = gt_img_cat/127.5 - 1

        sample = {'input': in_img_cat, 'gt': gt_img_cat, 'fov': fov, 'dir': sub_dir}

        if self.transform:
            sample = self.transform(sample)

        return sample

    def __len__(self):
        return self.data_len

    def _concat_img_full(self, sub_dir, prefix='gt_'):
        images = []
        for i in range(6):
            im_name = prefix + str(i+1) + '.jpg'
            im_path = os.path.join(self.root_dir, sub_dir, im_name)
            images.append(self._imread(im_path))

        empty = np.zeros_like(images[0])
        img_top = np.hstack((empty, images[4], empty, empty))
        img_concat = np.hstack((images[2], images[0], images[3], images[1]))
        img_bot = np.hstack((empty, images[5], empty, empty))

        img_concat_full = np.vstack((img_top, img_concat, img_bot))
        return img_concat_full

    def _concat_img(self, sub_dir, prefix='new_img_'):
        images = []
        for i in range(6):
            im_name = prefix + str(i+1) + '.jpg'
            im_path = os.path.join(self.root_dir, sub_dir, im_name)
            images.append(self._imread(im_path))

        img_concat = np.hstack((images[2], images[0], images[3], images[1]))

        return img_concat

    def _concat_img_pad(self, sub_dir, prefix='im_'):
        images = []
        for i in range(4):
            im_name = prefix + str(i+1) + '.jpg'
            im_path = os.path.join(self.root_dir, sub_dir, im_name)
            images.append(self._imread_pad(im_path))

        img_concat = np.hstack((images[2], images[0], images[3], images[1]))
        img_concat_full = np.vstack((np.zeros_like(img_concat), img_concat, np.zeros_like(img_concat)))
        return img_concat_full

    def _read_fov(self, sub_dir, prefix='fov.txt'):
        out = np.zeros((1,128))
        file_path = os.path.join(self.root_dir, sub_dir, prefix)
        with open(file_path) as f:
            for line in f:
                idx = line.strip()
                idx = int(int(idx)/2)
                out[0, idx-1] = 1
        return idx

    def _read_rand_img(self, sub_dir, prefix='gt_'):
        images = []
        gts = []
        fov = self.generate_random_fov()

        for i in range(4):
            img_path = os.path.join(self.root_dir, sub_dir, prefix + str(i+1) + '.jpg')
            im = self._imread(img_path)
            images.append(self.generate_crop_img(im, fov))
            gts.append(self.generate_pad_img(im * 0 + 255, fov))

        img_concat = np.hstack((images[2], images[0], images[3], images[1]))
        gt_concat = np.hstack((gts[2], gts[0], gts[3], gts[1]))
        fov = int(fov/2) # downsample image twice
        return img_concat, gt_concat, fov

    def _read_pano(self, sub_dir, prefix='pano_', scale=1.0):
        im_path_ = os.path.join(self.root_dir, sub_dir, prefix)
        print(im_path_)
        im_list = glob.glob(im_path_)
        img = self._imread(im_list[0])
        img_rsz = cv2.resize(img, (0,0), fx=scale, fy=scale)
        return img_rsz

    def _imread(self, x):
        img = cv2.imread(x)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        return img

    def _imread_rsz(self,x):
        img = cv2.imread(x)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img = cv2.resize(img, (128, 128))
        return img

    def _imread_pad(self,x):
        img = self._imread(x)
        img = cv2.resize(img, (128, 128))
        pad = 64
        img = cv2.copyMakeBorder(img, 64, 64, 64, 64, cv2.BORDER_CONSTANT)
        return img

    def _edge_img(self, x, y):
        return

    def generate_random_fov(self):
        fov_range = np.arange(2,256,2)
        np.random.shuffle(fov_range)
        return fov_range[0]

    def generate_crop_img(self, img, fov):
        pad = int((256-fov)/2)
        x = img[pad:pad+fov,pad:pad+fov,:]
        x = cv2.resize(x,(256,256))
        return x

    def generate_pad_img(self, img, fov):
        pad = int((256-fov)/2)
        img = cv2.resize(img, (fov,fov))
        img = cv2.copyMakeBorder(img, pad, pad, pad, pad, cv2.BORDER_CONSTANT)
        return img
