from util.base import *
from skimage.measure import compare_psnr, compare_mse, compare_ssim

def to_numpy(x):
    x = x.cpu()
    x = ((x.detach().numpy() + 1) / 2)
    x = np.transpose(x[0,:,:,:], (1,2,0))
    return x

def imresize_pad(x, sz, pad):
    img = cv2.resize(x, (sz,sz))
    img = cv2.copyMakeBorder(img, pad, pad, pad, pad, cv2.BORDER_CONSTANT)
    return img

def numpy_to_pano(in_img, out_h=512, out_w=1024, in_len=256):
    out_img = np.zeros((out_h, out_w, 3), dtype=np.uint8)
    coord = np.load('/home/juliussurya/workspace/360pano2/data/pano_coord_1024.npy')
    # coord = np.load('/home/juliussurya/workspace/360pano2/data/pano_coord_512.npy')

    y = coord[:, :, 0]  # load the coordinate
    x = coord[:, :, 1]  # load the coordinate
    out_img = in_img[y, x, :]
    return out_img

def min_visualize(x:torch.Tensor, map='rgb'):
    x = x.cpu()
    img = (x.detach().numpy() + 1) / 2
    if map == 'gray':
        img = np.transpose(img[0,:,:,:], (1,2,0))
        cv2.imshow('img',img)
    else:
        img = np.transpose(img[0,:,:,:], (1,2,0))
        plt.imshow(img)
        plt.show()

def calc_l1(im1, im2):
    out = np.abs(im1 - im2)
    out = np.mean(np.mean(out))
    return out

def calc_quanti(im1, im2):
    # im1 = (to_numpy(im1) * 1)#.astype(np.uint8)
    # im2 = (to_numpy(im2) * 1)#.astype(np.uint8)
    mse =  compare_mse(im1, im2)
    ssim = compare_ssim(im1, im2, multichannel=True, data_range=im2.max()-im2.min())
    psnr = compare_psnr(im2, im1, data_range=im2.max()-im2.min())
    return mse, psnr, ssim

def save_img(im, name):
    im = (to_numpy(im) * 255).astype(np.uint8)
    im = cv2.cvtColor(im, cv2.COLOR_RGB2BGR)
    cv2.imwrite(name, im)

def compute_sobel(im):
    kx = torch.tensor([[-1, 0, 1],[-2, 0, 2],[-1, 0, 1]])
    ky = torch.tensor([[-1, -2, -1],[0, 0, 0],[1, 2, 1]])
    kx = kx.expand((1,3,3,3)).type(torch.FloatTensor)
    ky = ky.expand((1,3,3,3)).type(torch.FloatTensor)

    outx = F.conv2d(im, kx, stride=1, padding=1)
    outy = F.conv2d(im, ky, stride=1, padding=1)
    return outx, outy


def cube_to_equirect(img, coord_src, device=None):
    batch_size, channels, height, width = list(img.size())
    height = int(width/2)

    # Reverse image channels  and flatten
    img = img.permute(1,0,2,3).contiguous()
    img_flat = img.view(channels, -1)

    # Get panorama coordinate source
    coord = np.load(coord_src)
    coord_x = torch.from_numpy(coord[:,:,1])
    coord_y = torch.from_numpy(coord[:,:,0])
    x = coord_x.view(-1).repeat(1, batch_size).long()
    y = coord_y.view(-1).repeat(1, batch_size).long() * width
    pano_coord = x + y

    # Warp image
    out_img_flat = img_flat.gather(1, pano_coord.repeat(channels, 1).long())

    # Reshape back and transpose the dimension
    out_img = out_img_flat.view(channels, batch_size, height, width).permute(1,0,2,3)

    return out_img

def to_one_hot(labels, num_class, device):
    # labels N X 1 X H X W --> softmax --> argmax(t, dim, keepdim=True)
    temp = torch.FLoatTensor(labels.size(0), num_class, labels.size(2), labels.size(3)).zero_()
    one_hot = temp.scatter_(1, labels.data, 1)
    return one_hot.to(device)


def colorize(out_segment, num_class, color_map=None):
    if color_map == None:
        colors = [[68, 73, 194],
              [127, 50, 126],
              [32, 148, 2],
              [37, 25, 131],
              [148, 114, 76],
              [135, 159, 254],
              [68, 118, 101],
              [217, 191, 81],
              [190, 102, 83],
              [208, 242, 121],
              [117, 106, 89],
              [28, 69, 185],
              [32, 241, 10],
              [238, 232, 203],
              [140, 229, 135],
              [128, 216, 231],
              [185, 255, 196],
              [73, 116, 106],
              [238, 209, 169],
              [114, 107, 223],
              [241, 51, 123]]

        segment = torch.argmax(out_segment, dim=1, keepdim=True)
        out_r = torch.zeros_like(segment)
        out_g = torch.zeros_like(segment)
        out_b = torch.zeros_like(segment)

        for i in range(len(colors)):
            out_r[segment==i] = colors[i][0]
            out_g[segment==i] = colors[i][1]
            out_b[segment==i] = colors[i][2]

        out = torch.cat((out_r, out_g, out_b), dim=1)

        return out


def resize_pad_tensor(x, size, pad, mode='bilinear'):
    out = F.interpolate(x, size=size, mode=mode)
    out_pad = F.pad(out, (pad, pad, pad, pad), 'constant', 0)
    return out_pad

def proc_fov_out(im_list, size, pad, equi_coord):
    im1 = resize_pad_tensor(im_list[0], size, pad)
    im2 = resize_pad_tensor(im_list[1], size, pad)
    im3 = resize_pad_tensor(im_list[2], size, pad)
    im4 = resize_pad_tensor(im_list[3], size, pad)

    im = torch.cat((im1, im2, im3, im4), -1)
    im_equi = cube_to_equirect(im, equi_coord)
    return im_equi

def create_mask_portion(size=[32,256]):
    hsize = size[0] * 4
    wsize = size[1] * 4
    pad = int((128 - size[0]) / 2)
    ones = torch.ones(hsize,wsize)
    out = F.pad(ones, (0, 0, pad * 4, pad * 4), 'constant', 0)
    return out

""" Create unaligned input mask """
def create_mask_ul(min=128, max=220):
    mlist = []
    flag = np.random.randint(3)

    # size are same, vertical position are different
    if flag == 0:
        horiz = np.random.randint(min, max)
        vert = np.random.randint(min, max)
        starth = 256 - horiz
        startv = 256 - vert
        sh = np.random.randint(0, starth)

        for i in range(4):
            sv = np.random.randint(0, startv)
            m = np.zeros((512,256))
            m[sv+128:sv+vert+128, sh:sh+horiz] = 1
            mlist.append(m)

    # size and position are different
    else:
        for i in range(4):
            horiz = np.random.randint(min, max)
            vert = np.random.randint(min, max)
            starth = 256 - horiz
            startv = 256 - vert
            sh = np.random.randint(0, starth)
            sv = np.random.randint(0, startv)

            m = np.zeros((512,256))
            m[sv+128:sv+vert+128, sh:sh+horiz] = 1
            mlist.append(m)

    mask = np.hstack((mlist[0], mlist[1], mlist[2], mlist[3]))
    mask = torch.from_numpy(mask)
    return mask.type(torch.FloatTensor)

def create_mask_outpaint():
    starth = 256 - 128
    startv = 128 - 64
    # sh = np.random.randint(0, starth)
    # sv = np.random.randint(0, startv)
    sh = 63
    sv = 31

    m = np.zeros((128, 256))
    m[sv:sv+64, sh:sh+128] = 1
    mask = torch.from_numpy(m)
    return mask.type(torch.FloatTensor)

def create_mask_rand(sz=(128,256)):
    hh = int(sz[1]/2)
    hv = int(sz[0]/2)
    starth = sz[1] -  int(sz[1]/2)
    startv = sz[0] -  int(sz[0]/2)
    sh = np.random.randint(0, starth)
    sv = np.random.randint(0, startv)

    m = np.zeros(sz)
    m[sv:sv+hv, sh:sh+hh] = 1
    mask = torch.from_numpy(m)
    return mask.type(torch.FloatTensor)

def create_mask_inp(size=256):
    m = np.zeros((512,1024)) + 1
    starth = 1024 - size
    startv = 512 - size
    sh = np.random.randint(0, starth)
    sv = np.random.randint(0, startv)
    m[sv:sv+size, sh:sh+size] = 0

    mask = torch.from_numpy(m)
    return mask.type(torch.FloatTensor)


