from util.base import *
from util.opt import Options
import util.utilities as utl
from util.utilities import min_visualize, calc_quanti, to_numpy, save_img, calc_l1
from dataset.dataset import Dataset
from model.Networks import Networks
from scipy.stats import gaussian_kde
import time

opt = Options(sys.argv[0])

device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

data_loader = Dataset('test',resize=opt.resize).load_data()


total_psnr = 0
total_mse = 0
count = 0

def read_img_to_tensor2(im_path, sz=(256,256)):
    im_ = cv2.imread(im_path)
    im_ = cv2.resize(im_, (256,256))
    im = cv2.cvtColor(im_, cv2.COLOR_BGR2RGB)
    im = im / 127.5 - 1
    tsr = torch.from_numpy(im.transpose(2,0,1))
    tsr = tsr.type(torch.FloatTensor)
    return torch.unsqueeze(tsr, 0), im_

def read_img_to_tensor(im_path):
    im = cv2.imread(im_path)
    im = cv2.cvtColor(im, cv2.COLOR_BGR2RGB)
    im = im / 127.5 - 1
    tsr = torch.from_numpy(im.transpose(2,0,1))
    tsr = tsr.type(torch.FloatTensor)
    return tsr

def write_output_single(folder_path, im_name='trained_input.jpg', model_path=None, net_type='large'):
    in_img = read_img_to_tensor(folder_path + '/' + im_name)
    in_img = torch.unsqueeze(in_img, 0)
    net = Networks(device, net_type=net_type)
    net.set_phase('test')
    # net.print_structure()
    net.load_model(model_path, False)
    start = time.time()
    net.load_input_batch(in_img, in_img, 0)

    if net_type == 'large':
        outs, outm, outl = net.Generator(net.in_img)
        end = time.time()
        print(end - start)
        im_name_splt = im_name.split('.')
        save_img(outs, folder_path + '/output'+ im_name_splt[0]+'_s.png')
        save_img(outm, folder_path + '/output'+ im_name_splt[0]+'_m.png')
        save_img(outl, folder_path + '/output'+ im_name_splt[0]+'_l.png')
        print('saving image')

    if net_type == 'medium':
        outs, outm = net.Generator(net.in_img_m)
        end = time.time()
        print(end - start)
        im_name_splt = im_name.split('.')
        save_img(outs, folder_path + '/output'+ im_name_splt[0]+'_s.png')
        save_img(outm, folder_path + '/output'+ im_name_splt[0]+'_m.png')
        print('saving image')


def write_output(output_path, model_path=None, net_type='small'):
    in_path = output_path[0]
    gt_path = output_path[1]
    out_path = output_path[2]
    net = Networks(device, net_type)
    # net.set_phase('test')
    net.print_structure()
    net.load_model_G(model_path, True)


    idx = 0
    for item in data_loader:
        mask = utl.create_mask_ul()
        mask = mask.type(torch.cuda.FloatTensor)
        in_img, gt_img = item['input'].to(device), item['gt'].to(device)
        net.load_input_batch(gt_img * mask, gt_img, 0)
        if net_type == 'small':
            # net.forward_small()
            g_out_s = net.Generator(net.in_img_s)
            print('saving image %d'% idx)
            save_img(net.in_img_s, os.path.join(in_path, 'input_' + str(idx) + '.png'))
            save_img(net.gt_img_s, os.path.join(gt_path, 'gt_' + str(idx) + '.png'))
            save_img(g_out_s, os.path.join(out_path, 'output_' + str(idx) + '.png'))
        elif net_type == 'medium':
            # net.forward_medium()
            g_out_s, g_out_m = net.Generator(net.in_img_m)
            print('saving image %d'% idx)
            save_img(net.in_img_m, os.path.join(in_path, 'input_' + str(idx) + '.png'))
            save_img(net.gt_img_m, os.path.join(gt_path, 'gt_' + str(idx) + '.png'))
            save_img(g_out_m, os.path.join(out_path, 'output_' + str(idx) + '.png'))
        elif net_type == 'large':
            g_out_s, g_out_m, g_out_l = net.Generator(net.in_img)
            # g_out_l = net.Generator(net.in_img)
            print('saving image %d'% idx)
            save_img(net.in_img, os.path.join(in_path, 'input_' + str(idx) + '.png'))
            # save_img(net.gt_img, os.path.join(gt_path, 'gt_' + str(idx) + '.png'))
            save_img(g_out_l, os.path.join(out_path, 'output_' + str(idx) + '.png'))
        idx += 1

def write_output_list(list_path):
    dir_idx = []
    fs = open(list_path, 'w')
    for item in data_loader:
        subdir = item['dir']
        dir_idx.append(subdir)
        fs.write(subdir[0] + '\n')
    fs.close()
    return dir_idx

def evaluate_output(folder_path, data_len):
    total_mse = []
    total_psnr = []
    total_ssim = []
    for i in range(0,data_len-2):
        print('Processing data %d/%d' %(i, data_len))
        gt_im_path = os.path.join(folder_path[0], 'gt',  'gt_' + str(i) + '.png')
        out_im_path = os.path.join(folder_path[1], 'output', 'output_' + str(i) + '.png')
        gt_im = cv2.imread(gt_im_path)
        out_im = cv2.imread(out_im_path)
        mse, psnr, ssim = calc_quanti(out_im, gt_im)
        print(mse, psnr, ssim)
        total_mse.append(mse)
        total_psnr.append(psnr)
        total_ssim.append(ssim)
    return total_mse, total_psnr, total_ssim

def evaluate_output_pix(folder_path, data_len):
    total_mse = []
    total_psnr = []
    total_ssim = []
    for i in range(1,data_len-1):
        print('Processing data %d/%d' %(i, data_len))
        gt_im_path = os.path.join(folder_path[0], 'gt',  'gt_' + str(i) + '.png')
        out_im_path = os.path.join(folder_path[1], 'output', 'output_' + str(i) + '.png')
        gt_im = cv2.imread(gt_im_path)
        out_im = cv2.imread(out_im_path)
        mse, psnr, ssim = calc_quanti(out_im, gt_im)
        print(mse, psnr, ssim)
        total_mse.append(mse)
        total_psnr.append(psnr)
        total_ssim.append(ssim)
    return total_mse, total_psnr, total_ssim

def evaluate_l1(folder_path, data_len):
    total_l1 = []
    for i in range(0,data_len - 2):
        print('Processing data %d/%d' %(i, data_len))
        gt_im_path = os.path.join(folder_path[0], 'gt',  'gt_' + str(i) + '.png')
        out_im_path = os.path.join(folder_path[1], 'output', 'output_' + str(i) + '.png')
        gt_im = cv2.imread(gt_im_path)
        out_im = cv2.imread(out_im_path)
        l1 = calc_l1(gt_im, out_im)
        print(l1)
        total_l1.append(l1)
    return total_l1

def evaluate_l1_pix(folder_path, data_len):
    total_l1 = []
    for i in range(1,data_len-1):
        print('Processing data %d/%d' %(i, data_len))
        gt_im_path = os.path.join(folder_path[0], 'gt',  'gt_' + str(i) + '.png')
        out_im_path = os.path.join(folder_path[1], 'output', 'output_' + str(i) + '.png')
        gt_im = cv2.imread(gt_im_path)
        out_im = cv2.imread(out_im_path)
        l1 = calc_l1(gt_im, out_im)
        print(l1)
        total_l1.append(l1)
    return total_l1

def write_to_file(in_list, out_file):
    fs = open(out_file, 'w')
    for item in in_list:
       fs.write(str(item) + '\n')
    fs.close()

def compute_kde(x, x_grid, bandwith=1.0, **kwargs):
    kde = gaussian_kde(x, bw_method=bandwith/x.std(ddof=1), **kwargs)
    return kde.evaluate(x_grid)

def read_list_from_file(filename):
    out = []
    fs = open(filename, 'r')
    for line in fs:
        line = line.strip()
        out.append(float(line))
    return out


output_path = ['../output_ul/in','../output_ul/gt','../output_ul/output2']
write_output(output_path, model_path='model_ul_large_final/model_ul_10000', net_type='large')
