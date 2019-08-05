from util.base import *
from util.opt import Options
import util.utilities as utl
from dataset.dataset import Dataset
from model.Networks import Networks, Networks_Wnet, NetworksSegment


device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

opt = Options(sys.argv[0])

# Define Dataset
# =================
data_loader = Dataset('train',resize=opt.resize).load_data()

# net = NetworksSegment(device, num_class=8)
net = Networks(device, opt.net)
net.set_phase('train')
net.print_structure()
# net.load_model('model_dog_small_0')
# net.load_model_G('model_sn_small_0', False)
# net.print_structure()
# net.load_model('large_190723/model_190716_large_10000',True)
# net.load_model_G('model_190712/model_n_medium_30000', False)
# net.load_model('model_190716_large_10000', True)

if opt.net == 'small':
    forward_call = net.train_small
    loss_call = net.compute_loss_small
    txt_summary_call = net.print_summary_small
    img_summary_call = net.write_imgs_summary_small
elif opt.net == 'medium':
    forward_call = net.train_medium
    loss_call = net.compute_loss_medium
    txt_summary_call = net.print_summary_small
    img_summary_call = net.write_imgs_summary_medium
else:
    forward_call = net.train_large
    loss_call = net.compute_loss_large
    txt_summary_call = net.print_summary_small
    img_summary_call = net.write_imgs_summary_large


total_step = 0
start = time.time()
for epoch in range(opt.total_epochs):
    step = 0

    for item in data_loader:
        mask = utl.create_mask_ul()
        in_img, gt_img, gt_fov = item['input'], item['gt'], item['fov']


        if in_img.size()[0] != opt.train_batch:
            break


        # Load image to network
        net.load_input_batch(in_img, gt_img, gt_fov)

        # Forward network
        forward_call() # net.forward_<type>()

        end = time.time()
        elapsed = end - start

        # Print network loss
        # # Add Tensorboard summary
        if step % 50 == 0:
            # net.write_scalars(step)
            txt_summary_call(epoch, step) # net.print_summary_<type>(epoch, step)
            print('Time elapsed', elapsed, 'seconds')
        if step % 100 == 0:
            img_summary_call(step) # net.write_imgs_summary_<type>(step)
            # net.write_scalar_summary('train/L1', total_step)
        if step % 1000 == 0:
            net.save_model(step, 'model_dog_' + opt.net)

        step += 1
        total_step += 1

