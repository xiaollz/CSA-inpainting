class Opion():
    
    def __init__(self):
            
        self.dataroot= r'/Data_SSD/shuyiqu/inpaint_data/img_celeba' #image dataroot
        self.maskroot= r'/Data_SSD/shuyiqu/inpaint_data/irregular_mask/disocclusion_img_mask'#mask dataroot
        self.batchSize= 16   # Need to be set to 1
        self.fineSize=256 # image size
        self.input_nc=3  # input channel size for first stage
        self.input_nc_g=6 # input channel size for second stage
        self.output_nc=3# output channel size
        self.ngf=64 # inner channel
        self.ndf=64# inner channel
        self.which_model_netD='basic' # patch discriminator
        
        self.which_model_netF='feature'# feature patch discriminator
        self.which_model_netG='unet_csa'# seconde stage network
        self.which_model_netP='unet_256'# first stage network
        self.triple_weight=1
        self.name='CSA_inpainting'
        self.n_layers_D='3' # network depth
        self.gpu_ids=[4]
        self.model='csa_net'
        self.checkpoints_dir=r'/Data_HDD/shuyiqu/csa/celeba' #
        self.norm='instance'
        self.fixed_mask=1
        self.use_dropout=False
        self.init_type='normal'
        self.mask_type='center'
        self.lambda_A=1
        self.threshold=5/16.0
        self.stride=1
        self.shift_sz=1 # size of feature patch
        self.mask_thred=1
        self.bottleneck=512
        self.gp_lambda=10.0
        self.ncritic=5
        self.constrain='MSE'
        self.strength=1
        self.init_gain=0.02
        self.cosis=1
        self.gan_type='lsgan'
        self.gan_weight=0.002
        self.overlap=4
        self.skip=0
        self.display_freq=10000
        self.print_freq=100
        self.save_latest_freq=10000
        self.save_epoch_freq=1
        self.continue_train=False
        self.epoch_count=1
        self.phase='train'
        self.which_epoch=''
        self.niter=2
        self.niter_decay=0
        self.beta1=0.5
        self.lr=0.0032
        self.lr_policy='lambda'
        self.lr_decay_iters=50
        self.isTrain=True
        self.save_dir=r'/Data_HDD/shuyiqu/csa/results/celeba'
        # for free-form mask generation
        self.image_shape=[256,256]
        self.mv=20
        self.ma=4.0
        self.ml=40
        self.mbw=8

import time
from util.data_load import Data_load
from models.models import create_model
import torch
import os
import torchvision
from torch.utils import data as dt
import torchvision.transforms as transforms

from tensorboardX import SummaryWriter

opt = Opion()

writer = SummaryWriter()

transform_mask = transforms.Compose(
    [transforms.Resize((opt.fineSize,opt.fineSize)),
     transforms.ToTensor(),
    ])
transform = transforms.Compose(
    [transforms.RandomHorizontalFlip(),
     transforms.Resize((opt.fineSize,opt.fineSize)),
     transforms.ToTensor(),
     transforms.Normalize(mean=[0.5] * 3, std=[0.5] * 3)])

dataset_train = Data_load(opt.dataroot, opt.maskroot, transform, transform_mask)
iterator_train = (dt.DataLoader(dataset_train, batch_size=opt.batchSize,shuffle=True))
print(len(dataset_train))
model = create_model(opt)
total_steps = 0

print("start")


for epoch in range(opt.epoch_count, opt.niter + opt.niter_decay + 1):
    
    epoch_start_time = time.time()
    epoch_iter = 0

#     image, mask, gt = [x.cuda() for x in next(iterator_train)]
    for image, mask in (iterator_train):
        iter_start_time = time.time()
        image=image.cuda()
        mask=mask.cuda()
        mask=mask[0][0]
        mask=torch.unsqueeze(mask,0)
        mask=torch.unsqueeze(mask,1)
        mask=mask.byte()

        total_steps += opt.batchSize
        epoch_iter += opt.batchSize
        model.set_input(image,mask) # it not only sets the input data with mask, but also sets the latent mask.
        model.set_gt_latent()
        model.optimize_parameters()

        if total_steps %opt.display_freq== 0:
            real_A,real_B,fake_B=model.get_current_visuals()
            #real_A=input, real_B=ground truth fake_b=output
            pic = (torch.cat([real_A, real_B, fake_B], dim=0) + 1) / 2.0
            torchvision.utils.save_image(pic, '%s/Epoch_(%d)_(%dof%d).jpg' % (
            opt.save_dir, epoch, total_steps + 1, len(dataset_train)), nrow=2)
            writer.add_image('input', real_A, total_steps)
            writer.add_image('ground_truth', real_B, total_steps)
            writer.add_image('output', fake_B, total_steps)
        if total_steps %1== 0:
            errors = model.get_current_errors()
            writer.add_scalar('loss/G_GAN', errors['G_GAN'], total_steps)
            writer.add_scalar('loss/G_L1', errors['G_L1'], total_steps)
            writer.add_scalar('loss/D', errors['D'], total_steps)
            writer.add_scalar('loss/F', errors['F'], total_steps)
            t = (time.time() - iter_start_time) / opt.batchSize
            print(errors)
            print(t)

    if epoch % opt.save_epoch_freq == 0:
        print('saving the model at the end of epoch %d, iters %d' %
                (epoch, total_steps))
        model.save(epoch)

    print('End of epoch %d / %d \t Time Taken: %d sec' %
            (epoch, opt.niter + opt.niter_decay, time.time() - epoch_start_time))

    model.update_learning_rate()

# export scalar data to JSON for external processing
# writer.export_scalars_to_json("./train_scalars.json")
# writer.close()

