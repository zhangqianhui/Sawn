import torch
from model.base_model import BaseModel
from model.networks import base_function
from model.losses.external import *
import model.networks as network
from collections import OrderedDict
from util import task, util
import itertools
import os
from util.DiffAugment_pytorch import DiffAugment
import numpy as np

seg_key = {0: 1, 1: 2, 2: 4, 3: 5, 4: 5, 5: 5, 6: 5}

class Pose(BaseModel):
    """
       Sawn Pose-Transfer
    """
    def name(self):
        return "Pose-Transfer"

    @staticmethod
    def modify_options(parser, is_train=True):
        parser.add_argument('--attn_layer', action=util.StoreList, metavar="VAL1,VAL2...", help="The number layers away from output layer") 
        parser.add_argument('--kernel_size', action=util.StoreDictKeyPair, metavar="KEY1=VAL1,KEY2=VAL2...", help="Kernel Size of Local Attention Block")

        parser.add_argument('--layers', type=int, default=3, help='number of layers in G')
        parser.add_argument('--netG', type=str, default='pose', help='The name of net Generator')
        parser.add_argument('--netD', type=str, default='res', help='The name of net Discriminator')
        parser.add_argument('--init_type', type=str, default='orthogonal', help='Initial type')

        # if is_train:
        parser.add_argument('--ratio_g2d', type=float, default=0.1, help='learning rate ratio G to D')
        parser.add_argument('--lambda_rec', type=float, default=5.0, help='weight for image reconstruction loss')
        parser.add_argument('--lambda_g', type=float, default=2.0, help='weight for generation loss')
        parser.add_argument('--lambda_style', type=float, default=0.5, help='weight for the VGG19 style loss')
        parser.add_argument('--lambda_content', type=float, default=0.0025, help='weight for the VGG19 content loss')
        parser.add_argument('--policy', type=str, default=None, help='policy for differ aug')

        parser.add_argument('--use_spect_g', action='store_false', help="whether use spectral normalization in generator")
        parser.add_argument('--use_spect_d', action='store_false', help="whether use spectral normalization in discriminator")
        parser.add_argument('--save_input', action='store_false', help="whether save the input images when testing")

        parser.set_defaults(use_spect_g=False)
        parser.set_defaults(use_spect_d=True)
        parser.set_defaults(save_input=False)

        return parser

    def __init__(self, opt):
        BaseModel.__init__(self, opt)

        self.FloatTensor = torch.cuda.FloatTensor if len(self.gpu_ids) > 0 \
            else torch.FloatTensor

        self.net_G = network.define_g(opt, image_nc=opt.image_nc, structure_nc=opt.structure_nc, ngf=64, img_f=512,
                                      layers=opt.layers, num_blocks=2, use_spect=opt.use_spect_g, attn_layer=opt.attn_layer, 
                                      norm='instance', activation='LeakyReLU', extractor_kz=opt.kernel_size)

        if len(opt.gpu_ids) > 1:
            self.net_G = torch.nn.DataParallel(self.net_G, device_ids=self.gpu_ids)

        if self.opt.dataset_mode == 'fashion':
            self.net_D = network.define_d(opt, ndf=32, img_f=128, layers=4, use_spect=opt.use_spect_d)

        if len(opt.gpu_ids) > 1:
            self.net_D = torch.nn.DataParallel(self.net_D, device_ids=self.gpu_ids)

        self.flow2color = util.flow2color()

        if self.isTrain:
            self.GANloss = AdversarialLoss(opt.gan_mode).to(opt.device)
            self.GANloss = torch.nn.DataParallel(self.GANloss, device_ids=opt.gpu_ids)
            self.L1loss = torch.nn.L1Loss()
            self.Vggloss = VGGLoss().to(opt.device)
            self.Vggloss = torch.nn.DataParallel(self.Vggloss, device_ids=opt.gpu_ids)
            # define the optimizer
            self.optimizer_G = torch.optim.Adam(itertools.chain(
                                               filter(lambda p: p.requires_grad, self.net_G.parameters())),
                                               lr=opt.lr, betas=(0.0, 0.999))
            self.optimizers.append(self.optimizer_G)
            self.optimizer_D = torch.optim.Adam(itertools.chain(
                                filter(lambda p: p.requires_grad, self.net_D.parameters())),
                                lr=opt.lr*opt.ratio_g2d, betas=(0.0, 0.999))
            self.optimizers.append(self.optimizer_D)

        self.loss_names = ['app_gen', 'content_gen', 'style_gen',
                           'ad_gen', 'dis_img_gen']
        self.visual_names = ['input_P1', 'input_P2', 'img_gen', 'flow_fields', 'masks', 'vis_input_SP2']
        self.model_names = ['G', 'D']

        # load the pre-trained model and schedulers
        self.setup(opt)

    def test(self):

        """Forward function used in test time"""
        img_gen, flow_fields, masks = self.net_G(self.input_P1, self.input_BP1, self.input_BP2, self.input_SP1)
        self.save_results(img_gen, data_name='vis')
        if self.opt.save_input or self.opt.phase == 'val':
            result = torch.cat([self.input_P1, img_gen, self.input_P2], 3)
            self.save_results(result, data_name='all')

    # Test texture transfer
    def testTexture(self):

        img_gen, _, _, _ = self.net_G(self.input_P1, self.input_BP1, self.input_SP1,
                                                 self.input_P2, self.input_BP2, self.input_SP2, self.opt.is_texturet, self.opt.seg_label)
        if self.opt.save_input or self.opt.phase == 'val':
            self.save_results(self.input_P1, data_name='ref')
            self.save_results(self.input_P2, data_name='gt')
        result = torch.cat([self.input_P1, img_gen, self.input_P2], 3)
        self.save_results(result, data_name='all')

    def forward(self):

        key = np.random.randint(0,7)
        seg_label = seg_key[key]
        if self.opt.is_texturet: # STPR
            self.img_gen, self.flow_fields, self.flow_fields_ref, self.masks = self.net_G(self.input_P2, self.input_BP2, self.input_SP2,
                                                                    self.input_P1, self.input_BP1, self.input_SP1, self.opt.is_texturet, seg_label)
        else:
            self.img_gen, self.flow_fields, self.masks = self.net_G(self.input_P1, self.input_BP1, self.input_BP2,
                                                                    self.input_SP1)

        self.warped = self.visi(self.flow_fields[-1], self.input_P1)
        self.seg = util.tensor2label(self.input_SP1, 8)
        _, _, inputh, inputw = self.input_P1.size()
        self.mask = torch.nn.functional.interpolate(self.masks[-1], (inputh, inputw))
        self.flow_resize = torch.nn.functional.interpolate(self.flow_fields[-1], (inputh, inputw))

    def backward_D_basic(self, netD, real, fake):

        # Real
        real_aug = DiffAugment(real, policy=self.opt.policy)
        fake_aug = DiffAugment(fake, policy=self.opt.policy)

        D_real = netD(real_aug)
        D_real_loss = self.GANloss(D_real, True, True).mean()
        D_fake = netD(fake_aug.detach())
        D_fake_loss = self.GANloss(D_fake, False, True).mean()
        D_loss = (D_real_loss + D_fake_loss) * 0.5

        D_loss.backward()

        return D_loss

    def backward_D(self):

        base_function._unfreeze(self.net_D)
        self.loss_dis_img_gen = self.backward_D_basic(self.net_D, self.input_P2, self.img_gen)

    def backward_G(self):

        img_gen_aug = DiffAugment(self.img_gen, policy=self.opt.policy)
        loss_app_gen = self.L1loss(self.img_gen, self.input_P2)
        self.loss_app_gen = loss_app_gen * self.opt.lambda_rec

        base_function._freeze(self.net_D)
        D_fake = self.net_D(img_gen_aug)
        self.loss_ad_gen = self.GANloss(D_fake, True, False).mean() * self.opt.lambda_g

        loss_content_gen, loss_style_gen = self.Vggloss(self.img_gen, self.input_P2)
        self.loss_style_gen = loss_style_gen.mean() * self.opt.lambda_style
        self.loss_content_gen = loss_content_gen.mean() * self.opt.lambda_content

        total_loss = 0
        for name in self.loss_names:
            if name != 'dis_img_gen':
                total_loss += getattr(self, "loss_" + name)
        total_loss.backward()

    def optimize_parameters(self):
        """update network weights"""
        self.forward()

        self.optimizer_D.zero_grad()
        self.backward_D()
        self.optimizer_D.step()

        self.optimizer_G.zero_grad()
        self.backward_G()
        self.optimizer_G.step()

    def get_current_visuals(self):

        height, width = self.input_P1.size(2), self.input_P1.size(3)
        input_P1 = util.tensor2im(self.input_P1.data)
        input_P2 = util.tensor2im(self.input_P2.data)
        warped = util.tensor2im(self.warped.detach())
        mask = util.tensor2im(self.mask.detach())

        self.flow2color = util.flow2color()
        flowfield = self.flow2color(self.flow_resize)
        flowfield = util.tensor2im(flowfield)

        input_BP1 = util.draw_pose_from_map(self.input_BP1.data)[0]
        input_BP2 = util.draw_pose_from_map(self.input_BP2.data)[0]

        img_gen = util.tensor2im(self.img_gen.data)
        vis = np.zeros((height, width * 9, 3)).astype(np.uint8)  # h, w, c
        vis[:, :width, :] = input_P1
        vis[:, width:width * 2, :] = input_BP1
        vis[:, width * 2:width * 3, :] = input_P2
        vis[:, width * 3:width * 4, :] = input_BP2
        vis[:, width * 4:width * 5, :] = img_gen
        vis[:, width * 5:width * 6, :] = warped
        vis[:, width * 6:width * 7, :] = mask
        vis[:, width * 7:width * 8, :] = self.seg
        vis[:, width * 8:width * 9, :] = flowfield
        ret_visuals = OrderedDict([('vis', vis)])

        return ret_visuals

    def visi(self, flow_field, input):

        [b, _, h, w] = flow_field.size()
        _, _, inputh, inputw = input.size()

        source_copy = torch.nn.functional.interpolate(input, (h, w))
        x = torch.arange(w).view(1, -1).expand(h, -1).float()
        y = torch.arange(h).view(-1, 1).expand(-1, w).float()
        x = 2 * x / (w - 1) - 1
        y = 2 * y / (h - 1) - 1
        grid = torch.stack([x, y], dim=0).float().cuda()
        grid = grid.unsqueeze(0).expand(b, -1, -1, -1)
        flow_x = (2 * flow_field[:, 0, :, :] / (w - 1)).view(b, 1, h, w)
        flow_y = (2 * flow_field[:, 1, :, :] / (h - 1)).view(b, 1, h, w)
        flow = torch.cat((flow_x, flow_y), 1)

        grid = (grid + flow).permute(0, 2, 3, 1)
        warp = torch.nn.functional.grid_sample(source_copy, grid)
        warp = torch.nn.functional.interpolate(warp, (inputh, inputw))

        return warp

    def set_input(self, input):

        self.input = input
        input_P1, input_BP1, input_SP1  = input['P1'], input['BP1'], input['SP1']
        input_P2, input_BP2, input_SP2 = input['P2'], input['BP2'], input['SP2']

        if len(self.gpu_ids) > 0:
            self.input_P1 = input_P1.cuda(self.gpu_ids[0], async=True)
            self.input_BP1 = input_BP1.cuda(self.gpu_ids[0], async=True)
            self.input_P2 = input_P2.cuda(self.gpu_ids[0], async=True)
            self.input_BP2 = input_BP2.cuda(self.gpu_ids[0], async=True)
            self.input_SP1 = input_SP1.cuda(self.gpu_ids[0], async=True)
            self.input_SP2 = input_SP2.cuda(self.gpu_ids[0], async=True)

        self.image_paths=[]
        for i in range(self.input_P1.size(0)):
            self.image_paths.append(os.path.splitext(input['P1_path'][i])[0] + '_2_' + input['P2_path'][i])


