import numpy as np
import torch
import os
from collections import OrderedDict
from torch.autograd import Variable
import itertools
import util.util as util
from util.image_pool import ImagePool
from .base_model import BaseModel
from . import networks
import random
import math
import sys
import pdb

class VAEModel(BaseModel):
    def name(self):
        return 'VAEModel'

    def initialize(self, opt):
        BaseModel.initialize(self, opt)
        nb = opt.batchSize
        size = opt.fineSize


        self.input_A = self.Tensor(nb, opt.input_nc, size, size)
        self.input_B = self.Tensor(nb, opt.output_nc, size, size)

        self.netE_AB = networks.define_G(opt.input_nc, opt.output_nc,
                                        opt.ngf, opt.which_model_netG, opt.norm, not opt.no_dropout, opt.init_type, self.gpu_ids)

        if self.isTrain:
            use_sigmoid = opt.no_lsgan
            self.netD_B = networks.define_D(opt.output_nc, opt.ndf,
                                            opt.which_model_netD,
                                            opt.n_layers_D, opt.norm, use_sigmoid, opt.init_type, self.gpu_ids)
            self.netD_rot_B = networks.define_D(opt.output_nc, opt.ndf,
                                               opt.which_model_netD,
                                               opt.n_layers_D, opt.norm, use_sigmoid, opt.init_type, self.gpu_ids)
            self.netD_vf_B = networks.define_D(opt.output_nc, opt.ndf,
                                               opt.which_model_netD,
                                               opt.n_layers_D, opt.norm, use_sigmoid, opt.init_type, self.gpu_ids)

        if not self.isTrain or opt.continue_train:
            which_epoch = opt.which_epoch
            self.load_network(self.netE_AB, 'E_AB', which_epoch)
            if self.isTrain:
                self.load_network(self.netD_B, 'D_B', which_epoch)
                self.load_network(self.netD_rot_B, 'D_rot_B', which_epoch)
                self.load_network(self.netD_vf_B, 'D_vf_B', which_epoch)

        if self.isTrain:
            self.old_lr = opt.lr
            self.fake_B_pool = ImagePool(opt.pool_size)
            self.fake_rot_B_pool = ImagePool(opt.pool_size)
            self.fake_vf_B_pool = ImagePool(opt.pool_size)
            self.aeLoss = networks.GANLoss(use_lsgan = opt.no_lsgan, tensor=self.Tensor)
            self.criterionIdt = torch.nn.L1Loss()
            self.criterionGc = torch.nn.L1Loss()
            self.optimizer_G = torch.optim.Adam(itertools.chain(self.netE_AB.parameters()), lr=opt.lr, betas=(opt.beta1, 0.999))
            self.optimizer_D_B = torch.optim.Adam(itertools.chain(self.netD_B.parameters(), self.netD_rot_B.parameters(), self.netD_vf_B.parameters()), lr=opt.lr, betas=(opt.beta1, 0.999))

            self.optimizers = []
            self.schedulers = []
            self.optimizers.append(self.optimizer_G)
            self.optimizers.append(self.optimizer_D_B)
            for optimizer in self.optimizers:
                self.schedulers.append(networks.get_scheduler(optimizer, opt))

        print('---------- Networks initialized -------------')
        networks.print_network(self.netE_AB)
        if self.isTrain:
            networks.print_network(self.netD_B)
            networks.print_network(self.netD_rot_B)
            networks.print_network(self.netD_vf_B)
        print('-----------------------------------------------')

    def set_input(self, input):
        AtoB = self.opt.which_direction == 'AtoB'
        input_A = input['A' if AtoB else 'B']
        input_B = input['B' if AtoB else 'A']
        self.input_A.resize_(input_A.size()).copy_(input_A)
        self.input_B.resize_(input_B.size()).copy_(input_B)
        self.image_paths = input['A_paths' if AtoB else 'B_paths']

    def backward_D_basic(self, netD, real, fake, netD_rot, real_rot, fake_rot, netD_vf, real_vf, fake_vf):
        pred_real = netD(real)
        loss_D_real = self.aeLoss(pred_real, True)
        pred_fake = netD(fake.detach())
        loss_D_fake = self.aeLoss(pred_fake, False)
        loss_D = (loss_D_real + loss_D_fake) * 0.5

        pred_real_rot = netD_rot(real_rot)
        loss_D_rot_real = self.aeLoss(pred_real_rot, True)
        pred_fake_rot = netD_rot(fake_rot.detach())
        loss_D_rot_fake = self.aeLoss(pred_fake_rot, False)
        if self.mix < 0.5:
            loss_D += (loss_D_rot_real + loss_D_rot_fake) * 0.5
        else:
            loss_D += 0

        pred_real_vf = netD_vf(real_vf)
        loss_D_vf_real = self.aeLoss(pred_real_vf, True)
        pred_fake_vf = netD_vf(fake_vf.detach())
        loss_D_vf_fake = self.aeLoss(pred_fake_vf, False)
        if self.mix >= 0.5:
            loss_D += (loss_D_vf_real + loss_D_vf_fake) * 0.5
        else:
            loss_D += 0

        loss_D.backward()
        return loss_D

    def get_image_paths(self):
        return self.image_paths

    def rot90(self, tensor, direction):
        tensor = tensor.transpose(2, 3)
        size = self.opt.fineSize
        inv_idx = torch.arange(size-1, -1, -1).long().cuda()
        if direction == 0:
          tensor = torch.index_select(tensor, 3, inv_idx)
        else:
          tensor = torch.index_select(tensor, 2, inv_idx)
        return tensor

    def forward(self):
        input_A = self.input_A.clone()
        input_B = self.input_B.clone()

        self.real_A = self.input_A
        self.real_B = self.input_B

        size = self.opt.fineSize
        self.mix = random.random()
        self.real_rot_A = self.rot90(input_A.clone(), 0)
        self.real_rot_B = self.rot90(input_B.clone(), 0)
        inv_idx = torch.arange(size-1, -1, -1).long().cuda()
        self.real_vf_A = torch.index_select(input_A.clone(), 2, inv_idx)
        self.real_vf_B = torch.index_select(input_B.clone(), 2, inv_idx)

    def get_gc_rot_loss(self, AB, AB_gc, direction):
        loss_gc = 0.0

        if direction == 0:
          AB_gt = self.rot90(AB_gc.clone().detach(), 1)
          loss_gc = self.criterionGc(AB, AB_gt)
          AB_gc_gt = self.rot90(AB.clone().detach(), 0)
          loss_gc += self.criterionGc(AB_gc, AB_gc_gt)
        else:
          AB_gt = self.rot90(AB_gc.clone().detach(), 0)
          loss_gc = self.criterionGc(AB, AB_gt)
          AB_gc_gt = self.rot90(AB.clone().detach(), 1)
          loss_gc += self.criterionGc(AB_gc, AB_gc_gt)

        loss_gc = loss_gc*self.opt.lambda_AB*self.opt.lambda_gc
        return loss_gc

    def get_gc_vf_loss(self, AB, AB_gc):
        loss_gc = 0.0

        size = self.opt.fineSize

        inv_idx = torch.arange(size-1, -1, -1).long().cuda()

        AB_gt = torch.index_select(AB_gc.clone().detach(), 2, inv_idx)
        loss_gc = self.criterionGc(AB, AB_gt)

        AB_gc_gt = torch.index_select(AB.clone().detach(), 2, inv_idx)
        loss_gc += self.criterionGc(AB_gc, AB_gc_gt)

        loss_gc = loss_gc*self.opt.lambda_AB*self.opt.lambda_gc
        return loss_gc


    def backward_D_B(self):
        fake_B = self.fake_B_pool.query(self.fake_B)
        fake_rot_B = self.fake_rot_B_pool.query(self.fake_rot_B)
        fake_vf_B = self.fake_vf_B_pool.query(self.fake_vf_B)
        loss_D_B = self.backward_D_basic(self.netD_B, self.real_B, fake_B, self.netD_rot_B, self.real_rot_B, fake_rot_B, self.netD_vf_B, self.real_vf_B, fake_vf_B)
        self.loss_D_B = loss_D_B.item()

    def backward_G(self):
        fake_B = self.netE_AB.forward(self.real_A)
        pred_fake = self.netD_B.forward(fake_B)
        loss_E_AB = self.aeLoss(pred_fake, True)*self.opt.lambda_G

        fake_rot_B = self.netE_AB.forward(self.real_rot_A)
        pred_fake = self.netD_rot_B.forward(fake_rot_B)
        if self.mix < 0.5:
            loss_G_gc_AB = self.aeLoss(pred_fake, True)*self.opt.lambda_G
        else:
            loss_G_gc_AB = 0

        fake_vf_B = self.netE_AB.forward(self.real_vf_A)
        pred_fake = self.netD_vf_B.forward(fake_vf_B)
        if self.mix >= 0.5:
            loss_G_gc_AB += self.aeLoss(pred_fake, True)*self.opt.lambda_G
        else:
            loss_G_gc_AB += 0

        if self.mix < 0.5:
            loss_gc = self.get_gc_rot_loss(fake_B, fake_rot_B, 0)
        else:
            loss_gc = self.get_gc_vf_loss(fake_B, fake_vf_B)

        if self.opt.identity > 0:
            idt_A = self.netE_AB(self.real_B)
            loss_idt = self.criterionIdt(idt_A, self.real_B) * self.opt.lambda_AB * self.opt.identity

            idt_gc_A = self.netE_AB(self.real_rot_B)

            if self.mix < 0.5:
                loss_idt_gc = self.criterionIdt(idt_gc_A, self.real_rot_B) * self.opt.lambda_AB * self.opt.identity
            else:
                idt_gc_A = self.netE_AB(self.real_vf_B)
                loss_idt_gc = self.criterionIdt(idt_gc_A, self.real_vf_B) * self.opt.lambda_AB * self.opt.identity

            self.idt_A = idt_A.data
            self.idt_gc_A = idt_gc_A.data
            self.loss_idt = loss_idt.item()
            self.loss_idt_gc = loss_idt_gc.item()
        else:
            loss_idt = 0
            loss_idt_gc = 0
            self.loss_idt = 0
            self.loss_idt_gc = 0

        loss_G = loss_E_AB + loss_G_gc_AB + loss_gc + loss_idt + loss_idt_gc

        loss_G.backward()

        self.fake_B = fake_B.data
        self.fake_rot_B = fake_rot_B.data
        self.fake_vf_B = fake_vf_B.data

        self.loss_E_AB = loss_E_AB.item()
        self.loss_G_gc_AB= loss_G_gc_AB.item()
        self.loss_gc = loss_gc.item()

    def optimize_parameters(self):
        self.forward()
        self.optimizer_G.zero_grad()
        self.backward_G()
        self.optimizer_G.step()
        self.optimizer_D_B.zero_grad()
        self.backward_D_B()
        self.optimizer_D_B.step()

    def get_current_errors(self):
        ret_errors = OrderedDict([('D_B', self.loss_D_B), ('E_AB', self.loss_E_AB),
                                  ('Gc', self.loss_gc), ('G_gc_AB', self.loss_G_gc_AB)])

        if self.opt.identity > 0.0:
            ret_errors['idt'] = self.loss_idt
            ret_errors['idt_gc'] = self.loss_idt_gc

        return ret_errors

    def get_current_visuals(self):
        real_A = util.tensor2im(self.real_A.data)
        real_B = util.tensor2im(self.real_B.data)

        fake_B = util.tensor2im(self.fake_B)

        ret_visuals = OrderedDict([('real_A', real_A), ('fake_B', fake_B), ('real_B', real_B)])
        return ret_visuals

    def save(self, label):
        self.save_network(self.netE_AB, 'E_AB', label, self.gpu_ids)
        self.save_network(self.netD_B, 'D_B', label, self.gpu_ids)
        self.save_network(self.netD_rot_B, 'D_rot_B', label, self.gpu_ids)
        self.save_network(self.netD_vf_B, 'D_vf_B', label, self.gpu_ids)


    def test(self):
        self.real_A = Variable(self.input_A)
        self.real_B = Variable(self.input_B)

        self.fake_B = self.netE_AB.forward(self.real_A).data
