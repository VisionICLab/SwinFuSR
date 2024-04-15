import torch
import torch.nn as nn
import torch.nn.functional as F
from torchmetrics import StructuralSimilarityIndexMeasure
from torchmetrics import PeakSignalNoiseRatio
from segmentation_models_pytorch.utils import base
import piqa
from .discriminator import Discriminator , DiscriminatorTSRGAN,Discriminator_VGG_96

"from CoReFusion"
import torch.optim as optim

def device_as(t1, t2):
    """
    Moves t1 to the device of t2
    """
    return t1.to(t2.device)
class ContrastiveLoss(nn.Module):
    """
    Vanilla Contrastive loss, also called InfoNceLoss as in SimCLR paper
    """
    def __init__(self, batch_size, temperature=0.5):
        super().__init__()
        self.batch_size = batch_size
        self.temperature = temperature
        self.mask = (~torch.eye(batch_size * 2, batch_size * 2, dtype=bool)).float()

    def calc_similarity_batch(self, a, b):
        representations = torch.cat([a, b], dim=0)
        return F.cosine_similarity(representations.unsqueeze(1), representations.unsqueeze(0), dim=2)

    def forward(self, proj_1, proj_2):
        """
        proj_1 and proj_2 are batched embeddings [batch, embedding_dim]
        where corresponding indices are pairs
        z_i, z_j in the SimCLR paper
        """
        batch_size = proj_1.shape[0]
        z_i = F.normalize(proj_1, p=2, dim=1)
        z_j = F.normalize(proj_2, p=2, dim=1)

        similarity_matrix = self.calc_similarity_batch(z_i, z_j)

        sim_ij = torch.diag(similarity_matrix, batch_size)
        sim_ji = torch.diag(similarity_matrix, -batch_size)

        positives = torch.cat([sim_ij, sim_ji], dim=0)

        nominator = torch.exp(positives / self.temperature)

        denominator = device_as(self.mask, similarity_matrix) * torch.exp(similarity_matrix / self.temperature)

        all_losses = -torch.log(nominator / torch.sum(denominator, dim=1))
        loss = torch.sum(all_losses) / (2 * self.batch_size)
        return loss

'''from https://github.com/researchmm/TTSR/blob/master/loss/loss.py'''
class AdversarialLoss(nn.Module):
    def __init__(self, use_cpu=False, num_gpu=1, gan_type='WGAN_GP', gan_k=1, 
        lr_dis=1e-4, train_crop_size=40):

        super(AdversarialLoss, self).__init__()
        self.gan_type = gan_type
        self.gan_k = gan_k
        self.device = torch.device('cpu' if use_cpu else 'cuda')
        self.discriminator = Discriminator_VGG_96().to(self.device)
        if (num_gpu > 1):
            self.discriminator = nn.DataParallel(self.discriminator, list(range(num_gpu)))
        if (gan_type in ['WGAN_GP', 'GAN']):
            self.optimizer = optim.Adam(
                self.discriminator.parameters(),
                betas=(0, 0.9), eps=1e-8, lr=lr_dis
            )
        else:
            raise SystemExit('Error: no such type of GAN!')

        self.bce_loss = torch.nn.BCELoss().to(self.device)
            
    def forward(self, fake, real):

        fake_detach = fake.detach()

        for _ in range(self.gan_k):
            self.optimizer.zero_grad()

            d_real = self.discriminator(real)
            d_fake = self.discriminator(fake_detach)

            
            if (self.gan_type.find('WGAN') >= 0):
                loss_d = (d_fake - d_real).mean()
                if self.gan_type.find('GP') >= 0:
                    epsilon = torch.rand(real.size(0), 1, 1, 1).to(self.device)
                    epsilon = epsilon.expand(real.size())
                    hat = fake_detach.mul(1 - epsilon) + real.mul(epsilon)
                    hat.requires_grad = True
                    d_hat = self.discriminator(hat)
                    gradients = torch.autograd.grad(
                        outputs=d_hat.sum(), inputs=hat,
                        retain_graph=True, create_graph=True, only_inputs=True
                    )[0]
                    gradients = gradients.view(gradients.size(0), -1)
                    gradient_norm = gradients.norm(2, dim=1)
                    gradient_penalty = 10 * gradient_norm.sub(1).pow(2).mean()
                    loss_d += gradient_penalty

            elif (self.gan_type == 'GAN'):
                valid_score = torch.ones(real.size(0), 1).to(self.device)
                fake_score = torch.zeros(real.size(0), 1).to(self.device)
                real_loss = self.bce_loss(torch.sigmoid(d_real), valid_score)
                fake_loss = self.bce_loss(torch.sigmoid(d_fake), fake_score)
                loss_d = (real_loss + fake_loss) / 2.

            # Discriminator update
            loss_d.backward()
            self.optimizer.step()

        d_fake_for_g = self.discriminator(fake)
        if (self.gan_type.find('WGAN') >= 0):
            loss_g = -d_fake_for_g.mean()
        elif (self.gan_type == 'GAN'):
            loss_g = self.bce_loss(torch.sigmoid(d_fake_for_g), valid_score)

        # Generator loss
        return loss_g
  
    def state_dict(self):
        D_state_dict = self.discriminator.state_dict()
        D_optim_state_dict = self.optimizer.state_dict()
        return D_state_dict, D_optim_state_dict

class custom_loss(base.Loss):
    def __init__(self, batch_size,dico_weight,device='cuda:0'):
        super().__init__()
        self.ssim = StructuralSimilarityIndexMeasure().to(device)
        self.mse = nn.MSELoss().to(device)
        self.L1 = nn.L1Loss().to(device)
        self.contrast = ContrastiveLoss(batch_size).to(device)
        self.psnr = PeakSignalNoiseRatio().to(device)
        self.lpips =  piqa.LPIPS(network='vgg').to(device)
        self.adversarial = AdversarialLoss().to(device)
        self.dico_weight = dico_weight

    def forward(self, y_pr, y_gt, ft1=None, ft2=None):
        loss = 0
        #print(y_pr, y_gt, ft1, ft2)
        if self.dico_weight["l1"]!=0:
            #print("l1",self.L1(y_pr, y_gt))
            loss+=self.dico_weight["l1"]*self.L1(y_pr, y_gt)
        if self.dico_weight["mse"]!=0:
            loss+=self.dico_weight["mse"]*self.mse(y_pr, y_gt)
        if self.dico_weight["ssim"]!=0:
            loss+=self.dico_weight["ssim"]*(1-self.ssim(y_pr, y_gt))
        if self.dico_weight["psnr"]!=0:
            loss+=self.dico_weight["psnr"]*(1 - self.psnr(y_pr, y_gt)/40)
        if self.dico_weight["contrast"]!=0:
            #print("contrast",self.contrast(ft1, ft2))
            loss+=self.dico_weight["contrast"]*(self.contrast(ft1, ft2))
        if self.dico_weight["lpips"]!=0:
            if y_pr.shape[1] == 1:
                y_pr = y_pr.repeat(1,3,1,1)
            if y_gt.shape[1] == 1:
                y_gt = y_gt.repeat(1,3,1,1)
            loss+=self.dico_weight["lpips"]*self.adversarial(y_pr, y_gt)
        if self.dico_weight["adversarial"]!=0:
            loss+=self.dico_weight["adversarial"]*self.adversarial(y_pr, y_gt)
        return loss
        