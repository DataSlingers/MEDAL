import torch
import torch.nn.functional as F

def reconstruction_loss(x, x_recon, type="mse"):
    """Standard MSE loss for reconstruction."""
    if type == "mse":
        loss = F.mse_loss(x_recon, x, reduction='mean')
    return loss

def kl_divergence(z):
    """KL divergence for Variational Autoencoder regularization."""
    mean = torch.mean(z, dim=0)
    std = torch.std(z, dim=0)
    return -0.5 * torch.sum(1 + torch.log(std**2) - mean**2 - std**2)

def distillation_loss(student_z, teacher_z):
    """Encourages student embeddings to match teacher embeddings."""
    return F.mse_loss(student_z, teacher_z, reduction='mean')
    #return F.l1_loss(student_z, teacher_z, reduction='mean')

def get_loss_function(lambda_kl = 0, lambda_d = 0):
    """Returns a loss function that includes reconstruction + KL + distillation."""
    
    def loss_fn(x, x_recon, student_z, teacher_z=None):
        recon_loss = reconstruction_loss(x, x_recon)
        # kl_loss = kl_divergence(student_z)
        
        if teacher_z is not None:
            distill_loss = distillation_loss(student_z, teacher_z)
            return recon_loss + lambda_d * distill_loss
        else:
            return recon_loss 
    
    return loss_fn


