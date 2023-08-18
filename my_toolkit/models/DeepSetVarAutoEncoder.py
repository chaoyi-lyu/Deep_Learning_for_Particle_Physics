import torch
from torch import nn
from my_toolkit.utils.layers.fspool import FSPool
from my_toolkit.utils.loss.chamfer_distance import ChamferDistance as chamfer_dist

class ParticleEncoder(nn.Module):
    """
    A neural network that maps each particle to the latent space.

    Args:
        the number of features per particle
        the number of nodes in hidden layer 1
        the number of nodes in hidden layer 2
        ...
        the dimension of latent space
    """
    def __init__(self, num_inputs, *num_hidden, num_outputs=8, activ='ReLU'):
        super().__init__()
        self.latent = num_outputs
        self.num_features = num_inputs
        self.num_hidden = num_hidden
        activ_dic = {'ReLU': nn.ReLU(), 
                     'ELU': nn.ELU(),  
                     'LReLU': nn.LeakyReLU(), }
        self.activ = activ_dic[activ]
        layers = [nn.Linear(self.num_features, self.num_hidden[0]), self.activ]
        for i,_ in enumerate(self.num_hidden):
            layers.append( nn.Linear(self.num_hidden[i-1], self.num_hidden[i]) )
            layers.append( self.activ )
        del layers[2:4]
        layers.append( nn.Linear(self.num_hidden[-1], self.latent*2) )
        self.stack_layers = nn.Sequential(*layers)
        
    def forward(self, x):
        output = self.stack_layers(x)
        return output

class ParticleConv(nn.Module):
    """
    Map all particles in an event to the representation in the latent space.
    Each particle corresponds to one channel.

    Args:
        the number of particles in the events
        
    """
    def __init__(self, particle_encoder):
        super().__init__()
        self.particle_encoder = particle_encoder
        self.num_features = self.particle_encoder.num_features

    def forward(self, x):
        channels = []
        self.num_particles = int(x.shape[1] / self.num_features)
        x = x.reshape(x.shape[0], -1, self.num_features)
        for i in range(self.num_particles):
            particle = x[:,i,:]
            channels.append( self.particle_encoder(particle) )
        return torch.stack(channels, 2)
    
class Encoder(nn.Module):
    """
    The encoder that map events to the latent space.
    FSPooling layer is applied to combine the per-particle latent spaces into an event-level latent representation.
    After the Pooling layer, we reparameterize the system with a Gaussian prior.
    """
    def __init__(self, particle_encoder):
        super().__init__()
        self.particle_encoder = particle_encoder
        self.num_features = self.particle_encoder.num_features
        self.particle_conv = ParticleConv(self.particle_encoder)
        self.fspool = FSPool(self.particle_encoder.latent*2, 20)

    def forward(self, x):
        x = self.particle_conv(x)
        x, perm = self.fspool(x)
        samples = self.Sampling(x)
        return x[:,::2], x[:,1::2], samples

    def Sampling(self, inputs):
        mus = inputs[:,::2]
        logvars = inputs[:,1::2]
        stds = torch.exp(0.5*logvars)
        eps = torch.randn_like(stds)
        return mus + eps*stds

class Decoder(nn.Module):
    """
    A multilayer perceptron used for decode.
    The network is splited to two outputs: the regression of particle momenta, and the classification of particle species.
    """
    def __init__(self, num_particles, num_inputs, *num_hidden, activ='ReLU'):
        super().__init__()
        self.num_particles = num_particles
        self.latent = num_inputs 
        self.num_hidden = num_hidden
        activ_dic = {'ReLU': nn.ReLU(), 
                     'ELU': nn.ELU(),  
                     'LReLU': nn.LeakyReLU(), }
        self.activ = activ_dic[activ]
        layers = [nn.Linear(self.latent, self.num_hidden[0]), self.activ]
        for i,_ in enumerate(self.num_hidden):
            layers.append( nn.Linear(self.num_hidden[i-1], self.num_hidden[i]) )
            layers.append( self.activ )
        del layers[2:4]
        self.stack_layers = nn.Sequential(*layers)
        self.linear1 = nn.Linear(self.num_hidden[-1], self.num_particles*4)
        self.linear2 = nn.Linear(self.num_hidden[-1], self.num_particles*9)
        self.logsoftmax = nn.LogSoftmax(dim=2)

    def forward(self, x):
        x = self.stack_layers(x)
        kine_pred = self.linear1(x)
        kine_pred = kine_pred.reshape(-1, self.num_particles, 4)
        pre_class = self.linear2(x)
        class_pred = self.logsoftmax(pre_class.reshape(-1, self.num_particles, 9))
        return kine_pred, class_pred
    
class AutoEncoder(nn.Module):
    """
    The autoencoder that combines the encoder and decoder.
    """
    def __init__(self, model, num_particles, activ='ReLU'):
        super().__init__()
        self.encoder = Encoder(model)
        self.num_particles = num_particles
        self.decoder = Decoder(self.num_particles, model.latent, 256, 256)

    def forward(self, x):
        mean, log_var, sample = self.encoder(x)
        kine_pred, class_pred = self.decoder(sample)
        return mean, log_var, kine_pred, class_pred  

class LossFunc(nn.Module):
    """
    The loss function has three parts. The Chamfer loss: the distance between
    """
    def __init__(self, beta, w):
        super().__init__()
        self.beta = beta
        self.w = w
        self.chd = chamfer_dist()

    def forward(self, kine_input, class_input, kine_pred, class_pred, mu, log_var):
        Chamfer_loss, *indxs = self.ChamferLoss(kine_input, kine_pred)
        Class_loss = self.ClassLoss(class_input, class_pred, indxs)
        KL_loss = self.KLDivergence(mu, log_var)
        
        total_loss = (1 - self.beta) * (Chamfer_loss + self.w * Class_loss) + self.beta * KL_loss
        
        return total_loss

    def ChamferLoss(self, kine_input, kine_pred):
        dist1, dist2, idx1, idx2 = self.chd(kine_input, kine_pred)
        dist_loss = dist1.sum() + dist2.sum()
        return dist_loss, idx1, idx2

    def ClassLoss(self, class_input, class_pred, indxs):
        sorted_pred = torch.zeros_like(class_pred)
        sorted_input = torch.zeros_like(class_input)
        idx1, idx2 = indxs
        for i in range(idx1.shape[0]):
            for j in range(idx1.shape[1]):
                sorted_pred[i,j,:] = class_pred[i,idx1[i,j],:]
                sorted_input[i,j,:] = class_input[i,idx2[i,j],:]
        loss = (sorted_pred * class_input + sorted_input * class_pred).sum()
        return loss

    def KLDivergence(self, mu, log_var):
        KLD = -0.5 * torch.sum(1 + log_var - mu.pow(2) - log_var.exp())
        return KLD


