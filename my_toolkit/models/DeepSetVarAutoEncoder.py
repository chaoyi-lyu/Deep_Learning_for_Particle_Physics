import torch
from torch import nn
from my_toolkit.utils.layers.fspool import FSPool

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
        
    def forward(self, x):
        layers = [nn.Linear(self.num_features, self.num_hidden[0]), self.activ]
        for i,_ in enumerate(self.num_hidden):
            layers.append( nn.Linear(self.num_hidden[i-1], self.num_hidden[i]) )
            layers.append( self.activ )
        del layers[2:4]
        layers.append( nn.Linear(self.num_hidden[0], self.latent*2) )
        model = nn.Sequential(*layers)
        return model(x)

class ParticleConv(nn.Module):
    """
    Map all particles in an event to the representation in the latent space.
    After this operation, each particle corresponds to one channel.

    Args:
        the neural network used to map each particle
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

    def forward(self, x):
        x = self.particle_conv(x)
        x, perm = FSPool(self.particle_encoder.latent*2, 1)(x)
        samples = self.Sampling(x)
        return x[:,::2], x[:,1::2], samples

    def Sampling(self, inputs):
        mus = inputs[:,::2]
        logvars = inputs[:,1::2]
        stds = torch.exp(0.5*logvars)
        eps = torch.randn_like(stds)
        return mus + eps*stds

    