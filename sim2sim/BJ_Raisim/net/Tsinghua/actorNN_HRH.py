import torch
import os
import numpy as np


class Configuration:
    def __init__(self):
        self.env = self.env()
        self.net = self.net()
        self.model = self.model()
    class env:
        num_output = 12
        num_actor_input = 512
        num_sense_input = 58*6
        num_sense_output = 16

    class net:
        dim_actor_latent = [256, 128, 64, 32]
        dim_sense_latent = [128, 64, 32]
        actor_domain = 58

    class model:
        def __init__(self):
            self.path = ''

class ActorNetwork(torch.nn.Module):
    def __init__(self, cfg: Configuration):
        super().__init__()
        self.activation = torch.nn.ELU()
        self.cfg = Configuration()
        

        self.actor_encoder1 = torch.nn.Linear(cfg.env.num_actor_input, cfg.net.dim_actor_latent[0])
        self.actor_encoder2 = torch.nn.Linear(cfg.net.dim_actor_latent[0], cfg.net.dim_actor_latent[1])
        self.actor_encoder3 = torch.nn.Linear(cfg.net.dim_actor_latent[1], cfg.net.dim_actor_latent[2])
        self.actor_encoder4 = torch.nn.Linear(cfg.net.dim_actor_latent[2], cfg.net.dim_actor_latent[3])
        self.sense_encoder1 = torch.nn.Linear(cfg.env.num_sense_input, cfg.net.dim_sense_latent[0])
        self.sense_encoder2 = torch.nn.Linear(cfg.net.dim_sense_latent[0], cfg.net.dim_sense_latent[1])
        self.sense_encoder3 = torch.nn.Linear(cfg.net.dim_sense_latent[1], cfg.net.dim_sense_latent[2])
        self.sense_encoder4 = torch.nn.Linear(cfg.net.dim_sense_latent[2], cfg.env.num_sense_output)

        self.decoder = torch.nn.Linear(in_features=cfg.net.dim_actor_latent[3], out_features=cfg.env.num_output)

    def forward(self, inputs):
        actor_desc = inputs['actor_desc']
        sense_desc = inputs['sense_desc']
        
        sense_x = self.activation(self.sense_encoder1(sense_desc))
        sense_x = self.activation(self.sense_encoder2(sense_x))
        sense_x = self.activation(self.sense_encoder3(sense_x))
        sense_x = self.sense_encoder4(sense_x)

        x = torch.cat((actor_desc[ :self.cfg.net.actor_domain], sense_x), dim=0)
        dim_nospace = self.cfg.net.actor_domain + self.cfg.env.num_sense_output
        x = torch.cat((x, actor_desc[ dim_nospace:]), dim=0)

        actor_x = self.activation(self.actor_encoder1(x))
        actor_x = self.activation(self.actor_encoder2(actor_x))
        actor_x = self.activation(self.actor_encoder3(actor_x))
        actor_x = self.activation(self.actor_encoder4(actor_x))

        return self.decoder(actor_x)


class ModelLoad:
    def __init__(self, dtype=torch.float, device='cpu'):
        self.dtype = dtype
        self.device = device
        self.cfg = Configuration()
        self.net = ActorNetwork(self.cfg).to(self.device)
        self.dtype = dtype

    def load_model(self,):
        model = torch.load(self.cfg.model.path,map_location=self.device)['model_state_dict']#!!!!!!!,map_location=self.device
        self.remapping(model)
        
    def remapping(self, model):
        for n, p in self.net.named_parameters():
            if n == 'actor_encoder1.weight':
                p.data = model['actor.net.0.weight']
            if n == 'actor_encoder1.bias':
                p.data = model['actor.net.0.bias']
            if n == 'actor_encoder2.weight':
                p.data = model['actor.net.2.weight']
            if n == 'actor_encoder2.bias':
                p.data = model['actor.net.2.bias']
            if n == 'actor_encoder3.weight':
                p.data = model['actor.net.4.weight']
            if n == 'actor_encoder3.bias':
                p.data = model['actor.net.4.bias']
            if n == 'actor_encoder4.weight':
                p.data = model['actor.net.6.weight']
            if n == 'actor_encoder4.bias':
                p.data = model['actor.net.6.bias']
            if n == 'decoder.weight':
                p.data = model['actor.net.8.weight']
            if n == 'decoder.bias':
                p.data = model['actor.net.8.bias']
            if n == 'sense_encoder1.weight':
                p.data = model['actor.ontology_sense.net.0.weight']
            if n == 'sense_encoder1.bias':
                p.data = model['actor.ontology_sense.net.0.bias']
            if n == 'sense_encoder2.weight':
                p.data = model['actor.ontology_sense.net.2.weight']
            if n == 'sense_encoder2.bias':
                p.data = model['actor.ontology_sense.net.2.bias']
            if n == 'sense_encoder3.weight':
                p.data = model['actor.ontology_sense.net.4.weight']
            if n == 'sense_encoder3.bias':
                p.data = model['actor.ontology_sense.net.4.bias']
            if n == 'sense_encoder4.weight':
                p.data = model['actor.ontology_sense.net.6.weight']
            if n == 'sense_encoder4.bias':
                p.data = model['actor.ontology_sense.net.6.bias']

    def inference(self, input_state, input_sense):
        inputs = {}

        inputs['actor_desc'] = input_state
        inputs['sense_desc'] = input_sense
        outputs = self.net(inputs)
        return outputs

