import torch
import numpy as np
from autoencoder import CAE
import pickle
import random


class Model(object):

    def __init__(self, modelname):
        self.model = CAE()
        model_dict = torch.load(modelname, map_location='cpu')
        self.model.load_state_dict(model_dict)
        self.model.eval

    def process(self, traj):
        return self.model.process(traj)

    def forward(self, traj):
        return self.model.forward(traj)

    def encoder(self, x):
        return self.model.encoder(x)

    def decoder(self, z_with_context):
        return self.model.decoder(z_with_context)



def predict(traj, model):
    x, context, r_star = model.process(traj)
    z = model.encoder(x)
    z -= 1.0
    z_with_context = torch.cat((z, context), 0)
    r_pred = model.decoder(z_with_context)
    return r_star.numpy(), r_pred.detach().numpy()


def get_latent(traj, model):
    x, context, r_star = model.process(traj)
    z = model.encoder(x)
    return z.item()



def main():

    dataname = "dataset.pkl"
    modelname = 'CAE_model.pt'

    data = pickle.load(open(dataname, "rb"))
    model = Model(modelname)

    r_star, r_pred = predict(data[0], model)
    print(r_star, r_pred)



if __name__ == "__main__":
    main()
