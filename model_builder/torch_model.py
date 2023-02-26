import os
import copy
import numpy as np
import torch
import torch.nn as nn
from tqdm import tqdm
from torch import optim
from model_builder.modules import Generator_conditional, EMA_model
import pandas as pd
from torch.utils.data import TensorDataset, DataLoader

def get_data(args):
    data_path = args["dataset_path"]
    data = pd.read_csv(data_path)
    label = torch.from_numpy(data["sex"].to_numpy())
    inputs = torch.from_numpy(data.drop("sex",axis=1).to_numpy())
    dataset = TensorDataset(inputs,label)
    return DataLoader(dataset)

class Diffusion:
    def __init__(self, noise_steps=1000, beta_start=1e-4, beta_end=0.02, data_shape=15):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        self.noise_steps = noise_steps
        self.beta_start = beta_start
        self.beta_end = beta_end

        self.beta = self.prepare_noise_schedule().to(self.device)
        self.alpha = 1. - self.beta
        self.alpha_hat = torch.cumprod(self.alpha, dim=0)

        self.data_shape = data_shape 

    def prepare_noise_schedule(self):
        #return torch.cos(torch.linspace(self.beta_start, self.beta_end, self.noise_steps))
        return torch.linspace(self.beta_start, self.beta_end, self.noise_steps)

    def noise_data(self, x,t):
        sqrt_alpha_hat = torch.sqrt(self.alpha_hat[t])#[:, None]
        sqrt_one_minus_alpha_hat = torch.sqrt(1 - self.alpha_hat[t])#[:, None]
        Ɛ = torch.normal(0.5, 0.24, size=x.shape).to(self.device)#torch.randn_like(x)
        return sqrt_alpha_hat * x + sqrt_one_minus_alpha_hat * Ɛ, Ɛ

    def sample_timesteps(self, n):
        n_shape = n.shape[0] 
        return torch.randint(low=1, high=self.noise_steps, size=(n_shape,))

    



def train(args):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    dataloader = get_data(args)
    model = Generator_conditional(num_classes=args["num_classes"])
    model = model.to(device)
    optimizer = optim.Adam(model.parameters(), lr=args["lr"])
    mse = nn.MSELoss()
    
    diffusion = Diffusion()
    l = len(dataloader)
    ema = EMA_model(0.995)
    ema_model = copy.deepcopy(model).eval().requires_grad_(False)


    best_mae_score = 100
    for epoch in range(args["epochs"]):
        mae = nn.L1Loss()
        origin_data = []
        predicted_data = []
        pbar = tqdm(dataloader)
        for i, (input_data, labels) in enumerate(pbar):
            input_data = torch.squeeze(input_data).type(torch.float) #The reason for this line is cuda is producing none reasonable error if the shape isn't [15], probably can't use batch_size, (the error isn't produced if we run on cpu)
            origin_data.append(input_data)
            input_data = input_data.to(device)
            labels = labels.to(device)
            t = diffusion.sample_timesteps(input_data).to(device)
            x_t, noise = diffusion.noise_data(input_data, t)
            x_t = x_t.type(torch.float).to(device)
            noise = noise.type(torch.float).to(device)

            if np.random.random() < 0.1:
                labels = None
            predicted_noise = model(x_t, t, labels)
            predicted_data.append(predicted_noise)
            loss = mse(predicted_noise, input_data)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            ema.step_ema(ema_model, model)

            pbar.set_postfix(MSE=loss.item())


        if epoch % 10 == 0:

            print(f"epoch: {epoch} out of {args['epochs']}")
            run_name = args["run_name"]
            main_path = os.path.join("models", run_name)
            if not os.path.exists(main_path):
                os.makedirs(main_path)
            torch.save(optimizer.state_dict(), os.path.join(main_path, f"optim.pt"))
        mae_score = mae(predicted_noise,input_data)
        if mae_score < best_mae_score:
            torch.save(model, os.path.join(main_path, f"model.pt"))
            torch.save(ema_model, os.path.join(main_path, f"ema_model.pt"))
            print("-----------------saved model-----------------")
            best_mae_score = mae_score
        print(f"Mean Absolute Error: {mae_score}")
        origin_data = []
        predicted_data = []


def launch():
    import argparse
    parser = argparse.ArgumentParser()
    args = parser.parse_args()
    args.epochs = 300
    args.run_name = "DDPM_conditional"
    args.batch_size = 3
    args.data_shape = 15
    args.num_classes = 2
    args.dataset_path = "/media/akm/My Work/Programming/AWS/AWS nano degree final project/Dataset/processed_data.csv"
    args.lr = 0.0001
    train(args)

