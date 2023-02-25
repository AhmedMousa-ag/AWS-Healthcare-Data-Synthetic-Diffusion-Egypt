import os
import copy
import numpy as np
import torch
import torch.nn as nn
from tqdm import tqdm
from torch import optim
from model_builder.modules import Generator_conditional, EMA
from torch.utils.tensorboard import SummaryWriter
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
    def __init__(self, noise_steps=1000, beta_start=1e-4, beta_end=0.02, data_shape=16): #device="cuda:0"):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu") #device

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
        #x = x
        #print(f"x device: {x.get_device()}, t device: {t.get_device()}")
        #x_shape = x.shape#.reshape(-1,1).shape#.to(self.device)
        #print(f"x shape: {x_shape}")
        #t = t.type(torch.float).to(self.device)
        #t = (torch.matmul(torch.ones(x_shape).to(self.device), t)).long()#.to(self.device)
        sqrt_alpha_hat = torch.sqrt(self.alpha_hat[t])[:, None]
        sqrt_one_minus_alpha_hat = torch.sqrt(1 - self.alpha_hat[t])[:, None]
        Ɛ = torch.randn_like(x)
        return sqrt_alpha_hat * x + sqrt_one_minus_alpha_hat * Ɛ, Ɛ

    def sample_timesteps(self, n):
        #print(f"n shape: {n.shape}")
        n_shape = n.shape[0] #torch.squeeze(n)
        return torch.randint(low=1, high=self.noise_steps, size=(n_shape,))

    def sample(self, model, n, labels, cfg_scale=3):
        n = n#.to(self.device)
        labels = labels
        model = model
        model.eval()
        with torch.no_grad():
            x = torch.randn(n).to(self.device)
            for i in tqdm(reversed(range(1, self.noise_steps)), position=0):
                t = (torch.ones(n).to(self.device) * i).long().to(self.device)
                predicted_noise = model(x, t, labels)
                if cfg_scale > 0:
                    uncond_predicted_noise = model(x, t, None)
                    predicted_noise = torch.lerp(uncond_predicted_noise, predicted_noise, cfg_scale)
                alpha = self.alpha[t][:, None]
                alpha_hat = self.alpha_hat[t][:, None]
                beta = self.beta[t][:, None]
                if i > 1:
                    noise = torch.randn_like(x)
                else:
                    noise = torch.zeros_like(x)
                x = 1 / torch.sqrt(alpha) * (x - ((1 - alpha) / (torch.sqrt(1 - alpha_hat))) * predicted_noise) + torch.sqrt(beta) * noise
        model.train()
        x = (x.clamp(-1, 1) + 1) / 2
        x = (x * 255).type(torch.uint8)
        return x

def train(args):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    dataloader = get_data(args)
    model = Generator_conditional(num_classes=args["num_classes"])
    model = model.to(device)
    optimizer = optim.AdamW(model.parameters(), lr=args["lr"])
    mse = nn.MSELoss()
    diffusion = Diffusion()
    l = len(dataloader)
    ema = EMA(0.995)
    ema_model = copy.deepcopy(model).eval().requires_grad_(False)

    for epoch in range(args["epochs"]):
        pbar = tqdm(dataloader)
        for i, (input_data, labels) in enumerate(pbar):
            input_data = torch.squeeze(input_data) #The reason for this line is cuda is producing none reasonable error if the shape isn't [16], probably can't use batch_size, (the error isn't produced if we run on cpu)
            input_data = input_data.to(device)
            labels = labels.to(device)
            t = diffusion.sample_timesteps(input_data).to(device)
            x_t, noise = diffusion.noise_data(input_data, t)#.to(device)
            x_t = x_t.type(torch.float).to(device)
            noise = noise.type(torch.float).to(device)
            if np.random.random() < 0.1:
                labels = None
            predicted_noise = model(x_t, t, labels)
            loss = mse(noise, predicted_noise)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            ema.step_ema(ema_model, model)

            pbar.set_postfix(MSE=loss.item())

        if epoch % 10 == 0:
            #labels = torch.arange(10).long()#.to(device)
           # sampled_images = diffusion.sample(model, n=len(labels), labels=labels)
           # ema_sampled_images = diffusion.sample(ema_model, n=len(labels), labels=labels)
            print(f"epoch: {epoch} out of {args['epochs']}")
            run_name = args["run_name"]
            main_path = os.path.join("models", run_name)
            if not os.path.exists(main_path):
                os.makedirs(main_path)
            torch.save(model.state_dict(), os.path.join(main_path, f"ckpt.pt"))
            torch.save(ema_model.state_dict(), os.path.join(main_path, f"ema_ckpt.pt"))
            torch.save(optimizer.state_dict(), os.path.join(main_path, f"optim.pt"))


def launch():
    import argparse
    parser = argparse.ArgumentParser()
    args = parser.parse_args()
    args.run_name = "DDPM_conditional"
    args.epochs = 300
    args.batch_size = 3
    args.data_shape = 16
    args.num_classes = 2
    args.dataset_path = "/media/akm/My Work/Programming/AWS/AWS nano degree final project/Dataset/processed_data.csv"
   # args.device = "cuda:0"
    args.lr = 3e-4
    train(args)

