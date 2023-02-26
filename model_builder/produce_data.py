import torch
import os
import numpy as np

class noise_data_producer():
    def __init__(self,model=None,model_path=None) -> None:
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        if model:
            self.model = model.to(self.device)
        else:
            self.model = self.load_model(model_path).to(self.device)

    def produce_fake_data(self,num_of_data: int, label: int = None):
        produced_data = []
        for i in range(num_of_data):
            if not label:
                if np.random.random() < 0.95:
                    label = torch.tensor([0]).to(self.device)
                else:
                    label = torch.tensor([1]).to(self.device)
                #label = torch.randint(low=0,high=2,size=(1,)).to(self.device)
            noise_sample,time_step = self.produce_noise_sample()
            predicted_data =np.array(self.model(noise_sample,time_step,label).cpu())

            produced_data.append(np.concatenate((predicted_data,np.array(label.cpu())),axis=0))
        return produced_data
    
    def produce_noise_sample(self, x=[15],):
        "returns a noisy data and a time_step"
        time_step = torch.ones(x).to(self.device).type(torch.float).to(self.device)
       # x = torch.tensor(time_step).to(device)
        noise =torch.normal(0.5, 0.24, size=time_step.shape).to(self.device)
        #noise = torch.randn_like(time_step).to(self.device)
        return noise,time_step
    
    def load_model(self,model_path):
        print("loaded model")
        path = os.path.join("models", "DDPM_conditional","ema_model.pt")
        return torch.load(path)
    

