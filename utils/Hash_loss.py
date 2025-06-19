import torch
import torch.nn.functional as F
from loguru import logger

class DPSHLoss(torch.nn.Module):
    def __init__(self, config, bit):
        super(DPSHLoss, self).__init__()
        self.U = torch.zeros(config["num_train"], bit).half().to(config["device"])
        # self.U = torch.zeros(config["num_train"], bit).float().to(config["device"])
        self.Y = torch.zeros(config["num_train"], config["n_class"]).float().to(config["device"])
    
        # self.one = torch.ones(config["batch_size"], bit).float().to(config["device"])
        self.bit = bit
        # self.cls_loss = torch.nn.CrossEntropyLoss()


    def forward(self, u, y, ind, config):
        B,N = u.shape
        self.U[ind, :] = u.data
        self.Y[ind, :] = y.float()

        s = (y @ self.Y.t() > 0).float()
        inner_product = u @ self.U.t() * 0.5

        likelihood_loss = torch.log(1 + (torch.exp(-(torch.abs(inner_product))))) + inner_product.clamp(min=0) - s * inner_product

        likelihood_loss = likelihood_loss.mean()

        # quantization_loss =(u.abs()-1).pow(2).mean()
        quantization_loss = (u - u.sign()).pow(2).mean()

        return likelihood_loss + config["alpha"]*quantization_loss, likelihood_loss, quantization_loss
        # return likelihood_loss





