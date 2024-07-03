import numpy as np
import os
import torch
from torch import nn
from torch import optim
from torch.optim.lr_scheduler import ReduceLROnPlateau
from torch.utils.data import DataLoader
import torch.nn.functional as F
import pytorch_lightning as pl
from dataset import AirfoilDataset


class NetworkModule(pl.LightningModule):
    
    ########## ---------- INITIALIZATION ---------- ##########
    
    def __init__(self,
                 architecture = 'paper',
                 pin_memory=False,
                 lr_patience=2,
                 lr_factor=0.1,
                 lr_threshold=1e-4,
                 optimizer="Adam",
                 batch_size=64,
                 learning_rate=1e-2,
                 airfoil_shape="simple",
                 angles = "full",
                 normalize = True,
                 train_airfoils = "all",
                 val_airfoils = "all",
                 test_airfoils = "all",
                 verbose = False,
                 weight_penalty = 0.2,
                 reg_layers = [1,1,1,1,1,1]
                 ) -> None:
        super().__init__()
        
        self.learning_rate  = learning_rate
        self.batch_size     = batch_size
        self.optimizer      = optimizer
        self.lr_factor      = lr_factor
        self.lr_patience    = lr_patience
        self.lr_threshold   = lr_threshold
        self.pin_memory     = pin_memory
        self.airfoil_shape  = airfoil_shape
        self.angles         = angles
        self.normalize      = normalize
        self.architecture   = architecture
        self.verbose        = verbose
        self.train_airfoils = train_airfoils
        self.val_airfoils   = val_airfoils
        self.test_airfoils  = test_airfoils
        self.weight_penalty = weight_penalty
        self.reg_layers = reg_layers
        
        self.construct_network()
        
    def configure_optimizers(self):
        optimizer = optim.Adam(self.parameters(),lr=self.learning_rate)
        lr_scheduler = ReduceLROnPlateau(optimizer=optimizer,
                                         mode="min",
                                         factor=self.lr_factor,
                                         patience=self.lr_patience,
                                         verbose=self.verbose,
                                         threshold=self.lr_threshold,
                                         min_lr=1e-7)
        return {"optimizer": optimizer, 
                "lr_scheduler": {"scheduler":lr_scheduler, 
                                "monitor":  "val_loss"}}

    ########## ---------- DATALOADERS ---------- ##########

    def prepare_data(self) -> None:
        return super().prepare_data()
    
    def train_dataloader(self):
        if self.train_airfoils == "all":
            self.train_airfoils = np.arange(6,30)
            split = "train"
        else:
            split = "all"
        return DataLoader(AirfoilDataset(mode=split,foils=self.train_airfoils,angles=self.angles),
                          batch_size=self.batch_size,
                          pin_memory=self.pin_memory,
                          shuffle=True)
    def val_dataloader(self):
        if self.val_airfoils == "all":
            self.val_airfoils = np.arange(6,30)
            split = "val"
        else:
            split = "all"
        return DataLoader(AirfoilDataset(mode=split,foils=self.val_airfoils,angles=self.angles),
                        batch_size=self.batch_size,
                        pin_memory=self.pin_memory,
                        shuffle=False)
    def test_dataloader(self):
        if self.test_airfoils == "all":
            self.test_airfoils = np.arange(6,30)
            split = "test"
        else:
            split = "all"
        return DataLoader(AirfoilDataset(mode=split,foils=self.test_airfoils,angles=self.angles),
                        batch_size=self.batch_size,
                        pin_memory=self.pin_memory,
                        shuffle=False)

    ########## ---------- FORWARD PASS ---------- ##########

    def training_step(self,batch,bid):
        loss = self.general_step(batch)
        self.log("train_loss",loss,on_step=False,on_epoch=True)
        return loss

    def validation_step(self,batch,bid):
        loss = self.general_step(batch)
        self.log("val_loss",loss,on_step=False,on_epoch=True)
        return loss
    
    def test_step(self,batch,bid):
        loss = self.general_step(batch)
        self.log("test_loss",loss,on_step=False,on_epoch=True)
        return loss

    def general_step(self,batch):
        # Compile input
        (input,target) = batch
        shape = input[1]
        supp = input[2]
        
        if self.architecture == 'paper': 
            target = target[:,0,None]
        if "nophys" in self.architecture:
            supp = supp[:,3:]
        
        # Run forward and backward pass
        output = self.forward((shape,supp))
        loss = F.mse_loss(output,target)
        
        if self.architecture == "regularization":
            for i in range(6):
                if self.reg_layers[i]:
                    loss = loss + self.weight_penalty * torch.mean(torch.abs(self.model[i+1][0].weight[:,-5:]))
        
        return loss
    
    def forward(self, input):
        (shape,supp) = input

        if self.architecture == "regularization":
            x = self.model[0](shape)
            for l in range(6):
                if self.reg_layers[l]:
                    x = torch.cat((x,supp),dim=1)
                x = self.model[l+1](x)
            return x
        
        # Turn shape into latent space
        latent = self.encoder(shape)
        # Add supplementary physics data
        concat = torch.cat((latent,supp),dim=1)
        # Run second part to estimate coefficients
        coeff = self.regressor(concat)
        return coeff

    ########## ---------- NETWORK ---------- ##########
    
    def construct_network(self):
        
        if self.architecture == "dummy":
            self.encoder = nn.Sequential(
                nn.Linear(40,20)
            )
            self.regressor = nn.Sequential(
                nn.Linear(25,3)
            )
        
        if self.architecture == "regularization":
            # First layer
            layers = [nn.Sequential(nn.Linear(40,20),nn.ReLU())]

            # 5 Hidden layers
            for l in range(len(self.reg_layers)-1):
                if self.reg_layers[l]:
                    layers.append(nn.Sequential(nn.Linear(25,20),nn.ReLU()))
                else:
                    layers.append(nn.Sequential(nn.Linear(20,20),nn.ReLU()))

            # Output layer
            if self.reg_layers[-1]:
                layers.append(nn.Sequential(nn.Linear(25,3),nn.ReLU()))
            else:
                layers.append(nn.Sequential(nn.Linear(20,3),nn.ReLU()))

            self.model = nn.ModuleList(layers)
        
        if self.architecture == "paper":
            self.encoder = nn.Sequential(
                nn.Linear(40,20),nn.ReLU(),
                nn.Linear(20,20),nn.ReLU(),
                nn.Linear(20,20),nn.ReLU(),
            )
            self.regressor = nn.Sequential(
                nn.Linear(25,20),nn.ReLU(),
                nn.Linear(20,1))
        
        if self.architecture == "l6_i2":
            self.encoder = nn.Sequential(
                nn.Linear(40,20),nn.ReLU(),
                nn.Linear(20,20),nn.ReLU())
            self.regressor = nn.Sequential(
                nn.Linear(25,20),nn.ReLU(),
                nn.Linear(20,20),nn.ReLU(),
                nn.Linear(20,20),nn.ReLU(),
                nn.Linear(20,20),nn.ReLU(),
                nn.Linear(20,3))

        if self.architecture == "l6_i2_nophys":
            self.encoder = nn.Sequential(
                nn.Linear(40,20),nn.ReLU(),
                nn.Linear(20,20),nn.ReLU())
            self.regressor = nn.Sequential(
                nn.Linear(22,20),nn.ReLU(),
                nn.Linear(20,20),nn.ReLU(),
                nn.Linear(20,20),nn.ReLU(),
                nn.Linear(20,20),nn.ReLU(),
                nn.Linear(20,3))
    
        if self.architecture == "l6_i3":
            self.encoder = nn.Sequential(
                nn.Linear(40,20),nn.ReLU(),
                nn.Linear(20,20),nn.ReLU(),
                nn.Linear(20,20),nn.ReLU())
            self.regressor = nn.Sequential(
                nn.Linear(25,20),nn.ReLU(),
                nn.Linear(20,20),nn.ReLU(),
                nn.Linear(20,20),nn.ReLU(),
                nn.Linear(20,3))
    
        if self.architecture == "l6_i5":
            self.encoder = nn.Sequential(
                nn.Linear(40,20),nn.ReLU(),
                nn.Linear(20,20),nn.ReLU(),
                nn.Linear(20,20),nn.ReLU(),
                nn.Linear(20,20),nn.ReLU(),
                nn.Linear(20,20),nn.ReLU())
            self.regressor = nn.Sequential(
                nn.Linear(25,20),nn.ReLU(),
                nn.Linear(20,3))

        if self.architecture == "l6_i6":
            self.encoder = nn.Sequential(
                nn.Linear(40,20),nn.ReLU(),
                nn.Linear(20,20),nn.ReLU(),
                nn.Linear(20,20),nn.ReLU(),
                nn.Linear(20,20),nn.ReLU(),
                nn.Linear(20,20),nn.ReLU(),
                nn.Linear(20,20),nn.ReLU())
            self.regressor = nn.Sequential(
                nn.Linear(25,3))

        if self.architecture == "l6_i6_nophys":
            self.encoder = nn.Sequential(
                nn.Linear(40,20),nn.ReLU(),
                nn.Linear(20,20),nn.ReLU(),
                nn.Linear(20,20),nn.ReLU(),
                nn.Linear(20,20),nn.ReLU(),
                nn.Linear(20,20),nn.ReLU(),
                nn.Linear(20,20),nn.ReLU())
            self.regressor = nn.Sequential(
                nn.Linear(22,3))

        if self.architecture == "l6_i4":
            self.encoder = nn.Sequential(
                nn.Linear(40,20),nn.ReLU(),
                nn.Linear(20,20),nn.ReLU(),
                nn.Linear(20,20),nn.ReLU(),
                nn.Linear(20,20),nn.ReLU())
            self.regressor = nn.Sequential(
                nn.Linear(25,20),nn.ReLU(),
                nn.Linear(20,20),nn.ReLU(),
                nn.Linear(20,3))

        if self.architecture == "l6_i4_nophys":
            self.encoder = nn.Sequential(
                nn.Linear(40,20),nn.ReLU(),
                nn.Linear(20,20),nn.ReLU(),
                nn.Linear(20,20),nn.ReLU(),
                nn.Linear(20,20),nn.ReLU())
            self.regressor = nn.Sequential(
                nn.Linear(22,20),nn.ReLU(),
                nn.Linear(20,20),nn.ReLU(),
                nn.Linear(20,3))

        if self.architecture == "l6_i5_nophys":
            self.encoder = nn.Sequential(
                nn.Linear(40,20),nn.ReLU(),
                nn.Linear(20,20),nn.ReLU(),
                nn.Linear(20,20),nn.ReLU(),
                nn.Linear(20,20),nn.ReLU(),
                nn.Linear(20,20),nn.ReLU())
            self.regressor = nn.Sequential(
                nn.Linear(22,20),nn.ReLU(),
                nn.Linear(20,3))

        if self.architecture == "l6_i3_nophys":
            self.encoder = nn.Sequential(
                nn.Linear(40,20),nn.ReLU(),
                nn.Linear(20,20),nn.ReLU(),
                nn.Linear(20,20),nn.ReLU())
            self.regressor = nn.Sequential(
                nn.Linear(22,20),nn.ReLU(),
                nn.Linear(20,20),nn.ReLU(),
                nn.Linear(20,20),nn.ReLU(),
                nn.Linear(20,3))
        
        if self.architecture == "l6_i1_nophys":
            self.encoder = nn.Sequential(
                nn.Linear(40,20),nn.ReLU())
            self.regressor = nn.Sequential(
                nn.Linear(22,20),nn.ReLU(),
                nn.Linear(20,20),nn.ReLU(),
                nn.Linear(20,20),nn.ReLU(),
                nn.Linear(20,20),nn.ReLU(),
                nn.Linear(20,20),nn.ReLU(),
                nn.Linear(20,3))
        
        if self.architecture == "l6_i1":
            self.encoder = nn.Sequential(
                nn.Linear(40,20),nn.ReLU())
            self.regressor = nn.Sequential(
                nn.Linear(25,20),nn.ReLU(),
                nn.Linear(20,20),nn.ReLU(),
                nn.Linear(20,20),nn.ReLU(),
                nn.Linear(20,20),nn.ReLU(),
                nn.Linear(20,20),nn.ReLU(),
                nn.Linear(20,3))
        
        
        
    
    def count_parameters(self):
        return int(sum(p.numel() for p in self.parameters() if p.requires_grad))