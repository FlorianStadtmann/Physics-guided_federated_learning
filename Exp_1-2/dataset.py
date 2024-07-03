import numpy as np
import torch
from torch.utils import data
import pickle
import turbine_data as turbine

class AirfoilDataset(data.Dataset):
    def __init__(self,
                 mode="all",
                 foils=np.arange(6,30),
                 angles="full") -> None:
        '''mode: "all", "train", "val", "test". Split (from selected foils) a subset into (train/val/test) or use the entire foil(s).
        foils: list of airfoils to draw from.
            If separating sets by using different airfoils for training, validation and testing, declare foils and set mode="all".
        angles: "limited" [-30,30] or "full" - all angles
        foilshape: "simple" or "full", simple uses only (reduced from 200) 20 coordinates to describe a shape (40 floats)
        normalize: Bool, use normalised input (strongly recommended)
        '''
        # Load the dataset (dictionary, 1.3MB)
        
        #self.filename = "dataset/tensored_data.pickle"
        self.filename = "dataset/tensored_data.pickle"
        with open(self.filename,"rb") as f:
            self.data = pickle.load(f)
        
        # Angles of attack
        if angles=="limited":
            aots = turbine.get_aots()[50:150]
        if angles=="full":
            aots = turbine.get_aots()
        
        # This is the actual length of the dataset.
        self.keys = [(foil,a) for foil in foils for a in aots]
        
        # Train, val and test set 80/10/10 split
        # Shuffling with the same seed ensures there is no leakage between sets
        N = len(self.keys)
        indices = np.arange(N)
        np.random.seed(42)
        np.random.shuffle(indices)
        split1 = N//10
        split2 = N//5
        self.test_set   = indices[:split1]
        self.val_set    = indices[split1:split2]
        self.train_set  = indices[split2:]
        
        if mode == "train": self.keys = [self.keys[i] for i in self.train_set]
        if mode == "val":   self.keys = [self.keys[i] for i in self.val_set]
        if mode == "test":  self.keys = [self.keys[i] for i in self.test_set]
        
    def __len__(self):
        return len(self.keys)
    
    def __getitem__(self, index):
        '''
        Returns a sample
            (input, target)
                input[0]: Bool; whether simplified physics result is available
                input[1]: 1d tensor: Shape of airfoil; a list of coordinates (x and y)
                input[2]: 1d tensor: Supplementary information. Physics result (if available, otherwise 0) and flow conditions
            
                target: tensor of airfoil coefficients
        '''
        return self.data[self.keys[index]]