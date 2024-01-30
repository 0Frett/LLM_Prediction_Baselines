# Import necessary packages.
import numpy as np
import pandas as pd
import torch
import os
import torch.nn as nn
from torchvision import transforms
import torchvision.transforms as transforms
from PIL import Image
# "ConcatDataset" and "Subset" are possibly useful when doing semi-supervised learning.
from torch.utils.data import ConcatDataset, DataLoader, Subset, Dataset
from torchvision.datasets import DatasetFolder, VisionDataset
# This is for the progress bar.
from tqdm.auto import tqdm
import json
import random
from transformers import AdamW, get_linear_schedule_with_warmup
import torch.nn as nn
import torch.optim as optim
from torchvision.models import resnet50
import argparse
from accelerate import Accelerator
import pickle
from transformers import AutoTokenizer, AutoModel
import pandas as pd
from sklearn.preprocessing import MinMaxScaler

# Fix random seed for reproducibility
def same_seeds(seed):
	torch.manual_seed(seed)
	if torch.cuda.is_available():
			torch.cuda.manual_seed(seed)
			torch.cuda.manual_seed_all(seed)
	np.random.seed(seed)
	random.seed(seed)
	torch.backends.cudnn.benchmark = False
	torch.backends.cudnn.deterministic = True

def get_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument('-s', '--seed', type=int, default=49)
    parser.add_argument('-d', '--device', type=str, default='cuda')
    parser.add_argument('-test', '--img_test_path', type=str, default='')
    parser.add_argument('-m', '--model_path', type=str, default='')
    parser.add_argument('-ms', '--mod_name', type=str, default='')

    return parser

class AIR(Dataset):
    def __init__(self, root_dir):
        self.root_dir = root_dir
        self.image_list = os.listdir(root_dir)      

    def __len__(self):
        return len(self.image_list)

    def __getitem__(self, idx):
        img_name = os.path.join(self.root_dir, self.image_list[idx])
        image = Image.open(img_name).convert('RGB')
        test_transform = transforms.Compose(
            [transforms.Resize(16), transforms.ToTensor()]
        )
        image = test_transform(image)
        tensor_1d = image.view(-1)
        label = float(self.image_list[idx].split('_')[0])

        return tensor_1d, label

class MultimodalModel(nn.Module):
    def __init__(self, pretrained_model_path):
        super(MultimodalModel, self).__init__()
        input_size = 768  
        self.concat = nn.Sequential(
            nn.Linear(input_size, 512),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(512, 32),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(32, 1)
        )
        self.concat.load_state_dict(torch.load(os.path.join(pretrained_model_path, "concat_best.ckpt")))

    def forward(self, x):
        output = self.concat(x)
        return output


if __name__ == '__main__':
    parser = get_parser()
    args = parser.parse_args()
    device = args.device if torch.cuda.is_available() else "cpu"
    mod = args.mod_name
    same_seeds(args.seed)

    model = MultimodalModel(args.model_path)
    test_set = AIR(args.img_test_path)
    criterion = nn.MSELoss()
    test_loader = DataLoader(test_set, batch_size=1, shuffle=False, pin_memory=True)

    model.to(device)

    print("Testing ...")
    model.eval()
    valid_loss = []
    with torch.no_grad():
        dev_loss= 0
        for i, data in enumerate(tqdm(test_loader)):
            emb, label = data
            label = label.view(-1, 1)
            emb, label = emb.to(device), label.to(device)
            emb = emb.to(torch.float32)
            output = model.forward(emb)

            label = label.to(torch.float32)
            loss = criterion(output, label)
            valid_loss.append(loss.item())

    print(f"{mod}")
    print(f"Testing | loss = {(sum(valid_loss) / len(valid_loss)):.3f}")
