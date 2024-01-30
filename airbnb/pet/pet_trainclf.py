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
    parser.add_argument('-s', '--seed', type=int, default=42)
    parser.add_argument('-d', '--device', type=str, default='cuda')
    parser.add_argument('-train', '--img_train_path', type=str, default='')
    parser.add_argument('-valid', '--img_valid_path', type=str, default='')
    parser.add_argument('-test', '--img_test_path', type=str, default='')
    parser.add_argument('-ms', '--model_save_dir', type=str, default='')
    parser.add_argument('--is_reg', action='store_true')

    return parser


class DData(Dataset):
    def __init__(self, root_dir:str, is_reg:bool):
        self.root_dir = root_dir
        self.image_list = os.listdir(root_dir)
        self.is_reg = is_reg    

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
        if self.is_reg:
            label = float(self.image_list[idx].split('_')[0])
        else:
            label = int(self.image_list[idx].split('_')[0])

        return tensor_1d, label

class MultimodalModel(nn.Module):
    def __init__(self):
        super(MultimodalModel, self).__init__()
        input_size = 768 
        self.concat = nn.Sequential(
            nn.Linear(input_size, 512),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(512, 32),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(32, 5)
        )
    def forward(self, x):
        output = self.concat(x)
        return output

if __name__ == '__main__':
    parser = get_parser()
    args = parser.parse_args()
    device = args.device if torch.cuda.is_available() else "cpu"
    model_save_dir = args.model_save_dir
    os.makedirs(model_save_dir, exist_ok=True)
    same_seeds(args.seed)
    is_reg = args.is_reg

    train_set = DData(args.img_train_path, is_reg)
    valid_set = DData(args.img_valid_path, is_reg)
    test_set = DData(args.img_test_path, is_reg)
    model = MultimodalModel()

    # hyperparameters
    num_epoch = 500
    validation = True
    logging_step = 10
    learning_rate = 1e-5
    early_stop_step = 50
    optimizer = AdamW(model.parameters(), lr=learning_rate)
    if is_reg:
        criterion = nn.MSELoss()
    else:
        criterion = nn.CrossEntropyLoss()
    
    train_batch_size = 4
    gradient_accumulation_steps = 1

    train_loader = DataLoader(train_set, batch_size=train_batch_size, shuffle=True, pin_memory=True)
    dev_loader = DataLoader(valid_set, batch_size=16, shuffle=False, pin_memory=True)
    test_loader = DataLoader(test_set, batch_size=1, shuffle=False, pin_memory=True)

    # Change "fp16_training" to True to support automatic mixed 
    # precision training (fp16)	
    fp16_training = True
    if fp16_training:    
        accelerator = Accelerator(mixed_precision="fp16")
    else:
        accelerator = Accelerator()

    total_step = (len(train_loader)//gradient_accumulation_steps) * num_epoch

    model, optimizer, train_loader = accelerator.prepare(model, optimizer, train_loader) 
    model.to(device)

    model.train()

    print("Start Training ...")
    best_acc = 0.0
    train_losses_his = []
    valid_losses_his = []
    early_stop = 0
    for epoch in range(num_epoch):
        train_loss = []
        train_accs = []
        
        for index, data in enumerate(tqdm(train_loader)):	
            # Load all data into GPU
            emb, label = data
            #label = label.view(-1, 1)
            emb, label = emb.to(device), label.to(device)
            emb = emb.to(torch.float32)
            output = model.forward(emb)

            label = label.to(torch.long)
            #label = label.to(torch.float32)
            loss = criterion(output, label)
            optimizer.zero_grad()
            accelerator.backward(loss)
            optimizer.step()

            acc = (output.argmax(dim=-1) == label).float().mean()

            train_loss.append(loss.item())
            train_accs.append(acc)

        # Print training loss and accuracy over past logging step
        print(f"Epoch {epoch + 1} | acc = {(sum(train_accs) / len(train_accs)):.3f} | loss = {(sum(train_loss) / len(train_loss)):.3f}")
        train_losses_his.append((sum(train_loss) / len(train_loss)))
        

        if validation:
            print("Evaluating Dev Set ...")
            model.eval()
            valid_loss = []
            valid_accs = []
            with torch.no_grad():
                dev_loss= 0
                for index, data in enumerate(tqdm(dev_loader)):
                    emb, label = data
                    #label = label.view(-1, 1)
                    emb, label = emb.to(device), label.to(device)
                    emb = emb.to(torch.float32)
                    output = model.forward(emb)

                    label = label.to(torch.long)
                    #label = label.to(torch.float32)
                    loss = criterion(output, label)
                    acc = (output.argmax(dim=-1) == label).float().mean()
                    valid_loss.append(loss.item())
                    valid_accs.append(acc)
                print(f"Validation | Epoch {epoch + 1} | acc = {(sum(valid_accs) / len(valid_accs)):.3f} | loss = {(sum(valid_loss) / len(valid_loss)):.3f}")
                valid_losses_his.append((sum(valid_loss)/ len(valid_loss)))
            model.train()

            if best_acc < (sum(valid_accs) / len(valid_accs)):
                early_stop = 0
                print(f"Best model found at epoch {epoch}, saving model")
                torch.save(model.concat.state_dict(), os.path.join(model_save_dir,"concat_best.ckpt"))
                best_acc = (sum(valid_accs) / len(valid_accs))
            else:
                early_stop += 1
        if early_stop > early_stop_step:
            break

    print("best_acc :", best_acc)
    with open(f'./{model_save_dir}/train_loss.pkl', 'wb') as f:
        pickle.dump(train_losses_his, f)
    with open(f'./{model_save_dir}/valid_loss.pkl', 'wb') as f:
        pickle.dump(valid_losses_his, f)