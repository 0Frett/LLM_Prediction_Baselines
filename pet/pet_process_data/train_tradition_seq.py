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
from torch import einsum
import torch.nn.functional as F
import torch.optim as optim
from torchvision.models import resnet50
import argparse
from accelerate import Accelerator
import pickle
from transformers import AutoTokenizer, AutoModel
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
from transformers import ViTImageProcessor
from einops import rearrange, repeat
from einops.layers.torch import Rearrange

import math
import warnings


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
    parser.add_argument('-df', '--df_path', type=str, default='./data/split_data/train.csv')
    parser.add_argument('-dfv', '--df_valid_path', type=str, default='./data/split_data/valid.csv')
    parser.add_argument('--df_test_path', type=str, default='./data/split_data/test.csv')
    parser.add_argument('--train_num_df', type=str, default='./data/split_data/train_num.csv')
    parser.add_argument('--valid_num_df', type=str, default='./data/split_data/valid_num.csv')
    parser.add_argument('--test_num_df', type=str, default='./data/split_data/test_num.csv')
    parser.add_argument('--train_categ_df', type=str, default='./data/split_data/train_categ.csv')
    parser.add_argument('--valid_categ_df', type=str, default='./data/split_data/valid_categ.csv')
    parser.add_argument('--test_categ_df', type=str, default='./data/split_data/test_categ.csv')
    parser.add_argument('-tm', '--pretrained_text_model', type=str, default='allenai/longformer-base-4096')
    parser.add_argument('-tim', '--pretrained_image_model', type=str, default='google/vit-base-patch16-224')
    parser.add_argument('-img', '--img_dir', type=str, default='./data/train_images')
    parser.add_argument('-ms', '--model_save_dir', type=str, default='saved_traditional_seq_full')
    parser.add_argument('--text_ok', action='store_true')
    parser.add_argument('--img_ok', action='store_true')
    parser.add_argument('--tabular_ok', action='store_true')
    return parser


class NumericalEmbedder(nn.Module):
    def __init__(self, dim, num_numerical_types):
        super().__init__()
        self.weights = nn.Parameter(torch.randn(num_numerical_types, dim))
        self.biases = nn.Parameter(torch.randn(num_numerical_types, dim))

    def forward(self, x):
        x = rearrange(x, 'b n -> b n 1')
        return x * self.weights + self.biases

class PetFinderDataset(Dataset):
    def __init__(self, df, img_dir, categ_df, num_df, pretrained_text_model='allenai/longformer-base-4096', pretrained_image_model ='google/vit-base-patch16-224', transform=None):
        self.df = df
        self.categ_df = categ_df
        self.num_df = num_df
        self.img_dir = img_dir
        self.transform = transform
        self.image_processor = ViTImageProcessor.from_pretrained(pretrained_image_model)
        self.tokenizer = AutoTokenizer.from_pretrained(pretrained_text_model)

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        # Image
        img_name = os.path.join(self.img_dir, self.df.iloc[idx]['PetID']+'-1.jpg')
        image = Image.open(img_name)
        image = image.convert('RGB')
        if self.transform:
            image_input = self.transform(image)
        
        image_input = self.image_processor(images=image, return_tensors="pt")

        # Concatenate title and description
        text = "[CLS] " + str(self.df.iloc[idx]['Description']) + " [SEP] "
        text_input = self.tokenizer(text, max_length=512, add_special_tokens=False, truncation=True, 
                          padding='max_length', return_tensors="pt",return_token_type_ids=True)
        input_ids = text_input['input_ids'].squeeze(0)
        token_type_ids = text_input['token_type_ids'].squeeze(0)
        attention_mask = text_input['attention_mask'].squeeze(0)

        label = torch.tensor(self.df.iloc[idx]['AdoptionSpeed'], dtype=torch.int)
        value_df = self.df.drop(['PetID','AdoptionSpeed','Description'], axis=1)

        x_categ = torch.tensor(self.categ_df.iloc[idx], dtype=torch.int)
        x_num = torch.tensor(self.num_df.iloc[idx])

        return image_input, x_categ, x_num, input_ids, token_type_ids, attention_mask , label


class MultimodalModel(nn.Module):
    def __init__(self,out_dim=5,categories=(2,3,3,3,3,15,4,2,2,2,2,2,2,2,2),num_continuous= 9, pretrained_text_model = "allenai/longformer-base-4096", pretrained_image_model ='google/vit-base-patch16-224',device='cuda',img_ok=False,text_ok=False,tabular_ok=False):
        super(MultimodalModel, self).__init__()
        

        self.num_special_tokens = 2
        self.num_categories = len(categories)
        self.num_unique_categories = sum(categories)
        self.img_ok = img_ok
        self.text_ok = text_ok
        self.tabular_ok = tabular_ok
        total_tokens = self.num_unique_categories + self.num_special_tokens
        dim = 768

        if self.num_unique_categories > 0:
            categories_offset = nn.functional.pad(torch.tensor(list(categories)), (1, 0), value = self.num_special_tokens)
            self.categories_offset = categories_offset.cumsum(dim = -1)[:-1].to(device)

            # categorical embedding

            self.categorical_embeds = nn.Embedding(total_tokens, dim)
        self.num_continuous = num_continuous

        if self.num_continuous > 0:
            self.numerical_embedder = NumericalEmbedder(dim, self.num_continuous)

        # cls token

        self.cls_token = nn.Parameter(torch.randn(1, 1, dim))

        # Concatenate 50(tabular) + 2048 (image) + 768 (text)= 2866
        # parameters: 1474013
        # -> 768*1024 + 1024*512 + 512*64 +  64*5
        self.text_embedding = AutoModel.from_pretrained(pretrained_text_model).embeddings
        self.image_embedding = AutoModel.from_pretrained(pretrained_image_model).embeddings
        # self.image_embedding = EmbeddingStem(position_embedding_dropout=0.1)
        self.encoder = AutoModel.from_pretrained(pretrained_text_model).encoder
        self.classifier = nn.Sequential(
             nn.Dropout(0.3),
             nn.Linear(768, out_dim),
        )
        # for param in self.text_embedding.parameters():
        #     param.requires_grad = False
        # for param in self.image_embedding.parameters():
        #     param.requires_grad = False


    def forward(self, image_input, x_categ, x_num, input_ids, token_type_ids, attention_mask):
        xs = []
        if self.num_unique_categories > 0:
            x_categ = x_categ + self.categories_offset

            x_categ = self.categorical_embeds(x_categ)

            xs.append(x_categ)

        # add numerically embedded tokens
        if self.num_continuous > 0:
            x_num = self.numerical_embedder(x_num)
            xs.append(x_num)
        # concat categorical and numerical

        tabular_embed = torch.cat(xs, dim = 1)

        # append cls tokens
        b = tabular_embed.shape[0]
        cls_tokens = repeat(self.cls_token, '1 1 d -> b 1 d', b = b)
        tabular_embed = torch.cat((cls_tokens, tabular_embed), dim = 1).to(input_ids.device)
        text_embed = self.text_embedding(input_ids)
        image_embed = self.image_embedding(image_input['pixel_values'].squeeze(1))

        if self.tabular_ok and self.text_ok and self.img_ok:
            combined_embed = torch.cat([image_embed, tabular_embed, text_embed], dim=1)
            pad_length = 1024 - combined_embed.size(1)
        elif self.img_ok and self.text_ok:
            combined_embed = torch.cat([image_embed, text_embed], dim=1)
            pad_length = 1024 - combined_embed.size(1)
        elif self.tabular_ok and self.text_ok:
            combined_embed = torch.cat([tabular_embed, text_embed], dim=1)
            pad_length = 1024 - combined_embed.size(1)
        elif self.img_ok and self.tabular_ok:
            combined_embed = torch.cat([image_embed, tabular_embed], dim=1)
            pad_length = 512 - combined_embed.size(1)
        elif self.text_ok:
            combined_embed = torch.cat([text_embed], dim=1)
            pad_length = 512 - combined_embed.size(1)
        elif self.tabular_ok:
            combined_embed = torch.cat([tabular_embed], dim=1)
            pad_length = 512 - combined_embed.size(1)
        elif self.img_ok:
            combined_embed = torch.cat([image_embed], dim=1)
            pad_length = 512 - combined_embed.size(1)
        else:
            combined_embed = torch.cat([image_embed, tabular_embed, text_embed], dim=1)
            pad_length = 1024 - combined_embed.size(1)

        if text_embed.shape[1] < combined_embed.shape[1]:
            new_attention_mask = torch.ones((combined_embed.shape[0], combined_embed.shape[1]-text_embed.shape[1]), dtype=torch.long).to(input_ids.device)
            combined_attention_mask = torch.cat([new_attention_mask, attention_mask], dim=1)
        else:
            new_attention_mask = torch.ones((combined_embed.shape[0], combined_embed.shape[1]), dtype=torch.long).to(input_ids.device)
            combined_attention_mask = torch.cat([new_attention_mask], dim=1)

        padding_embed = torch.zeros((combined_embed.shape[0], pad_length, combined_embed.shape[2])).to(combined_embed.device)
        combined_embed = torch.cat([combined_embed, padding_embed], dim=1)

        padding_attention_mask = torch.zeros((combined_embed.shape[0], pad_length), dtype=torch.long).to(combined_attention_mask.device)
        combined_attention_mask = torch.cat([combined_attention_mask, padding_attention_mask], dim=1)

        combined_embed = combined_embed.to(torch.float32)

        text_output = self.encoder(hidden_states=combined_embed, attention_mask=combined_attention_mask).last_hidden_state
        # text_output = self.pooler(x)
        input_mask_expanded = combined_attention_mask.unsqueeze(-1).expand(text_output.size())
        sum_embeddings = torch.sum(text_output * input_mask_expanded, 1)
        mean_embeddings = sum_embeddings / input_mask_expanded.sum(1)
        output = self.classifier(mean_embeddings)
        return output

if __name__ == '__main__':
    parser = get_parser()
    args = parser.parse_args()
    seed = args.seed
    device = args.device if torch.cuda.is_available() else "cpu"
    df_path = args.df_path
    df_valid_path = args.df_valid_path
    df_test_path = args.df_test_path
    train_num_df = args.train_num_df
    valid_num_df = args.valid_num_df
    test_num_df = args.test_num_df
    train_categ_df = args.train_categ_df
    valid_categ_df = args.valid_categ_df
    test_categ_df = args.test_categ_df
    pretrained_text_model = args.pretrained_text_model
    pretrained_image_model = args.pretrained_image_model
    img_dir = args.img_dir
    write_emb = True
    model_save_dir = args.model_save_dir
    text_ok = args.text_ok
    img_ok = args.img_ok
    tabular_ok = args.tabular_ok
    os.makedirs(model_save_dir,exist_ok=True)
    same_seeds(seed)
    train_df = pd.read_csv(df_path)
    valid_df = pd.read_csv(df_valid_path)
    test_df = pd.read_csv(df_test_path)
    train_num_df = pd.read_csv(train_num_df)
    valid_num_df = pd.read_csv(valid_num_df)
    test_num_df = pd.read_csv(test_num_df)
    train_categ_df = pd.read_csv(train_categ_df)
    valid_categ_df = pd.read_csv(valid_categ_df)
    test_categ_df = pd.read_csv(test_categ_df)
    

    model = MultimodalModel(out_dim=5, pretrained_text_model = pretrained_text_model,device=device,img_ok=img_ok,text_ok=text_ok,tabular_ok=tabular_ok)
    train_set = PetFinderDataset(train_df , img_dir, train_categ_df, train_num_df, pretrained_text_model=pretrained_text_model, pretrained_image_model= pretrained_image_model)
    valid_set = PetFinderDataset(valid_df , img_dir, valid_categ_df, valid_num_df,pretrained_text_model=pretrained_text_model, pretrained_image_model= pretrained_image_model)
    test_set = PetFinderDataset(test_df , img_dir, test_categ_df, test_num_df,pretrained_text_model=pretrained_text_model, pretrained_image_model= pretrained_image_model)

    # hyperparameters
    num_epoch = 8
    validation = True
    logging_step = 30
    learning_rate = 1e-5
    optimizer = AdamW(model.parameters(), lr=learning_rate)
    criterion = nn.CrossEntropyLoss()
    train_batch_size = 4
    gradient_accumulation_steps = 1

    train_loader = DataLoader(train_set, batch_size=train_batch_size, shuffle=True, pin_memory=True)
    dev_loader = DataLoader(valid_set, batch_size=1, shuffle=False, pin_memory=True)
    test_loader = DataLoader(test_set, batch_size=1, shuffle=False, pin_memory=True)

    # Change "fp16_training" to True to support automatic mixed 
    # precision training (fp16)	
    fp16_training = True
    if fp16_training:    
        accelerator = Accelerator(mixed_precision="fp16")
    else:
        accelerator = Accelerator()

    total_step = (len(train_loader)//gradient_accumulation_steps) * num_epoch
    scheduler = get_linear_schedule_with_warmup(optimizer,int(total_step *0.1),
                                                            total_step)

    # Documentation for the toolkit:  https://huggingface.co/docs/accelerate/

    model, optimizer, train_loader, scheduler = accelerator.prepare(model, optimizer, train_loader, scheduler) 
    model.to(device)


    model.train()

    print("Start Training ...")
    best_acc = 0.0
    train_losses_his = []
    valid_losses_his = []

    for epoch in range(num_epoch):
        train_loss = []
        train_accs = []
        step = 1
        for index, data in enumerate(tqdm(train_loader)):	
            # Load all data into GPU
            image_input, x_categ, x_num, input_ids, token_type_ids, attention_mask ,label = data
            image_input, x_categ, x_num, input_ids, token_type_ids, attention_mask ,label = image_input.to(device), x_categ.to(device), x_num.to(device), input_ids.to(device), token_type_ids.to(device),attention_mask.to(device), label.to(device)
            output = model(image_input, x_categ, x_num, input_ids, token_type_ids, attention_mask)
            label = label.to(torch.long)
            
            loss = criterion(output,label)

            optimizer.zero_grad()
            
            accelerator.backward(loss)
            optimizer.step()
            scheduler.step()

            acc = (output.argmax(dim=-1) == label).float().mean()

            train_loss.append(loss.item())
            train_accs.append(acc)
            step+=1

            # Print training loss and accuracy over past logging step
            if step % logging_step == 0:
                print(f"Epoch {epoch + 1} | Step {step} | acc = {(sum(train_accs) / len(train_accs)):.3f} | loss = {(sum(train_loss) / len(train_loss)):.3f}")
        train_losses_his.append((sum(train_loss) / len(train_loss)))
        

        if validation:
            print("Evaluating Dev Set ...")
            model.eval()
            valid_loss = []
            valid_accs = []
            with torch.no_grad():
                dev_loss= 0
                for index, data in enumerate(tqdm(dev_loader)):
                    image_input, x_categ, x_num, input_ids, token_type_ids, attention_mask ,label = data
                    image_input, x_categ, x_num, input_ids, token_type_ids, attention_mask ,label = image_input.to(device), x_categ.to(device), x_num.to(device), input_ids.to(device), token_type_ids.to(device),attention_mask.to(device), label.to(device)
                    output = model(image_input, x_categ, x_num, input_ids, token_type_ids, attention_mask)
                    label = label.to(torch.long)
                    loss = criterion(output,label)
                    acc = (output.argmax(dim=-1) == label).float().mean()
                    valid_loss.append(loss.item())
                    valid_accs.append(acc)
                print(f"Validation | Epoch {epoch + 1} | acc = {(sum(valid_accs) / len(valid_accs)):.3f} | loss = {(sum(valid_loss) / len(valid_loss)):.3f}")
                valid_losses_his.append((sum(valid_loss)/ len(valid_accs)))
            model.train()

            if best_acc < (sum(valid_accs) / len(valid_accs)):
                print(f"Best model found at epoch {epoch}, saving model")
                torch.save(model.image_embedding.state_dict(), os.path.join(model_save_dir,"image_embedding.ckpt"))
                torch.save(model.text_embedding.state_dict(), os.path.join(model_save_dir,"text_embedding.ckpt"))
                torch.save(model.encoder.state_dict(), os.path.join(model_save_dir,"encoder.ckpt"))
                torch.save(model.classifier.state_dict(), os.path.join(model_save_dir,"concat_best.ckpt"))
                torch.save(model.numerical_embedder.state_dict(), os.path.join(model_save_dir,"numerical_embedder.ckpt"))
                torch.save(model.categorical_embeds.state_dict(), os.path.join(model_save_dir,"categorical_embeds.ckpt"))
                torch.save(model.cls_token, os.path.join(model_save_dir,"cls_token.ckpt"))
                best_acc = (sum(valid_accs) / len(valid_accs))
    with open(f'./{model_save_dir}/train_foundation.pkl', 'wb') as f:
        pickle.dump(train_losses_his, f)
    with open(f'./{model_save_dir}/valid_foundation.pkl', 'wb') as f:
        pickle.dump(valid_losses_his, f)
