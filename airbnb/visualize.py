import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
from torchvision import transforms
import torchvision.transforms as transforms
from torch.utils.data import ConcatDataset, DataLoader, Subset, Dataset
import os
from umap import UMAP
import argparse


def get_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_name', type=str, required=True)
    parser.add_argument('--method', type=str, required=True)
    parser.add_argument('--text2image', type=str, required=True)
    parser.add_argument('--img2tab', type=str, required=True)
    parser.add_argument('--text', type=str, required=True)
    parser.add_argument('--img', type=str, required=True)
    parser.add_argument('--tab', type=str, required=True)
    
    return parser

class DDATA(Dataset):
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

        return tensor_1d

if __name__ == '__main__':
    parser = get_parser()
    args = parser.parse_args()
    data_name = args.data_name
    method_name = args.method
    #embeddings path
    text2image_path = args.text2image
    img2tab_path = args.img2tab
    text_path = args.text
    img_path = args.img
    tab_path = args.tab

    umap = UMAP(n_components=2, random_state=0)

    test_set = DDATA(text2image_path)
    test_loader = DataLoader(test_set, batch_size=1184, shuffle=False, pin_memory=True)
    for i, data in enumerate(test_loader):
        embeddings_transfer_text2image = data

    tests = DDATA(img2tab_path)
    test_loaders = DataLoader(tests, batch_size=1184, shuffle=False, pin_memory=True)
    for i, data in enumerate(test_loaders):
        embeddings_transfer_img2tab = data

    tt = DDATA(text_path)
    ttlo = DataLoader(tt, batch_size=1184, shuffle=False, pin_memory=True)
    for i, data in enumerate(ttlo):
        embeddings_transfer_text = data

    ttq = DDATA(img_path)
    ttloq = DataLoader(ttq, batch_size=1184, shuffle=False, pin_memory=True)
    for i, data in enumerate(ttloq):
        embeddings_transfer_image = data

    ttqq = DDATA(tab_path)
    ttloqq = DataLoader(ttqq, batch_size=1184, shuffle=False, pin_memory=True)
    for i, data in enumerate(ttloqq):
        embeddings_transfer_tab = data

    embeddings_transfer_text2image_umap = umap.fit_transform(embeddings_transfer_text2image)
    embeddings_transfer_img2tab_umap = umap.fit_transform(embeddings_transfer_img2tab)
    embeddings_transfer_text_umap = umap.fit_transform(embeddings_transfer_text)
    embeddings_transfer_image_umap = umap.fit_transform(embeddings_transfer_image)
    embeddings_transfer_tab_umap = umap.fit_transform(embeddings_transfer_tab)


    plt.figure(figsize=(15, 6)) 
    plt.subplot(1, 2, 1)
    plt.scatter(embeddings_transfer_text_umap[:, 0], embeddings_transfer_text_umap[:, 1], color='#F7903D', alpha=0.9, label='text')
    plt.scatter(embeddings_transfer_image_umap[:, 0], embeddings_transfer_image_umap[:, 1], color='#4D85BD', alpha=0.9, label='image')
    plt.scatter(embeddings_transfer_text2image_umap[:, 0], embeddings_transfer_text2image_umap[:, 1], color='#BD81C0', alpha=0.9, label='transfer text to image')
    plt.title('Embeddings before/after translation via UMAP')
    plt.legend()


    plt.subplot(1, 2, 2)
    plt.scatter(embeddings_transfer_image_umap[:, 0], embeddings_transfer_image_umap[:, 1], color='#F7903D', alpha=0.9, label='image')
    plt.scatter(embeddings_transfer_tab_umap[:, 0], embeddings_transfer_tab_umap[:, 1], color='#4D85BD', alpha=0.9, label='table')
    plt.scatter(embeddings_transfer_img2tab_umap[:, 0], embeddings_transfer_img2tab_umap[:, 1], color='#BD81C0', alpha=0.9, label='transfer image to table')
    plt.title('Embeddings before/after translation via UMAP')
    plt.legend()

    plt.tight_layout()
    plt.savefig(f"visualization/{data_name}_{method_name}.png", bbox_inches='tight')
    plt.show()