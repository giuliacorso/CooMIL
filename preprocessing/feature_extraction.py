from torch.utils.data import Dataset, DataLoader
import torch
from PIL import Image
import pickle
import os
from PIL import Image
import torch
import argparse
from conch.open_clip_custom import create_model_from_pretrained


class MyDataset(Dataset):
    def __init__(self, image):
        self.patches = []
        for patch in os.listdir(image):
            if patch.endswith('.jpg'):
                self.patches.append(os.path.join(image, patch))

    def __len__(self):
        return len(self.patches)

    def __getitem__(self, idx):
        patch_path = self.patches[idx]
        return patch_path


def main(job, res, conch_checkpoints, images_dir, results_root):

    conch_model, conch_processor = create_model_from_pretrained("conch_ViT-B-16", checkpoint_path=conch_checkpoints)

    if torch.cuda.is_available():
        device = torch.device("cuda")
    else:
        device = torch.device("cpu")

    conch_model.to(device)
    conch_model = conch_model.eval()

    if res == 'patch':
        images_dir = os.path.join(images_dir, 'patches')  # patch
    elif res == 'region':
        images_dir = os.path.join(images_dir, 'regions')  # region

    if res == 'patch':
        with open(f'job_{job}_patch.txt', 'r') as f:
            all_images = f.read().splitlines()
    elif res == 'region':
        with open(f'job_{job}_region.txt', 'r') as f:
            all_images = f.read().splitlines()

    all_images = [os.path.join(images_dir, img) for img in all_images]

    features_conch = []

    for i, img in enumerate(all_images):
        dataset = MyDataset(image=os.path.join(images_dir, img))
        if res == 'patch':
            dataloader = DataLoader(dataset=dataset, batch_size=64, num_workers=0, shuffle=False)
        elif res == 'region':
            dataloader = DataLoader(dataset=dataset, batch_size=1, num_workers=0, shuffle=False)

        values = []
        keys = []

        print(f'Processing image {i}/{len(all_images)}')
        
        for i, images in enumerate(dataloader):
            if i == 0:
                print(f'Extracting image features for {img}')
        
            if i % 100 == 0:
                print(f'{i}/{len(dataloader)}')
            
            image = [Image.open(path) for path in images]

            image_conch = [conch_processor(img).unsqueeze(0) for img in image]
            image_conch = torch.cat(image_conch, dim=0)
            image_conch = image_conch.to(device)      

            with torch.inference_mode():
                image_features_conch = conch_model.encode_image(image_conch, proj_contrast=False, normalize=False)

            keys.extend(images)
            # batch_size=N (patch)
            if res == 'patch':
                for i in range(image_features_conch.shape[0]):
                    values.append(image_features_conch[i, :].cpu())
            # batch_size=1 (region)
            elif res == 'region':
                values.extend(image_features_conch.cpu())
            
        features_conch = dict(zip(keys, values))

        if res == 'patch':
            conch_root = os.path.join(results_root, 'feats_conch_patch')
        elif res == 'region':
            conch_root = os.path.join(results_root, 'feats_conch_region')
        os.makedirs(conch_root, exist_ok=True)
        name = img.split('/')[-1]
        with open(os.path.join(conch_root, f'{name}.pkl'), 'wb') as handle:
            pickle.dump(features_conch, handle)
        
        values = []
        keys = []

parser = argparse.ArgumentParser(description='Compute features from CONCH embedder')
parser.add_argument('--job', default=0, type=int, help='Job number')
parser.add_argument('--res', default='patch', type=str, help='Resolution level (patch or region)')
parser.add_argument('--conch_checkpoints', type=str, help='Path to CONCH model checkpoint')
parser.add_argument('--images_dir', type=str, help='Directory containing images')
parser.add_argument('--results_root', type=str, help='Root directory for results')
args = parser.parse_args()

main(job=args.job, res=args.res, conch_checkpoints=args.conch_checkpoints, images_dir=args.images_dir, results_root=args.results_root)
