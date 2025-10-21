import torch
import os
import pickle
import argparse

def main(images_dir, results_root, save_root):
    os.makedirs(save_root, exist_ok=True)

    region_path = os.path.join(results_root, 'feats_conch_region')
    patch_path = os.path.join(results_root, 'feats_conch_patch')

    regions = os.listdir(region_path)
    
    for r in regions:
        with open(os.path.join(region_path, r), 'rb') as f:
            region = pickle.load(f)
        with open(os.path.join(patch_path, r), 'rb') as f:
            patch = pickle.load(f)

        slide = {}

        slide_region = []
        for k, v in region.items():
            slide_region.append(v.squeeze(0))
        slide['region'] = torch.stack(slide_region, dim=0)

        xy = []
        for reg in region.keys():
          c = reg.split('/')[-1]
          c = c.split('.')[0]
          x = c.split('_')[2]
          y = c.split('_')[4]
          xy.append((int(x),int(y)))

        coords_all = {}
        coords = []
        for el in xy:
          x = el[0]
          y = el[1]
          for i in range(8):
            for j in range(8):
              coord = (x+(i*512), y+(j*512))
              coords.append(coord)
          coords_all[tuple(el)] = coords
          coords = []

        name = r.split('.pkl')[0]
        base = f'{images_dir}/patches/{name}'
  
        slide_patch = []
        each = []
        for k, v in coords_all.items():
            paths = [os.path.join(base, f'_x_{coord[0]}_y_{coord[1]}.jpg') for coord in v]
            for path in paths:
              each.append(patch[path].squeeze(0))
            each = torch.stack(each,dim=0).reshape(8,8,-1)
            slide_patch.append(each)
            each = []
        slide['patch'] = torch.stack(slide_patch, dim=0)

        print(slide['region'].shape)
        print(slide['patch'].shape)

        with open(os.path.join(save_root, f'{name}.pkl'), 'wb') as handle:
                pickle.dump(slide, handle)


parser = argparse.ArgumentParser(description='Compute features from CONCH embedder')
parser.add_argument('--images_dir', type=str, help='Directory containing images')
parser.add_argument('--results_root', type=str, help='Root directory for feature extraction results')
parser.add_argument('--save_root', type=str, help='Root directory for results')
args = parser.parse_args()

main(images_dir=args.images_dir, results_root=args.results_root, save_root=args.save_root)
