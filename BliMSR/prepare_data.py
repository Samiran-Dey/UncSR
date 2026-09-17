import argparse
import os
from glob import glob
import torch
import torchvision.transforms as T

from dataset_random import create_dataset, read_dicom, get_LR_image

def parse_args():
  p = argparse.ArgumentParser(description='Prepare HR/LR tensor datasets for BliMSR from a folder of DICOM CT volumes.')
  p.add_argument('--hr_path', required=True, help='Root folder of DICOM CT volumes.')
  p.add_argument('--save_path', required=True, help='Output .pt file.')
  p.add_argument('--mode', choices=['train', 'multiview'], default='train',
                  help='train: (hr, lr) pairs for trainer.py. multiview: (hr, lr_1, ..., lr_n) for conformal.py.')
  p.add_argument('--n_views', type=int, default=3, help='Stochastic LR views per HR image (multiview mode).')
  return p.parse_args()

def main():
  args = parse_args()
  os.makedirs(os.path.dirname(args.save_path) or '.', exist_ok=True)

  if args.mode == 'train':
    data = create_dataset(args.hr_path)
  else:
    hr_images = read_dicom(glob(os.path.join(args.hr_path, '**', '*.dcm'), recursive=True))
    to_tensor = T.ToTensor()
    data = [(to_tensor(hr), *[to_tensor(get_LR_image(hr)) for _ in range(args.n_views)]) for hr in hr_images]

  torch.save(data, args.save_path)
  print(f'Saved {len(data)} samples to {args.save_path}')

if __name__ == '__main__':
  main()
