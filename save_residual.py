import argparse
import os
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

from utils import build_generator

def parse_args():
  p = argparse.ArgumentParser(description='Run a trained SR generator over a dataset and save its per-pixel residual.')
  p.add_argument('--generator_module', default='BliMSR.generator', help='Python module containing the generator class.')
  p.add_argument('--generator_class', default='Generator')
  p.add_argument('--checkpoint', default='BliMSR/checkpoints/gen_150.pth', help='Trained base SR model weights.')
  p.add_argument('--data_path', default='BliMSR/data/train_data.pt', help='(hr, lr) tensor pairs.')
  p.add_argument('--save_path', default='BliMSR/data/res_data.pt')
  p.add_argument('--device', default='cuda' if torch.cuda.is_available() else 'cpu')
  return p.parse_args()

def main():
  args = parse_args()
  os.makedirs(os.path.dirname(args.save_path) or '.', exist_ok=True)
  gen = build_generator(args.generator_module, args.generator_class, args.checkpoint, args.device)

  dl = DataLoader(torch.load(args.data_path), batch_size=1, shuffle=False)
  res_data = []
  for hr, lr in tqdm(dl, desc='computing residuals'):
    hr, lr = hr.to(args.device), lr.to(args.device)
    with torch.no_grad():
      sr = gen(lr)
    res = abs(hr - sr)
    res_data.append((hr[0].cpu(), sr[0].cpu(), res[0].cpu(), lr[0].cpu()))

  torch.save(res_data, args.save_path)
  print(f'Saved {len(res_data)} samples to {args.save_path}')

if __name__ == '__main__':
  main()
