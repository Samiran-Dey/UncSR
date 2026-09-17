import argparse
import os
import numpy as np
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

from utils import build_generator, refine, mean_filter, expand_filter

#nonconformity score = |hr-sr| / predicted residual, averaged over the available LR views, at full/patch/pixel level
def calibrate(gen_sr, gen_res, dl, device, alpha, patch_size):
  scores_full, scores_patch, scores_pixel = [], [], []
  mfilter = mean_filter(patch_size, device)

  for data in tqdm(dl, desc='calibrating'):
    hr, *lrs = data
    hr = hr.to(device)
    s_views = []
    for lr in lrs:
      lr = lr.to(device)
      with torch.no_grad():
        sr = gen_sr(lr)
        res = gen_res(lr)
      s_views.append(abs(hr - sr) / res)
    s_avg = torch.stack(s_views).mean(0)
    s_avg[hr <= 0.12] = 0.0 #ignore background pixels

    scores_full.append(torch.mean(s_avg).detach())
    scores_patch.append(mfilter(s_avg)[0].detach())
    scores_pixel.append(s_avg[0].detach())

  n = len(dl)
  q_level = min(max(np.ceil((n + 1) * (1 - alpha)) / n, 0), 1)

  qhat_full = torch.quantile(torch.tensor(scores_full), q_level)
  qhat_patch = torch.quantile(torch.cat(scores_patch, dim=0), q_level, dim=0)
  qhat_pixel = torch.quantile(torch.cat(scores_pixel, dim=0), q_level, dim=0)
  return qhat_full, qhat_patch, qhat_pixel

#build a prediction interval sr +- qhat*res_pred and measure marginal/conditional coverage and interval width (uncertainty)
def evaluate(gen_sr, gen_res, dl, qhat, level, beta, device, patch_size, log_fn):
  qhat = qhat.to(device) if level != 'unit' else qhat
  efilter = expand_filter(patch_size, device) if level == 'patch' else None
  marginal_hits, conditional_cov, uncertainty = 0, [], []

  for data in tqdm(dl, desc=f'{level} level testing'):
    hr, *lrs = data
    hr = hr.to(device)
    srs, intervals = [], []
    for lr in lrs:
      lr = lr.to(device)
      with torch.no_grad():
        sr = gen_sr(lr)
        res = gen_res(lr)
      q = efilter(qhat.unsqueeze(0)) if level == 'patch' else qhat
      lb, ub = refine(sr - res * q), refine(sr + res * q)
      srs.append(sr)
      intervals.append((lb, ub))

    lb0, ub0 = intervals[0]
    uncertainty.append(torch.mean((abs(ub0 - hr) + abs(lb0 - hr)) / 2).detach())

    if torch.mean((hr >= lb0).float()) >= beta and torch.mean((hr <= ub0).float()) >= beta:
      marginal_hits += 1

    if len(srs) > 1:
      hits = sum(
        1 for sr in srs[1:]
        if torch.mean((sr >= lb0).float()) >= beta and torch.mean((sr <= ub0).float()) >= beta
      )
      conditional_cov.append(hits / (len(srs) - 1))

  marginal_cov = marginal_hits / len(dl)
  mean_uncertainty = float(torch.mean(torch.tensor(uncertainty)))
  mean_conditional_cov = float(np.mean(conditional_cov)) if conditional_cov else None

  log_fn(f"Level -> {level}\nMarginal coverage -> {marginal_cov}\nConditional coverage -> {mean_conditional_cov}\nUncertainty -> {mean_uncertainty}\n")
  return marginal_cov, mean_conditional_cov, mean_uncertainty

def parse_args():
  p = argparse.ArgumentParser(description='Calibrate and evaluate generic conformal prediction intervals for any SR model.')
  p.add_argument('--sr_generator_module', default='BliMSR.generator', help='Swap these to use another model.')
  p.add_argument('--sr_generator_class', default='Generator')
  p.add_argument('--sr_checkpoint', default='BliMSR/checkpoints/gen_150.pth')
  p.add_argument('--res_generator_module', default='BliMSR.generator')
  p.add_argument('--res_generator_class', default='Generator')
  p.add_argument('--res_checkpoint', default='BliMSR/checkpoints/gen_res_150.pth')
  p.add_argument('--cal_data', default='BliMSR/data/conf_data.pt', help='(hr, lr_1, ..., lr_n) tuples, n>=1.')
  p.add_argument('--test_data', default='BliMSR/data/test_data.pt')
  p.add_argument('--log_dir', default='BliMSR/checkpoints/conformal/')
  p.add_argument('--alpha', type=float, default=0.1, help='Miscoverage rate; target coverage = 1-alpha.')
  p.add_argument('--patch_size', type=int, default=32)
  p.add_argument('--device', default='cuda' if torch.cuda.is_available() else 'cpu')
  return p.parse_args()

def main():
  args = parse_args()
  os.makedirs(args.log_dir, exist_ok=True)
  log_name = os.path.join(args.log_dir, 'conformal_log.txt')

  def log(msg):
    print(msg)
    with open(log_name, "a") as log_file:
      log_file.write(msg + '\n')

  device = args.device
  gen_sr = build_generator(args.sr_generator_module, args.sr_generator_class, args.sr_checkpoint, device)
  gen_res = build_generator(args.res_generator_module, args.res_generator_class, args.res_checkpoint, device)

  cal_dl = DataLoader(torch.load(args.cal_data), batch_size=1, shuffle=False)
  test_dl = DataLoader(torch.load(args.test_data), batch_size=1, shuffle=False)

  log('Calibrating ...')
  qhat_full, qhat_patch, qhat_pixel = calibrate(gen_sr, gen_res, cal_dl, device, args.alpha, args.patch_size)
  log(f"qhat_full={qhat_full}\nqhat_patch={qhat_patch}\nqhat_pixel={qhat_pixel}\n")

  beta = 1 - args.alpha
  evaluate(gen_sr, gen_res, test_dl, qhat_full, 'full', beta, device, args.patch_size, log)
  evaluate(gen_sr, gen_res, test_dl, qhat_patch, 'patch', beta, device, args.patch_size, log)
  evaluate(gen_sr, gen_res, test_dl, qhat_pixel, 'pixel', beta, device, args.patch_size, log)
  evaluate(gen_sr, gen_res, test_dl, 1.0, 'unit', beta, device, args.patch_size, log)

if __name__ == '__main__':
  main()
