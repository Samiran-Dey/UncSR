import argparse
import os
import numpy as np
import torch
import torch.optim as optim
import torchvision.transforms as T
from torch.autograd import Variable
from torch.utils.data import DataLoader
from skimage.metrics import structural_similarity
from tqdm import tqdm

from utils import load_class

def gradient_loss(preds, targets):
  gr_loss = 0
  for pred, target in zip(preds, targets):
    gr1 = np.array(np.gradient(pred[0].detach().cpu().numpy()))
    gr2 = np.array(np.gradient(target[0].detach().cpu().numpy()))
    gr_loss += np.mean(abs(gr1 - gr2))
  return gr_loss / len(preds)

def ssim_loss(preds, targets):
  to_pil = T.ToPILImage()
  sm_loss = 0
  for pred, target in zip(preds, targets):
    a, b = np.asarray(to_pil(pred[0])), np.asarray(to_pil(target[0]))
    sm_loss += (1 - abs(structural_similarity(a, b))) / 2
  return sm_loss / len(preds)

def perceptual_loss(preds, target, feature_extractor, loss_function):
  sr, hr = preds.repeat(1, 3, 1, 1), target.repeat(1, 3, 1, 1)
  w = (0.5, .25, .125, 0.0625, 0.0625)
  return sum(wi * loss_function(pf, hf) for wi, pf, hf in zip(w, feature_extractor(sr), feature_extractor(hr)))

def parse_args():
  p = argparse.ArgumentParser(description='Train a residual predictor with the same architecture as the base SR model.')
  p.add_argument('--generator_module', default='BliMSR.generator', help='Swap to use another model.')
  p.add_argument('--generator_class', default='Generator')
  p.add_argument('--discriminator_module', default='BliMSR.discriminator')
  p.add_argument('--discriminator_class', default='DiscriminatorSN')
  p.add_argument('--feature_extractor_module', default='BliMSR.feature_extractor')
  p.add_argument('--feature_extractor_class', default='vgg19')
  p.add_argument('--data_path', default='BliMSR/data/res_data.pt', help='(hr, sr, res, lr) tuples from save_residual.py')
  p.add_argument('--checkpoints_dir', default='BliMSR/checkpoints/')
  p.add_argument('--batch_size', type=int, default=4)
  p.add_argument('--start_epoch', type=int, default=1)
  p.add_argument('--epochs', type=int, default=150)
  p.add_argument('--lr', type=float, default=1e-5)
  p.add_argument('--save_every', type=int, default=10)
  p.add_argument('--resume_gen', default=None, help='Generator checkpoint to resume training from.')
  p.add_argument('--resume_disc', default=None, help='Discriminator checkpoint to resume training from.')
  p.add_argument('--device', default='cuda' if torch.cuda.is_available() else 'cpu')
  return p.parse_args()

def main():
  args = parse_args()
  os.makedirs(args.checkpoints_dir, exist_ok=True)
  device = args.device
  Tensor = torch.cuda.FloatTensor if device.startswith('cuda') else torch.Tensor

  gen = load_class(args.generator_module, args.generator_class)().to(device)
  disc = load_class(args.discriminator_module, args.discriminator_class)().to(device)
  feature_extractor = load_class(args.feature_extractor_module, args.feature_extractor_class)().to(device)
  feature_extractor.eval()
  if args.resume_gen:
    gen.load_state_dict(torch.load(args.resume_gen, map_location=device))
  if args.resume_disc:
    disc.load_state_dict(torch.load(args.resume_disc, map_location=device))

  train_dl = DataLoader(torch.load(args.data_path), batch_size=args.batch_size, shuffle=False)

  optimizer_G = optim.Adam(gen.parameters(), lr=args.lr)
  optimizer_D = optim.Adam(disc.parameters(), lr=args.lr)
  loss_function = torch.nn.L1Loss().to(device)
  gan_loss = torch.nn.BCEWithLogitsLoss().to(device)
  scaler = torch.cuda.amp.GradScaler()
  log_name = os.path.join(args.checkpoints_dir, 'residual_loss_log.txt')

  for epoch in range(args.start_epoch, args.epochs + 1):
    e_loss_G, e_loss_D = [], []

    for hr, sr_base, res, lr in tqdm(train_dl):
      target = res.to(device) #regression target: residual map |hr - sr_base|
      lr = lr.to(device)
      valid = Variable(Tensor(np.ones((target.shape[0], 1))), requires_grad=False)
      fake = Variable(Tensor(np.zeros((target.shape[0], 1))), requires_grad=False)

      with torch.cuda.amp.autocast():
        pred_res = gen(lr)

        content_loss = loss_function(pred_res, target)
        dssim_loss = ssim_loss(pred_res, target)
        gr_loss = gradient_loss(pred_res, target)
        perc_loss = perceptual_loss(pred_res, target, feature_extractor, loss_function)
        adv_loss = gan_loss(disc(pred_res), valid)
        loss_G = perc_loss + 0.01 * adv_loss + 0.1 * content_loss + gr_loss + dssim_loss

        optimizer_G.zero_grad()
        scaler.scale(loss_G).backward()
        scaler.step(optimizer_G)
        scaler.update()
        e_loss_G.append(float(loss_G))

        pred_real = disc(target)
        pred_fake = disc(pred_res.detach())
        loss_D = (gan_loss(pred_real, valid) + gan_loss(pred_fake, fake)) / 2

        optimizer_D.zero_grad()
        scaler.scale(loss_D).backward()
        scaler.step(optimizer_D)
        scaler.update()
        e_loss_D.append(float(loss_D))

    message = f"{epoch}/{args.epochs} -- Gen Loss: {sum(e_loss_G) / len(e_loss_G)} -- Disc Loss: {sum(e_loss_D) / len(e_loss_D)}"
    print(message)
    with open(log_name, "a") as log_file:
      log_file.write(message + '\n')

    if epoch % args.save_every == 0 or epoch == args.epochs:
      torch.save(gen.state_dict(), os.path.join(args.checkpoints_dir, f'gen_res_{epoch}.pth'))
      torch.save(disc.state_dict(), os.path.join(args.checkpoints_dir, f'disc_res_{epoch}.pth'))

if __name__ == '__main__':
  main()
