import importlib
import torch

def load_class(module_path, class_name):
  module = importlib.import_module(module_path)
  return getattr(module, class_name)

def build_generator(module_path, class_name, checkpoint, device):
  model = load_class(module_path, class_name)().to(device)
  model.load_state_dict(torch.load(checkpoint, map_location=device))
  model.eval()
  return model

# clip a prediction-interval bound back into the valid [0, 1] image range
def refine(image):
  image = image.clone()
  image[image > 1] = 1.0
  image[image < 0] = 0.0
  return image

def mean_filter(patch_size, device):
  f = torch.nn.Conv2d(1, 1, kernel_size=patch_size, stride=patch_size, bias=False)
  f.weight = torch.nn.Parameter(torch.ones((1, 1, patch_size, patch_size)) / (patch_size * patch_size))
  return f.to(device)

def expand_filter(patch_size, device):
  f = torch.nn.ConvTranspose2d(1, 1, kernel_size=patch_size, stride=patch_size, bias=False)
  f.weight = torch.nn.Parameter(torch.ones((1, 1, patch_size, patch_size)))
  return f.to(device)
