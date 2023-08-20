# For plotting
import numpy as np
import matplotlib.pyplot as plt
# For conversion
from skimage.color import lab2rgb, rgb2lab, rgb2gray
from skimage import io
# For everything
import torch
import torch.nn as nn
import torch.nn.functional as F
# For our model
import torchvision.models as models
from torchvision import datasets, transforms
# For utilities
import os, shutil, time
import numpy as np
import config
import os
#from PIL import Image
import PIL.Image
import torchvision.transforms as T
from torch.utils.data import Dataset, DataLoader
from torchvision.utils import save_image
from IPython.display import Image, display
import sys

use_gpu = torch.cuda.is_available()


class MapDataset(Dataset):
    def __init__(self, root_dir, set_data):
        self.root_dir = root_dir
        self.set_data = set_data
        self.list_files = os.listdir(self.root_dir)
        
        # Initialize dataset, you may use a second dataset for validation if required
        # Use the input transform to convert images to grayscale
        self.input_transform = T.Compose([T.ToTensor(),
                                          T.Resize(size=(256,256)),
                                          T.Grayscale(),
                                          T.Normalize((0.5), (0.5))
                                          ])
        # Use this on target images(colorful ones)
        self.target_transform = T.Compose([T.ToTensor(),
                                           T.Resize(size=(256,256)),
                                           T.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])

    def __len__(self):
        return len(self.list_files)

    def __getitem__(self, index):
        img_file = self.list_files[index]
        img_path = os.path.join(self.root_dir, img_file)
        print(img_path)
        image = np.array(PIL.Image.open(img_path))

        '''input_image = image[:, :600, :]
        target_image = image[:, 600:, :]

        augmentations = config.both_transform(image=input_image, image0=target_image)
        input_image = augmentations["image"]
        target_image = augmentations["image0"]

        input_image = config.transform_only_input(image=input_image)["image"]
        target_image = config.transform_only_mask(image=target_image)["image"]'''
        

        if self.set_data == "train":

            index = len(image[: int(len(image) * .80)])

            image = image[:index]

        else:

            index = len(image[: int(len(image) * .80)])

            image = image[index:]

        
        


        input_image = self.input_transform(image)

        print(len(input_image))

        target_image = self.target_transform(image)

        #print(len(input_image))
        #print(len(target_image))

        return input_image, target_image


class ConvertFunc(datasets.ImageFolder):
  '''Custom images folder, which converts images to grayscale before loading'''
  def __getitem__(self, index):
    path, target = self.imgs[index]
    img = self.loader(path)

    if self.transform is not None:
      orig_image = self.transform(img)
      orig_image = np.asarray(orig_image)
      lab_image = rgb2lab(orig_image)
      lab_image = (lab_image + 128) / 255
      ab_image = lab_image[:, :, 1:3]
      ab_image = torch.from_numpy(ab_image.transpose((2, 0, 1))).float()
      orig_image = rgb2gray(orig_image)
      orig_image = torch.from_numpy(orig_image).unsqueeze(0).float()

    if self.target_transform is not None:
      target = self.target_transform(target)
    return orig_image, ab_image, target


class ModelNet(nn.Module):
  def __init__(self, input_size=128):
    super(ModelNet, self).__init__()
    FEATURESIZE = 128

    resnet = models.resnet18(num_classes=365) 
    resnet.conv1.weight = nn.Parameter(resnet.conv1.weight.sum(dim=1).unsqueeze(1)) 
    self.midlevel_resnet = nn.Sequential(*list(resnet.children())[0:6])

    ## Second half: Upsampling
    self.upsample = nn.Sequential(     
      nn.Conv2d(FEATURESIZE, 128, kernel_size=3, stride=1, padding=1),
      nn.BatchNorm2d(128),
      nn.ReLU(),
      nn.Upsample(scale_factor=2),
      nn.Conv2d(128, 64, kernel_size=3, stride=1, padding=1),
      nn.BatchNorm2d(64),
      nn.ReLU(),
      nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=1),
      nn.BatchNorm2d(64),
      nn.ReLU(),
      nn.Upsample(scale_factor=2),
      nn.Conv2d(64, 32, kernel_size=3, stride=1, padding=1),
      nn.BatchNorm2d(32),
      nn.ReLU(),
      nn.Conv2d(32, 2, kernel_size=3, stride=1, padding=1),
      nn.Upsample(scale_factor=2)
    )

  def forward(self, input):

    features_midlevel = self.midlevel_resnet(input)
    result = self.upsample(features_midlevel)
    return result


class CalcuAvg(object):
  def __init__(self):
    self.reset()
  def reset(self):
    self.val, self.avg, self.sum, self.count = 0, 0, 0, 0
  def update(self, val, n=1):
    self.val = val
    self.sum += val * n
    self.count += n
    self.avg = self.sum / self.count

def to_rgb(input_gray, input_ab, save_path=None, save_name=None):
  
  plt.clf() # clear matplotlib 
  color = torch.cat((input_gray, input_ab), 0).numpy() 
  color = color.transpose((1, 2, 0))  
  color[:, :, 0:1] = color[:, :, 0:1] * 100
  color[:, :, 1:3] = color[:, :, 1:3] * 255 - 128   
  color = lab2rgb(color.astype(np.float64))
  input_gray = input_gray.squeeze().numpy()
  if save_path is not None and save_name is not None: 
    plt.imsave(arr=input_gray, fname='{}{}'.format(save_path['grayscale'], save_name), cmap='gray')
    plt.imsave(arr=color, fname='{}{}'.format(save_path['colorized'], save_name))


def validate(load_validation, model, crit, save_images, epoch):
  model.eval()

  batch = CalcuAvg(), 
  data_time = CalcuAvg() 
  losses = CalcuAvg()

  stop = time.time()

  saved_image = False
  for i, (input_gray, input_ab, target) in enumerate(load_validation):
    data_time.update(time.time() - stop)

    # Use GPU
    if use_gpu: input_gray, input_ab, target = input_gray.cuda(), input_ab.cuda(), target.cuda()

    output_ab = model(input_gray) 
    loss = crit(output_ab, input_ab)
    losses.update(loss.item(), input_gray.size(0))

    # Save images to file
    if save_images and not saved_image:
      saved_image = True
      for j in range(min(len(output_ab), 10)): # save at most 5 images
        save_path = {'grayscale': 'outputs/gray/', 'colorized': 'outputs/color/'}
        save_name = 'img-{}-epoch-{}.jpg'.format(i * load_validation.batch_size + j, epoch)
        to_rgb(input_gray[j].cpu(), input_ab=output_ab[j].detach().cpu(), save_path=save_path, save_name=save_name)

    # Record time to do forward passes and save images
    batch.update(time.time() - stop)
    stop = time.time()

    # Print model accuracy -- in the code below, val refers to both value and validation
    if i % 25 == 0:
      print('Validate: [{0}/{1}]\t'
            'Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
            'Loss {loss.val:.4f} ({loss.avg:.4f})\t'.format(
             i, len(load_validation), batch_time=batch, loss=losses))

    print('Finished validation.')
    return losses.avg


def train(train_loader, model, criterion, optimizer, epoch):
    print('Starting training epoch {}'.format(epoch))
    model.train()

    # Prepare value counters and timers
    batch_time, data_time, losses = CalcuAvg(), CalcuAvg(), CalcuAvg()

    end = time.time()
    for i, (input_gray, input_ab, target) in enumerate(train_loader):
        
        # Use GPU if available
        if use_gpu: input_gray, input_ab, target = input_gray.cuda(), input_ab.cuda(), target.cuda()

        # Record time to load data (above)
        data_time.update(time.time() - end)

        # Run forward pass
        output_ab = model(input_gray) 
        loss = criterion(output_ab, input_ab) 
        losses.update(loss.item(), input_gray.size(0))

        # Compute gradient and optimize
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        # Record time to do forward and backward passes
        batch_time.update(time.time() - end)
        end = time.time()

        # Print model accuracy -- in the code below, val refers to value, not validation
        if i % 25 == 0:
            print('Epoch: [{0}][{1}/{2}]\t'
                'Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
                'Data {data_time.val:.3f} ({data_time.avg:.3f})\t'
                'Loss {loss.val:.4f} ({loss.avg:.4f})\t'.format(
                epoch, i, len(train_loader), batch_time=batch_time,
                data_time=data_time, loss=losses)) 

    print('Finished training epoch {}'.format(epoch))





if __name__ == "__main__":

  mode = sys.argv[1]

    
  '''os.makedirs('images/train/class/', exist_ok=True) # 40,000 images
  os.makedirs('images/val/class/', exist_ok=True)   #  1,000 images
  for i, file in enumerate(os.listdir('landscape_images')):
      if i < 900: # first 1000 will be val
          os.rename('landscape_images/' + file, 'images/val/class/' + file)
      else: # others will be val
          os.rename('landscape_images/' + file, 'images/train/class/' + file)'''




  #dataset = MapDataset("landscape_images","train")
  #train_loader = DataLoader(dataset, batch_size=5)

  #dataset = MapDataset("landscape_images","val")
  #val_loader = DataLoader(dataset, batch_size=5)

  # Training
  transform_train = transforms.Compose([transforms.RandomResizedCrop(224), transforms.RandomHorizontalFlip()])
  image_train = ConvertFunc('images/train', transform_train)
  load_train = torch.utils.data.DataLoader(image_train, batch_size=64, shuffle=True)

  # Validation 
  transform_validation = transforms.Compose([transforms.Resize(256), transforms.CenterCrop(224)])
  image_validation = ConvertFunc('images/val' , transform_validation)
  load_validation = torch.utils.data.DataLoader(image_validation, batch_size=64, shuffle=False)




  model = ModelNet()

  crit = nn.MSELoss()


  optimizer = torch.optim.Adam(model.parameters(), lr=1e-2, weight_decay=0.0)


  if use_gpu: 
      crit = crit.cuda()
      model = model.cuda()

  os.makedirs('outputs/color', exist_ok=True)
  os.makedirs('outputs/gray', exist_ok=True)
  os.makedirs('checkpoints', exist_ok=True)

  
  save_images = True
  best_losses = 1e10
  epochs = 200


  if mode == "train":


    # Train model
    for epoch in range(epochs):
    # Train for one epoch, then validate
      train(load_train, model, crit, optimizer, epoch)
      with torch.no_grad():
          losses = validate(load_validation, model, crit, save_images, epoch)
      # Save checkpoint and replace old best model if current model is better
      if losses < best_losses:
          best_losses = losses
          torch.save(model.state_dict(), 'checkpoints/model-epoch-{}-losses-{:.3f}.pth'.format(epoch+1,losses))

  
  if mode == "test":

    # Validation 
    test_transforms = transforms.Compose([transforms.Resize(256), transforms.CenterCrop(224)])
    test_imagefolder = ConvertFunc('images/test' , test_transforms)
    test_loader = torch.utils.data.DataLoader(test_imagefolder, batch_size=64, shuffle=False)

    pretrained = torch.load('checkpoints/model-epoch-66-losses-0.003.pth', map_location=lambda storage, loc: storage)
    model.load_state_dict(pretrained) 

    # Validate
    save_images = True
    with torch.no_grad():
      validate(test_loader, model, crit, save_images, 0)