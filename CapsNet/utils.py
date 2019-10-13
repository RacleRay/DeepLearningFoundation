"""https://github.com/XifengGuo/CapsNet-Pytorch/blob/master/utils.py"""

import matplotlib.pyplot as plt
from PIL import Image
import numpy as np
import csv
import math
import torch
from torchvision import transforms, datasets
from torch.autograd import Variable


def show_reconstruction(model, test_loader, n_images, args):
    model.eval()
    for x, _ in test_loader:
        x = Variable(x[:min(n_images, x.size(0))].cuda(), volatile=True)
        _, x_recon = model(x)
        data = np.concatenate([x.data, x_recon.data])
        img = combine_images(np.transpose(data, [0, 2, 3, 1]))
        image = img * 255
        Image.fromarray(image.astype(np.uint8)).save(args.save_dir + "/real_and_recon.png")
        print()
        print('Reconstructed images are saved to %s/real_and_recon.png' % args.save_dir)
        print('-' * 70)
        plt.imshow(plt.imread(args.save_dir + "/real_and_recon.png", ))
        plt.show()
        break


def load_mnist(path='./data', download=False, batch_size=100, shift_pixels=2):
    """
    Construct dataloaders for training and test data.
    shift_pixels: maximum number of pixels to shift in each direction
    """
    kwargs = {'num_workers': 1, 'pin_memory': True}

    train_loader = torch.utils.data.DataLoader(
        datasets.MNIST(path, train=True, download=download,
                       transform=transforms.Compose([transforms.RandomCrop(size=28, padding=shift_pixels),
                                                     transforms.ToTensor()])),
        batch_size=batch_size, shuffle=True, **kwargs)

    test_loader = torch.utils.data.DataLoader(
        datasets.MNIST(path, train=False, download=download,
                       transform=transforms.ToTensor()),
        batch_size=batch_size, shuffle=True, **kwargs)

    return train_loader, test_loader


def plot_log(filename, show=True):
    # load data
    keys = []
    values = []
    with open(filename, 'r') as f:
        reader = csv.DictReader(f)
        for row in reader:
            if keys == []:
                for key, value in row.items():
                    keys.append(key)
                    values.append(float(value))
                continue

            for _, value in row.items():
                values.append(float(value))

        values = np.reshape(values, newshape=(-1, len(keys)))

    fig = plt.figure(figsize=(4,6))
    fig.subplots_adjust(top=0.95, bottom=0.05, right=0.95)
    fig.add_subplot(211)
    epoch_axis = 0
    for i, key in enumerate(keys):
        if key == 'epoch':
            epoch_axis = i
            values[:, epoch_axis] += 1
            break
    for i, key in enumerate(keys):
        if key.find('loss') >= 0:  # loss
            print(values[:, i])
            plt.plot(values[:, epoch_axis], values[:, i], label=key)
    plt.legend()
    plt.title('Training loss')

    fig.add_subplot(212)
    for i, key in enumerate(keys):
        if key.find('acc') >= 0:  # acc
            plt.plot(values[:, epoch_axis], values[:, i], label=key)
    plt.legend()
    plt.grid()
    plt.title('Accuracy')

    # fig.savefig('result/log.png')
    if show:
        plt.show()


def combine_images(generated_images):
    num = generated_images.shape[0]
    width = int(math.sqrt(num))
    height = int(math.ceil(float(num)/width))
    shape = generated_images.shape[1:3]
    image = np.zeros((height*shape[0], width*shape[1]),
                     dtype=generated_images.dtype)
    for index, img in enumerate(generated_images):
        i = int(index/width)
        j = index % width
        image[i*shape[0]:(i+1)*shape[0], j*shape[1]:(j+1)*shape[1]] = \
            img[:, :, 0]
    return image