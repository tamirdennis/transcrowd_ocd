from copy import deepcopy

import numpy as np
import torch
from torch.utils.data import DataLoader
from torchvision import transforms
from torchvision.datasets import mnist
from torchvision.transforms import ToTensor

import Lenet5
from nerf_utils.nerf import get_minibatches, positional_encoding
from nerf_utils.tiny_nerf import VeryTinyNerfModel
from TransCrowd_utils.datasets_utils import pre_data, listDataset
from TransCrowd_utils.models import base_patch16_384_token, base_patch16_384_gap

import torch.nn as nn
import os


def wrapper_dataset(config, args, device):
    if args.datatype == 'tinynerf':

        data = np.load(args.data_train_path)
        images = data["images"]
        # Camera extrinsics (poses)
        tform_cam2world = data["poses"]
        tform_cam2world = torch.from_numpy(tform_cam2world).to(device)
        # Focal length (intrinsics)
        focal_length = data["focal"]
        focal_length = torch.from_numpy(focal_length).to(device)

        # Height and width of each image
        height, width = images.shape[1:3]

        # Near and far clipping thresholds for depth values.
        near_thresh = 2.0
        far_thresh = 6.0

        # Hold one image out (for test).
        testimg, testpose = images[101], tform_cam2world[101]
        testimg = torch.from_numpy(testimg).to(device)

        # Map images to device
        images = torch.from_numpy(images[:100, ..., :3]).to(device)
        num_encoding_functions = 10
        # Specify encoding function.
        encode = positional_encoding
        # Number of depth samples along each ray.
        depth_samples_per_ray = 32
        model = VeryTinyNerfModel(num_encoding_functions=num_encoding_functions)
        # Chunksize (Note: this isn't batchsize in the conventional sense. This only
        # specifies the number of rays to be queried in one go. Backprop still happens
        # only after all rays from the current "bundle" are queried and rendered).
        # Use chunksize of about 4096 to fit in ~1.4 GB of GPU memory (when using 8
        # samples per ray).
        chunksize = 4096
        batch = {}
        batch['height'] = height
        batch['width'] = width
        batch['focal_length'] = focal_length
        batch['testpose'] = testpose
        batch['near_thresh'] = near_thresh
        batch['far_thresh'] = far_thresh
        batch['depth_samples_per_ray'] = depth_samples_per_ray
        batch['encode'] = encode
        batch['get_minibatches'] = get_minibatches
        batch['chunksize'] = chunksize
        batch['num_encoding_functions'] = num_encoding_functions
        train_ds, test_ds = [], []
        for img, tfrom in zip(images, tform_cam2world):
            batch['input'] = tfrom
            batch['output'] = img
            train_ds.append(deepcopy(batch))
        batch['input'] = testpose
        batch['output'] = testimg
        test_ds = [batch]
    elif args.datatype == 'mnist':
        model = Lenet5.NetOriginal()
        train_transform = transforms.Compose(
            [
                transforms.ToTensor()
            ])
        train_dataset = mnist.MNIST(
            "\data\mnist", train=True, download=True, transform=ToTensor())
        test_dataset = mnist.MNIST(
            "\data\mnist", train=False, download=True, transform=ToTensor())
        train_loader = DataLoader(train_dataset, batch_size=1, shuffle=True)
        test_loader = DataLoader(test_dataset, batch_size=1)
        train_ds, test_ds = [], []
        for idx, data in enumerate(train_loader):
            train_x, train_label = data[0], data[1]
            train_x = train_x[:, 0, :, :].unsqueeze(1)
            batch = {'input': train_x, 'output': train_label}
            train_ds.append(deepcopy(batch))
        for idx, data in enumerate(test_loader):
            train_x, train_label = data[0], data[1]
            train_x = train_x[:, 0, :, :].unsqueeze(1)
            batch = {'input': train_x, 'output': train_label}
            test_ds.append(deepcopy(batch))
    elif args.datatype == 'transcrowd':
        train_file = '/home/tamirdenis/projects/TransCrowd/npydata/ShanghaiA_train.npy'
        test_file = '/home/tamirdenis/projects/TransCrowd/npydata/ShanghaiA_test.npy'
        
        with open(train_file, 'rb') as outfile:
            train_list = np.load(outfile).tolist()
        with open(test_file, 'rb') as outfile:
            test_list = np.load(outfile).tolist()

        print(len(train_list), len(test_list))

        os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu_id

        if args.model_type == 'token':
            model = base_patch16_384_token(pretrained=False, num_classes=1000)
        else:
            model = base_patch16_384_gap(pretrained=True)
        model = nn.DataParallel(model, device_ids=[0, 1, 2, 3])
        model = model.cuda()

        # torch.set_num_threads(args.workers)

        train_data = pre_data(train_list, args, train=True)
        test_data = pre_data(test_list, args, train=False)
        train_loader = torch.utils.data.DataLoader(
                    listDataset(train_data, None,
                                shuffle=True,
                                transform=transforms.Compose([
                                    transforms.ToTensor(),

                                    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                                         std=[0.229, 0.224, 0.225]),
                                ]),
                                train=True,
                                batch_size=1,
                                num_workers=args.workers,
                                args=args),
            batch_size=1, drop_last=False)
        test_loader = torch.utils.data.DataLoader(
            listDataset(test_data, None,
                                shuffle=False,
                                transform=transforms.Compose([
                                    transforms.ToTensor(), transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                                                                std=[0.229, 0.224, 0.225]),

                                ]),
                                args=args, train=False),
            batch_size=1)
        train_ds, test_ds = [], []
        for idx, data in enumerate(train_loader):
            fname, train_x, train_label = data
            batch = {'input': train_x, 'output': train_label}
            train_ds.append(deepcopy(batch))
        for idx, data in enumerate(test_loader):
            fname, test_x, test_label = data
            batch = {'input': test_x, 'output': test_label}
            test_ds.append(deepcopy(batch))

    else:
        "implement on your own"
        pass
    return train_ds, test_ds, model
