import math

import nni
import torch
from torchvision import transforms
import json
from pathlib import Path
import numpy as np

from source.utils.TransCrowd_utils.datasets_utils import listDataset


def validate(Pre_data, model, args):
    print('begin test')
    batch_size = 1
    test_loader = torch.utils.data.DataLoader(
        listDataset(Pre_data, args['crop_size'],
                            shuffle=False,
                            transform=transforms.Compose([
                                transforms.ToTensor(),
                                transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                                     std=[0.229, 0.224, 0.225]),
                                transforms.RandomCrop(args['crop_size'])

                            ]),
                            args=args, train=False),
        batch_size=1)

    model.eval()

    mae = 0.0
    mse = 0.0
    results = {}

    for i, (fname, img, gt) in enumerate(test_loader):
        fname = fname[0]
        img = img.cuda()
        if len(img.shape) == 5:
            img = img.squeeze(0)
        if len(img.shape) == 3:
            img = img.unsqueeze(0)

        with torch.no_grad():
            pred = model(img)
        float_gt = float(gt[0])
        float_pred = float(pred[0])
        results[fname] = {'gt': float_gt, 'pred': float_pred}

        mae += abs(float_gt - float_pred)
        mse += abs(float_gt - float_pred) * abs(float_gt - float_pred)

        if i % 15 == 0:
            print('{fname} Gt {gt:.2f} Pred {pred}'.format(fname=fname, gt=float_gt, pred=float_pred))

    mae = mae * 1.0 / (len(test_loader) * batch_size)
    mse = math.sqrt(mse / (len(test_loader)) * batch_size)

    nni.report_intermediate_result(mae)
    print(' \n* MAE {mae:.3f}\n'.format(mae=mae), '* MSE {mse:.3f}'.format(mse=mse))

    return mae, results


def save_results_in_json(results, json_p):
    with open(json_p, 'w') as json_file:
        json.dump(results, json_file)


def _get_video_name_from_img_name(img_name):
    img_last_name_index = img_name.split('_')[-1]
    return img_name.split(f'_{img_last_name_index}')[0]


def get_results_per_video(results):
    results_per_video = {}
    for file_p, file_d in results.items():
        file_name = Path(file_p).name
        file_video_name = _get_video_name_from_img_name(file_name)
        if file_video_name not in results_per_video:
            results_per_video[file_video_name] = {'gt': file_d['gt'], 'preds': [file_d['pred']]}
        else:
            results_per_video[file_video_name]['preds'].append(file_d['pred'])
    videos_errors = []
    for vid_name, vid_d in results_per_video.items():
        print()
        print(vid_name)
        print('GT ', vid_d['gt'])
        vid_d['preds_mean'] = np.mean(vid_d['preds'])
        vid_d['preds_std'] = np.std(vid_d['preds'])
        videos_errors.append(np.abs(vid_d['gt'] - vid_d['preds_mean']))
        print(vid_d['preds_mean'], vid_d['preds_std'])
    print('videos mae: ', np.mean(videos_errors))

    return results_per_video