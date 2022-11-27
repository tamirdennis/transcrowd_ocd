import argparse

parser = argparse.ArgumentParser(description='TransCrowd')

# Data specifications
parser.add_argument('--dataset_version_dir', type=str,
                    help='')

parser.add_argument('--gt_dir', type=str, default='data/blood tests in excel',
                    help='')

parser.add_argument('--gt_key', type=str, default='WBC',
                    help='')

parser.add_argument('--crop_size', type=int, nargs='+', default=(972, 1296),
                    help='')

parser.add_argument('--workers', type=int, default=16,
                    help='load data workers')

parser.add_argument('--results_dir', type=str, default='./results/veye_transcrowd_batch1_fullsize',
                    help='folder for saving results')

parser.add_argument('--use_existing_results', action='store_true',
                    help='if using the results in {results_dir}/images_results.json')


# Model specifications
parser.add_argument('--pre', type=str, default='./save_file/veye_transcrowd_batch1_fullsize/model_best.pth',
                    help='pre-trained model directory')


# Optimization specifications
parser.add_argument('--batch_size', type=int, default=8,
                    help='input batch size for training')
parser.add_argument('--gpu_id', type=str, default='0',
                    help='gpu id')

# nni config
parser.add_argument('--model_type', type=str, default='token',
                    help='model type')

args = parser.parse_args()
return_args = parser.parse_args()
