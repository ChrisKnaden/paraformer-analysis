import argparse
import os
import random
import numpy as np
import torch
import torch.backends.cudnn as cudnn
from networks.vit_seg_modeling import VisionTransformer as ViT_seg
from networks.vit_seg_modeling import CONFIGS as CONFIGS_ViT_seg
from trainer import trainer_dataset
import os
from networks.vit_seg_modeling_L2HNet import L2HNet

parser = argparse.ArgumentParser()
parser.add_argument('--dataset', type=str, default='Poland', help='experiment_name')
parser.add_argument('--max_epochs', type=int, default=100, help='maximum epoch number to train')
parser.add_argument('--batch_size', type=int, default=16, help='batch_size per gpu')
parser.add_argument('--base_lr', type=float,  default=0.01, help='segmentation network learning rate')
parser.add_argument('--seed', type=int, default=1234, help='random seed')
parser.add_argument('--CNN_width', type=int, default=64, help='L2HNet_width_size, default is 64: light mode. Set to 128: normal mode')
parser.add_argument('--savepath', type=str)
parser.add_argument('--gpu', type=str, help='Select GPU number to train' )
parser.add_argument('--save_interval', type=int, default=20, help='Save model every N epochs')
parser.add_argument('--pretrained_weights', type=str, default=None, help='Path to pre-trained weights (.pth or .npz)')
args = parser.parse_args()

os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu
if __name__ == "__main__":
    vit_patches_size=16
    img_size=224
    cudnn.benchmark = True
    cudnn.deterministic = False
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed(args.seed)
    dataset_name = args.dataset
    dataset_config = {
        'Chesapeake': {
            'list_dir': './dataset/CSV_list/Chesapeake_NewYork.csv',
            'num_classes': 17
        },
        'Poland': {
            'list_dir': './dataset/CSV_list/Poland.csv',
            'num_classes': 11
        },
        'NRW': {
            'list_dir': './dataset/CSV_list/NRW.csv',
            'num_classes': 11
        }
    }# Create a config to your own dataset here
    if args.batch_size != 24 and args.batch_size % 6 == 0:
        args.base_lr *= args.batch_size / 24
    args.num_classes = dataset_config[dataset_name]['num_classes']
    args.list_dir = dataset_config[dataset_name]['list_dir']
    args.is_pretrain = True
    snapshot_path = args.savepath 
    if not os.path.exists(snapshot_path):
        os.makedirs(snapshot_path)
    config_vit = CONFIGS_ViT_seg["ViT-B_16"]
    config_vit.n_classes = args.num_classes
    config_vit.patches.grid = (int(img_size / vit_patches_size), int(img_size / vit_patches_size))

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu") # Hopefully, noone will use CPU for training
    net = ViT_seg(config_vit, backbone=L2HNet(width=args.CNN_width), img_size=img_size,
                  num_classes=config_vit.n_classes).to(device)

    # Add option to load pretrained weights as .pth or .npz file
    if args.pretrained_weights:
        if args.pretrained_weights.endswith('.pth'):
            print(f"Loading weights from .pth file: {args.pretrained_weights}")
            net.load_state_dict(torch.load(args.pretrained_weights, map_location=device))
        elif args.pretrained_weights.endswith('.npz'):
            print(f"Loading weights from .npz file: {args.pretrained_weights}")
            try:
                net.load_from(weights=np.load(args.pretrained_weights))
            except AttributeError:
                print("Error: 'load_from' method not found in ViT_seg. Cannot load .npz file.")
                print("Please ensure your ViT_seg model has a method to load .npz weights.")
                exit()
        else:
            print(f"Unsupported pretrained weight file format: {args.pretrained_weights}")
            print("Please provide a .pth or .npz file.")
            exit()
    elif hasattr(config_vit, 'pretrained_path') and config_vit.pretrained_path:
        print(
            f"No command-line pretrained weights specified. Loading from config_vit.pretrained_path: {config_vit.pretrained_path}")
        if config_vit.pretrained_path.endswith('.pth'):
            net.load_state_dict(torch.load(config_vit.pretrained_path, map_location=device))
        elif config_vit.pretrained_path.endswith('.npz'):
            try:
                net.load_from(weights=np.load(config_vit.pretrained_path))
            except AttributeError:
                print("Error: 'load_from' method not found in ViT_seg. Cannot load .npz file from config.")
                exit()
        else:
            print(f"Unsupported pretrained weight file format in config: {config_vit.pretrained_path}")
            print("Please ensure the path in config is a .pth or .npz file.")
            exit()
    else:
        print(
            "No pretrained weights specified (neither via command line nor in config). Starting training from scratch or using default initialization.")

    net.eval()

    trainer_dataset(args, net, snapshot_path)
