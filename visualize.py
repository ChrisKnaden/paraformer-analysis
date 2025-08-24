import argparse
import os
import rasterio
import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
import matplotlib.cm as cm
from torch.utils.data import Dataset, DataLoader
from rasterio.windows import Window
from tqdm import tqdm

try:
    from networks.vit_seg_modeling_L2HNet import L2HNet
    from networks.vit_seg_modeling import VisionTransformer as ViT_seg
    from networks.vit_seg_modeling import CONFIGS as CONFIGS_ViT_seg
except ImportError:
    print("Error: Could not import network definitions.")
    print("Please ensure 'networks/vit_seg_modeling.py' and 'networks/vit_seg_modeling_L2HNet.py' are accessible.")
    exit()

IMAGE_MEANS = np.array([106.13, 110.64, 95.32]) # Poland
IMAGE_STDS = np.array([45.93, 38.80, 33.04]) # Poland

# --- Label and Colormap Definitions ---
LABEL_CLASSES = [0, 10, 20, 30, 40, 50, 60, 70, 80, 90, 95, 100]
LABEL_CLASS_COLORMAP = {
    0: (0, 0, 0), 10: (0, 100, 0), 20: (255, 187, 34), 30: (255, 255, 76),
    40: (240, 150, 255), 50: (255, 0, 0), 60: (180, 180, 180), 70: (240, 240, 240),
    80: (0, 100, 200), 90: (0, 150, 160), 95: (0, 207, 117), 100: (250, 230, 160)
}
LABEL_IDX_COLORMAP = {idx: LABEL_CLASS_COLORMAP[c] for idx, c in enumerate(LABEL_CLASSES)}

# --- mIoU Evaluation Mappings ---
CLASS_COLORMAP_MIOU = {
    0: (255, 0, 0),  # Built-up
    1: (0, 100, 0),  # Tree cover
    2: (255, 255, 76),  # Low vegetation
    3: (0, 100, 200),  # Water
    4: (0, 0, 0)  # Nothing/Void
}
# Map from model prediction index -> evaluation class index
IDX_TO_EVAL_MAP = {
    0: 0, 1: 1, 2: 2, 3: 2, 4: 2, 5: 0, 6: 2, 7: 2, 8: 3, 9: 2, 10: 1, 11: 2
}
# Map from high-resolution ground truth value -> evaluation class index
GT_TO_EVAL_MAP = {
    0: 4, 1: 2, 2: 2, 3: 0, 4: 0, 5: 1, 6: 3, 7: 2, 8: 0
}
# Map from low-resolution label value -> evaluation class index
LR_RAW_TO_EVAL_MAP = {val: IDX_TO_EVAL_MAP.get(idx, 4) for idx, val in enumerate(LABEL_CLASSES)}

# --- Vectorized mapping functions for speed ---
map_pred_indices_to_eval_v = np.vectorize(lambda x: IDX_TO_EVAL_MAP.get(x, 4))
map_gt_values_to_eval_v = np.vectorize(lambda x: GT_TO_EVAL_MAP.get(x, 4))
map_lr_values_to_eval_v = np.vectorize(lambda x: LR_RAW_TO_EVAL_MAP.get(x, 4))

class TileInferenceDataset(Dataset):
    def __init__(self, fn, chip_size, stride, transform=None, verbose=False):
        self.fn = fn
        self.chip_size = chip_size
        self.transform = transform

        with rasterio.open(self.fn) as f:
            height, width = f.height, f.width
            self.profile = f.profile.copy()

        self.chip_coordinates = []
        # Create a list of top-left coordinates (y, x) for each chip
        for y in list(range(0, height - self.chip_size, stride)) + [height - self.chip_size]:
            for x in list(range(0, width - self.chip_size, stride)) + [width - self.chip_size]:
                self.chip_coordinates.append((y, x))
        self.num_chips = len(self.chip_coordinates)

        if verbose:
            print(f"Constructed TileInferenceDataset: Tiling {width}x{height} image into {self.num_chips} chips.")

    def __getitem__(self, idx):
        y, x = self.chip_coordinates[idx]
        with rasterio.open(self.fn) as f:
            # Read only the first 3 channels (assumed RGB) for visualization
            img = np.rollaxis(
                f.read((1, 2, 3), window=Window(x, y, self.chip_size, self.chip_size)), 0, 3)

        original_chip = img.copy()

        if self.transform is not None:
            img = self.transform(img)

        return img, original_chip, np.array((y, x))

    def __len__(self):
        return self.num_chips


class ProcessVisualizationDataset(Dataset):
    def __init__(self, hr_image_fn, lr_label_fn, hr_gt_fn, chip_size, stride):
        self.hr_image_fn, self.lr_label_fn, self.hr_gt_fn = hr_image_fn, lr_label_fn, hr_gt_fn
        self.chip_size = chip_size
        with rasterio.open(hr_image_fn) as f:
            height, width = f.height, f.width
        self.chip_coordinates = []
        for y in list(range(0, height - chip_size, stride)) + [height - chip_size]:
            for x in list(range(0, width - chip_size, stride)) + [width - chip_size]:
                self.chip_coordinates.append((y, x))

    def __getitem__(self, idx):
        y, x = self.chip_coordinates[idx]
        window = Window(x, y, self.chip_size, self.chip_size)

        with rasterio.open(self.hr_image_fn) as f:
            hr_chip = np.rollaxis(f.read((1, 2, 3), window=window), 0, 3)
        with rasterio.open(self.lr_label_fn) as f:
            lr_chip_raw = f.read(1, window=window)
        with rasterio.open(self.hr_gt_fn) as f:
            gt_chip_raw = f.read(1, window=window)

        # Normalize image for model input
        hr_chip_tensor = (torch.from_numpy(hr_chip).permute(2, 0, 1).float() - torch.tensor(IMAGE_MEANS).float().view(3,
                                                                                                                      1,
                                                                                                                      1)) / torch.tensor(
            IMAGE_STDS).float().view(3, 1, 1)

        return hr_chip, hr_chip_tensor, lr_chip_raw, gt_chip_raw, np.array((y, x))

    def __len__(self):
        return len(self.chip_coordinates)

def image_transforms(img: np.ndarray) -> torch.Tensor:
    img = (img.astype(np.float32) - IMAGE_MEANS) / IMAGE_STDS
    img = np.rollaxis(img, 2, 0)  # HWC to CHW
    return torch.from_numpy(img).float()


def save_prediction_as_geotiff(data_array, output_fn, original_profile, colormap):
    pred_profile = original_profile.copy()
    pred_profile["driver"] = "GTiff"
    pred_profile["dtype"] = "uint8"
    pred_profile["count"] = 1
    pred_profile["nodata"] = 255  # Use a value outside of class indices

    with rasterio.open(output_fn, "w", **pred_profile) as f:
        f.write(data_array, 1)
        f.write_colormap(1, colormap)
    print(f"Saved prediction to {output_fn}")


def save_rgb_image(data_array, output_fn):
    Image.fromarray(data_array.astype(np.uint8)).save(output_fn)
    print(f"Saved image to {output_fn}")


def save_heatmap(heatmap_array, output_fn, percentile=(2, 98), colormap_name='viridis'):
    vmin, vmax = np.percentile(heatmap_array, percentile)
    clipped_map = np.clip(heatmap_array, vmin, vmax)
    normalized_map = (clipped_map - vmin) / (vmax - vmin)
    colormap = getattr(cm, colormap_name)
    colored_map = (colormap(normalized_map)[:, :, :3] * 255).astype(np.uint8)
    Image.fromarray(colored_map).save(output_fn)
    print(f"Saved heatmap to {output_fn}")


def get_feature_heatmap(feature_tensor, aggregation='mean'):
    if aggregation == 'mean':
        return torch.mean(feature_tensor, dim=0)
    elif aggregation == 'max':
        return torch.max(feature_tensor, dim=0)[0]
    elif aggregation == 'l2':
        return torch.linalg.norm(feature_tensor, ord=2, dim=0)
    else:
        raise ValueError(f"Unknown aggregation method: {aggregation}")


def create_heatmap_from_patches(patch_heatmap, patch_size, output_size):
    grid_h, grid_w = patch_heatmap.shape
    full_heatmap = torch.zeros(output_size, device=patch_heatmap.device)
    for i in range(grid_h):
        for j in range(grid_w):
            value = patch_heatmap[i, j]
            y_start, x_start = i * patch_size, j * patch_size
            full_heatmap[y_start:y_start + patch_size, x_start:x_start + patch_size] = value
    return full_heatmap


def map_eval_classes_to_rgb(eval_classes, colormap):
    rgb_image = np.zeros((*eval_classes.shape, 3), dtype=np.uint8)
    for eval_class, color in colormap.items():
        rgb_image[eval_classes == eval_class] = color
    return rgb_image


def create_error_overlay(base_image_chip, pred_eval, gt_eval, alpha=0.6):
    grayscale_pil = Image.fromarray(base_image_chip).convert('L').convert('RGB')
    highlight_layer = Image.new('RGB', grayscale_pil.size, color='deeppink')

    # An error is a mismatch where neither the prediction nor the ground truth is a 'void' class (4)
    error_mask = (pred_eval != gt_eval) & (gt_eval != 4) & (pred_eval != 4)
    alpha_mask_pil = Image.fromarray((error_mask * 255).astype(np.uint8), 'L')

    composite = grayscale_pil.copy()
    composite.paste(highlight_layer, (0, 0), alpha_mask_pil)

    final_image = Image.blend(grayscale_pil, composite, alpha=alpha)
    return np.array(final_image)


def visualize_branches(args, model):
    print("\n--- Running Visualization Mode: [branches] ---")
    os.makedirs(args.output_dir, exist_ok=True)

    CHIP_SIZE = args.img_size
    PADDING = 112
    HALF_PADDING = PADDING // 2
    CHIP_STRIDE = CHIP_SIZE - PADDING

    print(f"Processing {args.input_tif}")
    dataset = TileInferenceDataset(args.input_tif, CHIP_SIZE, CHIP_STRIDE, transform=image_transforms, verbose=True)
    dataloader = DataLoader(dataset, batch_size=args.batch_size, num_workers=4, pin_memory=True)

    with rasterio.open(args.input_tif) as src:
        height, width = src.height, src.width
        input_profile = src.profile.copy()

    # Create accumulators for each branch's probabilities
    output_cnn = np.zeros((args.num_classes, height, width), dtype=np.float32)
    output_vit = np.zeros((args.num_classes, height, width), dtype=np.float32)
    output_fused = np.zeros((args.num_classes, height, width), dtype=np.float32)

    # Weighted kernel to reduce edge artifacts from tiling
    kernel = np.ones((CHIP_SIZE, CHIP_SIZE), dtype=np.float32)
    kernel[HALF_PADDING:-HALF_PADDING, HALF_PADDING:-HALF_PADDING] = 5
    counts = np.zeros((height, width), dtype=np.float32)

    # Save original image for reference
    with rasterio.open(args.input_tif) as src:
        img_rgb = np.rollaxis(src.read((1, 2, 3)), 0, 3)
        p2, p98 = np.percentile(img_rgb, (2, 98))
        img_rescaled = np.clip((img_rgb - p2) * 255.0 / (p98 - p2), 0, 255)
        save_rgb_image(img_rescaled, os.path.join(args.output_dir, "a_original_image.png"))

    # Run inference
    for data, _, coords in tqdm(dataloader, desc="Inferring on tiles"):
        data = data.cuda()
        with torch.no_grad():
            logits_cnn, logits_vit = model(data)
            probs_cnn = F.softmax(logits_cnn, dim=1).cpu().numpy()
            probs_vit = F.softmax(logits_vit, dim=1).cpu().numpy()
            probs_fused = F.softmax((logits_cnn + logits_vit) / 2, dim=1).cpu().numpy()

        for j in range(data.shape[0]):
            y, x = coords[j]
            output_cnn[:, y:y + CHIP_SIZE, x:x + CHIP_SIZE] += probs_cnn[j] * kernel
            output_vit[:, y:y + CHIP_SIZE, x:x + CHIP_SIZE] += probs_vit[j] * kernel
            output_fused[:, y:y + CHIP_SIZE, x:x + CHIP_SIZE] += probs_fused[j] * kernel
            counts[y:y + CHIP_SIZE, x:x + CHIP_SIZE] += kernel

    # Normalize accumulated probabilities and get hard predictions
    counts[counts == 0] = 1e-6
    pred_cnn_hard = (output_cnn / counts).argmax(axis=0).astype(np.uint8)
    pred_vit_hard = (output_vit / counts).argmax(axis=0).astype(np.uint8)
    pred_fused_hard = (output_fused / counts).argmax(axis=0).astype(np.uint8)

    # Save prediction maps
    save_prediction_as_geotiff(pred_cnn_hard, os.path.join(args.output_dir, "b_prediction_cnn_branch.tif"),
                               input_profile, LABEL_IDX_COLORMAP)
    save_prediction_as_geotiff(pred_vit_hard, os.path.join(args.output_dir, "c_prediction_vit_branch.tif"),
                               input_profile, LABEL_IDX_COLORMAP)
    save_prediction_as_geotiff(pred_fused_hard, os.path.join(args.output_dir, "d_prediction_fused.tif"), input_profile,
                               LABEL_IDX_COLORMAP)
    print("\nBranch visualization finished!")


def visualize_features(args, model):
    print("\n--- Running Visualization Mode: [features] ---")
    os.makedirs(args.output_dir, exist_ok=True)

    CHIP_SIZE, PADDING = args.img_size, 112
    HALF_PADDING, CHIP_STRIDE = PADDING // 2, CHIP_SIZE - PADDING

    print(f"Processing {args.input_tif} to generate feature heatmaps...")
    dataset = TileInferenceDataset(args.input_tif, CHIP_SIZE, CHIP_STRIDE, transform=image_transforms, verbose=True)
    dataloader = DataLoader(dataset, batch_size=args.batch_size, num_workers=4, pin_memory=True)

    with rasterio.open(args.input_tif) as src:
        height, width = src.height, src.width

    # Create accumulators for feature heatmaps
    heatmap_cnn = np.zeros((height, width), dtype=np.float32)
    heatmap_vit = np.zeros((height, width), dtype=np.float32)
    heatmap_hybrid = np.zeros((height, width), dtype=np.float32)
    heatmap_fused_decoder = np.zeros((height, width), dtype=np.float32)

    kernel = np.ones((CHIP_SIZE, CHIP_SIZE), dtype=np.float32)
    kernel[HALF_PADDING:-HALF_PADDING, HALF_PADDING:-HALF_PADDING] = 5
    counts = np.zeros((height, width), dtype=np.float32)

    # Save original image for reference
    with rasterio.open(args.input_tif) as src:
        img_rgb = np.rollaxis(src.read((1, 2, 3)), 0, 3)
        p2, p98 = np.percentile(img_rgb, (2, 98))
        img_rescaled = np.clip((img_rgb - p2) * 255.0 / (p98 - p2), 0, 255)
        save_rgb_image(img_rescaled, os.path.join(args.output_dir, "a_original_image.png"))

    # Run inference and extract features
    for data, _, coords in tqdm(dataloader, desc="Extracting features from tiles"):
        data = data.cuda()
        with torch.no_grad():
            _, _, features_cnn, features_vit, features_hybrid = model(data, visualize_features=True)

        for j in range(data.shape[0]):
            y, x = coords[j]
            agg_method = 'mean'

            # Get heatmaps from decoder branches
            f_cnn = get_feature_heatmap(features_cnn[j], agg_method)
            f_vit = get_feature_heatmap(features_vit[j], agg_method)
            f_fused = (f_cnn + f_vit) / 2.0

            f_cnn_up = F.interpolate(f_cnn.unsqueeze(0).unsqueeze(0), size=(CHIP_SIZE, CHIP_SIZE), mode='bilinear',
                                     align_corners=False).squeeze()
            f_vit_up = F.interpolate(f_vit.unsqueeze(0).unsqueeze(0), size=(CHIP_SIZE, CHIP_SIZE), mode='bilinear',
                                     align_corners=False).squeeze()
            f_fused_up = F.interpolate(f_fused.unsqueeze(0).unsqueeze(0), size=(CHIP_SIZE, CHIP_SIZE), mode='bilinear',
                                       align_corners=False).squeeze()

            heatmap_cnn[y:y + CHIP_SIZE, x:x + CHIP_SIZE] += f_cnn_up.cpu().numpy() * kernel
            heatmap_vit[y:y + CHIP_SIZE, x:x + CHIP_SIZE] += f_vit_up.cpu().numpy() * kernel
            heatmap_fused_decoder[y:y + CHIP_SIZE, x:x + CHIP_SIZE] += f_fused_up.cpu().numpy() * kernel

            # Get heatmap from hybrid backbone (patch-based)
            h, w = int(np.sqrt(features_hybrid.shape[1])), int(np.sqrt(features_hybrid.shape[1]))
            f_hybrid_reshaped = features_hybrid[j].permute(1, 0).contiguous().view(features_hybrid.shape[2], h, w)
            f_hybrid_patch_map = get_feature_heatmap(f_hybrid_reshaped, agg_method)
            patch_size = CHIP_SIZE // h
            f_hybrid_up = create_heatmap_from_patches(f_hybrid_patch_map, patch_size, (CHIP_SIZE, CHIP_SIZE))
            heatmap_hybrid[y:y + CHIP_SIZE, x:x + CHIP_SIZE] += f_hybrid_up.cpu().numpy() * kernel

            counts[y:y + CHIP_SIZE, x:x + CHIP_SIZE] += kernel

    # Normalize and save heatmaps
    counts[counts == 0] = 1e-6
    save_heatmap(heatmap_cnn / counts, os.path.join(args.output_dir, "b_heatmap_cnn_decoder.png"))
    save_heatmap(heatmap_vit / counts, os.path.join(args.output_dir, "c_heatmap_vit_decoder.png"))
    save_heatmap(heatmap_hybrid / counts, os.path.join(args.output_dir, "d_heatmap_hybrid_backbone.png"))
    save_heatmap(heatmap_fused_decoder / counts, os.path.join(args.output_dir, "e_heatmap_fused_decoder.png"))
    print("\nFeature visualization finished!")


def visualize_process(args, model):
    print("\n--- Running Visualization Mode: [process] ---")
    os.makedirs(args.output_dir, exist_ok=True)

    with rasterio.open(args.hr_image_tif) as src:
        height, width = src.height, src.width

    # Define accumulators for the full-sized output images
    plot_data = {
        'a_hr_image': np.zeros((height, width, 3), dtype=np.uint8),
        'b_lr_label': np.zeros((height, width, 3), dtype=np.uint8),
        'c_cnn_pred': np.zeros((height, width, 3), dtype=np.uint8),
        'd_mask_label': np.zeros((height, width, 3), dtype=np.uint8),
        'e_lr_errors': np.zeros((height, width, 3), dtype=np.uint8),
        'f_mask_errors': np.zeros((height, width, 3), dtype=np.uint8),
        'g_fused_pred': np.zeros((height, width, 3), dtype=np.uint8),
        'h_gt_label': np.zeros((height, width, 3), dtype=np.uint8),
    }

    CHIP_SIZE = args.img_size
    CHIP_STRIDE = CHIP_SIZE - (CHIP_SIZE // 4)
    dataset = ProcessVisualizationDataset(args.hr_image_tif, args.lr_label_tif, args.hr_gt_tif, CHIP_SIZE, CHIP_STRIDE)
    dataloader = DataLoader(dataset, batch_size=args.batch_size, num_workers=4, pin_memory=True)

    model.eval()
    for hr_chip_np, hr_chip_tensor, lr_chip_raw, gt_chip_raw, coords in tqdm(dataloader, desc="Processing tiles"):
        hr_chip_tensor = hr_chip_tensor.cuda()
        with torch.no_grad():
            logits_cnn, logits_vit = model(hr_chip_tensor)
            cnn_pred_indices = torch.argmax(logits_cnn, dim=1).cpu().numpy()
            final_pred_indices = torch.argmax((logits_cnn + logits_vit) / 2, dim=1).cpu().numpy()

        for i in range(hr_chip_tensor.shape[0]):
            y, x = coords[i]

            # Map raw model/label values to evaluation classes (0-4)
            cnn_pred_eval = map_pred_indices_to_eval_v(cnn_pred_indices[i])
            final_pred_eval = map_pred_indices_to_eval_v(final_pred_indices[i])
            lr_label_eval = map_lr_values_to_eval_v(lr_chip_raw[i].numpy())
            hr_gt_eval = map_gt_values_to_eval_v(gt_chip_raw[i].numpy())

            # The "Mask Label" is where the CNN prediction agrees with the noisy LR label
            mask_label_eval = np.where(cnn_pred_eval == lr_label_eval, lr_label_eval, 4)  # 4 is void

            # Create error overlays
            lr_error_chip = create_error_overlay(hr_chip_np[i].numpy(), lr_label_eval, hr_gt_eval)
            mask_error_chip = create_error_overlay(hr_chip_np[i].numpy(), mask_label_eval, hr_gt_eval)

            # Place processed chips into the full-size arrays
            s = (slice(y, y + CHIP_SIZE), slice(x, x + CHIP_SIZE))
            plot_data['a_hr_image'][s] = hr_chip_np[i].numpy()
            plot_data['b_lr_label'][s] = map_eval_classes_to_rgb(lr_label_eval, CLASS_COLORMAP_MIOU)
            plot_data['c_cnn_pred'][s] = map_eval_classes_to_rgb(cnn_pred_eval, CLASS_COLORMAP_MIOU)
            plot_data['d_mask_label'][s] = map_eval_classes_to_rgb(mask_label_eval, CLASS_COLORMAP_MIOU)
            plot_data['e_lr_errors'][s] = lr_error_chip
            plot_data['f_mask_errors'][s] = mask_error_chip
            plot_data['g_fused_pred'][s] = map_eval_classes_to_rgb(final_pred_eval, CLASS_COLORMAP_MIOU)
            plot_data['h_gt_label'][s] = map_eval_classes_to_rgb(hr_gt_eval, CLASS_COLORMAP_MIOU)

    # Save each full-sized image individually
    print("\nSaving individual output images...")
    for name, image_array in plot_data.items():
        output_path = os.path.join(args.output_dir, f"{name}.png")
        save_rgb_image(image_array, output_path)

    print("\nProcess visualization finished!")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="L2H-Net Model Visualization Tool")
    parser.add_argument('--gpu', type=str, default='0', help='Select GPU number to use.')
    subparsers = parser.add_subparsers(dest='mode', required=True, help='Visualization mode')

    # --- Common arguments for all modes ---
    common_parser = argparse.ArgumentParser(add_help=False)
    common_parser.add_argument('--model_path', type=str, required=True, help='Path to the trained model .pth file.')
    common_parser.add_argument('--output_dir', type=str, required=True,
                               help='Directory to save the visualization outputs.')
    common_parser.add_argument('--num_classes', type=int, default=11, help='Number of classes for the model.')
    common_parser.add_argument('--CNN_width', type=int, default=64, help='L2HNet width size (64: light, 128: normal).')
    common_parser.add_argument('--img_size', type=int, default=224, help='Input patch size of the network.')
    common_parser.add_argument('--batch_size', type=int, default=16, help='Batch size for inference.')

    # --- Parser for 'branches' mode ---
    parser_branches = subparsers.add_parser('branches', help='Visualize CNN, ViT, and fused branch predictions.',
                                            parents=[common_parser])
    parser_branches.add_argument('--input_tif', type=str, required=True,
                                 help='Path to the single .tif image to process.')
    parser_branches.set_defaults(func=visualize_branches)

    # --- Parser for 'features' mode ---
    parser_features = subparsers.add_parser('features', help='Visualize feature activation heatmaps.',
                                            parents=[common_parser])
    parser_features.add_argument('--input_tif', type=str, required=True,
                                 help='Path to the single .tif image to process.')
    parser_features.set_defaults(func=visualize_features)

    # --- Parser for 'process' mode ---
    parser_process = subparsers.add_parser('process',
                                           help='Visualize training process components (labels, errors, etc.).',
                                           parents=[common_parser])
    parser_process.add_argument('--hr_image_tif', type=str, required=True,
                                help='Path to the high-resolution source image.')
    parser_process.add_argument('--lr_label_tif', type=str, required=True,
                                help='Path to the low-resolution (noisy) label map.')
    parser_process.add_argument('--hr_gt_tif', type=str, required=True,
                                help='Path to the high-resolution ground truth label map.')
    parser_process.set_defaults(func=visualize_process)

    args = parser.parse_args()

    # --- Environment and Model Setup ---
    os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # Configure model architecture
    config_vit = CONFIGS_ViT_seg["ViT-B_16"]
    config_vit.n_classes = args.num_classes
    vit_patches_size = 16
    config_vit.patches.grid = (args.img_size // vit_patches_size, args.img_size // vit_patches_size)

    # Load model
    net = ViT_seg(config_vit, backbone=L2HNet(width=args.CNN_width), img_size=args.img_size,
                  num_classes=config_vit.n_classes).to(device)

    print(f"Loading model from: {args.model_path}")
    net.load_state_dict(torch.load(args.model_path, map_location=device))
    print("Model loaded successfully.")

    # --- Run selected visualization function ---
    args.func(args, net)