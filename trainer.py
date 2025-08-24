import logging
import os
import random
import sys
import numpy as np
import torch
import torch.optim as optim
from sklearn.metrics import confusion_matrix
from tensorboardX import SummaryWriter
from torch.nn.modules.loss import CrossEntropyLoss
from torch.utils.data import DataLoader
from tqdm import tqdm
import pandas as pd
import torch.nn.functional as F
import utils
import rasterio
from rasterio.windows import Window
from rasterio.errors import RasterioError
from torch.utils.data.dataset import IterableDataset
import csv

device = torch.device("cuda" if torch.cuda.is_available() else "cpu") # Hopefully, noone will use CPU for training
print("Using device:", device)

class StreamingGeospatialDataset(IterableDataset):
    
    def __init__(self, imagery_fns, label_fns=None, groups=None, chip_size=256, num_chips_per_tile=200, windowed_sampling=False, image_transform=None, label_transform=None, nodata_check=None, verbose=False):
        if label_fns is None:
            self.fns = imagery_fns
            self.use_labels = False
        else:
            self.fns = list(zip(imagery_fns, label_fns)) 
            
            self.use_labels = True

        self.groups = groups

        self.chip_size = chip_size
        self.num_chips_per_tile = num_chips_per_tile
        self.windowed_sampling = windowed_sampling

        self.image_transform = image_transform
        self.label_transform = label_transform
        self.nodata_check = nodata_check

        self.verbose = verbose

        if self.verbose:
            print("Constructed StreamingGeospatialDataset")

    def stream_tile_fns(self):
        worker_info = torch.utils.data.get_worker_info()
        if worker_info is None: # In this case we are not loading through a DataLoader with multiple workers
            worker_id = 0
            num_workers = 1
        else:
            worker_id = worker_info.id
            num_workers = worker_info.num_workers
        # We only want to shuffle the order we traverse the files if we are the first worker (else, every worker will shuffle the files...)
        if worker_id == 0:
            np.random.shuffle(self.fns) # in place

        if self.verbose:
            print("Creating a filename stream for worker %d" % (worker_id))

        # This logic splits up the list of filenames into `num_workers` chunks. Each worker will recieve ceil(num_filenames / num_workers) filenames to generate chips from. If the number of workers doesn't divide the number of filenames evenly then the last worker will have fewer filenames.
        N = len(self.fns)
        num_files_per_worker = int(np.ceil(N / num_workers))
        lower_idx = worker_id * num_files_per_worker
        upper_idx = min(N, (worker_id+1) * num_files_per_worker)
        for idx in range(lower_idx, upper_idx):

            label_fn = None
            if self.use_labels:
                
                img_fn, label_fn = self.fns[idx]
            else:
                img_fn = self.fns[idx]

            if self.groups is not None:
                group = self.groups[idx]
            else:
                group = None

            if self.verbose:
                print("Worker %d, yielding file %d" % (worker_id, idx))

            yield (img_fn, label_fn, group)

    def stream_chips(self):
        for img_fn, label_fn, group in self.stream_tile_fns():
            num_skipped_chips = 0

            # Open file pointers
            img_fp = rasterio.open(img_fn, "r")
            label_fp = rasterio.open(label_fn, "r") if self.use_labels else None
            
            height, width = img_fp.shape
            if self.use_labels: # garuntee that our label mask has the same dimensions as our imagery
                t_height, t_width = label_fp.shape
                assert height == t_height and width == t_width

            # If we aren't in windowed sampling mode then we should read the entire tile up front
            img_data = None
            label_data = None
            try:
                if not self.windowed_sampling:
                    img_data = np.rollaxis(img_fp.read(3), 0, 3)
                    if self.use_labels:
                        label_data = label_fp.read().squeeze() # assume the label geotiff has a single channel
            except RasterioError as e:
                print("WARNING: Error reading in entire file, skipping to the next file")
                continue

            for i in range(self.num_chips_per_tile):
                # Select the top left pixel of our chip randomly 
                x = np.random.randint(0, width-self.chip_size)
                y = np.random.randint(0, height-self.chip_size)

                # Read imagery / labels
                img = None
                labels = None
                if self.windowed_sampling:
                    try:
                        img = np.rollaxis(img_fp.read(window=Window(x, y, self.chip_size, self.chip_size)), 0, 3)
                        # print(img.shape)
                        if self.use_labels:
                            labels = label_fp.read(window=Window(x, y, self.chip_size, self.chip_size)).squeeze()
                    except RasterioError:
                        print("WARNING: Error reading chip from file, skipping to the next chip")
                        continue
                else:
                    img = img_data[y:y+self.chip_size, x:x+self.chip_size, :]
                    if self.use_labels:
                        labels = label_data[y:y+self.chip_size, x:x+self.chip_size]

                # Check for no data
                if self.nodata_check is not None:
                    if self.use_labels:
                        skip_chip = self.nodata_check(img, labels)
                    else:
                        skip_chip = self.nodata_check(img)

                    if skip_chip: # The current chip has been identified as invalid by the `nodata_check(...)` method
                        num_skipped_chips += 1
                        continue

                # Transform the imagery
                if self.image_transform is not None:
                    if self.groups is None:
                        img = self.image_transform(img)
                    else:
                        img = self.image_transform(img, group)
                else:
                    img = torch.from_numpy(img).squeeze()

                # Transform the labels
                if self.use_labels:
                    if self.label_transform is not None:
                        if self.groups is None:
                            
                            labels = self.label_transform(labels)
                        else:
                            print(label_fn)
                            labels = self.label_transform(labels, group)
                            print(labels)
                    else:
                        labels = torch.from_numpy(labels).squeeze()

                # Note, that img should be a torch "Double" type (i.e. a np.float32) and labels should be a torch "Long" type (i.e. np.int64)
                if self.use_labels:
                     yield img, labels
                else:
                     yield img
            # Close file pointers
            img_fp.close()
            if self.use_labels:
                label_fp.close()

            if num_skipped_chips>0 and self.verbose:
                print("We skipped %d chips on %s" % (img_fn))

    def __iter__(self):
        if self.verbose:
            print("Creating a new StreamingGeospatialDataset iterator")
        return iter(self.stream_chips())

def image_transforms(img):
    img = (img - utils.IMAGE_MEANS) / utils.IMAGE_STDS
    img = np.rollaxis(img, 2, 0).astype(np.float32)
    img = torch.from_numpy(img)
    return img

def label_transforms(labels):
    labels = utils.LABEL_CLASS_TO_IDX_MAP[labels]
    labels = torch.from_numpy(labels)
    return labels

# Map GT to mIoU classes for evaluation
def label_transforms_val(labels):
    map_func = np.vectorize(utils.GT_TO_EVAL_MAP.get)
    labels = map_func(labels)
    labels = torch.from_numpy(labels)
    return labels

def nodata_check(img, labels):
    return np.any(labels == 0) or np.any(np.sum(img == 0, axis=2) == 4)

def validate_and_get_miou(model, valloader, num_eval_classes, device):
    # Map training indices to evaluation classes
    map_fn = np.vectorize(utils.IDX_TO_EVAL_MAP.get)
    model.eval()
    conf_matrix = np.zeros((num_eval_classes, num_eval_classes), dtype=np.int64)

    with torch.no_grad():
        for image_batch, label_batch in tqdm(valloader, desc="Validation"):
            image_batch = image_batch.to(device)

            # Get fused prediction from both branches for robust evaluation
            outputs1, outputs2 = model(image_batch)
            outputs = F.softmax((outputs1 + outputs2) / 2, dim=1)
            preds = torch.argmax(outputs, dim=1).cpu().numpy()

            gt = label_batch.cpu().numpy()

            # Remap predictions and ground truth to evaluation classes
            preds_eval = map_fn(preds)
            gt_eval = map_fn(gt)

            # Calculate mIoU, ignoring the 'Nothing' class (index 4)
            # The classes to include in the mean are [0, 1, 2, 3]
            valid_pixels = (gt_eval != 4)
            gt_valid = gt_eval[valid_pixels]
            preds_valid = preds_eval[valid_pixels]

            # Add to confusion matrix
            conf_matrix += confusion_matrix(gt_valid, preds_valid, labels=np.arange(num_eval_classes))

    # Calculate IoU from the confusion matrix
    intersection = np.diag(conf_matrix)
    ground_truth_set = np.sum(conf_matrix, axis=1)
    predicted_set = np.sum(conf_matrix, axis=0)
    union = ground_truth_set + predicted_set - intersection

    iou = np.nan_to_num(intersection / union)

    mean_iou = np.mean(iou)

    model.train()
    return mean_iou, iou


def trainer_dataset(args, model, snapshot_path):
    logging.basicConfig(filename=snapshot_path + "/log.txt", level=logging.INFO,
                        format='[%(asctime)s.%(msecs)03d] %(message)s', datefmt='%H:%M:%S')
    logging.getLogger().addHandler(logging.StreamHandler(sys.stdout))
    logging.info(str(args))
    base_lr = args.base_lr
    batch_size = args.batch_size

    # --- Load input data and split into Train/Validation ---
    input_dataframe = pd.read_csv(args.list_dir)
    logging.info(f"Loaded {len(input_dataframe)} total samples from {args.list_dir}")

    val_candidates_df = input_dataframe[
        input_dataframe['gt_label_fn'].notna() & (input_dataframe['gt_label_fn'] != '')].copy()
    logging.info(f"Found {len(val_candidates_df)} samples with ground truth labels, eligible for validation.")

    np.random.seed(args.seed)
    # Either 20 % of the total samples or all val_candidates
    validation_split_size = min(len(val_candidates_df), int(len(input_dataframe) * 0.2))
    val_indices = np.random.choice(val_candidates_df.index, size=validation_split_size, replace=False)
    val_df = val_candidates_df.loc[val_indices]

    train_df = input_dataframe.copy()

    logging.info(f"Using {len(train_df)} tiles for training.")
    logging.info(f"Using {len(val_df)} tiles for validation.")

    # Get file paths for the training set (uses LR labels)
    train_image_fns = train_df["image_fn"].values
    train_label_fns = train_df["label_fn"].values

    # Get file paths for the validation set (uses HR GT labels)
    val_image_fns = val_df["image_fn"].values
    val_label_fns = val_df["gt_label_fn"].values

    NUM_TRAIN_CHIPS_PER_TILE = 50
    NUM_VAL_CHIPS_PER_TILE = 50
    CHIP_SIZE = 224
    db_train = StreamingGeospatialDataset(
        imagery_fns=train_image_fns, label_fns=train_label_fns, groups=None, chip_size=CHIP_SIZE,
        num_chips_per_tile=NUM_TRAIN_CHIPS_PER_TILE, windowed_sampling=True, image_transform=image_transforms,
        label_transform=label_transforms, nodata_check=nodata_check
    )
    db_val = StreamingGeospatialDataset(
        imagery_fns=val_image_fns, label_fns=val_label_fns, groups=None, chip_size=CHIP_SIZE,
        num_chips_per_tile=NUM_VAL_CHIPS_PER_TILE, windowed_sampling=True, image_transform=image_transforms,
        label_transform=label_transforms_val, nodata_check=nodata_check
    )

    def worker_init_fn(worker_id):
        random.seed(args.seed + worker_id)

    trainloader = DataLoader(db_train, batch_size=batch_size, num_workers=8, pin_memory=True,
                             worker_init_fn=worker_init_fn)
    valloader = DataLoader(db_val, batch_size=batch_size, num_workers=8, pin_memory=True, worker_init_fn=worker_init_fn)

    model = model.to(device)

    model.train()
    ce_loss = CrossEntropyLoss(ignore_index=0)
    optimizer = optim.SGD(model.parameters(), lr=base_lr, momentum=0.9, weight_decay=0.0001)
    writer = SummaryWriter(snapshot_path + '/log')
    iter_num = 0
    max_epoch = args.max_epochs
    num_batches_per_epoch = int(len(train_image_fns) * NUM_TRAIN_CHIPS_PER_TILE / batch_size)
    max_iterations = max_epoch * len(train_image_fns) * NUM_TRAIN_CHIPS_PER_TILE
    logging.info(f"{num_batches_per_epoch} training batches per epoch.")
    logging.info(f"{max_iterations} max iterations for LR scheduler.")
    best_miou = -1.0

    log_csv_path = os.path.join(snapshot_path, 'training_log.csv')
    csv_headers = [
        'epoch', 'avg_loss1', 'avg_loss2', 'total_avg_loss', 'mIoU'
    ]
    class_names = list(utils.CLASS_COLORMAP_MIOU.keys())
    for class_name in class_names:
        csv_headers.append(f'iou_class_{class_name}')

    with open(log_csv_path, 'w', newline='') as f:
        writer_csv = csv.writer(f)
        writer_csv.writerow(csv_headers)
    logging.info(f"CSV log created at: {log_csv_path}")

    iterator = range(max_epoch)
    for epoch_num in iterator:
        loss1 = []
        loss2 = []
        for i_batch, (image_batch, label_batch) in tqdm(enumerate(trainloader), total=num_batches_per_epoch):
            image_batch, label_batch = image_batch.to(device), label_batch.to(device)
            outputs1, outputs2 = model(image_batch)
            t_output = F.softmax((outputs1), dim=1)  # Created mask label
            t_output = t_output.argmax(axis=1)
            mask_output = torch.where(t_output == label_batch, label_batch, 0)
            loss_ce1 = ce_loss(outputs1, label_batch[:].long())  # General CE loss for CNN branch
            loss_ce2 = ce_loss(outputs2, mask_output[:].long())  # Mask CE (mce) loss for ViT branch
            loss = 0.5 * loss_ce1 + 0.5 * loss_ce2

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            lr_ = base_lr * (1.0 - iter_num / max_iterations) ** 0.9
            for param_group in optimizer.param_groups:
                param_group['lr'] = lr_
            loss1.append(loss_ce1.item())
            loss2.append(loss_ce2.item())
            iter_num += 1

        avg_loss1 = np.mean(loss1)
        avg_loss2 = np.mean(loss2)
        total_avg_loss = avg_loss1 * 0.5 + avg_loss2 * 0.5
        logging.info(
            f'Epoch {epoch_num + 1} Training --- Loss1: {avg_loss1:.4f}, Loss2: {avg_loss2:.4f}, Total: {total_avg_loss:.4f}')
        writer.add_scalar('train/loss', total_avg_loss, epoch_num)

        # Validation Phase
        val_miou, val_iou_per_class = validate_and_get_miou(model, valloader, utils.NUM_EVAL_CLASSES, device)

        logging.info(f'Epoch {epoch_num + 1} Validation --- mIoU: {val_miou:.4f}')
        iou_log_str = ", ".join([f"Class-{c}: {iou:.4f}" for c, iou in zip(class_names, val_iou_per_class)])
        logging.info(f"IoU per class: {iou_log_str}")

        writer.add_scalar('val/mIoU', val_miou, epoch_num)
        for c, iou in zip(class_names, val_iou_per_class):
            writer.add_scalar(f'val/IoU_class_{c}', iou, epoch_num)

        iou_list = val_iou_per_class.tolist() if isinstance(val_iou_per_class, np.ndarray) else val_iou_per_class

        data_row = [
                       epoch_num + 1,
                       avg_loss1,
                       avg_loss2,
                       total_avg_loss,
                       val_miou
                   ] + iou_list

        with open(log_csv_path, 'a', newline='') as f:
            writer_csv = csv.writer(f)
            writer_csv.writerow(data_row)

        if val_miou > best_miou:
            logging.info(f"🚀 Validation mIoU improved from {best_miou:.4f} to {val_miou:.4f}!")
            best_miou = val_miou

            save_filename = f'model_epoch_{epoch_num + 1}_miou_{val_miou:.4f}.pth'
            save_mode_path = os.path.join(snapshot_path, save_filename)

            torch.save(model.state_dict(), save_mode_path)
            logging.info(f"   New best model saved to: {save_mode_path}")

        if (epoch_num + 1) % args.save_interval == 0:
            save_mode_path = os.path.join(snapshot_path, 'epoch_' + str(epoch_num) + '.pth')
            torch.save(model.state_dict(), save_mode_path)
            logging.info("save model to {}".format(save_mode_path))

        if epoch_num >= max_epoch - 1:
            save_mode_path = os.path.join(snapshot_path, 'epoch_' + str(epoch_num) + '.pth')
            torch.save(model.state_dict(), save_mode_path)
            logging.info("save model to {}".format(save_mode_path))
        else:
            logging.info(
                f"Validation mIoU ({val_miou:.4f}) did not improve from best ({best_miou:.4f}). Not saving model.")

    writer.close()
    save_mode_path = os.path.join(snapshot_path, 'latest_model.pth')
    torch.save(model.state_dict(), save_mode_path)
    logging.info(f"Training finished. Final model saved to {save_mode_path}")
    logging.info(f"Training statistics saved to {log_csv_path}")
    return "Training Finished!"
