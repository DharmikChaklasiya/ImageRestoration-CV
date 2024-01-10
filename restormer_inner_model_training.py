import random
from typing import List

import torch
from torch import nn
from torch.nn import functional as F
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR
from tqdm import tqdm

from base_model_training import Phase, save_model_and_history, LossHistory, DatasetPartMetaInfo, \
    save_datasetpart_metainfo, preload_images_from_drive, load_input_image_parts
from image_loader import GroundTruthLabelDataset, ImageTensorGroup, ImageCropper
from performance_visualization import ImagePerformance, LabelAndPrediction, update_report_samples_for_epoch, \
    update_report_with_losses


def randomly_crop_images(inputs, ground_truths, img_tensor_groups: List[ImageTensorGroup]):
    new_inputs = []
    new_ground_truths = []

    for input, ground_truth, img_tensor_group in zip(inputs, ground_truths, img_tensor_groups):
        new_input, new_ground_truth = randomly_crop_image(input, ground_truth, img_tensor_group)
        # print(f"Cropped image to shape : {new_input.shape}")
        new_inputs.append(new_input)
        new_ground_truths.append(new_ground_truth)
    return torch.stack(new_inputs), torch.stack(new_ground_truths)


def randomly_crop_image(input, ground_truth, img_tensor_group: ImageTensorGroup):
    x_min, y_min, x_max, y_max = img_tensor_group.pose_prediction_labels
    cropper = ImageCropper()
    try:
        x_start, y_start, x_end, y_end = cropper.randomly_crop_image(input.shape, x_min, y_min, x_max, y_max, new_bbox_width=256, new_bbox_height=256)
    except Exception as e:
        raise ValueError("Problem with img : {} and bounding rectangle {}, {}, {}, {}".format(img_tensor_group.image_group.formatted_image_index, x_min, y_min, x_max, y_max)) from e
    new_input = input[:, y_start:y_end, x_start:x_end]
    new_ground_truth = ground_truth[:, y_start:y_end, x_start:x_end]
    return new_input, new_ground_truth


def train_model_on_one_batch(batch_part: DatasetPartMetaInfo, model: nn.Module, device, super_batch_info: str,
                             model_file_name: str):
    #batch_part.limit_dataset_size(20)

    sorted_image_groups, image_group_map = load_input_image_parts(batch_part)
    sorted_image_tensor_groups, image_tensor_group_map = preload_images_from_drive(batch_part, sorted_image_groups,
                                                                                   super_batch_info,
                                                                                   focal_stack_indices=[0, 2, 7])

    predication_and_labels_dataset = GroundTruthLabelDataset(sorted_image_tensor_groups)
    loss_history: LossHistory = batch_part.get_loss_history(model_file_name)

    train_loader, val_loader, eval_dataloader = batch_part.create_dataloaders(predication_and_labels_dataset,
                                                                              train_batch_size=1)

    num_epochs = 50

    loss_function = F.l1_loss  # ssim_based_loss  # F.mse_loss
    optimizer = AdamW(model.parameters(), lr=0.001, weight_decay=1e-4)
    lr_scheduler = CosineAnnealingLR(optimizer, T_max=num_epochs, eta_min=1e-6)
    html_file_path = 'model_run_restormer.html'

    def evaluate_rows_print_images_to_report(eval_dataloader, model, epoch):
        model.eval()
        performance = []
        with torch.no_grad():

            eval_dataloader_with_progress = tqdm(eval_dataloader, desc="Processing Evaluation")

            for inputs, ground_truth, img_group_indices in eval_dataloader_with_progress:
                outputs, loss = forward_pass_and_loss(device, ground_truth, inputs, loss_function, model)

                for gt, out, img_group_idx in zip(ground_truth, outputs, img_group_indices):
                    performance.append(ImagePerformance(loss.item(), gt, LabelAndPrediction(gt, out),
                                                        image_group_map[img_group_idx]))

        update_report_samples_for_epoch(epoch + 1, performance, html_file_path)

    for epoch in range(num_epochs):
        model.train()

        dataloader_with_progress = tqdm(train_loader, desc="Processing Batches")

        for i, batch in enumerate(dataloader_with_progress):
            inputs, ground_truth, img_group_indices = batch

            if len(img_group_indices) > 1:
                raise ValueError("We expect batchsize of 1 here for training!")

            optimizer.zero_grad()

            inputs, ground_truth = randomly_crop_images(inputs, ground_truth,
                                                        [image_tensor_group_map[img_group_indices[0]]])

            outputs, loss = forward_pass_and_loss(device, ground_truth, inputs, loss_function, model)

            loss.backward()
            optimizer.step()

            loss_history.add_current_running_loss(i, loss.item(), Phase.TRAINING)

            dataloader_with_progress.set_description(
                f"{super_batch_info}-part:{batch_part.part_name}-epoch {epoch + 1}/{num_epochs}-Batch {i + 1}/{len(train_loader)} Processing {img_group_indices}, "
                f"Loss: {loss.item():.4f}, Avg loss so far: {loss_history.current_running_loss.current_avg_train_loss:.4f}"
            )

            if i > 0 and (i % 20 == 0 or i == len(train_loader) - 1):
                update_report_with_losses(epoch + 1, loss_history, html_file_path)

        lr_scheduler.step()

        # Print average loss for the epoch
        epoch_loss = loss_history.current_running_loss.running_loss / len(train_loader)

        model.eval()

        val_dataloader_with_progress = tqdm(val_loader, desc="Processing Validation")

        with torch.no_grad():
            for i, batch in enumerate(val_dataloader_with_progress):
                inputs, ground_truth, img_group_indices = batch

                outputs, loss = forward_pass_and_loss(device, ground_truth, inputs, loss_function, model)

                loss_history.add_current_running_loss(i, loss.item(), Phase.VALIDATION)

                if i % 20 == 0 or i == len(val_loader) - 1:
                    update_report_with_losses(epoch + 1, loss_history, html_file_path)

        avg_val_loss = loss_history.current_running_loss.running_loss / len(val_loader)

        evaluate_rows_print_images_to_report(eval_dataloader, model, epoch)

        should_save = loss_history.add_loss(epoch_loss, avg_val_loss)
        if should_save:
            save_model_and_history(model, loss_history, model_file_name)
            save_datasetpart_metainfo(batch_part)
            print(f"\n\nModel saved in {super_batch_info} - epoch {epoch}")

        print(f"\n{super_batch_info}-part:{batch_part.part_name}-epoch {epoch + 1}/{num_epochs}, "
              f"Loss: {epoch_loss:.6f}, Validation Loss: {avg_val_loss:.6f}")


def forward_pass_and_loss(device, ground_truth, inputs, loss_function, model):
    inputs, ground_truth = inputs.to(device), ground_truth.to(device)
    try:
        outputs = model(inputs)
    except Exception as e:
        raise ValueError("Error while handling inputs of shape : "+str(inputs[0].shape)) from e
    loss = loss_function(outputs, ground_truth)
    return outputs, loss
