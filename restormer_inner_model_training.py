import random
from typing import List, Optional, Dict

import torch
from torch import nn
from torch.nn import functional as F
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR
from tqdm import tqdm

from base_model_training import Phase, save_model, LossHistory, DatasetPartMetaInfo, \
    save_datasetpart_metainfo, preload_images_from_drive, load_input_image_parts, AccentedLoss, ModelRunSummary
from image_loader import GroundTruthLabelDataset, ImageTensorGroup, ImageCropper
from parameter_file_parser import BoundingBox
from performance_visualization import ImagePerformance, LabelAndPrediction, update_report_samples_for_epoch, \
    update_report_with_losses, update_report_with_sample_training_images


def randomly_crop_images(inputs, ground_truths, img_tensor_groups: List[ImageTensorGroup]) -> \
        (torch.Tensor, torch.Tensor, List[BoundingBox]):
    new_inputs = []
    new_ground_truths = []
    new_bbxes = []

    for input, ground_truth, img_tensor_group in zip(inputs, ground_truths, img_tensor_groups):
        new_input, new_ground_truth, new_bbx = randomly_crop_image(input, ground_truth, img_tensor_group)
        # print(f"Cropped image to shape : {new_input.shape}")
        new_inputs.append(new_input)
        new_ground_truths.append(new_ground_truth)
        new_bbxes.append(new_bbx)
    return torch.stack(new_inputs), torch.stack(new_ground_truths), new_bbxes


def randomly_crop_image(input, ground_truth, img_tensor_group: ImageTensorGroup) -> (torch.Tensor, torch.Tensor, BoundingBox):
    orig_bbox: Optional[BoundingBox] = None

    try:
        orig_bbox = BoundingBox(*img_tensor_group.pose_prediction_labels)
        bbox, moved_orig_bbx = ImageCropper().randomly_crop_image(input.shape, orig_bbox,
                                                  new_bbox_width=256, new_bbox_height=256)
    except Exception as e:
        raise ValueError(
            "Problem with img : {} and bounding rectangle {}".format(img_tensor_group.image_group.formatted_image_index,
                                                                     orig_bbox)) from e
    new_input = input[:, bbox.y_min:bbox.y_max, bbox.x_min:bbox.x_max]
    new_ground_truth = ground_truth[:, bbox.y_min:bbox.y_max, bbox.x_min:bbox.x_max]
    return new_input, new_ground_truth, moved_orig_bbx


def train_model_on_one_batch(batch_part: DatasetPartMetaInfo, model_run_summary: ModelRunSummary, super_batch_info: str):
    batch_part.limit_validation_dataset_size(400)

    sorted_image_groups, image_group_map = load_input_image_parts(batch_part)
    image_tensor_group_map: Dict[str, ImageTensorGroup]
    sorted_image_tensor_groups, image_tensor_group_map = preload_images_from_drive(batch_part, sorted_image_groups,
                                                                                   super_batch_info,
                                                                                   focal_stack_indices=[0, 2, 7])

    predication_and_labels_dataset = GroundTruthLabelDataset(sorted_image_tensor_groups)
    loss_history: LossHistory = model_run_summary.get_loss_history()

    train_loader, val_loader, eval_dataloader = batch_part.create_dataloaders(predication_and_labels_dataset,
                                                                              train_batch_size=1)

    num_epochs = 50

    loss_function = F.l1_loss  # ssim_based_loss  # F.mse_loss
    optimizer = AdamW(model_run_summary.model.parameters(), lr=0.0001, weight_decay=1e-5)
    lr_scheduler = CosineAnnealingLR(optimizer, T_max=num_epochs, eta_min=1e-6)

    def evaluate_rows_print_images_to_report(eval_dataloader, model, epoch):
        model.eval()
        performance = []
        with torch.no_grad():

            eval_dataloader_with_progress = tqdm(eval_dataloader, desc="Processing Evaluation")

            img_group_idxes: List[str]
            for inputs, ground_truth, img_group_idxes in eval_dataloader_with_progress:
                bboxes = [BoundingBox(*image_tensor_group_map[idx].pose_prediction_labels) for idx in img_group_idxes]
                outputs, loss = forward_pass_and_loss(model_run_summary, ground_truth, inputs, loss_function,
                                                      bboxes)

                for gt, out, img_group_idx in zip(ground_truth, outputs, img_group_idxes):
                    performance.append(ImagePerformance(loss.item(), gt, LabelAndPrediction(gt, out),
                                                        image_group_map[img_group_idx]))

        update_report_samples_for_epoch(epoch + 1, performance, model_run_summary.get_html_summary_path())

    for epoch in range(num_epochs):
        model_run_summary.model.train()

        dataloader_with_progress = tqdm(train_loader, desc="Processing Batches")

        for i, batch in enumerate(dataloader_with_progress):
            inputs, ground_truths, img_group_indices = batch

            optimizer.zero_grad()

            inputs, ground_truths, newbbxes = randomly_crop_images(inputs, ground_truths,
                                                         [image_tensor_group_map[index] for index in img_group_indices])

            outputs, loss = forward_pass_and_loss(model_run_summary, ground_truths, inputs, loss_function, newbbxes)

            loss.backward()
            optimizer.step()

            loss_history.add_current_running_loss(i, loss.item(), Phase.TRAINING)

            dataloader_with_progress.set_description(
                f"{super_batch_info}-part:{batch_part.part_name}-epoch {epoch + 1}/{num_epochs}-Batch {i + 1}/{len(train_loader)} Processing {img_group_indices}, "
                f"Loss: {loss.item():.4f}, Avg loss so far: {loss_history.current_running_loss.current_avg_train_loss:.4f}"
            )

            if i > 0 and (i % 20 == 0 or i == len(train_loader) - 1):
                update_report_with_losses(epoch + 1, loss_history, model_run_summary.get_html_summary_path())
                update_report_with_sample_training_images(epoch + 1, inputs, ground_truths, newbbxes,
                                                          [image_tensor_group_map[index] for index in
                                                           img_group_indices], model_run_summary.get_html_summary_path())

        lr_scheduler.step()

        # Print average loss for the epoch
        epoch_loss = loss_history.current_running_loss.running_loss / len(train_loader)

        model_run_summary.model.eval()

        val_dataloader_with_progress = tqdm(val_loader, desc="Processing Validation")

        with torch.no_grad():
            for i, batch in enumerate(val_dataloader_with_progress):
                inputs, ground_truths, img_group_indices = batch

                bboxes = [BoundingBox(*image_tensor_group_map[index].pose_prediction_labels)
                          for index in img_group_indices]

                outputs, loss = forward_pass_and_loss(model_run_summary, ground_truths, inputs, loss_function, bboxes)

                loss_history.add_current_running_loss(i, loss.item(), Phase.VALIDATION)

                if i % 20 == 0 or i == len(val_loader) - 1:
                    update_report_with_losses(epoch + 1, loss_history, model_run_summary.get_html_summary_path())

        avg_val_loss = loss_history.current_running_loss.running_loss / len(val_loader)

        evaluate_rows_print_images_to_report(eval_dataloader, model_run_summary.model, epoch)

        model_run_summary.update(loss_history)

        should_save = loss_history.add_loss(epoch_loss, avg_val_loss)
        if should_save:
            save_model(model_run_summary)
            print(f"\n\nModel saved in {super_batch_info} - epoch {epoch + 1}")

        print(f"\n{super_batch_info}-part:{batch_part.part_name}-epoch {epoch + 1}/{num_epochs}, "
              f"Loss: {epoch_loss:.6f}, Validation Loss: {avg_val_loss:.6f}")


def forward_pass_and_loss(model_run_summary: ModelRunSummary, ground_truth, inputs, loss_function, bboxes: List[BoundingBox]):
    inputs, ground_truth = inputs.to(model_run_summary.device), ground_truth.to(model_run_summary.device)
    try:
        outputs = model_run_summary.model(inputs)
    except Exception as e:
        raise ValueError("Error while handling inputs of shape : " + str(inputs[0].shape)) from e

    # right now, I don't see the need for an accented loss here at all - the restormer handles this very nicely
    # combined_loss = AccentedLoss(loss_function).calculate_combined_loss(bboxes, outputs, ground_truth)

    loss = loss_function(outputs, ground_truth)
    return outputs, loss
