from typing import Dict

import torch
from torch import optim, nn
from torch.nn import functional as F
from tqdm import tqdm

from base_model_training import Phase, save_model_and_history, LossHistory, DatasetPartMetaInfo, \
    save_datasetpart_metainfo
from image_loader import load_input_image_parts, ImageTensorGroup, GroundTruthLabelDataset
from performance_visualization import ImagePerformance, LabelAndPrediction, update_report_samples_for_epoch, \
    update_report_with_losses


def train_model_on_one_batch(batch_part: DatasetPartMetaInfo, model: nn.Module, device, super_batch_info: str, model_file_name: str):
    sorted_image_groups, image_group_map = load_input_image_parts([batch_part.part_name])
    sorted_image_tensor_groups = []
    image_tensor_group_map: Dict[str, ImageTensorGroup] = {}
    focal_stack_indices = [0, 1, 2, 3, 7, 10, 15, 20, 25, 30]

    for img_group in tqdm(sorted_image_groups, desc="Preloading images for batches: {}".format(batch_part.part_name)):
        image_tensor_group = ImageTensorGroup(img_group, focal_stack_indices)
        image_tensor_group.load_images()
        image_tensor_group_map[img_group.formatted_image_index] = image_tensor_group
        sorted_image_tensor_groups.append(image_tensor_group)

    pose_prediction_label_dataset = GroundTruthLabelDataset(sorted_image_tensor_groups)
    loss_history: LossHistory = batch_part.get_loss_history(model_file_name)

    train_loader, val_loader, eval_dataloader = batch_part.create_dataloaders(pose_prediction_label_dataset)

    loss_function = F.l1_loss  # ssim_based_loss  # F.mse_loss
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    # Number of training epochs
    num_epochs = 50
    html_file_path = 'model_run_unet.html'

    def evaluate_rows_print_images_to_report(eval_dataloader, model, epoch):
        model.eval()
        performance = []
        with torch.no_grad():

            eval_dataloader_with_progress = tqdm(eval_dataloader, desc="Processing Evaluation")

            for inputs, ground_truth, img_group_indices in eval_dataloader_with_progress:
                inputs, ground_truth = inputs.to(device), ground_truth.to(device)
                outputs = model(inputs)

                for gt, out in zip(ground_truth, outputs):
                    loss = loss_function(outputs, ground_truth)
                    performance.append(ImagePerformance(loss.item(), gt, LabelAndPrediction(gt, out),
                                                        image_group_map[img_group_indices[0]]))

        update_report_samples_for_epoch(epoch + 1, performance, html_file_path)

    for epoch in range(num_epochs):
        model.train()

        dataloader_with_progress = tqdm(train_loader, desc="Processing Batches")

        for i, batch in enumerate(dataloader_with_progress):
            inputs, ground_truth, img_group = batch

            inputs, ground_truth = inputs.to(device), ground_truth.to(device)

            optimizer.zero_grad()

            outputs = model(inputs)

            loss = loss_function(outputs, ground_truth)

            loss.backward()
            optimizer.step()

            loss_history.add_current_running_loss(i, loss.item(), Phase.TRAINING)

            dataloader_with_progress.set_description(
                f"{super_batch_info}-part:{batch_part.part_name}-epoch {epoch + 1}/{num_epochs}-Batch {i + 1}/{len(train_loader)} Processing {img_group}, "
                f"Loss: {loss.item():.4f}, Avg loss so far: {loss_history.current_running_loss.current_avg_train_loss:.4f}"
            )

            if i % 20 == 0 or i == len(train_loader) - 1:
                update_report_with_losses(epoch + 1, loss_history, html_file_path)

        # Print average loss for the epoch
        epoch_loss = loss_history.current_running_loss.running_loss / len(train_loader)

        model.eval()

        val_dataloader_with_progress = tqdm(val_loader, desc="Processing Validation")

        with torch.no_grad():
            for i, batch in enumerate(val_dataloader_with_progress):
                inputs, ground_truth, img_group = batch

                inputs, ground_truth = inputs.to(device), ground_truth.to(device)

                outputs = model(inputs)

                loss = loss_function(outputs, ground_truth)
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
