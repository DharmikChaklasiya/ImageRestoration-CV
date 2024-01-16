import torch
from torch import optim, nn
from torch.nn import functional as F
from tqdm import tqdm

from base_model_training import Phase, save_model, LossHistory, DatasetPartMetaInfo, \
    save_datasetpart_metainfo, preload_images_from_drive, load_input_image_parts, AccentedLoss
from image_loader import GroundTruthLabelDataset
from parameter_file_parser import BoundingBox
from performance_visualization import ImagePerformance, LabelAndPrediction, update_report_samples_for_epoch, \
    update_report_with_losses


def train_model_on_one_batch(batch_part: DatasetPartMetaInfo, model: nn.Module, posepred_model: nn.Module, device,
                             super_batch_info: str, model_file_name: str):
    sorted_image_groups, image_group_map = load_input_image_parts(batch_part)

    sorted_image_tensor_groups, image_tensor_group_map = preload_images_from_drive(batch_part, sorted_image_groups,
                                                                                   super_batch_info)

    prediction_and_labels_dataset = GroundTruthLabelDataset(sorted_image_tensor_groups)
    loss_history: LossHistory = batch_part.get_loss_history(model_file_name)

    train_loader, val_loader, eval_dataloader = batch_part.create_dataloaders(prediction_and_labels_dataset)

    loss_function = F.l1_loss  # ssim_based_loss  # F.mse_loss
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    # Number of training epochs
    num_epochs = 50
    html_file_path = 'model_run_unet_with_posepred.html'

    def evaluate_rows_print_images_to_report(eval_dataloader, model, epoch):
        model.eval()
        performance = []
        with torch.no_grad():

            eval_dataloader_with_progress = tqdm(eval_dataloader, desc="Processing Evaluation")

            for inputs, ground_truth, img_group_indices in eval_dataloader_with_progress:
                inputs, ground_truth = inputs.to(device), ground_truth.to(device)

                outputs, loss = forward_call_and_loss_calc(ground_truth, inputs, loss_function, model, posepred_model)

                for gt, out, img_group_index in zip(ground_truth, outputs, img_group_indices):
                    performance.append(ImagePerformance(loss.item(), gt, LabelAndPrediction(gt, out),
                                                        image_group_map[img_group_index]))

        update_report_samples_for_epoch(epoch + 1, performance, html_file_path)

    for epoch in range(num_epochs):
        model.train()

        dataloader_with_progress = tqdm(train_loader, desc="Processing Batches")

        for i, batch in enumerate(dataloader_with_progress):
            inputs, ground_truth, img_group = batch

            inputs, ground_truth = inputs.to(device), ground_truth.to(device)

            optimizer.zero_grad()

            outputs, loss = forward_call_and_loss_calc(ground_truth, inputs, loss_function, model, posepred_model)

            loss.backward()
            optimizer.step()

            loss_history.add_current_running_loss(i, loss.item(), Phase.TRAINING)

            dataloader_with_progress.set_description(
                f"{super_batch_info}-part:{batch_part.part_name}-epoch {epoch + 1}/{num_epochs}-Batch {i + 1}/{len(train_loader)} Processing {img_group}, "
                f"Loss: {loss.item():.4f}, Avg loss so far: {loss_history.current_running_loss.current_avg_train_loss:.4f}"
            )

            if i > 0 and (i % 20 == 0 or i == len(train_loader) - 1):
                update_report_with_losses(epoch + 1, loss_history, html_file_path)

        epoch_loss = loss_history.current_running_loss.running_loss / len(train_loader)

        model.eval()

        val_dataloader_with_progress = tqdm(val_loader, desc="Processing Validation")

        with torch.no_grad():
            for i, batch in enumerate(val_dataloader_with_progress):
                inputs, ground_truth, img_group = batch

                inputs, ground_truth = inputs.to(device), ground_truth.to(device)

                outputs, loss = forward_call_and_loss_calc(ground_truth, inputs, loss_function, model, posepred_model)

                loss_history.add_current_running_loss(i, loss.item(), Phase.VALIDATION)

                if i % 20 == 0 or i == len(val_loader) - 1:
                    update_report_with_losses(epoch + 1, loss_history, html_file_path)

        avg_val_loss = loss_history.current_running_loss.running_loss / len(val_loader)

        evaluate_rows_print_images_to_report(eval_dataloader, model, epoch)

        should_save = loss_history.add_loss(epoch_loss, avg_val_loss)
        if should_save:
            save_model(model, loss_history, model_file_name)
            save_datasetpart_metainfo(batch_part)
            print(f"\n\nModel saved in {super_batch_info} - epoch {epoch}")

        print(f"\n{super_batch_info}-part:{batch_part.part_name}-epoch {epoch + 1}/{num_epochs}, "
              f"Loss: {epoch_loss:.6f}, Validation Loss: {avg_val_loss:.6f}")


def forward_call_and_loss_calc(ground_truth, inputs, loss_function, model, posepred_model):
    posepred_model.eval()
    pose_outputs = posepred_model(inputs)

    outputs = model(inputs)

    combined_loss = AccentedLoss(loss_function).calculate_combined_loss(
        [BoundingBox(*pose_output) for pose_output in pose_outputs], outputs, ground_truth)

    return outputs, combined_loss
