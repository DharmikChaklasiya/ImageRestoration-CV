import random
from typing import Dict

import torch
from torch import optim, nn
from torch.utils.data import random_split, DataLoader, Subset
from tqdm import tqdm

from base_model_training import Phase, LossHistory, DatasetPartMetaInfo
from image_loader import load_input_image_parts, ImageTensorGroup, PosePredictionLabelDataset
from performance_visualization import ImagePerformance, LabelAndPrediction, \
    update_report_samples_for_epoch, update_report_with_losses


def train_model_on_one_batch(batch_part: DatasetPartMetaInfo, model: nn.Module, device):
    sorted_image_groups, image_group_map = load_input_image_parts([batch_part.part_name])
    sorted_image_tensor_groups = []
    image_tensor_group_map: Dict[str, ImageTensorGroup] = {}
    focal_stack_indices = [0, 1, 2, 3, 7, 10, 15, 20, 25, 30]

    for img_group in tqdm(sorted_image_groups, desc="Preloading images for batches: {}".format(batch_part.part_name)):
        image_tensor_group = ImageTensorGroup(img_group, focal_stack_indices)
        image_tensor_group.load_images()
        image_tensor_group_map[img_group.formatted_image_index] = image_tensor_group
        sorted_image_tensor_groups.append(image_tensor_group)

    pose_prediction_label_dataset = PosePredictionLabelDataset(sorted_image_tensor_groups)
    loss_history = batch_part.loss_history

    train_loader, val_loader, eval_dataloader = batch_part.create_dataloaders(pose_prediction_label_dataset)

    loss_function = torch.nn.MSELoss()

    optimizer = optim.Adam(model.parameters(), lr=0.00001)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=100, gamma=0.1)
    # Number of training epochs
    num_epochs = 50
    html_file_path = 'model_run_pose_pred.html'

    # If need be, we can call this to get min/max values for x and y: image_loader.normalize_x_and_y_labels()
    def evaluate_rows_print_images_to_report(evaluated_model: nn.Module, device, eval_dataloader):
        evaluated_model.eval()
        performance = []
        with torch.no_grad():

            eval_dataloader_with_progress = tqdm(eval_dataloader, desc="Processing Evaluation")

            for eval_inputs, pose_prediction_labels_eval, img_group_indices in eval_dataloader_with_progress:
                eval_inputs, pose_prediction_labels_eval = eval_inputs.to(device), pose_prediction_labels_eval.to(
                    device)
                eval_outputs = evaluated_model(eval_inputs)

                for pose_pred_label, out in zip(pose_prediction_labels_eval, eval_outputs):
                    eval_loss = loss_function(out, pose_pred_label)
                    first_img_group_index = img_group_indices[0]
                    gt = image_tensor_group_map[first_img_group_index].ground_truth_tensor
                    performance.append(
                        ImagePerformance(eval_loss.item(), gt,
                                         LabelAndPrediction(pose_pred_label, out),
                                         image_group_map[first_img_group_index]))

        update_report_samples_for_epoch(epoch + 1, performance, html_file_path)

    for epoch in range(num_epochs):
        model.train()

        dataloader_with_progress = tqdm(train_loader, desc="Processing Batches")

        for i, batch in enumerate(dataloader_with_progress):
            inputs, pose_prediction_labels, img_group_index = batch

            inputs, pose_prediction_labels = inputs.to(device), pose_prediction_labels.to(device)

            optimizer.zero_grad()

            outputs = model(inputs)

            loss = loss_function(outputs, pose_prediction_labels)

            try:
                loss.backward()
            except RuntimeError as e:
                print(f"Error in epoch {epoch}, batch {i}")
                print(f"Inputs type: {inputs.dtype}, Labels type: {pose_prediction_labels.dtype}")
                print(f"Model outputs type: {outputs.dtype}, Loss value type: {loss.dtype}")
                raise e

            optimizer.step()

            loss_history.add_current_running_loss(i, loss.item(), Phase.TRAINING)
            current_loss = loss.item()

            dataloader_with_progress.set_description(
                f"Epoch {epoch + 1}/{num_epochs} - Batch {i + 1}/{len(train_loader)} "
                f"Processing {img_group_index}, Loss: {current_loss:.4f}, "
                f"Avg loss so far: {loss_history.current_avg_train_loss:.4f}"
            )

            if i % 20 == 0 or i == len(train_loader) - 1:
                update_report_with_losses(epoch + 1, loss_history, html_file_path)

        epoch_loss = loss_history.running_loss / len(train_loader)

        model.eval()

        val_dataloader_with_progress = tqdm(val_loader, desc="Processing Validation")

        with torch.no_grad():
            for i, batch in enumerate(val_dataloader_with_progress):
                inputs, pose_prediction_labels, img_group_index = batch

                inputs, pose_prediction_labels = inputs.to(device), pose_prediction_labels.to(device)

                outputs = model(inputs)

                loss = loss_function(outputs, pose_prediction_labels)
                loss_history.add_current_running_loss(i, loss.item(), Phase.VALIDATION)

                if i % 20 == 0 or i == len(val_loader) - 1:
                    update_report_with_losses(epoch + 1, loss_history, html_file_path)

        avg_val_loss = (loss_history.running_loss / len(val_loader))

        loss_history.add_loss(epoch_loss, avg_val_loss)

        scheduler.step()

        evaluate_rows_print_images_to_report(model, device, eval_dataloader)

        print(f"\nEpoch {epoch + 1}/{num_epochs}, Loss: {epoch_loss:.4f}, Validation Loss: {avg_val_loss:.4f}")
