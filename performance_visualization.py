import base64
import os
import webbrowser
from io import BytesIO, StringIO
from typing import List

import numpy as np
import torch
from bs4 import BeautifulSoup
from matplotlib import pyplot as plt
from torchvision.transforms import ToPILImage

from LFR.python.image_group import ImageGroup
from base_model_training import Phase


class LossHistory:
    def __init__(self):
        self.running_loss = 0.0
        self.current_avg_val_loss = None
        self.current_avg_train_loss = None
        self.training_losses = []
        self.validation_losses = []

    def add_loss(self, training_loss, validation_loss):
        self.training_losses.append(training_loss)
        self.validation_losses.append(validation_loss)
        self.running_loss = 0.0
        self.current_avg_val_loss = None
        self.current_avg_train_loss = None

    def get_loss_plot_base64(self):
        plt.figure()
        epochs = np.arange(1, len(self.training_losses) + 1)
        plt.plot(epochs, self.training_losses, label='Training Loss')
        plt.plot(epochs, self.validation_losses, label='Validation Loss')

        current_epoch = len(epochs) + 1

        # Plot current average losses if available
        if self.current_avg_train_loss is not None:
            plt.scatter(current_epoch, self.current_avg_train_loss, color='blue', marker='x',
                        label='Current Avg Training Loss')

        if self.current_avg_val_loss is not None:
            plt.scatter(current_epoch, self.current_avg_val_loss, color='orange', marker='x',
                        label='Current Avg Validation Loss')

        plt.xlabel('Epoch')
        plt.ylabel('Loss')
        plt.title('Loss Over Epochs')
        plt.legend()

        max_loss = 0
        if self.training_losses and self.validation_losses:
            max_loss = max(max(self.training_losses), max(self.validation_losses))
        if self.current_avg_val_loss:
            max_loss = max(max_loss, self.current_avg_val_loss)
        if self.current_avg_train_loss:
            max_loss = max(max_loss, self.current_avg_train_loss)
        plt.ylim(bottom=0, top=max_loss + 0.1 * max_loss)

        plt.xticks(epochs)
        plt.tight_layout()

        buffer = BytesIO()
        plt.savefig(buffer, format='png')
        plt.close()  # Close the figure to free memory
        buffer.seek(0)
        image_png = buffer.getvalue()
        buffer.close()

        return base64.b64encode(image_png).decode('utf-8')

    def add_current_running_loss(self, batch_num, running_loss, phase):
        if phase == Phase.TRAINING:
            if batch_num == 0:
                self.running_loss = 0.0
            self.running_loss += running_loss
            self.current_avg_train_loss = self.running_loss / (batch_num + 1)
        elif phase == Phase.VALIDATION:
            if batch_num == 0:
                self.running_loss = 0.0
            self.running_loss += running_loss
            self.current_avg_val_loss = self.running_loss / (batch_num + 1)
        else:
            raise ValueError("Unknown phase : " + phase)


def tensor_to_base64(tensor):
    """Convert a PyTorch tensor to a Base64-encoded image."""
    pil_image = ToPILImage()(tensor.cpu())  # Convert tensor to PIL image
    buffered = BytesIO()
    pil_image.save(buffered, format="PNG")  # Save image to a buffer
    return base64.b64encode(buffered.getvalue()).decode("utf-8")  # Encode to base64


class LabelAndPrediction:
    def __init__(self, label: torch.Tensor, prediction: torch.Tensor):
        self.label: torch.Tensor = label
        self.prediction: torch.Tensor = prediction

    def is_coordinate_prediction(self):
        # Check if label_and_prediction is a 1D tensor (x prediction)
        return isinstance(self.label, torch.Tensor) and self.label.dim() == 1


class ImagePerformance:
    def __init__(self, metric, ground_truth_img: torch.Tensor, label_and_prediction: LabelAndPrediction, image_group):
        self.metric = metric
        self.ground_truth: torch.Tensor = ground_truth_img
        self.label_and_prediction = label_and_prediction
        self.image_group: ImageGroup = image_group

        self.rank = None

    def get_label(self):
        return self.rank + " " + self.image_group.formatted_image_index


def update_report_with_losses(epoch, loss_history: LossHistory, html_file_path):
    soup = create_or_get_html_file(html_file_path)

    # Find or create the loss info section
    loss_info_section = soup.find('div', id='loss-info')
    if not loss_info_section:
        raise ValueError("loss_info-div not found")

    # Clear the previous contents and update with new info
    loss_info_section.clear()
    epoch_title = soup.new_tag('h2')
    epoch_title.string = f'Epoch {epoch}'
    loss_info_section.append(epoch_title)

    training_loss_paragraph = soup.new_tag('p')
    training_loss_paragraph.string = f'Training Loss: {loss_history.current_avg_train_loss:.4f}'
    loss_info_section.append(training_loss_paragraph)

    if loss_history.current_avg_val_loss:
        validation_loss_paragraph = soup.new_tag('p')
        validation_loss_paragraph.string = f'Validation Loss: {loss_history.current_avg_val_loss:.4f}'
        loss_info_section.append(validation_loss_paragraph)

    # Get the base64-encoded loss plot image from the LossHistory object
    loss_plot_base64 = loss_history.get_loss_plot_base64()

    # Create an img tag for the loss plot and add it to the HTML
    loss_plot_img = soup.new_tag('img', src=f"data:image/png;base64,{loss_plot_base64}", style="display:block;")
    loss_info_section.append(loss_plot_img)

    write_update_file(html_file_path, soup)


def create_or_get_html_file(html_file_path):
    # Open the existing HTML file or create a new structure if it doesn't exist
    if os.path.exists(html_file_path):
        with open(html_file_path, 'r') as file:
            soup = BeautifulSoup(file, 'html.parser')
    else:
        soup = BeautifulSoup(
            '<html>'
            '           <head>'
            '               <meta http-equiv="refresh" content="3">'
            '               <title>Training Output</title>'
            '           </head>'
            '           <body>'
            '               <h1>Model training run information</h1>'
            '               <h2>Loss information</h2>'
            '               <div id="loss-info"></div>'
            '               <h2>Training Output Images</h2>'
            '               <div id="samples-info"></div>'
            '           </body>'
            '       </html>',
            'html.parser')
    return soup


def convert_to_rgb(image_tensor_in):
    return image_tensor_in.repeat(3, 1, 1)


def update_report_samples_for_epoch(epoch: int, performance: List[ImagePerformance], html_file_path: str):
    images_info = rank_performances(performance)

    soup = create_or_get_html_file(html_file_path)

    # Find the body tag or create it if not exist
    samples_info_section = soup.find('div', id='samples-info')
    if not samples_info_section:
        raise ValueError("samples-info not found")

    epoch_section = soup.new_tag('div')
    epoch_title = soup.new_tag('h2')
    epoch_title.string = f'Epoch {epoch}'
    epoch_section.append(epoch_title)

    # Append images with descriptions
    for img_perf in images_info:
        assert isinstance(img_perf, ImagePerformance)
        description = soup.new_tag('p')
        description_str = f'{img_perf.rank} (Loss: {img_perf.metric:.4f}, Image Index: {img_perf.image_group.formatted_image_index}) '
        description.append(description_str)

        # Create a hyperlink to the image output
        link = soup.new_tag('a', href=img_perf.image_group.base_output_path)
        link.string = f"{img_perf.image_group.base_output_path}"
        description.append(link)

        epoch_section.append(description)

        if img_perf.label_and_prediction.is_coordinate_prediction():

            if len(img_perf.image_group.output_file_names) <= 2:
                raise ValueError("We should have filenames, but haven't for : " + img_perf.image_group.base_output_path)

            median_image_filename = img_perf.image_group.output_file_names[
                len(img_perf.image_group.output_file_names) // 2]

            color_red = (1.0, 0.0, 0.0)  # Red color
            color_green = (0.0, 1.0, 0.0)  # Green color

            updated_img = add_horizontal_line(convert_to_rgb(img_perf.ground_truth), img_perf.label_and_prediction.prediction[0], color_red)
            updated_img = add_horizontal_line(updated_img, img_perf.label_and_prediction.label[0], color_green)
            updated_img = add_vertical_line(updated_img, img_perf.label_and_prediction.prediction[1], color_red)
            updated_img = add_vertical_line(updated_img, img_perf.label_and_prediction.label[1], color_green)
            gt_base64 = tensor_to_base64(updated_img)
            img_gt = soup.new_tag('img', src=f"data:image/png;base64,{gt_base64}", width="300")
            median_image_tag = soup.new_tag('img', src=median_image_filename, width="300")
            epoch_section.append(img_gt)
            epoch_section.append(median_image_tag)

            label_pred_desc = soup.new_tag('p')
            label_pred_desc.string = (
                f'Label (x,y) : ({img_perf.label_and_prediction.label[0]}, {img_perf.label_and_prediction.label[1]}) '
                f'Prediction (x,y) : ({img_perf.label_and_prediction.prediction[0]}, {img_perf.label_and_prediction.prediction[1]})'
            )
            epoch_section.append(label_pred_desc)
        else:
            # Convert tensors to Base64 and append images
            gt_base64 = tensor_to_base64(img_perf.label_and_prediction.label)
            out_base64 = tensor_to_base64(img_perf.label_and_prediction.prediction)
            img_gt = soup.new_tag('img', src=f"data:image/png;base64,{gt_base64}", width="300")
            img_out = soup.new_tag('img', src=f"data:image/png;base64,{out_base64}", width="300")
            epoch_section.append(img_gt)
            epoch_section.append(img_out)

    samples_info_section.insert(0, epoch_section)

    max_sections = 5

    while len(samples_info_section.contents) > max_sections:
        oldest_section = samples_info_section.contents[-1]
        oldest_section.decompose()

    write_update_file(html_file_path, soup)

    if epoch == 1:
        webbrowser.open('file://' + os.path.realpath(html_file_path))


def rank_performances(performance):
    ranks = ["Best", "2nd-Best", "3rd-Best", "Median-1", "Median", "Median+1", "3rd-Worst", "2nd-Worst", "Worst"]
    performance.sort(key=lambda x: x.metric)
    performance_length = len(performance)
    median_index = performance_length // 2
    images_info = []
    # Append top 3 performances
    for i, img_perf in enumerate(performance[:3]):
        img_perf.rank = ranks[i]
        images_info.append(img_perf)
    # Handling for median performances
    if performance_length % 2 == 0:
        # Even number of performances (two medians)
        median_performances = performance[median_index - 2:median_index + 1]
    else:
        # Odd number of performances
        median_performances = performance[median_index - 1:median_index + 2]
    # Append median performances
    for i, img_perf in enumerate(median_performances):
        img_perf.rank = ranks[3 + i]  # Offset by 3 for the "median" ranks
        images_info.append(img_perf)
    # Append bottom 3 performances
    for i, img_perf in enumerate(performance[-3:]):
        img_perf.rank = ranks[6 + i]
        images_info.append(img_perf)
    return images_info


def add_vertical_line(image_tensor_in: torch.Tensor, y_position, color):
    """
    Add a vertical line to the image tensor.
    :param image_tensor_in: Tensor of shape [C, H, W]
    :param y_position: X position of the line, in the range [-1.0, 1.0]
    :return: Modified image tensor

    Args:
        color:
    """
    image_tensor = image_tensor_in.clone()
    C, H, W = image_tensor.shape
    # Normalize x_position to pixel coordinates
    pixel_y = int((1 - y_position) * W / 2)

    # Clamp to ensure the x coordinate is within the image width
    pixel_y = max(0, min(W - 1, pixel_y))

    for i, c in enumerate(color):
        image_tensor[i, :, pixel_y] = c

    return image_tensor


def add_horizontal_line(image_tensor_in: torch.Tensor, x_position, color):
    """
    Add a horizontal line to the image tensor.
    :param image_tensor_in: Tensor of shape [C, H, W]
    :param x_position: X position of the line, in the range [-1.0, 1.0]
    :return: Modified image tensor
    """
    image_tensor = image_tensor_in.clone()
    C, H, W = image_tensor.shape
    # Normalize x_position to pixel coordinates
    pixel_x = int((1 - x_position) * H / 2)

    # Clamp to ensure the x coordinate is within the image height
    pixel_x = max(0, min(H - 1, pixel_x))

    for i, c in enumerate(color):
        image_tensor[i, pixel_x, :] = c

    return image_tensor


def write_update_file(html_file_path, soup):
    # Prepare the entire content in a buffer
    buffer = StringIO()
    buffer.write(str(soup))
    buffer_content = buffer.getvalue()
    buffer.close()

    try:
        # Overwrite the file with the new content in one go
        with open(html_file_path, 'w', encoding='utf-8') as file:
            file.write(buffer_content)
    except OSError as e:
        print(f"Error writing to file: {e}")
        print(f"File path attempted: {html_file_path}")
