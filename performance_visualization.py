import base64
import os
from io import BytesIO, StringIO
from typing import List

import numpy as np
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
            raise ValueError("Unknown phase : "+phase)


def tensor_to_base64(tensor):
    """Convert a PyTorch tensor to a Base64-encoded image."""
    pil_image = ToPILImage()(tensor.cpu())  # Convert tensor to PIL image
    buffered = BytesIO()
    pil_image.save(buffered, format="PNG")  # Save image to a buffer
    return base64.b64encode(buffered.getvalue()).decode("utf-8")  # Encode to base64


class ImagePerformance:
    def __init__(self, metric, ground_truth, output, image_group):
        self.metric = metric
        self.ground_truth = ground_truth
        self.output = output
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


def update_report_samples_for_epoch(epoch: int, images_info: List[ImagePerformance], html_file_path: str):
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
        description = soup.new_tag('p')
        description_str = f'{img_perf.rank} (Loss: {img_perf.metric:.4f}, Image Index: {img_perf.image_group.formatted_image_index}) '
        description.append(description_str)

        # Create a hyperlink to the image output
        link = soup.new_tag('a', href=img_perf.image_group.base_output_path)
        link.string = f"{img_perf.image_group.base_output_path}"
        description.append(link)

        epoch_section.append(description)

        # Convert tensors to Base64 and append images
        gt_base64 = tensor_to_base64(img_perf.ground_truth)
        out_base64 = tensor_to_base64(img_perf.output)
        img_gt = soup.new_tag('img', src=f"data:image/png;base64,{gt_base64}", width="300")
        img_out = soup.new_tag('img', src=f"data:image/png;base64,{out_base64}", width="300")
        epoch_section.append(img_gt)
        epoch_section.append(img_out)

    samples_info_section.insert(0, epoch_section)

    write_update_file(html_file_path, soup)


def write_update_file(html_file_path, soup):
    # Prepare the entire content in a buffer
    buffer = StringIO()
    buffer.write(str(soup))
    buffer_content = buffer.getvalue()
    buffer.close()

    # Overwrite the file with the new content in one go
    with open(html_file_path, 'w', encoding='utf-8') as file:
        file.write(buffer_content)
