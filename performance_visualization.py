import base64
import os
from io import BytesIO
from typing import List

from bs4 import BeautifulSoup
from torchvision.transforms import ToPILImage


def tensor_to_base64(tensor):
    """Convert a PyTorch tensor to a Base64-encoded image."""
    pil_image = ToPILImage()(tensor.cpu())  # Convert tensor to PIL image
    buffered = BytesIO()
    pil_image.save(buffered, format="PNG")  # Save image to a buffer
    return base64.b64encode(buffered.getvalue()).decode("utf-8")  # Encode to base64


class ImagePerformance:
    def __init__(self, metric, ground_truth, output, image_index):
        self.metric = metric
        self.ground_truth = ground_truth
        self.output = output
        self.image_index = image_index

        self.rank = None

    def get_label(self):
        return self.rank + " "+self.image_index


def update_html_for_epoch(epoch: int, images_info: List[ImagePerformance], html_file_path: str):
    if os.path.exists(html_file_path):
        with open(html_file_path, 'r') as file:
            soup = BeautifulSoup(file, 'html.parser')
    else:
        default_html = '<html><head><title>Training Output</title></head><body><h1>Training Output Images</h1></body></html>'
        soup = BeautifulSoup(default_html, 'html.parser')

    # Find the body tag or create it if not exist
    body = soup.find('body')
    if not body:
        body = soup.new_tag('body')
        soup.html.append(body)

    epoch_section = soup.new_tag('div')
    epoch_title = soup.new_tag('h2')
    epoch_title.string = f'Epoch {epoch}'
    epoch_section.append(epoch_title)

    # Append images with descriptions
    for img_perf in images_info:
        description = soup.new_tag('p')
        description.string = f'{img_perf.rank} (Loss: {img_perf.metric:.4f}, Image Index: {img_perf.image_index})'
        epoch_section.append(description)

        # Convert tensors to Base64 and append images
        gt_base64 = tensor_to_base64(img_perf.ground_truth)
        out_base64 = tensor_to_base64(img_perf.output)
        img_gt = soup.new_tag('img', src=f"data:image/png;base64,{gt_base64}", width="300")
        img_out = soup.new_tag('img', src=f"data:image/png;base64,{out_base64}", width="300")
        epoch_section.append(img_gt)
        epoch_section.append(img_out)

    # Insert the new epoch section at the beginning of the body
    soup.body.insert(0, epoch_section)
    body.insert(0, epoch_section)

    # Write the updated content back to the HTML file
    with open(html_file_path, 'w') as file:
        file.write(str(soup))
