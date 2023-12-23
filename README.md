# COMPUTER VISION IMAGE RESTORATION PROJECT D7

For preprocessing, cleaning images and running AOS go to LFR/python/AOS_integrator.ipynb - please follow the readme there to install dependencies for AOS

Later, for running the model, go back to the main directory and execute:

- conda env create -f environment.yml

- conda activate compvis-model

If you have GPU support locally, make sure you have cuda installed 
(nvcc --version should emit the currently installed version, if not installed, go to nvidia to install)
and execute something like the following adapted to your environment
(from the pytorch getting started page):

conda install pytorch torchvision torchaudio pytorch-cuda=12.1 -c pytorch -c nvidia

