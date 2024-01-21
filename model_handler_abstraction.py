
from abc import ABC, abstractmethod

class AbstractModelHandler(ABC):
    @abstractmethod
    def load_model(self):
        pass

    @abstractmethod
    def load_inputs(self, input_path):
        pass

    @abstractmethod
    def process_inputs(self, inputs):
        pass

    @abstractmethod
    def save_output(self, output):
        pass