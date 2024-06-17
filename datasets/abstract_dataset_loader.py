"""
Abstract dataset class to load a dataset and provide entries.
"""
from abc import ABC, abstractmethod

class DatasetLoader(ABC):
    registered_datasets = ["morehopqa", "morehopqa-150"]
    
    @abstractmethod
    def items(self): 
        """Should iterate over data and return items"""
        pass

    @staticmethod
    def create(dataset_name):
        from datasets.morehopqa_loader import MorehopqaLoader, Morehopqa150Loader
        if dataset_name == "morehopqa":
            return MorehopqaLoader()
        elif dataset_name == "morehopqa-150":
            return Morehopqa150Loader()
        else:
            raise ValueError(f"Dataset {dataset_name} not found.")
        
