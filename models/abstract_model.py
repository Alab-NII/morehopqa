"""
Abstract model class to run evaluation. 
Should abstract away the prompt and loading of the model. 
Important: Cache all model answers.
"""

from abc import ABC, abstractmethod
from datetime import datetime


class AbstractModel(ABC):
    registered_models = ["gpt-3.5-turbo-direct", "gpt-4-turbo-direct", "gpt-4o-direct", "gemma-7b", "llama-8b", "llama-70b", "mistral-7b", "baseline"]

    @abstractmethod
    def get_answers_and_cache(self, dataset) -> dict:
        """Should iterate over dataset and cache all answers.
        
        Returns: dict of answers (key: id in initial dataset, value: model_answer)"""
        pass

    @staticmethod
    def create(model_name, output_file_name, prompt_generator):
        import models.openai_direct_model   
        import models.openai_batch_model    
        import models.gemma_7b
        import models.llama_8b
        import models.mistral_7b
        import models.llama_70b
        import models.baseline
        registered_models = {
        "gpt-3.5-turbo-direct": models.openai_direct_model.OpenAIDirectModel,
        "gpt-4-turbo-direct": models.openai_direct_model.OpenAIDirectModel,
        "gpt-4o-direct": models.openai_direct_model.OpenAIDirectModel,
        "gemma-7b": models.gemma_7b.Gemma7B,
        "llama-8b": models.llama_8b.Llama8b,
        "llama-70b": models.llama_70b.Llama70b,
        "mistral-7b": models.mistral_7b.Mistral7B,
        "baseline": models.baseline.Baseline
    }
        if model_name in registered_models:
            return registered_models[model_name](model_name=model_name, output_file_name=f"{output_file_name}_{model_name}_{datetime.now().strftime('%y%m%d-%H%M%S')}.json", prompt_generator=prompt_generator)
        
        raise ValueError(f"Model {model_name} not found.")