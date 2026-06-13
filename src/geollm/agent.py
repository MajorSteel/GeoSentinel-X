import json
from transformers import pipeline, AutoModelForCausalLM, AutoTokenizer
from .prompts import GEOLLM_SYSTEM_PROMPT, ANALYSIS_PROMPT_TEMPLATE

class GeoLLMAgent:
    """
    RAG-based Agent for interacting with Earth Observation predictions.
    Utilizes a lightweight local LLM (e.g., Llama-3-8B-Instruct or similar).
    """
    def __init__(self, model_id: str = "meta-llama/Meta-Llama-3-8B-Instruct"):
        """
        Initializes the HuggingFace pipeline. 
        Note: Requires appropriate hardware (GPU) and access tokens.
        """
        # Uncomment in production environment with GPU
        # self.tokenizer = AutoTokenizer.from_pretrained(model_id)
        # self.model = AutoModelForCausalLM.from_pretrained(model_id, device_map="auto", torch_dtype="auto")
        # self.pipeline = pipeline(
        #     "text-generation",
        #     model=self.model,
        #     tokenizer=self.tokenizer,
        #     max_new_tokens=512,
        # )
        self.pipeline = None # Placeholder for development

    def format_prompt(self, query: str, bbox: list, date_range: tuple, model_results: dict):
        """
        Injects metadata into the prompt template.
        """
        metadata_str = json.dumps(model_results, indent=2)
        prompt = ANALYSIS_PROMPT_TEMPLATE.format(
            bbox=bbox,
            start_date=date_range[0],
            end_date=date_range[1],
            metadata_json=metadata_str,
            query=query
        )
        return prompt

    def generate_response(self, query: str, bbox: list, date_range: tuple, model_results: dict):
        """
        Generates a natural language response based on model predictions.
        """
        prompt = self.format_prompt(query, bbox, date_range, model_results)
        messages = [
            {"role": "system", "content": GEOLLM_SYSTEM_PROMPT},
            {"role": "user", "content": prompt},
        ]
        
        if self.pipeline:
            outputs = self.pipeline(messages)
            return outputs[0]["generated_text"][-1]["content"]
        else:
            # Mock response for testing
            return (f"[Mock LLM] Based on the data from {date_range[0]} to {date_range[1]}, "
                    f"the model detects a 12% decrease in forest cover correlated with "
                    f"increased urban built-up areas. GradCAM highlights the eastern sector "
                    f"as the primary driver for this prediction.")

if __name__ == "__main__":
    agent = GeoLLMAgent()
    results = {
        "lulc_majority": "Urban",
        "change_detected": "Deforestation",
        "change_confidence": 0.94,
        "climate_trend": "Reduced precipitation"
    }
    response = agent.generate_response("Why is vegetation decreasing here?", [-122, 47, -121, 48], ("2020", "2023"), results)
    print(response)
