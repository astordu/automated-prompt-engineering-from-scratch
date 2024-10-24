import asyncio
import pandas as pd
from prompt_evaluator import PromptEvaluator
from vertexai.generative_models import HarmBlockThreshold, HarmCategory


if __name__ == "__main__":
    df_train = pd.read_csv('test.csv')  # Load your training data

    target_model_name = "gemini-1.5-flash"
    target_model_config = {
        "temperature": 0, "max_output_tokens": 1000
    }
    review_model_name = "gemini-1.5-flash" 
    review_model_config = {
        "temperature": 0, "max_output_tokens": 10 
    }
    safety_settings = {
        HarmCategory.HARM_CATEGORY_HATE_SPEECH: HarmBlockThreshold.BLOCK_NONE,
        HarmCategory.HARM_CATEGORY_DANGEROUS_CONTENT: HarmBlockThreshold.BLOCK_NONE,
        HarmCategory.HARM_CATEGORY_SEXUALLY_EXPLICIT: HarmBlockThreshold.BLOCK_NONE,
        HarmCategory.HARM_CATEGORY_HARASSMENT: HarmBlockThreshold.BLOCK_NONE,
    }
    
    review_prompt_template_path = 'review_prompt_template.txt'  # Path to the review prompt text file

    evaluator = PromptEvaluator(
        df_train, target_model_name, target_model_config, review_model_name, review_model_config, safety_settings, review_prompt_template_path
    )
    
    prompt = "在回答历史问题时，首先全面回顾相关历史背景和事件，确保对每个选项有深入的理解。接着，逐步推理每个选项的可能性，排除不符合条件的选项。最后，基于逻辑推理和历史事实，确定最符合条件的答案，并详细解释选择该答案的理由。" 
    asyncio.run(evaluator.main(prompt))