import argparse
import pandas as pd
from collections import defaultdict
from src.core.messages.message_manager import MessageManager
from src.core.predictors.predictor_manager import PredictionManager
from src.core.evaluation import Evaluation
from src.core.utils import load_yaml, check_file_exists
from tqdm import tqdm

def zero_shot(dataset_path: str, service: str = 'openai', model_name: str = 'gpt-4o-mini', message_type: str = 'openai', prompt_zero_shot_name: str = None, specialist_zero_shot_name: str = None):
    """
    Process the dataset using zero-shot predictions.

    Args:
        dataset_path (str): Path to the input dataset file.
        service (str): The service name ('openai' or other service). Defaults to 'openai'.
        model_name (str): The name of the model. Defaults to 'gpt-4o-mini'.
        message_type (str): The type of message to use ('openai' or other type). Defaults to 'openai'.
    """
    path_to_save = f'results/{model_name}_zero_shot.csv'
    check_file_exists(path_to_save)

    message_configs = load_yaml('configs/message.yaml')

    # Load the dataset
    df = pd.read_json(dataset_path, lines=True)

    # Initialize the MessageManager
    message_manager = MessageManager()

    prediction_manager = PredictionManager(service=service, model_name=model_name)

    evaluation = Evaluation()

    results_zero_shot = []
    results_evaluation = []
    
    for index, row in tqdm(df.iterrows(), total=len(df), desc="Zero Shot"):
        # Get the message type and text from the specific column
            prompt_zero_shot = message_configs[prompt_zero_shot_name].format(question = row['Question'])

            # Prepare the dictionary for formatting
            format_dict = defaultdict(str, domain=row.get('Domain', ''), subject=row.get('Subject', ''))

            specialist_zero_shot = message_configs[specialist_zero_shot_name].format_map(format_dict)

            # Generate the message
            messages_zero_shot = message_manager.generate_message(message_type, prompt_zero_shot, specialist_zero_shot)

            result_zero_shot = prediction_manager.predict(messages=messages_zero_shot)

            result_evaluation = evaluation.evaluate_result(result = result_zero_shot)

            results_zero_shot.append(result_zero_shot)
            results_evaluation.append(result_evaluation)
    
    df['Results'] = results_zero_shot
    df['Evaluation'] = results_evaluation

    df.to_csv(path_to_save, index=False)
    
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Process dataset using zero-shot predictions.')
    parser.add_argument('--dataset_path', type=str, default='dataset/TechHazardQA_translated.jsonl', help='Path to the input dataset file.')
    parser.add_argument('--service', type=str, default='openai', help='The service name (e.g., "openai" or other).')
    parser.add_argument('--model_name', type=str, default='gpt-4o-mini', help='The name of the model.')
    parser.add_argument('--message_type', type=str, default='openai', help='The type of message (e.g., "openai" or other).')
    parser.add_argument('--prompt_zero_shot_name', type=str, required=True, help='The template string for the zero-shot prompt.')
    parser.add_argument('--specialist_zero_shot_name', type=str, required=True, help='The template string for the specialist zero-shot prompt.')

    args = parser.parse_args()

    zero_shot(
        dataset_path=args.dataset_path,
        service=args.service,
        model_name=args.model_name,
        message_type=args.message_type,
        prompt_zero_shot_name=args.prompt_zero_shot_name,
        specialist_zero_shot_name=args.specialist_zero_shot_name
    )