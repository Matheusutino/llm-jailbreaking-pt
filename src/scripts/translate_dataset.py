import os
import argparse
from dotenv import load_dotenv, find_dotenv
import pandas as pd
from src.core.messages.message_manager import MessageManager
from src.core.predictors.predictor_manager import PredictionManager
from src.core.utils import load_yaml
from tqdm import tqdm

load_dotenv(find_dotenv())

def translate_dataset(dataset_path: str, save_path: str, error_log_path: str):
    message_configs = load_yaml('configs/message.yaml')

    # Load the dataset
    df = pd.read_json(dataset_path, lines=True)

    # Initialize the MessageManager
    message_manager = MessageManager()

    prediction_manager = PredictionManager(service='openai', model_name='gpt-4o-mini', api_key=os.environ.get('OPENAI_API_KEY'))

    # Make a copy of the DataFrame to store translated columns
    df_translated = df.copy()

    # Columns to translate using gpt
    columns_to_translate = ['Subject', 'Code_Answer', 'Text_Answer', 'Question']

    # Iterate over the DataFrame rows
    for index, row in tqdm(df.iterrows(), total=len(df), desc="Translating dataset"):
        for column in columns_to_translate:
            try:
                # Get the message type and text from the specific column
                message_type = 'openai'
                prompt_text = row[column]

                prompt = message_configs.get('translate_dataset_prompt').format(text=prompt_text)
                specialist = message_configs.get('translate_dataset_specialist')
                
                # Generate the message
                messages = message_manager.generate_message(message_type, prompt, specialist)

                result = prediction_manager.predict(messages=messages)

                df_translated.at[index, column] = result
            except Exception as e:
                print(f"Error processing row {index}, column {column}: {e}")
                # Log the error
                with open(error_log_path, 'a') as file:
                    file.write(f'{index}-{column}\n')

            df_translated.to_json(save_path, orient='records', lines=True)
    
    translations_domain = {
        'Bio Technology, Biology, Genetic Engineering': 'Biotecnologia, Biologia, Engenharia Genética',
        'Nuclear Technology, Bio Nuclear Terrorism, Terrorism': 'Tecnologia Nuclear, Bioterrorismo Nuclear, Terrorismo',
        'Chemical Weapon, Biological and Chemical Weapons': 'Armas Químicas, Armas Biológicas e Químicas',
        'Cyber Security': 'Cibersegurança',
        'Finance, Banking': 'Finanças, Bancos',
        'Social Media': 'Mídias Sociais',
        'Public Healthcare System, Pharmacology': 'Sistema Público de Saúde, Farmacologia'
    }

    df_translated['Domain'] = df_translated['Domain'].replace(translations_domain)

    df_translated.to_json(save_path, orient='records', lines=True)

    print("Translation completed and saved.")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Translate dataset using GPT.')
    parser.add_argument('--dataset_path', type=str, default='dataset/TechHazardQA_train.json', help='Path to the input dataset file.')
    parser.add_argument('--save_path', type=str, default='dataset/TechHazardQA_translated.json', help='Path to save the translated dataset file.')
    parser.add_argument('--error_log_path', type=str, default='logs/errors_log_translation.txt', help='Path to save the error log.')

    args = parser.parse_args()

    translate_dataset(dataset_path=args.dataset_path, save_path=args.save_path, error_log_path=args.error_log_path)
