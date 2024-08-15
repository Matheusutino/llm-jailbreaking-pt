from src.core.predictors.predictor_manager import PredictionManager

def run_predictor():
    prediction_manager = PredictionManager(service = 'huggingface', model_name = 'nicholasKluge/TeenyTinyLlama-460m')

    result = prediction_manager.predict(messages = 'olá, bom dia, qual é capital do Brasil?')

    print(result)

run_predictor()