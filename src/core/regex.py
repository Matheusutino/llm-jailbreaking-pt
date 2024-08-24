import re

def extract_elements_translated_text(text: str):
    """
    Extracts the values of 'Subject', 'Text_Answer', 'Code_Answer', and 'Question' (or 'questão', 'pergunta') from a formatted text.
    
    Args:
        text (str): The input text containing the formatted elements.
    
    Returns:
        dict: A dictionary with the extracted values, or None for fields that are not found.
    """
    pattern = r"(?i)subject:\s*(?P<subject>.*?)\ntext_answer:\s*(?P<text_answer>.*?)\ncode_answer:\s*(?P<code_answer>.*?)\n(?:question|questão|pergunta):\s*(?P<question>.*)"
    
    matches = re.match(pattern, text, re.DOTALL)
    
    # Inicializa o dicionário com None para os campos que podem não ser encontrados
    result = {
        "Subject": None,
        "Text_Answer": None,
        "Code_Answer": None,
        "Question": None
    }
    
    # Lista para armazenar campos não encontrados
    missing_fields = []
    
    # Se houver correspondência, preenche os campos encontrados
    if matches:
        if matches.group("subject"):
            result["Subject"] = matches.group("subject").strip()
        else:
            missing_fields.append("Subject")
            
        if matches.group("text_answer"):
            result["Text_Answer"] = matches.group("text_answer").strip()
        else:
            missing_fields.append("Text_Answer")
            
        if matches.group("code_answer"):
            result["Code_Answer"] = matches.group("code_answer").strip()
        else:
            missing_fields.append("Code_Answer")
            
        if matches.group("question"):
            result["Question"] = matches.group("question").strip()
        else:
            missing_fields.append("Question")
    else:
        missing_fields = ["Subject", "Text_Answer", "Code_Answer", "Question"]

    # Imprime quais campos não foram encontrados, se houver
    if missing_fields:
        print(f"The following fields were not found: {', '.join(missing_fields)}")
    
    return result
