import pandas as pd
from src.core.analysis import Analysis

def analysis_df():
    df = pd.read_json('dataset/TechHazardQA_train.json', lines=True)
    
    print(df)

    analysis = Analysis(df)

    n_unique_values_subject = analysis.count_unique_values('Subject')
    n_unique_values_domain = analysis.count_unique_values('Domain')

    unique_values_domain = analysis.get_unique_values('Domain')
    
    print(f"Number of unique values in 'Subject': {n_unique_values_subject}")
    print(f"Number of unique values in 'Domain': {n_unique_values_domain}")

    print(f"Unique values in 'Domain': {unique_values_domain}")

if __name__ == "__main__":
    analysis_df()
