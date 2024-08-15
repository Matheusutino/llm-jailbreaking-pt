import pandas as pd

class Analysis:
    def __init__(self, df: pd.DataFrame):
        """
        Initializes the class with a DataFrame.
        
        Args:
            df (pd.DataFrame): The DataFrame to be analyzed.
        """
        self.df = df

    def count_unique_values(self, column_name: str) -> int:
        """
        Counts the number of unique values in a specific column of the DataFrame.
        
        Args:
            column_name (str): The name of the column to count unique values.
        
        Returns:
            int: The number of unique values in the column.
        """
        if column_name not in self.df.columns:
            raise ValueError(f"The column '{column_name}' does not exist in the DataFrame.")
        
        return self.df[column_name].nunique()

    def get_unique_values(self, column_name: str) -> pd.Series:
        """
        Gets the unique values in a specific column of the DataFrame.
        
        Args:
            column_name (str): The name of the column to get unique values from.
        
        Returns:
            pd.Series: A series containing the unique values in the column.
        """
        if column_name not in self.df.columns:
            raise ValueError(f"The column '{column_name}' does not exist in the DataFrame.")
        
        return self.df[column_name].unique()