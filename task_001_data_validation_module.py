import pandas as pd
import numpy as np

class EquityDataValidator:
    def __init__(self, df):
        self.df = df

    def check_missing_values(self):
        """
        Check for missing values in the DataFrame.
        Returns a dictionary of columns with their missing value counts.
        """
        missing_values = self.df.isnull().sum()
        missing_report = missing_values[missing_values > 0].to_dict()

        return missing_report

    def check_outliers(self, columns, z_threshold=3):
        """
        Check for outliers in the specified columns using the Z-score method.

        Parameters:
        - columns: List of column names to check for outliers.
        - z_threshold: Threshold for the Z-score to classify outliers (default: 3).

        Returns:
        A dictionary where keys are column names and values are lists of row indices with outliers.
        """
        outlier_report = {}
        for col in columns:
            if self.df[col].dtype in [np.float64, np.int64]:
                mean = self.df[col].mean()
                std = self.df[col].std()
                z_scores = (self.df[col] - mean) / std
                outliers = self.df.index[np.abs(z_scores) > z_threshold].tolist()
                if outliers:
                    outlier_report[col] = outliers

        return outlier_report

    def check_format(self, column, expected_format):
        """
        Check if a column conforms to an expected format.

        Parameters:
        - column: Column name to check.
        - expected_format: A callable that takes a value and returns True if valid, False otherwise.

        Returns:
        A list of row indices where the format is incorrect.
        """
        invalid_rows = [i for i, val in enumerate(self.df[column]) if not expected_format(val)]

        return invalid_rows

    def validate(self):
        """
        Run all validation checks and return a comprehensive report.
        """
        report = {}

        # Missing values
        missing_report = self.check_missing_values()
        if missing_report:
            report['missing_values'] = missing_report

        # Outliers (example columns)
        numeric_columns = self.df.select_dtypes(include=[np.number]).columns.tolist()
        outlier_report = self.check_outliers(numeric_columns)
        if outlier_report:
            report['outliers'] = outlier_report

        # Incorrect formats (example column: UpdatedAt expected to be a valid date)
        def is_valid_date(value):
            try:
                pd.to_datetime(value)
                return True
            except Exception:
                return False

        format_issues = self.check_format('UpdatedAt', is_valid_date)
        if format_issues:
            report['format_issues'] = {'UpdatedAt': format_issues}

        return report

# Usage Example
if __name__ == "__main__":
    # Load data from a CSV file
    df = pd.read_csv("equity_data.csv")

    # Validate the DataFrame
    validator = EquityDataValidator(df)
    validation_report = validator.validate()

    # Display validation report
    print("Validation Report:", validation_report)
