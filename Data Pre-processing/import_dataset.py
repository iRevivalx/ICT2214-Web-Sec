import pandas as pd
import re

df = pd.read_json("hf://datasets/CyberNative/Code_Vulnerability_Security_DPO/secure_programming_dpo.json", lines=True)

# Filter rows where language is 'python'
python_rows = df[df['lang'] == 'python'].copy()
php_rows = df[df['lang'] == 'php'].copy()

# Remove comments in dataset
def remove_comments(dataframe, lang):
    def process_column(value):
        if isinstance(value, str):
            if lang == "python":
                # Remove Python comments (lines starting with #)
                return re.sub(r'^\s*#.*$', '', value, flags=re.MULTILINE)
            elif lang == "php":
                # Remove PHP single-line comments (// and #) and multi-line comments (/* */)
                value = re.sub(r'^\s*//.*$', '', value, flags=re.MULTILINE)  # Single-line //
                value = re.sub(r'^\s*#.*$', '', value, flags=re.MULTILINE)   # Single-line #
                value = re.sub(r'/\*.*?\*/', '', value, flags=re.DOTALL)     # Multi-line /* */
                return value
        return value

    dataframe['chosen'] = dataframe['chosen'].apply(process_column)
    dataframe['rejected'] = dataframe['rejected'].apply(process_column)
    return dataframe

# Apply the function to remove comments
python_rows = remove_comments(python_rows, "python")
php_rows = remove_comments(php_rows, "php")

def remove_first_last_lines(dataframe):
    # Function to process each column value by removing the first and last lines
    def process_column(value):
        if isinstance(value, str):
            lines = value.split("\n")
            if len(lines) > 2:
                # Remove the first and last lines
                return "\n".join(lines[1:-1])
        return value

    # Apply the function to both 'chosen' and 'rejected' columns
    dataframe['chosen'] = dataframe['chosen'].apply(process_column)
    dataframe['rejected'] = dataframe['rejected'].apply(process_column)
    return dataframe

# Apply the function to php_rows
python_rows = remove_first_last_lines(python_rows)

# 2 times for php cuz of the brackets
php_rows = remove_first_last_lines(php_rows)
php_rows = remove_first_last_lines(php_rows)
    
# Export to CSV
python_rows.to_csv('python_vuln_CyberNative.csv', index=False)
php_rows.to_csv('php_vuln_CyberNative.csv', index=False)
