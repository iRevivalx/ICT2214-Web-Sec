# ICT2214-Web-Sec
P1CG1 - ICT2214 Web Security

## Project Description
This project aims to develop a machine learning-powered web security scanner for C++, Python and PHP code. Our tool will help identify common coding vulnerabilities, including:
- Outdated dependencies and their associated vulnerabilities
- Detection of vulnerabilities related to OWASP Top 10

We use a Convolutional Neural Network (CNN) to detect unsafe coding practices from code snippets.
If the CNN predicts vulnerabilities with low confidence, a Large Language Model (LLM) will validate and confirm the findings.
The model is trained on a dataset containing vulnerable and non-vulnerable C++, Python and PHP code.

## Details of the Model (C++)
### Deep Learning Model
- A 1D CNN-based classifier to analyze C++ code snippets
- Uses word embeddings and convolutional layers to detect patterns in vulnerable code
- Handles binary classification (Vulnerable vs. Non-Vulnerable Code)

### Data Preprocessing
- Tokenization of C++ code
- Padding to ensure equal-length sequences
- Label encoding for target values

### Dataset Processing & Filtering
- Filters out invalid entries and ensures label consistency
- Targets specific Common Weakness Enumeration (CWE) vulnerabilities
- Balances dataset by including both vulnerable and non-vulnerable code

### Evaluation & Metrics
- Computes accuracy, precision, recall, and F1-score
- Generates training and validation accuracy plots
- Saves trained model for deployment

### Results
- Accuracy: 97.61%
- Precision: 95.28%
- Recall: 97.61%
- F1-score: 96.43%

## Details of the Model (Python)
### Deep Learning Model
- A 1D CNN-based classifier to analyze Python code snippets
- Uses word embeddings and convolutional layers to detect patterns in vulnerable code
- Handles binary classification (Vulnerable vs. Non-Vulnerable Code)
- Requires Function by Function Submission

### Data Preprocessing
- Split of safe & vulnerable code
- Tokenization of Python code
- Padding to ensure equal-length sequences (300)
- Label encoding for target values (1 or 0)

### Dataset Processing & Filtering
- Removal of Python comments
- Balances dataset by including both vulnerable and non-vulnerable code

### Evaluation & Metrics
- Computes accuracy, precision, recall, and F1-score
- Generates training and validation accuracy plots
- Saves trained model for deployment

### Results
- Accuracy: %
- Precision: %
- Recall: %
- F1-score: %

## Details of the Model (PHP)
### Deep Learning Model
- A 1D CNN-based classifier to analyze PHP code snippets
- Uses word embeddings and convolutional layers to detect patterns in vulnerable code
- Handles binary classification (Vulnerable vs. Non-Vulnerable Code)
- Requires Function by Function Submission

### Data Preprocessing
- Split of safe & vulnerable code
- Tokenization of PHP code
- Padding to ensure equal-length sequences (300)
- Label encoding for target values (1 or 0)

### Dataset Processing & Filtering
- Removal of PHP comments & opening braces
- Balances dataset by including both vulnerable and non-vulnerable code

### Evaluation & Metrics
- Computes accuracy, precision, recall, and F1-score
- Generates training and validation accuracy plots
- Saves trained model for deployment

### Results
- Accuracy: %
- Precision: %
- Recall: %
- F1-score: %


## Submission Deadline
9 March 2025 Sunday 2359 hours
