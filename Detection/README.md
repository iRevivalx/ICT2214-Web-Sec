# Detection

This directory contains scripts and test cases related to the detection functionality of SafeScan.

## Files

### Detection Script
- **`detect.py`**: The main script responsible for detecting vulnerabilities or performing security analysis.

### Test Cases
- **`test.cpp`**: Contains test cases for verifying the functionality of `detect.py` in C++.
- **`test.php`**: Contains test cases for verifying the functionality of `detect.py` in PHP.
- **`test.py`**: Contains test cases for verifying the functionality of `detect.py` in Python.

### Files Generated from Demo
- **`input_php_context.php`**: Some PHP code or configurations used for demo.
- **`security_report_php.html`**: The report generated for PHP security detection for demo.

## Usage

Run the `requirements.txt` to install the required dependencies to run the script:

```bash
pip install -r requirements.txt

Run the `detect.py` script to perform security analysis:

```bash
python detect.py --type cpp <filePath>      
python detect.py --type python <filePath> 
python detect.py --type php <filePath>    
python detect.py -help         
