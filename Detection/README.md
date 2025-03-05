# Detection

This directory contains scripts and test cases related to the detection functionality of SafeScan.

## Files

### Detection Script
- **`detect.py`**: The main script responsible for detecting vulnerabilities or performing security analysis.

### Test Cases
- **`test.cpp`**: Contains test cases for verifying the functionality of `detect.py` in C++.
- **`test.php`**: Contains test cases for verifying the functionality of `detect.py` in PHP.
- **`test.py`**: Contains test cases for verifying the functionality of `detect.py` in Python.

### Additional Files
- **`input_php_context.php`**: May contain PHP code or configurations related to testing.
- **`security_report_php.html`**: Possibly an output or report generated for PHP security detection.

## Usage

Run the `detect.py` script to perform security analysis:

```bash
python detect.py --type cpp <filePath>      
python detect.py --type python <filePath> 
python detect.py --type php <filePath>    
python detect.py -help         
