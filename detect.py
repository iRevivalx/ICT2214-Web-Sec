import argparse
import os 
os.environ["TF_ENABLE_ONEDNN_OPTS"] = "0"
import tensorflow as tf
import torch
from transformers import RobertaForSequenceClassification , AutoTokenizer 

def group_python_functions(code_lines):

    functions = []
    current_function= []
    start_of_file = True

    for line in code_lines:
        if start_of_file:
            current_function.append(line)
            if line.strip().startswith("def "):
                start_of_file = False
        elif line.strip().startswith("def "):
            functions.append("\n".join(current_function))
            current_function = [line]
        else:
            current_function.append(line)
    
    if current_function:  # Append the last function if any
        functions.append("\n".join(current_function))
        
    return functions

def remove_comment_python(line, in_multiline_comment):

    if in_multiline_comment:
        # Look for closing delimiter of multi-line comment
        if line.endswith('"""') or line.endswith("'''"):
            return None, False  # Exit multi-line comment mode
        return None, True  # Continue skipping lines

    # Check for single-line comment
    if line.strip().startswith("#"):
        return None, in_multiline_comment

    # Check for the start of a multi-line comment
    if line.strip().startswith("'''") or line.strip().startswith('"""'):
        return None, True  # Enter multi-line comment mode

    return line, in_multiline_comment


def print_banner():
    banner = """
    ==================================
      SafeScan Script 
    ==================================
      Use this script to submit:
      1) C code
      2) Python code
      3) PHP code
    ==================================
      Usage: 
      python detect.py --type c <filePath>      # Submit C file
      python detect.py --type python <filePath> # Submit Python file
      python detect.py --type php <filePath>    # Submit PHP file
      python detect.py -help         # Display this help message
    ==================================
    """
    print(banner)

def import_torch_model(file_path_name):
    file_path_name = os.path.join(os.getcwd(), file_path_name)
    #init the model from pretrained model microsoft codeBert
    model = RobertaForSequenceClassification.from_pretrained('microsoft/codebert-base')
    tokenizer = AutoTokenizer.from_pretrained('microsoft/codebert-base')

    #check for gpu 
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    #load model weights that is train before hand 
    #for torch model there is need to set to eval mode 
    model.load_state_dict(torch.load(file_path_name, weights_only=True, map_location=device))
    model.eval()
    model.to(device)

    return model, tokenizer

def import_tensorFlow_model(file_path_name):
    file_path_name = os.path.join(os.getcwd(), file_path_name)
    #check for gpu 
    if tf.config.list_physical_devices('GPU'):
        model = tf.keras.models.load_model(file_path_name)
        #model.summary()
        return model
    else:
        print("GPU not available, using CPU instead to load " , file_path_name)
        with tf.device('/CPU:0'):
            model = tf.keras.models.load_model(file_path_name)
            #for tensor there is no need to set to eval mode
            #model.summary()
            return model

def load_model(filepath):
    print("\nLoading model from: ", filepath)
    if filepath.endswith('.h5'):
        model = import_tensorFlow_model(filepath)
        return model, None
    elif filepath.endswith('.pth'):
        model,tokenizer = import_torch_model(filepath)
        return model,tokenizer
    else:
        raise ValueError("Unsupported model format. Only .h5 and .pth files are supported.")
    

def extract_code_from_file(file_path):
    cleaned_lines = []
    in_multiline_comment = False
    
    if file_path.endswith('.py'):
        with open(file_path, 'r') as file:
            for line in file:
                line, in_multiline_comment = remove_comment_python(line.rstrip(), in_multiline_comment)
                if line is not None:  # Avoid adding None values
                    cleaned_lines.append(line)
        return cleaned_lines  # List of cleaned Python code lines

    elif file_path.endswith('.php') or file_path.endswith('.c'):
        with open(file_path, 'r') as file:
            return file.readlines()  # Return a list of lines for consistency

    else:
        raise ValueError("Unsupported file format. Only .py, .php, and .c files are supported.")


def python_vul_detector(file_to_check):

    #start of everything here 
    print("Start detecting vulnerable statement in python")
    #this is to store the vulnerable statement after all the detection 
    vulnerable_statement = []
    group_functions = []
    

    #extract the code from file 
    cleaned_lines = extract_code_from_file(file_to_check)
    grouped_functions = group_python_functions(cleaned_lines)
    for i, func in enumerate(grouped_functions, 1):
        print(f"Function {i}:\n{func}\n{'-'*40}")
    

    #models that have the capabilites to detect python code ; shld include the file path of the pretrained model
    #edit this part to include more models
    models =["xformerBERT_python_model.pth" , "cnn_python_model.h5"]
    #loop thru the model 
    for filepath in models:
        model,tokenizer = load_model(filepath)
        print("Loaded model: ", filepath) 
        

    #at the end will return just list of tuples information like the score n stuff ??
    return vulnerable_statement

def main():
    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument("--type", choices=["c", "python", "php"], help="Specify the type of code to submit")
    parser.add_argument("file", type=str, nargs="?", help="Path to the file to check")
    parser.add_argument("-help", action="store_true", help="Show banner and usage information")
    
    args = parser.parse_args()
    

    #backbone , see where the program goes for which type of file 
    #load the model accordingly to the files submitted
    if args.help:
        print_banner()
    elif args.type and args.file:
        #check if the file match the type of the 
        if args.type == "c" and args.file.endswith(".c"):
            print(f"Submitting C code from file: {args.file}")

        elif args.type == "python" and args.file.endswith(".py"):
            print(f"Submitting Python code from file: {args.file}")
            vulnerable_statement = python_vul_detector(args.file)

        elif args.type == "php" and args.file.endswith(".php"):
            print(f"Submitting PHP code from file: {args.file}")

        else:
            print("Contradicting file type. Please provide a valid file.")
    else:
        print("Invalid usage. Use -help for more information.")

if __name__ == "__main__":
    main()
