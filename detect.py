import argparse
import os 
import re
import pandas as pd
from pprint import pprint
os.environ["TF_ENABLE_ONEDNN_OPTS"] = "0"
import tensorflow as tf
import torch
from transformers import RobertaForSequenceClassification , AutoTokenizer 
from tensorflow.keras.layers import TextVectorization
from keras.preprocessing.sequence import pad_sequences

def tokenize_data(code, tokenizer):

    encoding = tokenizer.batch_encode_plus(
        code.tolist(),  # Convert the DataFrame/Series to a list
        add_special_tokens=True,  # Adds the special tokens like [CLS] and [SEP]
        padding=True,  # Pad sequences to the same length
        truncation=True,  # Truncate sequences that are too long
        max_length=512,  # Adjust according to your model's max input length
        return_tensors="pt"  # Return PyTorch tensors
    )
    return encoding["input_ids"], encoding["attention_mask"]


def group_php_lines(cleaned_lines):
    chunks = []
    current_chunk = []

    for line in cleaned_lines:
        line = line.strip()

        if line.startswith("<?php"):
            if line.endswith("?>"):
                current_chunk.append(line)
                chunks.append("\n".join(current_chunk))
                current_chunk = []
            else:
                current_chunk.append(line)
        elif line.endswith("?>"):
            current_chunk.append(line)
            chunks.append("\n".join(current_chunk))
            current_chunk = []
        elif line:
            current_chunk.append(line)
        
    # If there is any remaining content in current_chunk, add it
    if current_chunk:
        chunks.append("\n".join(current_chunk))

    return chunks


def remove_comment_from_php(lines):
    cleaned_lines = []
    in_multiline_comment = False  # Reset multi-line comment mode

    for line in lines:  
        # If inside a multi-line comment
        if in_multiline_comment:
            # Look for the closing delimiter of a multi-line comment
            if line.endswith('*/'):
                in_multiline_comment = False  # Exit multi-line comment mode
            continue  # Skip the line
        
        # Check for single-line comments (//)
        if line.startswith('//'):
            continue  # Skip single-line comment lines

        # Check for the start of a multi-line comment (/*)
        if line.startswith('/*'):
            in_multiline_comment = True  # Enter multi-line comment mode
            continue  # Skip the comment line

        if line.startswith('#'):
            continue  # Skip the comment line
        
        if line:
            cleaned_lines.append(line)

    return cleaned_lines


def group_python_functions(code_lines):

    functions = []
    current_function= []
    start_of_file = True

    for line in code_lines:
        #print(line)
        if start_of_file:
            current_function.append(line)
            if line.startswith("def "):
                start_of_file = False
        elif line.startswith("def "):
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
      python detect.py --type cpp <filePath>      # Submit C file
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
                line = line.strip()
                if not line:  # Skip empty lines
                    continue  
                #print("Lines is ", line)
                line, in_multiline_comment = remove_comment_python(line, in_multiline_comment)
                if line is not None:
                    cleaned_lines.append(line)
        return cleaned_lines  # List of cleaned Python code lines

    elif file_path.endswith('.php') or file_path.endswith('.html'):
        in_php = False
        with open(file_path, 'r') as file:
            for line in file:
                line = line.strip()
                if line.startswith('<?php'):
                    if line.endswith('?>'):
                        in_php = False
                        cleaned_lines.append(line)
                        continue
                    else:
                        in_php = True
                        cleaned_lines.append(line)
                        continue
                if in_php:
                    cleaned_lines.append(line)
                    if line.endswith('?>'):
                        in_php = False
        cleaned_lines  = remove_comment_from_php(cleaned_lines)
        return cleaned_lines  # Return a list of lines for consistency

    elif file_path.endswith('.c') or file_path.endswith('.cpp'):

        #from gpt cause not sure not to sepearte .c file into functions
        with open(file_path, 'r') as file:
            lines = file.readlines()

        functions = []
        current_function = []
        inside_function = False
        brace_count = 0  # Track `{}` scope

        # Regular expression for function signatures (supports static, extern, etc.)
        function_pattern = re.compile(r"^\s*(static|extern|inline)?\s*(\w+)\s+(\w+)\s*\(.*\)\s*\{?")
        for line in lines:
            if line:
                line = line.strip()
                if not line:  # Skip empty lines
                    continue  
                if function_pattern.match(line):  # Function start
                    if current_function:  # Save the previous function before starting a new one
                        functions.append("\n".join(current_function))
                    current_function = [line]  # Start new function
                    inside_function = True
                    brace_count = line.count("{") - line.count("}")  # Count braces

                elif inside_function:
                    current_function.append(line)
                    brace_count += line.count("{") - line.count("}")  # Adjust brace count

                    if brace_count == 0:  # Function ends when braces are balanced
                        inside_function = False
                        functions.append("\n".join(current_function))
                        current_function = []

        return functions
    else:
        raise ValueError("Unsupported file format. Only .py, .php, and .c files are supported.")


def populate_chunk_confidence_yhat(group_chunks,predictions,probabilities,model,filepath):

    chunk_confidence_yhat = {}
    #populate the chunk_confidence_yhat
    for i, chunk in enumerate(group_chunks):
        #print(f"Chunk {i+1}:")
        # Get the predicted class (index of maximum probability)
        predicted_class = predictions[i].item()
        #print("Predicted class:", predicted_class)
        # Get the prob of predicted class
        confidence =  probabilities[i, predicted_class].item()  
        #print("Confidence:", confidence)
        #store to dict to pass to LLM to decide whether to investigate further
        chunk_confidence_yhat[i] = {"chunk": chunk, "yhat": predicted_class, "confidence": confidence, "model": filepath}

    return chunk_confidence_yhat


def python_vul_detector(file_to_check):

    #start of everything here 
    print("Start detecting vulnerable statement in python")
    #this is to store the vulnerable statement after all the detection 
    chunk_confidence_yhat = {}
    group_functions = []
    

    #extract the code from file 
    cleaned_lines = extract_code_from_file(file_to_check)
    #print(cleaned_lines)
    group_chunks = group_python_functions(cleaned_lines)
    for i, func in enumerate(group_chunks, 1):
        print(f"\nChunk {i}:\n{func}\n{'-'*40}")
    

    #models that have the capabilites to detect python code ; shld include the file path of the pretrained model
    #edit this part to include more models
    #models =["xformerBERT_python_model.pth" , "cnn_python_model.h5"]
    models =["xformerBERT_python_model.pth" ]
    #loop thru the model 
    for filepath in models:
        model,tokenizer = load_model(filepath)
        print("Loaded model: ", filepath) 
        
        #check if the tokenizer is not NUll
        #that means is xfer model cause i didnt load the tokenizer in the cnn model must load manual here 
        #tokenizer =  what tokenizer u used
        #this is a torch model
        if tokenizer is not None:
            #convert the group chunk list to pd series object 
            user_code = pd.Series(group_chunks)
            #put user code into tokenizer
            user_code_ids, user_code_mask = tokenize_data(user_code, tokenizer)
            #pass tokenize code to model and get y hat
            outputs = model(input_ids=user_code_ids, attention_mask=user_code_mask)
            
            # Apply sigmoid to logits to get probabilities
            logits = outputs.logits
            probabilities = torch.sigmoid(logits)
            #print("Probabilities:", probabilities)
            
            # Get the predicted class (index of maximum probability)
            predictions = torch.argmax(probabilities, dim=-1)
            #print("Predictions:", predictions)

            #populate chunk_confidence_yhat
            chunk_confidence_yhat = populate_chunk_confidence_yhat(group_chunks,predictions,probabilities,model,filepath)
        else:
            #this is a tensor model 
            print(model.summary())
            user_code = pd.Series(group_chunks)

            # Vectorize the input
            vectorizer = TextVectorization(max_tokens=20000, output_sequence_length=100, output_mode='int')
            vectorizer.adapt(user_code)
            X = vectorizer(user_code)
            #change X to float32 type ?????
            X_padded = pad_sequences(X, maxlen=300)

            # Make predictions on the test data
            y_pred = model.predict(X_padded)
            print(y_pred)


    #at the end will return just list of tuples information like the score n stuff ??
    return chunk_confidence_yhat

def php_vul_detector(file_to_check):
    #start of everything here 
    print("Start detecting vulnerable statement in php")
    #this is to store the vulnerable statement after all the detection 
    chunk_confidence_yhat = {}
    group_functions = []

    #extract the code from file 
    cleaned_lines = extract_code_from_file(file_to_check)
    #print(cleaned_lines)
    group_chunks = group_php_lines(cleaned_lines)
    for i, func in enumerate(group_chunks, 1):
        print(f"\nChunk {i}:\n{func}\n{'-'*40}")


    #models here load


    return chunk_confidence_yhat


def c_vul_detector(file_to_check):

    #start of everything here 
    print("Start detecting vulnerable statement in c")
    #this is to store the vulnerable statement after all the detection 
    chunk_confidence_yhat = {}
    group_chunks = extract_code_from_file(file_to_check)

    for i, func in enumerate(group_chunks, 1):
        print(f"\nChunk {i}:\n{func}\n{'-'*40}")

    return  chunk_confidence_yhat

def main():
    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument("--type", choices=["cpp", "python", "php"])
    parser.add_argument("file", type=str, nargs="?")
    parser.add_argument("-help", action="store_true")
    args = parser.parse_args()
    

    #backbone , see where the program goes for which type of file 
    #load the model accordingly to the files submitted
    if args.help:
        print_banner()
    elif args.type and args.file:
        #check if the file match the type of the 
        if args.type == "cpp" and (args.file.endswith(".c") or args.file.endswith(".cpp")):
            print(f"Submitting C code from file: {args.file}")
            chunk_confidence_yhat = c_vul_detector(args.file)

        elif args.type == "python" and args.file.endswith(".py"):
            print(f"Submitting Python code from file: {args.file}")
            chunk_confidence_yhat = python_vul_detector(args.file)
            pprint(chunk_confidence_yhat, sort_dicts=False)

        elif args.type == "php" and (args.file.endswith(".php") or args.file.endswith(".html")):

            print(f"Submitting PHP code from file: {args.file}")
            chunk_confidence_yhat = php_vul_detector(args.file)

        else:
            print("Contradicting file type. Please provide a valid file.")
    else:
        print("Invalid usage. Use -help for more information.")

if __name__ == "__main__":
    main()
