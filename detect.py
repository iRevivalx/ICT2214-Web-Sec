import argparse
import os 
import re
import pandas as pd
from pprint import pprint
os.environ["TF_ENABLE_ONEDNN_OPTS"] = "0"
os.environ["OPENAI_API_KEY"] = "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"
print(os.getenv("OPENAI_API_KEY"))
import tensorflow as tf
import numpy as np
import torch
from transformers import RobertaForSequenceClassification , AutoTokenizer 
from tensorflow.keras.layers import TextVectorization
from keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.preprocessing.text import Tokenizer
from openai import OpenAI


client = OpenAI(
    api_key = os.environ.get("OPENAI_API_KEY"),
)

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
        brace_count = 0  

        # Regex to match function signatures that explicitly start with "void"
        function_pattern = re.compile(r"^\s*void\s+(\w+)\s*\(.*\)\s*\{?$")

        for line in lines:
            stripped_line = line.rstrip()

            if function_pattern.match(stripped_line):  # Function start detected
                if current_function:  # Save previous function before starting a new one
                    functions.append("\n".join(current_function))

                current_function = [stripped_line]  
                inside_function = True
                brace_count = stripped_line.count("{") - stripped_line.count("}")  

            elif inside_function:
                current_function.append(stripped_line)
                brace_count += stripped_line.count("{") - stripped_line.count("}")  

                if brace_count == 0:  # Function ends when braces are balanced
                    inside_function = False
                    functions.append("\n".join(current_function))
                    current_function = []

        # Ensure the last function is added
        if current_function:
            functions.append("\n".join(current_function))

        return functions
    else:
        raise ValueError("Unsupported file format. Only .py, .php, and .c files are supported.")


def populate_chunk_confidence_yhat(chunk_confidence_yhat,group_chunks,predictions,probabilities,model,filepath,flatten):


    #print(probabilities)
    if flatten:
        #populate the chunk_confidence_yhat
        for i, chunk in enumerate(group_chunks):
            #print(f"Chunk {i+1}:")
            # Get the predicted class (index of maximum probability)
            predicted_class = predictions[i].item()
            if predicted_class == 1:
                confidence = probabilities[i].item()
            else:
                confidence = 1 - probabilities[i].item()
            #print("Confidence:", confidence)
            chunk_confidence_yhat.append({"chunk": chunk, "chunk_id" : i, "yhat": predicted_class, "confidence": confidence, "model": filepath})
            
    else:
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
            chunk_confidence_yhat.append({"chunk": chunk, "chunk_id" : i, "yhat": predicted_class, "confidence": confidence, "model": filepath})

    return chunk_confidence_yhat


def python_vul_detector(file_to_check):

    #start of everything here 
    print("Start detecting vulnerable statement in python")
    #this is to store the vulnerable statement after all the detection 
    chunk_confidence_yhat = []
    group_functions = []
    

    #extract the code from file 
    cleaned_lines = extract_code_from_file(file_to_check)
    #print(cleaned_lines)
    #print(cleaned_lines)
    group_chunks = group_python_functions(cleaned_lines)
    #print(group_chunks)

    if not group_chunks:
        print("No functions found in the file. Exiting program.")
        exit(1) 

    for i, func in enumerate(group_chunks, 1):
        print(f"\nChunk {i}:\n{func}\n{'-'*40}")
    
    
    #models that have the capabilites to detect python code ; shld include the file path of the pretrained model
    #edit this part to include more models
    #models =["xformerBERT_python_model.pth" , "cnn_python_model_new.h5"]
    #2 models availble for python but only use cnn cause it has better f1 score than xformer
    models =["cnn_python_model_new.h5" ]
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
            class_1_probs = probabilities[:, 1]
            predictions = (class_1_probs >= 0.64).long()

            #populate chunk_confidence_yhat
            chunk_confidence_yhat = populate_chunk_confidence_yhat(chunk_confidence_yhat,group_chunks,predictions,probabilities,model,filepath,flatten=False)
        else:
            #this is a tensor model 
            #print(model.summary())
            user_code = pd.Series(group_chunks)

            # Vectorize the input
            vectorizer = TextVectorization(max_tokens=20000, output_sequence_length=100, output_mode='int')
            vectorizer.adapt(user_code)
            X = vectorizer(user_code)
            #some padding
            X = pad_sequences(X, maxlen=300)

            # Make predictions on the test data
            y_pred = model.predict(X)
            #print(y_pred)
            # TODO: Finetune to prediction threshold
            y_pred_classes = (y_pred > 0.26).astype("int32")
            #print(y_pred_classes)

            """ print('Predicted_Probability', y_pred.flatten())               # Tis is my probability
            print('Predicted_Class', y_pred_classes.flatten())             # predicted class """

            #populate chunk_confidence_yhat
            chunk_confidence_yhat = populate_chunk_confidence_yhat(chunk_confidence_yhat,group_chunks,y_pred_classes.flatten(),y_pred.flatten(),model,filepath,flatten=True)


    #at the end will return just list of tuples information like the score n stuff ??
    return chunk_confidence_yhat

def php_vul_detector(file_to_check):
    #start of everything here 
    print("Start detecting vulnerable statement in php")
    #this is to store the vulnerable statement after all the detection 
    chunk_confidence_yhat = []
    group_functions = []

    #extract the code from file 
    cleaned_lines = extract_code_from_file(file_to_check)
    #print(cleaned_lines)
    group_chunks = group_php_lines(cleaned_lines)
    if not group_chunks:
        print("No functions found in the file. Exiting program.")
        exit(1) 

    for i, func in enumerate(group_chunks, 1):
        print(f"\nChunk {i}:\n{func}\n{'-'*40}")

    #models here load
    #models = ["xformerBERT_php_model.pth","cnn_php_model_new.h5"]
    #2 models availble for php but only use xformer cause it has better f1 score than cnn
    models = ["xformerBERT_php_model.pth"]
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
            class_1_probs = probabilities[:, 1]
            predictions = (class_1_probs >= 0.4).long()

            #populate chunk_confidence_yhat
            chunk_confidence_yhat = populate_chunk_confidence_yhat(chunk_confidence_yhat,group_chunks,predictions,probabilities,model,filepath,flatten=False)
        else:
            #this is a tensor model 
            #print(model.summary())
            user_code = pd.Series(group_chunks)

            # Vectorize the input
            vectorizer = TextVectorization(max_tokens=20000, output_sequence_length=100, output_mode='int')
            vectorizer.adapt(user_code)
            X = vectorizer(user_code)
            #some padding
            X = pad_sequences(X, maxlen=300)

            # Make predictions on the test data
            y_pred = model.predict(X)
            #print(y_pred)
            # TODO: Finetune to prediction threshold
            y_pred_classes = (y_pred > 0.16).astype("int32")
            #print(y_pred_classes)

            """ print('Predicted_Probability', y_pred.flatten())               # Tis is my probability
            print('Predicted_Class', y_pred_classes.flatten())             # predicted class """

            #populate chunk_confidence_yhat
            chunk_confidence_yhat = populate_chunk_confidence_yhat(chunk_confidence_yhat,group_chunks,y_pred_classes.flatten(),y_pred.flatten(),model,filepath,flatten=True)



    return chunk_confidence_yhat


def c_vul_detector(file_to_check):

    #start of everything here 
    print("Start detecting vulnerable statement in c")
    #this is to store the vulnerable statement after all the detection 
    chunk_confidence_yhat = []
    group_chunks = extract_code_from_file(file_to_check)
    #print(group_chunks)

    print("Group chunks type :" , type(group_chunks))
    if not group_chunks:
        print("No functions found in the file. Exiting program.")
        exit(1) 

    #print("Group chunks:" , group_chunks)
    for i, func in enumerate(group_chunks, 1):
        print(f"\nChunk {i}:\n{func}\n{'-'*40}")


    models = ["cnn_c++_model.h5"]
    for filepath in models:
        model = None
        tokenizer = None
        model,tokenizer = load_model(filepath)
        print("Loaded model: ", filepath) 

        if tokenizer is None:
            #this is tensor model
            user_code = pd.Series(group_chunks)
            # Initialize the tokenizer
            tokenizer = Tokenizer(num_words=20000, oov_token="")
            tokenizer.fit_on_texts(user_code)
            # Tokenize and pad sequences
            X_tokenized = tokenizer.texts_to_sequences(user_code)
            X = pad_sequences(X_tokenized, maxlen=300, padding="post", truncating="post")

            # Make predictions on the test data
            #print(model.summary())
            
            y_pred = model.predict(X)
            #print(y_pred)

            
            # Get model predictions
            y_pred = model.predict(X)
            # TODO: Finetune to prediction threshold
            y_pred_classes = (y_pred > 0.18).astype(int)  # Convert probabilities to class labels
            #print(y_pred_classes)
            #print('Predicted_Probability', y_pred)               # Tis is my probability
            #print('Predicted_Class', y_pred_classes)             # predicted class

            #populate chunk_confidence_yhat
            chunk_confidence_yhat = populate_chunk_confidence_yhat(chunk_confidence_yhat,group_chunks,y_pred_classes.flatten(),y_pred.flatten(),model,filepath,flatten=True)



    return  chunk_confidence_yhat

# Define Fine-Tuned Model IDs (Replace with your actual model IDs)
fine_tuned_models = {
    "Python": "ft:gpt-4o-mini-2024-07-18:websec::B1W99cp2",
    "PHP": "ft:gpt-4o-mini-2024-07-18:websec::B6BkXN2L",
    "C++": "ft:gpt-4o-mini-2024-07-18:websec::B6B5B4Aw"
}

def send_to_llm(code_chunk, language):
    """
    Send flagged code to the correct fine-tuned LLM model based on language, requesting HTML-formatted output.
    """
    if language not in fine_tuned_models:
        return f"Error: No fine-tuned model available for {language}."

    model_id = fine_tuned_models[language]

    full_prompt = f"""
    ### Code Security Audit: {language}

    **Your Task:**  
    You are a cybersecurity expert specializing in {language} vulnerability detection.  
    Carefully analyze the entire code snippet provided below as a complete unit.  
    Determine **whether** the code contains any security vulnerabilities.  

    ---

    ### **Response Guidelines:**  
    - If **no vulnerabilities** are found, return an HTML-formatted security report confirming the code's safety.  
    - If **vulnerabilities exist**, provide a **detailed security assessment** in HTML format as described below.  
    - Ensure the response is **unbiased** and based only on the code's actual security status.

    ---

    ### **Response Format (Strictly HTML)**:

    #### **If the code is secure:**  
    ```html
    <div class="report">
        <h2>✅ No Vulnerabilities Found</h2>
        <p>The provided {language} code has been reviewed, and no security risks were detected.</p>

        <h3>🔍 Explanation of Security:</h3>
        <p>Explain why the code is considered secure.</p>
        <p>Highlight any best practices used that contribute to its security.</p>
    </div>
    ```

    #### **If the code contains vulnerabilities:**  
    ```html
    <div class="report">
        <h2>🔍 Vulnerability Analysis for {language} Code</h2>

        <h3>Vulnerable Lines:</h3>
        <textarea readonly style="width:100%; height:auto;">
        line X: &lt;code&gt;
        line Y: &lt;code&gt;
        </textarea>

        <h3>🛑 Explanation of Vulnerabilities:</h3>
        <p>Explain why each identified line is vulnerable.</p>
        <p>Describe how an attacker could exploit this vulnerability.</p>

        <h3>✅ Secure Code Fix:</h3>
        <textarea readonly style="width:100%; height:auto;">{language.lower()}
        // Corrected version of the code with secure coding practices
        </textarea>

        <h3>🔄 Explanation of Fix:</h3>
        <p>Clearly outline what changes were made and how the fix improves security.</p>
    </div>
    ```

    ---

    ### **Analyze the following {language} code and return an HTML-formatted security assessment:**  
    ```{language.lower()}
    {code_chunk}
    ```
    """

    try:
        response = client.chat.completions.create(
            model=model_id,
            messages=[{"role": "user", "content": full_prompt}]
        )

        return response.choices[0].message.content  # This will be the HTML output

    except client.error.OpenAIError as e:
        return f"Error: {str(e)}"


import re

def process_results(chunk_confidence_yhat, language):
    """
    Process flagged chunks and send them to the correct fine-tuned LLM.
    Generates an HTML report.
    """
    final_results = []
    html_output = "<html><body><h1> Code Security Report</h1>"

    for chunk_data in chunk_confidence_yhat:
        if chunk_data['yhat'] == 1 :
            print(f"Sending Chunk {chunk_data['chunk_id']} to {language} fine-tuned LLM (Flagged by {chunk_data['model']})...")
            llm_result = send_to_llm(chunk_data["chunk"], language)

            # ✅ Extract only the actual HTML content (remove ```html ... ```)
            extracted_html = re.sub(r"```html\n(.*?)\n```", r"\1", llm_result, flags=re.DOTALL)

            chunk_data["llm_analysis"] = extracted_html  # Store cleaned HTML
            final_results.append(chunk_data)

            # Append extracted HTML output (which is already formatted)
            html_output += f"<h2>Chunk {chunk_data['chunk_id']} (Flagged by {chunk_data['model']})</h2>"
            html_output += extracted_html  # Insert clean HTML

    html_output += "</body></html>"

    # Save HTML report
    html_filename = f"security_report_{language.lower()}.html"
    with open(html_filename, "w", encoding="utf-8") as file:
        file.write(html_output)

    print(f"✅ Security report saved as {html_filename}")
    return final_results if final_results else None

def generate_html_report(report_data, filename="vulnerability_report.html"):
    """
    Generate an HTML report using the LLM's pre-formatted HTML output.
    """
    html_content = """
    <!DOCTYPE html>
    <html lang="en">
    <head>
        <meta charset="UTF-8">
        <meta name="viewport" content="width=device-width, initial-scale=1.0">
        <title>Vulnerability Report</title>
        <style>
            body {
                font-family: 'Arial', sans-serif;
                margin: 0;
                padding: 0;
                background-color: #f4f4f4;
                color: #333;
            }
            .container {
                width: 90%;
                max-width: 1000px;
                margin: 20px auto;
                padding: 20px;
                background-color: white;
                border-radius: 8px;
                box-shadow: 0 4px 8px rgba(0, 0, 0, 0.2);
            }
            h1 {
                background: #d9534f;
                color: white;
                padding: 15px;
                text-align: center;
                border-radius: 8px 8px 0 0;
                margin: -20px -20px 20px -20px;
            }
            .report {
                padding: 15px;
                margin-bottom: 20px;
                background: #fff3f3;
                border-left: 5px solid #d9534f;
                border-radius: 5px;
            }
            .footer {
                text-align: center;
                margin-top: 20px;
                padding: 10px;
                background: #333;
                color: white;
                font-size: 14px;
                border-radius: 0 0 8px 8px;
            }
        </style>
    </head>
    <body>
        <div class="container">
            <h1>🚀 Code Vulnerability Analysis Report</h1>
    """

    for item in report_data:
        flagged_by = item.get('model', 'Unknown Model')
        llm_analysis = item.get("llm_analysis", "⚠ No analysis received from LLM.")

        html_content += f"""
        <div class="report">
            <h2>🆔 Chunk ID: {item['chunk_id']} (Flagged by {flagged_by})</h2>
            {llm_analysis}  <!-- The LLM's generated HTML is directly inserted -->
        </div>
        """

    html_content += """
            <div class="footer">
                Generated by SafeScan | Secure Coding Matters 🚀
            </div>
        </div>
    </body>
    </html>
    """

    with open(filename, "w", encoding="utf-8") as file:
        file.write(html_content)

    print(f"\n✅ HTML Report Generated: {filename}")
    return filename


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
        language_map = {"cpp": "C++", "python": "Python", "php": "PHP"}
        language = language_map.get(args.type)

        #check if the file match the type of the 
        if args.type == "cpp" and (args.file.endswith(".c") or args.file.endswith(".cpp")):
            print(f"Submitting C code from file: {args.file}")
            chunk_confidence_yhat = c_vul_detector(args.file)

        elif args.type == "python" and args.file.endswith(".py"):
            print(f"Submitting Python code from file: {args.file}")
            chunk_confidence_yhat = python_vul_detector(args.file)

        elif args.type == "php" and (args.file.endswith(".php") or args.file.endswith(".html")):

            print(f"Submitting PHP code from file: {args.file}")
            chunk_confidence_yhat = php_vul_detector(args.file)

        else:
            print("Contradicting file type. Please provide a valid file.")

        if chunk_confidence_yhat:
            
            for chunk_data in chunk_confidence_yhat:
                print(f"Chunk ID: {chunk_data['chunk_id']}")
                print(f"Predicted Class (yhat): {chunk_data['yhat']}")
                print(f"Confidence: {chunk_data['confidence']}")
                print(f"Model: {chunk_data['model']}")
                print("-----")
            
            # Send low-confidence results to the correct LLM model
            print(f"\nProcessing low-confidence predictions with {language} fine-tuned LLM...")
            final_results = process_results(chunk_confidence_yhat, language)


            # print LLM results
            print("\nLLM Analysis Results")
            for chunk_data in final_results:
                print(f"\nChunk ID: {chunk_data['chunk_id']}")
                print(f"Flagged by: {chunk_data['model']}")  # Print which model flagged it
                print(f"LLM Analysis: {chunk_data['llm_analysis']}")
                print("-----")
        
        if final_results:
            print("\nLLM Analysis Results")
            for chunk_data in final_results:
                print(f"\nChunk ID: {chunk_data['chunk_id']}")
                print(f"Flagged by: {chunk_data['model']}")  # Print which model flagged it

                # ✅ Use .get() to avoid KeyError
                analysis = chunk_data.get("llm_analysis", "⚠ No analysis received from LLM.")
                print(f"LLM Analysis: {analysis}")
                print("-----")


    else:
        print("Invalid usage. Use -help for more information.")

if __name__ == "__main__":
    main()
