import argparse
import os 
import re
import pandas as pd
from pprint import pprint
os.environ["TF_ENABLE_ONEDNN_OPTS"] = "0"
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


def populate_chunk_confidence_yhat(chunk_confidence_yhat,group_chunks,predictions,probabilities,model,filepath,flatten):

    #test github push
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
    models =["xformerBERT_python_model.pth" , "cnn_python_model_new.h5"]
    #models =["xformerBERT_python_model.pth" ]
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
            # Predicted class (0 or 1) by thresholding at 0.5
            #predictions = (probabilities > 0.5).long()
            #print("Predictions:", predictions)
            predictions = torch.argmax(probabilities, dim=1)

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
            y_pred_classes = (y_pred > 0.5).astype("int32")
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
    models = ["xformerBERT_php_model.pth","cnn_php_model_new.h5"]
    #models = ["xformerBERT_php_model.pth"]
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
            # Predicted class (0 or 1) by thresholding at 0.5
            #predictions = (probabilities > 0.5).long()
            #print("Predictions:", predictions)
            predictions = torch.argmax(probabilities, dim=1)

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
            y_pred_classes = (y_pred > 0.5).astype("int32")
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


    models = ["cnn_c++.h5"]
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
            y_pred_classes = (y_pred > 0.015).astype(int)  # Convert probabilities to class labels
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

    ** Your Task:**  
    You are a cybersecurity expert specializing in {language} vulnerability detection.  
    Your job is to analyze the following code and return a **detailed security assessment** in **HTML format**.  

    ---
    ** Response Format (Strictly HTML)**:

    ```html
    <div class="report">
        <h2>🔍 Vulnerability Analysis for {language} Code</h2>

        <h3>Vulnerable Lines:</h3>
        <pre><code>
        line X: &lt;code&gt;
        line Y: &lt;code&gt;
        </code></pre>

        <h3>🛑 Explanation of Vulnerabilities:</h3>
        <p>Explain why each identified line is vulnerable.</p>
        <p>Describe how an attacker could exploit this vulnerability.</p>

        <h3>✅ Secure Code Fix:</h3>
        <pre><code>{language.lower()}
        // Corrected version of the code with secure coding practices
        </code></pre>

        <h3>🔄 Explanation of Fix:</h3>
        <p>Clearly outline what changes were made and how the fix improves security.</p>
    </div>
    ```

    ---
    **Now analyze the following {language} code and return an HTML-formatted security assessment:**  

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


def process_results(chunk_confidence_yhat, language):
    """
    Process flagged chunks and send them to the correct fine-tuned LLM.
    Generates an HTML report.
    """
    final_results = []
    html_output = "<html><body><h1> Code Security Report</h1>"

    for chunk_data in chunk_confidence_yhat:
        if chunk_data["confidence"] < 0.7:
            print(f"Sending Chunk {chunk_data['chunk_id']} to {language} fine-tuned LLM (Flagged by {chunk_data['model']})...")
            llm_result = send_to_llm(chunk_data["chunk"], language)
            chunk_data["llm_analysis"] = llm_result  # Store LLM result
            final_results.append(chunk_data)

            # Append LLM output (which is already in HTML format)
            html_output += f"<h2>Chunk {chunk_data['chunk_id']} (Flagged by {chunk_data['model']})</h2>"
            html_output += llm_result

    html_output += "</body></html>"

    # Save HTML report
    html_filename = f"security_report_{language.lower()}.html"
    with open(html_filename, "w", encoding="utf-8") as file:
        file.write(html_output)

    print(f"Security report saved as {html_filename}")
    return final_results if final_results else None


def generate_html_report(report_data, filename="vulnerability_report.html"):
    """
    Generate an HTML report from the LLM analysis results.
    """
    html_content = """
    <!DOCTYPE html>
    <html lang="en">
    <head>
        <meta charset="UTF-8">
        <meta name="viewport" content="width=device-width, initial-scale=1.0">
        <title>Vulnerability Report</title>
        <style>
            body { font-family: Arial, sans-serif; margin: 20px; padding: 20px; }
            .report { border: 1px solid #ccc; padding: 15px; margin-bottom: 20px; background-color: #f9f9f9; }
            h2 { color: #d9534f; }
            h3 { color: #5bc0de; }
            pre { background: #222; color: #eee; padding: 10px; border-radius: 5px; overflow-x: auto; }
        </style>
    </head>
    <body>
        <h1>🔍 Code Vulnerability Analysis Report</h1>
    """

    for item in report_data:
        html_content += f"""
        <div class="report">
            <h2>Chunk ID: {item['chunk_id']} (Flagged by {item['model']})</h2>
            {item['llm_html_analysis']}  <!-- GPT response in HTML format -->
        </div>
        """

    html_content += """
    </body>
    </html>
    """

    with open(filename, "w", encoding="utf-8") as file:
        file.write(html_content)

    print(f"\n HTML Report Generated: {filename}")
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
