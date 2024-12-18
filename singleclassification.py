{\rtf1\ansi\ansicpg1252\cocoartf2761
\cocoatextscaling0\cocoaplatform0{\fonttbl\f0\fswiss\fcharset0 Helvetica;}
{\colortbl;\red255\green255\blue255;}
{\*\expandedcolortbl;;}
\paperw11900\paperh16840\margl1440\margr1440\vieww11820\viewh9820\viewkind0
\pard\tx720\tx1440\tx2160\tx2880\tx3600\tx4320\tx5040\tx5760\tx6480\tx7200\tx7920\tx8640\pardirnatural\partightenfactor0

\f0\fs24 \cf0 import pandas as pd\
\
import torch\
\
inputdata = pd.read_csv("icd_ipdostable.csv")\
textparts = inputdata[["record_id", "d_ipdos_all","hist"]]\
textparts = textparts.copy()\
textparts = textparts[["record_id", "d_ipdos_all","hist"]]\
textparts.to_csv('ipdos.csv')\
\
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig, pipeline\
\
import os\
\
import setproctitle\
 \
# Change the name of the process as it appears in system utilities:\
setproctitle.setproctitle('dey1 - sleepdisorder')\
 \
# Ensure that GPUs are enumerated by their PCI bus ID:\
os.environ['CUDA_DEVICE_ORDER']='PCI_BUS_ID'\
 \
# Restrict cuda to access only certain devices, e.g. '0', '1', or '0,1' as entered in the MLMP calendar:\
os.environ['CUDA_VISIBLE_DEVICES'] = '0' # restart Kernel to apply changes\
 \
# Specify the model name\
model_name = "mistralai/Mixtral-8x7B-Instruct-v0.1"\
bnb_config = BitsAndBytesConfig(load_in_4bit = True, bnb_4bit_quant_type="nf4", bnb_4bit_use_double_quant=True)\
\
model = AutoModelForCausalLM.from_pretrained(model_name,quantization_config=bnb_config)\
tokenizer = AutoTokenizer.from_pretrained(model_name)\
# Load the tokenizer and model with trust_remote_code enabled\
\
# Set device\
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")\
model.to(device)\
\
print("Model and tokenizer loaded successfully.")\
\
# Ensure padding token is set\
if tokenizer.pad_token is None:\
    tokenizer.pad_token = tokenizer.eos_token  # Use EOS token as the PAD token\
    \
# Function to split text into overlapping chunks\
def split_text(text, max_chunk_size=512, overlap=50):\
    """Split the text into overlapping chunks."""\
    words = text.split()\
    chunks = []\
    for i in range(0, len(words), max_chunk_size - overlap):\
        chunk = " ".join(words[i:i + max_chunk_size])\
        chunks.append(chunk)\
    return chunks\
    \
# Function to check the token length of text\
def check_token_length(text, max_tokens=2000):\
    """Check if the text exceeds the maximum token limit."""\
    tokenized = tokenizer(text, return_tensors="pt", truncation=False)["input_ids"]\
    return len(tokenized[0]) > max_tokens\
    \
# Function to generate prompts with token length handling\
def generate_prompts(medical_records, categories, descriptions, examples, max_chunk_size=512, overlap=50):\
    """Generate prompts for all combinations of medical records and ICD codes/descriptions."""\
    prompts = []\
    record_indices = []\
    categories_list = []\
\
    for i, record in enumerate(medical_records):\
        # Check the token length of the full record\
        if check_token_length(record):\
            # Split the text into chunks if it exceeds the token limit\
            chunks = split_text(record, max_chunk_size=max_chunk_size, overlap=overlap)\
        else:\
            # Use the full text as a single chunk\
            chunks = [record]\
\
        for chunk in chunks:\
            for category, description, example in zip(categories, descriptions, examples):\
                prompt = (\
                        f"Does the following medical report mention a disease that falls into the category '\{category\}'? "\
                        f"Examples include: \{example\}.\\n\\n"\
                        f"Medical Report: \{chunk\}. \\n\\n"\
                        f"Answer strictly with True or False only: "\
                )\
                prompts.append(prompt)\
                record_indices.append(i)\
                categories_list.append(category)\
\
    return prompts, record_indices, categories_list\
\
# Query the model (single input at a time)\
def query_model_single(prompts):\
    """\
    Query the LLM one prompt at a time and return binary responses (True/False).\
    """\
    results = []\
    for prompt in prompts:\
        inputs = tokenizer(prompt, return_tensors="pt", padding=True, truncation=True).to(device)\
        outputs = model.generate(**inputs, max_new_tokens=100, temperature=0.0, do_sample=False)\
        response = tokenizer.decode(outputs[0], skip_special_tokens=True).strip().lower()\
        print(f"Raw Response: \{response\}")\
        # Extract only the generated response after the prompt\
        response = response.replace("answer strictly with true or false only:", "").strip()\
\
        if "true" in response.lower():\
            response = "true"\
        elif "false" in response.lower():\
            response = "false"\
        else:\
            response = "unclear"  # Handle unexpected responses\
\
        results.append(response)\
\
        # Debugging: Print raw response for each prompt\
        print(f"Prompt: \{prompt\}")\
        print(f"Response: \{response\}\\n")\
\
    # Validate that results are binary responses (True/False)\
    binary_results = [res in ("true", "false") for res in results]\
    print("Are results binary (True/False)?", binary_results)\
\
    return results\
\
\
# Process medical records with token length handling\
def process_medical_records(medical_text_csv, prompt_csv, output_csv):\
    """Process medical records one prompt at a time for ICD classification."""\
    # Load medical text and prompt data\
    medical_data = pd.read_csv(medical_text_csv)\
    prompt_data = pd.read_csv(prompt_csv, delimiter=';')  # Assuming semicolon-separated file\
\
    # Clean text by removing brackets\
    medical_data['cleaned_text'] = medical_data['d_ipdos_all'].str.replace(r"[()\\[\\]\{\}]", " ", regex=True)\
\
    # Optionally limit the data for testing\
    #slice_start = 50  # Starting row index\
    #slice_end = 100   # Ending row index\
    chunk = medical_data.iloc[295:395].reset_index(drop=True)\
    medical_data = chunk\
\
    # Validate required columns\
    if "d_ipdos_all" not in medical_data.columns:\
        raise ValueError("The medical text CSV must contain a 'd_ipdos_all' column.")\
    if not all(col in prompt_data.columns for col in ["Category", "Definition for the prompt", "Examples"]):\
        raise ValueError("The prompt CSV must contain 'Category', 'Definition for the prompt', and 'Examples' columns.")\
\
    # Extract relevant columns\
    medical_records = medical_data["cleaned_text"]\
    categories = prompt_data["Category"]\
    descriptions = prompt_data["Definition for the prompt"]\
    examples = prompt_data["Examples"]\
\
    # Prepare one-hot encoding columns for all unique ICD codes\
    for category in categories:\
        medical_data[f"ICD_\{category\}"] = 0  # Initialize all as False\
        medical_data[f"ICD_\{category\}"] = medical_data[f"ICD_\{category\}"].astype(int)  # Cast to integer\
\
    # Generate prompts with token length handling\
    prompts, record_indices, categories_list = generate_prompts(\
        medical_records=medical_records,\
        categories=categories,\
        descriptions=descriptions,\
        examples=examples,\
        max_chunk_size=512,\
        overlap=50\
    )\
\
    # Query the model (one prompt at a time)\
    responses = query_model_single(prompts)\
\
    # Map responses back to the DataFrame\
    for idx, response in enumerate(responses):\
        is_mentioned = response.lower() == "true"\
        record_index = record_indices[idx]\
        category = categories_list[idx]\
\
        # Debugging: Print mapping details\
        print(f"Mapping response: Record Index=\{record_index\}, Category=\{category\}, Response=\{response\}")\
\
        # Update the DataFrame only if the result is True\
        if is_mentioned:\
           medical_data.iloc[record_index, medical_data.columns.get_loc(f"ICD_\{category\}")] = 1\
\
    # Save results to a new CSV\
    medical_data.to_csv(output_csv, index=False)\
    print(f"Processed data saved to \{output_csv\}")\
\
\
# Example usage\
medical_text_csv = "ipdos.csv"  # Medical text file\
prompt_csv = "dieseasmini.csv"  # Prompt file\
output_csv = "classified_records.csv"  # Output file\
process_medical_records(medical_text_csv, prompt_csv, output_csv)\
}