import pandas as pd
from transformers import AutoTokenizer, AutoModel
import torch
import numpy as np
from tqdm import tqdm

# List of specific files to process
files_to_process = ["HarryPotter1.txt", "LOTR1.txt"]

# Load the model and tokenizer
tokenizer = AutoTokenizer.from_pretrained("AIDA-UPM/star")  # Assuming the same model as in the paper
model = AutoModel.from_pretrained("AIDA-UPM/star")

# Function to tokenize and process each chunk
def process_chunk(chunk_text):
    inputs = tokenizer(chunk_text, return_tensors="pt", truncation=True, padding=True, max_length=512)
    outputs = model(**inputs)
    return outputs.last_hidden_state

# Function to process chunks and store embeddings in the CSV
def process_chunks_and_store_embeddings(file_path):
    df = pd.read_csv(file_path)
    
    embeddings_list = []
    
    # Initialize the tqdm progress bar
    for i, row in tqdm(df.iterrows(), total=len(df), desc=f"Processing {file_path}"):
        chunk_text = row['chunk']
        embeddings = process_chunk(chunk_text)
        
        # Convert embeddings to a format that can be stored in a CSV (as a string)
        embeddings_np = embeddings.cpu().detach().numpy()
        embeddings_list.append(embeddings_np.tolist())  # Store as list to convert to JSON later
    
    # Add the embeddings as a new column in the dataframe
    df['embedding'] = embeddings_list
    
    # Save the dataframe with embeddings to a new CSV
    df.to_csv(f"{file_path}_with_embeddings.csv", index=False)
    print(f"Saved embeddings for {file_path} into {file_path}_with_embeddings.csv")

# Process chunks for each file and store embeddings in CSV
for filename in files_to_process:
    csv_file = f"{filename}_chunks.csv"
    process_chunks_and_store_embeddings(csv_file)
