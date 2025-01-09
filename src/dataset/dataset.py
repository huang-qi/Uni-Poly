import os
import json
import torch
import pandas as pd
from rdkit import Chem
from tqdm import tqdm
from torch_geometric.data import Dataset
from .geom_data import mol2coords
from .graph_data import mol_to_graph_data_obj_simple
from transformers import AutoTokenizer
from rdkit.Chem import rdFingerprintGenerator
import rdkit.Chem as Chem
from rdkit.Chem import AllChem
import numpy as np
from torch_geometric.data import Data

class UniDataset(Dataset):
    def __init__(self, root, dataset, smiles_model_name, text_model_name, transform=None, pre_transform=None):
        self.dataset = dataset
        self.root = root
        self.transform = transform
        self.pre_transform = pre_transform
        self.text_dict = json.load(open('./data/smiles_text_dict.json', 'r'))
        self.data_list = []
        self.smiles_tokenizer = AutoTokenizer.from_pretrained(smiles_model_name)
        self.text_tokenizer = AutoTokenizer.from_pretrained(text_model_name)
        
        # Define the path for the processed dataset
        processed_dir = os.path.join(self.root, 'processed')
        os.makedirs(processed_dir, exist_ok=True)
        self.processed_file = os.path.join(processed_dir, f"{self.dataset}.pt")
        
        if os.path.exists(self.processed_file):
            # If processed dataset exists, load it
            self.data_list = torch.load(self.processed_file)
            print(f"loaded {self.processed_file} with {len(self.data_list)} samples")  # Loaded the processed dataset from ...
            
        else:
            # If processed dataset doesn't exist, process and save it
            # Get maximum token length
            csv_path = f"{self.root}/raw/{self.dataset}.csv"
            self.max_length_smiles, self.max_length_text = self.get_max_token_length(csv_path, self.smiles_tokenizer, self.text_tokenizer,self.text_dict)
            
            self.process()
            torch.save(self.data_list, self.processed_file)
            print(f"processed and saved {self.processed_file} with {len(self.data_list)} samples")  # Processed and saved the dataset to ...

    def __len__(self):
        return len(self.data_list)

    def __getitem__(self, idx):
        return self.data_list[idx]


    def get_max_token_length(self, csv_path, smiles_tokenizer, text_tokenizer,text_dict):
        df = pd.read_csv(csv_path)
        max_smiles_length = 0
        max_text_length = 0
        
        for _, row in tqdm(df.iterrows(), total=len(df), desc="Checking token lengths"):
            smiles = row[0]
            
            # Tokenize SMILES
            smiles_tokens = self.smiles_tokenizer.encode(smiles)
            max_smiles_length = max(max_smiles_length, len(smiles_tokens))
            
            # Tokenize text input
            input_text = self.text_dict[smiles] if smiles in self.text_dict else ''
            text_tokens = self.text_tokenizer.encode(input_text)
            max_text_length = max(max_text_length, len(text_tokens))
        
        print(f"Max SMILES token length: {max_smiles_length}")
        print(f"Max text token length: {max_text_length}")
        
        return max_smiles_length, max_text_length


    def process(self):
        csv_path = f"{self.root}/raw/{self.dataset}.csv"
        df = pd.read_csv(csv_path)
        mfpgen = rdFingerprintGenerator.GetMorganGenerator(radius=2,fpSize=1024)
        for i, row in tqdm(df.iterrows(), total=len(df), desc="Processing dataset"):
            smiles = row[0]
            property = row[1]
            mol = Chem.MolFromSmiles(smiles)
            if mol is None:
                continue
            for atom in mol.GetAtoms():
                if atom.GetAtomicNum() == 0:
                    atom.SetAtomicNum(1)
            
            data = mol_to_graph_data_obj_simple(mol)
            
            # Labels
            data.y = torch.tensor([property], dtype=torch.float)
            data.smiles = smiles

            # Tokenize with dynamic max_length
            tokenizer_output = self.smiles_tokenizer(
                smiles,
                return_tensors='pt',
                max_length=self.max_length_smiles+5,
                padding='max_length',
                truncation=True
            )
            data.input_ids_smiles = tokenizer_output.input_ids
            data.attention_mask_smiles = tokenizer_output.attention_mask

            data.text = self.text_dict[smiles] if smiles in self.text_dict else ''
            tokenizer_output = self.text_tokenizer(
                data.text,
                return_tensors='pt',
                max_length=self.max_length_text+5,
                padding='max_length',
                truncation=True
            )
            data.input_ids_text = tokenizer_output.input_ids
            data.attention_mask_text = tokenizer_output.attention_mask
            data.fp = torch.tensor(mfpgen.GetFingerprint(mol), dtype=torch.float).unsqueeze(0)
            try:
                mol = Chem.MolFromSmiles(smiles)
                data.pos,data.z = mol2coords(mol)
            except Exception as e:
                print(e)
                print(f"Failed to generate 3D coordinates for {smiles}")
                continue
            self.data_list.append(data)

        # Explanation: Printing the total number of processed samples
        print("Dataset processed. Total samples:", len(self.data_list))
