import os
import argparse
import warnings
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import matplotlib.pyplot as plt
import json

from src.dataset import UniDataset
from src.modules import UniEncoderAttention
from src.utils import get_data_loader, compute_contrastive_loss

def parse_arguments():
    parser = argparse.ArgumentParser(description="Pretrain UniEncoderAttention Model")
    parser.add_argument(
        '--modalities',
        nargs='+',
        default=['smiles', 'text', 'graph', 'fp', 'geom'],
        help="List of modalities to use. Example: --modalities smiles text"
    )
    parser.add_argument(
        '--smiles_model_name',
        type=str,
        default="seyonec/PubChem10M_SMILES_BPE_450k",
        help="Pretrained model name or path for SMILES"
    )
    parser.add_argument(
        '--text_model_name',
        type=str,
        default="GT4SD/multitask-text-and-chemistry-t5-base-augm",
        help="Pretrained model name or path for Text"
    )
    parser.add_argument(
        '--gnn_model_name',
        type=str,
        default="./pretrained_models/Mole-BERT.pth",
        help="Pretrained GNN model path"
    )
    parser.add_argument(
        '--geom_model_name',
        type=str,
        default="./pretrained_models/schnet_qm9_heat_capacity_model.pth",
        help="Pretrained Geometry model path"
    )
    parser.add_argument(
        '--freeze_encoder',
        action='store_true',
        help="If set, freeze the pretrained model weights."
    )
    parser.add_argument(
        '--dataset_name',
        type=str,
        default='smi_all', 
        help="Name of the dataset for pretraining (unlabeled or labeled, but labels unused here)"
    )
    parser.add_argument(
        '--root',
        type=str,
        default='./data',
        help="Root directory of the dataset."
    )
    parser.add_argument(
        '--batch_size',
        type=int,
        default=32,
        help="Batch size for pretraining."
    )
    parser.add_argument(
        '--epochs',
        type=int,
        default=10,
        help="Number of pretraining epochs."
    )
    parser.add_argument(
        '--lr',
        type=float,
        default=1e-3,
        help="Learning rate for optimizer."
    )
    parser.add_argument(
        '--temperature',
        type=float,
        default=0.07,
        help="Temperature parameter for contrastive loss."
    )
    parser.add_argument(
        '--save_path',
        type=str,
        default='./pretrained_models/saved_pretrained_model.pth',
        help="Path to save the pretrained model."
    )
    return parser.parse_args()

def main():
    args = parse_arguments()
    
    # Get all available GPUs
    if torch.cuda.is_available():
        n_gpus = torch.cuda.device_count()
        print(f"Found {n_gpus} GPUs available")
        device = torch.device("cuda")
    else:
        print("No GPU available, using CPU")
        device = torch.device("cpu")

    # Ignore warnings
    warnings.filterwarnings("ignore")

    # Build dataset and DataLoader (using the same dataset for unsupervised training, only using input features)
    dataset = UniDataset(
        root=args.root,
        dataset=args.dataset_name,
        smiles_model_name=args.smiles_model_name,
        text_model_name=args.text_model_name
    )
    indices = np.arange(len(dataset))
    dataloader = get_data_loader(dataset, indices=indices, batch_size=args.batch_size, shuffle=False)

    # Initialize model
    model = UniEncoderAttention(
        joint_embedding_dim=256,
        smiles_model_name=args.smiles_model_name,
        text_model_name=args.text_model_name,
        gnn_model_name=args.gnn_model_name,
        geom_model_name=args.geom_model_name,
        modality_list=args.modalities,
        freeze_encoder=args.freeze_encoder
    )
    
    # Use all available GPUs for data parallel training
    if torch.cuda.device_count() > 1:
        print(f"Using {torch.cuda.device_count()} GPUs for data parallel training")
        model = nn.DataParallel(model)
    model = model.to(device)

    optimizer = optim.Adam(model.parameters(), lr=args.lr)

    # Create directory for saving loss curves and data
    os.makedirs('./plots/pretrain', exist_ok=True)
    
    # Record loss for each epoch
    losses = []
    
    model.train()
    for epoch in range(args.epochs):
        epoch_loss = 0.0
        for data in dataloader:
            data = data.to(device)
            print(data.x)
            print(data.edge_index)
            print(data.edge_attr)
            print(data.batch)
            print(data.smiles)
            print(data.input_ids_smiles)
            print(data.attention_mask_smiles)
            print(data.text)
            print(data.input_ids_text)
            print(data.attention_mask_text)
            print(data.fp)
            optimizer.zero_grad()
            _, embeddings = model(data)  # embeddings: [batch_size, num_modalities, embedding_dim]
            loss = compute_contrastive_loss(embeddings, temperature=args.temperature)
            loss.backward()
            optimizer.step()
            epoch_loss += loss.item()
        avg_loss = epoch_loss / len(dataloader)
        losses.append(avg_loss)
        print(f"Epoch [{epoch+1}/{args.epochs}] Contrastive Loss: {avg_loss:.4f}")

    # Save original loss data
    loss_data = {
        'epochs': list(range(1, args.epochs + 1)),
        'losses': losses
    }
    with open('./plots/pretrain/loss_data.json', 'w') as f:
        json.dump(loss_data, f, indent=4)
    print("Loss data saved at ./plots/pretrain/loss_data.json")

    # Plot loss curve
    plt.figure(figsize=(10, 6))
    plt.plot(range(1, args.epochs + 1), losses, marker='o')
    plt.title('Pretraining Loss Curve')
    plt.xlabel('Epoch')
    plt.ylabel('Contrastive Loss')
    plt.grid(True)
    plt.savefig('./plots/pretrain/loss_curve.png')
    plt.close()
    print("Loss curve saved at ./plots/pretrain/loss_curve.png")

    # If using DataParallel, need to handle module prefix when saving
    if isinstance(model, nn.DataParallel):
        torch.save(model.module.state_dict(), args.save_path)
    else:
        torch.save(model.state_dict(), args.save_path)
    print(f"Pretrained model saved at {args.save_path}")

if __name__ == "__main__":
    main()
