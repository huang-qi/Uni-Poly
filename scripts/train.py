import os
import argparse
import warnings
import torch
import torch.nn as nn
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

from src.dataset import UniDataset
from src.modules import UniEncoderAttention
from src.utils import scale_targets, train_and_evaluate, get_data_loader


def parse_arguments():
    parser = argparse.ArgumentParser(description="Train UniEncoderAttention Model")
    parser.add_argument(
        '--tasks',
        nargs='+',
        default=['tg', 'er', 'de', 'td', 'tm', 'iv', 'bc'],
        help="List of tasks to train on. Example: --tasks tg er de"
    )
    parser.add_argument(
        '--model_name',
        type=str,
        default='UniEncoderAttention',
        help="Name of the model."
    )
    parser.add_argument(
        '--modalities',
        nargs='+',
        default=['smiles', 'text', 'graph', 'fp', 'geom'],
        help="List of model modalities. Example: --modalities smiles text"
    )
    parser.add_argument(
        '--freeze_encoder',
        action='store_true',
        help="Freeze encoders weights if set."
    )
    parser.add_argument(
        '--pretrained_model_path',
        type=str,
        default=None,
        help="Path to the pretrained model."
    )
    parser.add_argument(
        '--epochs',
        type=int,
        default=100,
        help="Number of training epochs."
    )
    parser.add_argument(
        '--patience',
        type=int,
        default=10,
        help="Early stopping patience."
    )
    parser.add_argument(
        '--results_dir',
        type=str,
        default='./results/results.csv',
        help="Directory to save results CSV."
    )
    parser.add_argument(
        '--models_dir',
        type=str,
        default='./saved_models',
        help="Directory to save trained models."
    )
    parser.add_argument(
        '--batch_size',
        type=int,
        default=32,
        help="Batch size for training."
    )
    return parser.parse_args()


def main():
    args = parse_arguments()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # Ignore warnings
    warnings.filterwarnings("ignore")
    
    pre_trained_model_dict = {
        'smiles_model_name': "seyonec/PubChem10M_SMILES_BPE_450k",
        'text_model_name': "GT4SD/multitask-text-and-chemistry-t5-base-augm",
        'gnn_model_name': "./pretrained_models/Mole-BERT.pth",
        'geom_model_name': "./pretrained_models/schnet_qm9_heat_capacity_model.pth"
    }
    
    result_output_dir = args.results_dir
    model_output_dir = args.models_dir
    
    task_list = args.tasks
    dataset_name_list = ['smi_' + task for task in task_list]
    dataset_list = [
        UniDataset(
            root='./data',
            dataset=dataset_name,
            smiles_model_name=pre_trained_model_dict['smiles_model_name'],
            text_model_name=pre_trained_model_dict['text_model_name']
        )
        for dataset_name in dataset_name_list
    ]
    
    model_modality_list = args.modalities
    freeze_encoder = args.freeze_encoder
    pretrained_model_path = args.pretrained_model_path
    epochs = args.epochs
    patience = args.patience
    
    results = []
    
    for task in task_list:
        print(f"\nStarting task: {task}")
        dataset = dataset_list[task_list.index(task)]
        scaler = scale_targets(dataset, task)

        # Split dataset into train, validation and test sets
        train_val_indices, test_indices = train_test_split(range(len(dataset)), test_size=0.2, random_state=42)
        train_indices, val_indices = train_test_split(train_val_indices, test_size=0.1, random_state=42)

        # Get data loaders
        train_loader = get_data_loader(dataset, indices=train_indices, batch_size=args.batch_size, shuffle=True)
        val_loader = get_data_loader(dataset, indices=val_indices, batch_size=args.batch_size, shuffle=False)
        test_loader = get_data_loader(dataset, indices=test_indices, batch_size=args.batch_size, shuffle=False)

        # Initialize model
        model = UniEncoderAttention(
            joint_embedding_dim=256,
            smiles_model_name=pre_trained_model_dict['smiles_model_name'],
            text_model_name=pre_trained_model_dict['text_model_name'],
            gnn_model_name=pre_trained_model_dict['gnn_model_name'],
            geom_model_name=pre_trained_model_dict['geom_model_name'],
            modality_list=model_modality_list,
            freeze_encoder=freeze_encoder
        )
        
        if pretrained_model_path:
            model.load_state_dict(torch.load(pretrained_model_path))
            print(f"Loaded pretrained model from {pretrained_model_path}")
        
        model.to(device)
        print("Using GPU for model training." if torch.cuda.is_available() else "Using CPU for model training.")

        # Train and evaluate
        metrics = train_and_evaluate(
            model, scaler, train_loader, val_loader, test_loader,
            device, num_epochs=epochs, patience=patience
        )
        
        print(
            f"Test R2: {metrics['test_r2']:.4f}, "
            f"MAE: {metrics['test_mae']:.4f}, RMSE: {metrics['test_rmse']:.4f}"
        )

        # Save best model
        attention_weights = model.attention_visual_weights.detach().cpu().numpy()
        os.makedirs(os.path.join(model_output_dir, task), exist_ok=True)
        torch.save(model.state_dict(), os.path.join(model_output_dir, f'{task}/{args.model_name}_best.pth'))
        print(f"Model saved with R2: {metrics['test_r2']:.4f}")

        # Save results
        result = {
            'task': task,
            'model_name': args.model_name,
            'model_modality_list': model_modality_list,
            'avg_test_r2': float(f"{metrics['test_r2']:.4g}"),
            'std_test_r2': 0.0,
            'avg_test_mae': float(f"{metrics['test_mae']:.4g}"),
            'std_test_mae': 0.0,
            'avg_test_rmse': float(f"{metrics['test_rmse']:.4g}"),
            'std_test_rmse': 0.0,
            'attention_weights': attention_weights
        }
    
        results.append(result)

        # Save to CSV
        os.makedirs(os.path.dirname(result_output_dir), exist_ok=True)
        results_df = pd.DataFrame(results)
        results_df.to_csv(
            result_output_dir,
            mode='a',
            header=not os.path.exists(result_output_dir),
            index=False
        )
        print(f"Results have been appended to '{result_output_dir}'.")
        

if __name__ == "__main__":
    main()
