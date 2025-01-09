import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, List
from transformers import RobertaModel, AutoModelForSeq2SeqLM
from .geom import SchNetEncoder
from .graph import GNN_graphpred


class UniEncoderAttention(nn.Module):
    def __init__(
        self,
        joint_embedding_dim: int,
        smiles_model_name: Optional[str],
        text_model_name: Optional[str],
        gnn_model_name: Optional[str],
        geom_model_name: Optional[str],
        modality_list: List[str],
        output_attention_weights: bool = True,
        freeze_encoder: bool = False,
        num_heads: int = 8,
        ff_dim: Optional[int] = None,
        dropout: float = 0.1
    ):
        super().__init__()
        self.modality_list = modality_list
        self.joint_embedding_dim = joint_embedding_dim
        self.output_attention_weights = output_attention_weights
        self.freeze_encoder = freeze_encoder
        # Initialize encoders
        self.encoders = nn.ModuleDict({
            modality: EncoderModule(
                modality=modality,
                joint_embedding_dim=joint_embedding_dim,
                freeze_encoder=freeze_encoder,
                smiles_model_name=smiles_model_name,
                text_model_name=text_model_name,
                gnn_model_name=gnn_model_name,
                geom_model_name=geom_model_name,
            )
            for modality in modality_list
        })

        ff_dim = ff_dim or (joint_embedding_dim * 2)

        # Fusion module
        self.fusion_module = FusionModule(
            joint_embedding_dim=joint_embedding_dim,
            num_heads=num_heads,
            ff_dim=ff_dim,
            dropout=dropout
        )

        # Final MLP
        self.mlp = nn.Sequential(
            nn.Linear(joint_embedding_dim, 128),
            nn.ReLU(),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, 1)
        )

    def forward(self, data):
        # 对每种模态分别编码
        embeddings = []
        for modality in self.modality_list:
            embeddings.append(self.encoders[modality](data))
        # [batch_size, num_modalities, joint_embedding_dim]
        embeddings = torch.stack(embeddings, dim=1)

        # 融合
        fused_output, modality_attention = self.fusion_module(embeddings)
        self.unimodal_embedding = fused_output  # Store fused features

        self.attention_visual_weights = modality_attention

        output = self.mlp(fused_output)
        return output, embeddings


class EncoderModule(nn.Module):
    def __init__(
        self,
        modality: str,
        joint_embedding_dim: int,
        freeze_encoder: bool,
        smiles_model_name: Optional[str] = None,
        text_model_name: Optional[str] = None,
        gnn_model_name: Optional[str] = None,
        geom_model_name: Optional[str] = None
    ):
        super().__init__()
        self.modality = modality
        
        encoder, input_dim = self._initialize_encoder(
            modality=modality,
            smiles_model_name=smiles_model_name,
            text_model_name=text_model_name,
            gnn_model_name=gnn_model_name,
            geom_model_name=geom_model_name
        )
        self.encoder = encoder
        self.norm = nn.LayerNorm(input_dim) if encoder else None
        self.projection = self._create_projection(input_dim, joint_embedding_dim)

        if encoder and freeze_encoder:
            for param in self.encoder.parameters():
                param.requires_grad = False

    def _initialize_encoder(
        self,
        modality: str,
        smiles_model_name: Optional[str],
        text_model_name: Optional[str],
        gnn_model_name: Optional[str],
        geom_model_name: Optional[str]
    ):
        if modality == 'smiles':
            encoder = RobertaModel.from_pretrained(smiles_model_name)
            input_dim = encoder.config.hidden_size
        elif modality == 'text':
            encoder = AutoModelForSeq2SeqLM.from_pretrained(text_model_name).encoder
            input_dim = encoder.config.hidden_size
        elif modality == 'geom':
            encoder = SchNetEncoder(
                load_from_pretrain=geom_model_name,
                cutoff=10,
                max_num_neighbors=32,
                readout='mean'
            )
            input_dim = encoder.hidden_channels
        elif modality == 'graph':
            encoder = GNN_graphpred(
                num_layer=5,
                emb_dim=300,
                num_tasks=1,
                JK="last",
                drop_ratio=0,
                gnn_type="gin",
                graph_pooling="mean"
            )
            encoder.from_pretrained(model_file=gnn_model_name)
            input_dim = encoder.emb_dim
        elif modality == 'fp':
            # 如果是指纹，只有一个简单的全连接映射即可
            encoder = None
            input_dim = 1024
        else:
            raise ValueError(f"Unsupported modality: {modality}")
        return encoder, input_dim

    def _create_projection(self, input_dim: int, joint_embedding_dim: int):
        return nn.Sequential(
            nn.Linear(input_dim, joint_embedding_dim),
            nn.BatchNorm1d(joint_embedding_dim),
            nn.ReLU()
        )

    def _encode_text_like(self, data, input_ids_attr: str, attention_mask_attr: str):
        input_ids = getattr(data, input_ids_attr).to(self.encoder.device)
        attention_mask = getattr(data, attention_mask_attr).to(self.encoder.device)
        features = self.encoder(input_ids, attention_mask=attention_mask).last_hidden_state
        return self.projection(self.norm(features[:, 0, :]))


    def forward(self, data):
        if self.modality == 'smiles':
            return self._encode_text_like(data, 'input_ids_smiles', 'attention_mask_smiles')

        elif self.modality == 'text':
            return self._encode_text_like(data, 'input_ids_text', 'attention_mask_text')

        elif self.modality == 'graph':
            features, _ = self.encoder(data.x, data.edge_index, data.edge_attr, data.batch)
            return self.projection(self.norm(features))

        elif self.modality == 'geom':
            features = self.encoder(data.x3d, data.pos3d, data.batch3d)
            return self.projection(self.norm(features))

        elif self.modality == 'fp':
            return self.projection(data.fp)

        else:
            raise ValueError(f"Unsupported modality: {self.modality}")


class AttentionPooling(nn.Module):
    def __init__(self, joint_embedding_dim: int):
        super().__init__()
        self.attention = nn.Linear(joint_embedding_dim, 1)

    def forward(self, embeddings: torch.Tensor):
        """
        Args:
            embeddings: [batch_size, num_modalities, joint_embedding_dim]
        Returns:
            output: [batch_size, joint_embedding_dim]
            weights: [batch_size, num_modalities]
        """
        # 对每个模态的 embedding 做打分
        scores = self.attention(embeddings).squeeze(-1)  # [batch_size, num_modalities]
        weights = F.softmax(scores, dim=1)               # [batch_size, num_modalities]
        output = torch.sum(embeddings * weights.unsqueeze(-1), dim=1)  # [batch_size, joint_embedding_dim]
        return output, weights


class FusionModule(nn.Module):
    def __init__(
        self,
        joint_embedding_dim: int,
        num_heads: int,
        ff_dim: int,
        dropout: float = 0.1
    ):
        super().__init__()
        self.multihead_attn = nn.MultiheadAttention(
            embed_dim=joint_embedding_dim, 
            num_heads=num_heads, 
            dropout=dropout
        )
        self.layer_norm1 = nn.LayerNorm(joint_embedding_dim)
        self.feed_forward = nn.Sequential(
            nn.Linear(joint_embedding_dim, ff_dim),
            nn.ReLU(),
            nn.Linear(ff_dim, joint_embedding_dim)
        )
        self.layer_norm2 = nn.LayerNorm(joint_embedding_dim)
        self.dropout = nn.Dropout(dropout)
        self.attention_pooling = AttentionPooling(joint_embedding_dim)

    def forward(self, embeddings: torch.Tensor):
        """
        Args:
            embeddings: [batch_size, num_modalities, joint_embedding_dim]
        Returns:
            output: [batch_size, joint_embedding_dim]
            attention_weights: [batch_size, num_modalities]
        """
        # 先把 batch_size 放到第二个维度，以符合 MultiheadAttention 的输入格式
        # [num_modalities, batch_size, joint_embedding_dim]
        embeddings = embeddings.permute(1, 0, 2)

        attn_output, attn_weights = self.multihead_attn(
            embeddings, embeddings, embeddings
        )
        attn_output = attn_output.permute(1, 0, 2)  # [batch_size, num_modalities, joint_embedding_dim]

        # Add & Norm (第一次)
        fused_embedding = embeddings.permute(1, 0, 2) + attn_output
        fused_embedding = self.layer_norm1(fused_embedding)

        # 前馈网络
        ff_output = self.feed_forward(fused_embedding)
        ff_output = self.dropout(ff_output)

        # Add & Norm (第二次)
        fused_embedding = fused_embedding + ff_output
        fused_embedding = self.layer_norm2(fused_embedding)

        # 跨模态注意力池化
        output, attention_weights = self.attention_pooling(fused_embedding)

        return output, attention_weights
