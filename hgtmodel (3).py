from __future__ import annotations
import torch
import torch.nn as nn
from torch_geometric.nn import HGTConv, global_mean_pool, global_max_pool, global_add_pool, TopKPooling

class HGTImageFeatureExtractor(nn.Module):
    """
    A Heterogeneous Graph Transformer (HGT) based feature extractor that
    handles two modalities (vit, text) and produces a refined
    visual feature vector and updated text feature vectors.
    """
    def __init__(self, node_types, edge_types, input_dims, hidden_channels,
                 hgt_num_heads, hgt_num_layers, dropout_rate,
                 transformer_nhead, transformer_num_layers,
                 transformer_ff_multiplier, transformer_activation, shots,
                 pooling_ratio: float):
        """
        Initializes the HGTImageFeatureExtractor model.
        """
        super().__init__()

        self.node_types = node_types
        # Identify visual node types (will just be ['vit'])
        self.visual_node_types = [nt for nt in node_types if nt != 'text']
        self.edge_types = edge_types
        self.metadata = (self.node_types, self.edge_types)
        self.hidden_channels = hidden_channels
        self.num_hgt_layers = hgt_num_layers
        self.shots = shots

        # --- Validation Checks ---
        if hidden_channels % hgt_num_heads != 0:
            raise ValueError(f"HGT hidden_channels ({hidden_channels}) must be "
                             f"divisible by hgt_num_heads ({hgt_num_heads}).")
        if hidden_channels % transformer_nhead != 0:
            raise ValueError(f"Transformer hidden_channels ({hidden_channels}) must be "
                             f"divisible by transformer_nhead ({transformer_nhead}).")
        
        # --- Input Projection Layers ---
        self.input_proj = nn.ModuleDict()
        for node_type in self.node_types:
            self.input_proj[node_type] = nn.Linear(input_dims[node_type], hidden_channels)

        # --- Transformer Encoders ---
        self.transformer_encoders = nn.ModuleDict()
        for node_type in self.node_types:
            encoder_layer = nn.TransformerEncoderLayer(
                d_model=hidden_channels, nhead=transformer_nhead,
                dim_feedforward=hidden_channels * transformer_ff_multiplier,
                dropout=dropout_rate, activation=transformer_activation, batch_first=True
            )
            self.transformer_encoders[node_type] = nn.TransformerEncoder(
                encoder_layer, num_layers=transformer_num_layers
            )

        # --- HGT Convolutional Layers ---
        self.convs = nn.ModuleList()
        self.norms = nn.ModuleList()
        for _ in range(self.num_hgt_layers):
            conv = HGTConv(hidden_channels, hidden_channels, self.metadata, hgt_num_heads)
            self.convs.append(conv)
            norm_dict = nn.ModuleDict()
            for node_type in self.node_types:
                # if self.shots in [1,2,8]:
                norm_dict[node_type] = nn.LayerNorm(hidden_channels) #BatchNorm1d
                # elif self.shots in [4,16]:
                    # norm_dict[node_type] = nn.BatchNorm1d(hidden_channels) #BatchNorm1d
            self.norms.append(norm_dict)
            
        self.dropout = nn.Dropout(dropout_rate)
        
        # --- Pooling Layers ---
        # This will now only create a pool for 'vit'
        
        # if self.shots in [16]:
        #     pooling_ratio=0.5
        # elif self.shots in [8]:
        #     pooling_ratio=1.0
        
        self.topk_pools = nn.ModuleDict()
        for node_type in self.visual_node_types:
            self.topk_pools[node_type] = TopKPooling(hidden_channels, ratio=pooling_ratio)
        
    def forward(self, x_dict, edge_index_dict, batch_dict):
        """
        Defines the forward pass of the model.
        
        Returns a tuple:
        1. graph_visual_feature (torch.Tensor): A single pooled
           feature vector for the visual nodes in the batch. Shape: [batch_size, hidden_channels]
        2. updated_text_features (torch.Tensor): The updated features for the text
           nodes, from all graphs in the batch. Shape: [total_text_nodes, hidden_channels]
        """
        # --- 1. Apply Input Projection ---
        projected_x_dict = {}
        for node_type, x_features in x_dict.items():
            projected_x_dict[node_type] = self.input_proj[node_type](x_features)

        # --- 2. Apply Transformer Encoder ---
        transformed_x_dict = {}
        for node_type, x_features in projected_x_dict.items():
            batch = batch_dict[node_type]
            image_outputs = []
            # Process each image in the batch separately to maintain sequence integrity
            for image_idx in torch.unique(batch):
                image_mask = (batch == image_idx)
                patch_sequence = x_features[image_mask].unsqueeze(0) # Add batch dim
                transformer_out = self.transformer_encoders[node_type](patch_sequence)
                image_outputs.append(transformer_out.squeeze(0)) # Remove batch dim
            
            transformed_x_dict[node_type] = torch.cat(image_outputs, dim=0)

        # --- 3. Apply HGT convolutions with residual connections ---
        current_x_dict = transformed_x_dict
        for conv, norm_dict in zip(self.convs, self.norms):
            x_in = current_x_dict
            x_out = conv(current_x_dict, edge_index_dict)
            
            for node_type in x_in.keys():
                # Apply normalization based on the type (LayerNorm vs BatchNorm)
                norm_layer = norm_dict[node_type]
                if isinstance(norm_layer, nn.BatchNorm1d):
                    # BatchNorm expects [N, C]
                    processed_out = self.dropout(norm_layer(x_out[node_type]).relu())
                else:
                    # LayerNorm expects [N, *]
                    processed_out = self.dropout(norm_layer(x_out[node_type]).relu())
                
                current_x_dict[node_type] = x_in[node_type] + processed_out
        
        pooled_visual_features = []    
#         if self.shots == 4:
#             # For the 4-shot case, bypass TopKPooling and use global average pooling instead.
#             for node_type in self.visual_node_types: # Loop over ['vit']
#                 features = current_x_dict[node_type]
#                 batch = batch_dict[node_type]

#                 # Pool the nodes for each image.
#                 pooled = global_add_pool(features, batch) # Or global_add_pool
#                 pooled_visual_features.append(pooled)

#             # <<< CHANGED: No longer stack/mean, just get the single visual feature
#             if len(pooled_visual_features) != 1:
#                 raise RuntimeError(f"Expected 1 visual modality, but found {len(pooled_visual_features)} pooled features.")
#             graph_visual_feature = pooled_visual_features[0]
#             # <<< END CHANGE
            
#             all_updated_text_features = current_x_dict['text']

#             return graph_visual_feature, all_updated_text_features
        
        # --- 4. Saliency-Based Pooling for VISUAL nodes (non 4-shot case) ---
        for node_type in self.visual_node_types: # Iterates only over ['vit']
            features = current_x_dict[node_type]
            intra_edge_type = (node_type, 'intra_patch', node_type)
            edge_index = edge_index_dict.get(intra_edge_type, torch.empty(2, 0, device=features.device, dtype=torch.long))

            # Handle case with no intra-patch edges (e.g., single patch)
            if edge_index.numel() == 0:
                # Fallback to global pooling if TopK fails
                selected_features = features
                selected_batch = batch_dict[node_type]
            else:
                selected_features, _, _, selected_batch, _, _ = self.topk_pools[node_type](
                    x=features,
                    edge_index=edge_index,
                    batch=batch_dict[node_type]
                )
            
            # Apply global pooling ONLY on the subset of selected, important nodes.
            pooled_visual_features.append(global_add_pool(selected_features, selected_batch))
        
        # <<< CHANGED: No longer stack/mean, just get the single visual feature
        if len(pooled_visual_features) != 1:
            raise RuntimeError(f"Expected 1 visual modality, but found {len(pooled_visual_features)} pooled features.")
        graph_visual_feature = pooled_visual_features[0]
        # <<< END CHANGE
        
        # --- 5. Extract Updated TEXT features ---
        all_updated_text_features = current_x_dict['text']
        
        return graph_visual_feature, all_updated_text_features