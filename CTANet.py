import torch
import torch.nn as nn
from localbranch import LayerNorm, Local_block
from globalbranch import PatchEmbed, BasicLayer, PatchMerging
from AFFN import AFFN


class MainModel(nn.Module):
    """
    Multi-branch model class with global and local branches, combining convolutional and transformer-based
    layers for feature extraction and fusion.
    """

    def __init__(self, num_classes, patch_size=4, input_channels=3, embedding_dim=96, stage_depths=(2, 2, 2, 2),
                 attention_heads=(3, 6, 12, 24), window_size=7, use_qkv_bias=True, dropout_rate=0.0,
                 attention_dropout_rate=0.0, drop_path_rate=0.0, normalization_layer=nn.LayerNorm,
                 use_patch_norm=True, enable_checkpoint=False, hierarchical_fusion_dropout=0.0,
                 conv_stage_depths=(2, 2, 2, 2), conv_stage_dims=(96, 192, 384, 768),
                 conv_drop_path_rate=0.0, head_init_scale=1.0, **kwargs):
        super().__init__()

        # Initialize downsampling layers for local branch
        self.local_downsampling_layers = nn.ModuleList()  # Stem and three downsampling stages
        initial_conv = nn.Sequential(
            nn.Conv2d(input_channels, conv_stage_dims[0], kernel_size=4, stride=4),  # Initial convolution layer
            LayerNorm(conv_stage_dims[0], eps=1e-6, data_format="channels_first")  # Normalization
        )
        self.local_downsampling_layers.append(initial_conv)

        # Add additional downsampling layers for each stage
        for i in range(3):
            downsample = nn.Sequential(
                LayerNorm(conv_stage_dims[i], eps=1e-6, data_format="channels_first"),
                nn.Conv2d(conv_stage_dims[i], conv_stage_dims[i + 1], kernel_size=2, stride=2)
            )
            self.local_downsampling_layers.append(downsample)

        # Define drop path rates for local blocks
        drop_path_rates = [x.item() for x in torch.linspace(0, conv_drop_path_rate, sum(conv_stage_depths))]
        current_drop_path_idx = 0

        # Initialize local branch stages
        self.local_feature_stages = nn.ModuleList()
        for i in range(4):
            stage = nn.Sequential(
                *[Local_block(dim=conv_stage_dims[i], drop_rate=drop_path_rates[current_drop_path_idx + j])
                  for j in range(conv_stage_depths[i])]
            )
            self.local_feature_stages.append(stage)
            current_drop_path_idx += conv_stage_depths[i]

        # Define output layer for local branch
        self.local_output_norm = nn.LayerNorm(conv_stage_dims[-1], eps=1e-6)
        self.local_classification_head = nn.Linear(conv_stage_dims[-1], num_classes)
        self.local_classification_head.weight.data.mul_(head_init_scale)
        self.local_classification_head.bias.data.mul_(head_init_scale)

        # Global branch parameters
        self.num_classes = num_classes
        self.num_global_stages = len(stage_depths)
        self.embedding_dim = embedding_dim
        self.use_patch_norm = use_patch_norm
        self.final_global_feature_dim = int(embedding_dim * 2 ** (self.num_global_stages - 1))

        # Patch embedding for the global branch
        self.patch_embedding = PatchEmbed(
            patch_size=patch_size, in_c=input_channels, embed_dim=embedding_dim,
            norm_layer=normalization_layer if self.use_patch_norm else None
        )
        self.global_position_dropout = nn.Dropout(p=dropout_rate)

        # Define drop path rates for global attention layers
        global_drop_path_rates = [x.item() for x in torch.linspace(0, drop_path_rate, sum(stage_depths))]

        # Initialize global branch stages
        self.global_stage1 = BasicLayer(
            dim=int(embedding_dim * 2 ** 0), depth=stage_depths[0], num_heads=attention_heads[0],
            window_size=window_size, qkv_bias=use_qkv_bias, drop=dropout_rate,
            attn_drop=attention_dropout_rate, drop_path=global_drop_path_rates[:stage_depths[0]],
            norm_layer=normalization_layer, downsample=PatchMerging if 0 > 0 else None,
            use_checkpoint=enable_checkpoint
        )
        self.global_stage2 = BasicLayer(
            dim=int(embedding_dim * 2 ** 1), depth=stage_depths[1], num_heads=attention_heads[1],
            window_size=window_size, qkv_bias=use_qkv_bias, drop=dropout_rate,
            attn_drop=attention_dropout_rate, drop_path=global_drop_path_rates[stage_depths[0]:sum(stage_depths[:2])],
            norm_layer=normalization_layer, downsample=PatchMerging, use_checkpoint=enable_checkpoint
        )
        self.global_stage3 = BasicLayer(
            dim=int(embedding_dim * 2 ** 2), depth=stage_depths[2], num_heads=attention_heads[2],
            window_size=window_size, qkv_bias=use_qkv_bias, drop=dropout_rate,
            attn_drop=attention_dropout_rate,
            drop_path=global_drop_path_rates[sum(stage_depths[:2]):sum(stage_depths[:3])],
            norm_layer=normalization_layer, downsample=PatchMerging, use_checkpoint=enable_checkpoint
        )
        self.global_stage4 = BasicLayer(
            dim=int(embedding_dim * 2 ** 3), depth=stage_depths[3], num_heads=attention_heads[3],
            window_size=window_size, qkv_bias=use_qkv_bias, drop=dropout_rate,
            attn_drop=attention_dropout_rate, drop_path=global_drop_path_rates[sum(stage_depths[:3]):],
            norm_layer=normalization_layer, downsample=PatchMerging, use_checkpoint=enable_checkpoint
        )

        # Global pooling and classification head for the global branch
        self.global_norm = normalization_layer(self.final_global_feature_dim)
        self.global_avgpool = nn.AdaptiveAvgPool1d(1)
        self.global_classification_head = nn.Linear(self.final_global_feature_dim,
                                                    num_classes) if num_classes > 0 else nn.Identity()

        # Apply weights initialization
        self.apply(self._initialize_weights)


        self.finanl_feature_fusion = AFFN(drop_rate=hierarchical_fusion_dropout)

    def _initialize_weights(self, module):
        """
        Initializes weights for linear, layer normalization, and convolutional layers.
        """
        if isinstance(module, nn.Linear):
            nn.init.trunc_normal_(module.weight, std=0.02)
            if hasattr(module, 'bias') and module.bias is not None:
                nn.init.constant_(module.bias, 0)
        elif isinstance(module, nn.LayerNorm):
            nn.init.constant_(module.bias, 0)
            nn.init.constant_(module.weight, 1.0)
        elif isinstance(module, nn.Conv2d):
            nn.init.trunc_normal_(module.weight, std=0.2)
            if hasattr(module, 'bias') and module.bias is not None:
                nn.init.constant_(module.bias, 0)

    def forward(self, original_images, clahe_images=None):
        """
        Forward pass through the network.
        :param original_images: Original input images.
        :param clahe_images: Optional contrast-enhanced images.
        :return: Final classification output.
        """

        # Global Branch Processing
        # Step 1: Apply patch embedding to split images into patches and embed them
        global_features, height, width = self.patch_embedding(original_images)

        # Step 2: Apply positional dropout for regularization
        global_features = self.global_position_dropout(global_features)

        # Step 3: Pass through each stage of the global branch
        global_features_stage1, height, width = self.global_stage1(global_features, height, width)
        global_features_stage2, height, width = self.global_stage2(global_features_stage1, height, width)
        global_features_stage3, height, width = self.global_stage3(global_features_stage2, height, width)
        global_features_stage4, height, width = self.global_stage4(global_features_stage3, height, width)

        # Step 4: Reshape the global output for compatibility in fusion
        # Transpose and reshape to match the spatial format needed for fusion with local features
        global_features_stage4 = torch.transpose(global_features_stage4, 1, 2).view(global_features_stage4.shape[0], -1,
                                                                                    7, 7)

        # Local Branch Processing
        # Step 1: Apply initial downsampling (stem layer)
        local_features = self.local_downsampling_layers[0](original_images)

        # Step 2: Pass through each stage of the local branch with downsampling
        local_features_stage1 = self.local_feature_stages[0](local_features)
        local_features = self.local_downsampling_layers[1](local_features_stage1)

        local_features_stage2 = self.local_feature_stages[1](local_features)
        local_features = self.local_downsampling_layers[2](local_features_stage2)

        local_features_stage3 = self.local_feature_stages[2](local_features)
        local_features = self.local_downsampling_layers[3](local_features_stage2)

        local_features_stage4 = self.local_feature_stages[3](local_features)

        fused_features = self.final_feature_fusion_stage4(local_features_stage4, global_features_stage4, None)
        fused_features = self.local_output_norm(fused_features.mean(dim=[-2, -1]))
        # Classification Head # Pass the pooled and normalized features through the final classification layer
        final_output = self.local_classification_head(fused_features)
        return final_output


