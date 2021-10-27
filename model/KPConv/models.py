import torch.nn.functional as F
import numpy as np
from model.KPConv.modules import *


class KPEncoder(nn.Module):

    def __init__(self, config, normalize=False):
        super(KPEncoder, self).__init__()

        ########################
        # Parameters
        ########################
        # Current radius of convolution and feature dimension
        layer = 0
        r = config.first_subsampling_dl * config.conv_radius
        in_dim = config.in_feats_dim
        out_dim = config.first_feats_dim
        self.K = config.num_kernel_points
        self.epsilon = torch.nn.Parameter(torch.tensor(-5.0))
        self.final_feats_dim = config.final_feats_dim
        self.normalize = normalize
        ########################
        # List Encoder Blocks
        ########################
        # Save all block operations in a list of modules
        self.encoder_blocks = nn.ModuleList()
        self.encoder_skip_dims = []
        self.encoder_skips = []

        # Loop over consecutive blocks:
        for block_i, block in enumerate(config.architecture):
            # Check equivariance
            if ('equivariant' in block) and (not out_dim % 3 == 0):
                raise ValueError('Equivariant block but features dimension is not a fator of 3')

            # Detect change to next layer for skip connection
            if np.any([tmp in block for tmp in ['pool', 'strided', 'upsample', 'global']]):
                self.encoder_skips.append(block_i)
                self.encoder_skip_dims.append(in_dim)

            # Detect upsampling block to stop
            if 'upsample' in block:
                break

            # Decide block type and add corresponding block into block list
            self.encoder_blocks.append(block_decider(block,
                                                     r,
                                                     in_dim,
                                                     out_dim,
                                                     layer,
                                                     config))

            # Update dimension of input from output
            if 'simple' in block:
                in_dim = out_dim // 2
            else:
                in_dim = out_dim

            # Detect change to a subsampled layer
            if 'pool' in block or 'strided' in block:
                # Update radius and feature dimension for next layer
                layer += 1
                r *= 2
                out_dim *= 2

        
        ##############
        # Bottleneck
        ##############
        out_feats_dim = config.gnn_feats_dim
        self.bottle = nn.Conv1d(in_dim, out_feats_dim, kernel_size=1, bias=True)
        return


    def forward(self, batch):
        # Get input features
        x = batch['features'].clone().detach()

        #############################
        # Encoder Forward Part
        #############################
        skip_x = []
        for block_i, block_op in enumerate(self.encoder_blocks):
            if block_i in self.encoder_skips:
                skip_x.append(x)
            x = block_op(x, batch)
            #print(block_i, ' ', x.shape)
        feats = x.transpose(0, 1).unsqueeze(0)
        # print(feats.shape)
        feats = self.bottle(feats)

        if self.normalize:
            feats = F.normalize(feats, p=2, dim=1)

        return feats, skip_x


class KPDecoder(nn.Module):
    def __init__(self, config):
        super(KPDecoder, self).__init__()
        in_dim = config.in_feats_dim
        out_dim = config.first_feats_dim
        layer = 0
        r = config.first_subsampling_dl * config.conv_radius
        gnn_feats_dim = config.gnn_feats_dim
        #######################
        # Get Encoder Skip Info
        ########################
        self.encoder_skip_dims = []
        # Loop over consecutive blocks:
        for block_i, block in enumerate(config.architecture):
            # Check equivariance
            if ('equivariant' in block) and (not out_dim % 3 == 0):
                raise ValueError('Equivariant block but features dimension is not a fator of 3')

            # Detect change to next layer for skip connection
            if np.any([tmp in block for tmp in ['pool', 'strided', 'upsample', 'global']]):
                self.encoder_skip_dims.append(in_dim)

            # Detect upsampling block to stop
            if 'upsample' in block:
                break

            # Update dimension of input from output
            if 'simple' in block:
                in_dim = out_dim // 2
            else:
                in_dim = out_dim

            # Detect change to a subsampled layer
            if 'pool' in block or 'strided' in block:
                # Update radius and feature dimension for next layer
                out_dim *= 2
                layer += 1
                r *= 2

        ####################
        # Decoder
        ####################
        out_dim = gnn_feats_dim

        # Save all block operations in a list of modules
        self.decoder_blocks = nn.ModuleList()
        self.decoder_concats = []

        start_i = -1

        # Find first upsampling block
        for block_i, block in enumerate(config.architecture):
            if 'upsample' in block:
                start_i = block_i
                break

        # Loop over consecutive blocks
        for block_i, block in enumerate(config.architecture[start_i:]):
            # Add dimension of skip connection concat
            if block_i > 0 and 'upsample' in config.architecture[start_i + block_i - 1]:
                in_dim += self.encoder_skip_dims[layer]
                self.decoder_concats.append(block_i)
            # Apply the good block function defining tf ops
            self.decoder_blocks.append(block_decider(block,
                                                 r,
                                                 in_dim,
                                                 out_dim,
                                                 layer,
                                                 config))
            # Update dimension of input from output
            in_dim = out_dim

            # Detect change to a subsample layer
            if 'upsample' in block:
                # Update radius and feature dimension for next layer
                layer -= 1
                r *= 0.5
                out_dim = out_dim // 2


    def forward(self, batch, feats, skip_x):
        x = feats[0].transpose(0, 1)
        #print(x.shape, ' ', skip_x[-1].shape)
        #############################
        # Decoder Forward Part
        #############################
        # print(self.decoder_concats)
        for block_i, block_op in enumerate(self.decoder_blocks):
            #print(block_i, ' ', block_op, ' ', x.shape)
            if block_i in self.decoder_concats:
                x = torch.cat([x, skip_x.pop()], dim=1)
            #print(block_i, ' ', block_op, ' ', x.shape)
            x = block_op(x, batch)
            #print(x.shape)

        return x

