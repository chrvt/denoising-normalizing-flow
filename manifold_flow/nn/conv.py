import torch

from torch import nn
from torch.nn import functional as F
import logging


class GatedConv2d(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride, padding, dilation=1):
        super(GatedConv2d, self).__init__()

        self.conv = nn.Conv2d(in_channels, 2 * out_channels, kernel_size, stride, padding, dilation)

    def forward(self, inputs):
        temps = self.conv(inputs)
        outputs = F.glu(temps, dim=1)
        return outputs


class GatedConvTranspose2d(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride, padding, output_padding=0, dilation=1):
        super(GatedConvTranspose2d, self).__init__()

        self.conv_transpose = nn.ConvTranspose2d(in_channels, 2 * out_channels, kernel_size, stride, padding, output_padding, dilation=dilation)

    def forward(self, inputs):
        temps = self.conv_transpose(inputs)
        outputs = F.glu(temps, dim=1)
        return outputs


class SylvesterFlowConvEncoderNet(nn.Module):
    def __init__(self, context_features, last_kernel_shape=(7, 7)):
        super().__init__()
        self.context_features = context_features
        self.last_kernel_shape = last_kernel_shape

        self.gated_conv_layers = nn.ModuleList(
            [
                GatedConv2d(in_channels=1, out_channels=32, kernel_size=5, padding=2, stride=1),
                GatedConv2d(in_channels=32, out_channels=32, kernel_size=5, padding=2, stride=2),  # 2
                GatedConv2d(in_channels=32, out_channels=64, kernel_size=5, padding=2, stride=1),  # 3
                GatedConv2d(in_channels=64, out_channels=64, kernel_size=5, padding=2, stride=2),  # 4
                GatedConv2d(in_channels=64, out_channels=64, kernel_size=5, padding=2, stride=1),  # 5
                GatedConv2d(in_channels=64, out_channels=64, kernel_size=5, padding=2, stride=1),  # 6
                GatedConv2d(in_channels=64, out_channels=256, kernel_size=self.last_kernel_shape, padding=0, stride=1),  # 7
            ]
        )

        self.context_layer = nn.Linear(in_features=256, out_features=self.context_features)

    def forward(self, inputs):
        batch_size = inputs.shape[0]
        temps = inputs
        del inputs
        for gated_conv in self.gated_conv_layers:
            temps = gated_conv(temps)
        outputs = self.context_layer(temps.reshape(batch_size, -1))
        del temps
        return outputs


class SylvesterFlowConvDecoderNet(nn.Module):
    def __init__(self, latent_features, last_kernel_shape=(7, 7)):
        super().__init__()
        self.latent_features = latent_features
        self.last_kernel_shape = last_kernel_shape

        self.gated_conv_transpose_layers = nn.ModuleList(
            [
                GatedConvTranspose2d(in_channels=self.latent_features, out_channels=64, kernel_size=self.last_kernel_shape, padding=0, stride=1),
                GatedConvTranspose2d(in_channels=64, out_channels=64, kernel_size=5, padding=2, stride=1),  # 2
                GatedConvTranspose2d(in_channels=64, out_channels=32, kernel_size=5, padding=2, stride=2, output_padding=1),  # 3
                GatedConvTranspose2d(in_channels=32, out_channels=32, kernel_size=5, padding=2, stride=1),  # 4
                GatedConvTranspose2d(in_channels=32, out_channels=32, kernel_size=5, padding=2, stride=2, output_padding=1),  # 5
                GatedConv2d(in_channels=32, out_channels=32, kernel_size=5, padding=2, stride=1),  # 6
                GatedConv2d(in_channels=32, out_channels=1, kernel_size=1, padding=0, stride=1),  # 7
            ]
        )

    def forward(self, inputs):
        temps = inputs[..., None, None]
        del inputs
        for gated_conv_transpose in self.gated_conv_transpose_layers:
            temps = gated_conv_transpose(temps)
        outputs = temps
        del temps
        return outputs


class ResidualBlock(nn.Module):
    def __init__(self, in_channels, resample=None, activation=F.relu, dropout=None, first=False):
        super().__init__()
        self.in_channels = in_channels
        self.resample = resample
        self.activation = activation
        self.dropout = dropout

        self.residual_layer_1 = nn.Conv2d(in_channels=in_channels, out_channels=in_channels, kernel_size=3, padding=1)

        if resample is None:
            self.shortcut_layer = nn.Identity()
            self.residual_2_layer = nn.Conv2d(in_channels=in_channels, out_channels=in_channels, kernel_size=3, padding=1)
        elif resample == "down":
            self.shortcut_layer = nn.Conv2d(in_channels=in_channels, out_channels=2 * in_channels, kernel_size=3, stride=2, padding=1)
            self.residual_2_layer = nn.Conv2d(in_channels=in_channels, out_channels=2 * in_channels, kernel_size=3, stride=2, padding=1)
        elif resample == "up":
            self.shortcut_layer = nn.ConvTranspose2d(in_channels=in_channels, out_channels=in_channels // 2, kernel_size=3, stride=2, padding=1, output_padding=0 if first else 1)
            self.residual_2_layer = nn.ConvTranspose2d(in_channels=in_channels, out_channels=in_channels // 2, kernel_size=3, stride=2, padding=1, output_padding=0 if first else 1)

    def forward(self, inputs):

        shortcut = self.shortcut_layer(inputs)
        residual_1 = self.activation(inputs)
        residual_1 = self.residual_layer_1(residual_1)
        residual_2 = self.activation(residual_1)
        residual_2 = self.residual_2_layer(residual_2)

        return shortcut + residual_2


class ConvEncoder(nn.Module):
    def __init__(self, context_features, channels_multiplier, activation=F.relu):
        super().__init__()
        self.context_features = context_features
        self.channels_multiplier = channels_multiplier
        self.activation = activation

        self.initial_layer = nn.Conv2d(1, channels_multiplier, kernel_size=1)
        self.residual_blocks = nn.ModuleList(
            [
                ResidualBlock(in_channels=channels_multiplier),
                ResidualBlock(in_channels=channels_multiplier, resample="down"),
                ResidualBlock(in_channels=channels_multiplier * 2),
                ResidualBlock(in_channels=channels_multiplier * 2, resample="down"),
                ResidualBlock(in_channels=channels_multiplier * 4),
                ResidualBlock(in_channels=channels_multiplier * 4, resample="down"),
            ]
        )
        self.final_layer = nn.Linear(in_features=(4 * 4 * channels_multiplier * 8), out_features=context_features)

    def forward(self, inputs):
        temps = self.initial_layer(inputs)
        # print(temps.shape)
        for residual_block in self.residual_blocks:
            temps = residual_block(temps)
            # print(temps.shape)
        temps = self.activation(temps)
        outputs = self.final_layer(temps.reshape(-1, 4 * 4 * self.channels_multiplier * 8))
        # print(outputs.shape)
        return outputs


class ConvDecoder(nn.Module):
    def __init__(self, latent_features, channels_multiplier, activation=F.relu):
        super().__init__()
        self.latent_features = latent_features
        self.channels_multiplier = channels_multiplier
        self.activation = activation
        k = 2
        self.initial_layer = nn.Linear(in_features=latent_features, out_features=(k * k * channels_multiplier * 32))
        self.residual_blocks = nn.ModuleList(
            [
                ResidualBlock(in_channels=channels_multiplier * 32),
                ResidualBlock(in_channels=channels_multiplier * 32, resample="up"), #, first=True),
                ResidualBlock(in_channels=channels_multiplier * 16),
                ResidualBlock(in_channels=channels_multiplier * 16, resample="up"),
                ResidualBlock(in_channels=channels_multiplier * 8),
                ResidualBlock(in_channels=channels_multiplier * 8, resample="up"),
                ResidualBlock(in_channels=channels_multiplier * 4),
                ResidualBlock(in_channels=channels_multiplier * 4, resample="up"),
                ResidualBlock(in_channels=channels_multiplier * 2),
                ResidualBlock(in_channels=channels_multiplier * 2, resample="up"),
   
            ]
        )
        self.final_layer = nn.Conv2d(in_channels=channels_multiplier, out_channels=3, kernel_size=1)

    def forward(self, inputs):
        k = 2
        temps = self.initial_layer(inputs).reshape(-1, self.channels_multiplier * 32, k, k)
        for residual_block in self.residual_blocks:
            temps = residual_block(temps)
        temps = self.activation(temps)
        outputs = self.final_layer(temps)
        return outputs


class ModifiedConvEncoder(nn.Module):
    def __init__(self, h, w, channels_in, out_features, levels=4, channels_multiplier=1, activation=F.relu):
        super().__init__()
        self.channels_in = channels_in
        self.out_features = out_features
        self.channels_multiplier = channels_multiplier
        self.activation = activation

        self.initial_layer = nn.Conv2d(channels_in, channels_in * channels_multiplier, kernel_size=1)
        blocks = []
        for i in range(levels):
            blocks.append(ResidualBlock(in_channels=channels_in * channels_multiplier * 2 ** i))
            blocks.append(ResidualBlock(in_channels=channels_in * channels_multiplier * 2 ** i, resample="down"))
        self.residual_blocks = nn.ModuleList(blocks)

        self.flat_dim = (h // 2 ** levels) * (w // 2 ** levels) * channels_in * channels_multiplier * 2 ** levels
        self.final_layer = nn.Linear(in_features=self.flat_dim, out_features=out_features)

    def forward(self, inputs, context=None):
        assert context is None
        temps = self.initial_layer(inputs)
        for residual_block in self.residual_blocks:
            temps = residual_block(temps)
        temps = self.activation(temps)
        outputs = self.final_layer(temps.reshape(-1, self.flat_dim))
        return outputs



class infoGAN_encoder(nn.Module):
	def __init__(self, latent_dim,dropout):
            super().__init__()
            self.conv1 = nn.Conv2d(3, 64, 4, 2, 1)

            self.conv2 = nn.Conv2d(64, 128, 4, 2, 1, bias=True)
            self.bn2 = nn.BatchNorm2d(128)

            self.lin1 = nn.Linear(in_features=128*64*4, out_features=1024)
            self.bn3 = nn.BatchNorm1d(1024)
            self.lin2 = nn.Linear(in_features=1024, out_features=latent_dim)
            if dropout > 0:
                self.dropout = nn.Dropout(dropout)
            else: self.dropout = nn.Identity()
	def forward(self, x,context=None):
            x = F.leaky_relu(self.conv1(x), 0.1, inplace=True)  
            x = self.bn2(self.conv2(x))
            x = self.dropout(x)
            x = F.leaky_relu(x, 0.1, inplace=True)
            
            x = torch.flatten(x,start_dim=1)
            x = self.dropout(self.bn3(self.lin1(x)))
            x = F.leaky_relu(x, 0.1, inplace=True)
            x = self.lin2(x)
            return x
    
class infoGAN_decoder(nn.Module):
    def __init__(self,latent_dim,width,height,batch_size,dropout):
        super().__init__()
        self.batch_size = batch_size
        self.width = width
        self.height = height
        
        self.lin1 = nn.Linear(in_features=latent_dim, out_features=1024)
        self.bn1 = nn.BatchNorm1d(1024)
        
        self.lin2 = nn.Linear(in_features=1024, out_features=128 * (width // 4) * (height // 4)  )
        self.bn2 = nn.BatchNorm1d(128 * (width // 4) * (height // 4))


        self.dec_conv1 = nn.ConvTranspose2d(128, 64, 4, 2, 1, bias=True)
        self.bn3 = nn.BatchNorm2d(64)
        
        self.dec_conv2 = nn.ConvTranspose2d(64, 3, 4, 2, 1, bias=True)
        if dropout > 0:
            self.dropout = nn.Dropout(dropout)
        else: self.dropout = nn.Identity()
        
    def forward(self, x,context=None):
        batch_size = x.shape[0]
        net = F.relu(self.bn1(self.lin1(x)))
        net = F.relu(self.bn2(self.lin2(net)))
        net = self.dropout(net)
        
        net = net.reshape([batch_size,128, (self.width // 4), (self.height // 4)])
        
        net = F.relu(self.bn3(self.dec_conv1(net)))
        net = self.dropout(net)
        net = torch.sigmoid(self.dec_conv2(net))*256 #torch.tanh
        
        return net

class simple_decoder(nn.Module):
    def __init__(self,channels,width,height,dropout):
        super(simple_decoder,self).__init__()
        
        self.width = width
        self.height = height
        self.channels = channels
        
        #self.lin1 = nn.Linear(in_features=self.latent_dim, out_features=self.latent_dim)
        #self.bn1 = nn.BatchNorm1d(self.latent_dim)
        #kernel_size=4
        #same_padding = kernel_size // 2 
        self.dec_conv = nn.Conv2d(in_channels=self.channels, out_channels=self.channels, kernel_size=5, padding=2) #same_padding)
        #self.bn2 = nn.BatchNorm2d(self.channels)
        
        #self.dec_conv2 = nn.ConvTranspose2d(64, 3, 4, 2, 1, bias=True)
        if dropout > 0:
            self.dropout = nn.Dropout(dropout)
        else: self.dropout = nn.Identity()
        
    def forward(self, x,context=None):
        #batch_size = x.shape[0]
        #net = F.relu(self.bn1(self.lin1(x)))
        #net = self.dropout(net)

        net = torch.sigmoid(self.dec_conv(x))*256 #torch.tanh
        
        return net

class Conv2dSameSize(nn.Conv2d):
    def __init__(self, in_channels, out_channels, kernel_size):
        same_padding = kernel_size // 2  # Padding that would keep the spatial dims the same
        super().__init__(in_channels, out_channels, kernel_size, padding=same_padding)


def main():
    batch_size, channels, width, height = 16, 1, 28, 28
    inputs = torch.rand(batch_size, channels, width, height)

    net = ConvEncoder(context_features=24, channels_multiplier=16)
    outputs = net(inputs)

    net = ConvDecoder(latent_features=24, channels_multiplier=16)
    outputs = net(outputs)


if __name__ == "__main__":
    main()

