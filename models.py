import torch
import torch.nn as nn

class DoubleConv(nn.Module):
    """(Convolution => BatchNorm => ReLU) * 2"""
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.double_conv = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        return self.double_conv(x)


class UNetQualityInspector(nn.Module):
    """
    U-Net Quality Inspector.
    Inputs: (Batch, 3, 512, 512) -> [QV_t1, U_pred, V_pred]
    Outputs: (Batch, 1, 512, 512) -> [Predicted_Absolute_Error]
    """
    # DEFAULT CHANGED TO 3 CHANNELS
    def __init__(self, in_channels=3, out_channels=1, features=[64, 128, 256, 512]):
        super().__init__()
        self.ups = nn.ModuleList()
        self.downs = nn.ModuleList()
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)

        # Encoder Path
        curr_in_channels = in_channels
        for feature in features:
            self.downs.append(DoubleConv(curr_in_channels, feature))
            curr_in_channels = feature

        # Decoder Path
        for feature in reversed(features):
            self.ups.append(nn.ConvTranspose2d(feature * 2, feature, kernel_size=2, stride=2))
            self.ups.append(DoubleConv(feature * 2, feature))

        self.bottleneck = DoubleConv(features[-1], features[-1] * 2)
        self.final_conv = nn.Conv2d(features[0], out_channels, kernel_size=1)

    def forward(self, x):
        skip_connections = []

        # Encoder
        for down in self.downs:
            x = down(x)
            skip_connections.append(x)
            x = self.pool(x)

        # Bottleneck
        x = self.bottleneck(x)
        skip_connections = skip_connections[::-1]

        # Decoder
        for idx in range(0, len(self.ups), 2):
            x = self.ups[idx](x)
            skip_connection = skip_connections[idx // 2]
            
            if x.shape[2:] != skip_connection.shape[2:]:
                import torch.nn.functional as F
                x = F.interpolate(x, size=skip_connection.shape[2:], mode="bilinear", align_corners=False)
            
            concat_x = torch.cat((skip_connection, x), dim=1)
            x = self.ups[idx + 1](concat_x)

        return self.final_conv(x)


# ---------- v2 -------------#
################################################################

class DoubleConvV2(nn.Module):
    def __init__(self, in_channels, out_channels, dilation=1):
        super().__init__()
        self.double_conv = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=dilation, dilation=dilation, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.SiLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=dilation, dilation=dilation, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.SiLU(inplace=True)
        )
    
    def forward(self, x):
        return self.double_conv(x)



class UNetQualityInspectorV2(nn.Module):
    """
    U-Net Quality Inspector v2.
    Inputs: (Batch, 3, 512, 512) -> [QV_t1, U_pred, V_pred]
    Outputs: (Batch, 1, 512, 512) -> [Predicted_Absolute_Error]
    """
    def __init__(self, in_channels=3, out_channels=1, features=[64, 128, 256, 512]):
        super().__init__()
        self.ups = nn.ModuleList()
        self.downs = nn.ModuleList()
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)

        curr_in_channels = in_channels
        for feature in features:
            self.downs.append(DoubleConvV2(curr_in_channels, feature))
            curr_in_channels = feature
        
        # decoder
        for feature in reversed(features):
            self.ups.append(nn.ConvTranspose2d(feature * 2, feature, kernel_size=2, stride=2))
            self.ups.append(DoubleConvV2(feature * 2, feature))
        
        # UPGRADED BOTTLENECK: Dilated Receptive Field + Dropout
        self.bottleneck = nn.Sequential(
            DoubleConvV2(features[-1], features[-1] * 2, dilation=2), # Reduced to 1 DoubleConv
            nn.Dropout2d(p=0.2)
        )

        self.final_conv = nn.Conv2d(features[0], out_channels, kernel_size=1)

    def forward(self, x):
        skip_connections = []

        # Encoder
        for down in self.downs:
            x = down(x)
            skip_connections.append(x)
            x = self.pool(x)

        # Bottleneck (Dilated)
        x = self.bottleneck(x)
        skip_connections = skip_connections[::-1]

        # Decoder
        for idx in range(0, len(self.ups), 2):
            x = self.ups[idx](x)
            skip_connection = skip_connections[idx // 2]
            
            if x.shape[2:] != skip_connection.shape[2:]:
                x = F.interpolate(x, size=skip_connection.shape[2:], mode="bilinear", align_corners=False)
            
            concat_x = torch.cat((skip_connection, x), dim=1)
            x = self.ups[idx + 1](concat_x)

        return self.final_conv(x)

if __name__ == "__main__":
    # Sanity check with 3 channels
    mock_input = torch.randn(4, 3, 512, 512)
    model = UNetQualityInspector(in_channels=3, out_channels=1)
    with torch.no_grad():
        mock_output = model(mock_input)
    print(f"3-Channel Verification Success! Output shape: {mock_output.shape}")