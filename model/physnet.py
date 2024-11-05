import torch
import torch.nn as nn
import torch.nn.functional as F


'''
An implementation of PhysNet_3DCNN_ED model based on PhysNetED_BMVC.py
'''
class PhysNet_3DCNN_ED(nn.Module):
    def __init__(self, frames=128):
        super(PhysNet_3DCNN_ED, self).__init__()
        
        self.ConvBlock1 = nn.Sequential(
            nn.Conv3d(3, 16, [1,5,5],stride=1, padding=[0,2,2]),
            nn.BatchNorm3d(16),
            nn.ReLU(inplace=True),
        )

        self.ConvBlock2 = nn.Sequential(
            nn.MaxPool3d([1, 2, 2], stride=[1, 2, 2]),
            # conv 1
            nn.Conv3d(16, 32, [3, 3, 3], stride=1, padding=1),
            nn.BatchNorm3d(32),
            nn.ReLU(inplace=True),
            # conv 2
            nn.Conv3d(32, 64, [3, 3, 3], stride=1, padding=1),
            nn.BatchNorm3d(64),
            nn.ReLU(inplace=True),
        )
        
        self.ConvBlock3 = nn.Sequential(
            nn.MaxPool3d((2,2,2), stride=2),
            # conv 1
            nn.Conv3d(64, 64, [3, 3, 3], stride=1, padding=1),
            nn.BatchNorm3d(64),
            nn.ReLU(inplace=True),
            # conv 2
            nn.Conv3d(64, 64, [3, 3, 3], stride=1, padding=1),
            nn.BatchNorm3d(64),
            nn.ReLU(inplace=True),
        )
        self.ConvBlock4 = nn.Sequential(
            nn.MaxPool3d((2,2,2), stride=2),
            # conv 1
            nn.Conv3d(64, 64, [3, 3, 3], stride=1, padding=1),
            nn.BatchNorm3d(64),
            nn.ReLU(inplace=True),
            # conv 2
            nn.Conv3d(64, 64, [3, 3, 3], stride=1, padding=1),
            nn.BatchNorm3d(64),
            nn.ReLU(inplace=True),
        )
        
        self.ConvBlock5 = nn.Sequential(
            nn.MaxPool3d([1, 2, 2], stride=[1, 2, 2]),
            # conv 1
            nn.Conv3d(64, 64, [3, 3, 3], stride=1, padding=1),
            nn.BatchNorm3d(64),
            nn.ReLU(inplace=True),
            # conv 2
            nn.Conv3d(64, 64, [3, 3, 3], stride=1, padding=1),
            nn.BatchNorm3d(64),
            nn.ReLU(inplace=True),
        )
        
        self.upsample = nn.Sequential(
            # upsample 1  
           nn.ConvTranspose3d(in_channels=64, out_channels=64, kernel_size=[4,1,1], stride=[2,1,1], padding=[1,0,0]),   #[1, 128, 32]
           nn.BatchNorm3d(64),
           nn.ELU(),
           # upsample 2
           nn.ConvTranspose3d(in_channels=64, out_channels=64, kernel_size=[4,1,1], stride=[2,1,1], padding=[1,0,0]),   #[1, 128, 32]
           nn.BatchNorm3d(64),
           nn.ELU()
       )
        self.spatial_global_avgpool = nn.AdaptiveAvgPool3d((frames,1,1))
        self.ConvBlock6 = nn.Sequential(
            nn.Conv3d(64,1, [1,1,1], stride=1, padding=0)
        )
    
    def forward(self, x):
        # x [n, 3, T, 128,128] 
        [batch, channel, length, width, height] = x.shape
        x_visual = x
        # output_1 [n, 16, T, 128,128]
        output_1 = self.ConvBlock1(x)
        # output_6464 [n, 64, T, 64,64]
        output_6464 = self.ConvBlock2(output_1)
        # output_3232 [n, 64, T/2, 32,32]
        output_3232 = self.ConvBlock3(output_6464)
        # output_1616 [n, 64, T/4, 16,16]
        output_1616 = self.ConvBlock4(output_3232)
        # output_5 [n, 64, T/4, 8,8]
        output_5 = self.ConvBlock5(output_1616)
        # output_upsample [n, 64, T, 8,8]
        output_upsample = self.upsample(output_5)
        # output_pool [n, 64, T, 1,1]
        output_pool = self.spatial_global_avgpool(output_upsample) 
        # output [n, 1, T, 1,1]
        output = self.ConvBlock6(output_pool)    
        
        rppg = output.view(-1,length)   
        
        return rppg, x_visual, output_3232, output_1616
    

# test
if __name__ == '__main__':
    model = PhysNet_3DCNN_ED(frames=32).to('cuda')
    
    from torchsummary import summary
    summary(model, (3, 32, 128, 128))
    