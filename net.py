import torch
from torch import nn

import random
from math import log2, ceil

from torch.autograd import Function
from torch.nn import functional as F

from scaled_layers import set_scale, ScaledLinear, ScaledConv2d 

####################################################################################################################
################################################## Level 1 blocks ##################################################
####################################################################################################################

class FromRGB(nn.Module):
    def __init__(self, inp_c, oup_c):
        super(FromRGB, self).__init__()
        self.from_rgb = nn.Sequential(ScaledConv2d(inp_c, oup_c, 1, 1, 0), nn.LeakyReLU(0.2))
        self.downsample = nn.AvgPool2d(2)
        
    def forward(self, x, downsample=False):
        if downsample:
            return self.from_rgb(self.downsample(x.contiguous()))
        else:
            return self.from_rgb(x.contiguous())
    
    
class ToRGB(nn.Module):
    def __init__(self, inp_c, oup_c):
        super(ToRGB, self).__init__()
        self.to_rgb = ScaledConv2d(inp_c, oup_c, 1, 1, 0)
        self.upsample = nn.UpsamplingBilinear2d(scale_factor=2)
            
    def forward(self, x, upsample=False):
        if upsample:
            return self.to_rgb(self.upsample(x.contiguous()))
        else:
            return self.to_rgb(x.contiguous())

class BallProjection(nn.Module):
    """
    Constraint norm of an input noise vector to be sqrt(latent_code_size)
    """
    def __init__(self):
        super().__init__()

    def forward(self, x):
        return x.div((torch.mean(x.pow(2), dim=1, keepdim=True).add(1e-8)).pow(0.5))
    
        
class AdaIN(nn.Module):
    def __init__(self, n_channels, code):
        super().__init__()
        
        self.insance_norm = nn.InstanceNorm2d(n_channels, affine=False, eps=1e-8)
        self.A = ScaledLinear(code, n_channels * 2)
        
        # StyleGAN
        # self.A.linear.bias.data = torch.cat([torch.ones(n_channels), torch.zeros(n_channels)])
        
    def forward(self, x, style):
        """
        x - (N x C x H x W)
        style - (N x (Cx2))
        """        
        # Project project style vector(w) to  mu, sigma and reshape it 2D->4D to allow channel-wise operations        
        style = self.A(style)
        y = style.view(style.shape[0], 2, style.shape[1]//2).unsqueeze(3).unsqueeze(4)
        
        return torch.addcmul(y[:, 1], value=1., tensor1=y[:, 0] + 1, tensor2 = x)        

    
class IntermediateNoise(nn.Module):
    def __init__(self, inp_c):
        super().__init__()
        self.weight = nn.Parameter(torch.randn(1, inp_c, 1, 1), requires_grad=True)
    
    def forward(self, x):
        if self.training:
            noise = torch.randn(x.shape[0], 1, x.shape[-2], x.shape[-1]).to(x.device)
            return x + (noise * self.weight)
        else:
            return x
    
    
class BlurFunctionBackward(Function):
    """
    Official Blur implementation
    https://github.com/adambielski/perturbed-seg/blob/master/stylegan.py
    """
    @staticmethod
    def forward(ctx, grad_output, kernel, kernel_flip):
        ctx.save_for_backward(kernel, kernel_flip)

        grad_input = F.conv2d(
            grad_output, kernel_flip, padding=1, groups=grad_output.shape[1]
        )

        return grad_input

    @staticmethod
    def backward(ctx, gradgrad_output):
        kernel, kernel_flip = ctx.saved_tensors

        grad_input = F.conv2d(
            gradgrad_output, kernel, padding=1, groups=gradgrad_output.shape[1]
        )

        return grad_input, None, None


class BlurFunction(Function):
    @staticmethod
    def forward(ctx, input, kernel, kernel_flip):
        ctx.save_for_backward(kernel, kernel_flip)

        output = F.conv2d(input, kernel, padding=1, groups=input.shape[1])

        return output

    @staticmethod
    def backward(ctx, grad_output):
        kernel, kernel_flip = ctx.saved_tensors

        grad_input = BlurFunctionBackward.apply(grad_output, kernel, kernel_flip)

        return grad_input, None, None

blur = BlurFunction.apply

class Blur(nn.Module):
    def __init__(self, channel):
        super().__init__()

        weight = torch.tensor([[1, 2, 1], [2, 4, 2], [1, 2, 1]], dtype=torch.float32)
        weight = weight.view(1, 1, 3, 3)
        weight = weight / weight.sum()
        weight_flip = torch.flip(weight, [2, 3])

        self.register_buffer('weight', weight.repeat(channel, 1, 1, 1))
        self.register_buffer('weight_flip', weight_flip.repeat(channel, 1, 1, 1))

    def forward(self, input):
        return blur(input, self.weight, self.weight_flip)


####################################################################################################################
################################################## Level 2 blocks ##################################################
####################################################################################################################
    
class EncoderBlock(nn.Module):
    def __init__(self, inp_c, oup_c, code, final=False, blur_downsample=False):
        super().__init__()
        
        self.final = final
        self.blur_downsample = blur_downsample
        
        self.in1 = nn.InstanceNorm2d(inp_c, affine=False)
        self.in2 = nn.InstanceNorm2d(oup_c, affine=False)
        
        self.conv1 = ScaledConv2d(inp_c, inp_c, kernel_size=3, stride=1, padding=1)
        self.style_mapping1 = ScaledLinear(2 * inp_c, code)
        
        if final:
            self.fc = ScaledLinear(inp_c * 4 * 4, oup_c)
            self.style_mapping2 = ScaledLinear(oup_c, code)
        else:
            self.conv2 = ScaledConv2d(inp_c, oup_c, kernel_size=3, stride=1, padding=1)    
            self.style_mapping2 = ScaledLinear(2 * oup_c, code)
            
        self.act = nn.LeakyReLU(0.2)
        self.downsample = nn.AvgPool2d(2, 2)
        
        self.blur = Blur(inp_c)
        
    def forward(self, x):
        
        x = self.act(self.conv1(x))
        statistics1 = torch.cat([x.mean(dim=[2,3]), x.std(dim=[2,3])], dim=1)
        style1 = self.style_mapping1(statistics1)
        x = self.in1(x)
        
        if self.final:
            x = x.view(x.shape[0], -1)
            statistics2 = self.act(self.fc(x))
            
        else:    
            if self.blur_downsample:
                x = self.blur(x)
            x = self.downsample(self.act(self.conv2(x)))
            statistics2 = torch.cat([x.mean(dim=[2,3]), x.std(dim=[2,3])], dim=1)
            
        style2 = self.style_mapping2(statistics2)
        
        if not self.final:
            x = self.in2(x)
        
        return x, style1, style2
    
    
class GeneratorBlock(nn.Module):
    def __init__(self, inp_c, oup_c, code, initial=False, blur_upsample=False):
        super().__init__()
                
        self.initial = initial
        self.blur_upsample = blur_upsample

        
        # Learnable noise coefficients
        self.B1 = set_scale(IntermediateNoise(inp_c))
        self.B2 = set_scale(IntermediateNoise(oup_c))
        
        # Each Ada IN contains learnable parameters A
        self.ada_in1 = AdaIN(inp_c, code)
        self.ada_in2 = AdaIN(oup_c, code)
        
        # In case if it is the initial block, learnable constant is created
        if self.initial:
            self.constant = nn.Parameter(torch.randn(1, inp_c, 4, 4), requires_grad=True)
        else:
            self.conv1 = ScaledConv2d(inp_c, inp_c, kernel_size=3, padding=1)
            
        self.conv2 = ScaledConv2d(inp_c, oup_c, kernel_size=3, padding=1)
        
        self.upsample = nn.UpsamplingBilinear2d(scale_factor=2)
        self.blur = Blur(inp_c)
        self.activation = nn.LeakyReLU(0.2)
        
    def forward(self, x, w):
        """
        x - (N x C x H x W)
        w - (N x C), where A: (N x C) -> (N x (C x 2))
        """
        if self.initial:
            x = x.repeat(w.shape[0], 1, 1, 1)
        else:
            x = self.upsample(x)
            x = self.conv1(x)
            
            if self.blur_upsample:
                x = self.blur(x)
            
            x = self.activation(x)    
            
        x = self.B1(x)
        
        x = self.ada_in1(x, w)
        x = self.activation(self.conv2(x))
        
        x = self.B2(x)
            
        return self.ada_in2(x, w)
    
####################################################################################################################
################################################## Level 3 blocks ##################################################
####################################################################################################################
    
    
class MapingNetwork(nn.Module):
    def __init__(self, code=512, d=4):
        super().__init__()
        self.code = code
        self.act = nn.LeakyReLU(0.2)
        
        self.f = [BallProjection()]
        for _ in range(d-1):
            self.f.extend([ScaledLinear(code, code), nn.LeakyReLU(0.2)])
        self.f = self.f + [ScaledLinear(code, code)]
        self.f = nn.Sequential(*self.f)
    
    def forward(self, z1, scale, z2=None, p_mix=0.9):
        """
        Outputs latent code of size (bs x n_blocks x latent_code_size), performing style mixing
        """
        n_blocks = int(log2(scale) - 1)
        
        # Make latent code of style (bs x n_blocks x latent_code_size)
        style1 = self.f(z1)[:, None, :].repeat(1, n_blocks, 1)
        
        # Randomly decide if style mixing should be performed or not
        if (random.random() < p_mix) & (z2 is not None) & (n_blocks!=1):
            style2 = self.f(z2)[:, None, :].repeat(1, n_blocks, 1)
            layer_idx = torch.arange(n_blocks)[None, :, None].to(z1.device)
            mixing_cutoff = random.randint(1, n_blocks-1) #Insert style2 in 8x8 ... 1024x1024 blocks
            return torch.where(layer_idx < mixing_cutoff, style1, style2)
        else:
            return style1 # If style2 is not used
        
                       
class Discriminator(nn.Module):
    def __init__(self, code=512, d=3):
        super().__init__()
        
        self.disc = []
        for index in range(d - 1):            
            self.disc.extend([ScaledLinear(code, code), nn.LeakyReLU(0.2)])
        self.disc = self.disc + [ScaledLinear(code, 1)]
        self.disc = nn.Sequential(*self.disc)
                
    def forward(self, x):
        return self.disc(x)
    
    
class Encoder(nn.Module):
    def __init__(self, max_fm, code, fc_intital=True, blur_downsample=False):
        super().__init__()
        
        self.code = code       
        self.encoder = nn.ModuleList([
                                     EncoderBlock(max_fm//4, max_fm//2, code, final=False, blur_downsample=blur_downsample), #128
                                     EncoderBlock(max_fm//2, max_fm//2, code, final=False, blur_downsample=blur_downsample), #64
                                     EncoderBlock(max_fm//2, max_fm, code, final=False, blur_downsample=blur_downsample),    #32
                                     EncoderBlock(max_fm, max_fm, code, final=False, blur_downsample=blur_downsample),       #16
                                     EncoderBlock(max_fm, max_fm, code, final=False, blur_downsample=blur_downsample),       #8
                                     EncoderBlock(max_fm, max_fm, code, final=fc_intital, blur_downsample=blur_downsample),  #4
                                     ])
        
        self.fromRGB =  nn.ModuleList([FromRGB(3, max_fm//4),     #128
                                       FromRGB(3, max_fm//2),     #64
                                       FromRGB(3, max_fm//2),     #32
                                       FromRGB(3, max_fm),        #16
                                       FromRGB(3, max_fm),        #8
                                       FromRGB(3, max_fm)])       #4
        
    def forward(self, x, alpha=1.):
        n_blocks = int(log2(x.shape[-1]) - 1) # Compute the number of required blocks

        # In case of the first block, there is no blending, just return RGB image
        if n_blocks == 1:
            _, w1, w2 = self.encoder[-1](self.fromRGB[-1](x, downsample=False))
            return w1 + w2
            
        # Store w
        w = torch.zeros(x.shape[0], self.code).to(x.device)
        
        # Convert input from RGB and blend across 2 scales
        if alpha < 1:
            inp_top, w1, w2 = self.encoder[-n_blocks](self.fromRGB[-n_blocks](x, downsample=False))
            inp_left = self.fromRGB[-n_blocks+1](x, downsample=True)
            x = inp_left.mul(1 - alpha) + inp_top.mul(alpha)
        
        else: # Use top shortcut
            x, w1, w2 = self.encoder[-n_blocks](self.fromRGB[-n_blocks](x, downsample=False))

        w += (w1 + w2)

        for index in range(-n_blocks + 1, 0):
            x, w1, w2 = self.encoder[index](x)
            w += (w1 + w2)

        return w

class StyleGenerator(nn.Module):
    def __init__(self, max_fm, code, blur_upsample=False):
        super().__init__()
        
        self.generator = nn.ModuleList([GeneratorBlock(max_fm, max_fm, code, initial=True, blur_upsample=blur_upsample),           #4
                                        GeneratorBlock(max_fm, max_fm, code, initial=False, blur_upsample=blur_upsample),          #8
                                        GeneratorBlock(max_fm, max_fm, code, initial=False, blur_upsample=blur_upsample),          #16
                                        GeneratorBlock(max_fm, max_fm//2, code, initial=False, blur_upsample=blur_upsample),       #32
                                        GeneratorBlock(max_fm//2, max_fm//2, code, initial=False, blur_upsample=blur_upsample),    #64
                                        GeneratorBlock(max_fm//2, max_fm//4, code, initial=False, blur_upsample=blur_upsample)])   #128
        
        self.toRGB =  nn.ModuleList([ToRGB(max_fm, 3),           #4
                                     ToRGB(max_fm, 3),           #8
                                     ToRGB(max_fm, 3),           #16
                                     ToRGB(max_fm//2, 3),        #32
                                     ToRGB(max_fm//2, 3),        #64
                                     ToRGB(max_fm//4, 3)])       #128
        
    def get_blocks_parameters(self):
        pars = []
        for block in self.generator:
            named_block = list(block.named_parameters())
            for index in range(len(named_block)):
                if 'ada_in' not in named_block[index][0]:
                    pars.append(named_block[index][1])
        return pars
    
    def get_styles_parameters(self):
        # Get modules, corresponding to mapping from latent codes to Feature map's channel-wise coefficients
        return nn.ModuleList([module.ada_in1.A for module in self.generator] + \
                             [module.ada_in2.A for module in self.generator]).parameters()
    
    def ema(self, model, beta=0.999):
        """
        If Generator is used in running average regime, takes optimized model during training and
        adds it's weights into a linear combination
        """        
        runing_parameters = dict(self.named_parameters())
        for key in runing_parameters.keys():
            runing_parameters[key].data.mul_(beta).add_(1 - beta, dict(model.named_parameters())[key].data)
        
    def forward(self, w, scale, alpha=1):
        n_blocks = int(log2(scale) - 1) # Compute the number of required blocks
                
        # Take learnable constant as an input
        inp = self.generator[0].constant
        
        # In case of the first block, there is no blending, just return RGB image
        if n_blocks == 1:
            return self.toRGB[0](self.generator[0](inp, w[:, 0]), upsample=False)

        # If scale >= 8
        activations_2x = []
        for index in range(n_blocks):
            inp = self.generator[index](inp, w[:, index])

            # Save last 2 scales
            if index in [n_blocks-2, n_blocks-1]:
                activations_2x.append(inp)

        inp = self.toRGB[n_blocks-1](activations_2x[1], upsample=False)
        if alpha < 1: # In case if blending is applied            
            inp = (1 - alpha) * self.toRGB[n_blocks-2](activations_2x[0], upsample=True) + alpha * inp
        return inp