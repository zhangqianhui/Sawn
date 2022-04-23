import re
import torch
import torch.nn as nn
import functools
from torch.optim import lr_scheduler
import torch.nn.functional as F
from model.networks.gfla.resample2d import Resample2d
from model.networks.gfla.block_extractor  import BlockExtractor
from model.networks.gfla.local_attn_reshape   import LocalAttnReshape
from torch.nn.utils.spectral_norm import spectral_norm as SpectralNorm

######################################################################################
# base function for network structure
######################################################################################

class ResBlock3DEncoder(nn.Module):
    """
    Define a decoder block
    """
    def __init__(self, input_nc, output_nc, hidden_nc=None, norm_layer=nn.BatchNorm2d, nonlinearity= nn.LeakyReLU(),
                 use_spect=False, use_coord=False):
        super(ResBlock3DEncoder, self).__init__()

        hidden_nc = input_nc if hidden_nc is None else hidden_nc

        conv1 = spectral_norm(nn.Conv3d(input_nc,  hidden_nc, kernel_size=(3,3,3), stride=(1,1,1), padding=(1,1,1)), use_spect)
        conv2 = spectral_norm(nn.Conv3d(hidden_nc, output_nc, kernel_size=(3,4,4), stride=(1,2,2), padding=(0,1,1)), use_spect)
        bypass = spectral_norm(nn.Conv3d(input_nc, output_nc, kernel_size=1, stride=1, padding=0), use_spect)

        if type(norm_layer) == type(None):
            self.model = nn.Sequential(nonlinearity, conv1, nonlinearity, conv2,)
        else:
            self.model = nn.Sequential(norm_layer(input_nc), nonlinearity, conv1, 
                                       norm_layer(hidden_nc), nonlinearity, conv2,)

        self.shortcut = nn.Sequential(nn.AvgPool3d(kernel_size=(3,2,2), stride=(1,2,2)),bypass)

    def forward(self, x):
        out = self.model(x) + self.shortcut(x)
        return out  

class ADAIN(nn.Module):
    def __init__(self, norm_nc, feature_nc):
        super().__init__()

        self.param_free_norm = nn.InstanceNorm2d(norm_nc, affine=False)

        # The dimension of the intermediate embedding space. Yes, hardcoded.
        nhidden = 128
        use_bias=True

        self.mlp_shared = nn.Sequential(
            nn.Linear(feature_nc, nhidden, bias=use_bias),            
            nn.ReLU()
        )
        self.mlp_gamma = nn.Linear(nhidden, norm_nc, bias=use_bias)    
        self.mlp_beta = nn.Linear(nhidden, norm_nc, bias=use_bias)    

    def forward(self, x, feature):

        # Part 1. generate parameter-free normalized activations
        normalized = self.param_free_norm(x)
        # Part 2. produce scaling and bias conditioned on feature
        feature = feature.view(feature.size(0), -1)
        actv = self.mlp_shared(feature)
        gamma = self.mlp_gamma(actv)
        beta = self.mlp_beta(actv)

        # apply scale and bias
        gamma = gamma.view(*gamma.size()[:2], 1,1)
        beta = beta.view(*beta.size()[:2], 1,1)
        out = normalized * (1 + gamma) + beta

        return out

class Sawn(nn.Module):
    def __init__(self, num_features, eps=1e-5, activation='ReLU', param_free_norm_type='in'):
        super(Sawn, self).__init__()
        self.num_features = num_features
        self.eps = eps
        # weight and bias are dynamically assigned

        self.weight = None
        self.bias = None

        self.flow = None
        self.mask = None

        self.is_texturet = False

        self.weight_ref = None
        self.bias_ref = None

        self.flow_ref = None
        self.mask_ref = None
        self.seg = None

        self.kz = 5
        nonlinearity = get_nonlinearity_layer(activation_type=activation)
        attn1 = ExtractorAttn(num_features, self.kz, nonlinearity, softmax=True)
        setattr(self, 'attn', attn1)
        if param_free_norm_type == 'in':
            self.param_free_norm = nn.InstanceNorm2d(num_features, affine=False)
        elif param_free_norm_type == 'batch':
            self.param_free_norm = nn.BatchNorm2d(num_features, affine=False)
        else:
            raise ValueError('%s is not a recognized param-free norm type in SPADE'
                             % param_free_norm_type)

    def forward(self, x):

        model = getattr(self, 'attn')
        #assert self.weight is not None and self.bias is not None, "Please assign weight and bias before calling AdaIN!"
        normalized = self.param_free_norm(x)
        if self.is_texturet:
            weight_ = model(self.weight, x, self.flow)
            if self.mask is not None:
                weight_ = self.mask * weight_ + (1 - self.mask) * x
            bias_ = model(self.bias, x, self.flow)
            normalized_ = weight_ * normalized + bias_

        weight_warped = model(self.weight_ref, x, self.flow_ref)
        if self.mask_ref is not None:
            weight_warped = self.mask_ref * weight_warped + (1 - self.mask_ref) * x

        bias_warped = model(self.bias_ref, x, self.flow_ref)
        normalized_ref = normalized * weight_warped + bias_warped

        if self.is_texturet:
            _, _, h, w = x.size()
            self.seg = torch.nn.functional.interpolate(self.seg, (h, w))
            normalized_ref = self.seg * normalized_ref + (1 - self.seg) * normalized_

        return normalized_ref

    def __repr__(self):
        return self.__class__.__name__ + '(' + str(self.num_features) + ')'

def get_norm_layer(norm_type='batch'):
    """Get the normalization layer for the networks"""
    if norm_type == 'batch':
        norm_layer = functools.partial(nn.BatchNorm2d, momentum=0.1, affine=True)
    elif norm_type == 'instance':
        norm_layer = functools.partial(nn.InstanceNorm2d, affine=True)
    elif norm_type == 'adain':
        norm_layer = functools.partial(ADAIN)
    elif norm_type == 'Sawn':
        norm_layer = functools.partial(Sawn)
    elif norm_type == 'none':
        norm_layer = None
    else:
        raise NotImplementedError('normalization layer [%s] is not found' % norm_type)

    if norm_type != 'none':
        norm_layer.__name__ = norm_type

    return norm_layer


def get_nonlinearity_layer(activation_type='PReLU'):
    """Get the activation layer for the networks"""
    if activation_type == 'ReLU':
        nonlinearity_layer = nn.ReLU()
    elif activation_type == 'SELU':
        nonlinearity_layer = nn.SELU()
    elif activation_type == 'LeakyReLU':
        nonlinearity_layer = nn.LeakyReLU(0.1)
    elif activation_type == 'PReLU':
        nonlinearity_layer = nn.PReLU()
    else:
        raise NotImplementedError('activation layer [%s] is not found' % activation_type)
    return nonlinearity_layer


def get_scheduler(optimizer, opt):
    """Get the training learning rate for different epoch"""
    if opt.lr_policy == 'lambda':
        def lambda_rule(epoch):
            lr_l = 1.0 - max(0, epoch+1+1+opt.iter_count-opt.niter) / float(opt.niter_decay+1)
            return lr_l
        scheduler = lr_scheduler.LambdaLR(optimizer, lr_lambda=lambda_rule)
    elif opt.lr_policy == 'step':
        scheduler = lr_scheduler.StepLR(optimizer, step_size=opt.lr_decay_iters, gamma=0.1)
    elif opt.lr_policy == 'exponent':
        scheduler = lr_scheduler.ExponentialLR(optimizer, gamma=0.95)
    else:
        raise NotImplementedError('learning rate policy [%s] is not implemented', opt.lr_policy)
    return scheduler


def print_network(net):
    """print the network"""
    num_params = 0
    for param in net.parameters():
        num_params += param.numel()
    print(net)
    print('total number of parameters: %.3f M' % (num_params/1e6))


def init_net(net, init_type='normal', activation='relu', gpu_ids=[]):
    """print the network structure and initial the network"""
    print_network(net)

    if len(gpu_ids) > 0:
        assert(torch.cuda.is_available())
        net.cuda()
        net = torch.nn.DataParallel(net, gpu_ids)
    init_weights(net, init_type)
    return net


def _freeze(*args):
    """freeze the network for forward process"""
    for module in args:
        if module:
            for p in module.parameters():
                p.requires_grad = False


def _unfreeze(*args):
    """ unfreeze the network for parameter update"""
    for module in args:
        if module:
            for p in module.parameters():
                p.requires_grad = True

def spectral_norm(module, use_spect=True):
    """use spectral normal layer to stable the training process"""
    if use_spect:
        return SpectralNorm(module)
    else:
        return module

class Conv2dBlock(nn.Module):
    def __init__(self, input_dim ,output_dim, kernel_size, stride,
                 padding=0, norm='none', activation='relu', pad_type='zero'):
        super(Conv2dBlock, self).__init__()
        self.use_bias = True
        # initialize padding
        if pad_type == 'reflect':
            self.pad = nn.ReflectionPad2d(padding)
        elif pad_type == 'replicate':
            self.pad = nn.ReplicationPad2d(padding)
        elif pad_type == 'zero':
            self.pad = nn.ZeroPad2d(padding)
        else:
            assert 0, "Unsupported padding type: {}".format(pad_type)
        # initialize normalization
        norm_dim = output_dim
        if norm == 'bn':
            self.norm = nn.BatchNorm2d(norm_dim)
        elif norm == 'in':
            #self.norm = nn.InstanceNorm2d(norm_dim, track_running_stats=True)
            self.norm = nn.InstanceNorm2d(norm_dim, affine=True)
        elif norm == 'none' or norm == 'sn':
            self.norm = None
        else:
            assert 0, "Unsupported normalization: {}".format(norm)

        # initialize activation
        if activation == 'relu':
            self.activation = nn.ReLU(inplace=True)
        elif activation == 'lrelu':
            self.activation = nn.LeakyReLU(0.2, inplace=True)
        elif activation == 'prelu':
            self.activation = nn.PReLU()
        elif activation == 'selu':
            self.activation = nn.SELU(inplace=True)
        elif activation == 'tanh':
            self.activation = nn.Tanh()
        elif activation == 'none':
            self.activation = None
        else:
            assert 0, "Unsupported activation: {}".format(activation)

        # initialize convolution
        if norm == 'sn':
            self.conv = SpectralNorm(nn.Conv2d(input_dim, output_dim, kernel_size, stride, bias=self.use_bias))
        else:
            self.conv = nn.Conv2d(input_dim, output_dim, kernel_size, stride, bias=self.use_bias)

    def forward(self, x):
        x = self.conv(self.pad(x))
        if self.norm:
            x = self.norm(x)
        if self.activation:
            x = self.activation(x)
        return x

def coord_conv(input_nc, output_nc, use_spect=False, use_coord=False, with_r=False, **kwargs):
    """use coord convolution layer to add position information"""
    if use_coord:
        return CoordConv(input_nc, output_nc, with_r, use_spect, **kwargs)
    else:
        return spectral_norm(nn.Conv2d(input_nc, output_nc, **kwargs), use_spect)


######################################################################################
# Network basic function
######################################################################################
class AddCoords(nn.Module):
    """
    Add Coords to a tensor
    """
    def __init__(self, with_r=False):
        super(AddCoords, self).__init__()
        self.with_r = with_r

    def forward(self, x):
        """
        :param x: shape (batch, channel, x_dim, y_dim)
        :return: shape (batch, channel+2, x_dim, y_dim)
        """
        B, _, x_dim, y_dim = x.size()

        # coord calculate
        xx_channel = torch.arange(x_dim).repeat(B, 1, y_dim, 1).type_as(x)
        yy_cahnnel = torch.arange(y_dim).repeat(B, 1, x_dim, 1).permute(0, 1, 3, 2).type_as(x)
        # normalization
        xx_channel = xx_channel.float() / (x_dim-1)
        yy_cahnnel = yy_cahnnel.float() / (y_dim-1)
        xx_channel = xx_channel * 2 - 1
        yy_cahnnel = yy_cahnnel * 2 - 1

        ret = torch.cat([x, xx_channel, yy_cahnnel], dim=1)

        if self.with_r:
            rr = torch.sqrt(xx_channel ** 2 + yy_cahnnel ** 2)
            ret = torch.cat([ret, rr], dim=1)

        return ret

class CoordConv(nn.Module):
    """
    CoordConv operation
    """
    def __init__(self, input_nc, output_nc, with_r=False, use_spect=False, **kwargs):
        super(CoordConv, self).__init__()
        self.addcoords = AddCoords(with_r=with_r)
        input_nc = input_nc + 2
        if with_r:
            input_nc = input_nc + 1
        self.conv = spectral_norm(nn.Conv2d(input_nc, output_nc, **kwargs), use_spect)

    def forward(self, x):
        ret = self.addcoords(x)
        ret = self.conv(ret)

        return ret

class EncoderBlock(nn.Module):
    def __init__(self, input_nc, output_nc, norm_layer=nn.BatchNorm2d, nonlinearity=nn.LeakyReLU(),
                 use_spect=False, use_coord=False, stride=2, kernel_size=3):
        super(EncoderBlock, self).__init__()

        kwargs_down = {'kernel_size': kernel_size, 'stride': stride, 'padding': 1}
        kwargs_fine = {'kernel_size': 3, 'stride': 1, 'padding': 1}

        conv1 = coord_conv(input_nc,  output_nc, use_spect, use_coord, **kwargs_down)
        conv2 = coord_conv(output_nc, output_nc, use_spect, use_coord, **kwargs_fine)

        if type(norm_layer) == type(None):
            self.model = nn.Sequential(nonlinearity, conv1, nonlinearity, conv2,)
        else:
            self.model = nn.Sequential(norm_layer(input_nc),  nonlinearity, conv1, 
                                       norm_layer(output_nc), nonlinearity, conv2,)

    def forward(self, x):
        out = self.model(x)
        return out

class ResBlock(nn.Module):
    """
    Define an Residual block for different types
    """
    def __init__(self, input_nc, output_nc=None, hidden_nc=None, norm_layer=nn.BatchNorm2d, nonlinearity= nn.LeakyReLU(),
                learnable_shortcut=False, use_spect=False, use_coord=False):
        super(ResBlock, self).__init__()

        hidden_nc = input_nc if hidden_nc is None else hidden_nc
        output_nc = input_nc if output_nc is None else output_nc
        self.learnable_shortcut = True if input_nc != output_nc else learnable_shortcut

        kwargs = {'kernel_size': 3, 'stride': 1, 'padding': 1}
        kwargs_short = {'kernel_size': 1, 'stride': 1, 'padding': 0}

        conv1 = coord_conv(input_nc, hidden_nc, use_spect, use_coord, **kwargs)
        conv2 = coord_conv(hidden_nc, output_nc, use_spect, use_coord, **kwargs)

        if type(norm_layer) == type(None):
            self.model = nn.Sequential(nonlinearity, conv1, nonlinearity, conv2,)
        else:
            self.model = nn.Sequential(norm_layer(input_nc), nonlinearity, conv1, 
                                       norm_layer(hidden_nc), nonlinearity, conv2,)

        if self.learnable_shortcut:
            bypass = coord_conv(input_nc, output_nc, use_spect, use_coord, **kwargs_short)
            self.shortcut = nn.Sequential(bypass,)

    def forward(self, x):
        if self.learnable_shortcut:
            out = self.model(x) + self.shortcut(x)
        else:
            out = self.model(x) + x
        return out

class ResBlocks(nn.Module):
    """docstring for ResBlocks"""
    def __init__(self, num_blocks, input_nc, output_nc=None, hidden_nc=None, norm_layer=nn.BatchNorm2d, 
                 nonlinearity= nn.LeakyReLU(), learnable_shortcut=False, use_spect=False, use_coord=False):
        super(ResBlocks, self).__init__()
        hidden_nc = input_nc if hidden_nc is None else hidden_nc
        output_nc = input_nc if output_nc is None else output_nc

        self.model=[]
        if num_blocks == 1:
            self.model += [ResBlock(input_nc, output_nc, hidden_nc, 
                           norm_layer, nonlinearity, learnable_shortcut, use_spect, use_coord)]

        else:
            self.model += [ResBlock(input_nc, hidden_nc, hidden_nc, 
                           norm_layer, nonlinearity, learnable_shortcut, use_spect, use_coord)]
            for i in range(num_blocks-2):
                self.model += [ResBlock(hidden_nc, hidden_nc, hidden_nc, 
                               norm_layer, nonlinearity, learnable_shortcut, use_spect, use_coord)]
            self.model += [ResBlock(hidden_nc, output_nc, hidden_nc, 
                            norm_layer, nonlinearity, learnable_shortcut, use_spect, use_coord)]

        self.model = nn.Sequential(*self.model)

    def forward(self, inputs):
        return self.model(inputs)

class ResBlockDecoder(nn.Module):
    """
    Define a decoder block
    """
    def __init__(self, input_nc, output_nc, hidden_nc=None, norm_layer=nn.BatchNorm2d, nonlinearity= nn.LeakyReLU(),
                 use_spect=False, use_coord=False):
        super(ResBlockDecoder, self).__init__()

        hidden_nc = input_nc if hidden_nc is None else hidden_nc

        conv1 = spectral_norm(nn.Conv2d(input_nc, hidden_nc, kernel_size=3, stride=1, padding=1), use_spect)
        conv2 = spectral_norm(nn.ConvTranspose2d(hidden_nc, output_nc, kernel_size=3, stride=2, padding=1, output_padding=1), use_spect)
        bypass = spectral_norm(nn.ConvTranspose2d(input_nc, output_nc, kernel_size=3, stride=2, padding=1, output_padding=1), use_spect)

        if type(norm_layer) == type(None):
            self.model = nn.Sequential(nonlinearity, conv1, nonlinearity, conv2,)
        else:
            self.model = nn.Sequential(norm_layer(input_nc), nonlinearity, conv1, norm_layer(hidden_nc), nonlinearity, conv2,)

        self.shortcut = nn.Sequential(bypass)

    def forward(self, x):
        out = self.model(x) + self.shortcut(x)
        return out

class ResBlockEncoder(nn.Module):
    """
    Define a decoder block
    """
    def __init__(self, input_nc, output_nc, hidden_nc=None, norm_layer=nn.BatchNorm2d, nonlinearity= nn.LeakyReLU(),
                 use_spect=False, use_coord=False):
        super(ResBlockEncoder, self).__init__()

        hidden_nc = input_nc if hidden_nc is None else hidden_nc

        conv1 = spectral_norm(nn.Conv2d(input_nc, hidden_nc, kernel_size=3, stride=1, padding=1), use_spect)
        conv2 = spectral_norm(nn.Conv2d(hidden_nc, output_nc, kernel_size=4, stride=2, padding=1), use_spect)
        bypass = spectral_norm(nn.Conv2d(input_nc, output_nc, kernel_size=1, stride=1, padding=0), use_spect)

        if type(norm_layer) == type(None):
            self.model = nn.Sequential(nonlinearity, conv1, nonlinearity, conv2,)
        else:
            self.model = nn.Sequential(norm_layer(input_nc), nonlinearity, conv1, 
                                       norm_layer(hidden_nc), nonlinearity, conv2,)
        self.shortcut = nn.Sequential(nn.AvgPool2d(kernel_size=2, stride=2),bypass)

    def forward(self, x):
        out = self.model(x) + self.shortcut(x)
        return out

class Output(nn.Module):
    """
    Define the output layer
    """
    def __init__(self, input_nc, output_nc, kernel_size = 3, norm_layer=nn.BatchNorm2d, nonlinearity= nn.LeakyReLU(),
                 use_spect=False, use_coord=False):
        super(Output, self).__init__()

        kwargs = {'kernel_size': kernel_size, 'padding':0, 'bias': True}

        self.conv1 = coord_conv(input_nc, output_nc, use_spect, use_coord, **kwargs)

        if type(norm_layer) == type(None):
            self.model = nn.Sequential(nonlinearity, nn.ReflectionPad2d(int(kernel_size/2)), self.conv1, nn.Tanh())
        else:
            self.model = nn.Sequential(norm_layer(input_nc), nonlinearity, nn.ReflectionPad2d(int(kernel_size / 2)), self.conv1, nn.Tanh())

    def forward(self, x):
        out = self.model(x)

        return out
        
class Jump(nn.Module):
    """
    Define the output layer
    """
    def __init__(self, input_nc, output_nc, kernel_size = 3, norm_layer=nn.BatchNorm2d, nonlinearity= nn.LeakyReLU(),
                 use_spect=False, use_coord=False):
        super(Jump, self).__init__()

        kwargs = {'kernel_size': kernel_size, 'padding':0, 'bias': True}

        self.conv1 = coord_conv(input_nc, output_nc, use_spect, use_coord, **kwargs)

        if type(norm_layer) == type(None):
            self.model = nn.Sequential(nonlinearity, nn.ReflectionPad2d(int(kernel_size/2)), self.conv1)
        else:
            self.model = nn.Sequential(norm_layer(input_nc), nonlinearity, nn.ReflectionPad2d(int(kernel_size / 2)), self.conv1)

    def forward(self, x):
        out = self.model(x)
        return out

class ExtractorAttn(nn.Module):
    def __init__(self, feature_nc, kernel_size=4, nonlinearity=nn.LeakyReLU(), softmax=None):
        super(ExtractorAttn, self).__init__()
        self.kernel_size=kernel_size
        hidden_nc = 128
        softmax = nonlinearity if softmax is None else nn.Softmax(dim=1)
        self.extractor = BlockExtractor(kernel_size=kernel_size)
        self.reshape = LocalAttnReshape()
        self.fully_connect_layer = nn.Sequential(
                nn.Conv2d(2*feature_nc, hidden_nc, kernel_size=kernel_size, stride=kernel_size, padding=0),
                nonlinearity,
                nn.Conv2d(hidden_nc, kernel_size*kernel_size, kernel_size=1, stride=1, padding=0),
                softmax,)        
    def forward(self, source, target, flow_field):
        block_source = self.extractor(source, flow_field)
        block_target = self.extractor(target, torch.zeros_like(flow_field))
        attn_param = self.fully_connect_layer(torch.cat((block_target, block_source), 1))
        attn_param = self.reshape(attn_param, self.kernel_size)
        result = torch.nn.functional.avg_pool2d(attn_param*block_source, self.kernel_size, self.kernel_size)
        return result

    def hook_attn_param(self, source, target, flow_field):
        block_source = self.extractor(source, flow_field)
        block_target = self.extractor(target, torch.zeros_like(flow_field))
        attn_param_ = self.fully_connect_layer(torch.cat((block_target, block_source), 1))
        attn_param = self.reshape(attn_param_, self.kernel_size)
        result = torch.nn.functional.avg_pool2d(attn_param*block_source, self.kernel_size, self.kernel_size)
        return attn_param_, result

class LinearBlock(nn.Module):
    def __init__(self, input_nc, output_nc, norm_layer=nn.BatchNorm2d,  nonlinearity= nn.LeakyReLU(),
                 use_spect=False):
        super(LinearBlock, self).__init__()
        use_bias = True

        self.fc = spectral_norm(nn.Linear(input_nc, output_nc, bias=use_bias), use_spect)

        if type(norm_layer) == type(None):
            self.model = nn.Sequential(nonlinearity, self.fc)
        else:
            self.model = nn.Sequential(norm_layer(input_nc), nonlinearity, self.fc)


    def forward(self, x):
        out = self.model(x)
        return out
