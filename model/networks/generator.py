import re
import torch
import torch.nn as nn
from model.networks.base_network import BaseNetwork
from model.networks.base_function import *

class PoseGenerator(BaseNetwork):
    def __init__(self,  image_nc=3, structure_nc=18, output_nc=3, ngf=64, img_f=1024, layers=6, num_blocks=2, 
                norm='batch', activation='ReLU', attn_layer=[1,2], extractor_kz={'1':5,'2':5}, use_spect=True, use_coord=False):  
        super(PoseGenerator, self).__init__()

        self.decoder = Decoder(output_nc, ngf, img_f, layers, num_blocks,
                                                norm, activation, attn_layer, use_spect, use_coord)
        self.pose_encoder = PoseEncoder(structure_nc, ngf, img_f, layers, num_blocks,
                                                norm, activation, attn_layer, use_spect, use_coord)
        self.flow_net = PoseFlowNet(image_nc, structure_nc, ngf=32, img_f=256, encoder_layer=5, 
                                    attn_layer=attn_layer, norm=norm, activation=activation,
                                    use_spect=use_spect, use_coord=use_coord)

        self.style_encoder = StyleEncoder(image_nc, ngf, img_f, layers, norm, activation, use_spect, use_coord,
                                    styles_list=self.get_num_flow_params(self.decoder), attn_layer=attn_layer)

    def forward(self, source, source_B, target_B, source_S):

        feature_list = self.style_encoder(source, source_S)
        posecode = self.pose_encoder(target_B)
        flow_fields, masks = self.flow_net(source, source_B, target_B)

        self.assign_in_flow_params(flow_fields, self.decoder, masks)
        self.assign_in_feats_params(feature_list, self.decoder)

        image_gen = self.decoder(posecode)

        return image_gen, flow_fields, masks

    def assign_in_flow_params(self, flow, model, masks=None):
        i = 0
        for m in model.modules():
            if m.__class__.__name__ == "Sawn":
                m.flow_ref = flow[i//2]
                if masks is not None:
                    m.mask_ref = masks[i//2]
                i += 1

    def assign_in_feats_params(self, adain_params, model):
        i = 0
        for m in model.modules():
            if m.__class__.__name__ == "Sawn":
                adain_param = adain_params[i // 2]
                mean = adain_param[:, :m.num_features, :,:]
                std = adain_param[:, m.num_features:2*m.num_features, :, :]
                m.bias_ref = mean
                m.weight_ref = std
                i += 1

    def get_num_flow_params(self, model):
        num_params = []
        i = 0
        for m in model.modules():
            if m.__class__.__name__ == "Sawn":
                if i % 2:
                    num_params.append(m.num_features * 2)
                i += 1
        return num_params

# for texture transfer
class PoseGeneratorTexture(BaseNetwork):

    def __init__(self,  image_nc=3, structure_nc=18, output_nc=3, ngf=64, img_f=1024, layers=6, num_blocks=2,
                norm='batch', activation='ReLU', attn_layer=[1,2], extractor_kz={'1':5,'2':5}, use_spect=True, use_coord=False):
        super(PoseGeneratorTexture, self).__init__()

        self.decoder = Decoder(output_nc, ngf, img_f, layers, num_blocks,
                                                norm, activation, attn_layer, use_spect, use_coord)
        self.pose_encoder = PoseEncoder(structure_nc, ngf, img_f, layers, num_blocks,
                                                norm, activation, attn_layer, use_spect, use_coord)
        self.flow_net = PoseFlowNet(image_nc, structure_nc, ngf=32, img_f=256, encoder_layer=5,
                                    attn_layer=attn_layer, norm=norm, activation=activation,
                                    use_spect=use_spect, use_coord=use_coord)

        self.style_encoder = StyleEncoderTexture(image_nc, ngf, img_f, layers, norm, activation, use_spect, use_coord,
                                    styles_list=self.get_num_Inflow_params(self.decoder), attn_layer=attn_layer)

    def forward(self, source, source_B, source_S, source_ref, source_ref_B, source_ref_S, is_texturet=True, seg_label=4):

        feature_list, _ = self.style_encoder(source, source_S, source_ref, source_ref_S, 10)
        feature_list_ref, seg = self.style_encoder(source, source_S, source_ref, source_ref_S, seg_label)

        flow_fields, masks = self.flow_net(source, source_B, source_B)
        flow_fields_ref, masks_ref = self.flow_net(source, source_ref_B, source_B)

        self.assign_inflow_params(self.decoder, flow_fields, masks, flow_fields_ref, masks_ref, seg, is_texturet=is_texturet)
        self.assign_inflow_feats_params(feature_list, feature_list_ref, self.decoder)

        posecode = self.pose_encoder(source_B)
        image_gen = self.decoder(posecode)

        return image_gen, flow_fields, flow_fields_ref, masks

    def assign_inflow_params(self, model, flow, masks=None, flow_ref=None, masks_ref=None, seg=None, is_texturet=False):

        i = 0
        for m in model.modules():
            if m.__class__.__name__ == "Sawn":
                m.is_texturet = is_texturet
                m.flow = flow[i//2]
                m.flow_ref = flow_ref[i//2]
                m.seg = seg
                if masks is not None:
                    m.mask = masks[i//2]
                    m.mask_ref = masks_ref[i//2]
                i += 1

    def assign_inflow_feats_params(self, adain_params, adain_params_ref, model):

        i = 0
        for m in model.modules():
            if m.__class__.__name__ == "Sawn":

                adain_param = adain_params[i // 2]
                adain_param_ref = adain_params_ref[i // 2]

                mean = adain_param[:, :m.num_features, :,:]
                std = adain_param[:, m.num_features:2*m.num_features, :, :]
                m.bias = mean
                m.weight = std

                mean = adain_param_ref[:, :m.num_features, :,:]
                std = adain_param_ref[:, m.num_features:2*m.num_features, :, :]
                m.bias_ref = mean
                m.weight_ref = std

                i += 1

    def get_num_Inflow_params(self, model):
        num_params = []
        i = 0
        for m in model.modules():
            if m.__class__.__name__ == "Sawn":
                if i % 2:
                    num_params.append(m.num_features * 2)
                i += 1
        return num_params

class MLPConv(nn.Module):
    def __init__(self, input_dim, output_dim, norm='none', activ='relu'):

        super(MLPConv, self).__init__()
        self.model = []
        self.model += [Conv2dBlock(input_dim, input_dim, norm=norm, kernel_size=3, stride=1, activation=activ)]
        self.model += [Conv2dBlock(input_dim, output_dim, norm=norm, kernel_size=3, stride=1, activation='none')] # no output activations
        self.model = nn.Sequential(*self.model)

    def forward(self, x):
        return self.model(x)

class StyleEncoder(BaseNetwork):

    def __init__(self, input_nc=3, ngf=64, img_f=1024, layers=6, norm='batch',
                activation='ReLU', use_spect=True, use_coord=False, styles_list=None, attn_layer=[1,2]):
        super(StyleEncoder, self).__init__()

        self.layers = layers
        self.attn_layer = attn_layer
        self.seg_channel = 8
        self.style_dim = 256
        norm_layer = get_norm_layer(norm_type=norm)
        nonlinearity = get_nonlinearity_layer(activation_type=activation)
        # encoder part CONV_BLOCKS
        self.block0 = EncoderBlock(input_nc, ngf, norm_layer,
                                 nonlinearity, use_spect, use_coord, stride=1)
        mult = 1
        for i in range(layers):
            mult_prev = mult
            mult = min(2 ** (i + 1), img_f//ngf)
            block = EncoderBlock(ngf*mult_prev, ngf*mult, norm_layer,
                                 nonlinearity, use_spect, use_coord, kernel_size=3)
            setattr(self, 'encoder' + str(i), block)
            if self.layers - i in self.attn_layer:
                MLPConv = EncoderBlock(ngf * mult, self.style_dim // self.seg_channel, norm_layer,
                                     nonlinearity, use_spect, use_coord, stride=1, kernel_size=3)
                setattr(self, 'encoder_mlp' + str(i), MLPConv)
        for i in range(len(styles_list)):
            MLPConv = nn.Conv2d(self.style_dim, list(styles_list)[i], kernel_size=3, stride=1, padding=1, bias=True)
            setattr(self, 'encoder_mlp2' + str(i), MLPConv)

    def texture_enc(self, source):
        attn_list = []
        out = self.block0(source)
        #feature_list.append(out)
        for i in range(self.layers):
            model = getattr(self, 'encoder' + str(i))
            out = model(out)
            if self.layers - i in self.attn_layer:
                mlp = getattr(self, 'encoder_mlp' + str(i))
                attn = mlp(out)
                attn_list.append(attn)
        attn_list = list(reversed(attn_list))

        return attn_list

    def forward(self, source, sem):

        attn_list = []
        for i in range(sem.size(1)):
            semi = sem[:, i, :, :]
            semi = torch.unsqueeze(semi, 1)
            semi = semi.repeat(1, source.size(1), 1, 1)
            xi = source.mul(semi)
            if i == 0:
                out = self.texture_enc(xi)
                attn_list = out
            else:
                out = self.texture_enc(xi)
                for i in range(len(attn_list)):
                    attn_list[i] = torch.cat([attn_list[i], out[i]], dim=1)

        params_list = []
        for i in range(len(attn_list)):
            mlp2 = getattr(self, 'encoder_mlp2' + str(i))
            params_list.append(mlp2(attn_list[i]))
        return params_list

class StyleEncoderTexture(BaseNetwork):
    def __init__(self, input_nc=3, ngf=64, img_f=1024, layers=6, norm='batch',
                activation='ReLU', use_spect=True, use_coord=False, styles_list=None, attn_layer=[1,2], seg_label=4):
        super(StyleEncoderTexture, self).__init__()

        self.layers = layers
        self.attn_layer = attn_layer
        self.seg_channel = 8
        self.style_dim = 256
        self.seg_label = seg_label
        norm_layer = get_norm_layer(norm_type=norm)
        nonlinearity = get_nonlinearity_layer(activation_type=activation)
        # encoder part CONV_BLOCKS
        self.block0 = EncoderBlock(input_nc, ngf, norm_layer,
                                 nonlinearity, use_spect, use_coord)
        mult = 1
        for i in range(layers):
            mult_prev = mult
            mult = min(2 ** (i + 1), img_f//ngf)
            block = EncoderBlock(ngf*mult_prev, ngf*mult, norm_layer,
                                 nonlinearity, use_spect, use_coord)
            setattr(self, 'encoder' + str(i), block)
            if self.layers - i in self.attn_layer:
                MLPConv = EncoderBlock(ngf * mult, self.style_dim // self.seg_channel, norm_layer,
                                     nonlinearity, use_spect, use_coord, stride=1)
                setattr(self, 'encoder_mlp' + str(i), MLPConv)
        for i in range(len(styles_list)):
            MLPConv = nn.Conv2d(self.style_dim, list(styles_list)[i], kernel_size=3, stride=1, padding=1, bias=True)
            setattr(self, 'encoder_mlp2' + str(i), MLPConv)

    def texture_enc(self, source):

        attn_list = []
        out = self.block0(source)
        #feature_list.append(out)
        for i in range(self.layers):
            model = getattr(self, 'encoder' + str(i))
            out = model(out)
            if self.layers - i in self.attn_layer:
                mlp = getattr(self, 'encoder_mlp' + str(i))
                attn = mlp(out)
                attn_list.append(attn)
        attn_list = list(reversed(attn_list))

        return attn_list

    def forward(self, source, sem, sourceB, semB, seg_label):

        attn_list = []
        # get the refer style code.
        if seg_label < semB.size(1):
            semiB = semB[:, seg_label, :, :]
            semiB = torch.unsqueeze(semiB, 1)
            #semiB = semiB.repeat(1, sourceB.size(1), 1, 1)
            xiB = sourceB.mul(semiB)
            attn_listB = self.texture_enc(xiB)

        for i in range(sem.size(1)):
            semi = sem[:, i, :, :]
            semi = torch.unsqueeze(semi, 1)
            semi = semi.repeat(1, source.size(1), 1, 1)
            xi = source.mul(semi)
            if i == 0:
                if seg_label == 0:
                    attn_list = attn_listB
                else:
                    out = self.texture_enc(xi)
                    attn_list = out
            else:
                if i == seg_label:
                    out = attn_listB
                else:
                    out = self.texture_enc(xi)
                for i in range(len(attn_list)):
                    attn_list[i] = torch.cat([attn_list[i], out[i]], dim=1)

        params_list = []
        for i in range(len(attn_list)):
            mlp2 = getattr(self, 'encoder_mlp2' + str(i))
            params_list.append(mlp2(attn_list[i]))
        return params_list, sem[:, seg_label:seg_label+1, :, :]

class PoseEncoder(BaseNetwork):

    def __init__(self, structure_nc=18, ngf=64, img_f=1024, layers=6, num_blocks=2,
                norm='batch', activation='ReLU', attn_layer=[1,2], use_spect=True, use_coord=False):
        super(PoseEncoder, self).__init__()

        self.layers = layers
        self.attn_layer = attn_layer

        norm_layer = get_norm_layer(norm_type=norm)
        norm_layer_ = get_norm_layer(norm_type='Sawn')
        nonlinearity = get_nonlinearity_layer(activation_type=activation)

        self.block0 = EncoderBlock(structure_nc, ngf, norm_layer,
                                 nonlinearity, use_spect, use_coord)
        mult = 1
        for i in range(layers - 1):
            mult_prev = mult
            mult = min(2 ** (i + 1), img_f//ngf)
            block = EncoderBlock(ngf*mult_prev, ngf*mult, norm_layer,
                                 nonlinearity, use_spect, use_coord)
            setattr(self, 'encoder' + str(i), block)

    def forward(self, target_B):

        out = self.block0(target_B)
        for i in range(self.layers-1):
            model = getattr(self, 'encoder' + str(i))
            out = model(out)
        return out

class Decoder(BaseNetwork):
    def __init__(self, output_nc=3, ngf=64, img_f=1024, layers=6, num_blocks=2,
                norm='batch', activation='ReLU', attn_layer=[1,2], use_spect=True, use_coord=False):
        super(Decoder, self).__init__()

        self.layers = layers
        self.attn_layer = attn_layer

        norm_layer = get_norm_layer(norm_type=norm)
        norm_layer_ = get_norm_layer(norm_type='Sawn')
        nonlinearity = get_nonlinearity_layer(activation_type=activation)

        # decoder part
        mult = min(2 ** (layers-1), img_f//ngf)
        for i in range(layers):
            mult_prev = mult
            mult = min(2 ** (layers-i-2), img_f//ngf) if i != layers-1 else 1
            if num_blocks == 1:
                up = nn.Sequential(ResBlockDecoder(ngf*mult_prev, ngf*mult, None, norm_layer,
                                         nonlinearity, use_spect, use_coord))
            else:
                if layers - i in attn_layer:
                    up = nn.Sequential(ResBlocks(num_blocks-1, ngf*mult_prev, None, None, norm_layer,
                                                 nonlinearity, False, use_spect, use_coord),
                                       ResBlockDecoder(ngf*mult_prev, ngf*mult, None, norm_layer_,
                                                 nonlinearity, use_spect, use_coord))
                else:
                    up = nn.Sequential(ResBlocks(num_blocks-1, ngf*mult_prev, None, None, norm_layer,
                                                 nonlinearity, False, use_spect, use_coord),
                                       ResBlockDecoder(ngf*mult_prev, ngf*mult, None, norm_layer,
                                                 nonlinearity, use_spect, use_coord))
            setattr(self, 'decoder' + str(i), up)
        self.outconv = Output(ngf, output_nc, 3, None, nonlinearity, use_spect, use_coord)

    def forward(self, out):

        for i in range(self.layers):
            model = getattr(self, 'decoder' + str(i))
            out = model(out)
        out_image = self.outconv(out)
        return out_image

class PoseFlowNet(nn.Module):
    """docstring for FlowNet"""
    def __init__(self, image_nc, structure_nc, ngf=64, img_f=1024, encoder_layer=5, attn_layer=[1], norm='batch',
                activation='ReLU', use_spect=True, use_coord=False):
        super(PoseFlowNet, self).__init__()

        self.encoder_layer = encoder_layer
        self.decoder_layer = encoder_layer - min(attn_layer)
        self.attn_layer = attn_layer
        norm_layer = get_norm_layer(norm_type=norm)
        nonlinearity = get_nonlinearity_layer(activation_type=activation)
        input_nc = 2*structure_nc + image_nc

        self.block0 = EncoderBlock(input_nc, ngf, norm_layer,
                                 nonlinearity, use_spect, use_coord)
        mult = 1
        for i in range(encoder_layer-1):
            mult_prev = mult
            mult = min(2 ** (i + 1), img_f//ngf)
            block = EncoderBlock(ngf*mult_prev, ngf*mult,  norm_layer,
                                 nonlinearity, use_spect, use_coord)
            setattr(self, 'encoder' + str(i), block)         
        
        for i in range(self.decoder_layer):
            mult_prev = mult
            mult = min(2 ** (encoder_layer-i-2), img_f//ngf) if i != encoder_layer-1 else 1
            up = ResBlockDecoder(ngf*mult_prev, ngf*mult, ngf*mult, norm_layer, 
                                    nonlinearity, use_spect, use_coord)
            setattr(self, 'decoder' + str(i), up)
            
            jumpconv = Jump(ngf*mult, ngf*mult, 3, None, nonlinearity, use_spect, use_coord)
            setattr(self, 'jump' + str(i), jumpconv)

            if encoder_layer-i-1 in attn_layer:

                flow_out = nn.Conv2d(ngf*mult, 2, kernel_size=3,stride=1,padding=1,bias=True)
                setattr(self, 'output' + str(i), flow_out)
                flow_mask = nn.Sequential(nn.Conv2d(ngf*mult, 1, kernel_size=3,stride=1,padding=1,bias=True),
                                          nn.Sigmoid())
                setattr(self, 'mask' + str(i), flow_mask)

    def forward(self, source, source_B, target_B):
        flow_fields=[]
        masks=[]
        inputs = torch.cat((source, source_B, target_B), 1) 
        out = self.block0(inputs)

        result=[out]
        for i in range(self.encoder_layer-1):
            model = getattr(self, 'encoder' + str(i))
            out = model(out)
            result.append(out) 
        for i in range(self.decoder_layer):
            model = getattr(self, 'decoder' + str(i))
            out = model(out)

            model = getattr(self, 'jump' + str(i))
            jump = model(result[self.encoder_layer-i-2])
            out = out+jump

            if self.encoder_layer-i-1 in self.attn_layer:

                flow_field, mask = self.attn_output(out, i)
                flow_fields.append(flow_field)
                masks.append(mask)

        return flow_fields, masks

    def attn_output(self, out, i):

        model = getattr(self, 'output' + str(i))
        flow = model(out)
        model = getattr(self, 'mask' + str(i))
        mask = model(out)

        return flow, mask  

class FlowNetGenerator(BaseNetwork):

    def __init__(self, image_nc=3, structure_nc=18, output_nc=3, ngf=64,  img_f=1024, layers=6, norm='batch',
                activation='ReLU', encoder_layer=5, attn_layer=[1,2], use_spect=True, use_coord=False):  
        super(FlowNetGenerator, self).__init__()

        self.layers = layers
        self.attn_layer = attn_layer

        self.flow_net = PoseFlowNet(image_nc, structure_nc, ngf, img_f, 
                        encoder_layer, attn_layer=attn_layer,
                        norm=norm, activation=activation, 
                        use_spect=use_spect, use_coord= use_coord)

    def forward(self, source, source_B, target_B):
        flow_fields, masks = self.flow_net(source, source_B, target_B)
        return flow_fields, masks