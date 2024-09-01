'''
Darknet53 with channelattention
'''
from utils.parse_config import *
from models_config import Upsample, EmptyLayer, YOLOLayer
from attention_module import ChannelAttentionModule,SpatialAttention,ChannelAttention
import torch.nn as nn
import torch
import numpy as np
from utils.utils import to_cpu
from collections import OrderedDict

class Identity(nn.Module):
    def forward(self, x):
        return x

def create_modules(module_defs):
    """
    Constructs module list of layer blocks from module configuration in module_defs
    """
    hyperparams = module_defs.pop(0)
    output_filters = [int(hyperparams["channels"])]
    module_list = nn.ModuleList()
    for module_i, module_def in enumerate(module_defs):
        modules = nn.Sequential()

        if module_def["type"] == "CBR_Block":
            filters = int(module_def["filters"])
            output_filters.append(filters)
            modules.add_module(f"conv_{module_i}", CAR_Block(int(module_def["in_channels"]),
                                                            int(module_def["out_channels"]), int(module_def["times"])))

        if module_def["type"] == "convolutional":
            bn = int(module_def["batch_normalize"])
            filters = int(module_def["filters"])
            kernel_size = int(module_def["size"])
            pad = (kernel_size - 1) // 2
            modules.add_module(
                f"conv_{module_i}",
                nn.Conv2d(
                    in_channels=output_filters[-1],
                    out_channels=filters,
                    kernel_size=kernel_size,
                    stride=int(module_def["stride"]),
                    padding=pad,
                    bias=not bn,
                ),
            )
            if bn:
                modules.add_module(f"batch_norm_{module_i}", nn.BatchNorm2d(filters, momentum=0.9, eps=1e-5))
            if module_def["activation"] == "leaky":
                modules.add_module(f"leaky_{module_i}", nn.LeakyReLU(0.1))

        elif module_def["type"] == "maxpool":
            kernel_size = int(module_def["size"])
            stride = int(module_def["stride"])
            if kernel_size == 2 and stride == 1:
                modules.add_module(f"_debug_padding_{module_i}", nn.ZeroPad2d((0, 1, 0, 1)))
            maxpool = nn.MaxPool2d(kernel_size=kernel_size, stride=stride, padding=int((kernel_size - 1) // 2))
            modules.add_module(f"maxpool_{module_i}", maxpool)

        elif module_def["type"] == "upsample":
            upsample = Upsample(scale_factor=int(module_def["stride"]), mode="nearest")
            modules.add_module(f"upsample_{module_i}", upsample)

        elif module_def["type"] == "route":
            layers = [int(x) for x in module_def["layers"].split(",")]
            filters = sum([output_filters[:][i] for i in layers])
            modules.add_module(f"route_{module_i}", EmptyLayer())
        
        elif module_def["type"] == "shortcut": # shortcut layer的输出是前一层和前三层的输出的叠加.  x = layer_outputs[-1] + layer_outputs[layer_i]
            filters = output_filters[1:][int(module_def["from"])] # from=-3
            modules.add_module(f"shortcut_{module_i}", EmptyLayer())

        elif module_def["type"] == "yolo":
            anchor_idxs = [int(x) for x in module_def["mask"].split(",")]
            # Extract anchors
            anchors = [int(x) for x in module_def["anchors"].split(",")]
            anchors = [(anchors[i], anchors[i + 1]) for i in range(0, len(anchors), 2)]
            anchors = [anchors[i] for i in anchor_idxs]
            num_classes = int(module_def["classes"])
            img_size = int(hyperparams["height"])
            # Define detection layer
            yolo_layer = YOLOLayer(anchors, num_classes, img_size)
            modules.add_module(f"yolo_{module_i}", yolo_layer)
        # Register module list and number of output filters
        module_list.append(modules)
        output_filters.append(filters)
        # print(output_filters)
        # print(f'{module_def["type"]}', output_filters)
    return hyperparams, module_list
    
class CAR_Block(nn.Module):
    scale = 2

    def __init__(self, in_channels, out_channels, times=1):
        super(CAR_Block, self).__init__()
        self.times = times
        # self.sattention = SpatialAttention()
        # self.attention = CrossAttention(in_channel=out_channels)
        # self.attention = SELayer(out_channels)
        self.attention = ChannelAttention(out_channels)
        # self.attention = ChannelAttentionModule()
        self.conv1 = nn.Conv2d(in_channels, out_channels,  kernel_size=3, stride=1, padding=1, bias=False)
        self.norm_skip = nn.BatchNorm2d(out_channels)
        self.nonlin1 = nn.LeakyReLU(0.1, inplace=True)
        self.conv2 = nn.Conv2d(out_channels, out_channels*2, kernel_size=3, stride=1, padding=1, bias=False)
        self.nonlin2 = nn.LeakyReLU(0.1, inplace=True)
        self.conv3 = nn.Conv2d(out_channels*2, out_channels, kernel_size=1, stride=1, padding=0, bias=False)
        self.nonlin3 = nn.LeakyReLU(0.1, inplace=True)
        self.nonlin4 = nn.LeakyReLU(0.1, inplace=True)
        self.output = Identity()  # for an easy access to this block's output

        # need BatchNorm for each time step for training to work well
        for t in range(self.times):
            setattr(self, f'norm2_{t}', nn.BatchNorm2d(out_channels * self.scale))
            setattr(self, f'norm3_{t}', nn.BatchNorm2d(out_channels))
            

    def forward(self, inp):
        layer_out_block2 = []
        x = self.nonlin1(self.norm_skip(self.conv1(inp)))
        
        for t in range(self.times):
            if t == 0:
                skip = x
            else:
                skip = recurrent
                x = recurrent

            x = self.conv2(x)
            x = getattr(self, f'norm2_{t}')(x)
            x = self.nonlin2(x)

            x = self.conv3(x)
            x = getattr(self, f'norm3_{t}')(x)
            x = self.nonlin3(x)

            x = x + skip
            x = self.nonlin4(x)
       
            recurrent = self.attention(x)
        
            output = self.output(x)
            layer_out_block2.append(output)
  
        return output, layer_out_block2

class Darknet(nn.Module):
    """YOLOv3 object detection model"""

    def __init__(self, config_path, img_size=416):
        super(Darknet, self).__init__()
        self.module_defs = parse_model_config(config_path)
        # print(self.module_defs)
        self.hyperparams, self.module_list = create_modules(self.module_defs)
        # print(self.module_list)
        self.yolo_layers = [layer[0] for layer in self.module_list if hasattr(layer[0], "metrics")]
        self.img_size = img_size
        self.seen = 0
        self.header_info = np.array([0, 0, 0, self.seen, 0], dtype=np.int32)

    # #@pysnooper.snoop()
    def forward(self, x, targets=None):
        img_dim = x.shape[2]
        loss = 0
        layer_outputs, yolo_outputs = [], []
        for i, (module_def, module) in enumerate(zip(self.module_defs, self.module_list)):
            if module_def["type"] in ["convolutional", "upsample", "maxpool"]:
                x = module(x)
            elif module_def["type"] == "CBR_Block":
                x, layer_out = module(x)
                layer_outputs.extend(layer_out)
            elif module_def["type"] == "route":  # layer只有一个值，那么该route层的输出就是该层, 如果layer有两个值，则route层输出是对应两个层的特征图的融合。
                x = torch.cat([layer_outputs[int(layer_i)] for layer_i in module_def["layers"].split(",")], 1)
            elif module_def["type"] == "shortcut":
                layer_i = int(module_def["from"])
                x = layer_outputs[-1] + layer_outputs[layer_i]
            elif module_def["type"] == "yolo":
                # print(module[0])
                x, layer_loss = module[0](x, targets, img_dim) # 这里的module[0]是指yolo层
                loss += layer_loss
                yolo_outputs.append(x)
            layer_outputs.append(x)
        yolo_outputs = to_cpu(torch.cat(yolo_outputs, 1))
        return yolo_outputs if targets is None else (loss, yolo_outputs)

    def load_darknet_weights(self, weights_path):
        """Parses and loads the weights stored in 'weights_path'"""
        print('--->Load Darknet-53')
        # Open the weights file
        with open(weights_path, "rb") as f:
            header = np.fromfile(f, dtype=np.int32, count=5)  # First five are header values
            self.header_info = header  # Needed to write header when saving weights
            self.seen = header[3]  # number of images seen during training
            weights = np.fromfile(f, dtype=np.float32)  # The rest are weights

        # Establish cutoff for loading backbone weights
        cutoff = None
        if "darknet53.conv.74" in weights_path:
            cutoff = 75

        ptr = 0
        for i, (module_def, module) in enumerate(zip(self.module_defs, self.module_list)):
            if i == cutoff:
                break
            if module_def["type"] == "convolutional":
                conv_layer = module[0]
                if module_def["batch_normalize"]:
                    # Load BN bias, weights, running mean and running variance
                    bn_layer = module[1]
                    num_b = bn_layer.bias.numel()  # Number of biases
                    # Bias
                    bn_b = torch.from_numpy(weights[ptr : ptr + num_b]).view_as(bn_layer.bias)
                    bn_layer.bias.data.copy_(bn_b)
                    ptr += num_b
                    # Weight
                    bn_w = torch.from_numpy(weights[ptr : ptr + num_b]).view_as(bn_layer.weight)
                    bn_layer.weight.data.copy_(bn_w)
                    ptr += num_b
                    # Running Mean
                    bn_rm = torch.from_numpy(weights[ptr : ptr + num_b]).view_as(bn_layer.running_mean)
                    bn_layer.running_mean.data.copy_(bn_rm)
                    ptr += num_b
                    # Running Var
                    bn_rv = torch.from_numpy(weights[ptr : ptr + num_b]).view_as(bn_layer.running_var)
                    bn_layer.running_var.data.copy_(bn_rv)
                    ptr += num_b
                else:
                    # Load conv. bias
                    num_b = conv_layer.bias.numel()
                    conv_b = torch.from_numpy(weights[ptr : ptr + num_b]).view_as(conv_layer.bias)
                    conv_layer.bias.data.copy_(conv_b)
                    ptr += num_b
                # Load conv. weights
                num_w = conv_layer.weight.numel()
                conv_w = torch.from_numpy(weights[ptr : ptr + num_w]).view_as(conv_layer.weight)
                conv_layer.weight.data.copy_(conv_w)
                ptr += num_w
        print('--->Load done')

    def save_darknet_weights(self, path, cutoff=-1):
        """
            @:param path    - path of the new weights file
            @:param cutoff  - save layers between 0 and cutoff (cutoff = -1 -> all are saved)
        """
        fp = open(path, "wb")
        self.header_info[3] = self.seen
        self.header_info.tofile(fp)

        # Iterate through layers
        for i, (module_def, module) in enumerate(zip(self.module_defs[:cutoff], self.module_list[:cutoff])):
            if module_def["type"] == "convolutional":
                conv_layer = module[0]
                # If batch norm, load bn first
                if module_def["batch_normalize"]:
                    bn_layer = module[1]
                    bn_layer.bias.data.cpu().numpy().tofile(fp)
                    bn_layer.weight.data.cpu().numpy().tofile(fp)
                    bn_layer.running_mean.data.cpu().numpy().tofile(fp)
                    bn_layer.running_var.data.cpu().numpy().tofile(fp)
                # Load conv bias
                else:
                    conv_layer.bias.data.cpu().numpy().tofile(fp)
                # Load conv weights
                conv_layer.weight.data.cpu().numpy().tofile(fp)

        fp.close()

if __name__=='__main__':
    #device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    config_path = './config/carfnn_darknet_ssdd.cfg'
    model = Darknet(config_path)
    # print(len(model.module_defs))
    # print(len(model.module_list))
    #print(model.module_defs)
    print(model.module_list)
    # model_dict = model.state_dict()
    # for key in model_dict.keys():
    #     print(key)
    #summary(model, (1, 3, 416, 416))
    #summary(model)