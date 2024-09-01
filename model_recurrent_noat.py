'''
CORNet with channelattention
'''
from utils.parse_config import *
from models_config import Upsample, EmptyLayer, YOLOLayer
import torch.nn as nn
import torch
import numpy as np
from utils.utils import to_cpu
from collections import OrderedDict

class Flatten(nn.Module):
    def forward(self, x):
        return x.view(x.size(0), -1)

class Identity(nn.Module):
    def forward(self, x):
        return x

class CORblock_S(nn.Module):

    scale = 4  # scale of the bottleneck convolution channels

    def __init__(self, in_channels, out_channels, name, times=1):
        super(CORblock_S, self).__init__()

        self.times = times
        self.name = name

        self.conv_input = nn.Conv2d(in_channels, out_channels, kernel_size=1, bias=False)
        self.skip = nn.Conv2d(out_channels, out_channels,
                              kernel_size=1, stride=2, bias=False)
        self.norm_skip = nn.BatchNorm2d(out_channels)

        self.conv1 = nn.Conv2d(out_channels, out_channels * self.scale,
                               kernel_size=1, bias=False)
        self.nonlin1 = nn.LeakyReLU(0.1, inplace=True)

        self.conv2 = nn.Conv2d(out_channels * self.scale, out_channels * self.scale,
                               kernel_size=3, stride=2, padding=1, bias=False)
        self.nonlin2 = nn.LeakyReLU(0.1, inplace=True)

        self.conv3 = nn.Conv2d(out_channels * self.scale, out_channels,
                               kernel_size=1, bias=False)
        self.nonlin3 = nn.LeakyReLU(0.1, inplace=True)

        self.output = Identity()  # for an easy access to this block's output

        # need BatchNorm for each time step for training to work well
        for t in range(self.times):
            setattr(self, f'norm1_{t}', nn.BatchNorm2d(out_channels * self.scale))
            setattr(self, f'norm2_{t}', nn.BatchNorm2d(out_channels * self.scale))
            setattr(self, f'norm3_{t}', nn.BatchNorm2d(out_channels))

    def forward(self, inp):
        x = self.conv_input(inp)
        layer_outputs = []
        for t in range(self.times):
            if t == 0:
                skip = self.norm_skip(self.skip(x))
                self.conv2.stride = (2, 2)
            else:
                skip = x
                self.conv2.stride = (1, 1)

            x = self.conv1(x)
            x = getattr(self, f'norm1_{t}')(x)
            x = self.nonlin1(x)

            x = self.conv2(x)
            x = getattr(self, f'norm2_{t}')(x)
            x = self.nonlin2(x)

            x = self.conv3(x)
            x = getattr(self, f'norm3_{t}')(x)

            x = x + skip
            x = self.nonlin3(x)
            output = self.output(x)
            layer_outputs.append(output)

        return output, layer_outputs

def create_modules(module_defs):
    """
    Constructs module list of layer blocks from module configuration in module_defs
    """
    hyperparams = module_defs.pop(0)
    output_filters = [64, 128, 128, 256, 256, 256, 256, 512, 512]
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
        #print(f'{module_def["type"]}', output_filters)
    return hyperparams, module_list

class COR_model(nn.Module):
    def __init__(self):
        super(COR_model, self).__init__()
        self.v1 = nn.Sequential(OrderedDict([  # this one is custom to save GPU memory
            ('conv1', nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=3, bias=False)),
            ('norm1', nn.BatchNorm2d(64)),
            ('nonlin1', nn.LeakyReLU(0.1, inplace=True)),
            ('pool', nn.MaxPool2d(kernel_size=3, stride=2, padding=1)),
            ('conv2', nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=1, bias=False)),
            ('norm2', nn.BatchNorm2d(64)),
            ('nonlin2', nn.LeakyReLU(0.1, inplace=True)),
            ('output', Identity())
        ]))
        self.v2 = CORblock_S(64, 128, 'V2', times=2)
        self.v4 = CORblock_S(128, 256, 'V4', times=4)
        self.it = CORblock_S(256, 512, 'IT', times=2)

    def forward(self, inp):
        layer_out_block1 = []
        v1 = self.v1(inp)
        v2, layerout1 = self.v2(v1)
        v4, layerout2 = self.v4(v2)
        it, layerout3 = self.it(v4)
        #print('it:', it.shape) # torch.Size([8, 512, 14, 14])
        layer_out_block1.append(v1)
        layer_out_block1.extend(layerout1)
        layer_out_block1.extend(layerout2)
        layer_out_block1.extend(layerout3)
       
        return it, layer_out_block1

## 反馈空间CBAM-Attention+残差
class SpatialAttention(nn.Module):

    def __init__(self, kernel_size=7):
        super(SpatialAttention, self).__init__()
        assert kernel_size in (3, 7)
        padding = 3 if kernel_size == 7 else 1
        self.conv1_2d = nn.Conv2d(2, 1, kernel_size, stride=1, padding=padding, bias=False)
        self.sigmoid = nn.Sigmoid()
        self.maxpool = nn.MaxPool2d(kernel_size=2, stride=2, ceil_mode=True)

    def forward(self, x):
        x_avg_out = torch.mean(x, dim=1, keepdim=True)
        avg_out = torch.mean(x_avg_out, dim=2, keepdim=True)
        x_max_out, _ = torch.max(x, dim=1, keepdim=True)
        max_out, _ = torch.max(x_max_out, dim=2, keepdim=True)
        concat = torch.cat([avg_out, max_out], dim=1)
        cabm_feature = self.conv1_2d(concat)
        cabm_feature = self.sigmoid(cabm_feature)

        foaion = torch.mul(x, cabm_feature)
        return foaion

class CAR_Block(nn.Module):
    scale = 2

    def __init__(self, in_channels, out_channels, times=1):
        super(CAR_Block, self).__init__()
        self.times = times
        # self.attention = SpatialAttention()
        self.conv1 = nn.Conv2d(in_channels, out_channels,  kernel_size=1, stride=1, padding=0, bias=False)
        self.norm_skip = nn.BatchNorm2d(out_channels)
        self.nonlin1 = nn.LeakyReLU(0.1, inplace=True)
        self.conv2 = nn.Conv2d(out_channels, out_channels*2, kernel_size=3, stride=1, padding=1, bias=False)
        self.nonlin2 = nn.LeakyReLU(0.1, inplace=True)
        self.conv3 = nn.Conv2d(out_channels*2, out_channels, kernel_size=3, stride=1, padding=1, bias=False)
        self.nonlin3 = nn.LeakyReLU(0.1, inplace=True)
        self.output = Identity()  # for an easy access to this block's output

        # need BatchNorm for each time step for training to work well
        for t in range(self.times):
            setattr(self, f'norm2_{t}', nn.BatchNorm2d(out_channels * self.scale))
            setattr(self, f'norm3_{t}', nn.BatchNorm2d(out_channels))

    def forward(self, inp):
        layer_out_block2 = []
        x = self.nonlin1(self.norm_skip(self.conv1(inp)))

        for t in range(self.times):
            skip = x
            #if t == 0:
            #    skip = x
            #else:
            #    skip = recurrent
            #    x = recurrent

            x = self.conv2(x)
            x = getattr(self, f'norm2_{t}')(x)
            x = self.nonlin2(x)

            x = self.conv3(x)
            x = getattr(self, f'norm3_{t}')(x)
            x = self.nonlin3(x)

            x = x + skip
            #recurrent = self.attention(x)

            output = self.output(x)
            layer_out_block2.append(output)

        return output, layer_out_block2

class Darknet(nn.Module):
    """YOLOv3 object detection model"""

    def __init__(self, config_path, img_size=416):
        super(Darknet, self).__init__()
        self.module_defs = parse_model_config(config_path)
        # print(self.module_defs)
        self.COR_model = COR_model()
        # print(self.COR_model)
        self.hyperparams, self.module_list = create_modules(self.module_defs)
        # print(self.module_list)
        self.yolo_layers = [layer[0] for layer in self.module_list if hasattr(layer[0], "metrics")]
        self.img_size = img_size
        self.seen = 0
        self.header_info = np.array([0, 0, 0, self.seen, 0], dtype=np.int32)

    # #@pysnooper.snoop()
    def forward(self, inp, targets=None):
        img_dim = inp.shape[2]
        loss = 0
        layer_outputs, yolo_outputs = [], []
        layer_outputs_shape = []
        x, layer_out_block1= self.COR_model(inp)
        layer_outputs.extend(layer_out_block1)
        for i, (module_def, module) in enumerate(zip(self.module_defs, self.module_list)):
            if module_def["type"] in ["convolutional", "upsample", "maxpool"]:
                x = module(x)
                #print(f'{module_def["type"]}', x.shape)
            elif module_def["type"] == "CBR_Block":
                x, layer_out = module(x)
                #print('layer_out:', layer_out)
                layer_outputs.extend(layer_out)

            elif module_def["type"] == "route":
                x = torch.cat([layer_outputs[int(layer_i)] for layer_i in module_def["layers"].split(",")], 1)
                #print('route', x.shape)
            elif module_def["type"] == "yolo":
                # print(module[0])
                x, layer_loss = module[0](x, targets, img_dim)
                #print('yolo:', x.shape)
                loss += layer_loss
                yolo_outputs.append(x)
            if module_def["type"] != 'CBR_Block':
                layer_outputs.append(x)

        yolo_outputs = to_cpu(torch.cat(yolo_outputs, 1))
        return yolo_outputs if targets is None else (loss, yolo_outputs)

    def load_cornet_weights(self, weight_path):
        print('-->' + 'Loading Model......')
        _model = nn.Sequential(self.COR_model)
        pretrained_dict = torch.load('./weights/new_cornet_s.pth')
        model_dict = _model.state_dict()
        new_dict = OrderedDict()
        for i in model_dict.keys():
            if 'num_batches_tracked' not in i:
                new_dict[i] = model_dict[i]
        # 1. filter out unnecessary keys
        pretrained_dict = {k1: v for (k, v), k1 in zip(pretrained_dict.items(), new_dict)}
        for key in pretrained_dict.keys():
            weight = pretrained_dict[key]
        # 2. overwrite entries in the existing state dict
        model_dict.update(pretrained_dict)
        _model.load_state_dict(model_dict)
        print('-->' + 'Model loaded!')

    def load_darknet_weights(self, weights_path):
        """Parses and loads the weights stored in 'weights_path'"""

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
    config_path = './config/yolov3-model-re.cfg'
    model = Darknet(config_path)
    # print(len(model.module_defs))
    # print(len(model.module_list))
    print(model.module_defs)
    print(model.module_list)
    # model_dict = model.state_dict()
    # for key in model_dict.keys():
    #     print(key)
    #summary(model, (1, 3, 416, 416))
    #summary(model)