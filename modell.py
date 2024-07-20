import brevitas
import brevitas.nn as qnn
import torch
import torch.nn as nn
from brevitas.quant import Int8ActPerTensorFloat, Int8WeightPerTensorFloat

FEATURES_MAPS_VGG19 = [
    ("I",), # Initial input
    ("C", 3, 64, 3, 1, 1), # (32x32x3) -> (32x32x64)
    ("R",),
    ("C", 64, 64, 3, 1, 1), # (32x32x64) -> (32x32x64)
    ("R",),
    ("P", 2, 2, 0, 1, False), # (32x32x64) -> (16x16x64)
    ("I",),
    ("C", 64, 128, 3, 1, 1), # (16x16x64) -> (16x16x128)
    ("R",),
    ("C", 128, 128, 3, 1, 1), # (16x16x128) -> (16x16x128)
    ("R",),
    ("P", 2, 2, 0, 1, False), # (16x16x128) -> (8x8x128)
    ("I",),
    ("C", 128, 256, 3, 1, 1), # (8x8x128) -> (8x8x256)
    ("R",),
    ("C", 256, 256, 3, 1, 1), # (8x8x256) -> (8x8x256)
    ("R",),
    ("C", 256, 256, 3, 1, 1), # (8x8x256) -> (8x8x256)
    ("R",),
    ("C", 256, 256, 3, 1, 1), # (8x8x256) -> (8x8x256)
    ("R",),
    ("P", 2, 2, 0, 1, False), # (8x8x256) -> (4x4x256)
    ("I",),
    ("C", 256, 512, 3, 1, 1), # (4x4x256) -> (4x4x512)
    ("R",),
    ("C", 512, 512, 3, 1, 1), # (4x4x512) -> (4x4x512)
    ("R",),
    ("C", 512, 512, 3, 1, 1), # (4x4x512) -> (4x4x512)
    ("R",),
    ("C", 512, 512, 3, 1, 1), # (4x4x512) -> (4x4x512)
    ("R",),
    ("P", 2, 2, 0, 1, False), # (4x4x512) -> (2x2x512)
    ("I",),
    ("C", 512, 512, 3, 1, 1), # (2x2x512) -> (2x2x512)
    ("R",),
    ("C", 512, 512, 3, 1, 1), # (2x2x512) -> (2x2x512)
    ("R",),
    ("C", 512, 512, 3, 1, 1), # (2x2x512) -> (2x2x512)
    ("R",),
    ("C", 512, 512, 3, 1, 1), # (2x2x512) -> (2x2x512)
    ("R",),
    ("P", 2, 2, 0, 1, False), # (2x2x512) -> (1x1x512)
    ("I",),
]

class Fp32VGG19(nn.Module):
    def __init__(self, output_size: int):
        super(Fp32VGG19, self).__init__()
        self.output_size = output_size

        def make_layers(t):
            if t[0] == "P":
                return nn.AvgPool2d(kernel_size=t[1], stride=t[2], padding=t[3], ceil_mode=t[5])
            elif t[0] == "C":
                return nn.Conv2d(t[1], t[2], kernel_size=t[3], stride=t[4], padding=t[5])
            elif t[0] == "L":
                return nn.Linear(in_features=t[1], out_features=t[2])
            elif t[0] == "R":
                return nn.ReLU()
            else:
                raise NameError(f"{t} not defined")

        self.features = nn.Sequential(*[make_layers(t) for t in FEATURES_MAPS_VGG19 if t[0] != "I"])

        self.avgpool = nn.AdaptiveAvgPool2d(output_size=(1, 1))
        self.final_layer = nn.Linear(in_features=512 * 1 * 1, out_features=output_size)

    def forward(self, x):
        x = self.features(x)
        x = self.avgpool(x)
        x = nn.Flatten()(x)
        x = self.final_layer(x)
        return x

class QuantVGG19(nn.Module):
    def __init__(
        self,
        bit: int,
        output_size: int = 3,
        act_quant: brevitas.quant = Int8ActPerTensorFloat,
        weight_quant: brevitas.quant = Int8WeightPerTensorFloat,
    ):
        super(QuantVGG19, self).__init__()
        self.bit = bit

        def tuple2quantlayer(t):
            if t[0] == "R":
                return qnn.QuantReLU(return_quant_tensor=True, bit_width=bit, act_quant=act_quant)
            if t[0] == "P":
                return nn.AvgPool2d(kernel_size=t[1], stride=t[2], padding=t[3], ceil_mode=t[5])
            if t[0] == "C":
                return qnn.QuantConv2d(
                    t[1],
                    t[2],
                    kernel_size=t[3],
                    stride=t[4],
                    padding=t[5],
                    weight_bit_width=bit,
                    weight_quant=weight_quant,
                    return_quant_tensor=True,
                )
            if t[0] == "L":
                return qnn.QuantLinear(
                    in_features=t[1],
                    out_features=t[2],
                    weight_bit_width=bit,
                    weight_quant=weight_quant,
                    bias=True,
                    return_quant_tensor=True,
                )
            if t[0] == "I":
                identity_quant = t[1] if len(t) == 2 else bit
                return qnn.QuantIdentity(
                    bit_width=identity_quant, act_quant=act_quant, return_quant_tensor=True
                )

        self.features = nn.Sequential(*[tuple2quantlayer(t) for t in FEATURES_MAPS_VGG19])
        self.identity1 = qnn.QuantIdentity(
            bit_width=bit, act_quant=act_quant, return_quant_tensor=True
        )
        self.identity2 = qnn.QuantIdentity(
            bit_width=bit, act_quant=act_quant, return_quant_tensor=True
        )
        self.final_layer = qnn.QuantLinear(
            in_features=512 * 1 * 1,
            out_features=output_size,
            weight_quant=weight_quant,
            weight_bit_width=bit,
            bias=True,
            return_quant_tensor=True,
        )

    def forward(self, x):
        x = self.features(x)
        x = self.identity1(x)
        x = torch.flatten(x, 1)
        x = self.identity2(x)
        x = self.final_layer(x)
        return x.value
