import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision.models import vgg19


def calc_mean_std(features):
    """

    :param features: shape of features -> [batch_size, c, h ,w]
    :return: features_mean, features_std: shape of mean/std -> [batch_size, c, 1, 1]
    """

    batch_size, c = features.size()[: 2]
    features_mean = features.reshape(batch_size, c, -1).mean(dim=2).reshape(batch_size, c, 1, 1)
    features_std = features.reshape(batch_size, c, -1).std(dim=2).reshape(batch_size, c, 1, 1) + 1e-6

    return features_mean, features_std


def adain(content_features, style_features, weights=None):
    """
    Adaptive Instance Normlization

    :param content_features: shape -> [batch_size, c, h ,w]
    :param style_features: shape - > [batch_size, c, h ,w]
    :return: normalized_features: shape - > [batch_size, c, h ,w]
    """

    content_mean, content_std = calc_mean_std(content_features)

    if weights is not None:
        normalized_features = []
        for i in range(len(style_features)):
            style_mean, style_std = calc_mean_std(style_features[i])
            normalized_feature = style_std * ((content_features - content_mean) / content_std) + style_mean
            normalized_features.append(weights[i] * normalized_feature)
        return normalized_features

    else:
        style_mean, style_std = calc_mean_std(style_features)
        normalized_feature = style_std * ((content_features - content_mean) / content_std) + style_mean
        return normalized_feature


class VGGEncoder(nn.Module):
    def __init__(self):
        super(VGGEncoder, self).__init__()
        vgg = vgg19(pretrained=True).features  # 利用 features 去获取所有的模块
        self.slice1 = vgg[: 2]
        self.slice2 = vgg[2: 7]
        self.slice3 = vgg[7: 12]
        self.slice4 = vgg[12: 21]

        for param in self.parameters():  # 直接可以利用 self.parameter() 去获取模型中所有可训练的参数
            param.requires_grad = False

    def forward(self, images, output_last_feature=False, flag=False):

        if flag:
            features_list = []
            for image in images:
                h1 = self.slice1(image)
                h2 = self.slice2(h1)
                h3 = self.slice3(h2)
                h4 = self.slice4(h3)

                features_list.append(h4)

            if output_last_feature:
                return features_list


        h1 = self.slice1(images)
        h2 = self.slice2(h1)
        h3 = self.slice3(h2)
        h4 = self.slice4(h3)

        if output_last_feature:
            return h4

        else:
            return h1, h2, h3, h4


class RC(nn.Module):
    """
        A wrapper of ReflectionPad2 and Conv2d
    """

    def __init__(self, in_channels, out_channels, kernel_size=3, pad_size=1, activate=True):
        super(RC, self).__init__()
        self.pad = nn.ReflectionPad2d((1, 1, 1, 1))
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size)
        self.activated = activate

    def forward(self, x):
        h = self.pad(x)
        h = self.conv(h)

        if self.activated:
            return F.relu(h)
        else:
            return h


class Decoder(nn.Module):
    def __init__(self):
        super(Decoder, self).__init__()
        self.rc1 = RC(512, 256, 3, 1)
        self.rc2 = RC(256, 256, 3, 1)
        self.rc3 = RC(256, 256, 3, 1)
        self.rc4 = RC(256, 256, 3, 1)
        self.rc5 = RC(256, 128, 3, 1)
        self.rc6 = RC(128, 128, 3, 1)
        self.rc7 = RC(128, 64, 3, 1)
        self.rc8 = RC(64, 64, 3, 1)
        self.rc9 = RC(64, 3, 3, False)

    def forward(self, features):

        h = self.rc1(features)

        # 保证图像大小不改变
        h = F.interpolate(h, scale_factor=2)   # 上下采样插值函数，用于改变feature的尺寸，例如 h,w
        h = self.rc2(h)
        h = self.rc3(h)
        h = self.rc4(h)
        h = self.rc5(h)

        h = F.interpolate(h, scale_factor=2)
        h = self.rc6(h)
        h = self.rc7(h)

        h = F.interpolate(h, scale_factor=2)
        h = self.rc8(h)
        h = self.rc9(h)

        return h


class Model(nn.Module):

    def init_weight(self):
        for param in self.parameters():
            nn.init.kaiming_normal(param)

    def __init__(self, styleInterpWeights=None):
        super(Model, self).__init__()
        self.vgg_encoder = VGGEncoder()
        self.decoder = Decoder()


    def generate(self, content_images, style_images, alpha=1.0, styleInterpWeights=None):

        content_features = self.vgg_encoder(content_images, output_last_feature=True)

        if styleInterpWeights:
            style_features = self.vgg_encoder(style_images, output_last_feature=True, flag=True)
            t = adain(content_features, style_features, styleInterpWeights)
            t = torch.stack(t)
            t = torch.sum(t, dim=0)
            t = alpha * t + (1 - alpha) * content_features  # [batch_size, c, h ,w]
            out = self.decoder(t)  # [batch_size, c, h ,w]

        else:
            style_features = self.vgg_encoder(style_images, output_last_feature=True)

            t = adain(content_features, style_features)
            t = alpha * t + (1 - alpha) * content_features  # [batch_size, c, h ,w]
            out = self.decoder(t)  # [batch_size, c, h ,w]

        return out

    @staticmethod
    def calc_content_loss(out_features, t):
        return F.mse_loss(out_features, t)

    @staticmethod
    def calc_style_loss(content_middel_features, style_middel_features):
        loss = 0
        for c, s in zip(content_middel_features, style_middel_features):
            c_mean, c_std = calc_mean_std(c)
            s_mean, s_std = calc_mean_std(s)
            loss += F.mse_loss(c_mean, s_mean) + F.mse_loss(c_std, s_std)

        return loss

    def forward(self, content_images, style_images, alpha=1.0, lam=10):
        content_features = self.vgg_encoder(content_images, output_last_feature=True)
        style_features = self.vgg_encoder(style_images, output_last_feature=True)
        t = adain(content_features, style_features)
        t = alpha * t + (1 - alpha) * content_features
        out = self.decoder(t)

        output_features = self.vgg_encoder(out, output_last_feature=True)
        content_middle_features = self.vgg_encoder(out, output_last_feature=False)
        style_middle_features = self.vgg_encoder(style_features, output_last_feature=False)

        content_loss = self.calc_content_loss(output_features, t)
        style_loss = self.calc_style_loss(content_middle_features, style_middle_features)

        loss = content_loss + lam * style_loss

        return loss