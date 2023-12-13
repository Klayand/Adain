import os
import argparse

import cv2
from PIL import Image
import torch
from skimage import io, transform
from torchvision import transforms
from torchvision.utils import save_image
from model import Model
from color_transfer import ColorTransfer

normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                 std=[0.229, 0.224, 0.225])

trans = transforms.Compose([transforms.ToTensor(),
                            normalize])


def denorm(tensor, device):
    std = torch.Tensor([0.229, 0.224, 0.225]).reshape(-1, 1, 1).to(device)
    mean = torch.Tensor([0.485, 0.456, 0.406]).reshape(-1, 1, 1).to(device)
    res = torch.clamp(tensor * std + mean, 0, 1)

    return res


def resize_image(image_path):
    image = io.imread(image_path)
    if len(image.shape) == 3 and image.shape[-1] == 3:
        H, W, C = image.shape
        if H < W:
            ratio = W / H
            H = 512
            W = int(ratio * H)
        else:
            ratio = H / W
            W = 512
            H = int(ratio * W)

        # 可以放大或者缩小图像
        image = transform.resize(image, (H, W), mode='reflect', anti_aliasing=True)

    return image


def main():
    parser = argparse.ArgumentParser(description='AdaIN Style Transfer')
    parser.add_argument('--content', '-c', type=str, default=None,
                        help='Content image path e.g. content.jpg')
    parser.add_argument('--style', '-s', type=str, default=None,
                        help='Style image path e.g. image.jpg')
    parser.add_argument('--output_name', '-o', type=str, default=None,
                        help='Output path for generated image, no need to add ext, e.g. out')
    parser.add_argument('--alpha', '-a', type=float, default=1,
                        help='alpha control the fusion degree in Adain')
    parser.add_argument('--gpu', '-g', type=int, default=0,
                        help='GPU ID(nagative value indicate CPU)')
    parser.add_argument('--model_state_path', type=str, default='model/model_state.pth',
                        help='save directory for result and loss')
    parser.add_argument('--styleInterpWeights', '-w', type=str, default=None,
                        help='apply multi-styles interpolation(int)')
    parser.add_argument('--color_preserve', '-p', type=bool, default=False,
                        help='apply color preserve')

    args = parser.parse_args()

    multi_style = False
    styleInterpWeights = None

    if args.styleInterpWeights:
        styleInterpWeights = [int(weight) for weight in args.styleInterpWeights.split(',')]

        try:
            style_images_list = args.style.split(',')
        except:
            raise "only one style image..."

        if len(style_images_list) != len(styleInterpWeights):
            raise "the num of style images does`t match the num of weights"
        else:
            multi_style = True

    # set device on GPU if available, else CPU
    if torch.cuda.is_available() and args.gpu >= 0:
        device = torch.device(f'cuda:{args.gpu}')
        print(f'# CUDA available: {torch.cuda.get_device_name(0)}')
    else:
        device = 'cpu'

    # set model
    model = Model()
    if args.model_state_path is not None:
        model.load_state_dict(torch.load(args.model_state_path, map_location=lambda storage, loc: storage))
    model = model.to(device)

    # c = resize_image(args.content)
    # s = resize_image(args.style)

    style_images_tensor = []

    c = Image.open(args.content)

    if multi_style:
        for image in style_images_list:
            if args.color_preserve:
                # api may not matter the outcome ????
                s = ColorTransfer(args.content, image, preserve_paper=False, api=False).transfer_image

            else:
                s = Image.open(image)
            s_tensor = trans(s).unsqueeze(0).to(device)
            style_images_tensor.append(s_tensor)

        s_tensor = style_images_tensor

    else:
        if args.color_preserve:
            s = ColorTransfer(args.content, args.style, preserve_paper=False, api=True).transfer_image

        else:
            s = Image.open(args.style)
        s_tensor = trans(s).unsqueeze(0).to(device)

    c_tensor = trans(c).unsqueeze(0).to(device)

    with torch.no_grad():
        out = model.generate(c_tensor, s_tensor, args.alpha, styleInterpWeights)

    out = denorm(out, device)
    # c_tensor = denorm(c_tensor, device)
    # s_tensor = denorm(s_tensor, device)
    #
    # out = torch.cat([c_tensor, s_tensor, out], dim=0)
    out.to('cpu')

    if args.output_name is None:

        c_name = os.path.splitext(os.path.basename(args.content))[0]

        if styleInterpWeights is not None:
            name_list = []
            style_name = args.style.split(',')
            for name in style_name:
                name_list.append(name.split('/')[-1].split('.')[0])

            args.output_name = f'{c_name}_{"_".join(name_list)}'

        else:
            s_name = os.path.splitext(os.path.basename(args.style))[0]
            args.output_name = f'{c_name}_{s_name}'

    if args.color_preserve:
        save_image(out, f'{args.output_name}_color_preserve.jpg', nrow=1)
        o = Image.open(f'{args.output_name}_color_preserve.jpg')
    else:
        save_image(out, f'{args.output_name}.jpg', nrow=1)
        o = Image.open(f'{args.output_name}.jpg')

    demo = Image.new('RGB', (c.width * 2, c.height))
    o = o.resize(c.size)

    demo.paste(c, (0, 0))
    demo.paste(o, (c.width, 0))

    if styleInterpWeights is not None:
        demo_width = c.width
        o_width = 0

        for path in style_images_list:
            s = Image.open(path).resize((i // 4 for i in c.size))
            demo.paste(s, (demo_width, c.height - s.height))
            o.paste(s, (o_width, o.height - s.height))

            demo_width += s.width
            o_width += s.width

    else:
        s = Image.open(args.style)
        s = s.resize((i // 4 for i in c.size))
        demo.paste(s, (c.width, c.height - s.height))
        o.paste(s, (0, o.height - s.height))


    if args.color_preserve:
        demo.save(f'{args.output_name}_style_transfer_demo_color_preserve.jpg', quality=95)

        o.save(f'{args.output_name}_with_style_image_color_preserve.jpg', quality=95)

        print(f'result saved into files starting with {args.output_name}_color_preserve')

    else:
        demo.save(f'{args.output_name}_style_transfer_demo.jpg', quality=95)

        o.save(f'{args.output_name}_with_style_image.jpg', quality=95)

        print(f'result saved into files starting with {args.output_name}')


if __name__ == '__main__':
    main()