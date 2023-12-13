import numpy as np
import cv2
import argparse


def show_image(title, image, width = 300):
    # resize the image to have a constant width, just to make
    # displaying the images take up less screen real estate

    r = width / float(image.shape[1])
    dim = (width, int(image.shape[0] * r))
    resized = cv2.resize(image, dim, interpolation=cv2.INTER_AREA)

    # show the resized image
    cv2.imshow(title, resized)


def str2bool(string):
    if string.lower() in ('yes', 'true', 't', 'y', '1'):
        return True
    elif string.lower() in ('no', 'false', 'f', 'n', '0'):
        return False
    else:
        raise argparse.ArgumentTypeError("Boolean type expected.")


class ColorTransfer:
    """
        Transfer the color distribution from the source to the target
        image using the mean and standard deviations of the L*a*b color space

        Based on the 'Color Transfer between Images' paper by Reinhard et al.(2001)

        Also this is the new implementation of Luminance-only transfer.
    """
    def __init__(self, source_path, target_path, clip=True, preserve_paper=True, api=False):
        """

        :param source_path: source image file path
        :param target_path: target image file path

        :param clip:Should components of L*a*b image be scaled by np.clip
            before converting back to RGB|BGR color space?
            If False then components will be min-max scaled appropriately.
            Clipping will keep target image brightness truer to the input.
            Scaling will adjust image brightness to avoid washed out portions in the resulting color transfer that can be caused by clipping.

        :param preserve_paper:Should color transfer strictly follow methodology
            layed out in original paper? The method does not always produce aesthetically pleasing results.

            If False then L*a*b components will scaled using the reciprocal of the scaling factor
            proposed in the paper.
            This method seems to produce more consistently aesthetically pleasing results.

        :param api: Using for the AdaIN_transfer or not?
            True: act as the api for Luminance-only transfer
            Flase: just used for color transfer
        """

        self.source = cv2.imread(source_path)
        self.target = cv2.imread(target_path)

        source_lab = cv2.cvtColor(self.source, cv2.COLOR_BGR2LAB).astype('float32')
        target_lab = cv2.cvtColor(self.target, cv2.COLOR_BGR2LAB).astype('float32')

        # compute color statistics for the source and target image
        (l_mean_src, l_std_src, a_mean_src, a_std_src,
         b_mean_src, b_std_src) = self.image_stat(source_lab)

        (l_mean_tar, l_std_tar, a_mean_tar, a_std_tar,
         b_mean_tar, b_std_tar) = self.image_stat(target_lab)

        (l, a, b) = cv2.split(target_lab)


        if preserve_paper:
            # subtract the mean from the target image
            # scale by the standard deviations using paper proposed factor
            # add in the source mean
            l = (l_std_tar / l_std_src) * (l - l_mean_tar) + l_mean_src
            if not api:
                a = (a_std_tar / a_std_src) * (a - a_mean_tar) + a_mean_src
                b = (b_std_tar / b_std_src) * (b - b_mean_tar) + b_mean_src

        else:
            # scaled by the standard deviations using reciprocal of paper proposed factor
            l = (l_std_src / l_std_tar) * (l - l_mean_tar) + l_mean_src

            if not api:
                a = (a_std_src / a_std_tar) * (a - a_mean_tar) + a_mean_src
                b = (b_std_src / b_std_tar) * (b - b_mean_tar) + b_mean_src


        # clip/scale the pixel intensities to [0, 255] if they fall
        # outside this range
        l = self._scale_array(l, clip=clip)
        a = self._scale_array(a, clip=clip)
        b = self._scale_array(b, clip=clip)

        # merge the channels together and convert back to the RGB color
        # space, being sure to utilize the 8-bit unsigned integer data type
        transfer_image = cv2.merge([l, a, b])
        transfer_image = cv2.cvtColor(transfer_image.astype('uint8'), cv2.COLOR_LAB2BGR)

        self.transfer_image = transfer_image

    def image_stat(self, image):
        """

        :return: Tuples of mean and standard deviations for the L*, a*, b* channels
        """

        (l, a, b) = cv2.split(image)
        (lMean, lStd) = (l.mean(), l.std())
        (aMean, aStd) = (a.mean(), a.std())
        (bMean, bStd) = (b.mean(), b.std())

        return lMean, lStd, aMean, aStd, bMean, bStd

    def _scale_array(self, arr, clip=True):
        """
        Trim NumPy array values to be in [0, 255] range with option of clipping or scaling.

        :param arr: array to be trimmed to [0, 255] range
        :param clip: should array be scaled by np.clip?
                    if False then the input array will be min-max scaled to range.
                    [max([arr.min(), 0]), min([arr.max(), 255])]
        :return: NumPy array that has been scaled to be in [0, 255] range
        """
        if clip:
            scaled = np.clip(arr, 0, 255)
        else:
            scaled_range = (max([arr.min(), 0]), min([arr.max(), 255]))
            scaled = self._min_max_scale(arr, new_range=scaled_range)

        return scaled

    def _min_max_scale(self, arr, new_range=(0, 255)):
        """
        Perform min-max scaling to a NumPy array

        :param arr:
        :param new_range:
        :return:
        """
        mn = arr.min()
        mx = arr.max()

        # check if scaling needs to be done to be in new_range
        if mn < new_range[0] or mx > new_range[1]:
            scaled = (new_range[1] - new_range[0]) * (arr - mn) / (mx - mn) + new_range[0]

        else:
            scaled = arr

        return arr


if __name__ == '__main__':

    arg = argparse.ArgumentParser()
    arg.add_argument("-s", "--source", required=True,
                    help="Path to the source image")
    arg.add_argument("-t", "--target", required=True,
                    help="Path to the target image")
    arg.add_argument("-c", "--clip", type=str2bool, default='True',
                    help="Should np.clip scale L*a*b* values before final conversion to BGR? "
                         "Approptiate min-max scaling used if False.")
    arg.add_argument("-p", "--preservePaper", type=str2bool, default='True',
                    help="Should color transfer strictly follow methodology layed out in original paper?")
    arg.add_argument("-o", "--output", default= '.' ,help="Path to the output image (optional)")

    source_path = r'content/sailboat.jpg'
    tagret_path = r'style/picasso_self_portrait.jpg'

    source = cv2.imread(source_path)
    target = cv2.imread(tagret_path)

    color_transfer = ColorTransfer(source_path, tagret_path, preserve_paper=False, api=True).transfer_image
    show_image('Source', source)
    show_image("Target", target)
    show_image('Transfer', color_transfer)
    cv2.waitKey(0)
