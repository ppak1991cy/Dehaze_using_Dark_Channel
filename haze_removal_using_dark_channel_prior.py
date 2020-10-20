import numpy as np
import cv2


def Guidedfilter(im, p, r, eps):
    mean_I = cv2.boxFilter(im, cv2.CV_64F, (r, r))
    mean_p = cv2.boxFilter(p, cv2.CV_64F, (r, r))
    mean_Ip = cv2.boxFilter(im * p, cv2.CV_64F, (r, r))
    cov_Ip = mean_Ip - mean_I * mean_p

    mean_II = cv2.boxFilter(im * im, cv2.CV_64F, (r, r))
    var_I = mean_II - mean_I * mean_I

    a = cov_Ip / (var_I + eps)
    b = mean_p - a * mean_I

    mean_a = cv2.boxFilter(a, cv2.CV_64F, (r, r))
    mean_b = cv2.boxFilter(b, cv2.CV_64F, (r, r))

    q = mean_a * im + mean_b
    return q


class HazeRemovalTool(object):

    @staticmethod
    def _get_dark_channel(img, patch_size=15):
        height, width, _ = img.shape
        min_intensity = np.min(img, axis=2)  # 获取三个通道最小值
        dark_channel = np.zeros([height, width], dtype=np.float64)
        patch_idx_offset = [-int(patch_size / 2), patch_size - int(patch_size / 2) - 1]
        for i in range(height):
            for j in range(width):
                top = max(0, i + patch_idx_offset[0])
                bottom = min(height, i + patch_idx_offset[1])
                left = max(0, j + patch_idx_offset[0])
                right = min(width, j + patch_idx_offset[1])

                patch_min_intensity = min_intensity[top: bottom, left: right]
                patch_dark_channel = np.min(patch_min_intensity)

                dark_channel[i, j] = patch_dark_channel
        return dark_channel

    @staticmethod
    def _get_atmospheric_light(img, dark_channel, top_percent=0.001):
        height, width, channel = img.shape
        pixel_num = height * width

        dark_record = []
        for i in range(height):
            for j in range(width):
                dark_info = (i, j, dark_channel[i, j])
                dark_record.append(dark_info)
        dark_record = sorted(dark_record, key=lambda x: -x[2])

        top_num = int(pixel_num * top_percent)
        atmospheric_light = np.zeros(channel, dtype=np.float64)
        for i in range(top_num):
            for j in range(channel):
                atmospheric_light[j] += img[dark_record[i][0], dark_record[i][1]][j]
        atmospheric_light = atmospheric_light / top_num
        return atmospheric_light

    @staticmethod
    def _estimate_transmission(img, atmospheric_light, patch_size=15, omega=0.95):
        height, width, channel = img.shape
        img_normalized = np.zeros([height, width, channel], dtype=np.float64)

        for c in range(channel):
            img_normalized[:, :, c] = img[:, :, c] / atmospheric_light[c]
        transmission = 1 - omega * HazeRemovalTool._get_dark_channel(img_normalized, patch_size)
        return transmission

    @staticmethod
    def _refine_transmission(img, transmission):
        img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        img_gray = np.float64(img_gray) / 255
        r = 60
        eps = 0.0001
        transmission_refined = Guidedfilter(img_gray, transmission, r, eps)
        return transmission_refined

    @staticmethod
    def _recover_scene_radiance(img, transmission, atmospheric_light, threshold=0.1):
        img_recovered = np.empty(img.shape, img.dtype)
        transmission = cv2.max(transmission, threshold)

        for i in range(0, 3):
            img_recovered[:, :, i] = \
                (img[:, :, i] - atmospheric_light[i]) / transmission + atmospheric_light[i]

        return img_recovered

    @staticmethod
    def dehaze(img):
        I = img / 255
        dark_channel = HazeRemovalTool._get_dark_channel(I)
        atmospheric_light = HazeRemovalTool._get_atmospheric_light(I, dark_channel)
        transmission = HazeRemovalTool._estimate_transmission(I, atmospheric_light)
        transmission_refined = HazeRemovalTool._refine_transmission(img, transmission)
        img_recovered = HazeRemovalTool._recover_scene_radiance(I, transmission_refined, atmospheric_light)

        return img_recovered


if __name__ == '__main__':
    test_data = 'dataset/haze/tmp_3.jpg'

    haze_removal_tool = HazeRemovalTool()

    src = cv2.imread(test_data)
    img = haze_removal_tool.dehaze(src)
    cv2.imwrite('res_3.png', img * 255)
    cv2.waitKey()













