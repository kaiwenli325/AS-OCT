import logging
from os import listdir
from os.path import splitext
import numpy as np
import torch
from PIL import Image
from torch.utils.data import Dataset
import cv2


class BasicDataset(Dataset):
    def __init__(self, path, mode='train'):
        self.mode = mode
        self.da = False

        # Set your data path
        self.images_dir = path[0]
        self.masks_dir = path[1]
        self.landmarks_dir = path[2]

        self.scale = 0.5
        self.img_size = 1024
        self.ids = [splitext(file)[0] for file in listdir(self.images_dir) if not file.startswith('.')]
        if not self.ids:
            raise RuntimeError(f'No input file found in {self.images_dir}, make sure you put your images there')

        self.num_landmarks = 6
        self.sigma = 2
        self.heatmap_size = [int(self.img_size * self.scale), int(self.img_size * self.scale)]

        logging.info(f'Creating dataset with {len(self.ids)} examples')

    def __len__(self):
        return len(self.ids)

    def preprocess(self, image, scale, is_mask):
        h, w, c = image.shape
        newW, newH = int(scale * w), int(scale * h)
        assert newW > 0 and newH > 0, 'Scale is too small, resized images would have no pixel'
        img = cv2.resize(image, (newW, newH), interpolation=cv2.INTER_NEAREST if is_mask else cv2.INTER_CUBIC)

        if img.ndim == 2 and not is_mask:
            img = img[np.newaxis, ...]
        elif not is_mask:
            img = img.transpose((2, 0, 1))

        if not is_mask:
            img = img / 255

        if is_mask:
            # change colors denoted by '[0, 0, 0]' according to your segmentation masks
            img[np.logical_and.reduce(img == [0, 0, 0], axis=2)] = 0
            img[np.logical_and.reduce(img == [0, 0, 0], axis=2)] = 1
            img[np.logical_and.reduce(img == [0, 0, 0], axis=2)] = 2
            img[np.logical_and.reduce(img == [0, 0, 0], axis=2)] = 3
            img[np.logical_and.reduce(img == [0, 0, 0], axis=2)] = 4
            img[np.logical_and.reduce(img == [0, 0, 0], axis=2)] = 5
            img = img[:, :, 0]

        return img

    def load(self, filename):
        ext = splitext(filename)[1]
        if ext in ['.npz', '.npy']:
            return Image.fromarray(np.load(filename))
        elif ext in ['.pt', '.pth']:
            return Image.fromarray(torch.load(filename).numpy())
        else:
            img = cv2.imread(str(filename))
            return cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    def generate_heatmap(self, landmark_path):
        # Based on your labeling method, you should append the landmark coordinates
        # to 'landmarks' following the comment.
        # For 'landmarks_vis', '1' means that the landmark is visible
        # and '0' means that the landmark is not visible
        landmarks = []
        landmarks_vis = []
        landmarks.append()  # left_ss [col, row]
        landmarks.append()  # right_ss
        landmarks.append()  # left_iris_root
        landmarks.append()  # right_iris_root
        landmarks_vis.append()    # left_ss
        landmarks_vis.append()  # right_ss
        landmarks_vis.append()  # left_iris_root
        landmarks_vis.append()  # right_iris_root
        image_height = []  # imageHeight
        image_width = []  # imageWidth
        json_size = [image_width, image_height]

        heatmap = np.zeros((self.num_landmarks,
                               self.heatmap_size[1],
                               self.heatmap_size[0]),
                               dtype=np.float32)

        landmarks_vis = np.array(landmarks_vis, dtype=np.float32)

        tmp_size = self.sigma * 3

        for landmarks_id in range(len(landmarks)):
            feat_stride = np.array(json_size) / self.heatmap_size
            mu_x = int(landmarks[landmarks_id][0] / feat_stride[0] + 0.5)
            mu_y = int(landmarks[landmarks_id][1] / feat_stride[1] + 0.5)

            ul = [int(mu_x - tmp_size), int(mu_y - tmp_size)]
            br = [int(mu_x + tmp_size + 1), int(mu_y + tmp_size + 1)]

            # # Generate gaussian
            size = 2 * tmp_size + 1
            x = np.arange(0, size, 1, np.float32)
            y = x[:, np.newaxis]
            x0 = y0 = size // 2
            g = np.exp(- ((x - x0) ** 2 + (y - y0) ** 2) / (2 * self.sigma ** 2))

            g_x = max(0, -ul[0]), min(br[0], self.heatmap_size[0]) - ul[0]
            g_y = max(0, -ul[1]), min(br[1], self.heatmap_size[1]) - ul[1]

            img_x = max(0, ul[0]), min(br[0], self.heatmap_size[0])
            img_y = max(0, ul[1]), min(br[1], self.heatmap_size[1])

            if landmarks_id <= 1:  # ss
                heatmap[0][img_y[0]:img_y[1], img_x[0]:img_x[1]] = g[g_y[0]:g_y[1], g_x[0]:g_x[1]]
            else:  # ir
                heatmap[1][img_y[0]:img_y[1], img_x[0]:img_x[1]] = g[g_y[0]:g_y[1], g_x[0]:g_x[1]]

        return heatmap, landmarks_vis

    def __getitem__(self, idx):
        name = self.ids[idx]
        img_file = list(self.images_dir.glob(name + '.*'))
        landmark_file = list(self.landmarks_dir.glob(name + '.*'))
        mask_file = list(self.masks_dir.glob(name + '.*'))

        # data for seg
        assert len(img_file) == 1, f'Either no image or multiple images found for the ID {name}: {img_file}'
        assert len(mask_file) == 1, f'Either no mask or multiple masks found for the ID {name}: {mask_file}'
        img = self.load(img_file[0])
        mask = self.load(mask_file[0])

        assert img.size == mask.size, \
            'Image and mask {name} should be the same size, but are {img.size} and {mask.size}'

        img = self.preprocess(img, scale=self.scale, is_mask=False)
        mask = self.preprocess(mask, self.scale, is_mask=True)

        # data for landmark
        heatmap, landmarks_vis = self.generate_heatmap(landmark_file[0])

        return {
            'image': torch.as_tensor(img.copy()).float().contiguous(),
            'mask': torch.as_tensor(mask.copy()).long().contiguous(),
            'heatmap': torch.as_tensor(heatmap.copy()),
            'landmarks_vis': torch.as_tensor(landmarks_vis.copy())
        }
