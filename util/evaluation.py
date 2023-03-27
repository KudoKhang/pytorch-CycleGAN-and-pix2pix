import os
import time
import sys
sys.path.insert(0, '.')

import cv2
import numpy as np
from skimage.metrics import structural_similarity as ssim
from tqdm import tqdm


class Evaluation:
    def __init__(self, epoch, project_name):
        self.epoch = epoch
        self.root = f"results/{project_name}/test_{self.epoch}/images"
        self.id_images = list(set([name.split("_")[0] for name in os.listdir(self.root) if name.endswith("png")]))
    
    def convert_RGB2GRAY(self, *args):
        return [cv2.cvtColor(image, cv2.COLOR_BGR2GRAY) for image in args]


    def read_images(self, id):
        real_B = cv2.imread(os.path.join(self.root, id + "_real_B.png"))
        fake_B = cv2.imread(os.path.join(self.root, id + "_fake_B.png"))
        return real_B, fake_B

    def run(self):

        mse_metric = []
        psnr_metric = []
        ssim_metric = []

        for id in tqdm(self.id_images):
            real_B, fake_B = self.read_images(id)
            mse = np.mean((real_B - fake_B) ** 2)
            psnr = 20 * np.log10((np.max([real_B.max(), fake_B.max()])) / np.sqrt(mse)) if mse != 0 else 0

            real_B, fake_B = self.convert_RGB2GRAY(real_B, fake_B)
            ssim_result = ssim(real_B, fake_B)

            mse_metric.append(mse)
            psnr_metric.append(psnr)
            ssim_metric.append(ssim_result)

        print("MSE: ", np.mean(mse_metric))
        print("PSNR: ", np.mean(psnr_metric))
        print("SSIM: ", np.mean(ssim_metric))
        
        return {
            "MSE": np.mean(mse_metric),
            "PSNR": np.mean(psnr_metric),
            "SSIM": np.mean(ssim_metric)
        }
