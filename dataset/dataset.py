import numpy as np
import torch
import pydicom
from pydicom.pixel_data_handlers import apply_windowing
from pydicom.pixel_data_handlers.util import apply_modality_lut
import os
from .utils import normalize, XRayResizer


class DicomDataset():
    def __init__(
        self,
        imgpath,
        image_size=1792,
        seed=0
    ):
        super(DicomDataset, self).__init__()
        np.random.seed(seed)

        self.imgpath = imgpath
        self.xray_resizer = XRayResizer(1792)
        self.image_names = []
        for root, dirs, files in os.walk(imgpath):
            for file in files:
                if file.endswith('.dcm'):
                    self.image_names.append(file)

    def __str__(self):
        num_samples = len(self)
        return f'{num_samples}'

    def string(self):
        return str(self)

    def __len__(self):
        return len(self.image_names)

    def __getitem__(self, idx):
        image_name = self.image_names[idx]
        img_path = os.path.join(self.imgpath, image_name)

        dicom_obj = pydicom.filereader.dcmread(img_path)

        img = apply_modality_lut(dicom_obj.pixel_array, dicom_obj)
        img = apply_windowing(img, dicom_obj)

        # Photometric Interpretation to see if the image needs to be inverted
        mode = dicom_obj[0x28, 0x04].value
        bitstored = dicom_obj.BitsStored
        maxval = 2 ** bitstored - 1

        img = img.astype(np.float32)

        if mode == "MONOCHROME1":
            img = (maxval) - img
        elif mode == "MONOCHROME2":
            pass
        elif mode == "RGB":
            img = np.dot(img[..., :3], [0.2989, 0.5870, 0.1140])
        else:
            raise Exception("Unknown Photometric Interpretation mode")

        sample = normalize(img, img.max(), reshape=True)

        sample = self.xray_resizer(sample)

        sample = torch.from_numpy(sample)

        sample = sample.unsqueeze(0)

        return sample

