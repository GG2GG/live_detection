import os
import sys

# Add YOLOX to path
YOLOX_PATH = os.path.join(os.path.dirname(__file__), "YOLOX")
sys.path.insert(0, YOLOX_PATH)

from yolox.exp import get_exp
from yolox.utils import fuse_model, get_model_info
from yolox.data.data_augment import ValTransform
from yolox.utils.visualize import vis

import torch
import cv2
import time
import numpy as np

from yolox.utils import postprocess
from yolox.models import YOLOX

def main():
    # Config
    exp = get_exp("exps/default/yolox_s.py", None)
    ckpt_file = os.path.join("assets", "yolox_s.pth")

    model = exp.get_model()
    model.eval()

    # Load checkpoint
    ckpt = torch.load(ckpt_file, map_location="cpu")
    model.load_state_dict(ckpt["model"])

    model = fuse_model(model)
    print("Model Summary:", get_model_info(model, exp.test_size))

    decoder = None
    predictor = Predictor(model, exp, decoder=decoder, device="cpu", fp16=False)
    
    # Open webcam
    cap = cv2.VideoCapture(0)
    while True:
        ret, frame = cap.read()
        if not ret:
            break

        result_frame = predictor.inference(frame)
        cv2.imshow("YOLOX Webcam", result_frame)

        if cv2.waitKey(1) == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()


class Predictor:
    def __init__(self, model, exp, decoder=None, device="cpu", fp16=False):
        self.model = model
        self.cls_names = exp.class_names
        self.decode = decoder
        self.num_classes = exp.num_classes
        self.confthre = exp.test_conf
        self.nmsthre = exp.nmsthre
        self.test_size = exp.test_size
        self.device = device
        self.fp16 = fp16

        self.preproc = ValTransform(legacy=False)

    def inference(self, origin_img):
        img, ratio = self.preproc(origin_img, None, self.test_size)
        img = torch.from_numpy(img).unsqueeze(0).float()

        img = img.to(self.device)
        with torch.no_grad():
            outputs = self.model(img)
            outputs = postprocess(
                outputs, self.num_classes, self.confthre, self.nmsthre
            )
        if outputs[0] is None:
            return origin_img
        outputs[0][:, :4] /= ratio
        result_frame = vis(origin_img, outputs[0], self.cls_names)
        return result_frame


if __name__ == "__main__":
    main()
