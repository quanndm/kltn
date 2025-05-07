import torch
from monai.inferers import sliding_window_inference

# Assuming the variables roi_size, sw_batch_size, and overlap are lists
roi = (128, 128, 128)
roi_2d = (3, 256, 256) 
sw_batch_size = 1
overlap = 0.5

VAL_AMP = True
device = "cuda" if torch.cuda.is_available() else "cpu"

def model_inferer(input, model):
    def _compute(input):
        return sliding_window_inference(
            inputs=input,
            roi_size= roi,
            sw_batch_size=sw_batch_size,
            predictor=model,
            overlap=overlap,
        )

    if VAL_AMP:
        with torch.autocast(device):
            return _compute(input)
    else:
        return _compute(input)


def inference(input, model):
    def _compute(input):
        return sliding_window_inference(
            inputs=input,
            roi_size=roi,
            sw_batch_size=sw_batch_size,
            predictor=model,
            overlap=overlap,
        )
    if VAL_AMP:
        with torch.autocast(device):
            return _compute(input)
    else:
        return _compute(input)

        
def model_inferer_2d(input, model):
    def _compute(input):
        return sliding_window_inference(
            inputs=input,
            roi_size= roi_2d,
            sw_batch_size=sw_batch_size,
            predictor=model,
            overlap=overlap,
        )

    if VAL_AMP:
        with torch.autocast(device):
            return _compute(input)
    else:
        return _compute(input)


def inference_2d(input, model):
    def _compute(input):
        return sliding_window_inference(
            inputs=input,
            roi_size=roi_2d,
            sw_batch_size=sw_batch_size,
            predictor=model,
            overlap=overlap,
        )
    if VAL_AMP:
        with torch.autocast(device):
            return _compute(input)
    else:
        return _compute(input)