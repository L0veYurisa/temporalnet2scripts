import os
import glob
import requests
import json
import cv2
import numpy as np
import re
import sys
import torch
from PIL import Image
from pprint import pprint
import base64
from io import BytesIO
import torchvision.transforms.functional as F
from torchvision.io import read_video, read_image, ImageReadMode
from torchvision.models.optical_flow import Raft_Large_Weights
from torchvision.models.optical_flow import raft_large
from torchvision.io import read_video, read_image, ImageReadMode
from torchvision.utils import flow_to_image
import cv2
from torchvision.io import write_jpeg
import pickle

import argparse


def get_args():
    parser = argparse.ArgumentParser()

    parser.add_argument('--prompt', dest='prompt', default="")
    parser.add_argument('--negative-prompt', dest='negative_prompt', default="")

    parser.add_argument('--init-image', dest='init_image', default="./init.png")
    parser.add_argument('--input-dir', dest='input_dir', default="./Input_Images")
    parser.add_argument('--output-dir', dest='output_dir', default="./output")

    parser.add_argument('--width', default=720, type=int)
    parser.add_argument('--height', default=1280, type=int)

    return parser.parse_args()


args = get_args()


device = "cuda" if torch.cuda.is_available() else "cpu"

model = raft_large(weights=Raft_Large_Weights.DEFAULT, progress=False).to(device)
model = model.eval()

# Replace with the actual path to your image file and folder

os.makedirs(args.output_dir, exist_ok=True)


def get_image_paths(folder):
    image_extensions = ("*.jpg", "*.jpeg", "*.png", "*.bmp")
    files = []
    for ext in image_extensions:
        files.extend(glob.glob(os.path.join(folder, ext)))
    return sorted(files)


y_paths = get_image_paths(args.input_dir)


def get_controlnet_models():
    url = "http://localhost:7860/controlnet/model_list"

    temporalnet_model = None
    temporalnet_re = re.compile("^temporalnetversion2 \[.{8}\]")

    ip2p_model = None
    ip2p_re = re.compile("^control_.*ip2p.* \[.{8}\]")
    
    tile_model = None
    tile_re = re.compile("^control_.*tile.* \[.{8}\]")

    depth_model = None
    depth_re = re.compile("^control_.*depth.* \[.{8}\]")

    lineart_model = None
    lineart_re = re.compile("^control_.*lineart.* \[.{8}\]")
    
    softedge_model = None
    softedge_re = re.compile("^control_.*softedge.* \[.{8}\]")
    
    response = requests.get(url)
    if response.status_code == 200:
        models = json.loads(response.content)
    else:
        raise Exception("Unable to list models from the SD Web API! "
                        "Is it running and is the controlnet extension installed?")

    for model in models['model_list']:
        if temporalnet_model is None and temporalnet_re.match(model):
            temporalnet_model = model
        elif ip2p_model is None and ip2p_re.match(model):
            ip2p_model = model
        elif tile_model is None and tile_re.match(model):
            tile_model = model
        elif depth_model is None and depth_re.match(model):
            depth_model = model
        elif lineart_model is None and lineart_re.match(model):
            lineart_model = model
        elif softedge_model is None and softedge_re.match(model):
            softedge_model = model
    
    assert temporalnet_model is not None, "Unable to find the temporalnet2 model!  Ensure it's copied into the stable-diffusion-webui/extensions/models directory!"
    assert ip2p_model is not None, "Unable to find the ip2p_model model!  Ensure it's copied into the stable-diffusion-webui/extensions/models directory!"
    assert depth_model is not None, "Unable to find the depth model!  Ensure it's copied into the stable-diffusion-webui/extensions/models directory!"
    assert tile_model is not None, "Unable to find the tile model!  Ensure it's copied into the stable-diffusion-webui/extensions/models directory!"
    assert lineart_model is not None, "Unable to find the lineart model!  Ensure it's copied into the stable-diffusion-webui/extensions/models directory!"
    assert softedge_model is not None, "Unable to find the lineart model!  Ensure it's copied into the stable-diffusion-webui/extensions/models directory!"
    return temporalnet_model, ip2p_model, depth_model, tile_model, lineart_model, softedge_model


TEMPORALNET_MODEL, IP2P_MODEL, DEPTH_MODEL, TILE_MODEL, LINEART_MODEL, SOFTEDGE_MODEL  = get_controlnet_models()


def send_request(last_image_path, optical_flow_path,current_image_path):
    url = "http://localhost:7860/sdapi/v1/txt2img"
    
    with open(last_image_path, "rb") as b:
       last_image_encoded = base64.b64encode(b.read()).decode("utf-8")
    
    # Load and process the last image
    last_image = cv2.imread(last_image_path)
    last_image = cv2.cvtColor(last_image, cv2.COLOR_BGR2RGB)

    # Load and process the optical flow image
    flow_image = cv2.imread(optical_flow_path)
    flow_image = cv2.cvtColor(flow_image, cv2.COLOR_BGR2RGB)

    # Load and process the current image
    with open(current_image_path, "rb") as b:
       current_image = base64.b64encode(b.read()).decode("utf-8")


    # Concatenating the three images to make a 6-channel image
    six_channel_image = np.dstack((last_image, flow_image))

    # Serializing the 6-channel image
    serialized_image = pickle.dumps(six_channel_image)

    # Encoding the serialized image
    encoded_image = base64.b64encode(serialized_image).decode('utf-8')

    data = {
        "init_images": [current_image],
        "inpainting_fill": 0,
        "inpaint_full_res": True,
        "inpaint_full_res_padding": 1,
        "inpainting_mask_invert": 1,
        "resize_mode": 0,
        "prompt": args.prompt,
        "negative_prompt": args.negative_prompt,
        "alwayson_scripts": {
            "ControlNet":{
                "args": [
                    {
                        "input_image": current_image,
                        "module": "lineart_realistic",
                        "model": LINEART_MODEL,
                        "weight": 0.35,
                        "guidance": 1,
                        "pixel_perfect": True,
                        "resize_mode": 0,
                        "control_mode": 0,
                   },
                    {
                        "input_image": encoded_image,
                        "model": TEMPORALNET_MODEL,
                        "module": "none",
                        "weight": 1.0,
                        "guidance_end": 0.4,
                        "guidance_start": 0,
                        # "processor_res": 512,
                        "pixel_perfect": True,
                        "threshold_a": 64,
                        "threshold_b": 64,
                        "resize_mode": 0,
                        "control_mode": 2,
                    },
                    {
                        "input_image": current_image,
                        "model": IP2P_MODEL,
                        "module": "none",
                        "weight": 1.0,
                        "guidance": 1,
                        "guidance_end": 0.6,
                        "guidance_start": 0,
                        "pixel_perfect": True,
                        "resize_mode": 0,
                        "control_mode": 0,
                    },
                    #{
                    #    "input_image": current_image,
                    #    "model": TILE_MODEL,
                    #    "module": "tile_colorfix",
                    #    "weight": 0.4,
                    #    "guidance": 1,
                    #   "pixel_perfect": True,
                    #    "resize_mode": 0,
                    #    "control_mode": 0,
                   # },
                  
                ]
            }
        },
        "seed": 1849402459,
        "subseed": -1,
        "subseed_strength": -1,
        "sampler_index": "Euler a",
        "batch_size": 1,
        "n_iter": 1,
        "steps": 22,
        "cfg_scale": 4,
        "width": args.width,
        "height": args.height,
        "restore_faces": True,
        "include_init_images": True,
        "override_settings": {},
        "override_settings_restore_afterwards": True
    }
    response = requests.post(url, json=data)
    if response.status_code == 200:
        return response.content
    else:
        try:
            error_data = response.json()
            print("Error:")
            print(str(error_data))
            
        except json.JSONDecodeError:
            print(f"Error: Unable to parse JSON error data.")
        return None



def infer(frameA, frameB):
    
    
    input_frame_1 = read_image(str(frameA), ImageReadMode.RGB)
   
    input_frame_2 = read_image(str(frameB), ImageReadMode.RGB)
 
    
    #img1_batch = torch.stack([frames[0]])
    #img2_batch = torch.stack([frames[1]])

    img1_batch = torch.stack([input_frame_1])
    img2_batch = torch.stack([input_frame_2])
    
    
    weights = Raft_Large_Weights.DEFAULT
    transforms = weights.transforms()


    def preprocess(img1_batch, img2_batch):
        img1_batch = F.resize(img1_batch, size=[512, 512])
        img2_batch = F.resize(img2_batch, size=[512, 512])
        return transforms(img1_batch, img2_batch)

    img1_batch, img2_batch = preprocess(img1_batch, img2_batch)

    list_of_flows = model(img1_batch.to(device), img2_batch.to(device))

    predicted_flow = list_of_flows[-1][0]
    opitcal_flow_path = os.path.join(args.output_dir, f"flow_{i}.png")

    flow_img = flow_to_image(predicted_flow).to("cpu")
    flow_img = F.resize(flow_img, size=[args.height, args.width])

    write_jpeg(flow_img, opitcal_flow_path)

    return opitcal_flow_path

output_images = []
output_paths = []

# Initialize with the first image path

result = args.init_image
output_image_path = os.path.join(args.output_dir, f"output_image_0.png")

#with open(output_image_path, "wb") as f:
   # f.write(result)
    
last_image_path = args.init_image
for i in range(1, len(y_paths)):
    # Use the last image path and optical flow map to generate the next input
    optical_flow = infer(y_paths[i - 1], y_paths[i])
    
    # Modify your send_request to use the last_image_path
    result = send_request(last_image_path, optical_flow, y_paths[i])
    data = json.loads(result)

    for j, encoded_image in enumerate(data["images"]):
        if j == 0:
            output_image_path = os.path.join(args.output_dir, f"output_image_{i}.png")
            last_image_path = output_image_path
        else:
            output_image_path = os.path.join(args.output_dir, f"controlnet_image_{j}_{i}.png")

        with open(output_image_path, "wb") as f:
           f.write(base64.b64decode(encoded_image))
    print(f"Written data for frame {i}:")
