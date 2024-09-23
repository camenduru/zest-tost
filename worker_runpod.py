import os, json, requests, random, runpod

from diffusers import StableDiffusionXLControlNetInpaintPipeline, ControlNetModel
from rembg import remove
from PIL import Image
import torch
from ip_adapter import IPAdapterXL
from ip_adapter.utils import register_cross_attention_hook
from PIL import Image, ImageChops, ImageEnhance
import numpy as np
import cv2

import DPT.util.io
from torchvision.transforms import Compose
from DPT.dpt.models import DPTDepthModel
from DPT.dpt.transforms import Resize, NormalizeImage, PrepareForNet

def greet(input_image, material_exemplar, transform, model, ip_model):
    img = np.array(input_image)
    img_input = transform({"image": img})["image"]
    with torch.no_grad():
        sample = torch.from_numpy(img_input).unsqueeze(0)
        prediction = model.forward(sample)
        prediction = (
            torch.nn.functional.interpolate(
                prediction.unsqueeze(1),
                size=img.shape[:2],
                mode="bicubic",
                align_corners=False,
            )
            .squeeze()
            .cpu()
            .numpy()
        )
    depth_min = prediction.min()
    depth_max = prediction.max()
    bits = 2
    max_val = (2 ** (8 * bits)) - 1
    if depth_max - depth_min > np.finfo("float").eps:
        out = max_val * (prediction - depth_min) / (depth_max - depth_min)
    else:
        out = np.zeros(prediction.shape, dtype=depth.dtype)
    out = (out / 256).astype('uint8')
    depth_map = Image.fromarray(out).resize((1024, 1024))
    rm_bg = remove(input_image)
    target_mask = rm_bg.convert("RGB").point(lambda x: 0 if x < 1 else 255).convert('L').convert('RGB')
    mask_target_img = ImageChops.lighter(input_image, target_mask)
    invert_target_mask = ImageChops.invert(target_mask)
    gray_target_image = input_image.convert('L').convert('RGB')
    gray_target_image = ImageEnhance.Brightness(gray_target_image)
    factor = 1.0  # Try adjusting this to get the desired brightness
    gray_target_image = gray_target_image.enhance(factor)
    grayscale_img = ImageChops.darker(gray_target_image, target_mask)
    img_black_mask = ImageChops.darker(input_image, invert_target_mask)
    grayscale_init_img = ImageChops.lighter(img_black_mask, grayscale_img)
    init_img = grayscale_init_img
    ip_image = material_exemplar.resize((1024, 1024))
    init_img = init_img.resize((1024,1024))
    mask = target_mask.resize((1024, 1024))
    num_samples = 1
    images = ip_model.generate(pil_image=ip_image, image=init_img, control_image=depth_map, mask_image=mask, controlnet_conditioning_scale=0.9, num_samples=num_samples, num_inference_steps=30, seed=42)
    return images[0]

with torch.inference_mode():
    base_model_path = "/content/sdxl"
    image_encoder_path = "models/image_encoder"
    ip_ckpt = "sdxl_models/ip-adapter_sdxl_vit-h.bin"
    controlnet_path = "/content/diffusers/controlnet"
    device = "cuda"
    torch.cuda.empty_cache()
    controlnet = ControlNetModel.from_pretrained(controlnet_path, variant="fp16", use_safetensors=True, torch_dtype=torch.float16).to(device)
    pipe = StableDiffusionXLControlNetInpaintPipeline.from_pretrained(
        base_model_path,
        controlnet=controlnet,
        variant="fp16",
        use_safetensors=True,
        torch_dtype=torch.float16,
        add_watermarker=False,
    ).to(device)
    pipe.unet = register_cross_attention_hook(pipe.unet)
    ip_model = IPAdapterXL(pipe, image_encoder_path, ip_ckpt, device)
    model_path = "DPT/weights/dpt_hybrid-midas-501f0c75.pt"
    net_w = net_h = 384
    model = DPTDepthModel(
        path=model_path,
        backbone="vitb_rn50_384",
        non_negative=True,
        enable_attention_hooks=False,
    )
    normalization = NormalizeImage(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
    transform = Compose(
            [
                Resize(
                    net_w,
                    net_h,
                    resize_target=None,
                    keep_aspect_ratio=True,
                    ensure_multiple_of=32,
                    resize_method="minimal",
                    image_interpolation_method=cv2.INTER_CUBIC,
                ),
                normalization,
                PrepareForNet(),
            ]
        )
    model.eval()

def download_file(url, save_dir):
    os.makedirs(save_dir, exist_ok=True)
    file_name = url.split('/')[-1]
    file_path = os.path.join(save_dir, file_name)
    response = requests.get(url)
    response.raise_for_status()
    with open(file_path, 'wb') as file:
        file.write(response.content)
    return file_path

@torch.inference_mode()
def generate(input):
    values = input["input"]

    input_image = values['input_image_check']
    input_image = download_file(url=input_image, save_dir='/content')
    material_image = values['material_image']
    material_image = download_file(url=material_image, save_dir='/content')

    input_image1 = Image.open(input_image)
    input_image2 = Image.open(material_image)
    output_image = greet(input_image1, input_image2, transform, model, ip_model)
    output_image.save('/content/zest_tost.png')

    result = "/content/zest_tost.png"
    try:
        notify_uri = values['notify_uri']
        del values['notify_uri']
        notify_token = values['notify_token']
        del values['notify_token']
        discord_id = values['discord_id']
        del values['discord_id']
        if(discord_id == "discord_id"):
            discord_id = os.getenv('com_camenduru_discord_id')
        discord_channel = values['discord_channel']
        del values['discord_channel']
        if(discord_channel == "discord_channel"):
            discord_channel = os.getenv('com_camenduru_discord_channel')
        discord_token = values['discord_token']
        del values['discord_token']
        if(discord_token == "discord_token"):
            discord_token = os.getenv('com_camenduru_discord_token')
        job_id = values['job_id']
        del values['job_id']
        default_filename = os.path.basename(result)
        with open(result, "rb") as file:
            files = {default_filename: file.read()}
        payload = {"content": f"{json.dumps(values)} <@{discord_id}>"}
        response = requests.post(
            f"https://discord.com/api/v9/channels/{discord_channel}/messages",
            data=payload,
            headers={"Authorization": f"Bot {discord_token}"},
            files=files
        )
        response.raise_for_status()
        result_url = response.json()['attachments'][0]['url']
        notify_payload = {"jobId": job_id, "result": result_url, "status": "DONE"}
        web_notify_uri = os.getenv('com_camenduru_web_notify_uri')
        web_notify_token = os.getenv('com_camenduru_web_notify_token')
        if(notify_uri == "notify_uri"):
            requests.post(web_notify_uri, data=json.dumps(notify_payload), headers={'Content-Type': 'application/json', "Authorization": web_notify_token})
        else:
            requests.post(web_notify_uri, data=json.dumps(notify_payload), headers={'Content-Type': 'application/json', "Authorization": web_notify_token})
            requests.post(notify_uri, data=json.dumps(notify_payload), headers={'Content-Type': 'application/json', "Authorization": notify_token})
        return {"jobId": job_id, "result": result_url, "status": "DONE"}
    except Exception as e:
        error_payload = {"jobId": job_id, "status": "FAILED"}
        try:
            if(notify_uri == "notify_uri"):
                requests.post(web_notify_uri, data=json.dumps(error_payload), headers={'Content-Type': 'application/json', "Authorization": web_notify_token})
            else:
                requests.post(web_notify_uri, data=json.dumps(error_payload), headers={'Content-Type': 'application/json', "Authorization": web_notify_token})
                requests.post(notify_uri, data=json.dumps(error_payload), headers={'Content-Type': 'application/json', "Authorization": notify_token})
        except:
            pass
        return {"jobId": job_id, "result": f"FAILED: {str(e)}", "status": "FAILED"}
    finally:
        if os.path.exists(result):
            os.remove(result)
        if os.path.exists(input_image):
            os.remove(input_image)
        if os.path.exists(material_image):
            os.remove(material_image)

runpod.serverless.start({"handler": generate})