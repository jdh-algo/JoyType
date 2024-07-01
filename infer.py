import argparse
import torch
import numpy as np
import cv2
import random

from utils.render_text import *
from diffusers import StableDiffusionControlNetPipeline
from PIL import Image
from pytorch_lightning import seed_everything


def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '--load_path', default='jdh-algo/JoyType-v1-1M',
        help='The path to which the model is loaded. The model will be pulled from Hugging Face by default, '
            'if you want to load it locally, please pre-download the model and modify the path.'
    )
    parser.add_argument(
        '--prompt', default=None, required=True,
        help='A text description of the generated images.'
    )
    parser.add_argument(
        '--input_yaml', default=None, required=True,
        help='Edit the corresponding yaml file.'
    )
    parser.add_argument(
        '--img_name', default=None, required=True,
        help='Name of the generated images.'
    )
    parser.add_argument(
        '--img_size', default=[512, 512], type=int, 
        nargs='+', help='Image\'s size you want to generate.'
    )
    parser.add_argument(
        '--font', default='Arial',
        help='Font of the generated texts.'
    )
    parser.add_argument(
        '--n_prompt', 
        default='low-res, bad anatomy, extra digit, fewer digits, cropped, worst quality, low quality, watermark, '
            'unreadable text, messy words, distorted text, disorganized writing, advertising picture',
        help='Negtive prompt'
    )
    parser.add_argument(
        '--save_path', default='results',
        help='Path to save the generated images.'
    )
    parser.add_argument(
        '--batch_size', default=4, type=int,
        help='Batch size'
    )
    parser.add_argument(
        '--controlnet_scale', default=1., type=float,
        help='Controlnet conditioning scale'
    )
    parser.add_argument(
        '--cfg', default=7.5, type=float,
        help='Classifier free guidance'
    )
    parser.add_argument(
        '--steps', default=20, type=int,
        help='Inference steps'
    )
    parser.add_argument(
        '--seed', default=-1, type=int,
        help='If set to -1, a random seed will be created.'
    )
    parser.add_argument('--device', default='cuda', help='Device you want to use.')

    return parser.parse_args()


def canny(img):
    low_threshold = 64
    high_threshold = 100

    img = cv2.Canny(img, low_threshold, high_threshold)
    img = img[:, :, None]
    img = np.concatenate([img, img, img], axis=2)

    return Image.fromarray(img)


if __name__ == '__main__':
    args = get_args()

    if args.seed < 0:
        args.seed = random.randint(0, 2147483647)
    seed_everything(args.seed)

    pipe = StableDiffusionControlNetPipeline.from_pretrained(
        args.load_path,
        torch_dtype=torch.float32,
    ).to(args.device)

    shape = tuple(args.img_size)

    render_list = parse_yaml(args.input_yaml, shape, args.font)
    img = render_all_text(render_list, shape)

    controlnet_img = canny(np.array(img))

    batch_prompt = [args.prompt for _ in range(args.batch_size)]
    batch_n_prompt = [args.n_prompt for _ in range(args.batch_size)]
    batch_img = [controlnet_img for _ in range(args.batch_size)]

    images = pipe(
        batch_prompt,
        negative_prompt=batch_n_prompt,
        image=batch_img,
        controlnet_conditioning_scale=args.controlnet_scale,
        width=shape[0],
        height=shape[1],
        num_inference_steps=args.steps,
        guidance_scale=args.cfg
    ).images

    for i, image in enumerate(images):
        image.save(f"{args.save_path}/{args.img_name}_{i}.jpg")
