import os
import json
import argparse
from tqdm import tqdm
import numpy as np
import torch

from src.models.networks.wan_wrapper import WanTextEncoderWrapper


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--wan_dir", type=str, default="./resources/ckpts/Wan2.2-TI2V-5B", help="Path to Wan model directory")
    parser.add_argument("--caption_json_path", type=str, default="resources/captions/dl3dv_captions.json")
    parser.add_argument("--output_dir", type=str, default="dump/dl3dv/wan_prompt_embeds")

    args = parser.parse_args()

    text_encoder = WanTextEncoderWrapper(args.wan_dir).to("cuda")
    text_encoder.eval()

    captions = json.load(open(args.caption_json_path, "r"))
    for uid in tqdm(captions.keys(), ncols=125):
        short_captions, long_captions = captions[uid]["short_caption"], captions[uid]["long_caption"]

        with torch.no_grad():
            with torch.autocast("cuda", dtype=torch.bfloat16):
                short_embeds = text_encoder(short_captions)  # (B, N, D)
                long_embeds = text_encoder(long_captions)  # (B, N, D)

                short_embeds = short_embeds.float().cpu().numpy().astype(np.float16)
                long_embeds = long_embeds.float().cpu().numpy().astype(np.float16)

        os.makedirs(os.path.join(args.output_dir, uid), exist_ok=True)
        np.save(os.path.join(args.output_dir, uid, "short_embeds.npy"), short_embeds)
        np.save(os.path.join(args.output_dir, uid, "long_embeds.npy"), long_embeds)

    # captions = json.load(open(args.caption_json_path, "r"))
    # with torch.no_grad():
    #     with torch.autocast("cuda", dtype=torch.bfloat16):
    #         embeds = text_encoder(captions)  # (B, N, D)
    #         embeds = embeds.float().cpu().numpy().astype(np.float16)

    # os.makedirs(args.output_dir, exist_ok=True)
    # np.save(os.path.join(args.output_dir, "common_embeds.npy"), embeds)

    # captions = json.load(open(args.caption_json_path, "r"))
    # with torch.no_grad():
    #     with torch.autocast("cuda", dtype=torch.bfloat16):
    #         embeds = text_encoder(captions)  # (B, N, D)
    #         embeds = embeds.float().cpu().numpy().astype(np.float16)

    # os.makedirs(args.output_dir, exist_ok=True)
    # np.save(os.path.join(args.output_dir, "negative_embed.npy"), embeds)

if __name__ == "__main__":
    main()
