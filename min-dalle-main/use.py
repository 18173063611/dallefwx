import torch
from PIL import Image
import numpy as np
from min_dalle import MinDalle
import os
import argparse

def main():
    # 解析命令行参数
    parser = argparse.ArgumentParser(description='Generate images from text using min-DALL·E.')
    parser.add_argument('--text', type=str, required=True, help='The text prompt for generating images')
    parser.add_argument('--seed', type=int, default=-1, help='Random seed for reproducibility')
    parser.add_argument('--grid_size', type=int, default=4, help='Grid size for the generated image')
    parser.add_argument('--is_seamless', action='store_true', help='Generate seamless images')
    parser.add_argument('--temperature', type=float, default=1, help='Sampling temperature')
    parser.add_argument('--top_k', type=int, default=256, help='Top-k sampling')
    parser.add_argument('--supercondition_factor', type=int, default=32, help='Super conditioning factor')
    args = parser.parse_args()

    # 配置模型参数
    model = MinDalle(
        models_root='./pretrained',
        dtype=torch.float32,  # 可以改为 torch.float16 节省GPU内存
        device='cuda',  # 或 'cpu'
        is_mega=True,
        is_reusable=True
    )

    # 生成图像
    image = model.generate_image(
        text=args.text,
        seed=args.seed,
        grid_size=args.grid_size,
        is_seamless=args.is_seamless,
        temperature=args.temperature,
        top_k=args.top_k,
        supercondition_factor=args.supercondition_factor,
        is_verbose=False
    )

    # 显示图像
    image.show()

    # 处理文件名，去除不合法字符
    filename = "".join([c if c.isalnum() else "_" for c in args.text])
    filename = filename[:50]  # 限制文件名长度

    # 保存图像
    image.save(f'{filename}.png')

if __name__ == '__main__':
    main()




# python use.py --text "改为要生成的图片" --seed 42 --grid_size 4 --temperature 0.7 --top_k 128 --supercondition_factor 16


# --text：输入的文本描述，用于生成图像（必填）。
# --seed：随机种子，用于生成不同的图像（默认为-1）。
# --grid_size：生成图像网格的大小（默认为4）。
# --is_seamless：是否生成无缝图像（设置此参数则生成无缝图像）。
# --temperature：控制生成图像的多样性（默认为1）。
# --top_k：采样时考虑的最高概率的令牌数量（默认为256）。
# --supercondition_factor：控制生成图像与文本描述的一致性（默认为32）。