import os
import torch
import argparse
import gradio as gr
import numpy as np
from PIL import Image
import BEN2
import logging
from datetime import datetime


# 配置日志系统
def setup_logging():
    log_format = '\033[33m%(asctime)s \033[0m[%(threadName)s] %(levelname)s \033[32m[%(filename)s-%(funcName)s-%(lineno)d]\033[0m - %(message)s'
    logging.basicConfig(
        level=logging.INFO,
        format=log_format,
        handlers=[
            logging.StreamHandler(),
            logging.FileHandler("../log/matting_tool.log")
        ]
    )


setup_logging()
logger = logging.getLogger(__name__)

# 解析命令行参数
parser = argparse.ArgumentParser(description="BEN2: Background Erase Network")
parser.add_argument('--port', type=int, required=True, help="Gradio port")
parser.add_argument('--device', type=str, default='cuda')
args = parser.parse_args()

# 设置设备
if args.device == 'cuda':
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
else:
    device = torch.device('cpu')
logger.info(f"using device: {device}")

# 加载模型
try:
    model = BEN2.BEN_Base().to(device).eval()
    model_path = "/home/hx/workspace/projects/video_python/ai_change_cloth_auto/model/matting_model/BEN2/BEN2_Base.pth"
    model.loadcheckpoints(model_path)

    if device.type == 'cpu':
        model = model.float()  # 将模型转换为单精度
        torch.set_float32_matmul_precision('high')  # 可选，设置矩阵乘法精度
    logger.info(f"successfully load: {model_path}")
except Exception as e:
    logger.error(f"fail to load: {str(e)}")
    raise


def process_single_image(image):
    """处理单张图片"""
    try:
        start_time = datetime.now()
        logger.info("开始处理单张图片...")

        # 转换输入格式
        if isinstance(image, np.ndarray):
            logger.debug("输入为numpy数组，转换为PIL图像")
            image = Image.fromarray(image)
        elif isinstance(image, str):
            logger.debug(f"输入为文件路径: {image}")
            image = Image.open(image)
        else:
            error_msg = f"不支持的图片类型: {type(image)}"
            logger.error(error_msg)
            raise TypeError(error_msg)

        logger.debug("将图片转换为RGB格式")
        image = image.convert("RGB")

        # 执行抠图
        logger.info("正在进行抠图处理...")
        foreground = model.inference(image, refine_foreground=False)

        # 保存结果
        output_path = "/tmp/foreground.png"
        foreground.save(output_path, format="PNG")
        logger.info(f"抠图完成! 结果已保存到: {output_path}")

        elapsed = (datetime.now() - start_time).total_seconds()
        logger.info(f"单张图片处理耗时: {elapsed:.2f}秒")

        return foreground, output_path

    except Exception as e:
        logger.error(f"处理单张图片时出错: {str(e)}", exc_info=True)
        raise


def process_folder(folder_path, output_folder):
    """批量处理文件夹中的图片"""
    try:
        start_time = datetime.now()
        logger.info(f"开始批量处理文件夹: {folder_path}")

        if not os.path.isdir(folder_path):
            error_msg = f"输入文件夹不存在: {folder_path}"
            logger.error(error_msg)
            raise FileNotFoundError(error_msg)

        os.makedirs(output_folder, exist_ok=True)
        logger.info(f"创建输出文件夹: {output_folder}")

        processed_count = 0
        for image_item in os.listdir(folder_path):
            image_path = os.path.join(folder_path, image_item)
            if image_path.lower().endswith(('png', 'jpg', 'jpeg')):
                try:
                    logger.info(f"正在处理: {image_path}")
                    image = Image.open(image_path)
                    foreground = model.inference(image, refine_foreground=False)

                    output_path = os.path.join(output_folder, f"foreground-{image_item}")
                    foreground.save(output_path)
                    processed_count += 1
                    logger.debug(f"已保存结果到: {output_path}")
                except Exception as e:
                    logger.error(f"处理图片 {image_path} 时出错: {str(e)}", exc_info=True)
                    continue

        elapsed = (datetime.now() - start_time).total_seconds()
        logger.info(f"批量处理完成! 共处理 {processed_count} 张图片, 总耗时: {elapsed:.2f}秒")
        return f"所有图片处理完成! 共处理 {processed_count} 张图片, 结果保存在: {output_folder}"

    except Exception as e:
        logger.error(f"批量处理文件夹时出错: {str(e)}", exc_info=True)
        raise


# Gradio界面
with gr.Blocks(title="BEN2: Background Erase Network") as app:
    gr.Markdown("# 🖼️ BEN2: Background Erase Network")
    gr.Markdown("Support single image file or Batch image folder")

    with gr.Row():
        with gr.Column():
            image_input = gr.Image(label="上传图片/Upload image")
            process_button = gr.Button("开始抠图/Start", variant="primary")
        with gr.Column():
            image_output = gr.Image(label="抠图结果/Result")
            download_file = gr.File(label="下载 PNG/ Download")

    process_button.click(
        process_single_image,
        inputs=image_input,
        outputs=[image_output, download_file],
        api_name="single_image_matting"
    )

    with gr.Row():
        folder_input = gr.Textbox(label="输入图片文件夹路径/Input image folder path", placeholder="请输入包含图片的文件夹路径/Please enter the folder path containing the picture")
        output_folder_input = gr.Textbox(label="输出文件夹路径/Output image folder path", placeholder="请输入保存结果的文件夹路径/Please enter the folder path where you will save the result")
        process_folder_button = gr.Button("批量抠图/Batch process", variant="primary")

    folder_output_info = gr.Textbox(label="处理结果/Result")

    process_folder_button.click(
        process_folder,
        inputs=[folder_input, output_folder_input],
        outputs=folder_output_info,
        api_name="batch_matting"
    )

# 启动应用
try:
    logger.info(f"Start Gradio service, port: {args.port}")
    app.launch(server_name="0.0.0.0", server_port=args.port)
except Exception as e:
    logger.error(f"Fail to start Gradio service: {str(e)}", exc_info=True)
