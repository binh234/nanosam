import numpy as np

from nanosam import Predictor

import glob
import gradio as gr
import time
from PIL import ImageDraw
from utils import fast_process, format_results, point_prompt

# Most of our demo code is from [FastSAM Demo](https://huggingface.co/spaces/An-619/FastSAM). Huge thanks for AN-619.

# Load the pre-trained model
# image_encoder = "../data/resnet18_image_encoder.onnx"
# mask_decoder = "../data/mobile_sam_mask_decoder.onnx"
image_encoder = "../data/efficientvit_sam_l0_vit_h.encoder.onnx"
mask_decoder = "../data/sam_vit_h_mask_decoder.onnx"
provider = "openvino"

predictor = Predictor(image_encoder, mask_decoder, provider)

# Description
title = "<center><strong><font size='8'>Faster Segment Anything(NanoSAM)<font></strong></center>"

description_p = """ ##This is a demo of [Faster Segment Anything(MobileSAM) Model](https://github.com/ChaoningZhang/MobileSAM).
                # Instructions for point mode
                0. Restart by click the Restart button
                1. Select a point with Add Mask for the foreground (Must)
                2. Select a point with Remove Area for the background (Optional)
                3. Click the Start Segmenting.
                - Github [link](https://github.com/binh234/nanosam)
                - Model Card [link](https://huggingface.co/dragoswing/nanosam)
                We will provide box mode soon. 
                Enjoy!
              """

examples = [
    ["assets/picture3.jpg"],
    ["assets/picture4.jpg"],
    ["assets/picture5.jpg"],
    ["assets/picture6.jpg"],
    ["assets/picture1.jpg"],
    ["assets/picture2.jpg"],
    ["assets/dogs.jpg"],
]

css = "h1 { text-align: center } .about { text-align: justify; padding-left: 10%; padding-right: 10%; }"


def clear_point_data():
    global global_points, global_point_labels
    global_points = []
    global_point_labels = []


def set_image(image):
    global predictor
    print("Set image")
    clear_point_data()
    predictor.set_image(image)


def segment_with_points(
    image,
    better_quality=False,
    withContours=True,
    use_retina=True,
    mask_random_color=True,
):
    global global_points
    global global_point_labels
    global predictor

    points = np.array(global_points)
    point_labels = np.array(global_point_labels)
    if len(points) == 0 and len(point_labels) == 0:
        raise gr.Error("No points selected")

    img_w, img_h = image.size
    start = time.perf_counter()
    masks, scores, logits = predictor.predict(
        points=points,
        point_labels=point_labels,
    )
    end = time.perf_counter()
    print(f"Inference time: {end - start: .2f}s")

    results = format_results(masks[0], scores[0], logits[0], 0)

    annotations, _ = point_prompt(results, global_points, global_point_labels, img_h, img_w)
    annotations = np.array([annotations])

    fig = fast_process(
        annotations=annotations,
        image=image,
        scale=1,
        better_quality=better_quality,
        mask_random_color=mask_random_color,
        bbox=None,
        use_retina=use_retina,
        withContours=withContours,
    )

    # return fig, None
    return fig, image


def get_points_with_draw(image, label, evt: gr.SelectData):
    global global_points
    global global_point_labels

    x, y = evt.index[0], evt.index[1]
    point_radius, point_color = 15, (255, 255, 0) if label == "Add Mask" else (
        255,
        0,
        255,
    )
    global_points.append([x, y])
    global_point_labels.append(1 if label == "Add Mask" else 0)

    print(x, y, label == "Add Mask")

    draw = ImageDraw.Draw(image)
    draw.ellipse(
        [(x - point_radius, y - point_radius), (x + point_radius, y + point_radius)],
        fill=point_color,
    )
    return image


cond_img_p = gr.Image(label="Input with points", type="pil", interactive=True)

segm_img_p = gr.Image(label="Segmented Image with points", interactive=False, type="pil")

global_points = []
global_point_labels = []

with gr.Blocks(css=css, title="Faster Segment Anything(NanoSAM)") as demo:
    with gr.Row():
        with gr.Column(scale=1):
            # Title
            gr.Markdown(title)

    with gr.Tab("Point mode"):
        # Images
        with gr.Row(variant="panel"):
            with gr.Column(scale=0.8):
                cond_img_p.render()

            with gr.Column(scale=1):
                segm_img_p.render()

        # Submit & Clear
        with gr.Row():
            with gr.Column():
                with gr.Row():
                    add_or_remove = gr.Radio(
                        ["Add Mask", "Remove Area"],
                        value="Add Mask",
                    )

                    with gr.Column():
                        segment_btn_p = gr.Button("Start segmenting!", variant="primary")
                        clear_btn_p = gr.Button("Restart", variant="secondary")

                gr.Markdown("Try some of the examples below ⬇️")
                gr.Examples(
                    examples=examples,
                    inputs=[cond_img_p],
                    # outputs=segm_img_p,
                    fn=set_image,
                    run_on_click=True,
                    examples_per_page=4,
                )

            with gr.Column():
                # Description
                gr.Markdown(description_p)

    cond_img_p.upload(set_image, inputs=[cond_img_p])
    cond_img_p.select(get_points_with_draw, [cond_img_p, add_or_remove], cond_img_p)

    segment_btn_p.click(segment_with_points, inputs=[cond_img_p], outputs=[segm_img_p, cond_img_p])

    def clear():
        clear_point_data()
        return None, None

    clear_btn_p.click(clear, outputs=[cond_img_p, segm_img_p])

demo.queue()
demo.launch()
