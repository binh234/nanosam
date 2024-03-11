import numpy as np

from nanosam import Predictor, get_config

import gradio as gr
import time
from PIL import ImageDraw
from utils import fast_process, format_results, point_prompt

# Most of our demo code is from [FastSAM Demo](https://huggingface.co/spaces/An-619/FastSAM). Huge thanks for AN-619.

# Load the pre-trained model
image_encoder_path = "../data/sam_pphgv2_b4_image_encoder.onnx"
# image_encoder_path = "../data/efficientvit_l1_image_encoder.onnx"
mask_decoder_path = "../data/efficientvit_l1_mask_decoder.onnx"
# mask_decoder_path = "../data/mobile_sam_mask_decoder.onnx"
image_encoder_cfg = get_config("../configs/inference/encoder.yaml", overrides=[f"path={image_encoder_path}", "provider=cpu"])
mask_decoder_cfg = get_config("../configs/inference/decoder.yaml", overrides=[f"path={mask_decoder_path}"])
predictor = Predictor(image_encoder_cfg, mask_decoder_cfg)

# Description
title = "<center><strong><font size='8'>Faster Segment Anything(NanoSAM)<font></strong></center>"

description_p = """ ## This is a demo of [Faster Segment Anything(NanoSAM) Model](https://github.com/binh234/nanosam).
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


def get_empty_state():
    return {"points": [], "point_labels": [], "features": None}


def clear():
    return None, None, get_empty_state()


def set_image(image):
    state = get_empty_state()
    predictor.set_image(image)
    state["features"] = predictor.features
    return state


def segment_with_points(
    image,
    state,
    better_quality=False,
    withContours=True,
    use_retina=True,
    mask_random_color=True,
):
    global predictor

    points = np.asarray(state["points"])
    point_labels = np.asarray(state["point_labels"])
    if len(points) == 0 and len(point_labels) == 0:
        raise gr.Error("No points selected")
    if len(points) != len(point_labels):
        raise gr.Error("Mismatch length between points and point labels")
    if state["features"] is None:
        raise gr.Error("Image was not set correctly, please wait for a moment after uploading image before drawing points!")

    predictor.features = state["features"]
    predictor.image = image
    start = time.perf_counter()
    masks, scores, logits = predictor.predict(
        points=points,
        point_labels=point_labels,
    )
    end = time.perf_counter()
    print(f"Inference time: {end - start: .3f}s")

    # results = format_results(masks[0], scores[0], logits[0], 0)

    # img_w, img_h = image.size
    # annotations, _ = point_prompt(results, points, point_labels, img_h, img_w)
    # annotations = np.array([annotations])

    fig = fast_process(
        annotations=[masks[0, scores.argmax()] > 0],
        image=image,
        scale=1,
        better_quality=better_quality,
        mask_random_color=mask_random_color,
        bbox=None,
        use_retina=use_retina,
        withContours=withContours,
    )

    # return fig, None
    return fig


def get_points_with_draw(image, label, evt: gr.SelectData, state):
    x, y = evt.index[0], evt.index[1]
    point_radius, point_color = 15, (
        (255, 255, 0)
        if label == "Add Mask"
        else (
            255,
            0,
            255,
        )
    )
    state["points"].append([x, y])
    state["point_labels"].append(1 if label == "Add Mask" else 0)

    print(x, y, label == "Add Mask")

    draw = ImageDraw.Draw(image)
    draw.ellipse(
        [(x - point_radius, y - point_radius), (x + point_radius, y + point_radius)],
        fill=point_color,
    )
    return image, state


cond_img_p = gr.Image(label="Input with points", type="pil", interactive=True)

segm_img_p = gr.Image(label="Segmented Image with points", interactive=False, type="pil")

global_points = []
global_point_labels = []

with gr.Blocks(css=css, title="Faster Segment Anything(NanoSAM)") as demo:
    state = gr.State(value=get_empty_state())
    with gr.Row():
        with gr.Column(scale=1):
            # Title
            gr.Markdown(title)

    with gr.Tab("Point mode"):
        # Images
        with gr.Row(variant="panel"):
            with gr.Column(scale=1):
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
                        restart_btn_p = gr.Button("Restart", variant="secondary")

                gr.Markdown("Try some of the examples below ⬇️")
                gr.Examples(
                    examples=examples,
                    inputs=[cond_img_p],
                    outputs=[state],
                    fn=set_image,
                    run_on_click=True,
                    examples_per_page=4,
                )

            with gr.Column():
                # Description
                gr.Markdown(description_p)

    cond_img_p.upload(set_image, inputs=[cond_img_p], outputs=[state])
    cond_img_p.select(get_points_with_draw, [cond_img_p, add_or_remove, state], [cond_img_p, state])
    segment_btn_p.click(segment_with_points, [cond_img_p, state], [segm_img_p])
    restart_btn_p.click(clear, outputs=[cond_img_p, segm_img_p, state])

demo.queue().launch()
