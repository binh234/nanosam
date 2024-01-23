import matplotlib.pyplot as plt

from nanosam.models.paddle import build_model

import os
import paddle
import paddle.nn.functional as F
from ppcls.loss import build_loss
from ppcls.utils import config
from tqdm import tqdm

from nanosam.tools.paddle.loader import build_dataloader
from nanosam.tools.paddle.trt_model import TrtModel

if __name__ == "__main__":
    args = config.parse_args()
    config = config.get_config(args.config, overrides=args.override, show=False)

    output_dir = os.path.join(config["Global"]["output_dir"], config["Arch"]["name"])
    log_file = os.path.join(output_dir, f"log.txt")
    log_step = config["Global"]["print_batch_step"]
    num_epochs = config["Global"]["epochs"]
    teacher_size = config["Global"]["teacher_size"]
    student_size = config["Global"]["student_size"]
    batch_size = config["DataLoader"]["Train"]["sampler"]["batch_size"]

    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    if not os.path.exists(os.path.join(output_dir, "images")):
        os.makedirs(os.path.join(output_dir, "images"))

    device = paddle.set_device(config["Global"]["device"])
    image_encoder_trt = TrtModel(config["Teacher"]["path"], device)
    image_encoder_cnn = build_model(config["Arch"])

    # loss_function = build_loss(config["Loss"]["Train"])
    loss_function = F.smooth_l1_loss

    optimizer = paddle.optimizer.Adam(
        learning_rate=config["Global"].get("learning_rate", 3e-4),
        parameters=image_encoder_cnn.parameters(),
    )

    loader = build_dataloader(config["DataLoader"], "Train", device)

    checkpoint_path = os.path.join(output_dir, "checkpoint.pd")
    if os.path.exists(checkpoint_path):
        checkpoint = paddle.load(checkpoint_path)
        image_encoder_cnn.set_state_dict(checkpoint["model"])
        optimizer.set_state_dict(checkpoint["optimizer"])
        start_epoch = checkpoint["epoch"] + 1
    else:
        start_epoch = 0

    scaler = paddle.amp.GradScaler(init_loss_scaling=128)
    for epoch in range(start_epoch, num_epochs):
        epoch_loss = 0.0

        prog_bar = tqdm(enumerate(loader), total=len(loader), desc=f"Epoch {epoch + 1}")
        for cnt, batch in prog_bar:
            image = batch[0]
            if image.shape[0] != batch_size:
                continue

            with paddle.no_grad():
                features = image_encoder_trt(image)[0]

            if teacher_size != student_size:
                image = F.interpolate(image, (student_size, student_size), mode="bilinear")

            optimizer.clear_grad()
            with paddle.amp.auto_cast(level="O1"):
                output = image_encoder_cnn(image)
                loss = loss_function(output, features)

            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
            epoch_loss += float(loss)

            if (cnt + 1) % log_step == 0:
                prog_bar.set_postfix_str(f"loss: {epoch_loss / (cnt + 1):.5f}")

        epoch_loss /= len(loader)
        print(f"{epoch} - {epoch_loss: .5f}")

        with open(os.path.join(output_dir, "log.txt"), "a") as f:
            f.write(f"{epoch} - {epoch_loss}\n")

        paddle.save(
            {
                "model": image_encoder_cnn.state_dict(),
                "optimizer": optimizer.state_dict(),
                "epoch": epoch,
            },
            checkpoint_path,
        )

        plt.figure(figsize=(10, 10))
        plt.subplot(121)
        plt.imshow(features[0, 0].detach().cpu().numpy())
        plt.subplot(122)
        plt.imshow(output[0, 0].detach().cpu().numpy())
        plt.savefig(os.path.join(output_dir, "images", f"epoch_{epoch}.png"))
        plt.close()
