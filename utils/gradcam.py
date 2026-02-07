import os
import numpy as np
import tensorflow as tf
from tensorflow.keras.preprocessing import image
import matplotlib.pyplot as plt


def get_last_conv_layer(model):
    for layer in reversed(model.layers):
        if isinstance(layer, tf.keras.layers.Conv2D):
            return layer
    raise ValueError("No Conv2D layer found in the model.")


def generate_gradcam(model, img_path, output_path, target_size=(224, 224)):
    """
    Generate Grad-CAM heatmap for a single image and save overlay PNG.
    Works with Sequential models where building a new Model() causes errors.
    """
    # 1. Load and preprocess image
    img = image.load_img(img_path, target_size=target_size)
    x = image.img_to_array(img)
    x = np.expand_dims(x, axis=0) / 255.0

    # 2. Find last conv layer
    last_conv_layer = get_last_conv_layer(model)

    # 3. Use GradientTape on the whole model, tracking last conv output
    last_conv_output = None

    def forward_hook(layer, inputs, outputs):
        nonlocal last_conv_output
        last_conv_output = outputs

    # attach a temporary hook
    last_conv_layer._gradcam_hook = forward_hook

    @tf.custom_gradient
    def model_with_hook(inputs):
        # call model, but capture conv outputs via hook
        outputs = model(inputs)
        return outputs, lambda dy: (dy,)

    with tf.GradientTape() as tape:
        tape.watch(x)
        preds = model_with_hook(x)
        pred = preds[:, 0]

    # 4. Compute gradients w.r.t conv outputs
    grads = tape.gradient(pred, last_conv_output)

    pooled_grads = tf.reduce_mean(grads, axis=(0, 1, 2)).numpy()
    conv_outputs = last_conv_output[0].numpy()

    for i in range(pooled_grads.shape[0]):
        conv_outputs[:, :, i] *= pooled_grads[i]

    heatmap = np.mean(conv_outputs, axis=-1)
    heatmap = np.maximum(heatmap, 0)
    if np.max(heatmap) != 0:
        heatmap /= np.max(heatmap)

    # 5. Overlay on original image
    img_orig = image.load_img(img_path)
    img_orig = image.img_to_array(img_orig)

    heatmap_resized = tf.image.resize(
        heatmap[..., np.newaxis],
        (img_orig.shape[0], img_orig.shape[1])
    ).numpy().squeeze()

    plt.figure(figsize=(4, 4))
    plt.axis("off")
    plt.imshow(img_orig.astype("uint8"))
    plt.imshow(heatmap_resized, cmap="jet", alpha=0.35)
    plt.tight_layout()
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    plt.savefig(output_path, bbox_inches="tight", pad_inches=0)
    plt.close()

    return output_path
