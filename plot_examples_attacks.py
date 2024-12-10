import matplotlib.pyplot as plt
import numpy as np
import torch
from cleverhans.torch.attacks.fast_gradient_method import fast_gradient_method
from cleverhans.torch.attacks.projected_gradient_descent import (
    projected_gradient_descent,
)
from cleverhans.torch.attacks.spsa import spsa
from tqdm import tqdm

from alg.vae_new import bayes_classifier
from attack_black_box import load_model
from attacks.black_box import gaussian_perturbation_attack, sticker_attack
from attacks.momentum_iterative_method import momentum_iterative_method
from utils.utils import load_data

vae_type = "B"
data_name = "gtsrb"
infty = False
bbox = True


_, test_dataset = load_data(data_name, path="./data", labels=None, conv=True)
test_dataset = torch.utils.data.Subset(test_dataset, range(1))
images, _ = next(iter(torch.utils.data.DataLoader(test_dataset, batch_size=1)))
images = images.to("cuda" if torch.cuda.is_available() else "cpu")
encoder, generator = load_model(
    data_name, vae_type, 0, device="cuda" if torch.cuda.is_available() else "cpu"
)
encoder.eval()
if vae_type == "A":
    dec = (generator.pyz_params, generator.pxzy_params)
    from alg.lowerbound_functions import lowerbound_A as lowerbound
elif vae_type == "B":
    dec = (generator.pzy_params, generator.pxzy_params)
    from alg.lowerbound_functions import lowerbound_B as lowerbound
elif vae_type == "C":
    dec = (generator.pyzx_params, generator.pxz_params)
    from alg.lowerbound_functions import lowerbound_C as lowerbound
elif vae_type == "D":
    dec = (generator.pyzx_params, generator.pzx_params)
    from alg.lowerbound_functions import lowerbound_D as lowerbound
elif vae_type == "E":
    dec = (generator.pyz_params, generator.pzx_params)
    from alg.lowerbound_functions import lowerbound_E as lowerbound
elif vae_type == "F":
    dec = (generator.pyz_params, generator.pxz_params)
    from alg.lowerbound_functions import lowerbound_F as lowerbound
elif vae_type == "G":
    dec = (generator.pzy_params, generator.pxz_params)
    from alg.lowerbound_functions import lowerbound_G as lowerbound
else:
    raise ValueError(f"Unknown VAE type: {vae_type}")
enc_conv = encoder.encoder_conv
enc_mlp = encoder.enc_mlp
enc = (enc_conv, enc_mlp)

ll = "l2"
K = 10
dimY = 43 if data_name == "gtsrb" else 10

model = lambda x: bayes_classifier(
    x,
    (enc_conv, enc_mlp),
    dec,
    ll,
    dimY,
    lowerbound=lowerbound,
    K=10,
    beta=1.0,
)
sticker_sizes = [0, 0.05, 0.08, 0.1, 0.15, 0.2]
epsilons = [0, 0.01, 0.02, 0.05, 0.1, 0.2]
if bbox:
    images_adv = [
        [
            spsa(
                model,
                images,
                eps=eps,
                nb_iter=10,
                norm=np.inf,
                clip_min=0.0,
                clip_max=1.0,
                delta=0.01,
                sanity_checks=False,
                early_stop_loss_threshold=1e-3,
            )
            for eps in tqdm(epsilons, desc="SPSA")
        ],
        [
            gaussian_perturbation_attack(images, eps=eps)
            for eps in tqdm(epsilons, desc="Gaussian")
        ],
        [
            sticker_attack(images, sticker_size=eps)
            for eps in tqdm(sticker_sizes, desc="Sticker")
        ],
    ]

    # Plot images (one line per attack)
    fig, axes = plt.subplots(3, len(epsilons), figsize=(12, 6))
    for i, img in enumerate(images_adv):
        for j, image in enumerate(img):
            axes[i, j].imshow(image[0].permute(1, 2, 0).cpu().numpy())
            axes[i, j].axis("off")
            if i == 0:
                axes[i, j].set_title(f"eps={sticker_sizes[j]}")
            else:
                axes[i, j].set_title(f"eps={epsilons[j]}")
            if j == 0:
                axes[i, j].set_ylabel(["SPSA", "Gaussian", "Sticker"][i])
    plt.tight_layout()
    plt.savefig("attacks_bbox.png")


if infty:

    images_adv = [
        [
            fast_gradient_method(
                model,
                images,
                eps=eps,
                norm=np.inf,
                clip_min=0.0,
                clip_max=1.0,
                sanity_checks=False,
            )
            for eps in tqdm(epsilons, desc="FGSM")
        ],
        [
            projected_gradient_descent(
                model,
                images,
                eps=eps,
                eps_iter=0.01,
                nb_iter=100,
                norm=np.inf,
                clip_min=0.0,
                clip_max=1.0,
                rand_init=True,
                sanity_checks=False,
            )
            for eps in tqdm(epsilons, desc="PGD")
        ],
        [
            momentum_iterative_method(
                model,
                images,
                eps=eps,
                eps_iter=0.01,
                nb_iter=100,
                decay_factor=1.0,
                norm=np.inf,
                clip_min=0.0,
                clip_max=1.0,
                sanity_checks=False,
            )
            for eps in tqdm(epsilons, desc="MIM")
        ],
    ]

    # Plot images (one line per attack)
    fig, axes = plt.subplots(3, len(epsilons), figsize=(12, 6))
    for i, images in enumerate(images_adv):
        for j, image in enumerate(images):
            image = image.detach().cpu()
            axes[i, j].imshow(image[0].permute(1, 2, 0).numpy())
            axes[i, j].axis("off")
            if i == 0:
                axes[i, j].set_title(f"eps={epsilons[j]}")
            if j == 0:
                axes[i, j].set_ylabel(["FGSM", "PGD", "MIM"][i])
    plt.tight_layout()
    plt.savefig("attacks_infty.png")
