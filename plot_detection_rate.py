import matplotlib.pyplot as plt
import numpy as np
import json
from matplotlib import cm


def comp_detect_rate(dict_result, detection_method):
    """
    Compute the detection rate of a specific detection method.

    Parameters
    ----------
    dict_result : dict
        Dictionary containing the detection results.
    detection_method : str
        Name of the detection method.

    Returns
    -------
    list
        Detection rates of the detection method for different epsilons.
    """

    if detection_method == "Marginal":
        return dict_result["tp_logpx"]
    elif detection_method == "Joint":
        return dict_result["TP_logpxy"]
    elif detection_method == "KL":
        return dict_result["TP_kl"]
    else:
        raise ValueError("Invalid detection method.")
    
def plot_detection_rate(data_name, attack, epsilons, data_dir = "./detection_results", save_dir = "./detection_results"):
    """
    Plot the detection rate of one attack on all the VAE for different values of epsilon.
    
    Parameters
    ----------
    data_name : str
        Name of the dataset.
    attack : str
        Name of the attack.
    save_dir : str
        Directory to save the plot.
    """
    
    vae_types = ["A", "B", "C", "D", "E", "F", "G"]
    detection_methods = ["Marginal", "Joint", "KL"]
    detection_rates = {detect_method: {} for detect_method in detection_methods}

    for detect_method in detection_methods:
        for vae_type in vae_types:
            with open(f"{data_dir}/{data_name}_{vae_type}_{attack}_detection_results.json", "r") as f:
                detection_rate = json.load(f)
            
            detection_rates[detect_method][vae_type] = comp_detect_rate(detection_rate, detect_method)

    # Plotting
    # Create subplots for each detection method
    letter_to_title = {
        "A": "GFZ",
        "B": "GFY",
        "C": "GBZ",
        "D": "GBY",
        "E": "DFX",
        "F": "DFZ",
        "G": "DBX",
    }
    num_vae_types = len(vae_types)
    cmap = cm.get_cmap("rainbow", num_vae_types)

    fig, axes = plt.subplots(1, len(detection_methods), figsize=(18, 6))

    for idx, detect_method in enumerate(detection_methods):
        ax = axes[idx]
        for j, vae_type in enumerate(vae_types):
            if vae_type in detection_rates[detect_method]:
                ax.plot(
                    epsilons,
                    detection_rates[detect_method][vae_type],
                    marker="o",
                    label=letter_to_title[vae_type],
                    linewidth=2,
                    color=cmap(j))
        
        ax.set_title(f"{detect_method} Method")
        ax.set_xlabel("Epsilon")
        ax.set_ylabel("Detection Rate") if idx == 0 else None
        ax.legend(loc="upper left")
        ax.grid(True, ls="--")

    plt.tight_layout()
    save_path = f"{save_dir}/{data_name}_{attack}_detection_rates_subplots.png"
    plt.savefig(save_path)
    plt.close()

    print("Subplots saved successfully.")

if __name__ == "__main__":
    data_name = "MNIST"
    attack = "PGD"
    epsilons = [0, 0.1, 0.2, 0.3, 0.4, 0.5]
    plot_detection_rate(data_name, attack, epsilons)