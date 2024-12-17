# Coded by: Franck

import argparse
import subprocess


def run_script_with_args(
    script_path, model_names, attack_names, data_name, path_data, epsilons, save=False
):
    """
    Runs a Python script with varying -M and -A arguments and additional arguments.

    Parameters:
        script_path (str): Path to the Python script to execute.
        model_names (list): List of models attacked for the detection.
        attack_names (list): List of attacks used for the detection.
        **kwargs: Additional arguments to pass to the script as key-value pairs.
    """
    for m in model_names:
        for a in attack_names:
            command = [
                "python",
                script_path,
                "-M",
                str(m),
                "-A",
                str(a),
                "--path_data",
                str(path_data),
                "--data_name",
                str(data_name),
                "--epsilons",
            ]
            command.extend(str(epsilon) for epsilon in epsilons)
            if save:
                command.append("--save")
            try:
                print(f"Running: {' '.join(command)}")
                result = subprocess.run(
                    command, capture_output=True, text=True, check=True
                )
                print(f"Output:\n{result.stdout}")
            except subprocess.CalledProcessError as e:
                print(f"Error while running {' '.join(command)}:\n{e.stderr}")


# Example usage
if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Run a Python script with varying arguments."
    )

    # Add arguments for the script
    parser.add_argument(
        "script_to_run",
        type=str,
        default="./detect_attacks_logp.py",
        help="Path to the script to run.",
    )
    parser.add_argument(
        "-model_names",
        "-M",
        type=str,
        nargs="+",
        default=["A", "B", "C", "D", "E", "F", "G"],
        help="Name of the models to run the detection on.",
    )
    parser.add_argument(
        "-attack_names",
        "-A",
        type=str,
        nargs="+",
        default=["FGSM", "PGD", "MIM"],
        help="Name of the attacks to run the detection on.",
    )
    parser.add_argument(
        "--data_name", type=str, default="mnist", help="Name of the dataset."
    )
    parser.add_argument(
        "--path_data", type=str, default="./mnist_results", help="Path to the dataset."
    )
    parser.add_argument(
        "--epsilons",
        type=float,
        nargs="+",
        default=[0.1, 0.2, 0.3, 0.4, 0.5],
        help="List of epsilon values for the attacks",
    )
    parser.add_argument(
        "--save",
        action="store_true",
        default=False,
        help="Whether to save the results to a file.",
    )
    args = parser.parse_args()
    run_script_with_args(
        args.script_to_run,
        model_names=args.model_names,
        attack_names=args.attack_names,
        data_name=args.data_name,
        path_data=args.path_data,
        epsilons=args.epsilons,
        save=args.save,
    )
