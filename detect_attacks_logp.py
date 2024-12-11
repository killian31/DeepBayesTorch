import argparse
from tqdm import tqdm
import os, pickle, sys

import torch
import torch.nn as nn
import torch.nn.functional as F
import math

sys.path.append('../')
sys.path.append('../utils/')
from utils import utils
from alg.vae_new import bayes_classifier

def comp_lowerbound_func(vae_type):
    if vae_type == "A":
        from alg.lowerbound_functions import lowerbound_A as lowerbound_func
    elif vae_type == "B":
        from alg.lowerbound_functions import lowerbound_B as lowerbound_func
    elif vae_type == "C":
        from alg.lowerbound_functions import lowerbound_C as lowerbound_func
    elif vae_type == "D":
        from alg.lowerbound_functions import lowerbound_D as lowerbound_func
    elif vae_type == "E":
        from alg.lowerbound_functions import lowerbound_E as lowerbound_func
    elif vae_type == "F":
        from alg.lowerbound_functions import lowerbound_F as lowerbound_func
    elif vae_type == "G":
        from alg.lowerbound_functions import lowerbound_G as lowerbound_func
    else:
        raise ValueError(f"Unknown VAE type: {vae_type}")
    return lowerbound_func

def create_decoder(vae_type, generator):
    if vae_type == "A":
        dec = (generator.pyz_params, generator.pxzy_params)
    elif vae_type == "B":
        dec = (generator.pzy_params, generator.pxzy_params)
    elif vae_type == "C":
        dec = (generator.pyzx_params, generator.pxz_params)
    elif vae_type == "D":
        dec = (generator.pyzx_params, generator.pzx_params)
    elif vae_type == "E":
        dec = (generator.pyz_params, generator.pzx_params)
    elif vae_type == "F":
        dec = (generator.pyz_params, generator.pxz_params)
    elif vae_type == "G":
        dec = (generator.pzy_params, generator.pxz_params)
    else:
        raise ValueError(f"Unknown VAE type: {vae_type}")
    return dec

def comp_logp(logit, y, text, comp_logit_dist=False):
    """
    Compute log probabilities and related statistics.

    Args:
        logit (ndarray): logits of shape (N, nb_classes).
        y (ndarray): one-hot labels of shape (N, nb_classes).
        text (str): Tag for printing results.
        comp_logit_dist (bool): Whether to compute distribution of logits.

    Returns:
        List of computed statistics.
    """
    # logsumexp over classes
    logpx = torch.logsumexp(logit, axis=0)
    logpx_mean = torch.mean(logpx)
    logpx_std = torch.sqrt(torch.var(logpx))
    
    logpxy = torch.sum(y * logit, axis=1)
    logpxy_mean = []
    logpxy_std = []

    nb_classes = y.shape[1]
    for i in range(nb_classes):
        ind = torch.where(y[:, i] == 1)[0]
        if len(ind) > 0:
            logpxy_mean.append(torch.mean(logpxy[ind]))
            logpxy_std.append(torch.sqrt(torch.var(logpxy[ind])))
        else:
            # If no samples for this class, just append NaNs
            logpxy_mean.append(float('nan'))
            logpxy_std.append(float('nan'))

    print('%s: logp(x) = %.3f +- %.3f, logp(x|y) = %.3f +- %.3f' %
          (text, logpx_mean, logpx_std, torch.nanmean(torch.Tensor(logpxy_mean)), torch.nanmean(torch.Tensor(logpxy_std))))

    results = [logpx, logpx_mean, logpx_std, logpxy, logpxy_mean, logpxy_std]

    if comp_logit_dist:
        # Compute distribution of logits
        nb_classes = y.shape[1]
        logit_mean = []
        logit_std = []
        logit_kl_mean = []
        logit_kl_std = []
        softmax_mean_list = []
        # softmax of mean distribution
        for i in range(nb_classes):
            ind = torch.where(y[:, i] == 1)[0]
            if len(ind) > 0:
                logit_class = logit[ind]
                logit_mean.append(torch.mean(logit_class, axis=0))
                logit_std.append(torch.sqrt(torch.var(logit_class, axis=0)))

                # Compute softmax and KL divergence
                logit_tmp = logit_class - torch.logsumexp(logit_class, axis=1)[:, torch.newaxis]
                softmax_mean = torch.mean(torch.exp(logit_tmp), axis=0)
                softmax_mean_list.append(softmax_mean)

                # KL divergence from softmax_mean distribution to each sample's distribution
                # KL(Pmean || Pi) = sum(Pmean * (log(Pmean) - log(Pi)))
                # where Pi is from each sample logit_tmp
                # Actually, we want the mean KL over samples:
                # logit_tmp are log probabilities of each sample.
                # We can approximate KL by using the mean softmax distribution and comparing to each sample.
                # However, in original code, it seems a different approach was taken.
                # We'll follow the original logic closely.
                # The original code snippet seems incorrect in computing KL directly for each sample distribution.
                # Instead, it computed KL based on softmax_mean. We'll replicate that logic:
                # logit_kl = sum(Pmean * (log(Pmean)-log(pi))) over i, averaged over samples.
                # Actually, in original code, it seems to incorrectly apply logit_kl on each sample. 
                # We'll just skip this detail and keep the original approach.
                
                # We'll compute the KL divergence per sample and then average:
                # Pi = exp(logit_tmp)
                # KL(Pmean || Pi) = sum(Pmean * (log(Pmean) - log(Pi)))
                # But Pi differs per sample. We'll compute it for each sample:
                pi = torch.exp(logit_tmp)
                # For each sample:
                # kl_i = sum(Pmean * (log(Pmean)-log(pi_i)))
                kl_vals = []
                for sample_idx in range(pi.shape[0]):
                    kl_val = torch.sum(softmax_mean * (torch.log(softmax_mean) - logit_tmp[sample_idx]))
                    kl_vals.append(kl_val)
                kl_vals = torch.array(kl_vals)
                logit_kl_mean.append(torch.mean(kl_vals))
                logit_kl_std.append(torch.sqrt(torch.var(kl_vals)))
            else:
                # If no samples for that class
                logit_mean.append(torch.full((nb_classes,), torch.nan))
                logit_std.append(torch.full((nb_classes,), torch.nan))
                logit_kl_mean.append(torch.nan)
                logit_kl_std.append(torch.nan)
                softmax_mean_list.append(torch.full((nb_classes,), torch.nan))

        results.extend([logit_mean, logit_std, logit_kl_mean, logit_kl_std, softmax_mean_list])

    return results

def comp_detect(x, x_mean, x_std, alpha, plus):
    """
    Compute detection rate given a criterion:
    If plus=True: detect if x > x_mean + alpha * x_std
    else: detect if x < x_mean - alpha * x_std
    """
    if plus:
        detect_rate = torch.mean((x > x_mean + alpha * x_std).float())
    else:
        detect_rate = torch.mean((x < x_mean - alpha * x_std).float())
    return detect_rate * 100

def search_alpha(x, x_mean, x_std, target_rate=5.0, plus=False):
    """
    Binary search for alpha such that detection rate is close to target_rate.
    """
    alpha_min = 0.0
    alpha_max = 3.0
    alpha_now = 1.5
    detect_rate = comp_detect(x, x_mean, x_std, alpha_now, plus)
    T = 0
    while torch.abs(detect_rate - target_rate) > 0.01 and T < 20:
        if detect_rate > target_rate:
            alpha_min = alpha_now
        else:
            alpha_max = alpha_now
        alpha_now = 0.5 * (alpha_min + alpha_max)
        detect_rate = comp_detect(x, x_mean, x_std, alpha_now, plus)
        T += 1
    return alpha_now, detect_rate

def test_attacks(
    attack_method: str,
    epsilons: list,
    guard_name=str,
    path_data="./mnist_results",
    save=False, 
    victim_name='mlp',
    data_name='mnist',
    targeted=False
):
    """
    Evaluate detection metrics on clean and adversarial examples.

    Args:
        model (nn.Module): PyTorch model for evaluation.
        x_train, y_train: Training samples (for baseline stats).
        x_clean, y_clean: Clean test samples and labels.
        x_adv, y_adv: Adversarial samples and corresponding predicted labels from victim model.
        nb_classes (int): Number of classes.
        save (bool): Whether to save results to disk.
        guard_name (str): Identifier for the "guard" model.
        victim_name (str): Identifier for the victim model.
        data_name (str): Dataset name.
        targeted (bool): Whether the attack is targeted.

    Returns:
        results (dict): Dictionary of detection statistics.
    """
    success_rate_list = []
    l2_diff_mean_list = []
    l2_diff_std_list = []
    l0_diff_mean_list = []
    l0_diff_std_list = []
    li_diff_mean_list = []
    li_diff_std_list = []
    fp_logpx_list = []
    tp_logpx_list = []
    FP_logpxy_list = []
    TP_logpxy_list = []
    FP_kl_list = []
    TP_kl_list = []

    dimY = 10 if data_name != "gtsrb" else 43
    for epsilon in epsilons:
        print(f"Running detection for epsilon={epsilon}")
        # Craft adversarial examples
        filename_data = f"{guard_name}_{attack_method}_{data_name}_epsilon_{epsilon}.pkl"
        file_path = os.path.join(path_data, filename_data)
        if os.path.exists(file_path):
            with open(file_path, 'rb') as f:
                data= pickle.load(f)
                x_clean = torch.Tensor(data['x_clean'][0])
                y_clean = torch.Tensor(data['y_clean'][0])
                x_adv = torch.Tensor(data['x_adv'][0])
                y_adv = torch.Tensor(data['y_adv'][0])
                y_logit_clean = torch.Tensor(data['y_clean_logits'][0])
                y_logit_adv = torch.Tensor(data['y_adv_logits'][0])
        else:
            raise ValueError(f"File {file_path} not found, probably the attack has not be performed for epsilon={epsilon}.")

        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        # Identify successful attacks (where victim's adv prediction != clean label)
        correct_prediction = (torch.argmax(y_adv, dim=1) == torch.argmax(y_clean, dim=1))
        success_rate = 100.0 * (1 - torch.mean(correct_prediction.to(torch.float32)))
        ind_success = torch.where(correct_prediction == 0)[0]

        if len(ind_success) == 0:
            print('No successful attacks found.')
            return {}

        # Compute L2, L0, L_inf perturbations for successful attacks
        diff = x_adv[ind_success] - x_clean[ind_success]
        l2_diff = torch.sqrt(torch.sum(diff**2, dim=(1, 2, 3)))
        li_diff = torch.amax(torch.abs(diff), dim=(1, 2, 3))
        l0_diff = torch.sum(diff != 0, dim=(1, 2, 3))
        print('perturb for successful attack: L_2 = %.3f +- %.3f' % (torch.mean(l2_diff), torch.sqrt(torch.var(l2_diff))))
        print('perturb for successful attack: L_inf = %.3f +- %.3f' % (torch.mean(li_diff), torch.sqrt(torch.var(li_diff))))
        print('perturb for successful attack: L_0 = %.3f +- %.3f' % (torch.mean(l0_diff.to(torch.float32)), torch.sqrt(torch.var(l0_diff.to(torch.float32)))))

        # y_adv and y_clean are not one-hot labels, we need to convert them
        y_clean_one_hot = F.one_hot(torch.argmax(y_clean, dim=1), num_classes=dimY).to(torch.float32)
        y_adv_one_hot = F.one_hot(torch.argmax(y_adv, dim=1), num_classes=dimY).to(torch.float32)
        
        # Compute log probabilities
        # Also get training logits for baseline stats
        # y_logit_train = get_logits(model, x_train)
        # results_train = comp_logp(y_logit_train, y_train, 'train', comp_logit_dist=True)

        results_clean = comp_logp(y_logit_clean, y_clean_one_hot,'clean')
        results_adv = comp_logp(y_logit_adv[ind_success], y_adv_one_hot[ind_success], 'adv (wrong)')

        # Detection based on logp(x)
        # If guard_name in ['mlp', 'cnn'], plus=True else False (following original logic)
        plus = True if guard_name in ['mlp', 'cnn'] else False
        alpha, detect_rate = search_alpha(results_clean[0], results_clean[1], results_clean[2], plus=plus)
        fp_logpx = comp_detect(results_clean[0], results_clean[1], results_clean[2], alpha, plus=plus)
        tp_logpx = comp_detect(results_adv[0], results_clean[1], results_clean[2], alpha, plus=plus)
        print('false alarm rate (logp(x)):', fp_logpx)
        print('detection rate (logp(x)):', tp_logpx)

        # Detection based on logp(x|y)
        fp_rate = []
        tp_rate_vals = []
        for i in range(dimY):
            ind = torch.where(y_clean_one_hot[:, i] == 1)[0]
            if len(ind) == 0:
                continue
            alpha_c, _ = search_alpha(results_clean[3][ind], results_clean[4][i], results_clean[5][i], plus=plus)
            fp_c = comp_detect(results_clean[3][ind], results_clean[4][i], results_clean[5][i], alpha_c, plus=plus)
            fp_rate.append(fp_c)

            adv_ind = torch.where(y_adv_one_hot[ind_success][:, i] == 1)[0]
            if len(adv_ind) == 0:
                continue
            tp_c = comp_detect(results_adv[3][adv_ind], results_clean[4][i], results_clean[5][i], alpha_c, plus=plus)
            tp_rate_vals.append(tp_c)
        if len(tp_rate_vals) > 0:
            FP_logpxy = torch.mean(fp_rate)
            TP_logpxy = torch.mean(tp_rate_vals)
        else:
            FP_logpxy = torch.nan
            TP_logpxy = torch.nan
        print('false alarm rate (logp(x|y)):', FP_logpxy)
        print('detection rate (logp(x|y)):', TP_logpxy)

        # KL-based detection
        # Extract the logit distribution stats from training
        # last 5 results from results_train are [logit_mean, logit_std, logit_kl_mean, logit_kl_std, softmax_mean]
        logit_mean, logit_std, kl_mean, kl_std, softmax_mean_list = results_clean[-5:]

        # We need to compute KL on train and adv again per class
        fp_rate_kl = []
        tp_rate_kl = []
        
        for i in range(dimY):
            # compute KL for the training samples of class i
            ind = torch.where(y_clean_one_hot[:, i] == 1)[0]
            if len(ind) == 0 or torch.isnan(kl_mean[i]):
                continue
            # compute KL wrt. softmax_mean_list[i]
            logit_clean_i = y_clean[ind]
            logit_tmp = logit_clean_i - torch.logsumexp(logit_clean_i, axis=1, keepdim=True)
            pi = torch.exp(logit_tmp)
            pmean = softmax_mean_list[i]
            kl_values_train = []
            for j in range(pi.shape[0]):
                kl_val = torch.sum(pmean * (torch.log(pmean) - logit_tmp[j]))
                kl_values_train.append(kl_val)
            kl_values_train = torch.Tensor(kl_values_train)
            alpha_c, _ = search_alpha(kl_values_train, kl_mean[i], kl_std[i], plus=True)
            fp_c = comp_detect(kl_values_train, kl_mean[i], kl_std[i], alpha_c, plus=True)
            fp_rate_kl.append(fp_c)

            # adv
            adv_ind = torch.where(y_adv_one_hot[ind_success][:, i] == 1)[0]
            if len(adv_ind) == 0:
                continue
            logit_adv_i = y_logit_adv[ind_success][adv_ind]
            logit_tmp_adv = logit_adv_i - torch.logsumexp(logit_adv_i, axis=0)[:, torch.newaxis]
            pi_adv = torch.exp(logit_tmp_adv)
            kl_values_adv = []
            for j in range(pi_adv.shape[0]):
                kl_val = torch.sum(pmean * (torch.log(pmean) - logit_tmp_adv[j]))
                kl_values_adv.append(kl_val)
            kl_values_adv = torch.array(kl_values_adv)
            tp_c = comp_detect(kl_values_adv, kl_mean[i], kl_std[i], alpha_c, plus=True)
            tp_rate_kl.append(tp_c)
        if len(tp_rate_kl) > 0:
            FP_kl = torch.mean(fp_rate_kl)
            TP_kl = torch.mean(tp_rate_kl)
        else:
            FP_kl = torch.nan
            TP_kl = torch.nan
        print('false alarm rate (KL):', FP_kl)
        print('detection rate (KL):', TP_kl)

        success_rate_list.append(success_rate)
        l2_diff_mean_list.append(torch.mean(l2_diff))
        l2_diff_std_list.append(torch.sqrt(torch.var(l2_diff)))
        l0_diff_mean_list.append(torch.mean(l0_diff.to(torch.float32)))
        l0_diff_std_list.append(torch.sqrt(torch.var(l0_diff.to(torch.float32))))
        li_diff_mean_list.append(torch.mean(li_diff))
        li_diff_std_list.append(torch.sqrt(torch.var(li_diff)))
        fp_logpx_list.append(fp_logpx)
        tp_logpx_list.append(tp_logpx)
        FP_logpxy_list.append(FP_logpxy)
        TP_logpxy_list.append(TP_logpxy)
        FP_kl_list.append(FP_kl)
        TP_kl_list.append(TP_kl)


    results = {
        'success_rate': success_rate_list,
        'l2_diff_mean': l2_diff_mean_list,
        'l2_diff_std': l2_diff_std_list,
        'l0_diff_mean': l0_diff_mean_list,
        'l0_diff_std': l0_diff_std_list,
        'li_diff_mean': li_diff_mean_list,
        'li_diff_std': li_diff_std_list,
        'fp_logpx': fp_logpx_list,
        'tp_logpx': tp_logpx_list,
        'FP_logpxy': FP_logpxy_list,
        'TP_logpxy': TP_logpxy_list,
        'FP_kl': FP_kl_list,
        'TP_kl': TP_kl_list
    }

    # Optionally save results
    if save:
        if not os.path.exists('detection_results'):
            os.mkdir('detection_results')
        path = os.path.join('detection_results', guard_name)
        if not os.path.exists(path):
            os.mkdir(path)
        filename = f"{data_name}_{victim_name}"
        if targeted:
            filename += '_targeted'
        else:
            filename += '_untargeted'
        filename += '.pkl'
        pickle.dump(results, open(os.path.join(path, filename), 'wb'))
        print("results saved at", os.path.join(path, filename))

    return results

if __name__=="__main__":
    parser = argparse.ArgumentParser(description='Evaluate detection metrics on clean and adversarial examples.')

    parser.add_argument('--guard_name',
                        type=str,
                        default='A',
                        help='Identifier for the "guard" model.')
    parser.add_argument('--victim_name',
                        type=str,
                        default='mlp',
                        help='Identifier for the victim model.')
    parser.add_argument('--attack',
                        '-A',
                        type=str,
                        default='FGSM')
    parser.add_argument('--data_name',
                        type=str,
                        default='mnist',
                        help='Dataset name.')
    parser.add_argument('--targeted',
                        action='store_true',
                        default=False,
                        help='Whether the attack is targeted.')
    parser.add_argument('--save',
                        action='store_true',
                        default=False,
                        help='Whether to save results to disk.')
    parser.add_argument('--ll',
                        type=str,
                        default='bernoulli',
                        help='Likelihood type for the VAE model.')
    parser.add_argument(
        "--epsilons",
        type=float,
        nargs="+",
        default=[0.1, 0.2, 0.3, 0.4, 0.5],
        help="List of epsilon values for FGSM attack.",
    )
    parser.add_argument('--batch_size', '-B', type=int, default=100)
    parser.add_argument('--conv', '-C', action='store_true', default=False)

    args = parser.parse_args()
    result = test_attacks(
        attack_method=args.attack,
        epsilons=args.epsilons,
        guard_name=args.guard_name,
        victim_name=args.victim_name,
        data_name=args.data_name,
        targeted=args.targeted,
        save=args.save
    )