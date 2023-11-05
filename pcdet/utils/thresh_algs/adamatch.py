import torch
from torchmetrics import Metric
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from sklearn.mixture import GaussianMixture
palettes = dict(zip(['fp', 'tn', 'tp', 'fn'], sns.color_palette("hls", 4)))
import pandas as pd
import warnings

# TODO: can we delay the teacher/student weight updates until all the reset_state_interval number of samples are obtained?
# TODO: Test the effect of the reset_state_interval on the performance
# TODO: Set the FG of lbl data based on GTs

def _lbl(tensor, mask=None):
    return _lbl(tensor)[_lbl(mask)] if mask is not None else tensor.chunk(2)[0].squeeze(0)

def _ulb(tensor, mask=None):
    return _ulb(tensor)[_ulb(mask)] if mask is not None else tensor.chunk(2)[1].squeeze(0)


class AdaMatch(Metric):
    """
        Adamatch based relative Thresholding
        mean conf. of the top-1 prediction on the weakly aug source data multiplied by a user provided threshold

        Adamatch based Dist. Alignment
        Rectify the target unlabeled pseudo-labels by multiplying them by the ratio of the expected
        value of the weakly aug source labels E[YcapSL;w] to the expected
        value of the target labels E[YcapTU;w], obtaining the final pseudo-labels YtildaTU;w

        REF: UPS FRAMEWORK DA
        probs_x_ulb_w = accumulated_metrics['pl_scores_wa_unlab'].view(-1)
        probs_x_lb_s = accumulated_metrics['pl_scores_wa_lab'].view(-1)
        self.p_model = self.momentum  * self.p_model + (1 - self.momentum) * torch.mean(probs_x_ulb_w)
        self.p_target = self.momentum  * self.p_target + (1 - self.momentum) * torch.mean(probs_x_lb_s)
        probs_x_ulb_aligned = probs_x_ulb_w * (self.p_target + 1e-6) / (self.p_model + 1e-6)
    """
    full_state_update: bool = False

    def __init__(self, **configs):
        super().__init__(**configs)

        self.reset_state_interval = configs.get('RESET_STATE_INTERVAL', 32)
        self.prior_sem_fg_thresh = configs.get('SEM_FG_THRESH', 0.33)
        self.enable_plots = configs.get('ENABLE_PLOTS', False)
        self.fixed_thresh = configs.get('FIXED_THRESH', 0.9)
        self.momentum = configs.get('MOMENTUM', 0.9)
        self.temperature = configs.get('TEMPERATURE', 1.0)
        self.ulb_ratio = configs.get('ULB_RATIO', 0.5)
        self.states_name = ['sem_scores_wa', 'sem_scores_sa', 'conf_scores_wa', 'conf_scores_sa']
        self.class_names = ['Car', 'Pedestrian', 'Cyclist']
        self.iteration_count = 0

        # States are of shape (N, M, P) where N is # samples, M is # RoIs and P = 4 is the Car, Ped, Cyc, FG scores
        for name in self.states_name:
            self.add_state(name, default=[], dist_reduce_fx='cat')

        # mean_p_model aka p_target (_lbl(mean_p_model)) and p_model (_ulb(mean_p_model))
        self.mean_p_model = {s_name: None for s_name in self.states_name}
        self.var_p_model = {s_name: None for s_name in self.states_name}
        self.mean_p_max_model = {s_name: None for s_name in self.states_name}
        self.var_p_max_model = {s_name: None for s_name in self.states_name}
        self.labels_hist = {s_name: None for s_name in self.states_name}
        self.ratio = {'AdaMatch': None, 'SoftMatch': None}

        # Two fixed targets dists
        self.mean_p_model['uniform'] = (torch.ones(len(self.class_names)) / len(self.class_names)).cuda()
        self.mean_p_model['gt'] = torch.tensor([0.85, 0.1, 0.05]).cuda()

        # GMM
        self.gmm_policy=configs.get('GMM_POLICY', 'high')
        self.mu1=configs.get('MU1', 0.1)
        self.mu2=configs.get('MU2', 0.9)
        self.gmm = GaussianMixture(
            n_components=2,
            weights_init=[0.5, 0.5],
            means_init=[[self.mu1], [self.mu2]],
            precisions_init=[[[1.0]], [[1.0]]],
            init_params='k-means++',
            tol=1e-9,
            max_iter=1000
        )

    def update(self, **kwargs):
        for state_name in self.states_name:
            value = kwargs.get(state_name)
            if value is not None:
                getattr(self, state_name).append(value)

    def _accumulate_metrics(self):
        bs = len(self.sem_scores_wa[0])  # TODO: Refactor
        accumulated_metrics = {}
        for mname in self.states_name:
            mstate = getattr(self, mname)
            if isinstance(mstate, list):
                mstate = torch.cat(mstate, dim=0)
            splits = torch.split(mstate, int(self.ulb_ratio * bs), dim=0)
            lbl = torch.cat(splits[::2], dim=0)
            ulb = torch.cat(splits[1::2], dim=0)
            mstate = torch.cat([lbl, ulb], dim=0)
            mstate = mstate.view(-1, mstate.shape[-1])
            accumulated_metrics[mname] = mstate

        return accumulated_metrics

    def _get_mean_var_p_max_model_and_label_hist(self, max_scores, labels, fg_mask, hist_minlength=3, split=None, type='micro'):
        if split is None:
            split = ['lbl', 'ulb']
        if isinstance(split, str) and split in ['lbl', 'ulb']:
            _split = _lbl if split == 'lbl' else _ulb
            fg_max_scores = _split(max_scores, fg_mask)
            fg_labels = _split(labels, fg_mask)
            p_max_model = fg_labels.new_zeros(3, dtype=fg_max_scores.dtype).scatter_add_(0, fg_labels, fg_max_scores)
            fg_labels_hist = torch.bincount(fg_labels, minlength=hist_minlength)

            if type == 'micro':
                # overall mean and variance weighted by the label frequency
                mu_p_max_model = p_max_model.sum() / fg_labels_hist.sum()
                # unbiased_variance = Σ((x - μ)²) / (N - 1)
                var_p_max_model = ((fg_max_scores - mu_p_max_model) ** 2).sum() / (fg_labels_hist.sum() - 1)
            elif type == 'macro':
                norm_p_max_model = p_max_model/(fg_labels_hist + 1e-6)
                mu_p_max_model = norm_p_max_model.mean()
                var_p_max_model = norm_p_max_model.var(unbiased=True)
            return mu_p_max_model, var_p_max_model, fg_labels_hist
        elif isinstance(split, list) and len(split) == 2:
            mu_p_s0, var_p_s0, h_s0 = self._get_mean_var_p_max_model_and_label_hist(max_scores, labels, fg_mask, hist_minlength=hist_minlength, split=split[0], type=type)
            mu_p_s1, var_p_s1, h_s1 = self._get_mean_var_p_max_model_and_label_hist(max_scores, labels, fg_mask, hist_minlength=hist_minlength, split=split[1], type=type)
            return torch.vstack([mu_p_s0, mu_p_s1]).squeeze(), torch.vstack([var_p_s0, var_p_s1]), torch.vstack([h_s0, h_s1])
        else:
            raise ValueError(f"Invalid split type: {split}")

    def compute(self):
        results = {}

        if len(self.sem_scores_wa) < self.reset_state_interval:
            return

        self.iteration_count += 1
        accumulated_metrics = self._accumulate_metrics()
        for sname in ['sem_scores_wa', 'sem_scores_sa']:
            sem_scores = accumulated_metrics[sname]
            conf_scores = accumulated_metrics[sname.replace('sem', 'conf')]

            max_scores, labels = torch.max(sem_scores, dim=-1)
            fg_thresh = torch.tensor(self.prior_sem_fg_thresh, dtype=torch.float, device=sem_scores.device).unsqueeze(0)
            fg_thresh = fg_thresh.expand_as(sem_scores).gather(dim=-1, index=labels.unsqueeze(-1)).squeeze()
            fg_mask =  conf_scores.squeeze() > fg_thresh   # TODO: Make it dynamic. Also not the same for both labeled and unlabeled data
            hist_minlength = sem_scores.shape[-1]
            mean_p_max_model, var_p_max_model, labels_hist = self._get_mean_var_p_max_model_and_label_hist(max_scores, labels, fg_mask, hist_minlength)
            fg_scores_p_model_lbl = _lbl(sem_scores, fg_mask)
            mean_p_model_lbl, var_p_model_lbl = fg_scores_p_model_lbl.mean(dim=0), fg_scores_p_model_lbl.var(dim=0, unbiased=True)
            fg_scores_p_model_ulb = _ulb(sem_scores, fg_mask)
            mean_p_model_ulb, var_p_model_ulb = fg_scores_p_model_ulb.mean(dim=0), fg_scores_p_model_ulb.var(dim=0, unbiased=True)
            mean_p_model = torch.vstack([mean_p_model_lbl, mean_p_model_ulb])
            var_p_model = torch.vstack([var_p_model_lbl, var_p_model_ulb])

            self._update_ema('mean_p_max_model', mean_p_max_model, sname)
            self._update_ema('var_p_max_model', var_p_max_model, sname)
            self._update_ema('labels_hist', labels_hist, sname)
            self._update_ema('mean_p_model', mean_p_model, sname)
            self._update_ema('var_p_model', var_p_model, sname)

            self.log_results(results, sname=sname)
            if self.enable_plots:
                fig = self.draw_dist_plots(max_scores, labels, fg_mask, sname)
                results[f'dist_plots_{sname}'] = fig
                plt.close()

        ratio =  _lbl(self.mean_p_model['sem_scores_wa']) / (_ulb(self.mean_p_model['sem_scores_wa']) + 1e-6)
        self._update_ema('ratio', ratio, 'AdaMatch')
        results['ratio/lab_by_ulb_wa'] = self._arr2dict(self.ratio['AdaMatch'])
        ratio =  self.mean_p_model['uniform'] / (_ulb(self.mean_p_model['sem_scores_wa']) + 1e-6)
        self._update_ema('ratio', ratio, 'SoftMatch')
        results['ratio/uniform_by_ulb_wa'] = self._arr2dict(self.ratio['SoftMatch'])

        self.reset()
        return results

    def log_results(self, results, sname):
        results[f'mean_p_max_model_lbl/{sname}'] = _lbl(self.mean_p_max_model[sname])
        results[f'mean_p_max_model_ulb/{sname}'] = _ulb(self.mean_p_max_model[sname])
        results[f'var_p_max_model_lbl/{sname}'] = _lbl(self.var_p_max_model[sname])
        results[f'var_p_max_model_ulb/{sname}'] = _ulb(self.var_p_max_model[sname])
        results[f'labels_hist_lbl/{sname}'] = self._arr2dict(_lbl(self.labels_hist[sname]))
        results[f'labels_hist_ulb/{sname}'] = self._arr2dict(_ulb(self.labels_hist[sname]))
        # Bincount/histogram approach (labels_probs_lbl) is the sharpened
        # or one-hot version of the mean approach (mean_p_model_lbl)
        labels_probs = self.labels_hist[sname] / self.labels_hist[sname].sum(dim=-1, keepdim=True)
        unbiased_p_model = self.mean_p_model[sname] / self.labels_hist[sname]
        unbiased_p_model = unbiased_p_model / unbiased_p_model.sum(dim=-1, keepdim=True)
        results[f'unbiased_p_model_lbl/{sname}'] = self._arr2dict(_lbl(unbiased_p_model))
        results[f'unbiased_p_model_ulb/{sname}'] = self._arr2dict(_ulb(unbiased_p_model))
        results[f'labels_probs_lbl_or_sharpened_mean_p_model_lbl)/{sname}'] = self._arr2dict(_lbl(labels_probs))
        results[f'labels_probs_ulb_or_sharpened_mean_p_model_ulb)/{sname}'] = self._arr2dict(_ulb(labels_probs))
        results[f'mean_p_model_lbl/{sname}'] = self._arr2dict(_lbl(self.mean_p_model[sname]))
        results[f'mean_p_model_ulb/{sname}'] = self._arr2dict(_ulb(self.mean_p_model[sname]))
        results[f'var_p_model_lbl/{sname}'] = self._arr2dict(_lbl(self.var_p_model[sname]))
        results[f'var_p_model_ulb/{sname}'] = self._arr2dict(_ulb(self.var_p_model[sname]))
        results['threshold/AdaMatch'] = self._get_threshold(thresh_alg='AdaMatch')
        results['threshold/SoftMatch'] = self._get_threshold(thresh_alg='SoftMatch')
        results['threshold/FreeMatch'] = self._arr2dict(self._get_threshold(thresh_alg='FreeMatch'))

    def draw_dist_plots(self, max_scores, labels, fg_mask, tag, meta_info=''):

        max_scores_lbl = _lbl(max_scores, fg_mask)
        labels_lbl = _lbl(labels, fg_mask)

        max_scores_ulb = _ulb(max_scores, fg_mask)
        labels_ulb = _ulb(labels, fg_mask)

        BS = len(self.sem_scores_wa[0])
        WS = self.reset_state_interval * BS
        info = (f"Iter: {self.iteration_count}    Interval: {self.reset_state_interval}    " +
                f"BS: {BS}    W: {(self.iteration_count - 1) * WS} - {self.iteration_count * WS}    M: {meta_info}")
        fig, axes = plt.subplots(1, 2, figsize=(12, 6), sharex='col', sharey='row', layout="compressed")
        plt.suptitle(info, fontsize='small')
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            fg_scores_labels_lbl_df = pd.DataFrame(
                torch.cat([max_scores_lbl.view(-1, 1), labels_lbl.view(-1, 1)], dim=-1).cpu().numpy(),
                columns=['scores', 'labels'])
            fg_scores_labels_ulb_df = pd.DataFrame(
                torch.cat([max_scores_ulb.view(-1, 1), labels_ulb.view(-1, 1)], dim=-1).cpu().numpy(),
                columns=['scores', 'labels'])
            sns.histplot(data=fg_scores_labels_lbl_df, ax=axes[0], x='scores', hue='labels', kde=True).set(
                title=f"Dist of FG max-scores on WA LBL input {tag} ")
            sns.histplot(data=fg_scores_labels_ulb_df, ax=axes[1], x='scores', hue='labels', kde=True).set(
                title=f"Dist of FG max-scores on WA ULB input {tag}")
            plt.tight_layout()
        # plt.show()

        return fig.get_figure()

    def rectify_sem_scores(self, sem_scores_ulb, tag='AdaMatch'):

        if self.iteration_count == 0:
            print("Skipping rectification as iteration count is 0")
            return

        max_scores, labels = torch.max(sem_scores_ulb, dim=-1)
        fg_thresh = torch.tensor(self.prior_sem_fg_thresh, device=labels.device).unsqueeze(0).repeat(
            max_scores.shape[0], max_scores.shape[1], 1).gather(dim=2, index=labels.unsqueeze(-1)).squeeze()

        fg_mask = max_scores > fg_thresh

        rect_scores = sem_scores_ulb * self.ratio[tag]
        rect_scores /= rect_scores.sum(dim=-1, keepdims=True)
        sem_scores_ulb[fg_mask] = rect_scores[fg_mask]  # Only rectify FG rois

        return sem_scores_ulb

    def _update_ema(self, p_name, probs, tag):
        prob = getattr(self, p_name)
        prob[tag] = probs if prob[tag] is None else self.momentum * prob[tag] + (1 - self.momentum) * probs

    def get_mask(self, scores, thresh_alg='AdaMatch', DA=False):
        if thresh_alg == 'AdaMatch':
            if DA:
                scores = self.rectify_sem_scores(scores, tag=thresh_alg)
            max_scores, labels = torch.max(scores, dim=-1)
            fg_thresh = torch.tensor(self.prior_sem_fg_thresh, device=labels.device).unsqueeze(0).repeat(
            max_scores.shape[0], max_scores.shape[1], 1).gather(dim=2, index=labels.unsqueeze(-1)).squeeze()

            fg_mask = max_scores > fg_thresh
            thresh_mask = max_scores > self._get_threshold(tag='sem_scores_wa', thresh_alg=thresh_alg)
            return thresh_mask, fg_mask, None

        elif thresh_alg == 'FreeMatch':
            max_scores, labels = torch.max(scores, dim=-1)
            fg_thresh = torch.tensor(self.prior_sem_fg_thresh, device=labels.device).unsqueeze(0).repeat(
            max_scores.shape[0], max_scores.shape[1], 1).gather(dim=2, index=labels.unsqueeze(-1)).squeeze()
            fg_mask = max_scores > fg_thresh
            thresh = self._get_threshold(tag='sem_scores_wa', thresh_alg=thresh_alg)
            multi_thresh = torch.zeros_like(scores)
            multi_thresh[:, :] = thresh
            multi_thresh = multi_thresh.gather(dim=2, index=labels.unsqueeze(-1)).squeeze()
            return (max_scores > multi_thresh), fg_mask, None
        elif thresh_alg == 'SoftMatch':
            # max_scores, labels = torch.max(scores, dim=-1) # for ploting use labels from here
            # fg_thresh = torch.tensor(self.prior_sem_fg_thresh, device=labels.device).unsqueeze(0).repeat(
            # max_scores.shape[0], max_scores.shape[1], 1).gather(dim=2, index=labels.unsqueeze(-1)).squeeze()
            # fg_mask_raw = max_scores > fg_thresh
            if DA:
                scores = self.rectify_sem_scores(scores, tag=thresh_alg)
            max_scores, labels = torch.max(scores, dim=-1)
            fg_thresh = torch.tensor(self.prior_sem_fg_thresh, device=labels.device).unsqueeze(0).repeat(
            max_scores.shape[0], max_scores.shape[1], 1).gather(dim=2, index=labels.unsqueeze(-1)).squeeze()
            fg_mask = max_scores > fg_thresh
            mu = _ulb(self.mean_p_max_model['sem_scores_wa'])
            var = _ulb(self.var_p_max_model['sem_scores_wa'])
            n_sigma = 2
            lambda_p = torch.exp(-((torch.clamp(max_scores - mu, max=0.0) ** 2) / (2 * var / (n_sigma ** 2))))

            # fig, axs = plt.subplots(2, 2, figsize=(8, 8), layout="compressed")
            # axs = axs.flatten()
            # # Plot the histograms
            # axs[0].hist(max_scores[fg_mask_raw].view(-1).cpu().numpy(), bins=20, alpha=0.8, edgecolor='black', color='r', label='ulb-fg')
            # axs[1].hist(max_rect_scores[fg_mask_rect].view(-1).cpu().numpy(), bins=20, alpha=0.8, edgecolor='black', color='b', label='UA-ulb-fg')
            # axs[2].hist(max_rect_scores.view(-1).cpu().numpy(), bins=20, alpha=0.8, edgecolor='black', color='b', label='max-UA')
            # axs[3].hist(lambda_p.view(-1).cpu().numpy(), bins=20, alpha=0.8, edgecolor='black', color='c', label='lambda-p')
            # axs[3].axvline(mu.cpu().numpy())
            
            # # Add titles, labels, and legends
            # for ax in axs:
            #     ax.set_xlabel('score', fontsize='x-small')
            #     ax.set_ylabel('count', fontsize='x-small')
            #     # ax.set_ylim(0, 100)
            #     # ax.set_xlim(0, 1)

            # axs[0].set_title('fg-ulb', fontsize='small')
            # axs[1].set_title('UA(fg-ulb)', fontsize='small')
            # axs[2].set_title('max UA(p)', fontsize='small')
            # axs[3].set_title('labmda-p', fontsize='small')

            # plt.suptitle('softmatch', fontsize='small')
            # plt.show()
            # rectify_sem_scores_plots = fig.get_figure()
            # plt.close()

            return (max_scores > mu), fg_mask, lambda_p

    def _get_threshold(self, sem_scores_wa_lbl=None, tag='sem_scores_wa', thresh_alg='AdaMatch'):
        if thresh_alg == 'AdaMatch':
            if sem_scores_wa_lbl is None:
                return _lbl(self.mean_p_max_model[tag]) * self.fixed_thresh
            max_scores, labels = torch.max(sem_scores_wa_lbl, dim=-1)
            fg_mask = max_scores > self.prior_sem_fg_thresh
            fg_max_scores_lbl = max_scores[fg_mask]
            return fg_max_scores_lbl.mean() * self.fixed_thresh

        elif thresh_alg == 'FreeMatch':
            normalized_p_model = torch.div(_ulb(self.mean_p_model[tag]), _ulb(self.mean_p_model[tag]).max())
            return normalized_p_model * _ulb(self.mean_p_max_model[tag])
        elif thresh_alg == 'SoftMatch':
            return _ulb(self.mean_p_max_model[tag])

    def _arr2dict(self, array):
        if array.shape[-1] == 2:
            return {cls: array[cind] for cind, cls in enumerate(['Bg', 'Fg'])}
        elif array.shape[-1] == len(self.class_names):
            return {cls: array[cind] for cind, cls in enumerate(self.class_names)}
        else:
            raise ValueError(f"Invalid array shape: {array.shape}")