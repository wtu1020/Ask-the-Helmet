import torch
import torch.nn as nn
import torch.nn.functional as F
from copy import deepcopy

PLTAMY_DEFAULT_CONFIG = {
    "lr": 0.001,
    "temperature_scaling": 2.0,
    "memory_mode": "enhanced",

    "temporal_ema_momentum": 0.9,
    "temporal_mix_min": 0.2,
    "temporal_mix_max": 0.8,
    "distance_threshold": 0.6,# 0.6
    "conf_threshold": 0.78,#0.78
    "adaptive_threshold_gamma": 0.2,#0.2
    "min_adaptive_threshold": 0.65,
    "pseudo_conf_gate": True,

    # 微调软化融合率，更稳健
    "pseudo_max_mix": 0.55,

    "proto_temperature": 0.32,

    "lambda_proto_align": 0.65, #0.65
    "dynamic_align_scale": True,

    "memory_size": 100,
    "min_samples_per_class": 5,
    "use_ema_proto": True,

    "temp_smooth_base": 0.75,
    "smooth_uncertainty_bonus": 0.2,
    "feature_guided_smooth": True,
    "enable_temporal_dist_smooth": True,

    "enable_plain_entmin": False,

    "enable_memory": True,
    "enable_protoalign": True,

    "base_bn_momentum": 0.1,
    "bn_gamma": 2.0,

    "enable_cb_replay": True,
    "lambda_replay": 0.7,  # 略微提升重放权重，稳住流形

    "batch_diversity_threshold": 2.5,
    # 彻底移除了 entropy_block_threshold 相关的特设(Ad-hoc)保护机制
}


def get_pltamy_default_config(overrides=None):
    cfg = dict(PLTAMY_DEFAULT_CONFIG)
    if overrides:
        cfg.update(overrides)
    return cfg


class MyMem:
    def __init__(
            self,
            num_classes,
            capacity,
            conf_threshold,
            memory_mode="enhanced",
            use_ema_proto=True,
            min_samples_per_class=5,
            proto_temperature=0.35,
    ):
        self.capacity = capacity
        self.conf_threshold = conf_threshold
        self.num_classes = num_classes
        self.memory_mode = str(memory_mode).lower()
        self.use_ema_proto = use_ema_proto
        self.min_samples_per_class = min_samples_per_class
        self.proto_temperature = proto_temperature

        self.mem = []
        self.feats_mem = None
        self.pred_prob_mem = None
        self.class_mem = None
        self.age_mem = None
        self.lambda_age = 1
        self.lambda_conf = 1

        self.ema_momentum = 0.9
        self.class_centroid_ema = None
        self.class_proto_count = torch.zeros(num_classes)

    def __len__(self):
        return len(self.mem)

    def get_age_score(self, age):
        return torch.sigmoid(-age / self.capacity)

    def get_conf_score(self, conf):
        return torch.exp((conf - 1) / self.conf_threshold)

    def get_score(self, age, conf):
        return self.get_conf_score(conf) * self.lambda_conf + self.get_age_score(age) * self.lambda_age

    def update_mem_age(self):
        if self.age_mem is not None:
            self.age_mem += 1

    def update(self, x, fea, prob, conf):
        for i in range(x.shape[0]):
            c = conf[i]
            xx = x[i]
            feat_i = fea[i]
            prob_i = prob[i]

            if self.use_ema_proto:
                self._update_ema_proto(feat_i, prob_i, c)

            if len(self.mem) < self.capacity:
                self.add_to_mem(xx, feat_i, prob_i)
            elif self.memory_mode == "vanilla":
                self.remove_from_mem(0)
                self.add_to_mem(xx, feat_i, prob_i)
            else:
                conf_mem = self.pred_prob_mem.max(1)[0]
                score_mem = self.get_score(self.age_mem.to(conf_mem.device), conf_mem)
                class_occupance = torch.bincount(self.class_mem, minlength=self.num_classes)

                valid_mask = class_occupance > self.min_samples_per_class
                if valid_mask.sum() == 0:
                    replace_idx = score_mem.argmin()
                else:
                    masked_occupance = class_occupance.float()
                    masked_occupance[~valid_mask] = -1
                    prevalent_class = masked_occupance.argmax()
                    prevalent_samp_idx = torch.where(self.class_mem == prevalent_class)[0]
                    score_mem_prevalent = score_mem[prevalent_samp_idx]
                    idx = score_mem_prevalent.argmin()
                    replace_idx = prevalent_samp_idx[idx]

                score_samp = self.get_score(torch.tensor([0], device=c.device), c)
                if score_samp > score_mem[replace_idx]:
                    self.remove_from_mem(replace_idx)
                    self.add_to_mem(xx, feat_i, prob_i)

    def _update_ema_proto(self, feat, prob, conf):
        c = prob.argmax().item()
        feat_norm = F.normalize(feat.unsqueeze(0), p=2, dim=1)[0]

        if self.class_centroid_ema is None:
            self.class_centroid_ema = torch.zeros(self.num_classes, feat.shape[0], device=feat.device)

        weight = conf.item()
        momentum = 1.0 - (1.0 - self.ema_momentum) * weight

        if self.class_centroid_ema[c].sum() == 0:
            self.class_centroid_ema[c] = feat_norm.detach()
        else:
            self.class_centroid_ema[c] = momentum * self.class_centroid_ema[c] + (1 - momentum) * feat_norm.detach()
        self.class_proto_count[c] += 1

    def add_to_mem(self, x, fea, prob):
        self.mem.append(x)
        self.feats_mem = torch.cat([self.feats_mem, fea.unsqueeze(0)]) if self.feats_mem is not None else fea.unsqueeze(
            0)
        self.pred_prob_mem = (
            torch.cat([self.pred_prob_mem, prob.unsqueeze(0)])
            if self.pred_prob_mem is not None
            else prob.unsqueeze(0)
        )
        c = prob.argmax().item()
        self.class_mem = (
            torch.cat([self.class_mem, torch.tensor([c], device=fea.device)])
            if self.class_mem is not None
            else torch.tensor([c], device=fea.device)
        )
        self.age_mem = (
            torch.cat([self.age_mem, torch.tensor([0], device=fea.device)])
            if self.age_mem is not None
            else torch.tensor([0], device=fea.device)
        )

    def remove_from_mem(self, idx):
        self.mem.pop(idx)
        self.feats_mem = torch.cat([self.feats_mem[:idx], self.feats_mem[idx + 1:]])
        self.pred_prob_mem = torch.cat([self.pred_prob_mem[:idx], self.pred_prob_mem[idx + 1:]])
        self.class_mem = torch.cat([self.class_mem[:idx], self.class_mem[idx + 1:]])
        self.age_mem = torch.cat([self.age_mem[:idx], self.age_mem[idx + 1:]])

    def sample_balanced_batch(self, batch_size):
        if len(self.mem) == 0 or self.class_mem is None:
            return None

        unique_classes = self.class_mem.unique()
        num_classes_present = len(unique_classes)

        if num_classes_present < max(2, self.num_classes // 2):
            return None

        samples_per_class = max(1, batch_size // num_classes_present)

        replay_x = []
        for c in unique_classes:
            idx = torch.where(self.class_mem == c)[0]
            if len(idx) > 0:
                perm = torch.randperm(len(idx))[:samples_per_class]
                for i in idx[perm]:
                    replay_x.append(self.mem[i.item()].unsqueeze(0))

        if len(replay_x) == 0:
            return None

        return torch.cat(replay_x, dim=0)

    def get_pseudo_label(self, feature):
        valid_class = torch.zeros(self.num_classes, dtype=torch.bool, device=feature.device)

        if self.use_ema_proto and self.class_centroid_ema is not None:
            centroid_norm = F.normalize(self.class_centroid_ema.to(feature.device), p=2, dim=1)
            similarity = (centroid_norm @ feature.T) / self.proto_temperature
            valid_class = (self.class_proto_count > 0).to(feature.device)
        else:
            feats_mem = self.feats_mem
            label_mem = self.pred_prob_mem.argmax(1)
            class_centroid = torch.zeros(self.num_classes, feats_mem.shape[1], device=feature.device)
            for i in range(self.num_classes):
                feat_class = feats_mem[label_mem == i]
                if len(feat_class) == 0:
                    continue
                valid_class[i] = True
                feat_class = F.normalize(feat_class, p=2, dim=1)
                class_centroid[i] = feat_class.mean(0)

            similarity = (class_centroid @ feature.T) / self.proto_temperature

        similarity[~valid_class] = -1e9
        t = torch.softmax(similarity.T, dim=1)
        return t


class PLTAMY(nn.Module):
    def __init__(
            self,
            model,
            num_classes,
            lr=0.001,
            optim="Adam",
            conf_threshold=0.78,
            memory_size=150,
            temperature_scaling=2,
            temporal_ema_momentum=0.9,
            temporal_mix_min=0.2,
            temporal_mix_max=0.8,
            use_ema_proto=True,
            pseudo_conf_gate=True,
            pseudo_max_mix=0.55,
            memory_mode="enhanced",
            temp_smooth_base=0.75,
            smooth_uncertainty_bonus=0.2,
            dynamic_align_scale=True,
            adaptive_threshold_gamma=0.2,
            min_adaptive_threshold=0.65,
            feature_guided_smooth=True,
            enable_temporal_dist_smooth=True,
            enable_plain_entmin=False,
            min_samples_per_class=5,
            proto_temperature=0.35,
            lambda_proto_align=0.65,
            distance_threshold=0.6,
            enable_memory=True,
            enable_protoalign=True,
            base_bn_momentum=0.1,
            bn_gamma=2.0,
            enable_cb_replay=True,
            lambda_replay=0.7,
            batch_diversity_threshold=2.5,
    ):
        super(PLTAMY, self).__init__()
        self.num_classes = num_classes
        self.lr = lr

        self.conf_threshold = conf_threshold
        self.adaptive_threshold_gamma = adaptive_threshold_gamma
        self.min_adaptive_threshold = min_adaptive_threshold
        self.distance_threshold = distance_threshold

        self.temperature_scaling = temperature_scaling
        self.lambda_proto_align = lambda_proto_align
        self.dynamic_align_scale = dynamic_align_scale

        self.temporal_ema_momentum = min(max(temporal_ema_momentum, 0.0), 0.999)
        self.temporal_mix_min = min(max(temporal_mix_min, 0.0), 1.0)
        self.temporal_mix_max = min(max(temporal_mix_max, self.temporal_mix_min), 1.0)

        self.use_ema_proto = use_ema_proto
        self.pseudo_conf_gate = pseudo_conf_gate
        self.pseudo_max_mix = max(0.0, min(1.0, pseudo_max_mix))
        self.memory_mode = str(memory_mode).lower()
        self.is_vanilla_memory = self.memory_mode == "vanilla"

        self.temp_smooth_base = max(0.0, min(1.0, temp_smooth_base))
        self.smooth_uncertainty_bonus = smooth_uncertainty_bonus
        self.feature_guided_smooth = feature_guided_smooth
        self.enable_temporal_dist_smooth = enable_temporal_dist_smooth

        self.enable_plain_entmin = enable_plain_entmin
        self.enable_memory = enable_memory
        self.enable_protoalign = enable_protoalign
        self.enable_cb_replay = enable_cb_replay
        self.lambda_replay = float(lambda_replay)
        self.batch_diversity_threshold = float(batch_diversity_threshold)
        self.base_bn_momentum = float(base_bn_momentum)
        self.bn_gamma = float(bn_gamma)

        self.running_smooth_prob = None
        self.running_last_feat = None
        self.temporal_prob_ema = None

        self.total_samples_processed = 0
        self.diagnose_avg_loss_replay = 0.0

        self.mem = MyMem(
            num_classes,
            memory_size,
            conf_threshold,
            memory_mode=self.memory_mode,
            use_ema_proto=use_ema_proto,
            min_samples_per_class=min_samples_per_class,
            proto_temperature=proto_temperature,
        )

        self.model = model
        params, _ = collect_params(self.model)
        self.param_count = len(params)

        if optim == "SGD":
            self.optimizer = torch.optim.SGD(params, lr=lr)
        elif optim == "Adam":
            self.optimizer = torch.optim.Adam(params, lr=lr)
        else:
            raise NotImplementedError

        self.last_layer_input = None
        self._register_hook()
        self.batch_idx = 0

        print(f"\n{'=' * 50}")
        print("[INIT] PLTAMY V26 (RCMA) Loaded!")
        print("[INIT] Core idea: Replay-Compensated Marginal Alignment")
        print("[INIT] Note: Explicit BN freezing removed. Fully trusting RCMA for stabilization.")
        print(f"[INIT] memory_mode={self.memory_mode}")
        print(f"{'=' * 50}\n")

    def _register_hook(self):
        linear_layers = []
        for _, module in self.model.named_modules():
            if isinstance(module, nn.Linear):
                linear_layers.append(module)

        if not linear_layers:
            raise RuntimeError("Model has no linear layer for feature extraction.")

        last_linear = linear_layers[-1]

        def hook(module, input, output):
            self.last_layer_input = input[0].detach()

        last_linear.register_forward_hook(hook)

    def _get_temporal_prob_ema(self, device):
        if self.temporal_prob_ema is None:
            return torch.full((self.num_classes,), 1.0 / self.num_classes, device=device)
        return self.temporal_prob_ema.to(device)

    def _update_temporal_prob_ema(self, current_mean_prob):
        current_mean_prob = current_mean_prob.detach()
        current_mean_prob = current_mean_prob / current_mean_prob.sum().clamp(min=1e-6)
        if self.temporal_prob_ema is None:
            self.temporal_prob_ema = current_mean_prob
        else:
            self.temporal_prob_ema = (
                    self.temporal_ema_momentum * self.temporal_prob_ema.to(current_mean_prob.device)
                    + (1.0 - self.temporal_ema_momentum) * current_mean_prob
            )
            self.temporal_prob_ema = self.temporal_prob_ema / self.temporal_prob_ema.sum().clamp(min=1e-6)

    def forward(self, x):
        self.batch_idx += 1
        batch_size = x.shape[0]

        # ========== 阶段一：前向测试稳健推理 ==========
        self.model.eval()

        output_prob = torch.zeros(batch_size, self.num_classes, device=x.device)
        features = torch.zeros(batch_size, self._get_feature_dim(x), device=x.device)
        is_pseudo = torch.zeros(batch_size, dtype=torch.bool, device=x.device)
        pseudo_conf_tensor = torch.ones(batch_size, device=x.device)

        with torch.no_grad():
            if self.enable_temporal_dist_smooth:
                temp_prob_ema = self._get_temporal_prob_ema(x.device).detach()
            else:
                temp_prob_ema = torch.full((self.num_classes,), 1.0 / self.num_classes, device=x.device)
            max_ema_prob = temp_prob_ema.max().clamp_min(1e-6)

            if self.enable_memory:
                self.mem.update_mem_age()

            for i in range(batch_size):
                xx = x[i: i + 1]
                out = self.model(xx)
                fea = self.last_layer_input
                features[i] = fea[0]

                p = out.softmax(1)
                conf = p.max(1)[0].item()
                pred_class = p.argmax(1)[0].item()

                if self.is_vanilla_memory:
                    dynamic_threshold = self.conf_threshold
                    feat_sim = 1.0
                    is_conf_low = conf < dynamic_threshold
                    is_dist_far = False
                else:
                    class_ratio = (temp_prob_ema[pred_class] / max_ema_prob).clamp(1e-4, 1.0)
                    dynamic_threshold = self.conf_threshold * (class_ratio ** self.adaptive_threshold_gamma).item()
                    dynamic_threshold = max(self.min_adaptive_threshold, dynamic_threshold)

                    feat_sim = 1.0
                    if self.enable_memory and len(self.mem) > 0 and self.mem.class_centroid_ema is not None and \
                            self.mem.class_proto_count[pred_class] > 0:
                        centroid = self.mem.class_centroid_ema[pred_class].to(fea.device)
                        feat_sim = F.cosine_similarity(fea, centroid.unsqueeze(0)).item()

                    is_conf_low = conf < dynamic_threshold
                    is_dist_far = (feat_sim < self.distance_threshold) and (len(self.mem) > 0)

                if is_conf_low or is_dist_far:
                    if self.enable_memory and len(self.mem) > 0:
                        pseudo_prob = self.mem.get_pseudo_label(fea)
                        pseudo_conf = pseudo_prob.max(1)[0].item()
                        pseudo_conf_tensor[i] = pseudo_conf

                        if self.is_vanilla_memory:
                            output_prob[i] = pseudo_prob[0]
                            is_pseudo[i] = True
                        else:
                            if not self.pseudo_conf_gate or pseudo_conf > conf + 0.01:
                                mix_ratio = min(self.pseudo_max_mix, pseudo_conf)
                                fused_prob = mix_ratio * pseudo_prob + (1 - mix_ratio) * p
                                output_prob[i] = fused_prob[0]
                                is_pseudo[i] = True
                            else:
                                output_prob[i] = p[0]
                    else:
                        output_prob[i] = p[0]
                else:
                    output_prob[i] = p[0]
                    if self.enable_memory:
                        self.mem.update(xx, fea, p, torch.tensor([conf], device=x.device))

            if self.temp_smooth_base > 0:
                feat_norm = F.normalize(features, p=2, dim=1)
                smoothed_prob = torch.zeros_like(output_prob)
                for i in range(batch_size):
                    current_prob = output_prob[i]
                    curr_c = current_prob.max().item()
                    if self.running_smooth_prob is None:
                        smoothed_prob[i] = current_prob
                        self.running_smooth_prob = current_prob.clone()
                        self.running_last_feat = feat_norm[i].clone()
                    else:
                        prev_feat = self.running_last_feat
                        sim = torch.dot(feat_norm[i], prev_feat).clamp(0.0, 1.0)
                        uncertainty_bonus = self.smooth_uncertainty_bonus * (1.0 - curr_c)
                        alpha_raw = (
                                                self.temp_smooth_base + uncertainty_bonus) * sim if self.feature_guided_smooth else (
                                    self.temp_smooth_base + uncertainty_bonus)
                        alpha = min(alpha_raw.item(), 0.96)
                        smoothed_prob[i] = (1 - alpha) * current_prob + alpha * self.running_smooth_prob
                        self.running_smooth_prob = smoothed_prob[i].clone()
                        self.running_last_feat = feat_norm[i].clone()
                output_prob = smoothed_prob

        # ========== 阶段二：含 RCMA 的反向自适应 ==========
        self.total_samples_processed += batch_size

        with torch.enable_grad():
            self.model.eval()
            with torch.no_grad():
                temp_pred = self.model(x)
                temp_prob_base = F.softmax(temp_pred / self.temperature_scaling, dim=1)
                temp_mean_prob = temp_prob_base.mean(0).clamp(min=1e-6)
                temp_mean_prob = temp_mean_prob / temp_mean_prob.sum()
                temp_max_entropy = torch.log(torch.tensor(float(self.num_classes), device=x.device))
                temp_entropy = -torch.sum(temp_mean_prob * temp_mean_prob.log())
                current_marginal_ratio = (temp_entropy / temp_max_entropy).clamp(0.0, 1.0)

            # 核心变动：无论数据流多么单一（熵多么低），始终保持 train 模式更新 BN
            # 依靠下方的 RCMA (重放样本联合计算) 来对冲 BN 的统计量漂移
            self.model.train()

            pred = self.model(x)
            clean_feat = self.last_layer_input.detach().clone()

            pred_prob_base = F.softmax(pred / self.temperature_scaling, dim=1)
            output_conf = output_prob.max(1)[0]
            if self.enable_plain_entmin:
                pred_scaled = pred / self.temperature_scaling
                pred_prob_adapt = pred_prob_base
                ent = -(pred_prob_adapt * F.log_softmax(pred_scaled, dim=1)).sum(1)
                score_ent = torch.ones_like(output_conf)
            else:
                pred_scaled = pred / self.temperature_scaling
                pred_prob_adapt = pred_prob_base
                ent = -(pred_prob_adapt * F.log_softmax(pred_scaled, dim=1)).sum(1)
                score_ent = torch.ones_like(output_conf)

            loss_ent_min = ent if self.enable_plain_entmin else ent.mul(score_ent)
            loss_align_val = torch.zeros_like(loss_ent_min)
            if self.enable_memory and self.enable_protoalign and len(self.mem) > 0 and self.lambda_proto_align > 0:
                with torch.no_grad():
                    target_proto_dist = self.mem.get_pseudo_label(clean_feat).clamp(min=1e-6)
                logp_pred = F.log_softmax(pred / self.temperature_scaling, dim=1)
                dynamic_align_weight = self.lambda_proto_align * (
                    pseudo_conf_tensor if self.dynamic_align_scale else 1.0)
                loss_align_val = F.kl_div(logp_pred, target_proto_dist, reduction="none").mean(
                    1) * score_ent * dynamic_align_weight

            loss_cur_base = (loss_ent_min + loss_align_val).mean()

            # --- 🌟 V26 核心突破：RCMA 联合熵优化 ---
            rep_x = None
            if self.enable_cb_replay and self.enable_memory:
                rep_x = self.mem.sample_balanced_batch(batch_size)

            if rep_x is not None:
                rep_x = rep_x.to(x.device)
                pred_rep = self.model(rep_x)
                feat_rep = self.last_layer_input.detach().clone()
                prob_rep = F.softmax(pred_rep / self.temperature_scaling, dim=1)

                # 1. 强制重放样本自我确信
                loss_rep_ent = -(prob_rep * F.log_softmax(pred_rep / self.temperature_scaling, dim=1)).sum(1).mean()

                # 2. 强制重放样本对齐全局流形锚点 (极大地稳定了分类面)
                with torch.no_grad():
                    target_rep_dist = self.mem.get_pseudo_label(feat_rep).clamp(min=1e-6)
                logp_rep = F.log_softmax(pred_rep / self.temperature_scaling, dim=1)
                loss_rep_align = F.kl_div(logp_rep, target_rep_dist, reduction="batchmean")

                # 3. RCMA 联合边缘分布熵：将当前批次和重放批次物理拼合算熵！
                # 当前批次加权累加，过滤低置信噪点
                current_weighted_prob = pred_prob_base * score_ent.unsqueeze(-1)
                joint_prob_sum = current_weighted_prob.sum(0) + prob_rep.sum(0)
                total_weight = score_ent.sum() + prob_rep.shape[0]

                joint_mean_prob = (joint_prob_sum / total_weight.clamp(min=1e-6)).clamp(min=1e-6)
                joint_mean_prob = joint_mean_prob / joint_mean_prob.sum()

                loss_joint_div = -torch.sum(joint_mean_prob * torch.log(joint_mean_prob))

                # 总损失 = 当前置信/对抗/对齐 + 重放置信/对齐 - 联合多样性(因为要最大化熵，所以是负号)
                loss = loss_cur_base + self.lambda_replay * (loss_rep_ent + loss_rep_align) - loss_joint_div

                self.diagnose_avg_loss_replay += (loss_rep_ent + loss_rep_align).item() * batch_size
            else:
                # Early Warmup 阶段退回原始 EMA Diversity
                current_mean_prob = (pred_prob_base * score_ent.unsqueeze(-1)).sum(0) / score_ent.sum().clamp(min=1e-6)
                current_mean_prob = current_mean_prob.clamp(min=1e-6)
                current_mean_prob = current_mean_prob / current_mean_prob.sum().clamp(min=1e-6)

                if self.enable_plain_entmin:
                    loss = loss_cur_base
                else:
                    if self.enable_temporal_dist_smooth:
                        temporal_mix = self.temporal_mix_min + (
                                    self.temporal_mix_max - self.temporal_mix_min) * current_marginal_ratio
                        temporal_prob = self._get_temporal_prob_ema(pred.device).detach()
                        mixed_mean_prob = temporal_mix * current_mean_prob + (1.0 - temporal_mix) * temporal_prob
                        mixed_mean_prob = mixed_mean_prob / mixed_mean_prob.sum().clamp(min=1e-6)
                    else:
                        mixed_mean_prob = current_mean_prob

                    loss_div = entropy_from_prob(mixed_mean_prob)
                    loss = loss_cur_base - (current_marginal_ratio * loss_div)

            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()

            # 全局 EMA 更新：不再受保护机制打断
            current_mean_for_ema = pred_prob_base.mean(0).clamp(min=1e-6)
            current_mean_for_ema = current_mean_for_ema / current_mean_for_ema.sum()
            if self.enable_temporal_dist_smooth:
                self._update_temporal_prob_ema(current_mean_for_ema)

        if self.batch_idx <= 5 or self.batch_idx % 20 == 0:
            avg_loss_rep = self.diagnose_avg_loss_replay / self.total_samples_processed
            print("\n==================== PLTAMY V26 (RCMA) Monitor ====================")
            print(
                f"processed_samples={self.total_samples_processed} | batch_h_ratio={current_marginal_ratio.item():.4f}")
            if rep_x is not None:
                print(f"rcma_replay_penalty={avg_loss_rep:.4f}")
            print("=================================================================\n")

        output_logits = torch.log(torch.clamp(output_prob, min=1e-10))
        return output_logits

    def _get_feature_dim(self, x):
        with torch.no_grad():
            self.model(x[0:1])
            return self.last_layer_input.shape[1]

    @staticmethod
    def configure_model(model):
        model.train()
        return model


def collect_params(model):
    params = []
    names = []
    for nm, m in model.named_modules():
        if isinstance(m, (nn.BatchNorm1d, nn.BatchNorm2d, nn.LayerNorm)):
            for np, p in m.named_parameters():
                if np in ["weight", "bias"]:
                    params.append(p)
                    names.append(f"{nm}.{np}")
    return params, names


def KL(logit1, logit2, T, reverse=False):
    if reverse:
        logit1, logit2 = logit2, logit1
    logit1 = logit1 / T
    logit2 = logit2 / T
    p1 = logit1.softmax(1)
    logp1 = logit1.log_softmax(1)
    logp2 = logit2.log_softmax(1)
    return (p1 * (logp1 - logp2)).mean(1)


def entropy_from_prob(prob):
    prob = prob.clamp(min=1e-10)
    return -torch.sum(prob * prob.log())


def setup_pltamy(model, cfg, num_classes):
    cfg = get_pltamy_default_config(cfg)
    model = PLTAMY(
        model,
        num_classes,
        lr=cfg["lr"],
        optim="Adam",
        conf_threshold=cfg["conf_threshold"],
        memory_size=cfg["memory_size"],
        temperature_scaling=cfg["temperature_scaling"],
        memory_mode=cfg.get("memory_mode", "enhanced"),
        temporal_ema_momentum=cfg.get("temporal_ema_momentum", 0.9),
        temporal_mix_min=cfg.get("temporal_mix_min", 0.2),
        temporal_mix_max=cfg.get("temporal_mix_max", 0.8),
        use_ema_proto=cfg.get("use_ema_proto", True),
        pseudo_conf_gate=cfg.get("pseudo_conf_gate", True),
        pseudo_max_mix=cfg.get("pseudo_max_mix", 0.55),
        temp_smooth_base=cfg.get("temp_smooth_base", 0.75),
        feature_guided_smooth=cfg.get("feature_guided_smooth", True),
        enable_temporal_dist_smooth=cfg.get("enable_temporal_dist_smooth", True),
        enable_plain_entmin=cfg.get("enable_plain_entmin", False),
        enable_memory=cfg.get("enable_memory", True),
        enable_protoalign=cfg.get("enable_protoalign", True),
        min_samples_per_class=cfg.get("min_samples_per_class", 5),
        proto_temperature=cfg.get("proto_temperature", 0.32),
        lambda_proto_align=cfg.get("lambda_proto_align", 0.65),
        distance_threshold=cfg.get("distance_threshold", 0.6),
        enable_cb_replay=cfg.get("enable_cb_replay", True),
        lambda_replay=cfg.get("lambda_replay", 0.7),
        batch_diversity_threshold=cfg.get("batch_diversity_threshold", 2.5),
    )
    model = PLTAMY.configure_model(model)
    return model, None


def setup_pltamy_ablation(model, cfg, num_classes, IM_loss, PI_loss, label_correction):
    return setup_pltamy(model, cfg, num_classes)


def setup_pltamy_runtime(model, cfg, num_classes):
    return setup_pltamy(model, cfg, num_classes)
