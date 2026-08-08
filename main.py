from __future__ import annotations

import csv
import json
import random
import time
from pathlib import Path
from typing import Dict, List, Sequence, Tuple

import numpy as np
import torch
from scipy.stats import rankdata
from sklearn.ensemble import ExtraTreesRegressor
from torch import nn
from torch.nn import functional as F
from tqdm.auto import tqdm

from config import CONFIG, PACKAGE_RELEASE, RunConfig
from data_pipeline import ChunkRecord, discover_pdbbind_chunks, iter_split_batches
from encoder import RandomComplexEncoder
from physics_fingerprint import build_physics_fingerprint
from regressors import JointRegressor, ReservoirRegressor


# Set random seeds.
def seed_everything(seed: int) -> None:
    random.seed(int(seed))
    np.random.seed(int(seed) % (2**32))
    torch.manual_seed(int(seed))
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(int(seed))


# Write JSON output.
def write_json(path: Path, payload: Dict[str, object]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")


# Compute Pearson correlation.
def pearson_tensor(prediction: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    x = prediction.float().view(-1)
    y = target.float().view(-1)
    x = x - x.mean()
    y = y - y.mean()
    return (x * y).sum() / torch.sqrt(x.square().sum() * y.square().sum() + 1.0e-12)


# Compute regression metrics.
def regression_metrics(prediction: torch.Tensor, target: torch.Tensor) -> Dict[str, float]:
    prediction = prediction.float().view(-1).cpu()
    target = target.float().view(-1).cpu()
    residual = prediction - target
    centered = target - target.mean()
    spearman = float(np.corrcoef(rankdata(prediction.numpy()), rankdata(target.numpy()))[0, 1])
    return {
        "pearson": float(pearson_tensor(prediction, target)),
        "spearman": spearman,
        "rmse": float(torch.sqrt(residual.square().mean())),
        "mae": float(residual.abs().mean()),
        "bias": float(residual.mean()),
        "r2": float(1.0 - residual.square().sum() / (centered.square().sum() + 1.0e-12)),
        "n": float(target.numel()),
    }


# Copy model state to CPU.
def cpu_state_dict(model: nn.Module) -> Dict[str, torch.Tensor]:
    return {name: value.detach().cpu() for name, value in model.state_dict().items()}


# Build one frozen random encoder.
def make_random_encoder(cfg: RunConfig, seed: int, device: torch.device) -> RandomComplexEncoder:
    seed_everything(seed)
    encoder = RandomComplexEncoder(cfg).to(device)
    encoder.requires_grad_(False)
    encoder.eval()
    return encoder


# Build the encoder bank.
def make_encoder_bank(cfg: RunConfig, device: torch.device) -> Tuple[List[RandomComplexEncoder], List[str]]:
    encoders: List[RandomComplexEncoder] = []
    names: List[str] = []
    for seed in cfg.random_encoder_seeds:
        encoders.append(make_random_encoder(cfg, seed, device))
        names.extend((
            f"random_{seed}:protein_interaction",
            f"random_{seed}:ligand",
        ))
    encoders.append(make_random_encoder(cfg, cfg.extra_random_encoder_seed, device))
    names.append(f"random_{cfg.extra_random_encoder_seed}:protein_ligand")
    return encoders, names


# Encode one dataset split.
@torch.inference_mode()
def encode_split(
    split: str,
    records: Sequence[ChunkRecord],
    encoders: Sequence[RandomComplexEncoder],
    cfg: RunConfig,
    device: torch.device,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, List[str]]:
    total = sum(record.row_count for record in records)
    reservoir_parts: List[torch.Tensor] = []
    physics_parts: List[torch.Tensor] = []
    label_parts: List[torch.Tensor] = []
    complex_ids: List[str] = []
    progress = tqdm(
        total=total,
        desc=f"{split} reservoir+physics encoding",
        unit="graph",
        dynamic_ncols=True,
        mininterval=0.25,
    )
    for batch in iter_split_batches(records, cfg.graph_batch_size, split):
        physics_parts.append(build_physics_fingerprint(batch).cpu())
        if cfg.pin_memory and torch.cuda.is_available():
            batch.pin_memory_()
        batch = batch.to(device, non_blocking=cfg.pin_memory)
        branches: List[torch.Tensor] = []
        for index, encoder in enumerate(encoders):
            output = encoder(batch)
            if index < cfg.random_count:
                branches.append(torch.cat((
                    output["protein_pool"],
                    output["interaction_pool"],
                ), dim=-1).float())
                branches.append(output["ligand_pool"].float())
            else:
                branches.append(torch.cat((
                    output["protein_pool"],
                    output["ligand_pool"],
                ), dim=-1).float())
        reservoir_parts.append(torch.cat(branches, dim=-1).cpu())
        label_parts.append(batch.labels.float().cpu())
        complex_ids.extend(batch.complex_ids)
        progress.update(batch.num_graphs)
    progress.close()
    return (
        torch.cat(reservoir_parts),
        torch.cat(physics_parts),
        torch.cat(label_parts),
        complex_ids,
    )


# Standardize train and paired data.
def standardize_pair(train: torch.Tensor, other: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    mean = train.mean(dim=0, keepdim=True)
    std = torch.sqrt(train.var(dim=0, unbiased=False, keepdim=True) + 1.0e-8).clamp_min(1.0e-5)
    return (train - mean) / std, (other - mean) / std, mean, std


# Predict with a reservoir expert.
def predict_reservoir(
    model: ReservoirRegressor,
    value: torch.Tensor,
    batch_size: int,
    device: torch.device,
    target_mean: torch.Tensor,
    target_std: torch.Tensor,
) -> torch.Tensor:
    model.eval()
    parts: List[torch.Tensor] = []
    with torch.no_grad():
        for start in range(0, value.shape[0], int(batch_size)):
            selected = value[start : start + int(batch_size)].to(device)
            parts.append((model(selected).cpu() * target_std + target_mean))
    return torch.cat(parts)


# Predict with a joint expert.
def predict_joint(
    model: JointRegressor,
    reservoir: torch.Tensor,
    physics: torch.Tensor,
    batch_size: int,
    device: torch.device,
    target_mean: torch.Tensor,
    target_std: torch.Tensor,
) -> torch.Tensor:
    model.eval()
    parts: List[torch.Tensor] = []
    with torch.no_grad():
        for start in range(0, reservoir.shape[0], int(batch_size)):
            r = reservoir[start : start + int(batch_size)].to(device)
            p = physics[start : start + int(batch_size)].to(device)
            parts.append((model(r, p).cpu() * target_std + target_mean))
    return torch.cat(parts)


# Train one neural expert.
def train_neural_head(
    family: str,
    head_index: int,
    seed: int,
    train_reservoir: torch.Tensor,
    train_physics: torch.Tensor,
    train_target: torch.Tensor,
    val_reservoir: torch.Tensor,
    val_physics: torch.Tensor,
    val_target_raw: torch.Tensor,
    target_mean: torch.Tensor,
    target_std: torch.Tensor,
    cfg: RunConfig,
    device: torch.device,
) -> Tuple[nn.Module, Dict[str, float], List[Dict[str, float]], torch.Tensor]:
    seed_everything(seed)
    if family == "joint":
        model: nn.Module = JointRegressor(cfg, train_physics.shape[1]).to(device)
    else:
        model = ReservoirRegressor(cfg).to(device)
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=cfg.learning_rate,
        weight_decay=cfg.weight_decay,
        foreach=False,
        fused=False,
    )
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer,
        mode="max",
        factor=cfg.scheduler_factor,
        patience=cfg.scheduler_patience,
        min_lr=cfg.minimum_learning_rate,
    )

    train_r_device = train_reservoir.to(device)
    train_p_device = train_physics.to(device)
    train_y_device = train_target.to(device)
    generator = torch.Generator(device="cpu")
    generator.manual_seed(seed + 19)

    best_pearson = -1.0e9
    best_metrics: Dict[str, float] = {}
    best_state: Dict[str, torch.Tensor] = {}
    history: List[Dict[str, float]] = []
    patience = 0
    progress = tqdm(
        total=cfg.max_epochs,
        desc=f"{family} head {head_index + 1}",
        unit="epoch",
        dynamic_ncols=True,
        mininterval=0.25,
    )

    for epoch in range(1, cfg.max_epochs + 1):
        model.train()
        order = torch.randperm(train_reservoir.shape[0], generator=generator)
        loss_sum = 0.0
        batches = 0
        for start in range(0, order.numel(), cfg.train_batch_size):
            selected = order[start : start + cfg.train_batch_size].to(device)
            optimizer.zero_grad(set_to_none=True)
            batch_r = train_r_device.index_select(0, selected)
            batch_p = train_p_device.index_select(0, selected)
            if cfg.feature_noise_std > 0.0:
                batch_r = batch_r + torch.randn_like(batch_r) * cfg.feature_noise_std
                batch_p = batch_p + torch.randn_like(batch_p) * cfg.feature_noise_std
            if family == "joint":
                prediction = model(batch_r, batch_p)
            else:
                prediction = model(batch_r)
            target = train_y_device.index_select(0, selected)
            huber = F.smooth_l1_loss(prediction, target, beta=cfg.huber_beta)
            correlation = pearson_tensor(prediction, target)
            loss = huber + cfg.pearson_loss_weight * (1.0 - correlation)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), cfg.grad_clip_norm)
            optimizer.step()
            loss_sum += float(loss.detach())
            batches += 1

        if family == "joint":
            val_prediction = predict_joint(
                model, val_reservoir, val_physics, cfg.eval_batch_size,
                device, target_mean, target_std,
            )
        else:
            val_prediction = predict_reservoir(
                model, val_reservoir, cfg.eval_batch_size,
                device, target_mean, target_std,
            )
        metrics = regression_metrics(val_prediction, val_target_raw)
        scheduler.step(metrics["pearson"])
        row = {
            "family": family,
            "head": float(head_index),
            "seed": float(seed),
            "epoch": float(epoch),
            "loss": loss_sum / max(batches, 1),
            "lr": float(optimizer.param_groups[0]["lr"]),
            **metrics,
        }
        history.append(row)

        if metrics["pearson"] > best_pearson:
            best_pearson = metrics["pearson"]
            best_metrics = {**metrics, "epoch": float(epoch), "seed": float(seed), "family": family}
            best_state = cpu_state_dict(model)
            patience = 0
        else:
            patience += 1

        progress.set_postfix(
            best=f"{best_pearson:.4f}",
            loss=f"{row['loss']:.4f}",
            lr=f"{row['lr']:.1e}",
            r=f"{metrics['pearson']:.4f}",
            rmse=f"{metrics['rmse']:.3f}",
        )
        progress.update(1)
        if patience >= cfg.early_stop_patience:
            break

    progress.close()
    model.load_state_dict(best_state)
    if family == "joint":
        best_prediction = predict_joint(
            model, val_reservoir, val_physics, cfg.eval_batch_size,
            device, target_mean, target_std,
        )
    else:
        best_prediction = predict_reservoir(
            model, val_reservoir, cfg.eval_batch_size,
            device, target_mean, target_std,
        )
    return model, best_metrics, history, best_prediction


# Fit ExtraTrees experts.
def fit_extra_trees(
    train_physics_raw: torch.Tensor,
    train_target_raw: torch.Tensor,
    val_physics_raw: torch.Tensor,
    val_target_raw: torch.Tensor,
    cfg: RunConfig,
) -> Tuple[List[ExtraTreesRegressor], List[torch.Tensor], List[Dict[str, float]]]:
    models: List[ExtraTreesRegressor] = []
    predictions: List[torch.Tensor] = []
    metrics: List[Dict[str, float]] = []
    x_train = train_physics_raw.numpy()
    y_train = train_target_raw.numpy()
    x_val = val_physics_raw.numpy()
    for seed in cfg.tree_seeds:
        model = ExtraTreesRegressor(
            n_estimators=cfg.extra_trees_estimators,
            max_features=cfg.extra_trees_max_features,
            min_samples_leaf=cfg.extra_trees_min_samples_leaf,
            random_state=int(seed),
            n_jobs=cfg.extra_trees_n_jobs,
        )
        model.fit(x_train, y_train)
        prediction = torch.from_numpy(model.predict(x_val)).float()
        value = regression_metrics(prediction, val_target_raw)
        value.update({"family": "extra_trees", "seed": float(seed), "epoch": 0.0})
        models.append(model)
        predictions.append(prediction)
        metrics.append(value)
        print(
            f"ExtraTrees seed={seed}: Val r={value['pearson']:.4f} "
            f"rho={value['spearman']:.4f} RMSE={value['rmse']:.3f}",
            flush=True,
        )
    return models, predictions, metrics



# Learn softmax ensemble weights.
def weighted_ensemble(
    predictions: Sequence[torch.Tensor],
    target: torch.Tensor,
    cfg: RunConfig,
) -> Tuple[torch.Tensor, torch.Tensor, Dict[str, float]]:
    matrix = torch.stack([value.float() for value in predictions], dim=1)
    logits = torch.zeros(matrix.shape[1], requires_grad=True)
    optimizer = torch.optim.Adam([logits], lr=cfg.ensemble_weight_lr)
    for _ in range(cfg.ensemble_weight_steps):
        optimizer.zero_grad(set_to_none=True)
        weights = torch.softmax(logits, dim=0)
        prediction = matrix @ weights
        loss = 1.0 - pearson_tensor(prediction, target)
        loss = loss + cfg.ensemble_weight_l2 * (weights.square().sum())
        loss.backward()
        optimizer.step()
    weights = torch.softmax(logits.detach(), dim=0)
    prediction = matrix @ weights
    return weights, prediction, regression_metrics(prediction, target)


# Save final predictions.
def save_predictions(path: Path, complex_ids: Sequence[str], target: torch.Tensor, prediction: torch.Tensor) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(("complex_id", "target", "prediction", "residual"))
        for complex_id, true_value, predicted_value in zip(complex_ids, target.tolist(), prediction.tolist()):
            writer.writerow((
                complex_id,
                f"{true_value:.8f}",
                f"{predicted_value:.8f}",
                f"{predicted_value - true_value:.8f}",
            ))


# Save neural training history.
def save_history(path: Path, rows: Sequence[Dict[str, float]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    columns = ("family", "head", "seed", "epoch", "loss", "lr", "pearson", "spearman", "rmse", "mae", "bias", "r2", "n")
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=columns)
        writer.writeheader()
        for row in rows:
            writer.writerow({key: row.get(key, "") for key in columns})


# Save candidate metrics.
def save_candidate_metrics(path: Path, names: Sequence[str], metrics: Sequence[Dict[str, float]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    columns = ("candidate", "family", "seed", "epoch", "pearson", "spearman", "rmse", "mae", "bias", "r2", "n")
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=columns)
        writer.writeheader()
        for name, value in zip(names, metrics):
            writer.writerow({"candidate": name, **{key: value.get(key, "") for key in columns if key != "candidate"}})


# Run the complete training pipeline.
def main() -> None:
    cfg = CONFIG
    torch.set_num_threads(cfg.cpu_threads)
    if torch.cuda.is_available() and cfg.allow_tf32:
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True
    device = torch.device(cfg.device if cfg.device.startswith("cuda") and torch.cuda.is_available() else "cpu")
    Path(cfg.output_root).mkdir(parents=True, exist_ok=True)
    write_json(cfg.resolved_config_json, cfg.to_dict())
    start_time = time.perf_counter()

    train_records = discover_pdbbind_chunks(cfg.pdbbind_root, cfg.train_split)
    val_records = discover_pdbbind_chunks(cfg.pdbbind_root, cfg.val_split)
    test_records = discover_pdbbind_chunks(cfg.pdbbind_root, cfg.test_split)

    encoders, branch_names = make_encoder_bank(cfg, device)
    print(f"Package              : {PACKAGE_RELEASE}", flush=True)
    print(f"Device               : {device}", flush=True)
    print(f"Random encoders      : {list(cfg.random_encoder_seeds)}", flush=True)
    print(f"Extra random encoder : seed={cfg.extra_random_encoder_seed} (protein+ligand)", flush=True)
    print(f"Reservoir branches   : {len(branch_names)}", flush=True)
    print(f"Branch order         : {branch_names}", flush=True)
    print(f"Reservoir dimensions : {cfg.reservoir_feature_dim}D", flush=True)
    print("Physics fingerprint  : raw node/edge/cross-distance/phys-gate statistics", flush=True)
    print("Encoded vectors      : RAM only; no feature files are written", flush=True)
    print(f"Chunks               : Train={len(train_records)}, Val={len(val_records)}, Test={len(test_records)}", flush=True)

    train_r_raw, train_p_raw, train_y_raw, train_ids = encode_split(
        cfg.train_split, train_records, encoders, cfg, device
    )
    val_r_raw, val_p_raw, val_y_raw, val_ids = encode_split(
        cfg.val_split, val_records, encoders, cfg, device
    )
    print(f"Physics dimension    : {train_p_raw.shape[1]}D", flush=True)

    del encoders
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    train_r, val_r, reservoir_mean, reservoir_std = standardize_pair(train_r_raw, val_r_raw)
    train_p, val_p, physics_mean, physics_std = standardize_pair(train_p_raw, val_p_raw)
    target_mean = train_y_raw.mean()
    target_std = torch.sqrt(train_y_raw.var(unbiased=False) + 1.0e-8).clamp_min(1.0e-5)
    train_y = (train_y_raw - target_mean) / target_std

    candidate_names: List[str] = []
    candidate_models: List[Tuple[str, object]] = []
    candidate_predictions: List[torch.Tensor] = []
    candidate_metrics: List[Dict[str, float]] = []
    history_rows: List[Dict[str, float]] = []

    for head_index, seed in enumerate(cfg.reservoir_mlp_seeds):
        model, metrics, history, prediction = train_neural_head(
            "reservoir", head_index, seed,
            train_r, train_p, train_y,
            val_r, val_p, val_y_raw,
            target_mean, target_std, cfg, device,
        )
        candidate_names.append(f"reservoir_mlp_{seed}")
        candidate_models.append(("reservoir", model.cpu()))
        candidate_predictions.append(prediction)
        candidate_metrics.append(metrics)
        history_rows.extend(history)
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    for head_index, seed in enumerate(cfg.joint_mlp_seeds):
        model, metrics, history, prediction = train_neural_head(
            "joint", head_index, seed,
            train_r, train_p, train_y,
            val_r, val_p, val_y_raw,
            target_mean, target_std, cfg, device,
        )
        candidate_names.append(f"joint_mlp_{seed}")
        candidate_models.append(("joint", model.cpu()))
        candidate_predictions.append(prediction)
        candidate_metrics.append(metrics)
        history_rows.extend(history)
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    tree_models, tree_predictions, tree_metrics = fit_extra_trees(
        train_p_raw, train_y_raw, val_p_raw, val_y_raw, cfg
    )
    for seed, model, prediction, metrics in zip(cfg.tree_seeds, tree_models, tree_predictions, tree_metrics):
        candidate_names.append(f"extra_trees_{seed}")
        candidate_models.append(("extra_trees", model))
        candidate_predictions.append(prediction)
        candidate_metrics.append(metrics)

    save_history(cfg.history_csv, history_rows)
    save_candidate_metrics(cfg.candidate_metrics_csv, candidate_names, candidate_metrics)

    weighted_weights, val_final_prediction, val_final_metrics = weighted_ensemble(
        candidate_predictions, val_y_raw, cfg
    )
    ensemble_mode = "softmax_weighted"
    save_predictions(cfg.val_predictions_csv, val_ids, val_y_raw, val_final_prediction)

    checkpoint_payload = {
        "package_release": PACKAGE_RELEASE,
        "branch_names": branch_names,
        "candidate_names": candidate_names,
        "candidate_models": candidate_models,
        "candidate_val_metrics": candidate_metrics,
        "ensemble_mode": ensemble_mode,
        "weighted_weights": weighted_weights,
        "val_final_metrics": val_final_metrics,
        "scalers": {
            "reservoir_mean": reservoir_mean,
            "reservoir_std": reservoir_std,
            "physics_mean": physics_mean,
            "physics_std": physics_std,
            "target_mean": target_mean,
            "target_std": target_std,
        },
        "physics_dim": int(train_p_raw.shape[1]),
        "config": cfg.to_dict(),
    }
    torch.save(checkpoint_payload, cfg.best_checkpoint)

    encoders, _ = make_encoder_bank(cfg, device)
    test_r_raw, test_p_raw, test_y_raw, test_ids = encode_split(
        cfg.test_split, test_records, encoders, cfg, device
    )
    del encoders
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    test_r = (test_r_raw - reservoir_mean) / reservoir_std
    test_p = (test_p_raw - physics_mean) / physics_std

    test_candidate_predictions: List[torch.Tensor] = []
    for family, model in candidate_models:
        if family == "reservoir":
            neural = model.to(device)
            prediction = predict_reservoir(
                neural, test_r, cfg.eval_batch_size, device, target_mean, target_std
            )
            neural.cpu()
        elif family == "joint":
            neural = model.to(device)
            prediction = predict_joint(
                neural, test_r, test_p, cfg.eval_batch_size, device, target_mean, target_std
            )
            neural.cpu()
        else:
            prediction = torch.from_numpy(model.predict(test_p_raw.numpy())).float()
        test_candidate_predictions.append(prediction)

    matrix = torch.stack(test_candidate_predictions, dim=1)
    test_final_prediction = matrix @ weighted_weights

    test_metrics = regression_metrics(test_final_prediction, test_y_raw)
    save_predictions(cfg.test_predictions_csv, test_ids, test_y_raw, test_final_prediction)

    elapsed = time.perf_counter() - start_time
    payload = {
        "package_release": PACKAGE_RELEASE,
        "random_encoder_seeds": list(cfg.random_encoder_seeds),
        "extra_random_encoder_seed": cfg.extra_random_encoder_seed,
        "branch_names": branch_names,
        "reservoir_feature_dim": cfg.reservoir_feature_dim,
        "physics_feature_dim": int(train_p_raw.shape[1]),
        "candidate_names": candidate_names,
        "candidate_val_metrics": candidate_metrics,
        "ensemble_mode": ensemble_mode,
        "weighted_weights": weighted_weights.tolist(),
        "val_final_metrics": val_final_metrics,
        "test_metrics": test_metrics,
        "train_graphs": len(train_ids),
        "val_graphs": len(val_ids),
        "test_graphs": len(test_ids),
        "encoded_vectors_saved": False,
        "elapsed_minutes": elapsed / 60.0,
    }
    write_json(cfg.final_metrics_json, payload)

    print(
        "VAL FINAL : "
        f"mode={ensemble_mode} r={val_final_metrics['pearson']:.4f} "
        f"rho={val_final_metrics['spearman']:.4f} "
        f"RMSE={val_final_metrics['rmse']:.3f} MAE={val_final_metrics['mae']:.3f}",
        flush=True,
    )
    print(
        "TEST FINAL: "
        f"r={test_metrics['pearson']:.4f} rho={test_metrics['spearman']:.4f} "
        f"RMSE={test_metrics['rmse']:.3f} MAE={test_metrics['mae']:.3f} "
        f"R2={test_metrics['r2']:.4f}",
        flush=True,
    )
    print(f"Runtime   : {elapsed / 60.0:.1f} min", flush=True)


if __name__ == "__main__":
    main()
