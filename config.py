from __future__ import annotations

from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Dict, Tuple


PACKAGE_RELEASE = "RAVEN"


@dataclass
class RunConfig:
    pdbbind_root: str = "/usr/share/ollama/.ollama/zqydisk/Finals/data_bank/PL-heterograph-PDBbind-PhysGates"
    output_root: str = str(Path(__file__).resolve().parent / "output")

    device: str = "cuda"
    cpu_threads: int = 8
    allow_tf32: bool = True
    pin_memory: bool = True

    random_encoder_seeds: Tuple[int, ...] = (
        20260720,
        20260721,
        20260722,
        20260723,
        20260724,
        20260725,
        20260727,
        20260728,
        20260729,
        20260730,
        20260731,
        20260732,
        20260733,
        20260734,
        20260735,
        20260736,
        20260737,
        20260738,
        20260739,
        20260740,
        20260741,
        20260742,
        20260743,
        20260744,
        20260745,
        20260746,
        20260747,
        20260748,
        20260749,
        20260750,
        20260751,
        20260752,
    )
    extra_random_encoder_seed: int = 20260726
    reservoir_mlp_seeds: Tuple[int, ...] = (20260731, 20260732, 20260733)
    joint_mlp_seeds: Tuple[int, ...] = (20260741, 20260742, 20260743)
    tree_seeds: Tuple[int, ...] = (20260751, 20260752, 20260753)

    train_split: str = "Train"
    val_split: str = "Val"
    test_split: str = "Test"
    graph_batch_size: int = 128

    expected_node_dim: int = 41
    hidden_dim: int = 192
    edge_hidden_dim: int = 96
    message_layers: int = 4
    rbf_bins: int = 16
    representation_dim: int = 384

    random_structure_dim: int = 768
    random_ligand_dim: int = 384
    extra_random_global_dim: int = 768

    structure_adapter_hidden: int = 512
    structure_adapter_output: int = 256
    ligand_adapter_hidden: int = 256
    ligand_adapter_output: int = 128
    extra_random_adapter_hidden: int = 512
    extra_random_adapter_output: int = 256
    physics_adapter_hidden: int = 1024
    physics_adapter_output: int = 256
    adapter_dropouts: Tuple[float, float] = (0.15, 0.10)

    fusion_hidden_dims: Tuple[int, ...] = (2048, 512, 128)
    fusion_dropouts: Tuple[float, ...] = (0.35, 0.30, 0.25)
    branch_dropout_probability: float = 0.08
    feature_noise_std: float = 0.008

    train_batch_size: int = 512
    eval_batch_size: int = 1024
    max_epochs: int = 900
    early_stop_patience: int = 75
    learning_rate: float = 4.0e-4
    weight_decay: float = 1.5e-4
    grad_clip_norm: float = 5.0
    huber_beta: float = 0.5
    pearson_loss_weight: float = 0.08
    scheduler_factor: float = 0.5
    scheduler_patience: int = 12
    minimum_learning_rate: float = 1.0e-6

    extra_trees_estimators: int = 700
    extra_trees_max_features: float = 0.72
    extra_trees_min_samples_leaf: int = 2
    extra_trees_n_jobs: int = -1

    ensemble_weight_steps: int = 800
    ensemble_weight_lr: float = 0.05
    ensemble_weight_l2: float = 0.015

    # Count active random encoders.
    @property
    def random_count(self) -> int:
        return len(self.random_encoder_seeds)

    # Build reservoir branch dimensions.
    @property
    def reservoir_branch_dims(self) -> Tuple[int, ...]:
        return tuple(
            [self.random_structure_dim, self.random_ligand_dim] * self.random_count
            + [self.extra_random_global_dim]
        )

    # Compute total reservoir dimension.
    @property
    def reservoir_feature_dim(self) -> int:
        return sum(self.reservoir_branch_dims)

    # Resolve checkpoint path.
    @property
    def best_checkpoint(self) -> Path:
        return Path(self.output_root) / "checkpoint_best_physics_fusion_ensemble.pt"

    # Resolve training-history path.
    @property
    def history_csv(self) -> Path:
        return Path(self.output_root) / "training_history_all_neural_heads.csv"

    # Resolve candidate-metrics path.
    @property
    def candidate_metrics_csv(self) -> Path:
        return Path(self.output_root) / "candidate_val_metrics.csv"

    # Resolve validation-prediction path.
    @property
    def val_predictions_csv(self) -> Path:
        return Path(self.output_root) / "val_predictions_final.csv"

    # Resolve test-prediction path.
    @property
    def test_predictions_csv(self) -> Path:
        return Path(self.output_root) / "test_predictions_final.csv"

    # Resolve final-metrics path.
    @property
    def final_metrics_json(self) -> Path:
        return Path(self.output_root) / "final_metrics.json"

    # Resolve config-output path.
    @property
    def resolved_config_json(self) -> Path:
        return Path(self.output_root) / "resolved_config.json"

    # Export resolved configuration.
    def to_dict(self) -> Dict[str, object]:
        value = asdict(self)
        value.update({
            "package_release": PACKAGE_RELEASE,
            "random_count": self.random_count,
            "reservoir_branch_dims": list(self.reservoir_branch_dims),
            "reservoir_feature_dim": self.reservoir_feature_dim,
            "best_checkpoint": str(self.best_checkpoint),
            "encoding_storage": "RAM only; no encoded vector files are written",
            "selection_contract": "Val selects neural checkpoints and learns final softmax weights; Test is encoded afterwards",
            "encoder_contract": "thirty-two dual-output frozen random encoders plus one frozen random protein+ligand branch",
        })
        return value


CONFIG = RunConfig()
