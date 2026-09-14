import argparse
import csv
import datetime
import os
import sys
from pathlib import Path

import torch
from configmypy import ArgparseConfig, ConfigPipeline, YamlConfig
from torch.utils.data import DataLoader

from neuralop import get_model
from neuralop.data.datasets.Dynamic2DFlood import Dynamic2DFlood
from neuralop.models import DrainageAdapter
from utils.checkpoint_utils import load_state_dict_with_optional_lifting_expand
from utils.torch_utils import select_device


ROOT = Path(__file__).resolve().parent


def parse_args():
    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument("--config", type=str, default="region1_drainage_adapter.yaml")
    parser.add_argument("--device", type=str, default="0")
    parser.add_argument("--epochs", type=int, default=None)
    parser.add_argument("--hidden_channels", type=int, default=32)
    parser.add_argument("--dry_run", action="store_true")
    parser.add_argument("--dry_run_steps", type=int, default=1)
    parser.add_argument("--dry_run_samples", type=int, default=1)
    args, remaining = parser.parse_known_args()
    sys.argv = [sys.argv[0]] + remaining
    return args


def conf_get(section, name, default=None):
    try:
        return getattr(section, name)
    except Exception:
        try:
            return section[name]
        except Exception:
            return default


def parse_feature_list(value):
    if value is None:
        return []
    if isinstance(value, str):
        return [item.strip() for item in value.split(",") if item.strip()]
    return [str(item).strip() for item in value if str(item).strip()]


def move_to_device(obj, device):
    if torch.is_tensor(obj):
        return obj.to(device)
    if isinstance(obj, dict):
        return {k: move_to_device(v, device) for k, v in obj.items()}
    return obj


def without_drainage_features(inputs):
    return {k: v for k, v in inputs.items() if k != "drainage_features"}


def wmse_loss(pred, target, threshold_mm=30.0, wet_weight=20.0):
    weights = torch.ones_like(target)
    weights = torch.where(target > threshold_mm, weights * wet_weight, weights)
    return torch.mean(weights * (pred - target) ** 2)


def metrics(pred, target):
    err = pred - target
    return {
        "mae_mm": float(torch.mean(torch.abs(err)).item()),
        "rmse_mm": float(torch.sqrt(torch.mean(err ** 2)).item()),
        "peak_pred_mm": float(torch.max(pred).item()),
        "peak_target_mm": float(torch.max(target).item()),
        "peak_error_mm": float((torch.max(pred) - torch.max(target)).item()),
    }


def evaluate(base_model, adapter, loader, device, max_samples=None, steps=None):
    base_model.eval()
    adapter.eval()
    rows = []
    with torch.no_grad():
        for idx, (inputs, target, event_name) in enumerate(loader):
            if max_samples is not None and idx >= max_samples:
                break
            inputs = move_to_device(inputs, device)
            target = target.to(device).float()
            if steps is not None:
                target = target[:, :, :, :, :steps]
                inputs["rainfall"] = inputs["rainfall"][:, :, :, :, :steps]
                inputs["cumsum_rainfall"] = inputs["cumsum_rainfall"][:, :, :, :, :steps]
            b, _, h, w, t = target.shape
            base_pred = base_model(0, without_drainage_features(inputs), device, init_shape=(b, h, w, t), cache_key=None)
            adapted = adapter(base_pred, inputs)
            row = {"event": event_name[0] if isinstance(event_name, (list, tuple)) else str(event_name)}
            row.update({f"base_{k}": v for k, v in metrics(base_pred, target).items()})
            row.update({f"adapter_{k}": v for k, v in metrics(adapted, target).items()})
            rows.append(row)
    return rows


def main():
    args = parse_args()
    os.chdir(ROOT)
    pipe = ConfigPipeline([
        YamlConfig(args.config, config_name="default", config_folder="./configs"),
        ArgparseConfig(infer_types=True, config_name=None, config_file=None),
        YamlConfig(config_folder="../configs"),
    ])
    config = pipe.read_conf()
    device = select_device(args.device, 1)
    drainage_features = parse_feature_list(conf_get(config.data, "drainage_features", None))

    train_dataset = Dynamic2DFlood(
        data_root=config.data.data_root,
        split="train",
        location=config.data.train_location,
        train_list=config.data.train_list,
        test_list=config.data.test_list,
        wall_height=config.tfno2d.wall_height,
        drainage_features=drainage_features,
    )
    test_dataset = Dynamic2DFlood(
        data_root=config.data.data_root,
        split="test",
        location=config.data.train_location,
        train_list=config.data.train_list,
        test_list=config.data.test_list,
        wall_height=config.tfno2d.wall_height,
        drainage_features=drainage_features,
    )
    train_loader = DataLoader(train_dataset, batch_size=1, shuffle=True, num_workers=config.data.num_workers_train)
    test_loader = DataLoader(test_dataset, batch_size=1, shuffle=False, num_workers=config.data.num_workers_test)

    base_model = get_model(config).to(device)
    checkpoint = os.path.join(config.finetune.pretrained_dir, config.finetune.state_dict_name)
    info = load_state_dict_with_optional_lifting_expand(base_model, checkpoint, map_location=device)
    print(f"Loaded frozen base LarNO: loaded={len(info['loaded'])}, skipped={info['skipped'][:5]}")
    for param in base_model.parameters():
        param.requires_grad = False
    base_model.eval()

    adapter = DrainageAdapter(
        drainage_channels=len(drainage_features),
        hidden_channels=args.hidden_channels,
        nonpositive_residual=True,
    ).to(device)
    optimizer = torch.optim.Adam(adapter.parameters(), lr=float(config.opt.lr_max), weight_decay=float(config.opt.weight_decay))
    epochs = args.epochs or int(config.opt.n_epochs)

    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    exp_dir = Path(config.data.exp_root) / f"larno_drain_adapter_v1_itzi_sink_{timestamp}"
    exp_dir.mkdir(parents=True, exist_ok=True)

    if args.dry_run:
        rows = evaluate(
            base_model,
            adapter,
            test_loader,
            device,
            max_samples=args.dry_run_samples,
            steps=args.dry_run_steps,
        )
        print("Dry-run metrics:", rows[:1])
        return 0

    log_rows = []
    for epoch in range(epochs):
        adapter.train()
        losses = []
        for idx, (inputs, target, event_name) in enumerate(train_loader):
            inputs = move_to_device(inputs, device)
            target = target.to(device).float()
            b, _, h, w, t = target.shape
            with torch.no_grad():
                base_pred = base_model(
                    0, without_drainage_features(inputs), device, init_shape=(b, h, w, t), cache_key=None
                )
            adapted = adapter(base_pred, inputs)
            loss = wmse_loss(adapted[:, 0], target[:, 0])
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            optimizer.step()
            losses.append(float(loss.item()))
        mean_loss = sum(losses) / max(len(losses), 1)
        print(f"epoch={epoch+1}/{epochs} train_wmse={mean_loss:.6f}")
        log_rows.append({"epoch": epoch + 1, "train_wmse": mean_loss})

    torch.save(adapter.state_dict(), exp_dir / "drainage_adapter_state_dict.pt")
    eval_rows = evaluate(base_model, adapter, test_loader, device)
    with (exp_dir / "train_log.csv").open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(log_rows[0].keys()))
        writer.writeheader()
        writer.writerows(log_rows)
    with (exp_dir / "test_metrics.csv").open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(eval_rows[0].keys()))
        writer.writeheader()
        writer.writerows(eval_rows)
    print(f"Saved DrainAdapter experiment to {exp_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
