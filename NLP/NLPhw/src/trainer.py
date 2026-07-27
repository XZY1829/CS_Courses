import os
import sys
from contextlib import contextmanager, nullcontext
from typing import Any, Dict, Iterable, Iterator, Optional, Tuple, TypeVar

import torch
from torch.cuda.amp import GradScaler, autocast as cuda_autocast
from torch.nn.utils import clip_grad_norm_
from tqdm import tqdm

from src.metrics import compute_ner_metrics, ids_to_tag_sequences, ner_classification_report
from src.utils import format_metrics, save_json

T = TypeVar("T")


def _tqdm(iterable: Iterable[T], desc: str, **_kw: Any) -> tqdm:
    # disable=False keeps bars visible in wrapped/redirected terminals.
    return tqdm(
        iterable,
        desc=desc,
        file=sys.stderr,
        disable=False,
        mininterval=0.3,
        leave=True,
        dynamic_ncols=True,
        **_kw,
    )


def _clone_state_dict_to_cpu(state_dict: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
    return {k: v.detach().cpu().clone() for k, v in state_dict.items()}


def _amp_autocast(enabled: bool, device_type: str):
    if not enabled:
        return nullcontext()
    if hasattr(torch, "amp") and hasattr(torch.amp, "autocast"):
        return torch.amp.autocast(device_type=device_type, enabled=True)
    return cuda_autocast(enabled=True)


class ModelEMA:
    def __init__(self, model: torch.nn.Module, decay: float) -> None:
        if not (0.0 < decay < 1.0):
            raise ValueError("ema_decay must be in (0, 1).")
        self.decay = float(decay)
        self.shadow: Dict[str, torch.Tensor] = {
            k: v.detach().clone() for k, v in model.state_dict().items()
        }

    @torch.no_grad()
    def update(self, model: torch.nn.Module) -> None:
        for key, value in model.state_dict().items():
            source = value.detach()
            if key not in self.shadow:
                self.shadow[key] = source.clone()
                continue
            if torch.is_floating_point(source):
                self.shadow[key].mul_(self.decay).add_(source, alpha=1.0 - self.decay)
            else:
                self.shadow[key] = source.clone()

    @torch.no_grad()
    def copy_to(self, model: torch.nn.Module) -> None:
        model_state = model.state_dict()
        for key, value in model_state.items():
            shadow_val = self.shadow.get(key)
            if shadow_val is None:
                continue
            value.copy_(shadow_val.to(device=value.device, dtype=value.dtype))

    def state_dict(self) -> Dict[str, torch.Tensor]:
        return {k: v.detach().cpu().clone() for k, v in self.shadow.items()}


class Trainer:
    def __init__(
        self,
        model: torch.nn.Module,
        optimizer: torch.optim.Optimizer,
        device: torch.device,
        id2label: Dict[int, str],
        grad_clip: float = 5.0,
        aux_loss_weight: float = 0.0,
        scheduler: Optional[torch.optim.lr_scheduler.LRScheduler] = None,
        use_amp: bool = False,
        accumulation_steps: int = 1,
        ema_decay: float = 0.0,
        use_ema_for_eval: bool = False,
    ) -> None:
        self.model = model
        self.optimizer = optimizer
        self.device = device
        self.id2label = id2label
        self.grad_clip = float(grad_clip)
        self.aux_loss_weight = float(aux_loss_weight)
        self.scheduler = scheduler
        self.use_amp = bool(use_amp and device.type == "cuda")
        self.accumulation_steps = max(1, int(accumulation_steps))
        if self.use_amp:
            if hasattr(torch, "amp") and hasattr(torch.amp, "GradScaler"):
                self.scaler = torch.amp.GradScaler("cuda", enabled=True)
            else:
                self.scaler = GradScaler(enabled=True)
        else:
            self.scaler = None
        self.ema = ModelEMA(model, decay=float(ema_decay)) if ema_decay > 0.0 else None
        self.use_ema_for_eval = bool(use_ema_for_eval and self.ema is not None)
        self.global_step = 0

    def _move_batch(self, batch: Dict[str, Any]) -> Dict[str, Any]:
        moved = {
            "word_ids": batch["word_ids"].to(self.device),
            "tag_ids": batch["tag_ids"].to(self.device),
            "mask": batch["mask"].to(self.device),
        }
        if isinstance(batch["char_ids"], torch.Tensor):
            moved["char_ids"] = batch["char_ids"].to(self.device)
        else:
            moved["char_ids"] = None
        if isinstance(batch.get("pos_ids"), torch.Tensor):
            moved["pos_ids"] = batch["pos_ids"].to(self.device)
        else:
            moved["pos_ids"] = None
        if isinstance(batch.get("chunk_ids"), torch.Tensor):
            moved["chunk_ids"] = batch["chunk_ids"].to(self.device)
        else:
            moved["chunk_ids"] = None
        return moved

    @contextmanager
    def _ema_eval_scope(self) -> Iterator[None]:
        if self.ema is None:
            yield
            return
        backup_state = {k: v.detach().clone() for k, v in self.model.state_dict().items()}
        self.ema.copy_to(self.model)
        try:
            yield
        finally:
            self.model.load_state_dict(backup_state, strict=True)

    def train_one_epoch(self, train_loader) -> float:
        self.model.train()
        self.optimizer.zero_grad(set_to_none=True)
        total_loss = 0.0
        progress = _tqdm(train_loader, desc="Train")
        num_batches = max(1, len(train_loader))

        for step_idx, batch in enumerate(progress, start=1):
            batch = self._move_batch(batch)

            with _amp_autocast(enabled=self.use_amp, device_type=self.device.type):
                loss_dict = self.model(
                    word_ids=batch["word_ids"],
                    mask=batch["mask"],
                    tags=batch["tag_ids"],
                    char_ids=batch["char_ids"],
                    pos_ids=batch["pos_ids"],
                    chunk_ids=batch["chunk_ids"],
                    aux_loss_weight=self.aux_loss_weight,
                )
                loss = loss_dict["total_loss"] / self.accumulation_steps

            if self.use_amp:
                if self.scaler is None:
                    raise RuntimeError("AMP scaler is not initialized.")
                self.scaler.scale(loss).backward()
            else:
                loss.backward()

            should_step = (step_idx % self.accumulation_steps == 0) or (step_idx == num_batches)
            if should_step:
                did_optimizer_step = False
                if self.use_amp:
                    if self.scaler is None:
                        raise RuntimeError("AMP scaler is not initialized.")
                    if self.grad_clip > 0:
                        self.scaler.unscale_(self.optimizer)
                        clip_grad_norm_(self.model.parameters(), self.grad_clip)
                    prev_scale = float(self.scaler.get_scale())
                    self.scaler.step(self.optimizer)
                    self.scaler.update()
                    did_optimizer_step = float(self.scaler.get_scale()) >= prev_scale
                else:
                    if self.grad_clip > 0:
                        clip_grad_norm_(self.model.parameters(), self.grad_clip)
                    self.optimizer.step()
                    did_optimizer_step = True
                self.optimizer.zero_grad(set_to_none=True)

                if did_optimizer_step and self.scheduler is not None:
                    self.scheduler.step()
                if did_optimizer_step and self.ema is not None:
                    self.ema.update(self.model)
                if did_optimizer_step:
                    self.global_step += 1

            total_loss += float(loss_dict["total_loss"].item())
            lr = float(self.optimizer.param_groups[0]["lr"])
            progress.set_postfix(
                total=f"{loss_dict['total_loss'].item():.4f}",
                ner=f"{loss_dict['ner_loss'].item():.4f}",
                pos=f"{loss_dict['pos_loss'].item():.4f}",
                chunk=f"{loss_dict['chunk_loss'].item():.4f}",
                lr=f"{lr:.2e}",
            )

        return total_loss / num_batches

    @torch.no_grad()
    def evaluate(self, data_loader) -> Tuple[Dict[str, float], str]:
        self.model.eval()
        all_preds = []
        all_golds = []
        all_masks = []

        for batch in _tqdm(data_loader, desc="Eval"):
            batch = self._move_batch(batch)
            pred_ids = self.model.decode(
                word_ids=batch["word_ids"],
                mask=batch["mask"],
                char_ids=batch["char_ids"],
            )
            gold_ids = batch["tag_ids"].detach().cpu().tolist()
            mask = batch["mask"].detach().cpu().tolist()

            max_len = len(mask[0])
            padded_preds = [seq + [0] * (max_len - len(seq)) for seq in pred_ids]

            all_preds.extend(padded_preds)
            all_golds.extend(gold_ids)
            all_masks.extend(mask)

        pred_tags, gold_tags = ids_to_tag_sequences(
            pred_ids=all_preds, gold_ids=all_golds, mask=all_masks, id2label=self.id2label
        )
        metrics = compute_ner_metrics(pred_tags=pred_tags, gold_tags=gold_tags)
        report = ner_classification_report(pred_tags=pred_tags, gold_tags=gold_tags)
        return metrics, report

    def fit(
        self,
        train_loader,
        dev_loader,
        epochs: int,
        patience: int,
        output_dir: str,
        run_config: Dict[str, Any],
    ) -> Dict[str, Any]:
        best_f1 = -1.0
        best_epoch = -1
        early_stop_counter = 0
        history = []

        print(f"Starting training: up to {epochs} epochs, patience={patience}.", flush=True)
        print(
            f"Trainer options: amp={self.use_amp}, accumulation_steps={self.accumulation_steps}, "
            f"scheduler={'on' if self.scheduler is not None else 'off'}, "
            f"ema={'on' if self.ema is not None else 'off'}, "
            f"ema_eval={'on' if self.use_ema_for_eval else 'off'}.",
            flush=True,
        )
        for epoch in range(1, epochs + 1):
            print(f"Epoch {epoch}/{epochs} - running train + dev...", flush=True)
            train_loss = self.train_one_epoch(train_loader)
            if self.use_ema_for_eval:
                with self._ema_eval_scope():
                    dev_metrics, dev_report = self.evaluate(dev_loader)
            else:
                dev_metrics, dev_report = self.evaluate(dev_loader)

            current_lr = float(self.optimizer.param_groups[0]["lr"])
            epoch_record = {
                "epoch": epoch,
                "train_loss": train_loss,
                "dev_precision": dev_metrics["precision"],
                "dev_recall": dev_metrics["recall"],
                "dev_f1": dev_metrics["f1"],
                "lr": current_lr,
                "global_step": self.global_step,
            }
            history.append(epoch_record)

            print(
                f"[Epoch {epoch:02d}] "
                f"loss={train_loss:.4f} "
                f"lr={current_lr:.2e} "
                f"{format_metrics(dev_metrics)}",
                flush=True,
            )

            if dev_metrics["f1"] > best_f1:
                best_f1 = dev_metrics["f1"]
                best_epoch = epoch
                early_stop_counter = 0
                checkpoint = {
                    "epoch": epoch,
                    "model_state_dict": _clone_state_dict_to_cpu(self.model.state_dict()),
                    "optimizer_state_dict": self.optimizer.state_dict(),
                    "scheduler_state_dict": self.scheduler.state_dict() if self.scheduler is not None else None,
                    "scaler_state_dict": self.scaler.state_dict() if self.scaler is not None else None,
                    "dev_metrics": dev_metrics,
                    "dev_report": dev_report,
                    "config": run_config,
                    "global_step": self.global_step,
                    "best_model_variant": "ema" if self.use_ema_for_eval else "raw",
                }
                if self.ema is not None:
                    checkpoint["ema_state_dict"] = self.ema.state_dict()
                    with self._ema_eval_scope():
                        checkpoint["ema_model_state_dict"] = _clone_state_dict_to_cpu(
                            self.model.state_dict()
                        )
                torch.save(checkpoint, os.path.join(output_dir, "best_model.pt"))
            else:
                early_stop_counter += 1
                if early_stop_counter >= patience:
                    print(f"Early stopping triggered at epoch {epoch}.", flush=True)
                    break

        save_json(os.path.join(output_dir, "history.json"), {"history": history})
        return {"best_f1": best_f1, "best_epoch": best_epoch}
