import sys
from pathlib import Path

# add the parent dir (…/src) to sys.path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))


from config import Config
from data.data_loading import create_dataloaders
from models.model import TransformerDecoder
from models.trainer import Trainer
import torch
import optuna
import logging

try:
    import wandb  # type: ignore
except Exception:
    wandb = None


def _suggest_space(trial: optuna.trial.Trial, size: str) -> None:
    if size == "small":
        d_model = trial.suggest_categorical("d_model", [128, 256])
        n_layers = trial.suggest_int("n_layers", 2, 4)
        d_ff_choices = [512, 1024, 2048]
        batch_choices = [64, 128, 256]
        dropout = trial.suggest_float("dropout", 0.1, 0.3)
        lr = trial.suggest_float("lr", 3e-5, 5e-4, log=True)
    elif size == "medium":
        d_model = trial.suggest_categorical("d_model", [256, 512])
        n_layers = trial.suggest_int("n_layers", 4, 6)
        d_ff_choices = [1024, 2048, 4096]
        batch_choices = [32, 64, 128]
        dropout = trial.suggest_float("dropout", 0.1, 0.3)
        lr = trial.suggest_float("lr", 2e-5, 3e-4, log=True)
    else:  # large
        d_model = trial.suggest_categorical("d_model", [512, 768, 1024])
        n_layers = trial.suggest_int("n_layers", 6, 8, 16)
        d_ff_choices = [2048, 4096, 8192]
        batch_choices = [16, 32, 64]
        dropout = trial.suggest_float("dropout", 0.05, 0.3)
        lr = trial.suggest_float("lr", 1e-5, 2e-4, log=True)

    # heads must divide d_model
    valid_heads = [h for h in [4, 8, 16, 32] if d_model % h == 0 and h <= d_model]
    n_heads = trial.suggest_categorical("n_heads", valid_heads)

    d_ff = trial.suggest_categorical("d_ff", [c for c in d_ff_choices if c >= 2 * d_model and c % d_model == 0] or d_ff_choices)
    batch_size = trial.suggest_categorical("batch_size", batch_choices)

    # set on trial for later readback
    trial.set_user_attr("resolved_heads", n_heads)
    trial.set_user_attr("resolved_d_ff", d_ff)
    trial.set_user_attr("resolved_batch", batch_size)
    trial.set_user_attr("resolved_dropout", dropout)
    trial.set_user_attr("resolved_lr", lr)
    return


def _build_config_from_trial(trial: optuna.trial.Trial, size: str) -> Config:
    config = Config()
    _suggest_space(trial, size)
    config.d_model = trial.params["d_model"]
    config.n_layers = trial.params["n_layers"]
    config.n_heads = trial.params["n_heads"]
    config.d_ff = trial.params["d_ff"]
    config.dropout = trial.params["dropout"]
    config.batch_size = trial.params["batch_size"]
    config.lr = trial.params["lr"]
    # shorter training for search can be configured via user attrs, leave as is for now
    return config


def objective_factory(size: str):
    def objective(trial: optuna.trial.Trial):
        config = _build_config_from_trial(trial, size)

        # Enable wandb if available
        setattr(config, "use_wandb", True)

        # Data
        train_loader, val_loader, test_loader, tokenizer = create_dataloaders(config)
        print(f"Train loader: {len(train_loader)}")
        print(f"Val loader: {len(val_loader)}")
        print(f"Test loader: {len(test_loader)}")
        print(f"Vocab size: {config.vocab_size}")
        print(f"Max length: {config.max_length}")
        print(f"D model: {config.d_model}")
        print(f"N layers: {config.n_layers}")
        # Model
        model = TransformerDecoder(
            vocab_size=config.vocab_size,
            max_len=config.max_length,
            d_model=config.d_model,
            n_layers=config.n_layers,
            n_heads=config.n_heads,
            d_ff=config.d_ff,
            dropout=config.dropout,
            tie_weights=config.tie_weights,
        )

        # Trainer
        trainer = Trainer(model, config, tokenizer)

        run = None
        try:
            if wandb is not None:
                run = wandb.init(
                    project=getattr(config, "wandb_project", "SPT-Transformer-full"),
                    name=f"{size}-trial-{trial.number}",
                    config={
                        "size": size,
                        **trial.params,
                    },
                    reinit=True,
                )

            top1_acc = trainer.fit(train_loader, val_loader)

            # Optional test evaluation per trial
            test_metrics = trainer._validate(test_loader)
            if wandb is not None:
                wandb.log({
                    'test/acc_top1': test_metrics['top_k'][1],
                    'test/acc_top3': test_metrics['top_k'].get(3, 0.0),
                    'test/acc_top5': test_metrics['top_k'].get(5, 0.0),
                    'test/loss': test_metrics['loss'],
                    'test/f1': test_metrics['f1'],
                    'test/precision': test_metrics['precision'],
                    'test/recall': test_metrics['recall'],
                    'test/mrr': test_metrics['mrr'],
                    'test/perplexity': test_metrics['perplexity'],
                })

            if trial.should_prune():
                raise optuna.TrialPruned()

            # minimize 1 - val_top1
            return 1.0 - top1_acc

        except Exception as e:
            logging.warning(f"Trial failed due to {e}")
            raise optuna.TrialPruned()
        finally:
            if wandb is not None and run is not None:
                run.finish()

    return objective


def retrain_and_evaluate_best(size: str, trial: optuna.trial.FrozenTrial):
    config = Config()
    for key, value in trial.params.items():
        setattr(config, key, value)
    setattr(config, "use_wandb", True)

    train_loader, val_loader, test_loader, tokenizer = create_dataloaders(config)
    model = TransformerDecoder(
        vocab_size=config.vocab_size,
        max_len=config.max_length,
        d_model=config.d_model,
        n_layers=config.n_layers,
        n_heads=config.n_heads,
        d_ff=config.d_ff,
        dropout=config.dropout,
        tie_weights=config.tie_weights,
    )

    trainer = Trainer(model, config, tokenizer)

    run = None
    if wandb is not None:
        run = wandb.init(
            project=getattr(config, "wandb_project", "SPT-Transformer-full"),
            name=f"{size}-best-retrain",
            config={"size": size, **trial.params},
            reinit=True,
        )

    best_val = trainer.fit(train_loader, val_loader)
    test_metrics = trainer._validate(test_loader)

    # Save
    out_path = f"best_{size}_model.pth"
    torch.save(model.state_dict(), out_path)

    if wandb is not None:
        try:
            wandb.log({
                'final/val_acc_top1': best_val,
                'final/test_acc_top1': test_metrics['top_k'][1],
                'final/test_acc_top3': test_metrics['top_k'].get(3, 0.0),
                'final/test_acc_top5': test_metrics['top_k'].get(5, 0.0),
                'final/test_loss': test_metrics['loss'],
                'final/test_f1': test_metrics['f1'],
                'final/test_precision': test_metrics['precision'],
                'final/test_recall': test_metrics['recall'],
                'final/test_mrr': test_metrics['mrr'],
                'final/test_perplexity': test_metrics['perplexity'],
            })
            art = wandb.Artifact(f"{size}-model", type="model")
            art.add_file(out_path)
            wandb.log_artifact(art)
        except Exception:
            pass
        finally:
            run.finish() if run is not None else None

    return test_metrics


if __name__ == "__main__":
    # Run three separate studies for small, medium, large
    results = {}
    for size in ["small", "medium", "large"]:
        study = optuna.create_study(
            study_name=f"transformer_optimization_{size}",
            direction="minimize",
            storage="sqlite:///study.db",
            load_if_exists=True,
        )
        study.optimize(objective_factory(size), n_trials=50)

        print(f"Best trial for {size}:")
        best = study.best_trial
        print(f"  Value: {best.value}")
        print("  Params:")
        for key, value in best.params.items():
            print(f"    {key}: {value}")

        # Retrain with best params and evaluate on test
        test_metrics = retrain_and_evaluate_best(size, best)
        results[size] = {
            'best_val_objective': best.value,
            'test_top1': test_metrics['top_k'][1],
            'test_top3': test_metrics['top_k'].get(3, 0.0),
            'test_top5': test_metrics['top_k'].get(5, 0.0),
            'test_loss': test_metrics['loss'],
            'test_f1': test_metrics['f1'],
            'test_precision': test_metrics['precision'],
            'test_recall': test_metrics['recall'],
            'test_mrr': test_metrics['mrr'],
            'test_perplexity': test_metrics['perplexity'],
        }

    print("\n=== Summary (test set) ===")
    for size, m in results.items():
        print(f"{size.title()}: Top1={m['test_top1']:.4f} Top3={m['test_top3']:.4f} Top5={m['test_top5']:.4f} F1={m['test_f1']:.4f} PPL={m['test_perplexity']:.2f}")
