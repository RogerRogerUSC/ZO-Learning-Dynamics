from os import path
import torch
from tensorboardX import SummaryWriter
from tqdm import tqdm
from util import model_utils
from util import prepare_settings
from util import config_parser
from util import data_utils
from zo_llm.llm_trainer import LLM_trainer
from zo_llm.zo_optim import ZOOptimizer
import argparse


def setup_trainer(
    config: config_parser.MyConfig, device: torch.device, torch_dtype, train_loader
) -> LLM_trainer:
    model_inferences, metrics = prepare_settings.get_model_inferences_and_metrics(
        config.dataset, config
    )
    trainer = LLM_trainer(device=device, dataloader=train_loader, torch_dtype=torch_dtype)
    model = prepare_settings.get_model(
        dataset=config.dataset, model_setting=config, seed=config.seed
    ).to(device)

    zo_optimizer = ZOOptimizer.from_config(config, model=model)
    trainer.set_model_and_criterion(
        model,
        model_inferences.test_inference,
        metrics.test_loss,
        metrics.test_acc,
        zo_optimizer,
    )
    return trainer


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Config File")
    parser.add_argument(
        "--config-path",
        type=str,
        default="text_classification/uniform_opt.yaml",
        help="Path to the YAML configuration file",
    )
    args = parser.parse_args()

    config = config_parser.parse_config(args.config_path)
    device = config.get_device()
    torch_dtype = config.get_torch_dtype()
    train_loader, test_loader = data_utils.get_dataloaders(
        config, config.seed, config.get_hf_model_name()
    )
    # TODO make the trainer deterministic by seed.
    trainer = setup_trainer(config, device, torch_dtype, train_loader)

    if config.log_to_tensorboard:
        assert trainer.model
        tensorboard_sub_folder = "-".join(
            [
                trainer.model.model_name,
                model_utils.get_current_datetime_str(),
            ]
        )
        writer = SummaryWriter(
            path.join(
                "results",
                "zo_llm",
                config.dataset.value,
                config.log_to_tensorboard,
                tensorboard_sub_folder,
            )
        )

    with tqdm(total=config.iterations, desc="Training:") as t, torch.no_grad():
        for ite in range(config.iterations):
            step_loss, step_accuracy = trainer.train_one_step(ite)
            t.set_postfix({"Loss": step_loss, "Acc": step_accuracy})
            t.update(1)

            if config.log_to_tensorboard:
                writer.add_scalar("Loss/train", step_loss, ite)
                writer.add_scalar("Acc/train", step_accuracy, ite)

            if config.eval_iterations != 0 and (ite + 1) % config.eval_iterations == 0:
                eval_loss, eval_accuracy = trainer.eval_model(test_loader)
                if config.log_to_tensorboard:
                    writer.add_scalar("Loss/test", eval_loss, ite)
                    writer.add_scalar("Acc/test", eval_accuracy, ite)
