from os import path
import torch
from tensorboardX import SummaryWriter
from tqdm import tqdm
from util import model_utils
from util import prepare_settings
from util import config_parser
from util import data_utils
from zo_llm.llm_trainer import LLM_trainer


def setup_trainer(
    config: config_parser.MyConfig, device: torch.device, train_loader
) -> LLM_trainer:
    model_inferences, metrics = prepare_settings.get_model_inferences_and_metrics(
        config.dataset, config
    )
    trainer = LLM_trainer(device=device, dataloader=train_loader)
    model = prepare_settings.get_model(
        dataset=config.dataset, model_setting=config, seed=config.seed
    ).to(device)
    optimizer = prepare_settings.get_optimizer(
        model=model, dataset=config.dataset, optimizer_setting=config
    )
    grad_estimator = prepare_settings.get_gradient_estimator(
        model=model,
        device=device,
        config=config,
    )
    trainer.set_model_and_criterion(
        model,
        model_inferences.test_inference,
        metrics.test_loss,
        metrics.test_acc,
        optimizer,
        grad_estimator,
    )
    return trainer


if __name__ == "__main__":
    config = config_parser.parse_config("text_classification.yaml")
    device = torch.device(config.device)
    train_loader, test_loader = data_utils.get_dataloaders(
        config, config.seed, config.get_hf_model_name()
    )
    trainer = setup_trainer(config, device, train_loader)

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
