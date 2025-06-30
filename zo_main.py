from os import path
import torch
from tensorboardX import SummaryWriter
from tqdm import tqdm
from util import model_helpers
from exp_helper import prepare_settings
from exp_helper.cli_parser import (
    GeneralSetting,
    DeviceSetting,
    DataSetting,
    ModelSetting,
    OptimizerSetting,
    RGESetting,
)
from exp_helper.device import use_device
from exp_helper.data import get_dataloaders
from util.llm_trainer import LLM_trainer
from util.llm_trainer.LLM_trainer import eval_model


class CliSetting(
    GeneralSetting,
    DeviceSetting,
    DataSetting,
    ModelSetting,
    OptimizerSetting,
    RGESetting,
):
    """
    This is a replacement for regular argparse module.
    We used a third party library pydantic_setting to make command line interface easier to manage.
    Example:
    if __name__ == "__main__":
        args = CliSetting()

    args will have all parameters defined by all components.
    """

    pass


def setup_trainer(args: CliSetting, device: torch.device, train_loader) -> LLM_trainer:
    model_inferences, metrics = prepare_settings.get_model_inferences_and_metrics(
        args.dataset, args.model_setting
    )
    trainer = LLM_trainer(device=device)
    model = prepare_settings.get_model(
        dataset=args.dataset, model_setting=args.model_setting, seed=args.seed
    ).to(device)
    optimizer = prepare_settings.get_optimizer(
        model=model, dataset=args.dataset, optimizer_setting=args.optimizer_setting
    )
    grad_estimator = prepare_settings.get_gradient_estimator(
        model=model,
        device=device,
        rge_setting=args.rge_setting,
        model_setting=args.model_setting,
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
    args = CliSetting()
    print(args)
    device = use_device(args.device_setting)
    train_loader, test_loader = get_dataloaders(
        args.data_setting, args.seed, args.get_hf_model_name()
    )
    trainer = setup_trainer(args, device, train_loader)

    if args.log_to_tensorboard:
        assert trainer.model
        tensorboard_sub_folder = "-".join(
            [
                trainer.model.model_name,
                model_helpers.get_current_datetime_str(),
            ]
        )
        writer = SummaryWriter(
            path.join(
                "results",
                "zo_llm",
                args.dataset.value,
                args.log_to_tensorboard,
                tensorboard_sub_folder,
            )
        )

    with tqdm(total=args.iterations, desc="Training:") as t, torch.no_grad():
        for ite in range(args.iterations):
            step_loss, step_accuracy = trainer.train_one_step(ite)
            t.set_postfix({"Loss": step_loss, "Acc": step_accuracy})
            t.update(1)

            if args.log_to_tensorboard:
                writer.add_scalar("Loss/train", step_loss, ite)
                writer.add_scalar("Acc/train", step_accuracy, ite)

            if args.eval_iterations != 0 and (ite + 1) % args.eval_iterations == 0:
                eval_loss, eval_accuracy = eval_model(trainer.model, test_loader)
                if args.log_to_tensorboard:
                    writer.add_scalar("Loss/test", eval_loss, ite)
                    writer.add_scalar("Acc/test", eval_accuracy, ite)
