# Running command
#   python src/zo_dynamics/mnist_example.py
import numpy as np
from tqdm import tqdm
import copy
import torch
import torch.nn as nn
import torch.optim as optim
import torchvision

from util.metrics import Metric


class LeNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.feature = nn.Sequential(
            # 1
            nn.Conv2d(
                in_channels=1, out_channels=6, kernel_size=5, stride=1, padding=2
            ),  # 28*28->32*32-->28*28
            nn.Tanh(),
            nn.AvgPool2d(kernel_size=2, stride=2),  # 14*14
            # 2
            nn.Conv2d(in_channels=6, out_channels=16, kernel_size=5, stride=1),  # 10*10
            nn.Tanh(),
            nn.AvgPool2d(kernel_size=2, stride=2),  # 5*5
        )
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(in_features=16 * 5 * 5, out_features=60),
            nn.Tanh(),
            nn.Linear(in_features=60, out_features=42),
            nn.Tanh(),
        )
        self.Bob = nn.Sequential(
            nn.Linear(in_features=42, out_features=10),
        )

    def forward(self, x):
        z = self.classifier(self.feature(x))
        out = self.Bob(z)
        return out, z


def get_ld_metrics(model, model_delay, x, cos=False):
    def _get_adapt_mean(a, a_dly, cos):
        if cos:
            cos_sim = torch.nn.CosineSimilarity(dim=1, eps=1e-6)
            return torch.mean(cos_sim(a_dly, (a - a_dly))).detach().cpu()
        else:
            return torch.mean(torch.diag(torch.mm(a_dly, (a - a_dly).T))).detach().cpu()

    p, z = model(x)
    p_dly, z_dly = model_delay(x)
    z_adapt = _get_adapt_mean(z, z_dly, cos)
    p_adapt = _get_adapt_mean(p, p_dly, cos)
    return z_adapt, p_adapt


if __name__ == "__main__":
    FIGSIZE = 28
    EPOCHS = 100
    HID_SIZE = 128
    K_CLAS = 10
    X_DIM = FIGSIZE * FIGSIZE
    LR = 1e-3
    DEVICE = "mps"

    train_transform = torchvision.transforms.Compose(
        [
            torchvision.transforms.Resize(FIGSIZE),
            torchvision.transforms.ToTensor(),
            torchvision.transforms.Normalize((0.1307,), (0.3081,)),
        ]
    )
    np.random.seed(12345)
    rnd_perm = np.random.permutation(np.arange(0, 60000))
    full_trainset = torchvision.datasets.MNIST(
        "./data", train=True, download=True, transform=train_transform
    )
    train_subset = torch.utils.data.Subset(full_trainset, list(rnd_perm[:20000]))
    test_subset = torch.utils.data.Subset(full_trainset, list(rnd_perm[20000:40000]))

    seed_loader = torch.utils.data.DataLoader(train_subset, batch_size=2000, shuffle=False)
    train_loader = torch.utils.data.DataLoader(train_subset, batch_size=64, shuffle=True)
    test_loader = torch.utils.data.DataLoader(test_subset, batch_size=2000, shuffle=False)

    model = LeNet()
    model.to(DEVICE)
    optimizer = optim.SGD(model.parameters(), lr=LR, momentum=0.9, weight_decay=5e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=EPOCHS, eta_min=1e-6)

    results = {
        "train_loss": [],
        "test_loss": [],
        "test_acc": [],
        "z_adapt": [],
        "p_adapt": [],
        "z_adapt_cos": [],
        "p_adapt_cos": [],
    }

    cnt = 0
    track_gap = 500
    with tqdm(total=EPOCHS, desc="Training:") as t:
        for g in range(EPOCHS):
            model.train(True)
            for data_iter_step, (x, y) in enumerate(train_loader):
                cnt += 1
                model.train()
                if cnt % track_gap == 1:
                    model_delay = copy.deepcopy(model)
                # x, y = x.float().reshape(-1,FIGSIZE*FIGSIZE).cuda(), y.long().cuda()
                x, y = x.float().to(DEVICE), y.long().to(DEVICE)
                outputs, z = model(x)
                loss = nn.CrossEntropyLoss()(outputs, y)
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                results["train_loss"].append(loss.item())
                # ----------- Track adaptations
                if cnt % track_gap == 1:
                    z_adapt, p_adapt = get_ld_metrics(model, model_delay, x, cos=False)
                    results["z_adapt"].append(z_adapt)
                    results["p_adapt"].append(p_adapt)
                    z_adapt, p_adapt = get_ld_metrics(model, model_delay, x, cos=True)
                    results["z_adapt_cos"].append(z_adapt)
                    results["p_adapt_cos"].append(p_adapt)

            # --------- Evaluation for test loss
            model.eval()
            with torch.no_grad():
                test_loss = Metric("Test loss")
                test_acc = Metric("Test acc")
                for data_iter_step, (x, y) in enumerate(test_loader):
                    x, y = x.float().to(DEVICE), y.long().to(DEVICE)
                    outputs, z = model(x)
                    loss = nn.CrossEntropyLoss()(outputs, y)
                    test_loss.update(loss.item())
                    test_acc.update((outputs.argmax(dim=1) == y).sum().item())

                results["test_loss"].append(test_loss.avg)
                results["test_acc"].append(test_acc.avg / 2000)  # test batch size
                t.set_postfix({"Loss": test_loss.avg, "Acc": test_acc.avg / 2000})
            t.update(1)

    torch.save(model.state_dict(), "checkpoints/mnist_example.pth")
    scheduler.step()
