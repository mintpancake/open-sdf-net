import os
import time
import torch
from torch import nn
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter

from net import SDFNet
from loader import SDFData
from renderer import plot_sdf
from curve import Curve
from util import ensure_dir, read_config

DATA_PATH = '../curves/normalized_data/'
TRAIN_DATA_PATH = '../datasets/train/'
VAL_DATA_PATH = '../datasets/val/'
MODEL_PATH = '../models/'
RES_PATH = '../results/trained_heatmaps/'
MASK_PATH = '../curves/normalized_images/'
LOG_PATH = '../logs/'

CFG = read_config()


def train_sdf(name, delta, device, start_time):
    batch_size = CFG["batch_size"]
    learning_rate = CFG["learning_rate"]
    epochs = CFG["epochs"]
    regularization = CFG["weight_decay"]  # Default: 1e-2
    drop_last = CFG["drop_last"]

    train_data = SDFData(f'{TRAIN_DATA_PATH}{name}.txt')
    val_data = SDFData(f'{VAL_DATA_PATH}{name}.txt')

    train_dataloader = DataLoader(train_data, batch_size=batch_size, shuffle=True, drop_last=drop_last)
    val_dataloader = DataLoader(val_data, batch_size=batch_size, shuffle=True)

    model = SDFNet().to(device)
    if os.path.exists(f'{MODEL_PATH}{name}.pth'):
        model.load_state_dict(torch.load(f'{MODEL_PATH}{name}.pth'))

    loss_fn = nn.L1Loss().to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate, weight_decay=regularization)

    ensure_dir(LOG_PATH)
    writer = SummaryWriter(LOG_PATH)
    total_train_step = 0
    total_val_step = 0

    print('\nTraining SDF model...')

    for t in range(epochs):
        print(f'Epoch {t + 1} ({name})\n-------------------------------')

        # Training loop
        model.train()
        size = len(train_dataloader.dataset)
        for batch, (xy, sdf) in enumerate(train_dataloader):
            xy, sdf = xy.to(device), sdf.to(device)
            pred_sdf = model(xy)
            sdf = torch.reshape(sdf, pred_sdf.shape)
            loss = loss_fn(torch.clamp(pred_sdf, min=-delta, max=delta), torch.clamp(sdf, min=-delta, max=delta))

            loss.backward()
            optimizer.step()
            optimizer.zero_grad()

            if batch % 50 == 0:
                loss_value, current = loss.item(), batch * len(xy)
                print(f'loss: {loss_value:>7f}  [{current:>5d}/{size:>5d}]')

            total_train_step += 1
            if total_train_step % 200 == 0:
                writer.add_scalar(f'SDF training loss ({name})', loss.item(), total_train_step)

        # Evaluation loop
        model.eval()
        size = len(val_dataloader.dataset)
        val_loss = 0

        with torch.no_grad():
            for xy, sdf in val_dataloader:
                xy, sdf = xy.to(device), sdf.to(device)
                pred_sdf = model(xy)
                sdf = torch.reshape(sdf, pred_sdf.shape)
                loss = loss_fn(torch.clamp(pred_sdf, min=-delta, max=delta), torch.clamp(sdf, min=-delta, max=delta))
                val_loss += loss

        val_loss /= size
        end_time = time.time()
        print(f'Test Error: \n'
              f'  Avg loss: {val_loss:>8f} \n'
              f'  Time: {(end_time - start_time):>8f} \n')

        total_val_step += 1
        writer.add_scalar(f'SDF val loss ({name})', val_loss, total_val_step)

    ensure_dir(MODEL_PATH)
    torch.save(model.state_dict(), f'{MODEL_PATH}{name}.pth')
    print(f'Complete training SDF model with {epochs} epochs!')

    writer.close()

    return model


if __name__ == '__main__':
    print('Enter curve name:')
    name = input()
    curve = Curve()
    curve.load(DATA_PATH, name)

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f'Using {device}!')
    start_time = time.time()
    sdf_model = train_sdf(name, curve.delta, device, start_time)

    # Plot results
    print('Plotting results...')
    plot_sdf(sdf_model, curve, name, RES_PATH, MASK_PATH, device=device, is_net=True, show=False)
    print('Done!')
