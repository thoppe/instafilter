from tqdm import tqdm
from pathlib import Path
from model import ColorNet, ColorizedDataset
import torch
from torch.utils.data import DataLoader

device = "cuda"
loss_func = torch.optim.AdamW


def train_image_pair(
    f_source,
    f_target,
    f_save_model,
    batch_size=2 ** 10,
    n_epochs=30,
    max_learning_rate=0.01,
):

    data = ColorizedDataset(f_source, f_target, device=device)
    train_loader = DataLoader(data, batch_size=batch_size, shuffle=True)

    net = ColorNet()
    criterion = torch.nn.L1Loss()
    optimizer = loss_func(net.parameters())

    scheduler = torch.optim.lr_scheduler.OneCycleLR(
        optimizer,
        max_lr=max_learning_rate,
        steps_per_epoch=len(train_loader),
        epochs=n_epochs,
    )

    net.to(device)
    net.train()

    for epoch in tqdm(range(n_epochs)):
        # monitor training loss
        train_loss = 0.0

        for data, target in train_loader:

            optimizer.zero_grad()

            output = net(data)
            loss = criterion(output, target)
            loss.backward()
            optimizer.step()
            scheduler.step()

            # update running training loss
            train_loss += loss.item() * data.size(0)

        train_loss = train_loss / len(train_loader.dataset)
        print(f"Epoch: {epoch+1} \tTraining Loss: {train_loss:.6f}")

    torch.save(net.state_dict(), f_save_model)


if __name__ == "__main__":

    f_source = "samples/Normal.jpg"

    for f_target in Path('samples').glob('*.jpg'):
        
        f_model = Path("models") / f_target.name.replace(".jpg", ".pt")
                
        if f_target.name == f_source:
            continue
        
        if f_model.exists():
            continue
        
        print("Training", f_model)
        
        train_image_pair("samples/Normal.jpg", f_target, f_model)
