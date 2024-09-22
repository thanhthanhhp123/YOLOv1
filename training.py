import torch
import torch.optim as optim
import torch.nn as nn

from model import YOLOv1
from loss import YOLOLoss


device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

model = YOLOv1(S = 7, B = 2, C = 20).to(device)
loss = YOLOLoss(S = 7, B = 2, C = 20).to(device)

optim = optim.SGD(model.parameters(), lr = 1e-3, momentum = 0.9, weight_decay = 5e-4)

lr = 1e-3
for epoch in range(135):
    if epoch == 0:
        lr = 1e-3
    elif epoch == 75:
        lr = 1e-2
    elif epoch == 105:
        lr = 1e-3
    elif epoch == 135:
        lr = 1e-4

    for param_group in optim.param_groups:
        param_group['lr'] = lr

    # Training loop
    for i, (images, targets) in enumerate(train_loader):
        images = images.to(device)
        targets = targets.to(device)

        preds = model(images)
        loss_value = loss(preds, targets)

        optim.zero_grad()
        loss_value.backward()
        optim.step()

        if i % 10 == 0:
            print(f'Epoch {epoch}, step {i}, loss: {loss_value.item()}')

    # Save model
    torch.save(model.state_dict(), f'checkpoints/yolov1_epoch{epoch}.pth') 