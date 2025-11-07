import torch


class BaseLoss(torch.nn.Module):
    def __init__(self, *args, **kwargs):
        super(BaseLoss, self).__init__()

    def forward(self, *args, **kwargs):
        loss = torch.tensor(0, requires_grad=True)
        loss_items = {"cls_loss": loss.detach()}
        return loss, loss_items