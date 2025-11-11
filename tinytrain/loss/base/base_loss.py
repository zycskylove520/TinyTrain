import torch


class TTBaseLoss(torch.nn.Module):
    def __init__(self, *args, **kwargs):
        super(TTBaseLoss, self).__init__()

    def forward(self, *args, **kwargs):
        loss = torch.tensor(0, requires_grad=True)
        loss_items = {"cls_loss": loss.detach()}
        return loss, loss_items