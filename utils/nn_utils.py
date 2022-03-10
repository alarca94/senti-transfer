import torch


def get_predictions(outputs, loss_fn):
    if type(loss_fn) == torch.nn.BCELoss:
        return (outputs.squeeze(1) >= 0.5).to(torch.int16)
    elif type(loss_fn) == torch.nn.NLLLoss:
        return torch.argmax(outputs, dim=1)
    elif type(loss_fn) == torch.nn.MSELoss:
        return
    raise Exception('Invalid loss function!')
