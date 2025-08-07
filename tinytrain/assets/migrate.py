import torch


def Migrate(old_model_pt, new_model_pt):
    ckpt = torch.load(old_model_pt)
    model_args = ckpt['model_args']
    for i, module in enumerate(model_args["network"]):
        if module["module"] == "nn.Upsample":
            ckpt["model_args"]["network"][i]["module"] = "torch.nn.Upsample"
    torch.save(ckpt, new_model_pt)

if __name__ == '__main__':
    old_model = r"old_model.pt"
    new_model = r"new_model.pt"
    Migrate(old_model, new_model)
