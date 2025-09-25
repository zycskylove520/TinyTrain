import torch


def Migrate(old_model_pt, new_model_pt):
    ckpt = torch.load(old_model_pt, weights_only=False)
    model_args = ckpt['model_args']
    for i, module in enumerate(model_args["network"]):
        if module["module"] == "nn.Upsample":
            ckpt["model_args"]["network"][i]["module"] = "torch.nn.Upsample"
        if module["module"] == "Conv":
            ckpt["model_args"]["network"][i]["module"] = "CBA"
        if module["module"] == "Concat":
            value = ckpt["model_args"]["network"][i]["module"]["args"].pop("dimension", None)
            if value:
                ckpt["model_args"]["network"][i]["module"]["args"]["dim"] = value
    torch.save(ckpt, new_model_pt)

if __name__ == '__main__':
    old_model = r"best.pt"
    new_model = r"new_model.pt"
    Migrate(old_model, new_model)
