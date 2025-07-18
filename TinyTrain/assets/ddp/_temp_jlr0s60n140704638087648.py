
# Multi-GPU training temp file (should be automatically deleted after use)

if __name__ == "__main__":
    import sys
    sys.path.append(f"D:\project\python_code\TinyTrain-main")
    
    import pickle
    from builtins import NoneType
    from TinyTrain.global_var import ASSETS_PATH
    
    with open(ASSETS_PATH /"ddp/trainer.pkl", "rb") as f:
        trainer_obj = pickle.load(f)

    trainer = NoneType(config_manager=trainer_obj.config_manager, model=trainer_obj.model, callback=trainer_obj.callbacks)
    trainer.train()
