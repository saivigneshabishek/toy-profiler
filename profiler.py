import torch
from models import DINOv3
from torch.profiler import profile, ProfilerActivity, record_function

def profile_model(model, inputs, device="cpu", warmup=10):
    model = model.to(device).eval()
    x = inputs.to(device)

    if warmup > 0:
        print(f"Warming up for {warmup} iterations")
        with torch.inference_mode():
            for _ in range(warmup):
                _ = model.inference(x)
        if device == "cuda":
            torch.cuda.synchronize()

    activities = [ProfilerActivity.CPU]
    if device == "cuda":
        activities.append(ProfilerActivity.CUDA)

    print(f"========= {device} =========")
    with profile(activities=activities, record_shapes=True,
        profile_memory=True, with_stack=False) as prof:
        with torch.inference_mode():
            with record_function("model_inference"):
                _ = model.inference(x)

    print(prof.key_averages())
    prof.export_chrome_trace(f"trace_{device}.json")

if __name__ == "__main__":
    DINO = {
        "model": "dinov3_vits16",
        "dir": "/torch/dinov3",
        "ckpt": "/torch/ckpts/dinov3_vits16_pretrain_lvd1689m-08c60483.pth",
    }

    model = DINOv3(MODEL=DINO["model"], DIR=DINO["dir"], CKPT=DINO["ckpt"])
    x = torch.randn((1,3,128,128), dtype=torch.float32)

    profile_model(model, x, "cpu")
    profile_model(model, x, "cuda")
