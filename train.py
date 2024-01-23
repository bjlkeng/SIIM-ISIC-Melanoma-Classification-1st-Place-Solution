import os
import time
import random
import argparse
import numpy as np
import pandas as pd
import wandb

from datetime import datetime, timedelta
from tqdm import tqdm
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedKFold
import torch
from torch import amp
from torch import cuda
from torch.autograd.profiler import record_function
from torch.utils.data import DataLoader, Dataset
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.optim import lr_scheduler
from torch.utils.data.sampler import RandomSampler, SequentialSampler
from torch.optim.lr_scheduler import CosineAnnealingLR
from util import GradualWarmupSchedulerV2

# import apex
from dataset import get_df, get_transforms, MelanomaDataset
from models import Effnet_Melanoma, Resnest_Melanoma, Seresnext_Melanoma
from contextlib import nullcontext


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--kernel-type", type=str, required=True)
    parser.add_argument("--data-dir", type=str, default="/raid/")
    parser.add_argument("--data-folder", type=int, required=True)
    parser.add_argument("--image-size", type=int, required=True)
    parser.add_argument("--enet-type", type=str, required=True)
    parser.add_argument("--batch-size", type=int, default=48)
    parser.add_argument("--num-workers", type=int, default=6)
    parser.add_argument("--init-lr", type=float, default=3e-5)
    parser.add_argument("--out-dim", type=int, default=9)
    parser.add_argument("--n-epochs", type=int, default=15)
    parser.add_argument("--use-amp", action="store_true", default=False)
    parser.add_argument("--compile-mode", type=str, default="none")
    parser.add_argument("--mat-mul-precision", type=str, default="highest")
    parser.add_argument("--use-meta", action="store_true", default=False)
    parser.add_argument("--DEBUG", action="store_true")
    parser.add_argument("--model-dir", type=str, default="./weights")
    parser.add_argument("--log-dir", type=str, default="./logs")
    parser.add_argument("--CUDA_VISIBLE_DEVICES", type=str, default="0")
    parser.add_argument("--fold", type=str, default="0,1,2,3,4")
    parser.add_argument("--n-meta-dim", type=str, default="512,128")
    parser.add_argument("--use-warmup", action="store_true", default=False)
    parser.add_argument("--no-use-external", action="store_true", default=False)
    parser.add_argument("--binary-labels", action="store_true", default=False)
    parser.add_argument("--no-pretraining", action="store_true", default=False)
    parser.add_argument("--no-cosine", action="store_true", default=False)
    parser.add_argument("--no-augmentation", action="store_true", default=False)
    parser.add_argument("--use-profiler", action="store_true", default=False)
    parser.add_argument("--tags", type=str, default="empty_tag")

    args, _ = parser.parse_known_args()
    return args


def set_seed(seed=0):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True


def train_epoch(model, loader, optimizer):
    if args.use_amp:
        scaler = cuda.amp.GradScaler()

    model.train()
    train_loss = []
    bar = tqdm(loader)
    for data, target in bar:
        with record_function("## optimizer ##"):
            optimizer.zero_grad()

        with record_function("## forward ##"):
            if args.use_meta:
                data, meta = data
                data, meta, target = data.to(device), meta.to(device), target.to(device)
                if not args.use_amp:
                    logits = model(data, meta)
                    loss = criterion(logits, target)
                else:
                    with amp.autocast(device_type="cuda"):
                        logits = model(data, meta)
                        loss = criterion(logits, target)
            else:
                data, target = data.to(device), target.to(device)
                if not args.use_amp:
                    logits = model(data)
                    loss = criterion(logits, target)
                else:
                    with amp.autocast(device_type="cuda"):
                        logits = model(data)
                        loss = criterion(logits, target)

        if not args.use_amp:
            with record_function("## backward ##"):
                loss.backward()
                if args.image_size in [896, 576]:
                    torch.nn.utils.clip_grad_norm_(model.parameters(), 0.5)
            with record_function("## optimizer ##"):
                optimizer.step()
        else:
            with record_function("## backward ##"):
                scaler.scale(loss).backward()
            with record_function("## optimizer ##"):
                scaler.step(optimizer)
                scaler.update()

        loss_np = loss.detach().cpu().numpy()
        train_loss.append(loss_np)
        smooth_loss = sum(train_loss[-100:]) / min(len(train_loss), 100)
        bar.set_description("loss: %.5f, smth: %.5f" % (loss_np, smooth_loss))

    train_loss = np.mean(train_loss)
    return train_loss


def get_trans(img, I):
    if I >= 4:
        img = img.transpose(2, 3)
    if I % 4 == 0:
        return img
    elif I % 4 == 1:
        return img.flip(2)
    elif I % 4 == 2:
        return img.flip(3)
    elif I % 4 == 3:
        return img.flip(2).flip(3)


def val_epoch(model, loader, mel_idx, is_ext=None, n_test=1, get_output=False):
    model.eval()
    val_loss = []
    LOGITS = []
    PROBS = []
    TARGETS = []
    with torch.no_grad():
        for data, target in tqdm(loader):
            if args.use_meta:
                data, meta = data
                data, meta, target = data.to(device), meta.to(device), target.to(device)
                logits = torch.zeros((data.shape[0], args.out_dim)).to(device)
                probs = torch.zeros((data.shape[0], args.out_dim)).to(device)
                for I in range(n_test):
                    l = model(get_trans(data, I), meta)
                    logits += l
                    probs += l.softmax(1)
            else:
                data, target = data.to(device), target.to(device)
                logits = torch.zeros((data.shape[0], args.out_dim)).to(device)
                probs = torch.zeros((data.shape[0], args.out_dim)).to(device)
                for I in range(n_test):
                    l = model(get_trans(data, I))
                    logits += l
                    probs += l.softmax(1)
            logits /= n_test
            probs /= n_test

            LOGITS.append(logits.detach().cpu())
            PROBS.append(probs.detach().cpu())
            TARGETS.append(target.detach().cpu())

            loss = criterion(logits, target)
            val_loss.append(loss.detach().cpu().numpy())

    val_loss = np.mean(val_loss)
    LOGITS = torch.cat(LOGITS).numpy()
    PROBS = torch.cat(PROBS).numpy()
    TARGETS = torch.cat(TARGETS).numpy()

    if get_output:
        return LOGITS, PROBS
    else:
        acc = (PROBS.argmax(1) == TARGETS).mean() * 100.0
        auc = roc_auc_score((TARGETS == mel_idx).astype(float), PROBS[:, mel_idx])
        auc_20 = roc_auc_score(
            (TARGETS[is_ext == 0] == mel_idx).astype(float), PROBS[is_ext == 0, mel_idx]
        )
        return val_loss, acc, auc, auc_20


def run(
    fold, df, meta_features, n_meta_features, transforms_train, transforms_val, mel_idx
):
    if args.DEBUG:
        args.n_epochs = min(5, args.n_epochs)
        df_train = df[df["fold"] != fold].sample(args.batch_size * 5)
        df_valid = df[df["fold"] == fold].sample(args.batch_size * 5)
    else:
        df_train = df[df["fold"] != fold]
        df_valid = df[df["fold"] == fold]

    if args.no_augmentation:
        transforms_train = transforms_val
    dataset_train = MelanomaDataset(
        df_train, "train", meta_features, transform=transforms_train
    )
    dataset_valid = MelanomaDataset(
        df_valid, "valid", meta_features, transform=transforms_val
    )
    train_loader = torch.utils.data.DataLoader(
        dataset_train,
        batch_size=args.batch_size,
        sampler=RandomSampler(dataset_train),
        num_workers=args.num_workers,
    )
    valid_loader = torch.utils.data.DataLoader(
        dataset_valid, batch_size=args.batch_size, num_workers=args.num_workers
    )

    model = ModelClass(
        args.enet_type,
        n_meta_features=n_meta_features,
        n_meta_dim=[int(nd) for nd in args.n_meta_dim.split(",")],
        out_dim=args.out_dim,
        pretrained=(not args.no_pretraining),
    )
    if args.compile_mode != "none":
        print(f"Compiling model using mode: {args.compile_mode}")
        model = torch.compile(model, mode=args.compile_mode)

    if DP:
        model = apex.parallel.convert_syncbn_model(model)
    model = model.to(device)

    auc_max = 0.0
    auc_20_max = 0.0
    model_file = os.path.join(args.model_dir, f"{args.kernel_type}_best_fold{fold}.pth")
    model_file2 = os.path.join(
        args.model_dir, f"{args.kernel_type}_best_20_fold{fold}.pth"
    )
    model_file3 = os.path.join(
        args.model_dir, f"{args.kernel_type}_final_fold{fold}.pth"
    )

    optimizer = optim.Adam(model.parameters(), lr=args.init_lr)
    # BK: Use Pytorch mixed precision instead
    # if args.use_amp:
    #    model, optimizer = amp.initialize(model, optimizer, opt_level="O1")
    if DP:
        model = nn.DataParallel(model)

    if args.use_warmup:
        #     scheduler_cosine = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, args.n_epochs - 1)
        if args.no_cosine:
            scheduler_warmup = GradualWarmupSchedulerV2(
                optimizer, multiplier=10, total_epoch=1
            )
        else:
            scheduler_cosine = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(
                optimizer, args.n_epochs - 1
            )
            scheduler_warmup = GradualWarmupSchedulerV2(
                optimizer,
                multiplier=10,
                total_epoch=1,
                after_scheduler=scheduler_cosine,
            )

    if args.use_profiler:
        context = torch.profiler.profile(
            activities=[
                torch.profiler.ProfilerActivity.CPU,
                torch.profiler.ProfilerActivity.CUDA,
            ],
            schedule=torch.profiler.schedule(wait=0,
                                             warmup=args.n_epochs - 1,
                                             active=1,
                                             repeat=1),
            record_shapes=True,
            profile_memory=True,
            with_stack=True,
            on_trace_ready=trace_handler,
        )
    else:
        context = nullcontext()

    with context as prof:
        for epoch in range(1, args.n_epochs + 1):
            print(time.ctime(), f"Fold {fold}, Epoch {epoch}")
            #         scheduler_warmup.step(epoch - 1)

            train_loss = train_epoch(model, train_loader, optimizer)
            val_loss, acc, auc, auc_20 = val_epoch(
                model, valid_loader, mel_idx, is_ext=df_valid["is_ext"].values
            )

            content = (
                time.ctime()
                + " "
                + f'Fold {fold}, Epoch {epoch}, lr: {optimizer.param_groups[0]["lr"]:.7f}, train loss: {train_loss:.5f}, valid loss: {(val_loss):.5f}, acc: {(acc):.4f}, auc: {(auc):.6f}, auc_20: {(auc_20):.6f}.'
            )
            print(content)
            with open(
                os.path.join(args.log_dir, f"log_{args.kernel_type}.txt"), "a"
            ) as appender:
                appender.write(content + "\n")

            if args.use_warmup:
                scheduler_warmup.step()
                if epoch == 2:
                    scheduler_warmup.step()  # bug workaround

            if not args.DEBUG:
                wandb.log(
                    {
                        "Fold": fold,
                        "Epoch": epoch,
                        "Train Loss": train_loss,
                        "Valid Loss": val_loss,
                        "Accuracy": acc,
                        "AUC": auc,
                        "AUC_20": auc_20,
                    }
                )

            if not args.use_profiler:
                if auc > auc_max:
                    print("auc_max ({:.6f} --> {:.6f}). Saving model ...".format(auc_max, auc))
                    torch.save(model.state_dict(), model_file)
                    auc_max = auc
                if auc_20 > auc_20_max:
                    print(
                        "auc_20_max ({:.6f} --> {:.6f}). Saving model ...".format(
                            auc_20_max, auc_20
                        )
                    )
                    torch.save(model.state_dict(), model_file2)
                    auc_20_max = auc_20
            
            prof.step()

    torch.save(model.state_dict(), model_file3)


def main():
    df, df_test, meta_features, n_meta_features, mel_idx = get_df(
        args.kernel_type, args.out_dim, args.data_dir, args.data_folder, args.use_meta
    )

    if args.no_use_external:
        df = df[df["is_ext"] == 0].reset_index(drop=True)

    if args.binary_labels:
        df["target"] = (df["target"] == mel_idx).astype(int)
        assert args.out_dim == 2

    transforms_train, transforms_val = get_transforms(args.image_size)

    folds = [int(i) for i in args.fold.split(",")]
    print(f"Folds: {folds}")
    for fold in folds:
        run(
            fold,
            df,
            meta_features,
            n_meta_features,
            transforms_train,
            transforms_val,
            mel_idx,
        )


TIME_FORMAT_STR: str = "%b_%d_%H_%M_%S"


def trace_handler(prof: torch.profiler.profile):
    # Prefix for file names.
    timestamp = datetime.now().strftime(TIME_FORMAT_STR)
    file_prefix = f"profiler/{timestamp}_{args.kernel_type}"

    # Construct the trace file.
    prof.export_chrome_trace(f"{file_prefix}.json.gz")

    # Construct the memory timeline file.
    prof.export_memory_timeline(f"{file_prefix}.html", device="cuda:0")

    # Save trace to file
    prof.export_stacks(f"{file_prefix}.txt", "self_cuda_time_total") 

    # Dump to file
    with open(f"{file_prefix}.pstats", "w") as f:
        f.write('\n## CPU total time profiling ##\n')
        f.write(prof.key_averages(group_by_input_shape=True).table(sort_by="self_cpu_time_total", row_limit=30))
        f.write('\n\n')
        f.write('\n## CUDA total time profiling ##\n')
        f.write(prof.key_averages(group_by_input_shape=True).table(sort_by="self_cuda_time_total", row_limit=30))
        f.write('\n\n')
        f.write('\n## CPU Memory profiling ##\n')
        f.write(prof.key_averages().table(sort_by="self_cpu_memory_usage", row_limit=30))
        f.write('\n\n')
        f.write('\n## CUDA Memory profiling ##\n')
        f.write(prof.key_averages().table(sort_by="self_cuda_memory_usage", row_limit=30))
        f.write('\n\n')


if __name__ == "__main__":
    args = parse_args()
    assert args.compile_mode in ["none", "reduce-overhead", "max-autotune", "default"]
    assert args.mat_mul_precision in ["highest", "high", "medium"]
    if args.mat_mul_precision != "highest":
        torch.set_float32_matmul_precision(args.mat_mul_precision)

    os.makedirs(args.model_dir, exist_ok=True)
    os.makedirs(args.log_dir, exist_ok=True)
    os.environ["CUDA_VISIBLE_DEVICES"] = args.CUDA_VISIBLE_DEVICES

    if not args.DEBUG:
        wandb.init(
            project="SIIM_ISIC_Melanoma_Classification",
            name=args.kernel_type,
            tags=args.tags.split(","),
        )
        wandb.config.update(vars(args))

    if args.enet_type == "resnest101":
        ModelClass = Resnest_Melanoma
    elif args.enet_type == "seresnext101":
        ModelClass = Seresnext_Melanoma
    elif "efficientnet" in args.enet_type:
        ModelClass = Effnet_Melanoma
    else:
        raise NotImplementedError()

    # BK: Disable Apex
    # DP = len(os.environ['CUDA_VISIBLE_DEVICES']) > 1
    DP = False

    set_seed()

    device = torch.device("cuda")
    criterion = nn.CrossEntropyLoss()

    main()
