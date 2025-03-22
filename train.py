import cv2
import matplotlib.pyplot as plt
import numpy as np
import os
import h5py
import torch
import re
import asyncio

from PhysNetModel import PhysNet
from loss import ContrastLoss
from utils_train import EarlyStopping
from IrrelevantPowerRatio import IrrelevantPowerRatio
from PIL import Image

from utils_data import *
from utils_sig import *
from torch import optim
from tqdm import tqdm
from torch.utils.data import DataLoader
from torch.optim.lr_scheduler import ReduceLROnPlateau
from sacred import Experiment
from sacred.observers import FileStorageObserver
import torch.utils.benchmark as benchmark
from demo.utils_im import image_frequency_analysis, notch_filter_images, save_images_async

exp_name = "train_test"
ex = Experiment(exp_name, save_git_info=False)
#exp_name = "ubfc_train_test1"


if torch.cuda.is_available():
    device = torch.device('cuda')
    torch.backends.cudnn.enabled = True
    torch.backends.cudnn.benchmark = True
    device = torch.device('cpu')

else:
    device = torch.device('cpu')
    #print()


def check_run_id(folder_path):
    """
    Get the largest 1-2 digit numeric subfolder name in the given folder.

    Args:
        folder_path (str): The path to the folder to check.

    Returns:
        int or None: The largest numeric subfolder name as an integer, or None if no such folder exists.
    """
    # Regular expression to match 1-2 digit numbers
    numeric_pattern = re.compile(r'^\d{1,2}$')
    largest_number = None

    try:
        for entry in os.scandir(folder_path):
            if entry.is_dir() and numeric_pattern.match(entry.name):
                num = int(entry.name)
                if largest_number is None or num > largest_number:
                    largest_number = num
        return largest_number

    except FileNotFoundError:
        print(f"Folder not found: {folder_path}")
        return None

    except PermissionError:
        print(f"Permission denied for folder: {folder_path}")
        return None


# Benchmarking function
def benchmark_dataloader(dataset, num_workers, pin_memory, dl_device):
    print(f"[BENCHMARK] Running permutation: num_workers: {num_workers}, pin_memory:{pin_memory}, device: {dl_device}")
    #device = torch.device('cuda')
    dataloader = DataLoader(dataset, batch_size=2, num_workers=num_workers, pin_memory=pin_memory)

    if dl_device.type == "cuda":
        async_open = True
    else:
        async_open = False

    test_range = 20

    def transfer_batches():
        for i in range(test_range):
            imgs = next(iter(dataloader))
            #imgs = imgs.to(dl_device, non_blocking=async_open)
            i += 1

    return benchmark.Timer(stmt="transfer_batches()", globals={"transfer_batches": transfer_batches}).timeit(10)#


def check_dataloader_sample_iterations(dataloader):
    max_cnt = 50
    for i, batch in enumerate(dataloader):
        batch = batch.detach().numpy()
        batch = batch.astype(np.uint8)
        batch_1 = batch[0]
        batch_2 = batch[1]

        batch_1_T = np.transpose(batch_1, (1, 2, 3, 0))
        batch_2_T = np.transpose(batch_2, (1, 2, 3, 0))

        dataloader_test_path = "dataloader_testing"

        all_im_data_pil_b1 = [Image.fromarray(x) for x in batch_1_T]
        paths = [f"{dataloader_test_path}/batch{i}_1_{x}.jpg" for x in range(len(all_im_data_pil_b1))]
        formats = ["JPEG" for _ in range(len(all_im_data_pil_b1))]
        asyncio.run(save_images_async(image_batch=all_im_data_pil_b1, file_paths=paths, file_formats=formats))

        all_im_data_pil_b2 = [Image.fromarray(x) for x in batch_2_T]
        paths = [f"{dataloader_test_path}/batch{i}_2_{x}.jpg" for x in range(len(all_im_data_pil_b2))]
        formats = ["JPEG" for _ in range(len(all_im_data_pil_b2))]
        asyncio.run(save_images_async(image_batch=all_im_data_pil_b2, file_paths=paths, file_formats=formats))

        if i >= max_cnt:
            break


# Training loop logic
@ex.main
def training_loop(_run, total_epoch, T, S, lr, fold_dir, result_dir, fs, delta_t, K, in_ch, use_kfold_crossval, fold_idx, n_folds,
                  use_pretrained_model, freeze_layers):

    if use_kfold_crossval:
        exp_dir = f"{fold_dir}/{int(_run._id)}"
        print(f"Executing training loop (With K-Fold Crossvalidation): fold {fold_idx} ")
    else:
        exp_dir = f"{result_dir}/{int(_run._id)}"
        print(f"Executing training loop (Standard) run {fold_idx} ")

    benchmark_loaders = False
    benchmark_model_training = False
    check_dataloader_samples = False

    #TODO: confirm that LR Scheduling is being applied
    use_lr_scheduling = True
    pretrained_model_path = "ubfc_pretrained.pt"

    #ex.observers.append(FileStorageObserver(exp_dir))

    random_seed = 42
    early_stopping = None
    scheduler = None

    np.random.seed(random_seed)
    torch.manual_seed(random_seed)

    # get the training and test file path list by spliting the dataset
    # train_list, test_list = UBFC_LU_split() # TODO: you should define your function to split your dataset for training and testing
    train_list, test_list = Camvisim_LU_split(k_fold=use_kfold_crossval, num_folds=n_folds, fold_idx=fold_idx)
    # train_list, test_list = PURE_LU_split(k_fold=use_kfold_crossval, num_folds=n_folds, fold_idx=fold_idx)

    np.save(exp_dir+'/train_list.npy', train_list)
    np.save(exp_dir+'/test_list.npy', test_list)

    # define the dataloader
    dataset = H5Dataset(train_list, T, device) # please read the code about H5Dataset when preparing your dataset

    # Test different num_workers values
    if benchmark_loaders:
        results = {}
        for num_workers in [0]:  # , 2, 4, 8]:
            results[f'num_workers={num_workers}'] = {
               # "pin_memory=True_GPU": benchmark_dataloader(dataset, num_workers, pin_memory=True, dl_device=torch.device('cuda')),
                "pin_memory=False_GPU": benchmark_dataloader(dataset, num_workers, pin_memory=False, dl_device=torch.device('cuda')),
              #"pin_memory=True_CPU": benchmark_dataloader(dataset, num_workers, pin_memory=True, dl_device=torch.device('cpu')),
                "pin_memory=False_CPU": benchmark_dataloader(dataset, num_workers, pin_memory=False,dl_device=torch.device('cpu'))
            }

        for key, result in results.items():
            print(f"{key}: {result}")

    if benchmark_model_training:
        benchmark_model(dataset=dataset)

    dataloader = DataLoader(dataset, batch_size=2, # two videos for contrastive learning
                            shuffle=True, num_workers=0, pin_memory=False, drop_last=True) # TODO: If you run the code on Windows, remove num_workers=4

    if check_dataloader_samples:
        check_dataloader_sample_iterations(dataloader=dataloader)
        print("Exiting... ")
        os.system('exit')

    # define the model and loss
    model = PhysNet(S, in_ch=in_ch).to(device).train()

    if use_pretrained_model:
        print(f"[Model Training] NOTE: Training using pretrained model file {pretrained_model_path}")
        model.load_state_dict(torch.load(pretrained_model_path, weights_only=False))
        #lr = 1e-6  # learning rate
    else:
        print(f"[Model Training] NOTE: Training a new contrast-phys model from scratch...")

    layers_to_freeze = ['start', 'loop1']
    if freeze_layers:
        print(f"[MODEL TRAINING] Freezing layers: {layers_to_freeze}")
        for name, param in model.named_parameters():
            if any(layer in name for layer in layers_to_freeze):
                param.requires_grad = False

    # define the optimizer
    opt = optim.AdamW(model.parameters(), lr=lr)

    if use_lr_scheduling:
        scheduler = ReduceLROnPlateau(opt, mode='min', factor=0.1, patience=2, verbose=True)
        early_stopping = EarlyStopping(patience=10, min_delta=1e-2)

    loss_func = ContrastLoss(delta_t, K, fs, high_pass=40, low_pass=250)

    # define irrelevant power ratio
    IPR = IrrelevantPowerRatio(Fs=fs, high_pass=40, low_pass=250)

    all_epochs_mean_iprs = list()
    for e in range(total_epoch):
        if early_stopping.early_stop:
            print("[Model Training] Terminating due to early stopping....")
            break

        epoch_iprs = list()
        print(f"[Model Training] Starting Epoch {e}...")
        for it in tqdm(range(np.round(60 / (T / fs)).astype('int')), desc="Iterations", leave=False):
            # TODO: 60 means the video length of each video is 60s. If each video's length in your dataset is other value (e.g, 30s), you should use that value.
            #batch_progress = tqdm(total=len(dataloader), desc="Batch", leave=False)
            for imgs in dataloader:  # tqdm for each batch in the dataloader

                # model forward propagation
                model_output = model(imgs)
                rppg = model_output[:, -1]  # get rppg

                # define the loss functions
                loss, p_loss, n_loss = loss_func(model_output)

                # optimize
                opt.zero_grad()
                loss.backward()
                opt.step()

                # evaluate irrelevant power ratio during training
                ipr = torch.mean(IPR(rppg.clone().detach()))
                epoch_iprs.append(ipr)

                # save loss values and IPR
                ex.log_scalar("loss", loss.item())
                ex.log_scalar("p_loss", p_loss.item())
                ex.log_scalar("n_loss", n_loss.item())
                ex.log_scalar("ipr", ipr.item())
                #batch_progress.update(1)

            #batch_progress.close()
        # save model checkpoints
        torch.save(model.state_dict(), exp_dir+'/epoch%d.pt'%e)
        mean_epoch_ipr = np.sum(epoch_iprs)/len(epoch_iprs)

        # Schedulers for Learning Rate
        if use_lr_scheduling:
            early_stopping(mean_epoch_ipr)
            scheduler.step(mean_epoch_ipr)
            for param_group in opt.param_groups:
                print(f"Learning rate after step: {param_group['lr']}")

        print(f"[Model Training] Mean IPR for epoch {e}: {mean_epoch_ipr}")
        all_epochs_mean_iprs.append(mean_epoch_ipr)
    print(f"Done")


def run_training_loop(num_loops, fold_dir):
    for i in range(num_loops):
        # store experiment recording to the path
        if num_loops != 1:
            observer = FileStorageObserver(fold_dir)
            ex.observers[:] = [observer]
            ex.run(config_updates={"fold_dir": fold_dir, "fold_idx": i})


@ex.config
def my_config():
    # here are some hyperparameters in our method
    # hyperparams for model training
    total_epoch = 30 # total number of epochs for training the model
    in_ch = 3 # TODO: number of input video channels, in_ch=3 for RGB videos, in_ch=1 for NIR videos.

    # hyperparams for ST-rPPG block
    fs = 30 # video frame rate, TODO: modify it if your video frame rate is not 30 dataset_fps.
    T = fs * 10 # temporal dimension of ST-rPPG block, default is 10 seconds.
    S = 2 # spatial dimenion of ST-rPPG block, default is 2x2.

    # hyperparams for rPPG spatiotemporal sampling
    delta_t = int(T/2) # time length of each rPPG sample
    K = 4 # the number of rPPG samples at each spatial position
    use_pretrained_model = False
    freeze_layers = False
    use_kfold_crossval = use_kfold
    n_folds = num_folds

    if use_pretrained_model:
        lr = 1e-6  # learning rate
    else:
        lr = 1e-5  # learning rate

    result_dir = f'./results/{exp_name}' # store checkpoints and training recording
    if not use_kfold:
        ex.observers.append(FileStorageObserver(result_dir))

    else:
        num_loops = num_folds


def benchmark_model(dataset):
    # ================================ hyperparams start ===================================
    total_epoch = 1  # total number of epochs for training the model
    lr = 1e-5  # learning rate
    in_ch = 3  # TODO: number of input video channels, in_ch=3 for RGB videos, in_ch=1 for NIR videos.
    random_seed = 42
    use_pretrained_model = True
    pretrained_model_path = "ubfc_pretrained.pt"

    # hyperparams for ST-rPPG block
    fs = 25  # video frame rate, TODO: modify it if your video frame rate is not 30 dataset_fps.
    T = fs * 10  # temporal dimension of ST-rPPG block, default is 10 seconds.
    S = 2  # spatial dimenion of ST-rPPG block, default is 2x2.

    # hyperparams for rPPG spatiotemporal sampling
    delta_t = int(T / 2)  # time length of each rPPG sample
    K = 4  # the number of rPPG samples at each spatial position
    clip_len_t = 120
    # ================================ hyperparams end ===================================
    np.random.seed(random_seed)
    torch.manual_seed(random_seed)

    dataloader = DataLoader(dataset, batch_size=2, # two videos for contrastive learning
                            shuffle=True, num_workers=0, pin_memory=False, drop_last=True) # TODO: If you run the code on Windows, remove num_workers=4

    # define the model and loss
    model = PhysNet(S, in_ch=in_ch).to(device).train()
    if use_pretrained_model:
        print(f"[Model Benchmark] NOTE: Training using pretrained model file {pretrained_model_path}")
        model.load_state_dict(torch.load(pretrained_model_path, weights_only=False))
        lr = 1e-6  # learning rate

    loss_func = ContrastLoss(delta_t, K, fs, high_pass=40, low_pass=250)

    # define irrelevant power ratio
    IPR = IrrelevantPowerRatio(Fs=fs, high_pass=40, low_pass=250)

    # define the optimizer
    opt = optim.AdamW(model.parameters(), lr=lr)

    for e in range(total_epoch):
        print(f"[Model Benchmark] Starting Epoch {e}...")
        for it in range(np.round(clip_len_t / (T / fs)).astype('int')):
            print(f"[Model Benchmark] Starting Iteration {it}...")
        #for it in tqdm(range(np.round(clip_len_t / (T / fs)).astype('int')), desc="Iterations", leave=False):
            # TODO: 60 means the video length of each video is 60s. If each video's length in your dataset is other value (e.g, 30s), you should use that value.
            batch_progress = tqdm(total=len(dataloader), desc="Batch", leave=False)
            for imgs in dataloader:  # tqdm for each batch in the dataloader
                #imgs = imgs.to(device)

                # model forward propagation
                model_output = model(imgs)
                rppg = model_output[:, -1]  # get rppg

                # define the loss functions
                loss, p_loss, n_loss = loss_func(model_output)

                # optimize
                opt.zero_grad()
                loss.backward()
                opt.step()

                # evaluate irrelevant power ratio during training
                ipr = torch.mean(IPR(rppg.clone().detach()))
                batch_progress.update(1)


if __name__ == "__main__":
    use_kfold = True
    if use_kfold:
        num_folds = 5
        result_dir = f'./results/{exp_name}' # store checkpoints and training recording
        last_run_id = check_run_id(result_dir)
        if last_run_id:
            current_sacred_run_id = last_run_id + 1
        else:
            current_sacred_run_id = 1

        fold_dir = f"{result_dir}/{int(current_sacred_run_id)}/folds"
        if not os.path.isdir(fold_dir):
            os.makedirs(fold_dir)

        os.makedirs(fold_dir, exist_ok=True)
        run_training_loop(num_loops=num_folds, fold_dir=fold_dir)

    else:
        ex.run(config_updates={"fold_dir": "", "use_kfold_crossval": False, "fold_idx": 0, "n_folds": 1})
