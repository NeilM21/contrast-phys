import numpy as np
import h5py
import torch
from PhysNetModel import PhysNet
from utils_data import *
from utils_sig import *
from sacred import Experiment
from sacred.observers import FileStorageObserver
from pathlib import Path
import json
import time
from datetime import datetime


ex = Experiment('model_pred', save_git_info=False)


def get_lowest_ipr_epoch(train_model_path: str):
    """
    Checks a trained Contrast-Phys model folder to identify the best performing model in terms of the IPR value

    Args:
        train_model_path (str): Path where the .pt files of the trained model and its metrics reside.

    Returns:
        lowest_ipr_idx (int): The index of the model with the lowest IPR error
        lowest_ipr (float): The best IPR loss value out of all trained epochs
        lowest_ipr_model_path (str): The filepath of the model with the lowest IPR

    """
    json_path = f"{train_model_path}/metrics.json"
    config_path = f"{train_model_path}/config.json"
    lowest_ipr = 99999999999999999

    config_json = json.loads(open(config_path).read())
    metrics_json = json.loads(open(json_path).read())
    iprs = metrics_json["ipr"]["values"]
    num_epochs = len(list(Path(train_model_path).rglob("*.pt")))
    metric_per_epoch = len(iprs) // num_epochs
    ipr_by_epoch = list()
    for i in range(num_epochs):
        ipr_by_epoch.append(iprs[i * metric_per_epoch:(i + 1) * metric_per_epoch])
        mean_ipr = np.mean(ipr_by_epoch[i])
        if mean_ipr < lowest_ipr:
            lowest_ipr = mean_ipr
            lowest_ipr_idx = i
            lowest_ipr_model_path = list(Path(config_path).parent.glob(f"epoch{lowest_ipr_idx}.pt"))[0]

    return lowest_ipr_idx, np.round(lowest_ipr, 4), lowest_ipr_model_path


def is_kfold(test_dir):
    folds_path = os.path.join(test_dir, 'folds')

    # Check if 'folds' exists and is a directory
    if os.path.isdir(folds_path):
        entries = os.listdir(folds_path)
        # Exclude Sacred metadata folder
        entries = [x for x in entries if x not in ["_sources", "best_ipr", "last_epoch"]]
        print(f"[TESTING] Cross-validated model results found in {test_dir} ({len(entries)} folds. Testing all folds....")
        return True, len(entries), [os.path.join(folds_path, x) for x in entries]
    else:
        print(f"[TESTING] One set of model results found in {test_dir}. Testing the model...")
        return False, 1, None


def run_testing_loop(num_loops, fold_dir):
    for i in range(num_loops):
        # store experiment recording to the path
        if num_loops != 1:
            observer = FileStorageObserver(fold_dir)
            ex.observers[:] = [observer]
            ex.run(config_updates={"fold_dir": fold_dir, "fold_idx": i})

@ex.config
def my_config():
    # camvisim_r2l_mts best e is 5, exp num is 3, ubfc_train_test1 best e is 29, exp num is 6
    time_interval = 30  # get rppg for 30s video clips, too long clips might cause out of memory
    use_gpu_for_inference = False
    repo_model_cross_dataset_testing = False
    cross_dataset_source = None

    if torch.cuda.is_available() and use_gpu_for_inference:
        device = torch.device('cuda')
        torch.backends.cudnn.enabled = True
        torch.backends.cudnn.benchmark = True

    else:
        device = torch.device('cpu')

    #train_exp_dir = f'./results/ubfc_train_test1/{train_exp_num}'  # training experiment directory
    #train_exp_dir = f'./results/camvisim_r2l_mts/{train_exp_num}' # training experiment directory
    # train_exp_dir = f'./results/camvisim_train_cphys2_proper/{train_exp_num}' # training experiment directory
    #train_exp_dir = f"./results/'camvisim_train_cphys_nfiltered_proper_trainbenchmark_dataloader/{train_exp_num}" # training experiment directory
    #train_exp_dir = f'./results/ubfc_pretrained_model_camvisimnfilt_freeze_toencoder2/{train_exp_num}'  # training experiment directory
    #train_exp_dir = f'./results/ubfc_pretrained_model_camvisimnfilt_freeze_toloop1_crossval/{train_exp_num}'  # training experiment directory

    print(f"[TESTING] Model directory selected for testing is {train_exp_dir}...")
    #train_exp_dir = f'./results/ubfc_pretrained_pt/{train_exp_num}'  # training experiment directory
    use_kfold_crossval = use_kfold
    use_best_model = use_lowest_ipr_model

    if use_lowest_ipr_model:
        lowest_ipr_epoch, best_ipr, best_ipr_path = get_lowest_ipr_epoch(train_model_path=train_exp_dir)
        print(f"[TESTING - BEST IPR] Using model with the lowest IPR: epoch {lowest_ipr_epoch}: {best_ipr}")
        e = lowest_ipr_epoch

    else:
        e = 29  # the model checkpoint at epoch e
        print(f"[TESTING - CONST EPOCH] Using model with epoch {e}")
        #print(f"[TESTING - CONST EPOCH] Using UBFC pre-trained model")

    observer = FileStorageObserver(train_exp_dir)
    ex.observers[:] = [observer]


@ex.main
def my_main(_run, e, train_exp_dir, device, time_interval, repo_model_cross_dataset_testing, cross_dataset_source, use_kfold_crossval):
    pred_experiment_name = f"{Path(train_exp_dir).name}_window_10"
    _run.info['custom_id'] = pred_experiment_name  # Add to the run's metadata

    if repo_model_cross_dataset_testing:
        # cross-dataset testing only
        if cross_dataset_source == "camvisim":
            print("Cross Dataset Testing - CAMVISIM")
            new_test_data_list = list(Path("cphys_data_nfiltered/camvisim_r2l_mts").glob("*.h5"))
            new_test_data_list = [str(x) for x in new_test_data_list]

        elif cross_dataset_source == "pure":
            print("Cross Dataset Testing - PURE")
            new_test_data_list = list(Path("cphys_data/pure").glob("*.h5"))
            new_test_data_list = [str(x) for x in new_test_data_list]

        else:
            raise ValueError(f"Cross Dataset source {cross_dataset_source} incorrect! Expected one of [camvisim, pure]")

        test_list = new_test_data_list
        model_path = f"{train_exp_dir}/ubfc_pretrained.pt"
        model = PhysNet(2, 3).to(device).eval()

    else:
        # load test file paths
        test_list = list(np.load(train_exp_dir + '/test_list.npy'))
        model_path = train_exp_dir+'/epoch%d.pt'%(e)
        with open(train_exp_dir + '/config.json') as f:
            config_train = json.load(f)
            model = PhysNet(config_train['S'], config_train['in_ch']).to(device).eval()

        #run 14 = cphys2

    pred_exp_dir = train_exp_dir + '/%d'%(int(_run._id)) # prediction experiment directory
    # pred_exp_dir = f"{train_exp_dir}/{pred_experiment_name}"

    print(f"[TESTING INFERENCE] Loaded model weights {model_path}")
    model.load_state_dict(torch.load(model_path, map_location=device)) # load weights to the model
    #model.load_state_dict(torch.load('ubfc_pretrained.pt', map_location=device)) # load weights to the model

    @torch.no_grad()
    def dl_model(imgs_clip):
        # model inference
        img_batch = imgs_clip
        img_batch = img_batch.transpose((3, 0, 1, 2))
        img_batch = img_batch[np.newaxis].astype('float32')
        img_batch = torch.tensor(img_batch).to(device, non_blocking=True)

        rppg = model(img_batch)[:, -1, :]
        rppg = rppg[0].detach().cpu().numpy()
        return rppg

    total_inference_time = 0
    measure_inference = True

    for h5_path in test_list:
        h5_path = str(h5_path)
        if "pure" not in h5_path:
            subj_id = (h5_path.split('/')[-1]).split('.')[0]
        else:
            subj_id = (h5_path.split(os.sep)[-1]).split('.')[0]

        with h5py.File(h5_path, 'r') as f:
            imgs = f['imgs']
            bvp = f['bvp']
            fs = config_train['fs']

            duration = np.min([imgs.shape[0], bvp.shape[0]]) / fs
            num_blocks = int(duration // time_interval)

            rppg_list = []
            bvp_list = []

            for b in range(num_blocks):
                start_timer = time.time()
                rppg_clip = dl_model(imgs[b*time_interval*fs:(b+1)*time_interval*fs])
                end_timer = time.time()
                inference_time = end_timer - start_timer
                if measure_inference:
                    print(f"Time for inference of {subj_id}: {np.round(inference_time,2)}s")
                total_inference_time += inference_time

                rppg_list.append(rppg_clip)
                bvp_list.append(bvp[b*time_interval*fs:(b+1)*time_interval*fs])

            print(f"Total inference time: {total_inference_time}s")
            rppg_list = np.array(rppg_list)
            bvp_list = np.array(bvp_list)
            results = {'rppg_list': rppg_list, 'bvp_list': bvp_list}
            if repo_model_cross_dataset_testing:
                np.save(pred_exp_dir+'/'+h5_path.split('\\')[-1][:-3], results)
            else:
                if e != 29 and use_kfold_crossval:
                    if "pure" not in h5_path:
                        np.save(f"{best_epoch_save_dir}/{h5_path.split('/')[-1][:-3]}", results)
                    else:
                        np.save(f"{best_epoch_save_dir}/{h5_path.split(os.sep)[-1][:-3]}", results)

                elif e == 29 and use_kfold_crossval:
                    if "pure" not in h5_path:
                        np.save(f"{last_epoch_save_dir}/{h5_path.split('/')[-1][:-3]}", results)
                    else:
                        np.save(f"{last_epoch_save_dir}/{h5_path.split(os.sep)[-1][:-3]}", results)

                else:
                    np.save(pred_exp_dir+'/'+h5_path.split('/')[-1][:-3], results)


if __name__ == "__main__":
    train_exp_num = 7  # the training experiment number
    run_time = datetime.now().strftime("%Y_%m_%d_%H-%M-%S")
    use_lowest_ipr_model = True
    #train_exp_dir = f"results/'camvisim_train_cphys_nfiltered_proper_trainbenchmark_dataloader/15"
    #train_exp_dir = f"results/ubfc_train_test1/6"

    #train_exp_dir = f'./results/ubfc_pretrained_model_camvisimnfilt_freeze_toencoder2/{train_exp_num}'
    #train_exp_dir = f'./results/ubfc_pretrained_model_camvisimnfilt_freeze_toloop1_crossval/{train_exp_num}'
    train_exp_dir = f'./results/fresh_model_ubfc_lrsched/{train_exp_num}'
    #train_exp_dir = f'./results/ubfc_pretrained_model_camvisimnfilt_freeze_toloop1_crossval/{train_exp_num}'
    #train_exp_dir = f'./results/ubfc_pretrained_model_camvisimnfilt_freeze_toloop1_test2_crossval_seed72/{train_exp_num}'
    #train_exp_dir = f'./results/ubfc_pretrained_model_camvisimnfilt_retrain_nofreeze/{train_exp_num}'
    #train_exp_dir = f'./results/fresh_model_pure_lrsched_corrected/{train_exp_num}'
    #train_exp_dir = f'./results/ubfc_pretrained_model_pure_freeze_toloop1_crossval/{train_exp_num}'

    #train_exp_dir = f'./results/ubfc_pretrained_pt_repomodel'

    #fs = 30

    use_kfold, num_folds, fold_paths = is_kfold(train_exp_dir)
    if use_kfold:
        best_epoch_save_dir = f"{str(Path(fold_paths[0]).parent)}/best_ipr/{run_time}"
        last_epoch_save_dir = f"{str(Path(fold_paths[0]).parent)}/last_epoch/{run_time}"

        if use_lowest_ipr_model:
            os.makedirs(best_epoch_save_dir)
        else:
            os.makedirs(last_epoch_save_dir)

    for fold_idx in range(num_folds):
        if use_kfold:
            train_exp_dir = fold_paths[fold_idx]

        test_list = list(np.load(train_exp_dir + '/test_list.npy'))
        ex.run(config_updates={"train_exp_dir": train_exp_dir})

        #result_dir = f'./results/{exp_name}' # store checkpoints and training recording
    #last_run_id = check_run_id(result_dir)
    #if last_run_id:
    #    current_sacred_run_id = last_run_id + 1
    #else:
    #    current_sacred_run_id = 1

    #fold_dir = f"{result_dir}/{int(current_sacred_run_id)}/folds"
    #if not os.path.isdir(fold_dir):
    #    os.makedirs(fold_dir)

    #os.makedirs(fold_dir, exist_ok=True)

    #run_training_loop(num_loops=n_folds, fold_dir=fold_dir)