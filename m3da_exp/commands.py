from collections import defaultdict
from functools import partial
from typing import Sequence, Callable

import numpy as np
import torch
from deli import save_json, save_numpy
from dpipe.im.box import get_centered_box
from dpipe.im.shape_ops import crop_to_box
from dpipe.itertools import zip_equal
from dpipe.train import train
from tqdm import tqdm

from m3da_exp.batch_iter import SPATIAL_DIMS
from m3da_exp.predict.tta import predict_tta
from m3da_exp.predict.utils import load_x_hm, zoom_pred2true
from m3da_exp.torch.model import train_step_adabn_dummy
from m3da_exp.utils import flush, path_mkdir


def skip_predict(output_path):
    flush(f'>>> Passing the step of saving predictions into `{output_path}`')
    _ = path_mkdir(output_path)


def predict_to_folder(load_y, load_x, predict, test_ids, results_path, dims=SPATIAL_DIMS, zoom_order=1):
    results_path = path_mkdir(results_path)
    for _id in tqdm(test_ids):
        pred = zoom_pred2true(pred=predict(load_x(_id)), true=load_y(_id), dims=dims, order=zoom_order)
        save_numpy(np.uint8(np.argmax(pred, axis=0)), results_path / f'{_id}.npy.gz', compression=1)


def embed_to_folder(load_x, architecture, test_ids, results_path, patch_size, device):
    results_path = path_mkdir(results_path)
    for _id in tqdm(test_ids):
        x = load_x(_id)
        x = crop_to_box(x, get_centered_box(np.asarray(x.shape) // 2, patch_size), padding_values=np.min)
        _, emb = architecture.forward(torch.from_numpy(x[None, None, :]).to(device), return_features=True)
        save_numpy(np.float16(np.ravel(emb.numpy(force=True))), results_path / f'{_id}.npy.gz', compression=1)


def compute_test_metrics(load_y, load_x, predict, metrics, test_ids, results_path, random_state: np.random.RandomState,
                         dims=SPATIAL_DIMS, zoom_order=1, n_tta_iter=1, flip_dims=SPATIAL_DIMS, rot_dims=(-3, -2)):
    results_path = path_mkdir(results_path)
    results, results_tta = defaultdict(dict), defaultdict(dict)

    for _id in tqdm(test_ids):
        # 1. Single predict
        true, img = load_y(_id), load_x(_id)
        pred = zoom_pred2true(pred=predict(img), true=true, dims=dims, order=zoom_order)
        for metric_name, metric in metrics.items():
            results[metric_name][_id] = metric(true, pred)

        # 2. TTA predict
        if n_tta_iter > 1:
            pred_tta = predict_tta(img, true, predict, pred, n_tta_iter - 1, zoom_dims=dims, order=zoom_order,
                                   random_state=random_state, flip_dims=flip_dims, rot_dims=rot_dims)
            for metric_name, metric in metrics.items():
                results_tta[metric_name][_id] = metric(true, pred_tta)

    for metric_name, result in results.items():
        save_json(result, results_path / f'{metric_name}.json', indent=0)
    for metric_name, result in results_tta.items():
        save_json(result, results_path / f'tta_{metric_name}.json', indent=0)


def compute_test_metrics_hm(load_y, load_x, predict, metrics, test_ids, train_ids, results_path,
                            random_state: np.random.RandomState, dims=SPATIAL_DIMS, zoom_order=1, n_tta_iter=1,
                            flip_dims=SPATIAL_DIMS, rot_dims=(-3, -2)):
    _load_x = partial(load_x_hm, load_x=load_x, train_ids=train_ids, random_state=random_state)
    compute_test_metrics(load_y, _load_x, predict, metrics, test_ids, results_path, random_state, dims=dims,
                         zoom_order=zoom_order, n_tta_iter=n_tta_iter, flip_dims=flip_dims, rot_dims=rot_dims)


def compute_test_metrics_adabn_random(load_y, load_x, predict, metrics, test_ids, results_path, get_batch_iter, scaler,
                                      architecture, random_state: np.random.RandomState, dims=SPATIAL_DIMS,
                                      zoom_order=1, n_tta_iter=1, flip_dims=SPATIAL_DIMS, rot_dims=(-3, -2)):
    # create batch_iter and run several forward steps:
    batch_iter = get_batch_iter()
    train(train_step=partial(train_step_adabn_dummy, scaler=scaler),
          batch_iter=batch_iter, n_epochs=1, architecture=architecture)
    architecture.eval()

    compute_test_metrics(load_y, load_x, predict, metrics, test_ids, results_path, random_state, dims=dims,
                         zoom_order=zoom_order, n_tta_iter=n_tta_iter, flip_dims=flip_dims, rot_dims=rot_dims)


def aggregate_metric(xs, ys, metric, aggregate_fn=np.mean):
    return aggregate_fn([metric(x, y) for x, y in zip_equal(xs, ys)])


def compute_val_metrics(predict: Callable, load_x: Callable, load_y: Callable, ids: Sequence[str], metrics: dict):
    y_true = list(map(load_y, ids))
    y_pred = [predict(load_x(i)) for i in ids]
    return {name: metric(y_true, y_pred) for name, metric in metrics.items()}
