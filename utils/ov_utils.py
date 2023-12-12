"""
Functions toconvert and proces Open VINO models
adapted from https://docs.openvino.ai/2023.0/notebooks/notebooks.html
"""

import torch
from ultralytics.utils.metrics import ConfusionMatrix
from tqdm.notebook import tqdm


# def compile_ovmodel(model, core=None, device=None):
#     if device != "CPU":
#         model.reshape({0: [1, 3, -1, -1]})
#     compiled_model = core.compile_model(model, device)
#     return compiled_model

def compile_ovmodel(ovmodel=None, core=None, device=None):
    if device != "CPU":
        ovmodel.reshape(({0: [1, 3, 640, 640]}))
        compiled = core.compile_model(ovmodel, device, {"PERFORMANCE_HINT": "THROUGHPUT"})
    else:
        compiled = core.compile_model(ovmodel, device)

    return compiled


def test_ov_model(model=None, core=None, data_loader=None, validator=None, num_samples=None,
                  device=None):
    """
    Runs YOLOv8 openvino model validation on dataset and \
        returns metrics
    Parameters:
        model (Model): OpenVINO model
        core openvino core
        data_loader (torch.utils.data.DataLoader): dataset loader
        validator: instance of validator class
        num_samples (int, *optional*, None): validate model only on specified number samples,\
              if provided
    Returns:
        stats: (Dict[str, float]) - dictionary with aggregated accuracy metrics statistics, \
            key is metric name, value is metric value
    """
    validator.seen = 0
    validator.jdict = []
    validator.stats = []
    validator.batch_i = 1
    validator.confusion_matrix = ConfusionMatrix(nc=validator.nc, conf=validator.args.conf)
    compiled_model = compile_ovmodel(model, core=core, device=device)

    for batch_i, batch in enumerate(tqdm(data_loader, total=num_samples)):
        if num_samples is not None and batch_i == num_samples:
            break
        batch = validator.preprocess(batch)
        results = compiled_model(batch["img"])
        preds = torch.from_numpy(results[compiled_model.output(0)])
        preds = validator.postprocess(preds)
        validator.update_metrics(preds, batch)
    stats = validator.get_stats()
    return stats


def print_stats(stats, total_images: int, total_objects: int):
    """
    Helper function for printing accuracy statistic
    Parameters:
        stats: (Dict[str, float]) - dictionary with aggregated accuracy metrics 
        statistics, key is metric name, value is metric value
        total_images (int) -  number of evaluated images
        total_objects (int)
    Returns:
        None
    """
    print("Boxes:")
    mp, mr = stats['metrics/precision(B)'], stats['metrics/recall(B)']
    map50, mean_ap = stats['metrics/mAP50(B)'], stats['metrics/mAP50-95(B)']
    # Print results
    s = ('%20s' + '%12s' * 6) % (
        'Class', 'Images', 'Labels', 'Precision', 'Recall', 'mAP@.5', 'mAP@.5:.95')
    print(s)
    pf = '%20s' + '%12i' * 2 + '%12.3g' * 4  # print format
    print(pf % ('all', total_images, total_objects, mp, mr, map50, mean_ap))
    if 'metrics/precision(M)' in stats:
        s_mp, s_mr, s_map50, s_mean_ap = stats['metrics/precision(M)'], \
                                         stats['metrics/recall(M)'], stats['metrics/mAP50(M)'], \
                                         stats['metrics/mAP50-95(M)']
        # Print results
        s = ('%20s' + '%12s' * 6) % ('Class', 'Images', 'Labels', 'Precision', 'Recall',
                                     'mAP@.5', 'mAP@.5:.95')
        print(s)
        pf = '%20s' + '%12i' * 2 + '%12.3g' * 4  # print format
        print(pf % ('all', total_images, total_objects, s_mp, s_mr, s_map50, s_mean_ap))