import json
import warnings
from tqdm import tqdm
from scipy.ndimage import gaussian_filter
import numpy as np
import torch
import torch.nn.functional as F

from utils import get_residual_features, get_matched_ref_features
from utils import calculate_metrics, load_and_transform_vision_data
from ad_models.patchcore import get_patchcore
from datasets.mvtec import MVTEC
from datasets.visa import VISA
from datasets.btad import BTAD
from datasets.mvtec_3d import MVTEC3D
from datasets.mpdd import MPDD


warnings.filterwarnings('ignore')


def validate(args, encoder, projectors, normal_loader, test_loader, ref_features, device, class_name, residual=True, pro=False, dtype=torch.float):
    if projectors is not None:
        projectors.eval()
    
    patchcore_model = get_patchcore(encoder, projectors, device=device, dtype=dtype, residual=residual)
    patchcore_model.fit(normal_loader, ref_features, class_name, aligned=False)
    img_scores, pix_scores, gt_labels, gt_masks = patchcore_model.predict(
        test_loader, ref_features, class_name, aligned=False
    )
    img_scores = np.array(img_scores)
    pix_scores = np.asarray(pix_scores)
    gt_labels = np.concatenate(gt_labels)
    gt_masks = np.concatenate(gt_masks, axis=0)
    img_auc, img_ap, img_f1_score, pix_auc, pix_ap, pix_f1_score, pix_aupro = calculate_metrics(img_scores, pix_scores, gt_labels, gt_masks, pro=pro, only_max_value=True)
    
    metrics_norm = calculate_scores_by_feature_norm(args, encoder, projectors, test_loader, ref_features, device, class_name, 
                                                    residual=residual, pro=pro, dtype=dtype)
    img_auc_norm, img_ap_norm, img_f1_score_norm, pix_auc_norm, pix_ap_norm, pix_f1_score_norm, pix_aupro_norm = metrics_norm
    metrics = {}
    metrics['metrics'] = [img_auc, img_ap, img_f1_score, pix_auc, pix_ap, pix_f1_score, pix_aupro]
    metrics['metrics-norm'] = [img_auc_norm, img_ap_norm, img_f1_score_norm, pix_auc_norm, pix_ap_norm, pix_f1_score_norm, pix_aupro_norm]
    
    return metrics


def calculate_scores_by_feature_norm(args, encoder, projectors, test_loader, ref_features, device, class_name, residual=True, aligned=False, pro=False, dtype=torch.float):
    if aligned:
        if class_name in MVTEC.CLASS_NAMES:
            with open(f'./aligned/references/mvtec/{class_name}.json', 'r', encoding='utf-8') as f:
                matched_reference_paths = json.load(f)
        elif class_name in VISA.CLASS_NAMES:
            with open(f'./aligned/references/visa/{class_name}.json', 'r', encoding='utf-8') as f:
                matched_reference_paths = json.load(f)
        elif class_name in BTAD.CLASS_NAMES:
            with open(f'./aligned/references/btad/{class_name}.json', 'r', encoding='utf-8') as f:
                matched_reference_paths = json.load(f)
        elif class_name in MVTEC3D.CLASS_NAMES:
            with open(f'./aligned/references/mvtec3d/{class_name}.json', 'r', encoding='utf-8') as f:
                matched_reference_paths = json.load(f)
        elif class_name in MPDD.CLASS_NAMES:
            with open(f'./aligned/references/mpdd/{class_name}.json', 'r', encoding='utf-8') as f:
                matched_reference_paths = json.load(f)
                
    label_list, gt_mask_list = [], []
    s_list = [list() for _ in range(args.feature_levels)]
    progress_bar = tqdm(total=len(test_loader))
    progress_bar.set_description(f"Evaluating")
    for idx, batch in enumerate(test_loader):
        progress_bar.update(1)
        
        image, label, mask, _ = batch    
        gt_mask_list.append(mask.squeeze(1).cpu().numpy().astype(bool))
        label_list.append(label.cpu().numpy().astype(bool).ravel())
        
        image = image.to(dtype=dtype, device=device)
        size = image.shape[-1]
        
        if aligned:
            image_path = test_loader.dataset.image_paths[idx]
            rimage_paths = matched_reference_paths[image_path]
            rimages = load_and_transform_vision_data(rimage_paths, device, encoder.name)
            with torch.no_grad():
                features = encoder.encode_image_from_tensors(rimages.to(dtype=dtype, device=device), shape='seq')
                for l in range(len(features)):
                    _, _, c = features[l].shape
                    features[l] = features[l].reshape(-1, c)
            ref_features = features
        
        with torch.no_grad():
            features = encoder.encode_image_from_tensors(image, shape='img')
            
            if residual:
                mfeatures = get_matched_ref_features(features, ref_features)
                rfeatures = get_residual_features(features, mfeatures)
                if projectors is not None:
                    features = projectors(*rfeatures)
                else:
                    features = rfeatures
            else:
                if projectors is not None:
                    features = projectors(*features)
            
            for l in range(len(features)):
                e = features[l]  
                bs, dim, h, w = e.size()
                e = e.permute(0, 2, 3, 1).reshape(-1, dim)
                s = e.norm(dim=1)
                s = torch.sqrt(s + 1) - 1
                s = s.to(torch.float)

                s_list[l].append(s.reshape(bs, h, w))
    
    progress_bar.close()
    
    labels = np.concatenate(label_list)
    gt_masks = np.concatenate(gt_mask_list, axis=0)
    
    scores = aggregate_anomaly_scores(s_list, feature_levels=args.feature_levels, class_name=class_name, size=size)
    img_auc, img_ap, img_f1_score, pix_auc, pix_ap, pix_f1_score, pix_aupro = calculate_metrics(None, scores, labels, gt_masks, pro=pro, only_max_value=True)

    return img_auc, img_ap, img_f1_score, pix_auc, pix_ap, pix_f1_score, pix_aupro


def aggregate_anomaly_scores(logps_list, feature_levels=3, class_name=None, size=224):
    abnormal_map = [list() for _ in range(feature_levels)]
    for l in range(feature_levels):
        probs = torch.cat(logps_list[l], dim=0)  
        # upsample
        abnormal_map[l] = F.interpolate(probs.unsqueeze(1),
            size=size, mode='bilinear', align_corners=True).squeeze().cpu().numpy()
    
    # score aggregation
    scores = np.zeros_like(abnormal_map[0])
    for l in range(feature_levels):
        scores += abnormal_map[l]
    scores /= feature_levels
    
    for i in range(scores.shape[0]):
        scores[i] = gaussian_filter(scores[i], sigma=4)

    return scores