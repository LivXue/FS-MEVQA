import torch
from torchvision.ops import box_iou


def find_box_token(res):
    i0 = res.find('[box]')
    if i0 == -1:
        return []
    else:
        inds = [i0]

    while (i := res[inds[-1] + 1:].find('[box]')) != -1:
        inds.append(i + inds[-1] + 1)

    return inds


def find_obj(res, obj):
    lower_res = res.lower()
    box_inds = find_box_token(lower_res)
    obj_ids = []
    if obj == 'LOC':
        obj = 'located here'
    for i, ind in enumerate(box_inds):
        if (' ' + lower_res[:ind-1]).endswith(' ' + obj):
            obj_ids.append(i)

    return obj_ids


def vis_eval(exp_gts, exp_res):
    gts = {k: v['boxes'] for k, v in exp_gts.items()}
    res = exp_res

    scores = []
    for k in gts:
        cur_score = []
        gt, re = gts[k], res[k]
        re_exp = re['explanation'].replace('.', '').replace(';', '').replace(',', '').lower()
        gt_obj_n = 0
        for obj in gt:
            obj_ids = find_obj(re_exp, obj)
            min_n = min(len(gt[obj]), len(obj_ids))
            gt_obj_n += len(gt[obj])
            obj_ids = obj_ids[:min_n]
            for i, id in enumerate(obj_ids):
                if id >= len(re['boxes']) or re['boxes'][id] == []:
                    cur_score.append(0)
                    continue
                iou = box_iou(torch.tensor(gt[obj][i]), torch.tensor(re['boxes'][id]))
                iou = iou.max(dim=1)[0]
                iou = iou.mean()
                cur_score.append(iou)

            # compute empty sets
            cur_score.extend([0] * (max(gt_obj_n, len(re['boxes'])) - len(cur_score)))

        if len(cur_score) == 0:
            scores.append(0)
        else:
            scores.append(sum(cur_score) / len(cur_score))

    if len(scores) == 0:
        return 0
    else:
        return sum(scores) / len(scores) * 100
