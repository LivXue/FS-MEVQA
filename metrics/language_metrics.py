from .pycocoevalcap.tokenizer.ptbtokenizer import PTBTokenizer
from .pycocoevalcap.bleu.bleu import Bleu
from .pycocoevalcap.meteor.meteor import Meteor
from .pycocoevalcap.rouge.rouge import Rouge
from .pycocoevalcap.cider.cider import Cider
from .pycocoevalcap.spice.spice import Spice


def coco_eval(gts, res):
    if isinstance(gts, list):
        gts = {i: [gts[i]] for i in range(len(gts))}
    if isinstance(res, list):
        res = {i: [res[i]] for i in range(len(res))}
    assert gts.keys() == res.keys(), "ERROR: The keys of references and ground truths are unequal!"

    print('tokenization...')
    tokenizer = PTBTokenizer()
    gts = {k: [{'caption': v}] for k, v in gts.items()}
    res = {k: [{'caption': v}] for k, v in res.items()}
    gts = tokenizer.tokenize(gts)
    res = tokenizer.tokenize(res)

    # =================================================
    # Set up scorers
    # =================================================
    print('setting up scorers...')
    scorers = [
        (Bleu(4), ["Bleu_1", "Bleu_2", "Bleu_3", "Bleu_4"]),
        (Meteor(), "METEOR"),
        (Rouge(), "ROUGE_L"),
        (Cider(), "CIDEr"),
        (Spice(), "SPICE")  # comment SPICE for faster prototyping
    ]

    report_score = dict()
    # =================================================
    # Compute scores
    # =================================================
    for scorer, method in scorers:
        print('computing %s score...' % (scorer.method()))
        score, scores = scorer.compute_score(gts, res)
        if type(method) == list:
            for sc, scs, m in zip(score, scores, method):
                report_score[m] = sc * 100
        else:
            report_score[method] = score * 100
    return report_score


def lang_eval(exp_gts, exp_res):
    gts = {k: v['explanation'] for k, v in exp_gts.items()}
    res = {k: v['explanation'] for k, v in exp_res.items()}

    lang_scores = coco_eval(gts, res)

    return lang_scores
