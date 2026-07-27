from typing import Dict, List, Sequence, Tuple

from seqeval.metrics import classification_report, f1_score, precision_score, recall_score


def ids_to_tag_sequences(
    pred_ids: Sequence[Sequence[int]],
    gold_ids: Sequence[Sequence[int]],
    mask: Sequence[Sequence[bool]],
    id2label: Dict[int, str],
) -> Tuple[List[List[str]], List[List[str]]]:
    pred_tags: List[List[str]] = []
    gold_tags: List[List[str]] = []

    for p_seq, g_seq, m_seq in zip(pred_ids, gold_ids, mask):
        valid_len = int(sum(bool(x) for x in m_seq))
        p_tags = [id2label[int(tag_id)] for tag_id in p_seq[:valid_len]]
        g_tags = [id2label[int(tag_id)] for tag_id in g_seq[:valid_len]]
        pred_tags.append(p_tags)
        gold_tags.append(g_tags)

    return pred_tags, gold_tags


def compute_ner_metrics(pred_tags: List[List[str]], gold_tags: List[List[str]]) -> Dict[str, float]:
    return {
        "precision": float(precision_score(gold_tags, pred_tags)),
        "recall": float(recall_score(gold_tags, pred_tags)),
        "f1": float(f1_score(gold_tags, pred_tags)),
    }


def ner_classification_report(pred_tags: List[List[str]], gold_tags: List[List[str]]) -> str:
    return classification_report(gold_tags, pred_tags, digits=4)
