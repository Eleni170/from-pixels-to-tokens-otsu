import nltk
import alignscore

from common.params import *

nltk.download("punkt_tab")


def get_align_score_results(contexts, claims):
    scorer = alignscore.AlignScore(
        model="roberta-large",
        device=DEVICE,
        batch_size=32,
        ckpt_path="https://huggingface.co/yzha/AlignScore/resolve/main/AlignScore-large.ckpt",
        evaluation_mode="nli_sp"
    )

    scores = scorer.score(contexts=contexts, claims=claims)

    return scores
