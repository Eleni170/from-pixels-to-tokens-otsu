from minicheck.minicheck import MiniCheck


def get_mini_check_results(contexts, claims):
    scorer = MiniCheck(model_name="deberta-v3-large")
    _, raw_prob, _, _ = scorer.score(docs=contexts, claims=claims)

    return raw_prob
