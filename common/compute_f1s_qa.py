import spacy

nlp = spacy.load("en_core_web_sm")


def _normalize_text(text):
    doc = nlp(text.lower())

    tokens = [token.lemma_ for token in doc if not token.is_stop and not token.is_punct]

    return tokens


def compute_f1s_qa(predictions, references):
    f1s_qa = []
    for prediction, reference in zip(predictions, references):

        prediction = _normalize_text(prediction)
        reference = _normalize_text(reference)

        if len(prediction) == 0 or len(reference) == 0:
            f1s_qa.append(int(prediction == reference))
            continue

        common_tokens = set(prediction) & set(reference)

        if len(common_tokens) == 0:
            f1s_qa.append(0)
            continue

        precision = len(common_tokens) / len(prediction)
        recall = len(common_tokens) / len(reference)

        f1s_qa.append(2 * (precision * recall) / (precision + recall))

    return f1s_qa
