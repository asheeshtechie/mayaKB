import os

import pytest

torch = pytest.importorskip("torch")

from mayakb.core import registry  # noqa: E402
from mayakb.retrieval.rerankers.cross_encoder import CrossEncoderReranker  # noqa: E402


class FakeEncoder:
    def __init__(self):
        self.calls = []

    def predict(self, pairs, batch_size, activation_fn, show_progress_bar):
        self.calls.append((pairs, batch_size, activation_fn))
        logits = torch.tensor([float(len(text)) - 5.0 for _, text in pairs])
        return activation_fn(logits).numpy()


def test_scores_are_probabilities_in_input_order():
    encoder = FakeEncoder()
    reranker = CrossEncoderReranker(encoder=encoder, batch_size=8)
    scores = reranker.score("refunds", ["short", "a much longer passage"])
    assert 0.0 < scores[0] < scores[1] < 1.0
    pairs, batch_size, activation = encoder.calls[0]
    assert pairs == [("refunds", "short"), ("refunds", "a much longer passage")]
    assert batch_size == 8
    assert isinstance(activation, torch.nn.Sigmoid)


def test_no_texts_no_call():
    encoder = FakeEncoder()
    assert CrossEncoderReranker(encoder=encoder).score("q", []) == []
    assert encoder.calls == []


def test_registered_as_cross_encoder():
    assert registry.get("reranker", "cross_encoder") is CrossEncoderReranker


@pytest.mark.integration
@pytest.mark.skipif(not os.environ.get("MAYAKB_TEST_RERANKER"), reason="set MAYAKB_TEST_RERANKER=1 to use the model")
def test_real_cross_encoder_prefers_the_relevant_passage():
    reranker = CrossEncoderReranker()
    relevant, unrelated = reranker.score(
        "How long do refunds take?",
        ["Refunds are issued within 30 days of the return.", "Our office is closed on public holidays."],
    )
    assert relevant > unrelated
    assert 0.0 <= unrelated <= relevant <= 1.0
