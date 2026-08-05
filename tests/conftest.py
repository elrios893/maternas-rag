from __future__ import annotations

from unittest.mock import MagicMock, patch

import pytest


@pytest.fixture(autouse=True)
def mock_settings():
    with patch("src.settings.settings") as mock:
        mock.groq_api_key = "test-key"
        mock.groq_model = "test-model"
        yield mock


@pytest.fixture
def mock_groq_client():
    patcher1 = patch("src.classifiers.risk_detector.Groq", autospec=True)
    patcher2 = patch("src.classifiers.intent_classifier.Groq", autospec=True)
    mock_groq_cls1 = patcher1.start()
    mock_groq_cls2 = patcher2.start()
    mock_client = MagicMock()
    mock_groq_cls1.return_value = mock_client
    mock_groq_cls2.return_value = mock_client
    yield mock_client
    patcher1.stop()
    patcher2.stop()


@pytest.fixture(autouse=True)
def reset_cached_clients():
    import src.classifiers.risk_detector as rd
    import src.classifiers.intent_classifier as ic
    rd._groq_client = None
    ic._groq_client = None
    yield
    rd._groq_client = None
    ic._groq_client = None


def make_groq_response(text: str) -> MagicMock:
    choice = MagicMock()
    choice.message.content = text
    response = MagicMock()
    response.choices = [choice]
    return response
