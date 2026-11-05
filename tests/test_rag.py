from types import SimpleNamespace

from rag import Chunk, answer, chunk_text


def test_chunks_cover_text_with_overlap():
    text = " ".join(f"word{i}" for i in range(600))
    chunks = chunk_text(text, page=1, size=200, overlap=50)
    assert all(len(c.text) <= 200 for c in chunks)
    assert chunks[0].text.startswith("word0") and chunks[-1].text.endswith("word599")
    # consecutive chunks share some text
    assert chunks[0].text.split()[-1] in chunks[1].text


def test_empty_page_gives_no_chunks():
    assert chunk_text("   ", page=3) == []


def test_answer_sends_page_tagged_context():
    seen = {}

    class FakeClient:
        def chat_completion(self, **kw):
            seen.update(kw)
            msg = SimpleNamespace(content="42 [p. 7]")
            return SimpleNamespace(choices=[SimpleNamespace(message=msg)])

    out = answer("What is it?", [Chunk("The answer is 42.", 7)], client=FakeClient())
    assert out == "42 [p. 7]"
    assert "[p. 7] The answer is 42." in seen["messages"][0]["content"]
