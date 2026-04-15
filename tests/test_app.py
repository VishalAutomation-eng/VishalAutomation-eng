import unittest
from types import SimpleNamespace
from unittest.mock import Mock, patch

import common
import graph_rag
import main
import old_rag
import page_rag


def fake_embedding_vectors(texts):
    return [[float(len(t) % 11), float((len(t) * 2) % 7), 1.0] for t in texts]


class TestTextHelpers(unittest.TestCase):
    @patch("common.importlib.util.find_spec", side_effect=ModuleNotFoundError)
    def test_load_optional_module_nested_missing_parent(self, _mock_find):
        self.assertIsNone(common.load_optional_module("pdfminer.high_level"))

    def test_pdf_backend_status_shape(self):
        status = common.get_pdf_backend_status()
        self.assertIn("PyMuPDF", status)
        self.assertIn("pypdf", status)
        self.assertIn("pdfminer.six", status)

    def test_preprocess_text(self):
        text = "Hello\x00   world\n\nfrom\tPDF"
        self.assertEqual(common.preprocess_text(text), "Hello world from PDF")


class TestEmbeddingPipelines(unittest.TestCase):
    @patch("common.call_embedding_api", side_effect=lambda texts, **_: fake_embedding_vectors(texts))
    def test_semantic_chunk_text(self, _mock_embed):
        text = "A quick sentence. Another similar sentence. New topic starts here."
        chunks = common.semantic_chunk_text(text, max_sentences_per_chunk=2, similarity_threshold=-1)
        self.assertGreaterEqual(len(chunks), 2)

    @patch("common.call_embedding_api", side_effect=lambda texts, **_: fake_embedding_vectors(texts))
    def test_embedding_index_search(self, _mock_embed):
        idx = common.EmbeddingIndex([
            {"content": "python basics and loops", "metadata": {"id": 1}},
            {"content": "graph databases and nodes", "metadata": {"id": 2}},
        ])
        results = idx.search("python", k=1)
        self.assertEqual(len(results), 1)

    @patch("common.call_embedding_api", side_effect=lambda texts, **_: fake_embedding_vectors(texts))
    @patch("common.load_optional_module")
    def test_build_vector_index_fallback(self, mock_load_optional, _mock_embed):
        def side(name):
            if name in ("qdrant_client", "qdrant_client.models"):
                return None
            return object()
        mock_load_optional.side_effect = side
        idx = common.build_vector_index([{"content": "abc", "metadata": {}}])
        self.assertIsInstance(idx, common.EmbeddingIndex)

    @patch("old_rag.semantic_chunk_text", return_value=["chunk one", "chunk two"])
    @patch("common.call_embedding_api", side_effect=lambda texts, **_: fake_embedding_vectors(texts))
    def test_old_rag_build(self, _embed, _chunk):
        index = old_rag.build_old_rag_index(["page one", "page two"])
        self.assertTrue(hasattr(index, "search"))

    @patch("common.call_embedding_api", side_effect=lambda texts, **_: fake_embedding_vectors(texts))
    def test_page_rag_build(self, _embed):
        index = page_rag.build_page_rag_index(["alpha", "beta"])
        self.assertTrue(hasattr(index, "search"))

    @patch("graph_rag.call_embedding_api", side_effect=lambda texts, **_: fake_embedding_vectors(texts))
    def test_graph_rag_hybrid_search(self, _embed):
        idx = graph_rag.GraphRAGIndex(["Microsoft acquired GitHub. Satya announced vision."])
        out = idx.search("What did Microsoft acquire?", k=1)
        self.assertEqual(len(out), 1)


class TestExceptions(unittest.TestCase):
    @patch("common.extract_with_pdfminer", return_value=[])
    @patch("common.extract_with_pypdf", return_value=[])
    @patch("common.extract_with_pymupdf", return_value=[])
    def test_extract_pdf_pages_raises_when_no_text(self, *_mocks):
        with self.assertRaises(common.PDFExtractionError):
            common.extract_pdf_pages(b"fake-pdf")

    @patch("main.require_module")
    def test_call_qwen_malformed_response(self, mock_require_module):
        mock_response = Mock()
        mock_response.raise_for_status.return_value = None
        mock_response.json.return_value = {"unexpected": "format"}
        mock_response.text = '{"unexpected":"format"}'

        class FakeRequestException(Exception):
            pass

        fake_requests = SimpleNamespace(post=Mock(return_value=mock_response), RequestException=FakeRequestException)
        mock_require_module.return_value = fake_requests

        with self.assertRaises(main.LLMRequestError):
            main.call_qwen("http://example.com/v1", "model", "q", [])


if __name__ == "__main__":
    unittest.main()
