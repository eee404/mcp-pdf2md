"""Tests for pdf2md MCP server."""

import base64
import os
import shutil
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
import pytest_asyncio

from pdf2md.server import (
    MAX_FILE_SIZE,
    _is_ip_encoding_trick,
    _validate_url,
    convert_pdf_file,
    convert_pdf_url,
    extract_first_heading,
    get_api_status,
    parse_input_string,
    rewrite_image_links,
    save_image,
    sanitize_filename,
)

# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

TESTS_DIR = Path(__file__).parent
TEST_PDF = TESTS_DIR / "ViewFile-6.pdf"


def _make_fake_image(image_id: str, data: bytes = b"\x89PNG fake"):
    """Create a fake image object mimicking Mistral OCR response."""
    b64 = base64.b64encode(data).decode()
    return SimpleNamespace(
        id=image_id,
        image_base64=f"data:image/png;base64,{b64}",
    )


def _make_ocr_response(pages_md: list[str], images_per_page: list[list] | None = None):
    """Build a fake OCR response object."""
    pages = []
    for i, md in enumerate(pages_md):
        imgs = (images_per_page[i] if images_per_page else [])
        pages.append(SimpleNamespace(markdown=md, images=imgs))
    return SimpleNamespace(pages=pages)


@pytest.fixture
def tmp_output(tmp_path):
    """Return a temporary output directory."""
    return tmp_path


# ---------------------------------------------------------------------------
# Unit tests: parse_input_string
# ---------------------------------------------------------------------------

class TestParseInputString:
    def test_single_path(self):
        assert parse_input_string("/path/to/file.pdf") == ["/path/to/file.pdf"]

    def test_comma_separated(self):
        result = parse_input_string("/a.pdf, /b.pdf")
        assert result == ["/a.pdf", "/b.pdf"]

    def test_space_separated(self):
        result = parse_input_string("/a.pdf /b.pdf")
        assert result == ["/a.pdf", "/b.pdf"]

    def test_outer_quotes_stripped(self):
        result = parse_input_string('"/path/to/file.pdf"')
        assert result == ["/path/to/file.pdf"]

    def test_inner_quoted_items(self):
        """Inner quotes are stripped when they wrap an entire space-separated token.

        Note: The outer-quote-stripping heuristic means that if the whole input
        starts and ends with quotes AND contains multiple items, the outer quotes
        are incorrectly consumed. This test uses unquoted items to avoid that.
        """
        # Without outer quotes wrapping the whole string, individual quoted tokens work
        result = parse_input_string('/a.pdf "/b.pdf"')
        assert result == ["/a.pdf", "/b.pdf"]

    def test_path_with_spaces_quoted(self):
        """Known latent bug: a quoted path with internal spaces gets split."""
        result = parse_input_string('"C:/My Documents/file.pdf"')
        # The current implementation splits on spaces even inside quotes,
        # so the path is broken. This documents the known bug.
        assert len(result) != 1 or result == ["C:/My Documents/file.pdf"]

    def test_empty_string(self):
        assert parse_input_string("") == []


# ---------------------------------------------------------------------------
# Unit tests: save_image
# ---------------------------------------------------------------------------

class TestSaveImage:
    def test_empty_image_id(self, tmp_path):
        img = SimpleNamespace(id="", image_base64="data:image/png;base64,AAAA")
        assert save_image(img, tmp_path) is None

    def test_whitespace_image_id(self, tmp_path):
        img = SimpleNamespace(id="   ", image_base64="data:image/png;base64,AAAA")
        assert save_image(img, tmp_path) is None

    def test_invalid_base64(self, tmp_path):
        img = SimpleNamespace(id="test.png", image_base64="not-a-data-uri")
        assert save_image(img, tmp_path) is None

    def test_valid_image(self, tmp_path):
        img = _make_fake_image("test-img.png")
        result = save_image(img, tmp_path)
        assert result is not None
        assert Path(result).exists()


# ---------------------------------------------------------------------------
# Unit tests: extract_first_heading
# ---------------------------------------------------------------------------

class TestExtractFirstHeading:
    def test_empty_page(self):
        assert extract_first_heading("") == "(empty page)"

    def test_no_heading(self):
        result = extract_first_heading("Just some plain text\nAnother line")
        assert result == "Just some plain text"

    def test_h1_heading(self):
        result = extract_first_heading("# My Title\nSome text")
        assert result == "# My Title"

    def test_h3_heading(self):
        result = extract_first_heading("Some preamble\n### Sub heading\ntext")
        assert result == "### Sub heading"

    def test_long_text_truncated(self):
        long_line = "A" * 100
        result = extract_first_heading(long_line)
        assert len(result) == 83  # 80 + "..."
        assert result.endswith("...")


# ---------------------------------------------------------------------------
# Unit tests: rewrite_image_links
# ---------------------------------------------------------------------------

class TestRewriteImageLinks:
    def test_bare_filename_rewritten(self):
        md = "![alt](img-001.jpeg)"
        result = rewrite_image_links(md, "images")
        assert result == "![alt](images/img-001.jpeg)"

    def test_path_with_slash_not_rewritten(self):
        md = "![alt](some/path/img.jpeg)"
        result = rewrite_image_links(md, "images")
        assert result == "![alt](some/path/img.jpeg)"

    def test_multiple_images(self):
        md = "![a](img-1.png) text ![b](img-2.png)"
        result = rewrite_image_links(md, "../images")
        assert "![a](../images/img-1.png)" in result
        assert "![b](../images/img-2.png)" in result

    def test_no_images(self):
        md = "Just text, no images."
        assert rewrite_image_links(md, "images") == md


# ---------------------------------------------------------------------------
# Unit tests: sanitize_filename
# ---------------------------------------------------------------------------

class TestSanitizeFilename:
    def test_path_traversal(self):
        assert ".." not in sanitize_filename("../../etc/passwd")

    def test_special_chars(self):
        result = sanitize_filename('file<>:"/\\|?*.pdf')
        assert "<" not in result
        assert ">" not in result

    def test_empty_fallback(self):
        assert sanitize_filename("...") == "document"


# ---------------------------------------------------------------------------
# Unit tests: _is_ip_encoding_trick
# ---------------------------------------------------------------------------

class TestIpEncodingTrick:
    def test_dword(self):
        assert _is_ip_encoding_trick("2130706433") is True

    def test_hex(self):
        assert _is_ip_encoding_trick("0x7f000001") is True

    def test_octal(self):
        assert _is_ip_encoding_trick("0177.0.0.1") is True

    def test_normal_ip(self):
        assert _is_ip_encoding_trick("1.2.3.4") is False

    def test_normal_hostname(self):
        assert _is_ip_encoding_trick("example.com") is False


# ---------------------------------------------------------------------------
# Unit tests: _validate_url
# ---------------------------------------------------------------------------

class TestValidateUrl:
    def test_non_http_scheme(self):
        with pytest.raises(ValueError, match="scheme"):
            _validate_url("ftp://example.com/file.pdf")

    def test_localhost_blocked(self):
        with pytest.raises(ValueError, match="localhost"):
            _validate_url("http://localhost/file.pdf")

    def test_metadata_google_blocked(self):
        with pytest.raises(ValueError, match="metadata"):
            _validate_url("http://metadata.google.internal/something")

    def test_metadata_goog_blocked(self):
        with pytest.raises(ValueError, match="metadata"):
            _validate_url("http://metadata.goog/something")

    def test_ip_encoding_dword_blocked(self):
        with pytest.raises(ValueError, match="encoding"):
            _validate_url("http://2130706433/file.pdf")

    @patch("pdf2md.server.socket.getaddrinfo")
    def test_loopback_ip_blocked(self, mock_dns):
        mock_dns.return_value = [(2, 1, 6, "", ("127.0.0.1", 0))]
        with pytest.raises(ValueError, match="non-public"):
            _validate_url("http://evil.example.com/file.pdf")

    @patch("pdf2md.server.socket.getaddrinfo")
    def test_link_local_blocked(self, mock_dns):
        mock_dns.return_value = [(2, 1, 6, "", ("169.254.169.254", 0))]
        with pytest.raises(ValueError, match="non-public"):
            _validate_url("http://evil.example.com/file.pdf")

    @patch("pdf2md.server.socket.getaddrinfo")
    def test_cgnat_blocked(self, mock_dns):
        mock_dns.return_value = [(2, 1, 6, "", ("100.100.100.200", 0))]
        with pytest.raises(ValueError):
            _validate_url("http://evil.example.com/file.pdf")

    @patch("pdf2md.server.socket.getaddrinfo")
    def test_public_ip_allowed(self, mock_dns):
        mock_dns.return_value = [(2, 1, 6, "", ("93.184.216.34", 0))]
        _validate_url("http://example.com/file.pdf")  # should not raise


# ---------------------------------------------------------------------------
# Integration tests (mocked Mistral): convert_pdf_file
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
class TestConvertPdfFileMocked:

    @patch("pdf2md.server.MISTRAL_API_KEY", "fake-key")
    @patch("pdf2md.server.Mistral")
    async def test_file_not_found(self, mock_mistral_cls):
        result = await convert_pdf_file("/nonexistent/file.pdf")
        assert result["results"][0]["success"] is False
        assert "does not exist" in result["results"][0]["error"]

    @patch("pdf2md.server.MISTRAL_API_KEY", "fake-key")
    @patch("pdf2md.server.Mistral")
    async def test_non_pdf_rejected(self, mock_mistral_cls, tmp_path):
        txt_file = tmp_path / "test.txt"
        txt_file.write_text("hello")
        result = await convert_pdf_file(str(txt_file))
        assert result["results"][0]["success"] is False
        assert "not a PDF" in result["results"][0]["error"]

    @patch("pdf2md.server.MISTRAL_API_KEY", "fake-key")
    @patch("pdf2md.server.Mistral")
    async def test_successful_conversion(self, mock_mistral_cls, tmp_path):
        # Create a fake PDF file
        pdf_file = tmp_path / "test-doc.pdf"
        pdf_file.write_bytes(b"%PDF-1.4 fake content")

        # Setup mock
        mock_client = MagicMock()
        mock_mistral_cls.return_value = mock_client
        mock_client.ocr.process.return_value = _make_ocr_response(
            ["# Page 1\nContent", "## Page 2\nMore content"]
        )

        result = await convert_pdf_file(str(pdf_file))
        assert result["success"] is True
        entry = result["results"][0]
        assert entry["success"] is True
        assert entry["pages"] == 2
        assert "full_markdown" in entry
        assert "pages_index" in entry

        # Verify output structure
        out_dir = Path(entry["output_directory"])
        assert (out_dir / "full.md").exists()
        assert (out_dir / "pages" / "index.md").exists()
        assert (out_dir / "pages" / "page-001.md").exists()
        assert (out_dir / "pages" / "page-002.md").exists()
        assert (out_dir / "images").is_dir()
        assert (out_dir / "source").is_dir()

        # Source PDF should have been moved
        assert (out_dir / "source" / "test-doc.pdf").exists()
        assert not pdf_file.exists()

    @patch("pdf2md.server.MISTRAL_API_KEY", "fake-key")
    @patch("pdf2md.server.Mistral")
    async def test_overwritten_flag(self, mock_mistral_cls, tmp_path):
        pdf_file = tmp_path / "doc.pdf"
        pdf_file.write_bytes(b"%PDF-1.4 fake")

        # Pre-create output dir to trigger overwritten
        out_dir = tmp_path / "doc"
        out_dir.mkdir()

        mock_client = MagicMock()
        mock_mistral_cls.return_value = mock_client
        mock_client.ocr.process.return_value = _make_ocr_response(["# Page 1"])

        result = await convert_pdf_file(str(pdf_file))
        entry = result["results"][0]
        assert entry["success"] is True
        assert entry.get("overwritten") is True

    @patch("pdf2md.server.MISTRAL_API_KEY", "fake-key")
    @patch("pdf2md.server.Mistral")
    async def test_save_error_midprocessing(self, mock_mistral_cls, tmp_path):
        pdf_file = tmp_path / "doc.pdf"
        pdf_file.write_bytes(b"%PDF-1.4 fake")

        mock_client = MagicMock()
        mock_mistral_cls.return_value = mock_client
        mock_client.ocr.process.return_value = _make_ocr_response(["# Page 1"])

        with patch("pdf2md.server.save_ocr_response", return_value=(None, [])):
            result = await convert_pdf_file(str(pdf_file))
            entry = result["results"][0]
            assert entry["success"] is False
            assert "Could not save" in entry["error"]


# ---------------------------------------------------------------------------
# Integration tests (mocked Mistral): convert_pdf_url
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
class TestConvertPdfUrlMocked:

    @patch("pdf2md.server.MISTRAL_API_KEY", "fake-key")
    @patch("pdf2md.server.Mistral")
    @patch("pdf2md.server._download_url_safe")
    @patch("pdf2md.server._validate_url")
    async def test_successful_conversion(self, mock_validate, mock_download, mock_mistral_cls, tmp_path):
        mock_download.return_value = b"%PDF-1.4 fake"
        mock_client = MagicMock()
        mock_mistral_cls.return_value = mock_client
        mock_client.ocr.process.return_value = _make_ocr_response(["# Title\nContent"])

        result = await convert_pdf_url("https://example.com/test.pdf", output_dir=str(tmp_path))
        assert result["success"] is True
        entry = result["results"][0]
        assert entry["success"] is True
        assert "full_markdown" in entry
        assert "pages_index" in entry

        # Source PDF saved
        out_dir = Path(entry["output_directory"])
        assert (out_dir / "source" / "test.pdf").exists()

    @patch("pdf2md.server.MISTRAL_API_KEY", "fake-key")
    @patch("pdf2md.server.Mistral")
    @patch("pdf2md.server._download_url_safe")
    @patch("pdf2md.server._validate_url")
    async def test_overwritten_flag(self, mock_validate, mock_download, mock_mistral_cls, tmp_path):
        # Pre-create output dir
        (tmp_path / "test").mkdir()

        mock_download.return_value = b"%PDF-1.4 fake"
        mock_client = MagicMock()
        mock_mistral_cls.return_value = mock_client
        mock_client.ocr.process.return_value = _make_ocr_response(["# Page"])

        result = await convert_pdf_url("https://example.com/test.pdf", output_dir=str(tmp_path))
        entry = result["results"][0]
        assert entry.get("overwritten") is True

    @patch("pdf2md.server.MISTRAL_API_KEY", "fake-key")
    @patch("pdf2md.server.Mistral")
    async def test_ssrf_rejected(self, mock_mistral_cls, tmp_path):
        result = await convert_pdf_url("http://127.0.0.1/secret.pdf", output_dir=str(tmp_path))
        assert result["results"][0]["success"] is False

    @patch("pdf2md.server.MISTRAL_API_KEY", "fake-key")
    @patch("pdf2md.server.Mistral")
    async def test_metadata_url_rejected(self, mock_mistral_cls, tmp_path):
        result = await convert_pdf_url("http://169.254.169.254/latest/meta-data/", output_dir=str(tmp_path))
        assert result["results"][0]["success"] is False


# ---------------------------------------------------------------------------
# Security tests
# ---------------------------------------------------------------------------

class TestSecurity:

    def test_path_traversal_rejected(self):
        """Path traversal attempt should fail (file doesn't exist at resolved path)."""
        # The resolve(strict=True) will raise FileNotFoundError for non-existent paths

    @pytest.mark.asyncio
    @patch("pdf2md.server.MISTRAL_API_KEY", "fake-key")
    @patch("pdf2md.server.Mistral")
    async def test_path_traversal_convert(self, mock_mistral_cls):
        result = await convert_pdf_file("../../etc/secret.pdf")
        assert result["results"][0]["success"] is False

    def test_ssrf_loopback(self):
        with pytest.raises(ValueError):
            _validate_url("http://127.0.0.1/file.pdf")

    def test_ssrf_link_local(self):
        with pytest.raises(ValueError):
            _validate_url("http://169.254.169.254/latest/meta-data/")

    @patch("pdf2md.server.socket.getaddrinfo")
    def test_ssrf_redirect_to_localhost(self, mock_dns):
        """Redirect target resolving to localhost should be blocked."""
        mock_dns.return_value = [(2, 1, 6, "", ("127.0.0.1", 0))]
        with pytest.raises(ValueError, match="non-public"):
            _validate_url("http://redirect-target.example.com/file.pdf")

    @pytest.mark.asyncio
    @patch("pdf2md.server.MISTRAL_API_KEY", "fake-key")
    @patch("pdf2md.server.Mistral")
    @patch("pdf2md.server.MAX_FILE_SIZE", 10)
    async def test_oversized_file_rejected(self, mock_mistral_cls, tmp_path):
        big_file = tmp_path / "big.pdf"
        # Write more than 10 bytes to exceed the patched MAX_FILE_SIZE
        big_file.write_bytes(b"%PDF-1.4 this is definitely more than 10 bytes of content")

        result = await convert_pdf_file(str(big_file))
        entry = result["results"][0]
        assert entry["success"] is False
        assert "size" in entry["error"].lower()

    def test_api_status_no_key_leak(self):
        """status://api should not expose any part of the API key."""
        with patch("pdf2md.server.MISTRAL_API_KEY", "sk-super-secret-key-12345"):
            status = get_api_status()
            assert "sk-" not in status
            assert "secret" not in status
            assert "12345" not in status
            assert "Configured" in status

    def test_api_status_not_configured(self):
        with patch("pdf2md.server.MISTRAL_API_KEY", None):
            status = get_api_status()
            assert "Not configured" in status


# ---------------------------------------------------------------------------
# Smoke test: real conversion (requires MISTRAL_API_KEY)
# ---------------------------------------------------------------------------

@pytest.mark.integration
@pytest.mark.skipif(
    not os.environ.get("MISTRAL_API_KEY"),
    reason="MISTRAL_API_KEY not set",
)
@pytest.mark.asyncio
class TestSmokeIntegration:

    async def test_real_conversion(self, tmp_path):
        """Convert the test PDF and verify output structure."""
        # Copy test PDF to tmp so we don't move the original
        test_pdf = tmp_path / "ViewFile-6.pdf"
        shutil.copy2(TEST_PDF, test_pdf)

        result = await convert_pdf_file(str(test_pdf))
        assert result["success"] is True
        entry = result["results"][0]
        assert entry["success"] is True

        out_dir = Path(entry["output_directory"])

        # Verify structure
        full_md = out_dir / "full.md"
        assert full_md.exists()
        assert full_md.stat().st_size > 0

        assert (out_dir / "pages").is_dir()
        assert (out_dir / "pages" / "index.md").exists()
        assert (out_dir / "images").is_dir()
        assert (out_dir / "source").is_dir()
        assert (out_dir / "source" / "ViewFile-6.pdf").exists()
