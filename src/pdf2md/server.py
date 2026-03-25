import base64
import ipaddress
import os
import re
import shutil
import socket
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List
from urllib.parse import urlparse

import datauri
import httpx
from dotenv import load_dotenv
from filelock import AsyncFileLock
from mcp.server.fastmcp import FastMCP
from mistralai import Mistral

# Load environment variables
load_dotenv()

# API configuration
MISTRAL_API_KEY = os.environ.get("MISTRAL_API_KEY")

# Size limit: 200 MB
MAX_FILE_SIZE = 200 * 1024 * 1024
MAX_REDIRECTS = 5
MIN_HEADING_LENGTH = 6


@dataclass
class ConversionContext:
    doc_name: str
    total_pages: int

    @property
    def safe_doc_name(self) -> str:
        """Sanitize doc_name for HTML comment injection."""
        return self.doc_name.replace(">", "&gt;")


def save_image(image, images_dir: Path) -> str | None:
    """Saves a base64 encoded image from the OCR response into images_dir."""
    try:
        if not image.id or not image.id.strip():
            return None
        parsed_uri = datauri.DataURI(image.image_base64)
        # Use a sanitized version of image.id for the filename
        image_filename = "".join(c for c in image.id if c.isalnum() or c in ('-', '_', '.')).rstrip()
        if not image_filename:
            return None
        image_path = images_dir / image_filename
        with open(image_path, "wb") as file:
            file.write(parsed_uri.data)
        return str(image_path)
    except Exception as e:
        print(f"  Error saving image {image.id}: {e}")
        return None


def extract_first_heading(markdown_text: str) -> str:
    """Extract the first markdown heading from text, or first ~80 chars as fallback.

    Headings and fallback lines must have text of at least MIN_HEADING_LENGTH chars.
    """
    # Priority: markdown heading with text >= MIN_HEADING_LENGTH
    for line in markdown_text.split('\n'):
        line = line.strip()
        match = re.match(r'^(#{1,6})\s+(.+)', line)
        if match and len(match.group(2)) >= MIN_HEADING_LENGTH:
            return match.group(0)
    # Fallback: first non-empty line with text >= MIN_HEADING_LENGTH
    for line in markdown_text.split('\n'):
        line = line.strip()
        if line and len(line) >= MIN_HEADING_LENGTH:
            return line[:80] + ('...' if len(line) > 80 else '')
    # Last resort: first non-empty line truncated, or "(empty page)"
    for line in markdown_text.split('\n'):
        line = line.strip()
        if line:
            return line[:80] + ('...' if len(line) > 80 else '')
    return '(empty page)'


def rewrite_image_links(markdown: str, image_prefix: str) -> str:
    """Rewrite image links in markdown to point to the given prefix.

    Replaces ![alt](img-xxx.jpeg) with ![alt](<prefix>/img-xxx.jpeg).
    Only rewrites references that look like bare image filenames (no path separators).
    """
    def replacer(m):
        alt = m.group(1)
        src = m.group(2)
        # Only rewrite bare filenames (no directory separators)
        if '/' not in src and '\\' not in src:
            return f"![{alt}]({image_prefix}/{src})"
        return m.group(0)

    return re.sub(r'!\[([^\]]*)\]\(([^)]+)\)', replacer, markdown)


def save_images(ocr_response, images_dir: Path) -> List[str]:
    """Extract and save all images from the OCR response into images_dir."""
    saved_images = []
    for page in ocr_response.pages:
        for image in page.images:
            saved_path = save_image(image, images_dir)
            if saved_path:
                saved_images.append(saved_path)
    return saved_images


def write_full_markdown(ocr_response, output_dir: Path, ctx: ConversionContext) -> str:
    """Write full.md with all pages concatenated. Image links point to images/."""
    full_content = []
    for page in ocr_response.pages:
        full_content.append(page.markdown)
    combined = "\n\n".join(full_content)
    combined = rewrite_image_links(combined, "images")
    header = f"<!-- PDF: {ctx.safe_doc_name} | {ctx.total_pages} pages -->\n\n"
    full_path = output_dir / "full.md"
    with open(full_path, "wt", encoding='utf-8') as f:
        f.write(header + combined)
    return combined


def write_page_files(ocr_response, pages_dir: Path, ctx: ConversionContext) -> None:
    """Write individual page files (page-001.md, etc.) with image links to ../images/."""
    for i, page in enumerate(ocr_response.pages):
        page_num = i + 1
        page_filename = f"page-{page_num:03d}.md"
        page_path = pages_dir / page_filename
        content = rewrite_image_links(page.markdown, "../images")
        with open(page_path, "wt", encoding='utf-8') as f:
            if page_num == 1:
                f.write(f"<!-- PDF: {ctx.safe_doc_name} | {ctx.total_pages} pages | This is page 1 -->\n\n")
            f.write(content)


def write_index(ocr_response, pages_dir: Path, ctx: ConversionContext) -> None:
    """Write pages/index.md with metadata, links, and heading previews."""
    index_entries = []
    for i, page in enumerate(ocr_response.pages):
        page_num = i + 1
        page_filename = f"page-{page_num:03d}.md"
        heading = extract_first_heading(page.markdown)
        index_entries.append(f"- [{page_filename}]({page_filename}) — {heading}")

    disclaimer = (
        "<!-- Headings are extracted automatically and may not reflect actual page content.\n"
        "For documents with a table of contents, check the first few pages.\n"
        "To locate specific topics, use Grep on full.md rather than browsing page by page. -->\n\n"
    )

    index_path = pages_dir / "index.md"
    with open(index_path, "wt", encoding='utf-8') as f:
        f.write(f"# {ctx.doc_name} ({ctx.total_pages} pages)\n\n")
        f.write(disclaimer)
        f.write('\n'.join(index_entries))
        f.write('\n')


def save_ocr_response(ocr_response, output_dir: Path, ctx: ConversionContext):
    """Orchestrate saving OCR response to the new directory structure.

    Creates subdirectories (pages/, images/, .source/) and delegates to sub-functions.
    Returns (markdown_content, saved_images) or (None, []) on error.
    """
    pages_dir = output_dir / "pages"
    images_dir = output_dir / "images"
    source_dir = output_dir / ".source"

    pages_dir.mkdir(parents=True, exist_ok=True)
    images_dir.mkdir(parents=True, exist_ok=True)
    source_dir.mkdir(parents=True, exist_ok=True)

    try:
        saved_images = save_images(ocr_response, images_dir)
        markdown_content = write_full_markdown(ocr_response, output_dir, ctx)
        write_page_files(ocr_response, pages_dir, ctx)
        write_index(ocr_response, pages_dir, ctx)
        return markdown_content, saved_images
    except Exception as e:
        print(f"Error saving markdown files or processing images: {e}")
        return None, []


def parse_input_string(input_string: str) -> List[str]:
    """Parses a string of paths or URLs separated by spaces, commas, or newlines."""
    if (input_string.startswith('"') and input_string.endswith('"')) or \
       (input_string.startswith("'") and input_string.endswith("'")):
        input_string = input_string[1:-1]
    items = " ".join(input_string.replace(",", " ").split()).split()
    cleaned_items = []
    for item in items:
        if (item.startswith('"') and item.endswith('"')) or \
           (item.startswith("'") and item.endswith("'")):
            cleaned_items.append(item[1:-1])
        else:
            cleaned_items.append(item)
    return [item for item in cleaned_items if item]


def sanitize_filename(name: str) -> str:
    """Sanitize a filename: remove path traversal, special chars."""
    # Remove any directory components
    name = Path(name).name
    # Remove dangerous characters
    name = re.sub(r'[<>:"/\\|?*\x00-\x1f]', '_', name)
    # Remove leading dots to prevent hidden files
    name = name.lstrip('.')
    # Fallback
    if not name:
        name = "document"
    return name


def _safe_error_message(error: Exception) -> str:
    """Return a safe error message without leaking internal details."""
    return f"{type(error).__name__}: An error occurred during processing."


def _is_ip_encoding_trick(hostname: str) -> bool:
    """Detect IP encoding tricks: dword (pure numeric), hex (0x...), octal (leading zeros)."""
    # Dword notation: pure integer (e.g. 2130706433 = 127.0.0.1)
    if re.match(r'^\d+$', hostname):
        return True
    # Hex notation (e.g. 0x7f000001)
    if re.match(r'^0x[0-9a-fA-F]+$', hostname):
        return True
    # Dotted with octal octets — any octet with a leading zero (e.g. 0177.0.0.1)
    parts = hostname.split('.')
    if all(re.match(r'^\d+$', p) for p in parts) and len(parts) <= 4:
        for p in parts:
            if len(p) > 1 and p.startswith('0'):
                return True
    return False


# Blocked localhost hostnames
_LOCALHOST_NAMES = frozenset({
    'localhost', 'localhost.localdomain',
    'ip6-localhost', 'ip6-loopback',
})

# Blocked cloud metadata hostnames
_METADATA_HOSTNAMES = frozenset({
    'metadata.google.internal',
    'metadata.azure.internal',
    'metadata.goog',
})


def _validate_url(url_str: str) -> None:
    """Validate a URL for SSRF. Raises ValueError if unsafe."""
    parsed = urlparse(url_str)

    # Protocol check
    if parsed.scheme not in ('http', 'https'):
        raise ValueError(f"Unsupported URL scheme: {parsed.scheme}")

    hostname = parsed.hostname
    if not hostname:
        raise ValueError("URL has no hostname.")

    lower_host = hostname.lower()

    # Block localhost names
    if lower_host in _LOCALHOST_NAMES:
        raise ValueError("Access to localhost is blocked.")

    # Block cloud metadata hostnames
    if lower_host in _METADATA_HOSTNAMES:
        raise ValueError("Access to cloud metadata endpoints is blocked.")

    # Block IP encoding tricks (dword, hex, octal)
    if _is_ip_encoding_trick(lower_host):
        raise ValueError("IP encoding tricks are not allowed.")

    # Resolve hostname to IP and check
    try:
        addr_infos = socket.getaddrinfo(hostname, None)
    except socket.gaierror:
        raise ValueError(f"Cannot resolve hostname: {hostname}")

    for family, _, _, _, sockaddr in addr_infos:
        ip_str = sockaddr[0]
        ip = ipaddress.ip_address(ip_str)

        # Map IPv4-mapped IPv6 to IPv4 for consistent checks
        if isinstance(ip, ipaddress.IPv6Address) and ip.ipv4_mapped:
            ip = ip.ipv4_mapped

        if not ip.is_global:
            raise ValueError("URL resolves to a non-public IP address.")

        # Additional check for CGNAT range (100.64.0.0/10) - is_global may miss it on some versions
        if isinstance(ip, ipaddress.IPv4Address):
            cgnat = ipaddress.IPv4Network('100.64.0.0/10')
            if ip in cgnat:
                raise ValueError("URL resolves to a CGNAT address.")


async def _download_url_safe(http_client: httpx.AsyncClient, url: str) -> bytes:
    """Download a URL with SSRF protection and redirect validation.

    Follows redirects manually, revalidating each target URL.
    Enforces MAX_FILE_SIZE limit via streaming.
    """
    current_url = url
    for _ in range(MAX_REDIRECTS):
        _validate_url(current_url)

        async with http_client.stream("GET", current_url) as response:
            if response.status_code in (301, 302, 303, 307, 308):
                location = response.headers.get("location")
                if not location:
                    raise ValueError("Redirect without Location header.")
                # Resolve relative redirects
                current_url = str(httpx.URL(current_url).join(location))
                continue

            response.raise_for_status()

            # Stream with size check
            chunks = []
            total = 0
            async for chunk in response.aiter_bytes():
                total += len(chunk)
                if total > MAX_FILE_SIZE:
                    raise ValueError(f"File exceeds maximum size of {MAX_FILE_SIZE // (1024*1024)} MB.")
                chunks.append(chunk)

            return b"".join(chunks)

    raise ValueError("Too many redirects.")


# Create MCP server
mcp = FastMCP("PDF to Markdown Conversion Service")


@mcp.tool()
async def convert_pdf_url(url: str, output_dir: str = "./downloads") -> Dict[str, Any]:
    """
    Convert a PDF from a URL to Markdown. The output is saved in the specified output directory.

    Args:
        url: A single PDF URL or multiple URLs separated by spaces, commas, or newlines.
        output_dir: Directory where output folders will be created. Defaults to "./downloads".

    Returns:
        A dictionary with the conversion results.
    """
    if not MISTRAL_API_KEY:
        return {"success": False, "error": "Missing API key, please set environment variable MISTRAL_API_KEY"}

    try:
        client = Mistral(api_key=MISTRAL_API_KEY)
    except Exception as e:
        return {"success": False, "error": _safe_error_message(e)}

    urls = parse_input_string(url)
    results = []
    output_base = Path(output_dir)

    async with httpx.AsyncClient(timeout=120.0, follow_redirects=False) as http_client:
        for u in urls:
            try:
                _validate_url(u)
                pdf_bytes = await _download_url_safe(http_client, u)

                # Sanitize name from URL
                raw_name = Path(u.split('?')[0]).stem
                pdf_name = sanitize_filename(raw_name)

                # Create output directory structure
                doc_output_dir = output_base / f"PDF_{pdf_name}"
                overwritten = doc_output_dir.exists()
                doc_output_dir.mkdir(parents=True, exist_ok=True)

                lock_path = doc_output_dir / ".converting.lock"
                async with AsyncFileLock(lock_path, timeout=300):
                    source_dir = doc_output_dir / ".source"
                    source_dir.mkdir(parents=True, exist_ok=True)

                    # Save the downloaded PDF in .source/
                    source_pdf_path = source_dir / f"{pdf_name}.pdf"
                    with open(source_pdf_path, "wb") as f:
                        f.write(pdf_bytes)

                    base64_pdf = base64.b64encode(pdf_bytes).decode('utf-8')

                    ocr_response = client.ocr.process(
                        model="mistral-ocr-latest",
                        document={"type": "document_url", "document_url": f"data:application/pdf;base64,{base64_pdf}"},
                        include_image_base64=True
                    )

                    ctx = ConversionContext(doc_name=pdf_name, total_pages=len(ocr_response.pages))
                    markdown_content, saved_images = save_ocr_response(
                        ocr_response, doc_output_dir, ctx
                    )

                if markdown_content is not None:
                    result_entry = {
                        "url": u,
                        "success": True,
                        "pages": len(ocr_response.pages),
                        "images": saved_images,
                        "output_directory": str(doc_output_dir),
                        "content_length": len(markdown_content),
                        "full_markdown": str(doc_output_dir / "full.md"),
                        "pages_index": str(doc_output_dir / "pages" / "index.md"),
                    }
                    if overwritten:
                        result_entry["overwritten"] = True
                    results.append(result_entry)
                else:
                    results.append({"url": u, "success": False, "error": "Could not save markdown or images."})

            except ValueError as e:
                results.append({"url": u, "success": False, "error": str(e)})
            except httpx.RequestError as e:
                results.append({"url": u, "success": False, "error": f"Failed to download URL: {type(e).__name__}"})
            except Exception as e:
                results.append({"url": u, "success": False, "error": _safe_error_message(e)})

    return {"success": any(r.get("success", False) for r in results), "results": results}


@mcp.tool()
async def convert_pdf_file(file_path: str) -> Dict[str, Any]:
    """
    Convert a local PDF file to Markdown. Output is saved in a new folder named after the PDF in its original directory.

    Args:
        file_path: Path to a local PDF file or multiple paths separated by spaces, commas, or newlines.

    Returns:
        A dictionary with the conversion results.
    """
    if not MISTRAL_API_KEY:
        return {"success": False, "error": "Missing API key, please set environment variable MISTRAL_API_KEY"}

    try:
        client = Mistral(api_key=MISTRAL_API_KEY)
    except Exception as e:
        return {"success": False, "error": _safe_error_message(e)}

    file_paths = parse_input_string(file_path)
    results = []

    for path_str in file_paths:
        try:
            raw_path = Path(path_str)

            # Security: reject symlinks before resolving
            if raw_path.is_symlink():
                results.append({"file_path": path_str, "success": False, "error": "Symlinks are not allowed."})
                continue

            input_path = raw_path.resolve(strict=True)

            # Verify extension
            if input_path.suffix.lower() != '.pdf':
                results.append({"file_path": path_str, "success": False, "error": "File is not a PDF."})
                continue

            # Size check
            file_size = input_path.stat().st_size
            if file_size > MAX_FILE_SIZE:
                results.append({"file_path": path_str, "success": False, "error": f"File exceeds maximum size of {MAX_FILE_SIZE // (1024*1024)} MB."})
                continue

            # Create output directory structure
            output_dir = input_path.parent / f"PDF_{input_path.stem}"
            overwritten = output_dir.exists()
            output_dir.mkdir(parents=True, exist_ok=True)

            lock_path = output_dir / ".converting.lock"
            async with AsyncFileLock(lock_path, timeout=300):
                with open(input_path, "rb") as pdf_file:
                    base64_pdf = base64.b64encode(pdf_file.read()).decode('utf-8')

                ocr_response = client.ocr.process(
                    model="mistral-ocr-latest",
                    document={"type": "document_url", "document_url": f"data:application/pdf;base64,{base64_pdf}"},
                    include_image_base64=True
                )

                ctx = ConversionContext(doc_name=input_path.stem, total_pages=len(ocr_response.pages))
                markdown_content, saved_images = save_ocr_response(
                    ocr_response, output_dir, ctx
                )

                if markdown_content is not None:
                    # Move source PDF into .source/
                    source_dir = output_dir / ".source"
                    dest_pdf = source_dir / input_path.name
                    if input_path != dest_pdf:
                        shutil.move(str(input_path), str(dest_pdf))

            if markdown_content is not None:
                result_entry = {
                    "file_path": path_str,
                    "success": True,
                    "pages": len(ocr_response.pages),
                    "images": saved_images,
                    "output_directory": str(output_dir),
                    "content_length": len(markdown_content),
                    "full_markdown": str(output_dir / "full.md"),
                    "pages_index": str(output_dir / "pages" / "index.md"),
                }
                if overwritten:
                    result_entry["overwritten"] = True
                results.append(result_entry)
            else:
                results.append({"file_path": path_str, "success": False, "error": "Could not save markdown or images."})

        except FileNotFoundError:
            results.append({"file_path": path_str, "success": False, "error": "File does not exist."})
        except Exception as e:
            results.append({"file_path": path_str, "success": False, "error": _safe_error_message(e)})

    return {"success": any(r.get("success", False) for r in results), "results": results}


@mcp.prompt()
def default_prompt() -> str:
    """Create default tool usage prompt"""
    return """
PDF to Markdown Conversion Service provides two tools:

- `convert_pdf_url`: Converts PDFs from URLs. Accepts an `output_dir` parameter (defaults to `./downloads`).
- `convert_pdf_file`: Converts local PDF files. Output is saved to a new folder next to the original file.

Both tools always produce the same output structure:
- `PDF_document-name/full.md` — all pages concatenated
- `PDF_document-name/pages/` — individual page files with an `index.md`
- `PDF_document-name/images/` — extracted images
- `PDF_document-name/.source/` — original PDF

Please choose the appropriate tool based on the input type. For mixed inputs, call the tools separately.
"""


@mcp.prompt()
def pdf_prompt(path: str) -> str:
    """Create PDF processing prompt"""
    return f"""
Please convert the following PDF to Markdown format:

{path}

- If it's a URL, use `convert_pdf_url`.
- If it's a local file, use `convert_pdf_file`.

Converted files will be saved to the appropriate output directory based on the tool used.
"""


@mcp.resource("status://api")
def get_api_status() -> str:
    """Get API status information"""
    if not MISTRAL_API_KEY:
        return "API status: Not configured (missing MISTRAL_API_KEY)"
    return "API status: Configured"


@mcp.resource("help://usage")
def get_usage_help() -> str:
    """Get tool usage help information"""
    return """
# PDF to Markdown Conversion Service (Mistral AI)

## Tools:

1.  **`convert_pdf_url(url: str, output_dir: str = "./downloads")`**
    - Converts PDF from one or more URLs to Markdown.
    - **Output**: Saves files into a new sub-directory (named after the PDF) inside `output_dir`.
    - **Args**: `url` (A single URL or a list separated by spaces, commas, or newlines), `output_dir` (output base directory, defaults to `./downloads`).

2.  **`convert_pdf_file(file_path: str)`**
    - Converts one or more local PDF files to Markdown.
    - **Output**: Saves files into a new directory named after the PDF, located in the same directory as the original PDF.
    - **Args**: `file_path` (A single path or a list separated by spaces, commas, or newlines).

## Output Structure:

Both tools produce the same directory structure:

```
PDF_document-name/
├── full.md              ← all pages concatenated
├── pages/
│   ├── index.md         ← metadata, links, heading previews
│   ├── page-001.md
│   └── ...
├── images/
│   ├── img-001.jpeg
│   └── ...
└── .source/
    └── document-name.pdf
```

## Usage Examples:

```python
# Convert a PDF from a URL
await convert_pdf_url("https://arxiv.org/pdf/1706.03762.pdf")

# Convert a PDF from a URL with custom output directory
await convert_pdf_url("https://arxiv.org/pdf/1706.03762.pdf", output_dir="/tmp/pdfs")

# Convert a local PDF file
await convert_pdf_file("/path/to/my/document.pdf")

# Convert multiple local files
await convert_pdf_file("/path/doc1.pdf, /path/doc2.pdf")
```

## Conversion Results:
Each tool returns a dictionary containing a list of results for each file/URL processed. A successful conversion will include `full_markdown`, `pages_index`, `images`, and the `output_directory`.
"""


if __name__ == "__main__":
    if not MISTRAL_API_KEY:
        print("Warning: API key not set, please set environment variable MISTRAL_API_KEY")

    mcp.run()
