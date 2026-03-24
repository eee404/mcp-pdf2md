import base64
import os
from pathlib import Path
from typing import Any, Dict, List

import datauri
import httpx
from dotenv import load_dotenv
from mcp.server.fastmcp import FastMCP
from mistralai import Mistral

# Load environment variables
load_dotenv()

# API configuration
MISTRAL_API_KEY = os.environ.get("MISTRAL_API_KEY")

# Global variables
OUTPUT_DIR = "./downloads"

def set_output_dir(output_dir: str):
    """Set the output directory path for URL conversions."""
    global OUTPUT_DIR
    OUTPUT_DIR = os.path.normpath(output_dir)

def save_image(image, output_dir):
    """Saves a base64 encoded image from the OCR response."""
    try:
        parsed_uri = datauri.DataURI(image.image_base64)
        # Use a sanitized version of image.id for the filename
        image_filename = "".join(c for c in image.id if c.isalnum() or c in ('-', '_', '.')).rstrip()
        image_path = Path(output_dir) / image_filename
        with open(image_path, "wb") as file:
            file.write(parsed_uri.data)
        return str(image_path)
    except Exception as e:
        print(f"  Error saving image {image.id}: {e}")
        return None

def extract_first_heading(markdown_text):
    """Extract the first markdown heading from text, or first ~80 chars as fallback."""
    import re
    for line in markdown_text.split('\n'):
        line = line.strip()
        match = re.match(r'^(#{1,6})\s+(.+)', line)
        if match:
            return match.group(0)
    # Fallback: first non-empty line, truncated
    for line in markdown_text.split('\n'):
        line = line.strip()
        if line:
            return line[:80] + ('...' if len(line) > 80 else '')
    return '(empty page)'


def save_ocr_response_to_markdown_and_images(ocr_response, output_md_path, output_dir_for_images, split_pages=False):
    """
    Saves the OCR response markdown and images to disk.

    If split_pages is False (default): all pages are concatenated into a single .md file.
    If split_pages is True: each page is saved as page-001.md, page-002.md, etc.
    and an index.md is generated with links and heading previews.
    """
    output_dir = Path(output_dir_for_images)
    full_markdown_content = []
    saved_images = []

    try:
        if split_pages:
            total_pages = len(ocr_response.pages)
            index_entries = []

            for i, page in enumerate(ocr_response.pages):
                page_num = i + 1
                page_filename = f"page-{page_num:03d}.md"
                page_path = output_dir / page_filename

                with open(page_path, "wt", encoding='utf-8') as f:
                    f.write(page.markdown)

                full_markdown_content.append(page.markdown)

                for image in page.images:
                    saved_image_path = save_image(image, output_dir_for_images)
                    if saved_image_path:
                        saved_images.append(saved_image_path)

                heading = extract_first_heading(page.markdown)
                index_entries.append(f"- [{page_filename}]({page_filename}) — {heading}")

            doc_name = Path(output_md_path).stem
            index_path = output_dir / "index.md"
            with open(index_path, "wt", encoding='utf-8') as f:
                f.write(f"# {doc_name} ({total_pages} pages)\n\n")
                f.write('\n'.join(index_entries))
                f.write('\n')
        else:
            with open(output_md_path, "wt", encoding='utf-8') as f:
                for page in ocr_response.pages:
                    f.write(page.markdown)
                    full_markdown_content.append(page.markdown)
                    for image in page.images:
                        saved_image_path = save_image(image, output_dir_for_images)
                        if saved_image_path:
                            saved_images.append(saved_image_path)

        return "".join(full_markdown_content), saved_images
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

# Create MCP server
mcp = FastMCP("PDF to Markdown Conversion Service")

@mcp.tool()
async def convert_pdf_url(url: str, split_pages: bool = False) -> Dict[str, Any]:
    """
    Convert a PDF from a URL to Markdown. The output is saved in the directory specified by --output-dir.

    Args:
        url: A single PDF URL or multiple URLs separated by spaces, commas, or newlines.
        split_pages: If True, each page is saved as a separate file (page-001.md, etc.) with an index.md. If False (default), all pages are concatenated into a single .md file.

    Returns:
        A dictionary with the conversion results.
    """
    if not MISTRAL_API_KEY:
        return {"success": False, "error": "Missing API key, please set environment variable MISTRAL_API_KEY"}

    try:
        client = Mistral(api_key=MISTRAL_API_KEY)
    except Exception as e:
        return {"success": False, "error": f"Error initializing Mistral client: {e}"}

    urls = parse_input_string(url)
    results = []

    async with httpx.AsyncClient(timeout=120.0) as http_client:
        for u in urls:
            try:
                response = await http_client.get(u, follow_redirects=True)
                response.raise_for_status()
                base64_pdf = base64.b64encode(response.content).decode('utf-8')

                pdf_name = Path(u.split('?')[0]).stem
                # Create a specific subdirectory for this URL's content
                output_dir = Path(OUTPUT_DIR) / pdf_name
                output_dir.mkdir(parents=True, exist_ok=True)

                output_md_path = output_dir / f"{pdf_name}.md"

                ocr_response = client.ocr.process(
                    model="mistral-ocr-latest",
                    document={"type": "document_url", "document_url": f"data:application/pdf;base64,{base64_pdf}"},
                    include_image_base64=True
                )

                markdown_content, saved_images = save_ocr_response_to_markdown_and_images(
                    ocr_response, output_md_path, output_dir, split_pages=split_pages
                )

                if markdown_content is not None:
                    result_entry = {
                        "url": u,
                        "success": True,
                        "pages": len(ocr_response.pages),
                        "images": saved_images,
                        "output_directory": str(output_dir),
                        "content_length": len(markdown_content)
                    }
                    if split_pages:
                        result_entry["index_file"] = str(output_dir / "index.md")
                    else:
                        result_entry["markdown_file"] = str(output_md_path)
                    results.append(result_entry)
                else:
                    results.append({"url": u, "success": False, "error": "Could not save markdown or images."})

            except httpx.RequestError as e:
                results.append({"url": u, "success": False, "error": f"Failed to download URL: {e}"})
            except Exception as e:
                results.append({"url": u, "success": False, "error": f"Error processing URL '{u}': {e}"})

    return {"success": any(r.get("success", False) for r in results), "results": results}


@mcp.tool()
async def convert_pdf_file(file_path: str, split_pages: bool = False) -> Dict[str, Any]:
    """
    Convert a local PDF file to Markdown. Output is saved in a new folder named after the PDF in its original directory.

    Args:
        file_path: Path to a local PDF file or multiple paths separated by spaces, commas, or newlines.
        split_pages: If True, each page is saved as a separate file (page-001.md, etc.) with an index.md. If False (default), all pages are concatenated into a single .md file.

    Returns:
        A dictionary with the conversion results.
    """
    if not MISTRAL_API_KEY:
        return {"success": False, "error": "Missing API key, please set environment variable MISTRAL_API_KEY"}

    try:
        client = Mistral(api_key=MISTRAL_API_KEY)
    except Exception as e:
        return {"success": False, "error": f"Error initializing Mistral client: {e}"}

    file_paths = parse_input_string(file_path)
    results = []

    for path_str in file_paths:
        try:
            input_path = Path(path_str)
            if not input_path.exists() or not input_path.name.lower().endswith('.pdf'):
                results.append({"file_path": path_str, "success": False, "error": "File does not exist or is not a PDF."})
                continue

            # Create a new directory for output next to the original file
            output_dir = input_path.parent / input_path.stem
            output_dir.mkdir(parents=True, exist_ok=True)

            output_md_path = output_dir / f"{input_path.stem}.md"

            with open(input_path, "rb") as pdf_file:
                base64_pdf = base64.b64encode(pdf_file.read()).decode('utf-8')

            ocr_response = client.ocr.process(
                model="mistral-ocr-latest",
                document={"type": "document_url", "document_url": f"data:application/pdf;base64,{base64_pdf}"},
                include_image_base64=True
            )

            markdown_content, saved_images = save_ocr_response_to_markdown_and_images(
                ocr_response, output_md_path, output_dir, split_pages=split_pages
            )

            if markdown_content is not None:
                result_entry = {
                    "file_path": path_str,
                    "success": True,
                    "pages": len(ocr_response.pages),
                    "images": saved_images,
                    "output_directory": str(output_dir),
                    "content_length": len(markdown_content)
                }
                if split_pages:
                    result_entry["index_file"] = str(output_dir / "index.md")
                else:
                    result_entry["markdown_file"] = str(output_md_path)
                results.append(result_entry)
            else:
                results.append({"file_path": path_str, "success": False, "error": "Could not save markdown or images."})

        except Exception as e:
            results.append({"file_path": path_str, "success": False, "error": f"Error processing file '{path_str}': {e}"})

    return {"success": any(r.get("success", False) for r in results), "results": results}

@mcp.prompt()
def default_prompt() -> str:
    """Create default tool usage prompt"""
    return """
PDF to Markdown Conversion Service provides two tools:

- `convert_pdf_url`: Converts PDFs from URLs. Output is saved to the `--output-dir` directory.
- `convert_pdf_file`: Converts local PDF files. Output is saved to a new folder next to the original file.

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
    return f"API status: Configured\nMistral API key: {MISTRAL_API_KEY[:8]}..."

@mcp.resource("help://usage")
def get_usage_help() -> str:
    """Get tool usage help information"""
    return """
# PDF to Markdown Conversion Service (Mistral AI)

## Tools:

1.  **`convert_pdf_url(url: str)`**
    - Converts PDF from one or more URLs to Markdown.
    - **Output**: Saves files into a new sub-directory (named after the PDF) inside the global `--output-dir`.
    - **Args**: `url` (A single URL or a list separated by spaces, commas, or newlines).

2.  **`convert_pdf_file(file_path: str)`**
    - Converts one or more local PDF files to Markdown.
    - **Output**: Saves files into a new directory named after the PDF, located in the same directory as the original PDF.
    - **Args**: `file_path` (A single path or a list separated by spaces, commas, or newlines).

## Usage Examples:

```python
# Convert a PDF from a URL
await convert_pdf_url("https://arxiv.org/pdf/1706.03762.pdf")

# Convert a local PDF file
await convert_pdf_file("/path/to/my/document.pdf")

# Convert multiple local files
await convert_pdf_file("/path/doc1.pdf, /path/doc2.pdf")
```

## Conversion Results:
Each tool returns a dictionary containing a list of results for each file/URL processed. A successful conversion will include the path to the markdown file, a list of saved image paths, and the output directory.
"""

if __name__ == "__main__":
    if not MISTRAL_API_KEY:
        print("Warning: API key not set, please set environment variable MISTRAL_API_KEY")

    mcp.run()
