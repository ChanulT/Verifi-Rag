import json
from pathlib import Path
from typing import Optional  # <--- THIS WAS THE MISSING IMPORT


def json_to_markdown_chunks(json_path_str: str, output_path_str: Optional[str] = None):
    json_path = Path(json_path_str)

    if not json_path.exists():
        print(f"❌ Error: File not found at {json_path}")
        return

    try:
        with open(json_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
    except Exception as e:
        print(f"❌ Error loading JSON: {e}")
        return

    # Extract core metadata
    filename = data.get("filename", "Unknown Document")
    doc_id = data.get("document_id", "N/A")
    chunks = data.get("chunks", [])

    # Header for the Markdown file
    md_content = [
        f"# Document Analysis: {filename}",
        f"**Document ID:** `{doc_id}`",
        f"**Total Chunks:** {len(chunks)}",
        "\n-----",
        "\n"
    ]

    # Process each chunk
    for chunk in chunks:
        idx = chunk.get("index", "??")
        # Look for page number in standard fields or metadata
        page = chunk.get("page_number") or chunk.get("metadata", {}).get("page_number", "Unknown")
        content = chunk.get("content", "").strip()

        # Format the chunk output
        md_content.append(f"## 🧩 Chunk {idx} (Page {page})")
        md_content.append("\n" + content + "\n")
        md_content.append("\n---\n")

    # Final result
    final_markdown = "\n".join(md_content)

    # Output logic
    if output_path_str:
        with open(output_path_str, 'w', encoding='utf-8') as f:
            f.write(final_markdown)
        print(f"✅ Success! Markdown saved to: {output_path_str}")
    else:
        # Print to console if no file specified
        print(final_markdown)


if __name__ == "__main__":
    # === CONFIGURATION ===
    # Set this to your local intermediate JSON path
    input_json = r"C:\Users\chana\OneDrive\Documents\DEV\Verifi-Rag\services\ingestion-service\cache\intermediate\c15b75a1-0fec-48de-bef3-246cbec8ef0a.json"

    # Optional: Set this to save to a file
    output_md = None

    json_to_markdown_chunks(input_json, output_md)