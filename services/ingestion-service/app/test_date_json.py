import json
import logging
from pathlib import Path

# Import your utility function
try:
    from utils import extract_document_date
except ImportError:
    print("❌ Could not import 'utils.py'. Make sure it's in the same folder.")
    exit(1)

# Configure logging to see utils.py debug output
logging.basicConfig(level=logging.INFO)


def test_json_date(json_path_str: str):
    json_path = Path(json_path_str)

    if not json_path.exists():
        print(f"❌ File not found: {json_path}")
        return

    print(f"\n--- Testing Intermediate File: {json_path.name} ---")

    try:
        with open(json_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
    except Exception as e:
        print(f"❌ Failed to load JSON: {e}")
        return

    # 1. Get Filename
    filename = data.get("filename", "unknown_document.pdf")
    print(f"📄 Document Filename: {filename}")

    # 2. Get Content
    # First try full content if it exists in extraction
    content = data.get("extraction", {}).get("content")

    # If not, reconstruct from chunks (since intermediate files often only save a preview)
    if not content:
        print("ℹ️  'content' not found in extraction object, reconstructing from chunks...")
        chunks = data.get("chunks", [])
        # Sort chunks just in case
        chunks.sort(key=lambda x: x.get("index", 0))
        content = "\n\n".join([c.get("content", "") for c in chunks])

    if not content:
        print("❌ Could not find text content in JSON.")
        return

    print(f"✅ Loaded {len(content)} characters of text.")
    print(f"📝 Text Preview (First 500 chars):\n{'-' * 20}\n{content[:500].strip()}\n{'-' * 20}")

    # 3. Run the Date Extractor logic
    extracted_date, extracted_year = extract_document_date(
        filename=filename,
        content=content
    )

    # 4. Output Results
    print("\n🔍 EXTRACTION RESULTS:")
    print("=" * 30)
    if extracted_date:
        print(f"📅 FOUND DATE: {extracted_date}")
    else:
        print("⚠️  NO DATE FOUND")

    if extracted_year:
        print(f"📆 FOUND YEAR: {extracted_year}")
    else:
        print("⚠️  NO YEAR FOUND")
    print("=" * 30)


if __name__ == "__main__":
    # === CONFIGURATION ===
    # Replace with the path to your intermediate JSON file
    # Example: "./cache/intermediate/565fef5d-dc54-4be0...json"
    my_json_path = r"C:\Users\chana\OneDrive\Documents\DEV\Verifi-Rag\services\ingestion-service\cache\intermediate\b444b980-8da3-436d-93ed-0152a73cc1e1.json"

    test_json_date(my_json_path)