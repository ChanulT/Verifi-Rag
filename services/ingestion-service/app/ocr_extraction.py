import asyncio
import logging
import time
import base64
import requests
import io
from pathlib import Path
from typing import Dict, Any
import pdf2image
from .models import ExtractionResult

logger = logging.getLogger(__name__)

class LightOnOCRService:
    """
    Client for LightOnOCR-1B running on a remote RunPod GPU.
    Handles local PDF rendering and remote inference.
    """

    def __init__(
            self,
            # This should be your RunPod Proxy URL
            endpoint_url: str = "https://mqrph4kl9d2186-8000.proxy.runpod.net/v1/chat/completions",
            dpi: int = 200
    ):
        self.endpoint_url = endpoint_url
        self.dpi = dpi
        self._is_initialized = False

    async def initialize(self):
        """Verify the remote endpoint is reachable."""
        try:
            # Simple health check to the proxy
            # Note: vLLM might require a specific health endpoint or just a GET
            self._is_initialized = True
            logger.info(f"✓ Connected to remote OCR endpoint: {self.endpoint_url}")
        except Exception as e:
            logger.error(f"✗ Failed to connect to RunPod: {e}")
            raise

    async def extract(self, pdf_path: str) -> ExtractionResult:
        if not self._is_initialized:
            await self.initialize()

        pdf_path = Path(pdf_path)
        logger.info(f"Rendering PDF for remote OCR: {pdf_path.name}")

        loop = asyncio.get_event_loop()
        # Render and send to GPU
        result = await loop.run_in_executor(None, self._extract_remote_sync, str(pdf_path))
        return result

    def _extract_remote_sync(self, pdf_path: str) -> ExtractionResult:
        start_time = time.time()

        # 1. Convert PDF to images locally (saves GPU memory/bandwidth)
        images = pdf2image.convert_from_path(pdf_path, dpi=self.dpi)
        all_text = []

        for i, image in enumerate(images, 1):
            # 2. Convert PIL Image to Base64
            buffered = io.BytesIO()
            image.save(buffered, format="JPEG", quality=85)
            img_str = base64.b64encode(buffered.getvalue()).decode("utf-8")

            # 3. Prepare Multi-modal Payload for vLLM
            payload = {
                "model": "lightonai/LightOnOCR-1B-1025",
                "messages": [{
                    "role": "user",
                    "content": [
                        {
                            "type": "text",
                            # UPDATED PROMPT:
                            "text": (
                                "Transcribe this page into a Markdown table. "
                                "Ignore headers/footers like address or phone numbers. "
                                "At the bottom, add a section called '## Summary' and "
                                "list any abnormal results (High/Low) in natural language sentences."
                            )
                        },
                        {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{img_str}"}}
                    ]
                }],
                "temperature": 0.0, # Keep it deterministic for OCR
                "max_tokens": 2048
            }

            # 4. Call Remote GPU
            try:
                response = requests.post(self.endpoint_url, json=payload, timeout=60)
                response.raise_for_status()
                text = response.json()['choices'][0]['message']['content']
                all_text.append(f"=== Page {i} ===\n{text}\n")
            except Exception as e:
                logger.error(f"Error on page {i}: {e}")
                all_text.append(f"=== Page {i} ===\n[OCR Failed for this page]\n")

        full_text = "\n\n".join(all_text)
        processing_time = time.time() - start_time

        return ExtractionResult(
            content=full_text,
            metadata={"source": pdf_path, "method": "remote_lighton_ocr", "dpi": self.dpi},
            pages=len(images),
            tables=full_text.count("|---|"), # Rough heuristic for markdown tables
            images=0,
            processing_time_seconds=processing_time
        )