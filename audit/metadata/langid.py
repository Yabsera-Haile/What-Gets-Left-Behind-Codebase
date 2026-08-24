from __future__ import annotations

import logging
from typing import Optional

logger = logging.getLogger(__name__)

_ISO1_TO_ISO3 = {
    "en": "eng", "de": "deu", "fr": "fra", "es": "spa", "it": "ita", "pt": "por",
    "nl": "nld", "ru": "rus", "ja": "jpn", "zh": "cmn", "ar": "arb", "ko": "kor",
    "hi": "hin", "tr": "tur", "vi": "vie", "fa": "fas", "pl": "pol", "id": "ind",
    "fi": "fin", "sv": "swe", "da": "dan", "no": "nob", "cs": "ces", "el": "ell",
    "he": "heb", "ro": "ron", "hu": "hun", "th": "tha", "uk": "ukr", "bg": "bul",
    "ca": "cat", "bn": "ben", "ta": "tam", "te": "tel", "mr": "mar", "ur": "urd",
    "ms": "zsm", "tl": "tgl", "sw": "swh", "af": "afr", "hr": "hrv", "sr": "srp",
    "sk": "slk", "sl": "slv", "lt": "lit", "lv": "lav", "et": "est", "eu": "eus",
    "is": "isl", "ga": "gle", "cy": "cym", "ka": "kat", "hy": "hye", "az": "aze",
    "kk": "kaz", "uz": "uzb", "ne": "npi", "si": "sin", "km": "khm", "lo": "lao",
    "my": "mya", "am": "amh", "kn": "kan", "ml": "mal", "gu": "guj", "pa": "pan",
}


class LanguageIdentifier:

    def __init__(self, backend: str = "glotlid") -> None:
        self.requested_backend = backend
        self.backend = None
        self._model = None
        self._py3langid = None
        if backend == "glotlid":
            self._try_init_glotlid()
        if self.backend is None:
            self._init_py3langid()
        logger.info("LanguageIdentifier backend: %s (requested: %s)",
                    self.backend, self.requested_backend)

    def _try_init_glotlid(self) -> None:
        try:
            import fasttext
            from huggingface_hub import hf_hub_download
            path = hf_hub_download("cis-lmu/glotlid", "model.bin")
            self._model = fasttext.load_model(path)
            self.backend = "glotlid"
        except Exception as exc:
            logger.warning("GlotLID unavailable (%s); falling back to py3langid.", exc)

    def _init_py3langid(self) -> None:
        import py3langid
        self._py3langid = py3langid
        self.backend = "py3langid"

    def predict(self, text: str) -> tuple[str, str, float, str]:
        text = " ".join((text or "").split())
        if not text:
            return ("und", "", 0.0, self.backend)

        if self.backend == "glotlid":
            prob, label = self._model.f.predict(text, 1, 0.0, "strict")[0]
            code = label.replace("__label__", "")
            iso3, _, script = code.partition("_")
            return (iso3, script, float(prob), "glotlid")

        iso1, prob = self._py3langid.classify(text)
        iso3 = _ISO1_TO_ISO3.get(iso1, iso1)
        return (iso3, "", float(prob), "py3langid")
