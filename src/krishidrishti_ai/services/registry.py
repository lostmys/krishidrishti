from __future__ import annotations

from pathlib import Path

from krishidrishti_ai.config import load_config
from krishidrishti_ai.services.inference import DiagnosisService


class CropRegistry:
    """Lazily loads and caches one DiagnosisService per registered crop slug.

    The registry is fully data-driven: the slug→config-path mapping is
    supplied at construction time (read from the ``crops:`` block in
    default.yaml by the caller).  Adding a new Maharashtra crop requires
    one new entry in default.yaml — no code changes needed here.

    Usage::

        registry = CropRegistry({"tomato": Path("/abs/tomato.yaml"),
                                  "soyabean": Path("/abs/soyabean.yaml")})
        service = registry.get("tomato")   # lazy-loaded and cached
    """

    def __init__(self, crops: dict[str, str | Path]) -> None:
        # Normalise slugs to lowercase at construction time so comparisons
        # are always case-insensitive.
        self._crops: dict[str, Path] = {
            slug.lower(): Path(path) for slug, path in crops.items()
        }
        self._cache: dict[str, DiagnosisService] = {}

    # ------------------------------------------------------------------
    # Public interface
    # ------------------------------------------------------------------

    @property
    def available_crops(self) -> list[str]:
        """Sorted list of registered crop slugs."""
        return sorted(self._crops)

    def get(self, slug: str) -> DiagnosisService:
        """Return (and lazily create) the :class:`DiagnosisService` for *slug*.

        The service is constructed once and cached for the lifetime of this
        registry — subsequent calls for the same slug are free.

        Raises:
            KeyError: *slug* is not a registered crop.
            FileNotFoundError: The checkpoint file for this crop is missing.
        """
        slug = slug.lower()
        if slug not in self._crops:
            raise KeyError(slug)
        if slug not in self._cache:
            config = load_config(self._crops[slug])
            self._cache[slug] = DiagnosisService(config)
        return self._cache[slug]

    def checkpoint_ready(self, slug: str) -> bool:
        """Return ``True`` if the checkpoint file for *slug* exists on disk.

        Never raises — returns ``False`` on any error (missing config, bad
        YAML, missing key, etc.).
        """
        slug = slug.lower()
        if slug not in self._crops:
            return False
        try:
            config = load_config(self._crops[slug])
            return Path(config["paths"]["checkpoint_path"]).is_file()
        except Exception:
            return False
