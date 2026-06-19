from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable
from zipfile import ZipFile

import httpx

from ...paths import REAL_DATA_RAW_DIR, ensure_directories
from ..real_schema import DatasetCard, PreparedRealDataset


class BaseRealDatasetAdapter(ABC):
    dataset_card: DatasetCard

    @property
    def raw_dir(self) -> Path:
        ensure_directories()
        target = REAL_DATA_RAW_DIR / self.dataset_card.short_name
        target.mkdir(parents=True, exist_ok=True)
        return target

    @abstractmethod
    def download(self, force: bool = False) -> list[Path]:
        raise NotImplementedError

    @abstractmethod
    def prepare(self) -> PreparedRealDataset:
        raise NotImplementedError

    def manual_instructions(self) -> str:
        return self.dataset_card.note


def download_file(url: str, destination: Path, force: bool = False) -> Path:
    if destination.exists() and not force:
        return destination
    destination.parent.mkdir(parents=True, exist_ok=True)
    with httpx.Client(follow_redirects=True, timeout=120.0, headers={"User-Agent": "Mozilla/5.0"}) as client:
        response = client.get(url)
        response.raise_for_status()
        destination.write_bytes(response.content)
    return destination


def extract_zip(archive_path: Path, target_dir: Path) -> list[Path]:
    target_dir.mkdir(parents=True, exist_ok=True)
    with ZipFile(archive_path, "r") as handle:
        handle.extractall(target_dir)
        return [target_dir / member for member in handle.namelist()]


@dataclass(frozen=True)
class RegistryEntry:
    key: str
    adapter: BaseRealDatasetAdapter

    @property
    def card(self) -> DatasetCard:
        return self.adapter.dataset_card


def get_dataset_registry() -> dict[str, RegistryEntry]:
    from .chi2023_dke import CHI2023DKEAdapter
    from .convxai_iui2025 import ConvXAIIUI2025Adapter
    from .flora_ips import FloraIPSAdapter
    from .haiid import HAIIDAdapter
    from .pardos_chatgpt_tutoring import PardosChatGPTTutoringAdapter

    adapters: dict[str, BaseRealDatasetAdapter] = {
        "haiid": HAIIDAdapter(),
        "chi2023_dke": CHI2023DKEAdapter(),
        "convxai_iui2025": ConvXAIIUI2025Adapter(),
        "pardos_chatgpt_tutoring": PardosChatGPTTutoringAdapter(),
        "flora_ips": FloraIPSAdapter(),
    }
    return {key: RegistryEntry(key=key, adapter=adapter) for key, adapter in adapters.items()}


def get_integrated_adapters() -> dict[str, BaseRealDatasetAdapter]:
    return {
        key: entry.adapter
        for key, entry in get_dataset_registry().items()
        if entry.card.decision == "integrated"
    }


def get_manual_only_adapters() -> dict[str, BaseRealDatasetAdapter]:
    return {
        key: entry.adapter
        for key, entry in get_dataset_registry().items()
        if entry.card.decision == "manual-download only"
    }


def render_registry_markdown(entries: Iterable[RegistryEntry] | None = None) -> str:
    registry_entries = list(entries or get_dataset_registry().values())
    lines = [
        "| Key | Dataset | Decision | Auto-download | License | Source | Note |",
        "| --- | --- | --- | --- | --- | --- | --- |",
    ]
    for entry in registry_entries:
        card = entry.card
        lines.append(
            f"| {entry.key} | {card.name} | {card.decision} | "
            f"{'yes' if card.auto_download else 'no'} | {card.license_name} | {card.source} | {card.note} |"
        )
    return "\n".join(lines)
