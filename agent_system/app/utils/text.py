from __future__ import annotations

import re
from collections.abc import Iterable

TOKEN_PATTERN = re.compile(
    r"[A-Za-zА-Яа-яЁё0-9]+(?:[._+-][A-Za-zА-Яа-яЁё0-9]+)*",
    flags=re.UNICODE,
)

CANONICAL_SYNONYMS: dict[str, set[str]] = {
    "sand": {
        "sand",
        "sanding",
        "sand control",
        "sand production",
        "песок",
        "песка",
        "пескопроявление",
        "пескопроявления",
        "вынос песка",
    },
    "control": {"control", "management", "контроль", "контроля", "управление", "регулирование"},
    "method": {"method", "methods", "selection", "screening", "метод", "метода", "выбор", "подбор", "подобрать", "выбрать"},
    "calculate": {"calculate", "calculation", "compute", "расчет", "рассчитать", "расчитай", "посчитать", "оценить"},
    "monitoring": {"monitor", "monitoring", "наблюдение", "мониторинг", "surveillance"},
    "design": {"design", "strategy", "проектирование", "дизайн", "стратегия", "схема"},
    "reserves": {"reserve", "reserves", "запасы", "подсчет запасов", "подсчёт запасов"},
    "production": {"production", "production technology", "добыча", "эксплуатация", "разработка"},
    "screen": {"screen", "standalone screen", "фильтр", "экран"},
    "gravel": {"gravel", "gravel pack", "гравий", "гравийная набивка"},
    "chemical": {"chemical", "chemical consolidation", "химическое закрепление", "химический"},
    "horizontal": {"horizontal", "horizontal well", "горизонтальный", "горизонтальная скважина"},
    "reservoir": {"reservoir", "reservoirs", "коллектор", "пласт", "месторождение", "залежь"},
    "geology": {"geology", "geological", "геология", "геологический", "геологическое строение", "фации", "литология"},
    "classification": {"classification", "classify", "классификация", "классифицировать", "типизация", "виды"},
    "porosity": {"porosity", "пористость"},
    "permeability": {"permeability", "проницаемость"},
    "development": {"development", "develop", "разработка", "разрабатываемый", "система разработки", "этап разработки"},
    "eor": {
        "eor",
        "enhanced oil recovery",
        "improved oil recovery",
        "tertiary recovery",
        "методы увеличения нефтеотдачи",
        "повышение нефтеотдачи",
        "увеличение нефтеотдачи",
        "мун",
    },
    "spacing": {"spacing", "well spacing", "расстояние между скважинами", "сетка скважин", "плотность сетки"},
    "pressure": {"pressure", "давление", "пластовое давление"},
    "core": {"core", "керн", "керновый"},
    "fluid": {"fluid", "fluids", "флюид", "флюиды", "нефть", "газ", "вода"},
}

TYPO_CORRECTIONS: dict[str, str] = {
    "нефтеотачи": "нефтеотдачи",
    "нефтеотдчи": "нефтеотдачи",
    "увеличение нефтеодачи": "увеличение нефтеотдачи",
    "пескопроявлениe": "пескопроявление",
    "м ун": "мун",
    "расчитай": "рассчитай",
}

DOMAIN_HINTS: dict[str, set[str]] = {
    "PT": {
        "sand",
        "control",
        "production",
        "completion",
        "well",
        "screen",
        "gravel",
        "chemical",
        "production technology",
        "песок",
        "пескопроявление",
        "фильтр",
        "гравийная набивка",
    },
    "RE": {
        "eor",
        "reservoir",
        "geology",
        "classification",
        "reserves",
        "porosity",
        "permeability",
        "development",
        "spacing",
        "pressure",
        "коллектор",
        "геология",
        "месторождение",
        "классификация",
        "пористость",
        "проницаемость",
        "разработка",
        "сетка скважин",
        "запасы",
        "мун",
        "повышение нефтеотдачи",
    },
    "DT": {"drilling", "well construction", "бурение", "скважина", "ствол", "крепление"},
    "SF": {"surface", "facility", "pipeline", "processing", "surface facilities", "наземная инфраструктура", "трубопровод", "подготовка продукции"},
}


def typo_normalize_text(text: str) -> str:
    """Apply deterministic typo corrections before downstream analysis."""

    normalized = text
    for typo, corrected in TYPO_CORRECTIONS.items():
        normalized = re.sub(typo, corrected, normalized, flags=re.IGNORECASE)
    return normalized


def normalize_text(text: str) -> str:
    """Normalize multilingual text for deterministic matching."""

    normalized = typo_normalize_text(text.lower().replace("_", " ").replace("-", " "))
    normalized = normalized.replace("ё", "е")
    return re.sub(r"\s+", " ", normalized).strip()


def extract_tokens(text: str) -> set[str]:
    """Split normalized text into tokens with Cyrillic support."""

    return {match.group(0) for match in TOKEN_PATTERN.finditer(normalize_text(text))}


def canonical_variants(canonical: str) -> list[str]:
    """Return known variants for a canonical token."""

    return sorted(CANONICAL_SYNONYMS.get(canonical, set()))


def expand_with_canonical_tokens(text: str) -> set[str]:
    """Return raw tokens plus canonical tokens inferred from multilingual synonym matches."""

    normalized = normalize_text(text)
    tokens = extract_tokens(normalized)

    for canonical, variants in CANONICAL_SYNONYMS.items():
        if canonical in tokens:
            tokens.add(canonical)
            continue
        normalized_variants = [normalize_text(variant) for variant in variants]
        if any(variant in normalized for variant in normalized_variants):
            tokens.add(canonical)
    return tokens


def text_to_embedding_terms(text: str) -> list[str]:
    """Return raw and canonical terms used by local multilingual hashing embeddings."""

    tokens = extract_tokens(text)
    canonical_tokens = expand_with_canonical_tokens(text)
    return sorted(tokens.union(canonical_tokens))


def flatten_text(values: Iterable[str | None]) -> str:
    """Join arbitrary text fragments into a single normalized text block."""

    return " ".join(value for value in values if value)


def overlap(query_tokens: set[str], candidate_tokens: set[str]) -> list[str]:
    """Return sorted token overlap between two sets."""

    return sorted(query_tokens.intersection(candidate_tokens))


def enforce_russian_user_text(text: str) -> str:
    """Normalize user-facing text to Russian and remove retrieval artifacts."""

    cleaned = text
    replacements = {
        "Recommendation:": "Рекомендация:",
        "Justification:": "Обоснование:",
        "Limitations:": "Ограничения:",
        "Missing inputs:": "Недостающие данные:",
        "guidance:": "указание:",
        "Design basis must include": "Базис проектирования должен включать",
        "Method selection should compare": "При выборе метода следует сопоставлять",
        "Sand management starts from": "Управление пескопроявлением начинается с",
        "Primary method candidate": "Основной кандидат метода",
        "screening score": "скрининговая оценка",
        "selection criteria": "критерии выбора",
        "screen EOR methods compare options justify selection": "",
        "compare options": "",
        "justify selection": "",
        "screen methods": "сопоставление методов",
    }
    for source, target in replacements.items():
        cleaned = cleaned.replace(source, target)
    cleaned = re.sub(r"\bRecommendation\b", "Рекомендация", cleaned)
    cleaned = re.sub(r"\bJustification\b", "Обоснование", cleaned)
    cleaned = re.sub(r"\bLimitations\b", "Ограничения", cleaned)
    cleaned = re.sub(r"\bMissing inputs\b", "Недостающие данные", cleaned)
    cleaned = re.sub(r"\s{2,}", " ", cleaned)
    cleaned = re.sub(r"\n{3,}", "\n\n", cleaned)
    return cleaned.strip()


def has_excessive_latin(text: str) -> bool:
    """Return whether user-facing text contains too much unexplained Latin text."""

    allowed_tokens = {"EOR", "ASP", "PDF", "YAML", "CO2"}
    words = re.findall(r"[A-Za-z][A-Za-z0-9._-]*", text)
    suspicious = [word for word in words if word not in allowed_tokens and not word.lower().endswith((".pdf", ".yaml", ".yml"))]
    return len(suspicious) >= 6


def strip_metadata_leakage(text: str, document_titles: list[str] | None = None) -> str:
    """Remove title-like, keyword-like, and task-list fragments from user-facing text."""

    document_titles = document_titles or []
    lines = [line.strip() for line in text.splitlines() if line.strip()]
    kept: list[str] = []
    artifact_patterns = (
        "screen eor methods",
        "compare options",
        "justify selection",
        "дать определение",
        "классифицировать методы",
        "что такое методы увеличения нефтеотдачи",
    )
    verb_patterns = (
        " это ",
        " является ",
        " представляет собой ",
        " понимается ",
        " определяется ",
        " is ",
        " is defined as ",
        " refers to ",
        " means ",
    )
    keep_patterns = (
        " не хватает ",
        " выполнить нельзя ",
        " расчет ",
        " расчёт ",
        " результаты расчета ",
        " результаты расчёта ",
        " составляют ",
        " относится к категории ",
    )

    for line in lines:
        lower = f" {line.lower()} "
        if any(pattern in lower for pattern in keep_patterns):
            kept.append(line)
            continue
        if any(pattern in lower for pattern in artifact_patterns):
            continue
        if any(line.lower() == title.lower() for title in document_titles):
            continue
        comma_density = line.count(",") / max(len(line.split()), 1)
        if comma_density > 0.35 and not any(pattern in lower for pattern in verb_patterns):
            continue
        if len(line) < 40 and not any(pattern in lower for pattern in verb_patterns):
            continue
        kept.append(line)
    return "\n".join(kept).strip()
