from __future__ import annotations

import argparse
import json
from pathlib import Path


REQUIRED_INPUTS = [
    "area_km2",
    "net_pay_m",
    "ntg",
    "porosity",
    "oil_saturation",
    "oil_density_t_m3",
    "formation_volume_factor",
    "recovery_factor",
]


def classify_field(recoverable_reserves_mln_t: float) -> str:
    """Классификация месторождения по извлекаемым запасам, млн т."""

    if recoverable_reserves_mln_t < 10:
        return "Мелкое"
    if recoverable_reserves_mln_t < 30:
        return "Среднее"
    if recoverable_reserves_mln_t <= 300:
        return "Крупное"
    return "Уникальное"


def calculate_reserves_volumetric(
    area_km2: float,
    net_pay_m: float,
    ntg: float,
    porosity: float,
    oil_saturation: float,
    oil_density_t_m3: float,
    formation_volume_factor: float,
    recovery_factor: float,
) -> dict[str, float | str]:
    """Расчет запасов нефти объемным методом."""

    area_m2 = area_km2 * 1_000_000
    bulk_rock_volume_m3 = area_m2 * net_pay_m
    pore_volume_m3 = bulk_rock_volume_m3 * ntg * porosity
    oil_volume_reservoir_m3 = pore_volume_m3 * oil_saturation
    oil_volume_surface_m3 = oil_volume_reservoir_m3 / formation_volume_factor
    geological_reserves_t = oil_volume_surface_m3 * oil_density_t_m3
    recoverable_reserves_t = geological_reserves_t * recovery_factor
    recoverable_reserves_mln_t = recoverable_reserves_t / 1_000_000
    field_class = classify_field(recoverable_reserves_mln_t)

    return {
        "geological_reserves_t": geological_reserves_t,
        "recoverable_reserves_t": recoverable_reserves_t,
        "recoverable_reserves_mln_t": recoverable_reserves_mln_t,
        "classification": field_class,
    }


def run(inputs: dict[str, object]) -> dict[str, object]:
    """Controlled execution entrypoint for the agent system."""

    missing_inputs = [field for field in REQUIRED_INPUTS if field not in inputs]
    if missing_inputs:
        return {
            "status": "skipped",
            "summary": "Расчет запасов не выполнен: не хватает исходных данных.",
            "missing_inputs": missing_inputs,
            "outputs": {},
            "limitations": ["Для запуска расчета нужно заполнить все обязательные входные параметры."],
        }

    try:
        prepared_inputs = {
            field: _to_float(inputs[field])
            for field in REQUIRED_INPUTS
        }
    except (TypeError, ValueError) as exc:
        return {
            "status": "failed",
            "summary": "Расчет запасов не выполнен: входные данные не удалось привести к числу.",
            "outputs": {},
            "limitations": [str(exc)],
        }

    if prepared_inputs["formation_volume_factor"] <= 0:
        return {
            "status": "failed",
            "summary": "Расчет запасов не выполнен.",
            "outputs": {},
            "limitations": ["Объемный коэффициент нефти должен быть больше нуля."],
        }

    outputs = calculate_reserves_volumetric(**prepared_inputs)
    recoverable_mln = float(outputs["recoverable_reserves_mln_t"])
    recommendation = (
        "Геологические запасы составляют "
        f"{float(outputs['geological_reserves_t']):,.2f} т, "
        "извлекаемые запасы составляют "
        f"{float(outputs['recoverable_reserves_t']):,.2f} т "
        f"({recoverable_mln:.3f} млн т). "
        f"По величине извлекаемых запасов месторождение относится к категории: {outputs['classification']}."
    )
    return {
        "status": "success",
        "summary": "Расчет запасов объемным методом выполнен.",
        "recommendation": recommendation,
        "outputs": outputs,
        "assumptions": [
            "Все коэффициенты заданы в долях единицы.",
            "Расчет выполнен по детерминированной формуле объемного метода без учета диапазонов неопределенности.",
        ],
        "limitations": [
            "Результат чувствителен к качеству исходных карт, эффективной толщины, пористости и коэффициента извлечения.",
        ],
    }


def _to_float(value: object) -> float:
    if isinstance(value, str):
        value = value.strip().replace(",", ".")
    return float(value)


def _json_main(input_path: Path, output_path: Path) -> None:
    payload = json.loads(input_path.read_text(encoding="utf-8"))
    result = run(payload)
    output_path.write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")


def read_float(prompt: str) -> float:
    while True:
        try:
            return float(input(prompt).replace(",", "."))
        except ValueError:
            print("Ошибка: введите число.")


def main() -> None:
    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument("--input")
    parser.add_argument("--output")
    args, _ = parser.parse_known_args()

    if args.input and args.output:
        _json_main(Path(args.input), Path(args.output))
        return

    print("Расчет извлекаемых запасов нефти объемным методом")
    print("-" * 55)

    payload = {
        "area_km2": read_float("Площадь залежи, км²: "),
        "net_pay_m": read_float("Эффективная нефтенасыщенная толщина, м: "),
        "ntg": read_float("Коэффициент песчанистости (NTG), доли ед. (например 0.8): "),
        "porosity": read_float("Пористость, доли ед. (например 0.22): "),
        "oil_saturation": read_float("Нефтенасыщенность, доли ед. (например 0.75): "),
        "oil_density_t_m3": read_float("Плотность нефти, т/м³ (например 0.86): "),
        "formation_volume_factor": read_float("Объемный коэффициент нефти B_o (например 1.2): "),
        "recovery_factor": read_float("Коэффициент извлечения нефти, доли ед. (например 0.35): "),
    }

    result = run(payload)
    print("\nРезультаты расчета:")
    print(result["recommendation"])


if __name__ == "__main__":
    main()
