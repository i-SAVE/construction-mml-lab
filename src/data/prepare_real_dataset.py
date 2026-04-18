# Подключаем будущие аннотации типов.
from __future__ import annotations

# Импортируем argparse для CLI.
import argparse
# Импортируем json для записи summary-файла.
import json
# Импортируем dataclass для структуры сводки.
from dataclasses import asdict, dataclass
# Импортируем Path для путей к файлам.
from pathlib import Path

# Импортируем pandas для обработки таблиц.
import pandas as pd


# Класс хранит результат улучшения датасета.
@dataclass
class RealDataPrepSummary:
    # Размер до очистки.
    rows_before: int
    # Размер после очистки.
    rows_after: int
    # Число удалённых дублей.
    duplicates_removed: int
    # Количество удалённых константных колонок.
    constant_columns_removed: int
    # Список удалённых колонок из-за пропусков.
    high_missing_columns_removed: list[str]
    # Количество строк, удалённых из-за пустого target.
    empty_target_rows_removed: int


# Функция парсинга аргументов.
def parse_args() -> argparse.Namespace:
    # Создаём parser.
    parser = argparse.ArgumentParser()
    # Входной CSV с реальными данными.
    parser.add_argument("--input", type=str, required=True)
    # Выходной CSV после подготовки.
    parser.add_argument("--output", type=str, default="data/processed/real_construction_data_clean.csv")
    # Название целевой переменной.
    parser.add_argument("--target", type=str, default="SalePrice")
    # Порог доли пропусков для удаления признака.
    parser.add_argument("--missing-threshold", type=float, default=0.8)
    # Квантиль для клиппинга выбросов слева.
    parser.add_argument("--clip-lower", type=float, default=0.01)
    # Квантиль для клиппинга выбросов справа.
    parser.add_argument("--clip-upper", type=float, default=0.99)
    # Путь к JSON-отчёту.
    parser.add_argument("--summary-output", type=str, default="data/processed/real_data_summary.json")
    # Возвращаем аргументы.
    return parser.parse_args()


# Нормализация имён колонок для стабильного пайплайна.
def normalize_columns(df: pd.DataFrame) -> pd.DataFrame:
    # Копируем таблицу.
    out = df.copy()
    # Подрезаем пробелы и заменяем пробелы на подчёркивания.
    out.columns = [str(col).strip().replace(" ", "_") for col in out.columns]
    # Возвращаем обновлённую таблицу.
    return out


# Улучшение реального датасета: очистка, импутация, клиппинг.
def improve_real_dataset(
    df: pd.DataFrame,
    target: str,
    missing_threshold: float = 0.8,
    clip_lower: float = 0.01,
    clip_upper: float = 0.99,
) -> tuple[pd.DataFrame, RealDataPrepSummary]:
    # Проверяем наличие target.
    if target not in df.columns:
        # Если target отсутствует — выбрасываем ошибку.
        raise ValueError(f"Target column '{target}' not found")

    # Сохраняем размер до очистки.
    rows_before = len(df)

    # Нормализуем названия колонок.
    work = normalize_columns(df)
    # Актуализируем target на случай пробелов в имени.
    normalized_target = target.strip().replace(" ", "_")

    # Удаляем дубли строк.
    work = work.drop_duplicates().copy()
    # Считаем число удалённых дублей.
    duplicates_removed = rows_before - len(work)

    # Удаляем строки без целевой переменной.
    rows_before_target_drop = len(work)
    work = work[work[normalized_target].notna()].copy()
    # Считаем сколько удалили из-за пустого target.
    empty_target_rows_removed = rows_before_target_drop - len(work)

    # Считаем долю пропусков по признакам (кроме target).
    missing_share = work.drop(columns=[normalized_target]).isna().mean()
    # Находим признаки, где пропусков слишком много.
    high_missing_columns = sorted(missing_share[missing_share > missing_threshold].index.tolist())
    # Удаляем такие признаки.
    if high_missing_columns:
        work = work.drop(columns=high_missing_columns)

    # Ищем константные колонки (без target).
    feature_cols = [c for c in work.columns if c != normalized_target]
    constant_cols = [c for c in feature_cols if work[c].nunique(dropna=False) <= 1]
    # Удаляем константные признаки.
    if constant_cols:
        work = work.drop(columns=constant_cols)

    # Разделяем числовые и категориальные колонки.
    num_cols = work.select_dtypes(include=["number"]).columns.tolist()
    cat_cols = [c for c in work.columns if c not in num_cols]

    # Импутируем числовые признаки медианой.
    for col in num_cols:
        # Заполняем NaN медианой столбца.
        work[col] = work[col].fillna(work[col].median())

    # Импутируем категориальные признаки модой.
    for col in cat_cols:
        # Если мода существует, заполняем ею.
        if not work[col].mode(dropna=True).empty:
            work[col] = work[col].fillna(work[col].mode(dropna=True).iloc[0])
        else:
            # Если вся колонка пустая, ставим маркер неизвестного значения.
            work[col] = work[col].fillna("unknown")

    # Клиппинг выбросов для числовых признаков, кроме target.
    for col in num_cols:
        # Не трогаем целевую переменную.
        if col == normalized_target:
            continue
        # Считаем нижний и верхний квантили.
        low_q = work[col].quantile(clip_lower)
        high_q = work[col].quantile(clip_upper)
        # Ограничиваем значения указанным диапазоном.
        work[col] = work[col].clip(lower=low_q, upper=high_q)

    # Формируем summary.
    summary = RealDataPrepSummary(
        rows_before=rows_before,
        rows_after=len(work),
        duplicates_removed=duplicates_removed,
        constant_columns_removed=len(constant_cols),
        high_missing_columns_removed=high_missing_columns,
        empty_target_rows_removed=empty_target_rows_removed,
    )
    # Возвращаем очищенный датасет и summary.
    return work, summary


# Основной entrypoint для запуска через CLI.
def main() -> None:
    # Читаем аргументы.
    args = parse_args()
    # Готовим путь входного файла.
    input_path = Path(args.input)
    # Проверяем существование входного файла.
    if not input_path.exists():
        raise FileNotFoundError(f"Input CSV not found: {input_path}")

    # Читаем исходный CSV.
    raw = pd.read_csv(input_path)
    # Готовим улучшенный датасет.
    clean, summary = improve_real_dataset(
        raw,
        target=args.target,
        missing_threshold=args.missing_threshold,
        clip_lower=args.clip_lower,
        clip_upper=args.clip_upper,
    )

    # Создаём путь к выходному CSV.
    output_path = Path(args.output)
    # Создаём директорию для выходного CSV.
    output_path.parent.mkdir(parents=True, exist_ok=True)
    # Сохраняем очищенный датасет.
    clean.to_csv(output_path, index=False)

    # Создаём путь к summary JSON.
    summary_path = Path(args.summary_output)
    # Создаём директорию для summary JSON.
    summary_path.parent.mkdir(parents=True, exist_ok=True)
    # Сохраняем summary в JSON.
    summary_path.write_text(json.dumps(asdict(summary), ensure_ascii=False, indent=2), encoding="utf-8")

    # Печатаем краткий итог.
    print("Prepared dataset saved to:", output_path)
    print("Summary saved to:", summary_path)
    print("Rows before:", summary.rows_before)
    print("Rows after:", summary.rows_after)


# Точка входа.
if __name__ == "__main__":
    # Запускаем подготовку.
    main()
