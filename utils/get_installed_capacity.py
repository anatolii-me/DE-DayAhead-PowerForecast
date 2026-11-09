"""
Получение данных об установленной мощности генерации (Германия и др. страны)
из публичного API energy-charts.info (Fraunhofer ISE).
 
Документация API: https://api.energy-charts.info/
Эндпоинт: GET /installed_power
 
ВАЖНО: сам эндпоинт /installed_power не умеет фильтровать по датам —
он всегда отдаёт всю доступную историю (по годам или по месяцам).
Поэтому фильтрация по start/end происходит здесь, на стороне клиента,
после получения ответа.
"""
 
from __future__ import annotations
 
from typing import Iterable, Optional, Union
 
import pandas as pd
import requests
 
API_URL = "https://api.energy-charts.info/installed_power"
 
DateLike = Union[str, "pd.Timestamp"]
 
# API отдаёт даты не в ISO-формате: для time_step="monthly" это "MM.YYYY"
# (например "10.2002"), для time_step="yearly" — просто "YYYY". На случай
# изменений формата перебираем несколько известных вариантов по очереди.
_TIME_FORMATS = ("%m.%Y", "%Y", "%d.%m.%Y", "%Y-%m-%d", "%Y-%m")
 
 
def _parse_time(time_list: list) -> "pd.DatetimeIndex":
    """Парсит список дат из ответа API, пробуя известные форматы по очереди."""
    last_error: Optional[Exception] = None
    for fmt in _TIME_FORMATS:
        try:
            return pd.to_datetime(time_list, format=fmt, utc=True)
        except (ValueError, TypeError) as exc:
            last_error = exc
            continue
    # Ни один известный формат не подошёл — пробуем автоопределение pandas,
    # чтобы хотя бы дать осмысленную ошибку, если формат всё же изменился.
    try:
        return pd.to_datetime(time_list, utc=True)
    except Exception:
        raise ValueError(
            f"Не удалось распознать формат дат из ответа API. "
            f"Пример значения: {time_list[0]!r}"
        ) from last_error
 
 
def get_installed_power(
    start: DateLike,
    end: DateLike,
    sources: Optional[Iterable[str]] = None,
    country: str = "de",
    time_step: str = "monthly",
    installation_decommission: bool = False,
) -> pd.DataFrame:
    """
    Возвращает датафрейм с установленной мощностью по типам генерации
    за период [start, end] (UTC).
 
    Parameters
    ----------
    start, end : str | datetime-like
        Начало и конец периода в UTC. Подойдёт любая строка, которую
        понимает pandas.to_datetime, например "2024-01-01",
        "2024-01-01T00:00Z", "2024-01" и т.п. Границы включительны.
    sources : iterable[str] | None
        Какие источники/типы генерации нужны, например
        ["Solar", "Wind onshore", "Nuclear"]. Сопоставление
        регистронезависимое и по подстроке: "wind" вернёт сразу
        "Wind onshore" и "Wind offshore", если оба есть в ответе API.
        Если None — возвращаются ВСЕ доступные типы.
    country : str
        Код страны (по умолчанию "de" — Германия). Список доступных
        кодов — в описании API (api.energy-charts.info/openapi.json).
    time_step : str
        "yearly" или "monthly". Помесячные данные доступны только
        для Германии (country="de").
    installation_decommission : bool
        Если True — вернутся не накопленные значения мощности, а
        чистые приросты/выбытия за период (в МВт, а не в ГВт).
        По умолчанию False — как на графике energy-charts.info.
 
    Returns
    -------
    pd.DataFrame
        Первая колонка "date" (datetime64, UTC), далее — по одной
        колонке на каждый выбранный источник. Единицы измерения —
        ГВт (для накопителей батарей — ГВт*ч), либо МВт, если
        installation_decommission=True.
 
    Raises
    ------
    ValueError
        Если time_step некорректен, start > end, или ни один из
        запрошенных source не найден среди доступных в ответе API
        (в сообщении будет показан список доступных названий).
    requests.HTTPError
        Если API вернул ошибку (например, неверный country).
    """
    if time_step not in ("yearly", "monthly"):
        raise ValueError('time_step должен быть "yearly" или "monthly"')
 
    start_ts = pd.to_datetime(start, utc=True)
    end_ts = pd.to_datetime(end, utc=True)
    if start_ts > end_ts:
        raise ValueError("start не может быть позже end")
 
    resp = requests.get(
        API_URL,
        params={
            "country": country,
            "time_step": time_step,
            "installation_decommission": str(installation_decommission).lower(),
        },
        timeout=30,
    )
    resp.raise_for_status()
    payload = resp.json()
 
    dates = _parse_time(payload["time"])
    all_series = {
        pt["name"]: pt["data"] for pt in (payload.get("production_types") or [])
    }
 
    if sources is None:
        chosen = list(all_series.keys())
    else:
        chosen = []
        for wanted in sources:
            matches = [name for name in all_series if wanted.lower() in name.lower()]
            if not matches:
                available = ", ".join(sorted(all_series))
                raise ValueError(
                    f'Источник "{wanted}" не найден. '
                    f"Доступные источники для country={country!r}: {available}"
                )
            for m in matches:
                if m not in chosen:
                    chosen.append(m)
 
    df = pd.DataFrame({"date": dates})
    for name in chosen:
        df[name] = all_series[name]
 
    mask = (df["date"] >= start_ts) & (df["date"] <= end_ts)
    df = df.loc[mask].reset_index(drop=True)
 
    return df