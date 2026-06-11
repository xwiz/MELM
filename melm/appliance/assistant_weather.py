"""Small weather-cache adapter for the Local Assistant OS MVP."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timezone
import json
from pathlib import Path
from typing import Any
from urllib.parse import urlencode
from urllib.request import urlopen

from .local_assistant_router import LocalAssistantProfile


OPEN_METEO_GEOCODE_URL = "https://geocoding-api.open-meteo.com/v1/search"
OPEN_METEO_FORECAST_URL = "https://api.open-meteo.com/v1/forecast"


@dataclass(frozen=True)
class WeatherForecast:
    day: str
    forecast: str
    location: str
    source: str
    fetched_at: str
    temperature_min_c: float | None = None
    temperature_max_c: float | None = None
    precipitation_probability: int | None = None
    weather_code: int | None = None


@dataclass(frozen=True)
class WeatherCacheRefreshResult:
    location: str
    forecasts: tuple[WeatherForecast, ...]
    source: str
    network_used: bool
    fetched_at: str

    @property
    def weather_days(self) -> int:
        return len(self.forecasts)

    def to_dict(self) -> dict[str, Any]:
        return {
            "location": self.location,
            "source": self.source,
            "network_used": self.network_used,
            "fetched_at": self.fetched_at,
            "weather_days": self.weather_days,
            "forecasts": [forecast.__dict__ for forecast in self.forecasts],
        }


class OpenMeteoWeatherAdapter:
    """Fetch or replay a compact Open-Meteo forecast into local cache rows."""

    def refresh(
        self,
        profile: LocalAssistantProfile,
        *,
        location: str | None = None,
        offline_json: str | Path | None = None,
        live: bool = False,
        timeout_seconds: float = 8.0,
    ) -> WeatherCacheRefreshResult:
        target_location = location or profile.location
        fetched_at = _now()
        if offline_json is not None and not live:
            payload = json.loads(Path(offline_json).read_text(encoding="utf-8"))
            return _result_from_open_meteo_payload(
                payload,
                target_location=target_location,
                source="open_meteo_offline_fixture",
                network_used=False,
                fetched_at=fetched_at,
            )
        if not live:
            raise ValueError("pass live=True or an offline_json fixture")
        latitude, longitude, resolved_location = self._geocode(target_location, timeout_seconds=timeout_seconds)
        query = urlencode(
            {
                "latitude": latitude,
                "longitude": longitude,
                "daily": ",".join(
                    (
                        "weather_code",
                        "temperature_2m_max",
                        "temperature_2m_min",
                        "precipitation_probability_max",
                    )
                ),
                "forecast_days": 7,
                "timezone": "auto",
            }
        )
        payload = _get_json(f"{OPEN_METEO_FORECAST_URL}?{query}", timeout_seconds=timeout_seconds)
        payload["resolved_location"] = resolved_location
        return _result_from_open_meteo_payload(
            payload,
            target_location=resolved_location,
            source="open_meteo_api",
            network_used=True,
            fetched_at=fetched_at,
        )

    def _geocode(self, location: str, *, timeout_seconds: float) -> tuple[float, float, str]:
        query = urlencode({"name": location, "count": 1, "format": "json"})
        payload = _get_json(f"{OPEN_METEO_GEOCODE_URL}?{query}", timeout_seconds=timeout_seconds)
        results = payload.get("results") or []
        if not results:
            raise ValueError(f"no weather geocode result for {location!r}")
        first = results[0]
        resolved_parts = [
            str(first.get("name", location)),
            str(first.get("admin1", "")),
            str(first.get("country", "")),
        ]
        resolved = ", ".join(part for part in resolved_parts if part)
        return float(first["latitude"]), float(first["longitude"]), resolved or location


def weather_items_to_inventory_rows(result: WeatherCacheRefreshResult) -> tuple[dict[str, Any], ...]:
    return tuple(
        {
            "kind": "weather",
            "item_id": forecast.day,
            "payload": {
                "forecast": forecast.forecast,
                "location": forecast.location,
                "temperature_min_c": forecast.temperature_min_c,
                "temperature_max_c": forecast.temperature_max_c,
                "precipitation_probability": forecast.precipitation_probability,
                "weather_code": forecast.weather_code,
                "fetched_at": forecast.fetched_at,
            },
            "source": forecast.source,
            "license": "open_meteo_weather_api",
            "tags": ("weather", forecast.location, forecast.day),
        }
        for forecast in result.forecasts
    )


def _result_from_open_meteo_payload(
    payload: dict[str, Any],
    *,
    target_location: str,
    source: str,
    network_used: bool,
    fetched_at: str,
) -> WeatherCacheRefreshResult:
    daily = dict(payload.get("daily", {}))
    times = [str(item) for item in daily.get("time", [])]
    codes = _list_or_empty(daily.get("weather_code"))
    max_temps = _list_or_empty(daily.get("temperature_2m_max"))
    min_temps = _list_or_empty(daily.get("temperature_2m_min"))
    precip = _list_or_empty(daily.get("precipitation_probability_max"))
    forecasts = []
    for index, day_label in enumerate(("today", "tomorrow", "day_3", "day_4", "day_5", "day_6", "day_7")):
        if index >= len(times):
            break
        code = _optional_int(codes, index)
        max_temp = _optional_float(max_temps, index)
        min_temp = _optional_float(min_temps, index)
        rain = _optional_int(precip, index)
        forecasts.append(
            WeatherForecast(
                day=day_label,
                forecast=_forecast_sentence(
                    weather_code=code,
                    min_temp=min_temp,
                    max_temp=max_temp,
                    precipitation_probability=rain,
                ),
                location=str(payload.get("resolved_location") or target_location),
                source=source,
                fetched_at=fetched_at,
                temperature_min_c=min_temp,
                temperature_max_c=max_temp,
                precipitation_probability=rain,
                weather_code=code,
            )
        )
    if forecasts:
        week = _week_summary(forecasts)
        forecasts.append(
            WeatherForecast(
                day="week",
                forecast=week,
                location=forecasts[0].location,
                source=source,
                fetched_at=fetched_at,
            )
        )
    return WeatherCacheRefreshResult(
        location=forecasts[0].location if forecasts else target_location,
        forecasts=tuple(forecasts),
        source=source,
        network_used=network_used,
        fetched_at=fetched_at,
    )


def _forecast_sentence(
    *,
    weather_code: int | None,
    min_temp: float | None,
    max_temp: float | None,
    precipitation_probability: int | None,
) -> str:
    label = _weather_code_label(weather_code)
    temp = ""
    if min_temp is not None and max_temp is not None:
        temp = f", {round(min_temp)}-{round(max_temp)}C"
    rain = ""
    if precipitation_probability is not None:
        rain = f", {precipitation_probability}% precipitation chance"
    return f"{label}{temp}{rain}"


def _week_summary(forecasts: list[WeatherForecast]) -> str:
    rainy_days = sum(
        1
        for forecast in forecasts
        if (forecast.precipitation_probability or 0) >= 45
        or (forecast.weather_code or 0) in {51, 53, 55, 61, 63, 65, 80, 81, 82, 95}
    )
    high_temps = [forecast.temperature_max_c for forecast in forecasts if forecast.temperature_max_c is not None]
    warm = ""
    if high_temps:
        warm = f" highs around {round(sum(high_temps) / len(high_temps))}C"
    return f"{rainy_days} likely rainy day(s) this week{warm}".strip()


def _weather_code_label(code: int | None) -> str:
    if code is None:
        return "forecast available"
    if code == 0:
        return "clear"
    if code in {1, 2}:
        return "partly cloudy"
    if code == 3:
        return "cloudy"
    if code in {45, 48}:
        return "foggy"
    if code in {51, 53, 55, 56, 57}:
        return "drizzle"
    if code in {61, 63, 65, 66, 67}:
        return "rain"
    if code in {71, 73, 75, 77}:
        return "snow"
    if code in {80, 81, 82}:
        return "rain showers"
    if code in {95, 96, 99}:
        return "thunderstorm"
    return "forecast available"


def _get_json(url: str, *, timeout_seconds: float) -> dict[str, Any]:
    with urlopen(url, timeout=timeout_seconds) as response:
        return dict(json.loads(response.read().decode("utf-8")))


def _list_or_empty(value: Any) -> list[Any]:
    return list(value) if isinstance(value, list) else []


def _optional_int(values: list[Any], index: int) -> int | None:
    try:
        value = values[index]
    except IndexError:
        return None
    return None if value is None else int(value)


def _optional_float(values: list[Any], index: int) -> float | None:
    try:
        value = values[index]
    except IndexError:
        return None
    return None if value is None else float(value)


def _now() -> str:
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat()
