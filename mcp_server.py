#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
MCP-сервер с погодой на основе FastMCP (современный подход)
"""
from __future__ import annotations

import logging
import os
import sys
from typing import Any

import httpx
import mcp.types as types
from mcp.server.fastmcp import FastMCP
from dotenv import load_dotenv
import docker
import tempfile
import subprocess

# --------------------  ЛОГИРОВАНИЕ  --------------------
# Важно: пишем только в stderr для STDIO-based серверов
logging.basicConfig(
    level=logging.INFO,
    format="[%(asctime)s] %(levelname)s - %(name)s: %(message)s",
    datefmt="%H:%M:%S",
    handlers=[logging.StreamHandler(sys.stderr)]
)
log = logging.getLogger("weather-server")

load_dotenv()

API_KEY = os.getenv("OPEN_WEATHER")
if not API_KEY:
    log.error("OPEN_WEATHER не найден в окружении!")
    sys.exit(1)
else:
    log.info(f"OPEN_WEATHER загружен: {API_KEY[:5]}...")

# --------------------  FASTMCP СЕРВЕР  --------------------
# Инициализируем FastMCP сервер
mcp = FastMCP("weather")

# --------------------  ПОГОДНЫЕ ИНСТРУМЕНТЫ  --------------------
@mcp.tool()
async def get_weather(city_name: str) -> str:
    """Получить текущую погоду для города.
    
    Args:
        city_name: Название города (например, "Moscow", "London")
    """
    log.info(f"Запрашиваю погоду для города: {city_name}")
    
    url = (
        "http://api.openweathermap.org/data/2.5/weather"
        f"?q={city_name}&appid={API_KEY}&units=metric&lang=ru"
    )
    
    try:
        async with httpx.AsyncClient() as client:
            response = await client.get(url, timeout=30.0)
            response.raise_for_status()
            data = response.json()
        
        log.info(f"Погода получена: {data.get('name', city_name)} "
                 f"{data['main']['temp']}°C, {data['weather'][0]['description']}")
        
        return (
            f"Погода в {data['name']}, {data['sys']['country']}:\n"
            f"- Состояние: {data['weather'][0]['description'].title()}\n"
            f"- Температура: {data['main']['temp']:.1f}°C (ощущается как {data['main']['feels_like']:.1f}°C)\n"
            f"- Min/Max: {data['main']['temp_min']:.1f}°C / {data['main']['temp_max']:.1f}°C\n"
            f"- Влажность: {data['main']['humidity']}%\n"
            f"- Ветер: {data['wind']['speed']} м/с\n"
            f"- Давление: {data['main']['pressure']} гПа"
        )
        
    except httpx.HTTPStatusError as e:
        error_msg = f"Ошибка OpenWeather: HTTP {e.response.status_code}"
        if e.response.status_code == 404:
            error_msg += f" - Город '{city_name}' не найден"
        else:
            error_msg += f" - {e.response.text}"
        log.error(error_msg)
        return error_msg
        
    except Exception as e:
        error_msg = f"Ошибка при получении погоды: {e}"
        log.exception(error_msg)
        return error_msg

@mcp.tool()
async def get_weather_forecast(city_name: str, days: int = 3) -> str:
    """Получить прогноз погоды на несколько дней.
    
    Args:
        city_name: Название города
        days: Количество дней для прогноза (по умолчанию 3, максимум 5)
    """
    days = min(max(days, 1), 5)  # Ограничиваем от 1 до 5 дней
    
    log.info(f"Запрашиваю прогноз для {city_name} на {days} дней")
    
    # Сначала получаем координаты города
    url = (
        "http://api.openweathermap.org/data/2.5/weather"
        f"?q={city_name}&appid={API_KEY}&units=metric"
    )
    
    try:
        async with httpx.AsyncClient() as client:
            # Получаем текущую погоду для координат
            response = await client.get(url, timeout=30.0)
            response.raise_for_status()
            current_data = response.json()
            
            lat, lon = current_data['coord']['lat'], current_data['coord']['lon']
            
            # Теперь получаем прогноз по координатам
            forecast_url = (
                "https://api.openweathermap.org/data/2.5/forecast"
                f"?lat={lat}&lon={lon}&appid={API_KEY}&units=metric&lang=ru"
            )
            
            forecast_response = await client.get(forecast_url, timeout=30.0)
            forecast_response.raise_for_status()
            forecast_data = forecast_response.json()
        
        log.info(f"Прогноз получен для {current_data['name']}")
        
        # Группируем прогноз по дням
        daily_forecasts = {}
        for item in forecast_data['list']:
            date = item['dt_txt'].split(' ')[0]
            if date not in daily_forecasts:
                daily_forecasts[date] = []
            daily_forecasts[date].append(item)
        
        # Формируем ответ
        result = f"Прогноз погоды в {current_data['name']} на {days} дней:\n\n"
        
        count = 0
        for date, forecasts in daily_forecasts.items():
            if count >= days:
                break
            
            # Берем прогноз на полдень для каждого дня
            midday_forecast = None
            for forecast in forecasts:
                if '12:00:00' in forecast['dt_txt']:
                    midday_forecast = forecast
                    break
            
            if not midday_forecast and forecasts:
                midday_forecast = forecasts[len(forecasts)//2]
            
            if midday_forecast:
                result += (
                    f"{midday_forecast['dt_txt'][:10]}:\n"
                    f"- {midday_forecast['weather'][0]['description'].title()}\n"
                    f"- Температура: {midday_forecast['main']['temp']:.1f}°C\n"
                    f"- Ощущается как: {midday_forecast['main']['feels_like']:.1f}°C\n"
                    f"- Влажность: {midday_forecast['main']['humidity']}%\n"
                    f"- Ветер: {midday_forecast['wind']['speed']} м/с\n\n"
                )
                count += 1
        
        return result.strip()
        
    except httpx.HTTPStatusError as e:
        error_msg = f"Ошибка OpenWeather: HTTP {e.response.status_code}"
        if e.response.status_code == 404:
            error_msg += f" - Город '{city_name}' не найден"
        log.error(error_msg)
        return error_msg
        
    except Exception as e:
        error_msg = f"Ошибка при получении прогноза: {e}"
        log.exception(error_msg)
        return error_msg

@mcp.tool()
async def convert_temperature(value: float, from_unit: str = "celsius", to_unit: str = "fahrenheit") -> str:
    """Конвертировать температуру между различными единицами измерения.
    
    Args:
        value: Значение температуры
        from_unit: Исходная единица измерения (celsius, fahrenheit, kelvin)
        to_unit: Целевая единица измерения (celsius, fahrenheit, kelvin)
    """
    from_unit = from_unit.lower()
    to_unit = to_unit.lower()
    
    # Конвертируем в Цельсий сначала
    if from_unit == "celsius":
        celsius = value
    elif from_unit == "fahrenheit":
        celsius = (value - 32) * 5/9
    elif from_unit == "kelvin":
        celsius = value - 273.15
    else:
        return f"Ошибка: неизвестная исходная единица '{from_unit}'. Используйте: celsius, fahrenheit, kelvin"
    
    # Конвертируем из Цельсия в целевую единицу
    if to_unit == "celsius":
        result = celsius
    elif to_unit == "fahrenheit":
        result = celsius * 9/5 + 32
    elif to_unit == "kelvin":
        result = celsius + 273.15
    else:
        return f"Ошибка: неизвестная целевая единица '{to_unit}'. Используйте: celsius, fahrenheit, kelvin"
    
    # Определяем символы единиц
    symbols = {
        "celsius": "°C",
        "fahrenheit": "°F", 
        "kelvin": "K"
    }
    
    return f"{value} {symbols[from_unit]} = {result:.2f} {symbols[to_unit]}"

import json
from datetime import datetime
from pathlib import Path

@mcp.tool()
async def save_weather_data(data: str, filename: str = "weather_data.json") -> str:
    """Сохранить погодные данные в JSON файл.

    Args:
        data: Погодные данные в формате строки JSON
        filename: Имя файла для сохранения (по умолчанию "weather_data.json")
    """
    log.info(f"Сохранение погодных данных в файл: {filename}")

    try:
        # Parse the input data as JSON
        parsed_data = json.loads(data)

        # Add timestamp to the data
        if isinstance(parsed_data, dict):
            parsed_data["timestamp"] = datetime.now().isoformat()
            weather_records = [parsed_data]
        elif isinstance(parsed_data, list):
            for record in parsed_data:
                record["timestamp"] = datetime.now().isoformat()
            weather_records = parsed_data
        else:
            return "Ошибка: данные должны быть в формате JSON объекта или массива"

        # Define file path
        file_path = Path(filename)

        # Read existing data if file exists
        existing_data = []
        if file_path.exists():
            try:
                with open(file_path, 'r', encoding='utf-8') as f:
                    existing_data = json.load(f)
                if not isinstance(existing_data, list):
                    existing_data = []
            except json.JSONDecodeError:
                existing_data = []

        # Append new records
        existing_data.extend(weather_records)

        # Write back to file
        with open(file_path, 'w', encoding='utf-8') as f:
            json.dump(existing_data, f, ensure_ascii=False, indent=2)

        log.info(f"Погодные данные успешно сохранены в {filename}")
        return f"Данные успешно сохранены в {filename}. Всего записей: {len(existing_data)}"

    except json.JSONDecodeError as e:
        error_msg = f"Ошибка парсинга JSON: {e}"
        log.error(error_msg)
        return error_msg
    except Exception as e:
        error_msg = f"Ошибка при сохранении данных: {e}"
        log.exception(error_msg)
        return error_msg


@mcp.tool()
async def stub_tool(input: str) -> str:
    """Простая заглушка для тестов.

    Args:
        input: Любой текст для обработки
    """
    log.info(f"Вызвана заглушка с input: {input}")
    return f"stub-tool выполнен. Аргументы: {input}"

# --------------------  DOCKER PYTHON EXECUTION TOOL  --------------------
@mcp.tool()
async def execute_python_code(code: str) -> str:
    """Выполнить Python код в Docker контейнере.

    Args:
        code: Python код для выполнения
    """
    log.info("Выполнение Python кода в Docker контейнере...")

    try:
        # Initialize Docker client
        client = docker.from_env()

        # Check if the Python image exists, pull it if not
        try:
            client.images.get("python:3.9-slim")
        except docker.errors.ImageNotFound:
            log.info("Образ python:3.9-slim не найден, скачиваем...")
            client.images.pull("python:3.9-slim")

        # Create a temporary Python file with the code
        with tempfile.NamedTemporaryFile(mode='w', suffix='.py', delete=False) as f:
            f.write(code)
            temp_file_path = f.name

        try:
            # Run the code in a Python Docker container and handle timeout with a different approach
            # Using python:3.9-slim as a lightweight Python environment
            import time

            # For timeout handling, we'll use a different approach since the timeout parameter might not be available
            container = None
            container = client.containers.run(
                "python:3.9-slim",
                f"python {f'/tmp/{os.path.basename(temp_file_path)}'}",
                remove=False,  # We'll handle removal ourselves
                volumes={os.path.dirname(temp_file_path): {'bind': '/tmp', 'mode': 'ro'}},
                working_dir="/tmp",
                stdout=True,
                stderr=True,
                detach=True  # Run in background to manage it ourselves
            )

            # Wait for container to finish with a timeout approach
            start_time = time.time()
            while container.status != 'exited' and time.time() - start_time < 30:
                time.sleep(0.5)
                container.reload()

            # If container is still running after timeout, stop it
            if container.status != 'exited':
                container.stop()
                container.remove()
                return "Превышено время выполнения (30 секунд)"

            # Get logs and remove the container
            logs = container.logs(stdout=True, stderr=True)
            container.remove()

            # Decode the logs
            output = logs.decode('utf-8')
            log.info(f"Код успешно выполнен. Вывод: {output[:200]}...")
            return output if output else "Код выполнен без вывода"

        except docker.errors.ContainerError as e:
            error_output = e.stderr.decode('utf-8') if e.stderr else str(e)
            log.error(f"Ошибка выполнения кода в Docker: {error_output}")
            return f"Ошибка выполнения кода: {error_output}"

        except docker.errors.ImageNotFound:
            error_msg = "Docker образ 'python:3.9-slim' не найден. Пожалуйста, выполните: docker pull python:3.9-slim"
            log.error(error_msg)
            return error_msg

        except Exception as e:
            error_msg = f"Ошибка при выполнении Python кода в Docker: {str(e)}"
            log.error(error_msg)
            # Make sure container is removed even if there's an error
            if container is not None:
                try:
                    container.remove(force=True)
                except:
                    pass  # Container might not exist or already be removed
            return error_msg

        finally:
            # Clean up the temporary file
            if os.path.exists(temp_file_path):
                os.remove(temp_file_path)

    except Exception as e:
        error_msg = f"Ошибка инициализации Docker клиента: {str(e)}"
        log.error(error_msg)
        return error_msg

# --------------------  ЗАПУСК СЕРВЕРА  --------------------
def main():
    """Запуск MCP сервера."""
    log.info("Запуск Weather MCP сервера...")
    try:
        # FastMCP автоматически использует stdio транспорт
        log.info("Сервер запущен и готов к работе.")
        mcp.run()
    except KeyboardInterrupt:
        log.info("Сервер остановлен пользователем")
    except Exception as e:
        log.exception(f"Ошибка при запуске сервера: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main()