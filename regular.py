#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Регулярный AI-агент для проверки погоды в Томске
Проверяет погоду каждые 10 секунд с помощью MCP-сервера и сохраняет в JSON
Запускает второй процесс, который раз в минуту делает Summary всех данных
"""
from __future__ import annotations

import asyncio
import json
import os
import sys
import typing as tp
from contextlib import AsyncExitStack
from datetime import datetime
from pathlib import Path

import httpx
import openai
from mcp import ClientSession, StdioServerParameters
from mcp.client.stdio import stdio_client
from dotenv import load_dotenv

load_dotenv()

# --------------------  КОНФИГУРАЦИЯ  --------------------
CITY_NAME = "Томск"
CHECK_INTERVAL = 10  # секунды
SUMMARY_INTERVAL = 60  # секунды
WEATHER_DATA_FILE = "tomsk_weather_data.json"
BASE_DIR = Path(__file__).resolve().parent
SERVER_SCRIPT = str(BASE_DIR / "mcp_server.py")

# --------------------  ЛОГИРОВАНИЕ  --------------------
import logging
logging.basicConfig(
    level=logging.INFO,
    format="[%(asctime)s] %(levelname)s - %(name)s: %(message)s",
    handlers=[logging.StreamHandler(sys.stdout)]
)
log = logging.getLogger("regular-agent")

# --------------------  MCP КЛИЕНТ  --------------------
class MCPClient:
    """MCP клиент для взаимодействия с сервером погоды."""

    def __init__(self) -> None:
        self.session: ClientSession | None = None
        self.exit_stack = AsyncExitStack()
        self.tools: list[dict] = []
        self._running = False

    async def connect_to_server(self, server_script_path: str) -> None:
        """Подключается к MCP серверу."""
        log.info(f"Подключение к серверу: {server_script_path}")

        if not Path(server_script_path).exists():
            raise FileNotFoundError(f"Сервер не найден: {server_script_path}")

        # Параметры для запуска сервера
        server_params = StdioServerParameters(
            command=sys.executable,
            args=[server_script_path],
            env={**os.environ}
        )

        # Создаем транспорт и сессию через контекстный менеджер
        stdio_transport = await self.exit_stack.enter_async_context(
            stdio_client(server_params)
        )

        self.session = await self.exit_stack.enter_async_context(
            ClientSession(
                stdio_transport[0],
                stdio_transport[1],
                client_info={"name": "regular-agent", "version": "1.0.0"}
            )
        )

        # Инициализируем сессию
        await self.session.initialize()

        # Получаем список инструментов
        tools_result = await self.session.list_tools()
        self.tools = [
            {
                "type": "function",
                "function": {
                    "name": tool.name,
                    "description": tool.description or "",
                    "parameters": tool.inputSchema
                }
            }
            for tool in (tools_result.tools if tools_result else [])
        ]

        self._running = True
        log.info(f"Подключено к серверу. Доступно инструментов: {len(self.tools)}")

        if self.tools:
            for tool in self.tools:
                log.info(f"  - {tool['function']['name']}: {tool['function']['description']}")

    async def call_tool(self, name: str, arguments: dict) -> str:
        """Вызывает инструмент MCP сервера."""
        if not self._running or not self.session:
            return "[Ошибка] Сервер не подключен"

        try:
            result = await self.session.call_tool(name, arguments)

            # Объединяем все текстовые блоки в один ответ
            text_parts = []
            for block in result.content or []:
                if hasattr(block, 'text'):
                    text_parts.append(block.text)

            return "\n".join(text_parts) if text_parts else "Инструмент выполнен без результата"

        except Exception as exc:
            error_msg = f"Ошибка вызова инструмента {name}: {exc}"
            log.error(error_msg)
            return f"[Ошибка] {error_msg}"

    async def cleanup(self) -> None:
        """Освобождает ресурсы."""
        if self._running:
            self._running = False
            await self.exit_stack.aclose()
            log.info("Ресурсы MCP клиента освобождены")

    @property
    def available_tools(self) -> list[dict]:
        """Возвращает список доступных инструментов."""
        return self.tools

# --------------------  РЕГУЛЯРНЫЙ АГЕНТ  --------------------
class RegularAgent:
    """AI-агент, который регулярно проверяет погоду для заданного города и сохраняет данные."""

    def __init__(self) -> None:
        self.mcp_client = MCPClient()
        self.city_name = CITY_NAME
        self.interval = CHECK_INTERVAL
        self.data_file = WEATHER_DATA_FILE
        self.is_running = False

    async def check_weather(self) -> str:
        """Проверяет погоду для указанного города."""
        try:
            # Проверяем наличие инструмента get_weather
            if not any(tool['function']['name'] == 'get_weather' for tool in self.mcp_client.available_tools):
                return "[Ошибка] Инструмент get_weather недоступен"

            # Вызываем инструмент получения погоды
            raw_result = await self.mcp_client.call_tool("get_weather", {"city_name": self.city_name})
            
            # Extract numeric values from the raw result to save as structured data
            import re
            
            # Attempt to parse the weather information and extract data into a structured format
            weather_data = {
                "city": self.city_name,
                "raw_result": raw_result
            }
            
            # Extract temperature values
            temp_match = re.search(r'Температура: ([+-]?\d+\.?\d*)', raw_result)
            if temp_match:
                weather_data["temperature"] = float(temp_match.group(1))
            
            feels_like_match = re.search(r'ощущается как ([+-]?\d+\.?\d*)', raw_result)
            if feels_like_match:
                weather_data["feels_like"] = float(feels_like_match.group(1))
                
            temp_min_match = re.search(r'Min/Max: ([+-]?\d+\.?\d*)', raw_result)
            if temp_min_match:
                weather_data["temp_min"] = float(temp_min_match.group(1))
                
            temp_max_match = re.search(r'/ ([+-]?\d+\.?\d*)°C', raw_result)
            if temp_max_match:
                weather_data["temp_max"] = float(temp_max_match.group(1))
                
            humidity_match = re.search(r'Влажность: (\d+)%', raw_result)
            if humidity_match:
                weather_data["humidity"] = int(humidity_match.group(1))
                
            wind_match = re.search(r'Ветер: ([+-]?\d+\.?\d*) м/с', raw_result)
            if wind_match:
                weather_data["wind_speed"] = float(wind_match.group(1))
                
            pressure_match = re.search(r'Давление: (\d+) гПа', raw_result)
            if pressure_match:
                weather_data["pressure"] = int(pressure_match.group(1))
            
            condition_match = re.search(r'- Состояние: ([^\n]+)', raw_result)
            if condition_match:
                weather_data["condition"] = condition_match.group(1).strip()
                
            country_city_match = re.search(r'Погода в ([^,]+), ([^\n]+):', raw_result)
            if country_city_match:
                weather_data["city"] = country_city_match.group(1).strip()
                weather_data["country"] = country_city_match.group(2).strip()

            # Convert to JSON string for saving
            data_str = json.dumps(weather_data, ensure_ascii=False)
            
            # Save to file using the new tool
            save_result = await self.mcp_client.call_tool("save_weather_data", {
                "data": data_str,
                "filename": self.data_file
            })
            
            return f"{raw_result}\n\n{save_result}"

        except Exception as exc:
            error_msg = f"Ошибка при проверке погоды: {exc}"
            log.error(error_msg)
            return f"[Ошибка] {error_msg}"

    async def start_monitoring(self) -> None:
        """Запускает регулярную проверку погоды."""
        log.info(f"Запуск мониторинга погоды для {self.city_name} каждые {self.interval} секунд...")
        
        try:
            # Подключаемся к серверу
            await self.mcp_client.connect_to_server(SERVER_SCRIPT)
            self.is_running = True
            
            while self.is_running:
                # Получаем время начала
                start_time = datetime.now()
                
                # Проверяем погоду
                log.info(f"Проверка погоды для {self.city_name}...")
                weather_report = await self.check_weather()
                
                # Выводим результат в консоль
                timestamp = start_time.strftime("%Y-%m-%d %H:%M:%S")
                print(f"\n[{timestamp}] Погода в {self.city_name}:")
                print(weather_report)
                print("-" * 60)
                
                # Ждем указанное количество секунд
                for _ in range(self.interval):
                    if not self.is_running:
                        break
                    await asyncio.sleep(1)
                
        except KeyboardInterrupt:
            log.info("Мониторинг прерван пользователем")
        except Exception as exc:
            error_msg = f"Критическая ошибка: {exc}"
            log.exception(error_msg)
            print(error_msg)
        finally:
            # Освобождаем ресурсы
            await self.mcp_client.cleanup()

    def stop(self):
        """Останавливает мониторинг."""
        self.is_running = False

# --------------------  СУММАРИЗЕР АГЕНТ  --------------------
class SummaryAgent:
    """Агент, который создает суммаризацию погодных данных с помощью LLM."""

    def __init__(self, data_file: str = WEATHER_DATA_FILE):
        self.data_file = data_file
        self.openai_client = self.build_openai_client()

    def build_openai_client(self) -> openai.AsyncOpenAI:
        """Создает клиент OpenAI с настройками из окружения."""
        key = os.getenv("OPENAI_API_KEY")
        base = os.getenv("OPENAI_BASE_URL")
        verify = os.getenv("OPENAI_VERIFY_SSL", "true").lower() != "false"
        http = httpx.AsyncClient(verify=verify)
        return openai.AsyncOpenAI(api_key=key, base_url=base, http_client=http)

    async def create_summary(self) -> str:
        """Создает суммаризацию всех собранных погодных данных с помощью LLM."""
        try:
            # Чтение данных из файла
            if not Path(self.data_file).exists():
                return f"Файл {self.data_file} не найден. Нет данных для суммаризации."

            with open(self.data_file, 'r', encoding='utf-8') as f:
                try:
                    data = json.load(f)
                except json.JSONDecodeError:
                    return f"Файл {self.data_file} содержит некорректный JSON."

            if not data or not isinstance(data, list) or len(data) == 0:
                return "Нет данных для суммаризации."

            # Подготовка данных для LLM
            # Ограничим количество записей, чтобы не превысить лимит токенов
            recent_data = data[-20:] if len(data) > 20 else data  # Берем последние 20 записей

            # Формируем промпт для LLM
            prompt = f"""
Пожалуйста, создай подробную аналитическую сводку погоды в Томске на основе следующих данных:

Количество наблюдений: {len(data)}
Последние наблюдения:
"""

            for i, record in enumerate(recent_data):
                if isinstance(record, dict):
                    timestamp = record.get('timestamp', 'N/A')
                    condition = record.get('condition', 'N/A')
                    temp = record.get('temperature', 'N/A')
                    feels_like = record.get('feels_like', 'N/A')
                    humidity = record.get('humidity', 'N/A')
                    wind = record.get('wind_speed', 'N/A')
                    pressure = record.get('pressure', 'N/A')

                    prompt += f"""
{i+1}. Время: {timestamp}
   Погода: {condition}
   Температура: {temp}°C (ощущается как {feels_like}°C)
   Влажность: {humidity}%
   Ветер: {wind} м/с
   Давление: {pressure} гПа
"""

            prompt += """
Проанализируй тенденции, изменения, особенности погоды, и предоставь краткое и полное резюме наблюдений.
Выдели средние значения, экстремумы, а также возможные тенденции изменения погоды.
Ответь на русском языке в формате аналитического отчета.
"""

            # Вызов LLM для создания суммаризации
            response = await self.openai_client.chat.completions.create(
                model="glm-4.5-air",  # Используем ту же модель, что и в main.py
                messages=[{"role": "user", "content": prompt}],
                temperature=0.5,
                max_tokens=1024,
            )

            summary_text = response.choices[0].message.content or "Не удалось создать резюме."

            return summary_text

        except Exception as exc:
            error_msg = f"Ошибка при создании суммаризации с помощью LLM: {exc}"
            log.error(error_msg)
            return error_msg

    async def start_summarizing(self, interval: int = SUMMARY_INTERVAL) -> None:
        """Запускает регулярную суммаризацию данных с помощью LLM."""
        log.info(f"Запуск LLM-суммаризации данных каждые {interval} секунд...")

        while True:
            try:
                summary = await self.create_summary()

                timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
                print(f"\n{'='*60}")
                print(f"[{timestamp}] СУММАРИЗАЦИЯ ПОГОДНЫХ ДАННЫХ (LLM):")
                print(summary)
                print(f"{'='*60}")

                # Ждем указанный интервал
                await asyncio.sleep(interval)

            except KeyboardInterrupt:
                log.info("Суммаризация прервана пользователем")
                break
            except Exception as exc:
                error_msg = f"Ошибка в процессе суммаризации: {exc}"
                log.error(error_msg)
                print(error_msg)
                await asyncio.sleep(interval)

# --------------------  ГЛАВНАЯ ФУНКЦИЯ  --------------------
async def main() -> None:
    """Главная функция запуска регулярного агента и суммаризатора."""
    # Создаем агентов
    regular_agent = RegularAgent()
    summary_agent = SummaryAgent(WEATHER_DATA_FILE)

    # Запускаем задачи параллельно
    try:
        await asyncio.gather(
            regular_agent.start_monitoring(),
            summary_agent.start_summarizing()
        )
    except KeyboardInterrupt:
        print("\n\n👋 Агенты остановлены!")
        regular_agent.stop()
        # Закрываем соединение с OpenAI
        await summary_agent.openai_client.close()
    except Exception as exc:
        log.error(f"Ошибка в главной функции: {exc}")
        # Закрываем соединение с OpenAI
        await summary_agent.openai_client.close()
        raise

if __name__ == "__main__":
    asyncio.run(main())