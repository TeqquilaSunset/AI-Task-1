#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
OpenAI-совместимый чат-клиент с MCP-инструментами
(современный подход с FastMCP и улучшенной архитектурой)
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
from dotenv import load_dotenv
from mcp import ClientSession, StdioServerParameters
from mcp.client.stdio import stdio_client

load_dotenv()

# --------------------  КОНФИГУРАЦИЯ  --------------------
SYSTEM_PROMPT = "Ты помощник, который помогает с любыми вопросами. Ты можешь использовать доступные инструменты для получения информации. Используй только те инструменты которые у тебя есть. Если существующих инструментов не хватает не напиши пользователю об этом. Не придумывай новые иснтурменты. Вызывай инструменты в tool_calls"
SAVE_DIR = Path("saves")
SAVE_DIR.mkdir(exist_ok=True)

BASE_DIR = Path(__file__).resolve().parent
SERVER_SCRIPT = str(BASE_DIR / "mcp_server.py")
DOCKER_MCP_HOST = os.getenv("DOCKER_MCP_HOST", "localhost")
DOCKER_MCP_PORT = int(os.getenv("DOCKER_MCP_PORT", "9011"))  # Updated to 9011 for MCP gateway as per new docker-compose.yml
DOCKER_MCP_URL = f"http://{DOCKER_MCP_HOST}:{DOCKER_MCP_PORT}/mcp"

# --------------------  ЛОГИРОВАНИЕ  --------------------
import logging
logging.basicConfig(
    level=logging.INFO,
    format="[%(asctime)s] %(levelname)s - %(name)s: %(message)s",
    handlers=[logging.StreamHandler(sys.stderr)]
)
log = logging.getLogger("mcp-client")

# --------------------  УТИИЛИТЫ  --------------------
def build_openai_client() -> openai.AsyncOpenAI:
    """Создает клиент OpenAI с настройками из окружения."""
    key = os.getenv("OPENAI_API_KEY")
    base = os.getenv("OPENAI_BASE_URL")
    verify = os.getenv("OPENAI_VERIFY_SSL", "true").lower() != "false"
    http = httpx.AsyncClient(verify=verify)
    return openai.AsyncOpenAI(api_key=key, base_url=base, http_client=http)

def save_conversation(history: tp.List[dict], name: str | None = None) -> str:
    """Сохраняет историю разговора в JSON файл."""
    name = f"conversation_{datetime.now():%Y%m%d_%H%M%S}.json" if name is None else name
    if not name.endswith(".json"):
        name += ".json"
    
    path = SAVE_DIR / name
    try:
        path.write_text(json.dumps(history, ensure_ascii=False, indent=2))
        log.info(f"Разговор сохранен в {path}")
        return str(path)
    except Exception as exc:
        error_msg = f"Ошибка при сохранении: {exc}"
        log.error(error_msg)
        return error_msg

def load_conversation(name: str) -> tp.Tuple[tp.List[dict] | None, str]:
    """Загружает историю разговора из файла."""
    try:
        if name.isdigit():
            # Загрузка по номеру
            files = sorted(SAVE_DIR.glob("conversation_*.json"), reverse=True)
            idx = int(name) - 1
            if 0 <= idx < len(files):
                path = files[idx]
                return json.loads(path.read_text()), str(path)
            return None, "Неверный номер сохранения."
        
        # Загрузка по имени
        path = SAVE_DIR / (name if name.endswith(".json") else f"{name}.json")
        if path.exists():
            return json.loads(path.read_text()), str(path)
        return None, f"Файл {path} не найден."
        
    except Exception as exc:
        error_msg = f"Ошибка при загрузке: {exc}"
        log.error(error_msg)
        return None, error_msg

def list_saved_conversations() -> str:
    """Возвращает список сохраненных разговоров."""
    files = sorted(SAVE_DIR.glob("conversation_*.json"), reverse=True)
    if not files:
        return "Нет сохранённых разговоров."
    
    lines = ["Сохранённые разговоры:", "=" * 40]
    for idx, fp in enumerate(files, 1):
        # Извлекаем дату из имени файла
        ts_match = fp.stem.replace("conversation_", "")
        try:
            dt = datetime.strptime(ts_match, "%Y%m%d_%H%M%S")
            nice_date = dt.strftime("%Y-%m-%d %H:%M:%S")
        except:
            nice_date = fp.stem
        
        lines.append(f"{idx}. {nice_date} – {fp.name}")
    
    return "\n".join(lines)

async def create_summary(cli: openai.AsyncOpenAI, model: str, history: tp.List[dict]) -> str:
    """Создает краткое резюме разговора."""
    msgs = [m for m in history if m["role"] in ("user", "assistant")]
    if not msgs:
        return "Нет истории для резюме."
    
    text = "Пожалуйста, создай краткое резюме следующего диалога. Выдели основные темы и детали:\n\n"
    for msg in msgs:
        role = "Пользователь" if msg["role"] == "user" else "AI"
        text += f"{role}: {msg['content']}\n\n"
    
    try:
        resp = await cli.chat.completions.create(
            model=model,
            messages=[{"role": "user", "content": text}],
            temperature=0.3,
            max_tokens=512,
        )
        return resp.choices[0].message.content or "Не удалось создать резюме."
    except Exception as exc:
        error_msg = f"Ошибка при создании резюме: {exc}"
        log.error(error_msg)
        return error_msg

# --------------------  MCP КЛИЕНТ  --------------------
class MCPClient:
    """Современный MCP клиент с улучшенным управлением ресурсами."""
    
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
                client_info={"name": "mcp-client", "version": "1.0.0"}
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

# --------------------  DOCKER MCP КЛИЕНТ  --------------------
class DockerMCPClient:
    """Клиент для подключения к MCP серверу, работающему в Docker на порту 8002."""

    def __init__(self) -> None:
        self.tools: list[dict] = []
        self._connected = False
        # Use the correct endpoint for streamableHttp protocol
        self.mcp_url = DOCKER_MCP_URL  # Use the configured URL directly
        self.base_url = f"http://{DOCKER_MCP_HOST}:{DOCKER_MCP_PORT}"
        self.health_url = f"{self.base_url}/healthz"
        self.session = None
        self._tools_discovered = False
        self._sse_task = None

    async def connect_to_server(self) -> None:
        """Подключается к Docker MCP серверу по HTTP с использованием streamableHttp."""
        log.info(f"Подключение к Docker MCP серверу: {self.mcp_url}")

        try:
            async with httpx.AsyncClient(timeout=10.0) as client:
                # Check if the server is healthy first
                try:
                    health_response = await client.get(self.health_url)
                    log.info(f"Health check: {self.health_url} -> {health_response.status_code}")
                except Exception:
                    log.warning(f"Не удалось проверить health endpoint: {self.health_url}")

                # For streamableHttp protocol, we should directly try to get tools via POST
                # with proper headers following the MCP specification
                headers = {
                    "Content-Type": "application/json",
                    "Accept": "application/json, text/event-stream",  # Required by streamableHttp
                    # MCP protocol headers
                    "MCP-Version": "2024-11-05",
                    "Protocol-Version": "2024-11-05",
                    # Some implementations might expect specific headers
                }

                # According to error message, server expects JSON-RPC 2.0 format
                payload = {
                    "jsonrpc": "2.0",
                    "method": "tools/list",
                    "params": {},
                    "id": str(hash("tools_list_request"))[:8]  # Use a unique ID for JSON-RPC
                }

                # Post the request to get tools list
                tools_response = await client.post(
                    self.mcp_url,
                    json=payload,
                    headers=headers
                )

                log.info(f"Tools list request to {self.mcp_url} -> {tools_response.status_code}")

                if tools_response.status_code == 200:
                    # For streamableHttp, we need to handle the SSE response
                    await self._process_sse_response(tools_response.text)
                    self._connected = True  # Server is accessible
                elif tools_response.status_code in [400, 404, 405]:
                    # 400 Bad Request might mean the server is available but request format is wrong
                    # 404: endpoint not found
                    # 405: method not allowed
                    # In all these cases, the server is responding, so we consider it accessible
                    self._connected = True
                    log.info(f"Сервер streamableHttp доступен (статус: {tools_response.status_code}), но не удалось получить инструменты")
                else:
                    log.warning(f"MCP endpoint недоступен: {self.mcp_url}, статус: {tools_response.status_code}")

        except httpx.ConnectError:
            log.warning(f"Не удается подключиться к Docker MCP серверу: {self.base_url}")
            self._connected = False
        except httpx.TimeoutException:
            log.warning(f"Таймаут подключения к Docker MCP серверу: {self.base_url}")
            self._connected = False
        except Exception as exc:
            log.warning(f"Ошибка подключения к Docker MCP серверу: {exc}")
            self._connected = False

    async def _process_sse_response(self, response_text: str) -> None:
        """Process the SSE response to extract tools."""
        try:
            # Parse the response as SSE format
            # Handle both direct JSON and SSE-formatted responses
            if response_text.strip().startswith('event: message') or 'data: ' in response_text:
                # This is an SSE response, parse it
                lines = response_text.strip().split('\n')
                for i, line in enumerate(lines):
                    if line.startswith('data: '):
                        try:
                            import json
                            data_content = line[6:]  # Remove 'data: ' prefix
                            sse_data = json.loads(data_content)

                            # Check if this is a JSON-RPC 2.0 response with tools
                            if sse_data.get('jsonrpc') == '2.0' and 'result' in sse_data:
                                result = sse_data['result']
                                if isinstance(result, dict) and 'tools' in result:
                                    self._process_tools(result['tools'])
                                    self._tools_discovered = True
                                    log.info(f"Инструменты получены через SSE: {len(self.tools)}")
                                    for tool in self.tools:
                                        log.info(f"  - {tool['function']['name']}: {tool['function']['description']}")
                                    break  # Found tools, exit loop
                        except json.JSONDecodeError:
                            continue
            else:
                # Direct JSON response
                import json
                data = json.loads(response_text)
                if data.get('jsonrpc') == '2.0' and 'result' in data:
                    result = data['result']
                    if isinstance(result, dict) and 'tools' in result:
                        self._process_tools(result['tools'])
                        self._tools_discovered = True
                        log.info(f"Инструменты получены через JSON: {len(self.tools)}")
                        for tool in self.tools:
                            log.info(f"  - {tool['function']['name']}: {tool['function']['description']}")
        except Exception as e:
            log.warning(f"Ошибка парсинга SSE ответа: {e}")

    def _process_tools(self, tools_list):
        """Process the tools list from server response."""
        self.tools = []

        for tool in tools_list:
            # Convert tool to OpenAI format
            openai_tool = {
                "type": "function",
                "function": {
                    "name": tool.get('name', ''),
                    "description": tool.get('description', ''),
                    "parameters": tool.get('inputSchema', {})
                }
            }
            self.tools.append(openai_tool)



    async def call_tool(self, name: str, arguments: dict) -> str:
        """Вызывает инструмент Docker MCP сервера."""
        if not self._connected:
            return "[Предупреждение] Docker MCP сервер недоступен"

        try:
            # Send tool call request to the streamableHttp endpoint
            async with httpx.AsyncClient(timeout=30.0) as client:
                headers = {
                    "Content-Type": "application/json",
                    "Accept": "application/json, text/event-stream",  # Required by streamableHttp
                    "MCP-Version": "2024-11-05",
                    "Protocol-Version": "2024-11-05",
                }

                payload = {
                    "jsonrpc": "2.0",
                    "method": "tools/call",
                    "params": {
                        "name": name,
                        "arguments": arguments
                    },
                    "id": str(hash(f"tool_call_{name}"))[:8]  # Use a unique ID for JSON-RPC
                }

                response = await client.post(self.mcp_url, json=payload, headers=headers)

                if response.status_code == 200:
                    try:
                        content_type = response.headers.get("content-type", "")
                        response_text = response.text
                        if response_text.strip():
                            result = response.json()
                            # Handle different possible response formats
                            if isinstance(result, dict):
                                if "result" in result:
                                    return str(result["result"])
                                elif "content" in result:
                                    return str(result["content"])
                                elif "message" in result:
                                    return str(result["message"])
                                else:
                                    return str(result)
                            else:
                                return str(result)
                        else:
                            # Empty response is valid
                            return "Инструмент успешно выполнен (без результата)"
                    except Exception:
                        # Server responded with 200 but response is not valid JSON
                        # This can happen with streamableHttp protocols
                        return f"Инструмент успешно выполнен, ответ: {response.text[:200]}..."
                else:
                    return f"[Ошибка] Код статуса: {response.status_code}, Ответ: {response.text}"

        except httpx.ConnectError:
            return "[Ошибка] Не удается подключиться к Docker MCP серверу"
        except httpx.TimeoutException:
            return "[Ошибка] Таймаут выполнения инструмента"
        except Exception as exc:
            error_msg = f"Ошибка вызова инструмента {name}: {exc}"
            log.error(error_msg)
            return f"[Ошибка] {error_msg}"

    @property
    def available_tools(self) -> list[dict]:
        """Возвращает список доступных инструментов."""
        # In a real implementation, tools would be discovered via the SSE stream
        # using the MCP protocol's listTools endpoint over the SSE connection
        return self.tools

    @property
    def connected(self) -> bool:
        """Возвращает статус соединения."""
        return self._connected

# --------------------  ЧАТ КЛИЕНТ  --------------------
class ChatClient:
    """Основной чат-клиент с интеграцией MCP."""
    
    def __init__(self, model_name: str = "glm-4.5-air") -> None:
        self.model_name = model_name
        self.openai_client = build_openai_client()
        self.mcp_client = MCPClient()
        self.docker_mcp_client = DockerMCPClient()
        self.conversation: list[dict] = [{"role": "system", "content": SYSTEM_PROMPT}]
        self.temperature = 0.7

    async def process_query(self, query: str) -> str:
        """Обрабатывает запрос пользователя с использованием доступных инструментов."""
        log.info(f"Обработка запроса: {query[:50]}...")

        try:
            # Объединяем инструменты из обоих клиентов
            all_tools = []
            all_tools.extend(self.mcp_client.available_tools or [])
            all_tools.extend(self.docker_mcp_client.available_tools or [])

            # Начинаем цикл обработки, чтобы обработать потенциально несколько раундов вызовов инструментов
            messages = self.conversation + [{"role": "user", "content": query}]

            while True:  # Цикл для обработки нескольких раундов инструментов
                # Делаем запрос к OpenAI с доступными инструментами
                response = await self.openai_client.chat.completions.create(
                    model=self.model_name,
                    messages=messages,
                    tools=all_tools or None,
                    tool_choice="auto" if all_tools else None,
                    temperature=self.temperature,
                    max_tokens=2048,
                )

                assistant_message = response.choices[0].message
                content = assistant_message.content or ""

                # Проверяем, есть ли tool calls
                if assistant_message.tool_calls:
                    log.info(f"Обнаружены вызовы инструментов: {[tc.function.name for tc in assistant_message.tool_calls]}")

                    # Добавляем сообщение ассистента
                    messages.append({
                        "role": "assistant",
                        "content": content,
                        "tool_calls": [
                            {
                                "id": tc.id,
                                "type": "function",
                                "function": {
                                    "name": tc.function.name,
                                    "arguments": tc.function.arguments
                                }
                            }
                            for tc in assistant_message.tool_calls
                        ]
                    })

                    # Выполняем все tool calls
                    for tool_call in assistant_message.tool_calls:
                        tool_name = tool_call.function.name

                        try:
                            tool_args = json.loads(tool_call.function.arguments)

                            # Проверяем и корректируем формат аргументов, если они содержат XML-подобный формат
                            if isinstance(tool_args, str):
                                # Проверяем, не является ли строка XML-подобной
                                if '<arg_key>' in tool_args and '<arg_value>' in tool_args:
                                    # Парсим XML-подобный формат в словарь
                                    corrected_args = self._parse_xml_like_args(tool_args)
                                    tool_args = corrected_args
                            elif isinstance(tool_args, dict):
                                # Проверяем, не содержит ли словарь строки с XML-форматом
                                for key, value in tool_args.items():
                                    if isinstance(value, str) and '<arg_key>' in value and '<arg_value>' in value:
                                        tool_args[key] = self._parse_xml_like_args(value)

                            log.info(f"Вызов инструмента: {tool_name} с аргументами: {tool_args}")

                            # Определяем, какой клиент использовать для вызова инструмента
                            tool_result = await self._call_appropriate_tool(tool_name, tool_args)

                            log.info(f"Результат инструмента {tool_name}: {tool_result[:100]}...")

                            # Добавляем результат в контекст
                            messages.append({
                                "role": "tool",
                                "tool_call_id": tool_call.id,
                                "content": tool_result
                            })
                        except json.JSONDecodeError as e:
                            log.error(f"Ошибка декодирования аргументов инструмента {tool_name}: {e}")
                            # В случае ошибки декодирования, пробуем распарсить XML-подобный формат
                            try:
                                corrected_args = self._parse_xml_like_args(tool_call.function.arguments)
                                tool_result = await self._call_appropriate_tool(tool_name, corrected_args)

                                log.info(f"Результат инструмента {tool_name} (из XML-формата): {tool_result[:100]}...")

                                messages.append({
                                    "role": "tool",
                                    "tool_call_id": tool_call.id,
                                    "content": tool_result
                                })
                            except Exception as parse_error:
                                log.error(f"Ошибка парсинга аргументов инструмента {tool_name}: {parse_error}")
                                error_content = f"[Ошибка] Не удалось разобрать аргументы инструмента {tool_name}: {parse_error}"
                                messages.append({
                                    "role": "tool",
                                    "tool_call_id": tool_call.id,
                                    "content": error_content
                                })

                    # Продолжаем цикл, чтобы дать модели возможность сделать еще вызовы инструментов
                    # или вернуть окончательный ответ (после нескольких раундов инструментов)
                    continue
                else:
                    # Нет tool calls, значит это окончательный ответ
                    # Добавляем его в основную историю разговора
                    self.conversation.extend(messages[1:])  # Добавляем все сообщения кроме системного
                    return content

        except Exception as exc:
            error_msg = f"Ошибка при обработке запроса: {exc}"
            log.error(error_msg)
            return f"Извините, произошла ошибка: {exc}"

    def _parse_xml_like_args(self, xml_string: str) -> dict:
        """Парсит XML-подобную строку аргументов в словарь."""
        import re

        args_dict = {}

        # Регулярное выражение для поиска пар <arg_key> и <arg_value>
        pattern = r'<arg_key>(.*?)</arg_key>\s*<arg_value>(.*?)</arg_value>'
        matches = re.findall(pattern, xml_string, re.DOTALL)

        for key, value in matches:
            # Очищаем значения от лишних пробелов и тегов
            key = key.strip()
            value = value.strip()
            # Пытаемся определить тип значения
            if value.lower() in ('true', 'false'):
                value = value.lower() == 'true'
            elif value.isdigit():
                value = int(value)
            elif value.replace('.', '').isdigit():
                value = float(value)
            elif value.startswith('"') and value.endswith('"'):
                value = value[1:-1]  # Убираем кавычки

            args_dict[key] = value

        # Также проверяем общий формат, где вся строка может быть одним значением
        if not args_dict and xml_string.strip().startswith('<arg_key>'):
            # Попробуем найти все вхождения
            all_matches = re.findall(r'<arg_key>([^<]*)</arg_key>\s*<arg_value>([^<]*)</arg_value>', xml_string)
            for key, value in all_matches:
                key = key.strip()
                value = value.strip()
                if value.lower() in ('true', 'false'):
                    value = value.lower() == 'true'
                elif value.isdigit():
                    value = int(value)
                elif value.replace('.', '', 1).isdigit():
                    value = float(value)
                args_dict[key] = value

        return args_dict


    async def _call_appropriate_tool(self, name: str, arguments: dict) -> str:
        """Определяет, какой клиент использовать для вызова инструмента."""
        # Проверяем, принадлежит ли инструмент к обычному MCP клиенту
        stdio_tool_names = [tool['function']['name'] for tool in self.mcp_client.available_tools]
        docker_tool_names = [tool['function']['name'] for tool in self.docker_mcp_client.available_tools]

        log.info(f"Попытка вызвать инструмент '{name}' с аргументами: {arguments}")

        if name in stdio_tool_names:
            log.info(f"Вызов инструмента '{name}' через stdio клиент")
            return await self.mcp_client.call_tool(name, arguments)
        elif name in docker_tool_names:
            log.info(f"Вызов инструмента '{name}' через Docker клиент")
            return await self.docker_mcp_client.call_tool(name, arguments)
        else:
            # Если инструмент не найден ни в одном из списков, пробуем оба
            # Сначала пробуем обычный MCP клиент
            if self.mcp_client._running:  # Check if client is actually running
                result = await self.mcp_client.call_tool(name, arguments)
                if result and not (result.startswith("[Ошибка]") or result.startswith("[Предупреждение]")):
                    log.info(f"Инструмент '{name}' успешно вызван через stdio клиент")
                    return result

            # Если обычный клиент не дал результата, пробуем Docker-клиент
            if self.docker_mcp_client.connected:
                result = await self.docker_mcp_client.call_tool(name, arguments)
                if result and not (result.startswith("[Ошибка]") or result.startswith("[Предупреждение]")):
                    log.info(f"Инструмент '{name}' успешно вызван через Docker клиент")
                    return result

            return f"[Ошибка] Инструмент '{name}' не найден ни в одном из активных серверов MCP"

    async def start(self, server_script: str) -> None:
        """Запускает клиента и подключается к MCP серверам."""
        log.info("Запуск чат-клиента...")

        # Подключаемся к стандартному MCP серверу
        await self.mcp_client.connect_to_server(server_script)

        # Подключаемся к Docker MCP серверу (не выбрасываем исключение при недоступности)
        await self.docker_mcp_client.connect_to_server()

        log.info("Чат-клиент готов к работе!")

    async def cleanup(self) -> None:
        """Освобождает ресурсы."""
        await self.mcp_client.cleanup()
        # Docker MCP не требует специальной очистки соединений
        await self.openai_client.close()
        log.info("Клиент остановлен")

    def add_message(self, role: str, content: str) -> None:
        """Добавляет сообщение в историю."""
        self.conversation.append({"role": role, "content": content})

# --------------------  ИНТЕРАКТИВНЫЙ ИНТЕРФЕЙС  --------------------
async def interactive_chat(client: ChatClient) -> None:
    """Интерактивный чат-цикл."""
    print("=" * 60)
    print("🤖 Чат-клиент с MCP инструментами")
    print("Команды: quit/exit, save <имя>, load <имя>, temp <0-2>, clear, print")
    print("=" * 60)

    while True:
        try:
            user_input = input("\n👤 Вы: ").strip()
        except (EOFError, KeyboardInterrupt):
            print("\n\n👋 До свидания!")
            break

        if not user_input:
            continue

        # Обработка команд
        if user_input.lower() in ("quit", "exit"):
            print("👋 До свидания!")
            break

        if user_input.lower() == "clear":
            client.conversation = [{"role": "system", "content": SYSTEM_PROMPT}]
            print("🗑️ История очищена")
            continue

        if user_input.lower() == "print":
            print("=" * 50, "📋 История разговора:", "=" * 50, sep="\n")
            for i, msg in enumerate(client.conversation[1:], 1):
                print(f"{i}. {msg['role'].upper()}: {msg['content'][:100]}{'...' if len(msg['content']) > 100 else ''}")
            print("=" * 50)
            continue

        if user_input.lower().startswith("save "):
            name = user_input[5:].strip()
            path = save_conversation(client.conversation, name)
            print(f"💾 Сохранено: {path}")
            continue

        if user_input.lower().startswith("load "):
            name = user_input[5:].strip()
            loaded, msg = load_conversation(name)
            if loaded:
                client.conversation = loaded
                print(f"📂 Загружено: {msg}")
            else:
                print(f"❌ {msg}")
            continue

        if user_input.lower().startswith("temp "):
            try:
                temp = float(user_input[5:].strip())
                if 0.0 <= temp <= 2.0:
                    client.temperature = temp
                    print(f"🌡️ Температура установлена: {temp}")
                else:
                    print("⚠️ Температура должна быть от 0 до 2")
            except ValueError:
                print("⚠️ Пример: temp 0.7")
            continue

        # Обработка обычного запроса
        try:
            start_time = datetime.now()

            # Обрабатываем запрос (метод process_query сам добавляет сообщения в историю)
            response = await client.process_query(user_input)

            # Выводим ответ
            print(f"\n🤖 Ассистент: {response}")
            
            # Статистика
            elapsed = (datetime.now() - start_time).total_seconds()
            total_tools = len(client.mcp_client.available_tools) + len(client.docker_mcp_client.available_tools)
            print(f"⏱️ Время: {elapsed:.2f}с | Инструментов: {total_tools} (STDIO: {len(client.mcp_client.available_tools)}, Docker: {len(client.docker_mcp_client.available_tools)})")
            
        except Exception as exc:
            error_msg = f"❌ Ошибка: {exc}"
            log.error(error_msg)
            print(error_msg)

# --------------------  ГЛАВНАЯ ФУНКЦИЯ  --------------------
async def main() -> None:
    """Главная функция запуска."""
    # Создаем клиента
    client = ChatClient(model_name="glm-4.5-air")
    
    try:
        # Подключаемся к MCP серверу
        await client.start(SERVER_SCRIPT)
        
        # Запускаем интерактивный чат
        await interactive_chat(client)
        
    except KeyboardInterrupt:
        print("\n\n🛑 Прервано пользователем")
    except Exception as exc:
        error_msg = f"💥 Критическая ошибка: {exc}"
        log.exception(error_msg)
        print(error_msg)
    finally:
        # Освобождаем ресурсы
        await client.cleanup()

if __name__ == "__main__":
    asyncio.run(main())