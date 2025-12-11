from openai import OpenAI
import os
from dotenv import load_dotenv
from datetime import datetime
import httpx

load_dotenv()

SYSTEM_PROMPT = "Ты помощник, который помогает с любыми вопросами"

def create_summary_with_llm(client, model_name, conversation_history):
    """Create a summary of all previous user requests and AI responses using LLM"""
    # Exclude the system prompt (index 0) and only include user and assistant messages
    user_ai_messages = [msg for msg in conversation_history[1:] if msg["role"] in ["user", "assistant"]]

    if len(user_ai_messages) == 0:
        return "Нет истории диалога для создания резюме."

    # Format the conversation history for the LLM to summarize
    formatted_history = "Пожалуйста, создай краткое резюме следующей истории диалога. Выдели основные темы и детали диалога, которые могут понадобится при дальнейшем общении:\n\n"
    for i, message in enumerate(user_ai_messages):
        role = "Пользователь" if message["role"] == "user" else "AI"
        content = message["content"]
        formatted_history += f"{role}: {content}\n\n"

    try:
        # Send the formatted history to the LLM for summarization
        response = client.chat.completions.create(
            model=model_name,
            messages=[{"role": "user", "content": formatted_history}],
            temperature=0.3,
            max_tokens=2048
        )

        summary = response.choices[0].message.content
        return summary
    except Exception as e:
        return f"Ошибка при создании резюме: {e}"

def print_context(system_prompt, conversation_history):
    """Print the full context (system prompt + conversation history)"""
    print("="*50)
    print("ПОЛНЫЙ КОНТЕКСТ:")
    print("="*50)
    print(f"Системный промпт: {system_prompt}")
    print("-"*50)
    print("История разговора:")

    for i, message in enumerate(conversation_history[1:], 1):  # Skip system prompt
        role = message["role"].upper()
        content = message["content"]
        print(f"{i}. {role}: {content}")

    print("="*50)

def main():
    # Initialize OpenAI client
    # You can use either OpenAI API or an OpenAI-compatible service

    # Z.AI
    api_key = os.getenv("OPENAI_API_KEY")
    base_url = os.getenv("OPENAI_BASE_URL")
    model_name = "glm-4.5-air"

    # Disable SSL certificate verification
    http_client = httpx.Client(verify=False)

    if base_url:
        client = OpenAI(api_key=api_key, base_url=base_url, http_client=http_client)
    else:
        client = OpenAI(api_key=api_key, http_client=http_client)

    print("=" * 50)
    print("Консольный клиент OpenAI-совместимой модели. Введите 'quit' для выхода.")
    print("=" * 50)

    conversation = [
        {"role": "system", "content": f"{SYSTEM_PROMPT}"}
    ]

    temp = 1.0
    print(f"Используемая модель: {model_name}")

    while True:
        # Получение запроса от пользователя
        try:
            user_input = input("\nВы: ")
        except (EOFError, KeyboardInterrupt):
            print("\n\nВыход.")
            break

        if user_input.lower() == 'quit':
            print("До свидания!")
            break

        # Check if the user wants to change the temperature
        if user_input.lower().startswith('temp '):
            try:
                # Extract the numeric value after 'temp '
                temp_value = float(user_input[5:].strip())  # Skip 'temp ' (5 characters) and get the number
                if 0.0 <= temp_value <= 2.0:  # Validate temperature range
                    temp = temp_value
                    print(f"Температура установлена на {temp}")
                    continue  # Skip to the next iteration without sending to AI
                else:
                    print("Температура должна быть в диапазоне от 0.0 до 2.0")
                    continue
            except ValueError:
                print("Пожалуйста, укажите числовое значение для температуры, например: temp 0.7")
                continue

        # Check if the user wants to print the full context
        if user_input.lower() == 'print':
            print_context(SYSTEM_PROMPT, conversation)
            continue  # Skip to the next iteration without sending to AI

        # Check if the user wants to get a summary of previous requests
        if user_input.lower() == 'summary':
            print("Создание резюме предыдущей истории...")
            summary = create_summary_with_llm(client, model_name, conversation)
            print(f"\nРезюме: {summary}")

            # Replace conversation history with system prompt and summary only
            system_message = conversation[0]  # Keep the system prompt
            summary_message = f"Суммаризация предыдущего разговора: {summary}"
            conversation = [system_message, {"role": "assistant", "content": summary_message}]
            continue  # Skip to the next iteration without sending to AI

        # Добавление запроса в историю диалога
        conversation.append({"role": "user", "content": user_input})

        try:
            # Record start time for request
            start_time = datetime.now()

            # Отправка запроса к модели
            response = client.chat.completions.create(
                model=model_name,  # Используемая модель
                messages=conversation,  # История разговора
                temperature=temp,
                max_tokens=2048,
                # Note: The thinking parameter from GLM is removed as it's not compatible with OpenAI interface
            )

            # Получение и вывод ответа
            ai_response = response.choices[0].message.content

            conversation.append({"role": "assistant", "content": ai_response})

            # Calculate request time
            end_time = datetime.now()
            request_duration = end_time - start_time

            # Get token usage information
            usage_info = response.usage
            tokens_prompt = usage_info.prompt_tokens if hasattr(usage_info, 'prompt_tokens') else "N/A"
            tokens_completion = usage_info.completion_tokens if hasattr(usage_info, 'completion_tokens') else "N/A"
            tokens_total = usage_info.total_tokens if hasattr(usage_info, 'total_tokens') else "N/A"

            # Print the raw AI response and additional information
            print(f"\nTemperature: {temp}")
            print(f"\nAI: {ai_response}")

            # Print request time and token usage
            print(f"\n--- Справочная информация ---")
            print(f"Время запроса: {request_duration.total_seconds():.2f} секунд")
            print(f"Расход токенов:")
            print(f"  - Вопрос: {tokens_prompt}")
            print(f"  - Ответ: {tokens_completion}")
            print(f"  - Всего: {tokens_total}")
            print(f"-----------------------------")

        except Exception as e:
            print(f"\n Произошла ошибка: {e}")

if __name__ == "__main__":
    main()
