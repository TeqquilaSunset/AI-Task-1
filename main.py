from zai import ZaiClient
import os
import json
from dotenv import load_dotenv

load_dotenv()

SYSTEM_PROMPT = "Ты помощник, который помогает с любыми вопросами"

def main():
    client = ZaiClient(api_key=os.getenv("ZAI_API_KEY"))

    print("=" * 50)
    print("Консольный клиент GLM-4.5-flash. Введите 'quit' для выхода.")
    print("=" * 50)

    conversation = [
        {"role": "system", "content": f"{SYSTEM_PROMPT}"}
    ]

    temp= 1.0
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

        # Добавление запроса в историю диалога
        conversation.append({"role": "user", "content": user_input})

        temp = 1.8
        try:
            # Отправка запроса к модели
            response = client.chat.completions.create(
                model="GLM-4.6",  # Используемая модель
                messages=conversation,  # История разговора
                temperature=temp,
                max_tokens=2048,
                thinking={ "type": "disabled" }
                #response_format={"type": "json_object"} # Вроде удалось и без этой штуки добится нужного формата, но с ней надежнее
            )

            # Получение и вывод ответа
            ai_response = response.choices[0].message.content

            conversation.append({"role": "assistant", "content": ai_response})

            # First print the raw AI response
            print(f"\nTemperature: {temp}")
            print(f"\nAI: {ai_response}")

        except Exception as e:
            print(f"\n Произошла ошибка: {e}")

if __name__ == "__main__":
    main()
