from zai import ZaiClient
import os
import json
from dotenv import load_dotenv

load_dotenv()

SYSTEM_PROMPT = """
Ты — высокоточный климатический API, который отвечает исключительно в формате JSON.
Твой ответ должен быть валидным JSON объектом.
Ты не должен использовать Markdown или любые другие форматы.
Ты не должен использовать ``` в начале и в конце ответа.

ЖЁСТКОЕ ПРАВИЛО:
1. НЕ ПИШИ никаких объяснений, вводных фраз или дополнительного текста ДО или ПОСЛЕ JSON.
2. В ответе должен быть ТОЛЬКО ОДИН JSON объект.

СТРУКТУРА JSON (Обязательные поля):
{
  "statusMessage": "Выполненно, описание ошибки или не достающей информации",
  "statusCode": "0 - успех, 1 - ошибка",
  "city": "Название города (строка)",
  "unit": "Единица измерения (строка, например, Цельсий)",
  "source": "Источник данных (строка)",
  "temperatures": {
    "название_месяца_в_англ_н_регистре": {
      "ru": "Русское название месяца (строка)",
      "avg_temp": "Средняя температура (число с плавающей точкой)"
    }
    // ... и так для всех 12 месяцев.
  }
}
"""

def validate_ai_response(response_text):
    """
    Validates the AI response by parsing it as JSON and checking if it matches the expected schema.
    """
    try:
        # Try to parse the response as JSON
        parsed_response = json.loads(response_text)

        # Define the required fields in the JSON object
        required_fields = {"statusMessage", "statusCode", "city", "unit", "source", "temperatures"}

        # Check if all required fields are present
        if not all(field in parsed_response for field in required_fields):
            print("Ошибка: Ответ AI не содержит все обязательные поля JSON.")
            return None

        # Validate the structure of temperatures (should be a dictionary with month objects)
        if not isinstance(parsed_response["temperatures"], dict):
            print("Ошибка: Поле 'temperatures' должно быть объектом с месяцами.")
            return None

        # Check if temperature objects have the correct structure
        for month_key, month_value in parsed_response["temperatures"].items():
            if not isinstance(month_value, dict):
                print(f"Ошибка: Месяц '{month_key}' должен быть объектом.")
                return None
            if "ru" not in month_value or "avg_temp" not in month_value:
                print(f"Ошибка: Месяц '{month_key}' не содержит обязательных полей 'ru' и 'avg_temp'.")
                return None
            if not isinstance(month_value["ru"], str):
                print(f"Ошибка: Поле 'ru' для месяца '{month_key}' должно быть строкой.")
                return None
            if not isinstance(month_value["avg_temp"], (int, float)):
                print(f"Ошибка: Поле 'avg_temp' для месяца '{month_key}' должно быть числом.")
                return None

        # Validate data types for other fields
        if not isinstance(parsed_response["statusMessage"], str):
            print("Ошибка: Поле 'statusMessage' должно быть строкой.")
            return None

        if not isinstance(parsed_response["statusCode"], (str, int)):
            print("Ошибка: Поле 'statusCode' должно быть строкой или числом.")
            return None

        if not isinstance(parsed_response["city"], str):
            print("Ошибка: Поле 'city' должно быть строкой.")
            return None

        if not isinstance(parsed_response["unit"], str):
            print("Ошибка: Поле 'unit' должно быть строкой.")
            return None

        if not isinstance(parsed_response["source"], str):
            print("Ошибка: Поле 'source' должно быть строкой.")
            return None

        return parsed_response

    except json.JSONDecodeError:
        print("Ошибка: Ответ AI не является валидным JSON.")
        return None
    except Exception as e:
        print(f"Ошибка при валидации ответа AI: {e}")
        return None

def main():
    client = ZaiClient(api_key=os.getenv("ZAI_API_KEY"))

    print("=" * 50)
    print("Консольный клиент GLM-4.5-flash. Введите 'quit' для выхода.")
    print("=" * 50)

    conversation = [
        {"role": "system", "content": f"{SYSTEM_PROMPT}"}
    ]

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

        # Добавление запроса в историю диалога
        conversation.append({"role": "user", "content": user_input})

        try:
            # Отправка запроса к модели
            response = client.chat.completions.create(
                model="GLM-4.5-Air",  # Используемая модель
                messages=conversation,  # История разговора
                temperature=0.9,
                max_tokens=2048,
                thinking={ "type": "disabled" }
                #response_format={"type": "json_object"} # Вроде удалось и без этой штуки добится нужного формата, но с ней надежнее
            )

            # Получение и вывод ответа
            ai_response = response.choices[0].message.content

            # First print the raw AI response
            print(f"\nAI: {ai_response}")

            # Validate the AI response
            validated_response = validate_ai_response(ai_response)

            if validated_response:
                print("Удалось распарсить ответ.")
                # Use the validated response in the conversation history
                conversation.append({"role": "assistant", "content": json.dumps(validated_response, ensure_ascii=False)})
            else:
                print("Не удалось распарсить ответ.")
                # Optionally, you could ask the AI to try again here
                conversation.append({"role": "assistant", "content": ai_response})

        except Exception as e:
            print(f"\n Произошла ошибка: {e}")

if __name__ == "__main__":
    main()