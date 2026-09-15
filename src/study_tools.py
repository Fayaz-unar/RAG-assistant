```python
from .prompts import (
    create_summary_prompt,
    create_questions_prompt
)


def generate_summary(client, context):
    prompt = create_summary_prompt(context)

    response = client.chat.completions.create(
        model="openai/gpt-oss-120b",
        messages=[
            {
                "role": "system",
                "content": "You are an expert academic study assistant."
            },
            {
                "role": "user",
                "content": prompt
            }
        ],
        reasoning_effort="medium"
    )

    return response.choices[0].message.content


def generate_questions(
    client,
    context,
    question_type,
    number
):
    prompt = create_questions_prompt(
        context,
        question_type,
        number
    )

    response = client.chat.completions.create(
        model="openai/gpt-oss-120b",
        messages=[
            {
                "role": "system",
                "content": "You are an expert university question-paper generator."
            },
            {
                "role": "user",
                "content": prompt
            }
        ],
        reasoning_effort="medium"
    )

    return response.choices[0].message.content
```
