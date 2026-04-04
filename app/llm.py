from openai import OpenAI

client = OpenAI()

def ask_llm(prompt: str) -> str:
    res = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[{"role": "user", "content": prompt}],
        temperature=0.7
    )
    return res.choices[0].message.content