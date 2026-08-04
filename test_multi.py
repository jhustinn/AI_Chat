import asyncio
import httpx
import time

SERVER_URL = "http://localhost:8000/v1/chat/completions"

async def send_request(client: httpx.AsyncClient, prompt: str, request_id: int):
    payload = {
        "model": "qwen2.5-1.5b",
        "messages": [{"role": "user", "content": prompt}],
        "max_tokens": 100,
        "temperature": 0.7,
    }
    start = time.time()
    resp = await client.post(SERVER_URL, json=payload, timeout=60)
    elapsed = time.time() - start
    data = resp.json()
    reply = data["choices"][0]["message"]["content"]
    return request_id, reply, elapsed

async def main():
    prompts = [
        "Apa itu Python? Jawab singkat.",
        "Jelaskan machine learning dalam 2 kalimat.",
        "Sebutkan 3 bahasa pemrograman populer.",
        "Apa fungsi GPU dalam AI?",
        "Apa perbedaan CPU dan GPU?",
    ]

    print(f"Sending {len(prompts)} concurrent requests...\n")
    start_total = time.time()

    async with httpx.AsyncClient() as client:
        tasks = [send_request(client, p, i+1) for i, p in enumerate(prompts)]
        results = await asyncio.gather(*tasks)

    total_time = time.time() - start_total

    for req_id, reply, elapsed in results:
        print(f"--- Request {req_id} ({elapsed:.2f}s) ---")
        print(reply[:200])
        print()

    print(f"=== Total time for {len(prompts)} concurrent requests: {total_time:.2f}s ===")

if __name__ == "__main__":
    asyncio.run(main())
