import json
import openai
import time
openai.api_type = "azure"
openai.api_version = "2023-03-15-preview"
openai.api_base = "<azure openai endpoint>"
openai.api_key = "<azure openai key>"


def prompt_gen(room_type, style, prompt_file="prompts.json", max_retry=5):
    with open(prompt_file, 'r') as f:
        prompts = json.load(f)
    sys_prompt = prompts['system']
    user_prompt = prompts['user']
    for _ in range(max_retry):
        try:
            response = openai.ChatCompletion.create(
                engine="chatgpt",
                messages=[
                    {"role": "system", "content": sys_prompt},
                    {"role": "user", "content": user_prompt.replace("{room_type}", room_type.lower()).replace("{style}", style.lower())}
                ],
                temperature=1.,
                max_tokens=50,
                top_p=0.5,
                frequency_penalty=0,
                presence_penalty=0,
                stop=None
            )
            break
        except Exception as e:
            time.sleep(3)
            continue

    res = response['choices'][0]['message']['content']
    if '\n' in res:
        res = res.split('\n')[0]
    return res.replace("User:", "").replace("Assistant", "").strip().strip('.')


if __name__ == '__main__':
    print(prompt_gen("living room", "modern"))
