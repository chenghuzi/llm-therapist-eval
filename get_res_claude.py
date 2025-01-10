import pandas as pd
from openai import OpenAI
from common import api_base, api_key, sys_prompt
import tqdm

fn = "res/question-model-pairs.csv"
df = pd.read_csv(fn, header=0, index_col=0)


print(df)


client = OpenAI(
    base_url=api_base,
    api_key=api_key,
)
print(client.api_key)
err_count = 0
phbar = tqdm.tqdm(df.iterrows(), total=df.shape[0])
for id, row in phbar:
    if row["model"] != "claude3":
        continue
    if row["reply"] != "none":
        continue
    phbar.set_postfix({"err": err_count, "id": id, "Q": row["question"]})
    try:
        resp = client.chat.completions.create(
            model="claude-3-5-sonnet-20241022",
            messages=[
                {"role": "system", "content": sys_prompt},
                {"role": "user", "content": row["question"]},
            ],
            # 流式输出
            stream=False,
        )

        df.loc[id, "reply"] = resp.choices[0].message.content
        df.to_csv(fn)
    except Exception:
        err_count += 1
        continue
