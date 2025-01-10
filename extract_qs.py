import pandas as pd
import random
import copy

df = pd.read_csv('data/all_replies.xlsx - Sheet 1-no-modelinfo.csv',header=0, index_col=0)
df = df[['type', 'question']]
print(df.shape)
assert df.shape[0] % 3==0

n_qs = df.shape[0]//3

models = ['gpt4','claude3', 'llama3']
all_models = []
for q_id in range(n_qs):
    random.shuffle(models)
    all_models.extend(copy.deepcopy(models))
    # q = df.iloc[q_id*3]
    # print(q)
    # print(df.iloc[q_id*3:q_id*3+3])
    # print()
df['model'] = all_models
df['reply'] = "none"
df.to_csv('res/question-model-pairs.csv')
print(df)
# print(df['question'].unique())

# # column_name='question'
# # # Get boolean mask for unique values
# # unique_values_mask = ~df[column_name].duplicated(keep=False)
# # print(unique_values_mask)

# # Filter the DataFrame
# df = df[unique_values_mask]


# # df.groupby('type')
# print(df)