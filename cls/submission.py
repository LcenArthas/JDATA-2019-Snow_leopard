#用来集成模型的结果

import os
import pandas as pd


flie_list = os.listdir('./ensemble/test_b/')
new_df = pd.DataFrame()

for i, f in enumerate(flie_list):
    csv_f = './ensemble/test_b/' + f
    df = pd.read_csv(csv_f)
    video = df['video_name']

    v_class = df['class']
    new_df[str(i)] = v_class

video = list(video)
result = []
for index, row in new_df.iterrows():
    result.append(row.mode()[0])                           #求出众数

df = pd.DataFrame({'video_name': video, 'class': result})
df.to_csv('submisssion_191613.csv', index=False)