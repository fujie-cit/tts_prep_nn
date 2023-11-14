# %% [markdown]
# # 日本語平文と韻律記号付き音素列のペアデータの生成
# 
# CSJ書き起こしの平文（TRNのForm2）の各文に対してESPnetの前処理（PyOpenJTalkを用いた韻律記号付き音素列）を適用した結果をJSONファイル（all.json）に保存する

# %%
# CSJ の TRN ファイルがあるディレクトリ
CSJ_TRN_DIR = "/autofs/diamond/share/corpus/CSJ/TRN/Form2"

# %%
import os

# TRNファイルのリストを作成
trn_files = []
for root, dirs, files in os.walk(CSJ_TRN_DIR):
    for file in files:
        if file.endswith('.trn'):
            trn_files.append(os.path.join(root, file))

# %%
from csj_formatter import remove_tag_from_plain_tagged_string
from espnet_phoneme_tokenizer import pyopenjtalk_g2p_prosody

def read_and_format_trn_file(filename):
    # 講演ID
    session_id = os.path.basename(filename).split('.')[0]

    results = []
    with open(filename, 'r', encoding='sjis') as f:
        for line in f:
            utt_id, _, text = line.rstrip().split(' ', 2)
            channel = text[0]
            text = text[2:]

            if 'R' in text:
                continue
            if '<' in text:
                continue

            try:
                formatted_text = remove_tag_from_plain_tagged_string(text)
            except ValueError as e:
                # print(e)
                # print(text)
                continue

            if len(formatted_text) == 0:
                continue

            phoneme_text = ' '.join(pyopenjtalk_g2p_prosody(formatted_text))

            if len(phoneme_text) == 0:
                continue
            
            results.append({
                'session_id': session_id,
                'utt_id': int(utt_id),
                'channel': channel,
                'text': formatted_text,
                'phoneme_text': phoneme_text,
            })
            
        return results

# %%
import tqdm

results = []
for filename in tqdm.tqdm(trn_files):
    results.extend(read_and_format_trn_file(filename))

# %%
import json
json.dump(results, open('all.json', 'w', encoding='utf-8'), ensure_ascii=False)


