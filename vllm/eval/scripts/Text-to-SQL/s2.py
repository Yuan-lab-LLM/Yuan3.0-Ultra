# 读取jsonl文件
import json
# /icfsikc/wangcarol/output_test_40b/sql/ckpt_iter_0000332

# /icfsikc/wangcarol/output_test_40b/sql/ckpt_epoch0.4_iter086
# /icfsikc/wangcarol/output_test_40b/sql/ckpt_epoch0.8_iter172
# /icfsikc/wangcarol/output_test_40b/sql/ckpt_epoch1.2_iter258
# /icfsikc/wangcarol/output_test_40b/sql/ckpt_epoch1.6_iter344

rootpath = r'/mnt/beegfs/suntianxing/result1/shotcot/text2sql/split'

with open(f'{rootpath}/dev_spider_clear.jsonl', 'r', encoding='utf-8') as file:
    lines = file.readlines()
with open(f'{rootpath}/spider_gold.txt', "w", encoding="utf-8") as f, open(f"{rootpath}/spider_pred.txt", "w", encoding="utf-8") as g:
    for line in lines:
        # 处理每一行数据
        data = json.loads(line)
        # 将处理后的数据写入新的jsonl文件
        if data['gold']:
            f.write(data['gold'] + "\n")
        else:
            f.write("\n")
        if data['predict']:
            g.write(data['predict'].replace("\n", " ").replace(";", "") + "\n")
        else:
            g.write("\n")

print('echo >>>>>>>')
with open(f'{rootpath}/dev_bird_clear.jsonl', 'r', encoding='utf-8') as file:
    lines = file.readlines()
with open(f"{rootpath}/bird_gold.json", "w", encoding="utf-8") as f, open(f"{rootpath}/bird_pred.json", "w", encoding="utf-8") as g:
    res_1 = {}
    res_2 = {}
    for line in lines:
        # 处理每一行数据
        data = json.loads(line)
        # 将处理后的数据写入新的jsonl文件
        res_1[data['question_id']] = data['gold']

        if data['predict']:
            res_2[data['question_id']] = data['predict'].replace("\n", " ") + "\t----- bird -----\t" + data['gold'].split("\t----- bird -----\t")[1]
        else:
            res_2[data['question_id']] = "??\t----- bird -----\t??"

    json.dump(res_1, f, ensure_ascii=False, indent=4)
    json.dump(res_2, g, ensure_ascii=False, indent=4)

print('done >>>>')
