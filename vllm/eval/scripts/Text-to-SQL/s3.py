import re
import json

ckpt_list = ['ckpt_iter_6498']
file_list = ['spider', 'bird']

for ckpt_name in ckpt_list:
    for file_name in file_list:
        print(ckpt_name, file_name)
        input = rf'/mnt/beegfs/suntianxing/result1/longcot/text2sql/split/dev_{file_name}.jsonl'
        output = rf'/mnt/beegfs/suntianxing/result1/longcot/text2sql/split/dev_{file_name}_clear.jsonl'

        with open(input, 'r', encoding='utf-8') as file, open(output, 'w', encoding='utf-8') as wr:
            for line in file:
                data = json.loads(line)
                answer = data['answer']
                db_id = data['db_id']
                if file_name == 'spider':
                    gold = answer + '\t' + db_id
                else:
                    gold = f'{answer}\t----- bird -----\t{db_id}'
                
                solution = data['yuan_solution']
                predict = None  # 默认为 None

                # 无论有没有 </think>，都尝试从 solution 提取 SQL
                if '</think>' in solution:
                    result = solution.split('</think>')[-1]
                else:
                    result = solution  # 没有 </think>，整段作为候选

                pattern = r'```sql.*?\n(.*?)```'
                code_match = re.search(pattern, result, re.DOTALL)
                if code_match:
                    predict = code_match.group(1).strip()
                else:
                    # 去除特殊 token 后作为 fallback
                    predict = result.replace('<|end_of_sentence|>', '').replace('<eod>', '').strip()
                    # 若清理后为空，则设为 None（可选）
                    if not predict:
                        predict = None

                data['gold'] = gold
                data['predict'] = predict
                wr.write(json.dumps(data, ensure_ascii=False) + '\n')
