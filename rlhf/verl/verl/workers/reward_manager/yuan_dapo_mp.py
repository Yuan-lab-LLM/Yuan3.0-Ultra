# Copyright 2024 Bytedance Ltd. and/or its affiliates
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

from collections import defaultdict

import torch

from verl import DataProto
from verl.utils.reward_score import default_compute_score
from verl.workers.reward_manager import register
from datetime import datetime
import asyncio
from concurrent.futures import ProcessPoolExecutor
from functools import partial
import time
import traceback
import ast
import sys
sys.path.append('verl/utils/reward_score/')
import torch.multiprocessing as mp
from verl.utils.reward_score.eval_external.model import llm_process
import multiprocessing
from concurrent import futures
import numpy as np
from verl.utils.reward_score.eval_llm.model import llm_process_reward_process_scoring
import os
import re


LLM_PROCESS_METHOD_FLAG = os.environ.get("LLM_PROCESS_METHOD_FLAG", "").split(',')


def step1_process(reward_inputs,reward_func,reward_vllm_args,overlong_buffer_cfgs,max_resp_len):
    num_processes=36
    scores=[]
    reward_func_ = partial(reward_func,args=reward_vllm_args,overlong_buffer_cfg=overlong_buffer_cfgs,max_resp_len=max_resp_len)
    with multiprocessing.Pool(processes=num_processes) as pool:
        for result in pool.imap(reward_func_, reward_inputs):
            scores.append(result)
    return scores



def thread_func(func,args,questions):
    results = []
    if len(questions) < 1:
        return results
    with futures.ThreadPoolExecutor(max_workers=len(questions)) as executor:
        future_results = [
            executor.submit(func, q, args,q_ord)
            for q_ord,q in enumerate(questions)
        ]
        for future in futures.as_completed(future_results):
            try:
                result = future.result()
                results.append(result)
            except Exception as exc:
                print(f"Generated an exception: {exc}")
                print(f"Traceback: {traceback.format_exc()}")
                results.append(None)
    return results


def process_inputs_externalmodel(reward_inputs,scores,func,args):

    reward_api_lst = [reward_input["reward_vllm_api"] for reward_input in reward_inputs if reward_input["reward_method"] in LLM_PROCESS_METHOD_FLAG]
    base_n = len(reward_api_lst)
    
    def zigzag_map(n, total):
        block = n // total
        offset = n % total
        return offset if block % 2 == 0 else total -1 - offset
    
    questions = [[{**reward_input, 'index': idx}, score] for idx, (reward_input, score) in enumerate(zip(reward_inputs, scores))]
    questions = sorted(questions, key=lambda d: len(d[0].get("response", "")), reverse=True)
    content = {}
    all_results = []
    reward_process_num = 0
    for idx,data in enumerate(questions):
        api = data[0]["reward_vllm_api"]

        if api == "none":
            all_results.append(data)
            continue
        
        if data[0]["reward_method"] in LLM_PROCESS_METHOD_FLAG:
            api = reward_api_lst[zigzag_map(reward_process_num, base_n)]
            reward_process_num += 1
            questions[idx][0]["reward_vllm_api"] = api

        if api not in content:
            content[api] = [data]
        else:
            content[api].append(data)

    if content:
        def split_list_2(lst, n=4):
            """使用numpy将列表平均分成n份"""
            if len(lst)<n:
                return np.array_split(lst, len(lst))
            return np.array_split(lst, n)
        num_processes = 64
        split_num = num_processes // len(content.keys())
        print(f"num_processes: {num_processes}; split_num:{split_num}")
        input_lst = []
        for key, value in content.items():
            if split_num > 0:
                value_split = split_list_2(value, split_num)
                for item in value_split:
                    input_lst.append(list(item))
            else:
                input_lst.append(value)


        thread_func_ = partial(thread_func,func,args)
        with multiprocessing.Pool(processes=num_processes) as pool:
            for result in pool.imap(thread_func_, input_lst):
                all_results.extend(result)

    results_sorted = sorted(all_results, key=lambda x: x[0]['index'])

    return results_sorted


@register("yuan_dapo_mp")
class MPDapoYuanRewardManager:
    """The reward manager."""

    def __init__(self, tokenizer, num_examine, compute_score=None, reward_fn_key="data_source",**reward_kwargs) -> None:
        """
        Initialize the NaiveRewardManager instance.

        Args:
            tokenizer: The tokenizer used to decode token IDs into text.
            num_examine: The number of batches of decoded responses to print to the console for debugging purpose.
            compute_score: A function to compute the reward score. If None, `default_compute_score` will be used.
            reward_fn_key: The key used to access the data source in the non-tensor batch data. Defaults to
                "data_source".
            reward_kwargs (dict): The keyword arguments to pass to the reward function.[store reward vllm api].
        """
        self.tokenizer = tokenizer  # Store the tokenizer for decoding token IDs
        self.num_examine = num_examine  # the number of batches of decoded responses to print to the console
        self.compute_score = compute_score or default_compute_score
        self.reward_fn_key = reward_fn_key  # Store the key for accessing the data source

        qwen72b_inst_ip=reward_kwargs["qwen72b_inst_host"]
        qwen72b_vl_ip = reward_kwargs["qwen72b_vl_host"]
        yuan_m32_ip = reward_kwargs["yuanm32_host"]
        bert_ip = reward_kwargs["bert_host"]

        process_reward_ip = reward_kwargs["process_reward_host"]

        self.reward_vllm_args = reward_kwargs["reward_vllm_args"]
        self.n_samples_per_prompt=reward_kwargs.get("n_samples_per_prompt")

        self.bert_api=bert_ip[0]+":8019"
        self.qwen72b_inst_api = []
        self.qwen72b_vl_api = []
        self.yuan_m32_api = []
        self.process_reward_api = []
        self.debug_reward_data_path = reward_kwargs["debug_reward_data_path"]

        for ip in qwen72b_inst_ip:
            self.qwen72b_inst_api.append(ip + ':7210')
            self.qwen72b_inst_api.append(ip + ':7211')


        for ip in qwen72b_vl_ip:
            self.qwen72b_vl_api.append(ip + ':7220')
            self.qwen72b_vl_api.append(ip + ':7221')


        for ip in yuan_m32_ip:
            self.yuan_m32_api.append(ip + ':7230')
            self.yuan_m32_api.append(ip + ':7231')
            self.yuan_m32_api.append(ip + ':7232')
            self.yuan_m32_api.append(ip + ':7233')

        for ip in process_reward_ip:
            self.process_reward_api.append(ip + ':7240')
            self.process_reward_api.append(ip + ':7241')
            self.process_reward_api.append(ip + ':7242')
            self.process_reward_api.append(ip + ':7243')


        self.overlong_buffer_cfg = reward_kwargs["overlong_buffer"]
        self.max_resp_len = reward_kwargs["max_resp_len"]

        if self.overlong_buffer_cfg is not None:
            assert self.max_resp_len is not None, (
                f"max_resp_len must be provided if {overlong_buffer_cfg=}, but got None"
            )
            assert self.max_resp_len >= self.overlong_buffer_cfg.len, (
                "max_resp_len must be larger than overlong_buffer.len"
            )


    def get_reward_input(self,data_item):
        prompt_ids = data_item.batch["prompts"]
        prompt_length = prompt_ids.shape[-1]

        valid_prompt_length = data_item.batch["attention_mask"][:prompt_length].sum()
        valid_prompt_ids = prompt_ids[-valid_prompt_length:]

        response_ids = data_item.batch["responses"]
        valid_response_length = data_item.batch["attention_mask"][prompt_length:].sum()
        valid_response_ids = response_ids[:valid_response_length]

        prompt_str = self.tokenizer.decode(valid_prompt_ids, skip_special_tokens=False)
        response_str = self.tokenizer.decode(valid_response_ids) ## for detect format

        if '<eod>' not in response_str:
            overlong_filter_flag = True
        else:
            overlong_filter_flag = False

        ground_truth = data_item.non_tensor_batch["reward_model"]["ground_truth"]
        reward_method = data_item.non_tensor_batch["reward_method"]
        language = data_item.non_tensor_batch["language"]  if  "language" in data_item.non_tensor_batch  else None
        data_source = data_item.non_tensor_batch["data_source"] if  "data_source" in data_item.non_tensor_batch  else None

        reward_input = dict(question=prompt_str,response=response_str,response_id = valid_response_ids,answer=ground_truth,reward_method=reward_method,language=language,data_source=data_source)

        if "mllm" in reward_method or "rag_mmtab"==reward_method:
            if "image_path" in data_item.non_tensor_batch["extra_info"].keys():
                image_path_data = data_item.non_tensor_batch["extra_info"]["image_path"]
            elif "imagepath" in data_item.non_tensor_batch["extra_info"].keys():
                image_path_data = data_item.non_tensor_batch["extra_info"]["imagepath"]
            else:
                image_path_data = None

            try:
                reward_input["imagepath"] = ast.literal_eval(image_path_data)
            except:
                reward_input["imagepath"] = image_path_data

        elif "sql" in reward_method:
            reward_input["db_id"]=data_item.non_tensor_batch["extra_info"]["db_id"]
            reward_input["question_id"]=data_item.non_tensor_batch["extra_info"]["question_id"]
            reward_input["table"]=data_item.non_tensor_batch["extra_info"]["table"]

        if reward_method == "mllm_general2":
            reward_input["multi_modal_data"] = data_item.non_tensor_batch["multi_modal_data"]

        reward_input["enable_thinking_flag"] = data_item.non_tensor_batch["extra_info"]["enable_thinking_flag"]
        reward_input["overlong_filter_flag"] = overlong_filter_flag
        return reward_input


    def __call__(self, data: DataProto, return_dict=False):
        """We will expand this function gradually based on the available datasets"""
        # If there is rm score, we directly return rm score. Otherwise, we compute via rm_score_fn

        if "rm_scores" in data.batch.keys():
            if return_dict:
                return {"reward_tensor": data.batch["rm_scores"]}
            else:
                return data.batch["rm_scores"]

        reward_tensor = torch.zeros_like(data.batch["responses"], dtype=torch.float32)
        reward_extra_info = defaultdict(list)

        already_print_data_sources = {}

        group_qwen72b_inst_key=['mllm_general1','rag_alpaca', 'rag_simpleqa', 'rag_frames']
        group_qwen72b_vl_key = ['mllm_general2']
        group_yuan_m32_key = ['llm_general']
        group_process_reward_key = LLM_PROCESS_METHOD_FLAG
        number_qwen72b_inst = 0
        number_qwen72b_vl = 0
        number_yuan_m32 = 0
        number_process_reward = 0

        reward_counters={
                "llm_general":{"idx":-1, "response_list":[]},
                "mllm_general2":{"idx":-1, "response_list":[]}
        }

        def process_general_reward(method, data_slice, reward_input):
            counter = reward_counters[method]
            counter["idx"] += 1

            if counter["idx"] % self.n_samples_per_prompt == 0:
                counter["response_list"] = [
                self.get_reward_input(tmpdata)["response"]
                for tmpdata in data_slice
                ]

            reward_input["response_list"] = counter["response_list"]
            reward_input["reward_flag"] = (counter["idx"] % self.n_samples_per_prompt == 0)

            return reward_input



        total_tasks=len(data)
        i=0
        reward_inputs=[]
        overlong_filter_flags = []

        start_time=time.time()

        while True:
            data_item=data[i]
            reward_input=self.get_reward_input(data_item)
            reward_method=reward_input["reward_method"]
            enable_thinking_flag = reward_input["enable_thinking_flag"]
            overlong_filter_flags.append(reward_input['overlong_filter_flag'])

            if reward_method in ["llm_general","mllm_general2"]:
                data_slice=data[i:i+self.n_samples_per_prompt]
                reward_input = process_general_reward(reward_method,data_slice,reward_input)

            if reward_method in group_qwen72b_inst_key:
                reward_input["reward_vllm_api"]=self.qwen72b_inst_api[number_qwen72b_inst % len(self.qwen72b_inst_api)]
                number_qwen72b_inst += 1
            elif reward_method in group_qwen72b_vl_key:
                reward_input["reward_vllm_api"]=self.qwen72b_vl_api[number_qwen72b_vl % len(self.qwen72b_vl_api)]
                if reward_input.get("reward_flag", False):
                    number_qwen72b_vl += 1
            elif reward_method in group_yuan_m32_key:
                reward_input["reward_vllm_api"]=self.yuan_m32_api[number_yuan_m32 % len(self.yuan_m32_api)]
                if reward_input.get("reward_flag", False):
                    number_yuan_m32 += 1
            elif reward_method=="rag_summary":
                reward_input["reward_vllm_api"]=self.bert_api
            elif reward_method in group_process_reward_key:
                reward_input['reward_vllm_api'] = self.process_reward_api[number_process_reward % len(self.process_reward_api)]
                number_process_reward += 1
            else:
                reward_input["reward_vllm_api"]="none"

            reward_inputs.append(reward_input)
            i += 1
            if i>=total_tasks:
                break
        end_time = time.time()
        print(f"compute score step1 time input_queue: {end_time - start_time}")

        start_time = time.time()
        time_str = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
        torch.save(reward_inputs, f"{self.debug_reward_data_path}/reward_inputs_step1_{time_str}.pt")
        scores=step1_process(reward_inputs, self.compute_score,self.reward_vllm_args,self.overlong_buffer_cfg,self.max_resp_len)
        end_time = time.time()
        print(f"compute score step1 time: {end_time - start_time}")
        print("compute score step1 end....")

        torch.save(scores,f"{self.debug_reward_data_path}/reward_scores_1_{time_str}.pt")


        start_time = time.time()
        results=process_inputs_externalmodel(reward_inputs,scores,llm_process,self.reward_vllm_args)
        end_time = time.time()
        print(f"process_line_elapsed_time: {end_time - start_time}")
        reward_inputs = [item[0] for item in results]
        scores = [item[1] for item in results]
        print("compute score step2 end....")

        torch.save(reward_inputs,f"{self.debug_reward_data_path}/reward_inputs_2_{time_str}.pt")
        torch.save(scores,f"{self.debug_reward_data_path}/reward_scores_2_{time_str}.pt")

        # process score
        start_time = time.time()
        inputs=list(zip(reward_inputs,scores))
        group_size = 8
        group_inputs = [inputs[i:i+group_size] for i in range(0, len(inputs), group_size)]

        with multiprocessing.pool.ThreadPool(processes=128) as pool:
            scores = list(pool.imap(llm_process_reward_process_scoring,group_inputs))

        scores = sum(scores,[])
        end_time = time.time()
        print(f"process score elapsed_time: {end_time - start_time}")
        torch.save(scores,f"{self.debug_reward_data_path}/reward_scores_3_{time_str}.pt")

        reward_counters_results={
                "llm_general":{"idx":-1, "acc_scores":None,"llm_general_judgement":None},
                "mllm_general2":{"idx":-1, "acc_scores":None,"llm_general_judgement":None}
        }

        def process_general_reward_results(method, scores_data):
            counter = reward_counters_results[method]
            counter["idx"] += 1

            rep_idx = counter["idx"] % self.n_samples_per_prompt

            if rep_idx == 0:
                counter["acc_scores"] = scores_data["acc_score"]
                counter["llm_general_judgement"] = scores_data["llm_general_judgement"]
                scores_data["acc_score"] = counter["acc_scores"][rep_idx]
            else:
                scores_data["acc_score"] = counter["acc_scores"][rep_idx]
                scores_data["llm_general_judgement"] = counter["llm_general_judgement"]

            scores_data['reward_score'] = counter["acc_scores"][rep_idx] if scores_data['reward_score'] != -1.0 else scores_data['reward_score']

            return scores_data

        all_keys = set()
        for score in scores:
            if score:
                all_keys.update(score.keys())


        boxed_pattern = re.compile(r'\\boxed\{([^}]*)\}')
        yn_pattern = re.compile(r'\b(yes|no|true|false)\b', re.IGNORECASE)
        uid_dict_mllm = {}
        for i in range(total_tasks):
            i_uid = data[i].non_tensor_batch["uid"]
            i_reward_method=data[i].non_tensor_batch["reward_method"]
            if i_reward_method in {"mllm_normal", "mllm_choice"}:
                assert 'acc_score' in scores[i], f"check mllm acc_score"
                i_acc_score = scores[i]['acc_score']
                assert i_acc_score == 1 or i_acc_score == -1, f"i_acc_score can only =1 or =-1, now get {i_acc_score}"
                if i_uid not in uid_dict_mllm:
                    uid_dict_mllm[i_uid] = {}
                    uid_dict_mllm[i_uid]['acc_score_all'] = []
                    uid_dict_mllm[i_uid]['text_all'] = []

                uid_dict_mllm[i_uid]['acc_score_all'].append(i_acc_score)

                prompt_ids = data[i].batch["prompts"]
                prompt_length = prompt_ids.shape[-1]

                response_ids = data[i].batch["responses"]
                valid_response_length = data[i].batch["attention_mask"][prompt_length:].sum()
                valid_response_ids = response_ids[:valid_response_length]
                i_response_str = self.tokenizer.decode(valid_response_ids) ## for detect format

                i_answer = data[i].non_tensor_batch["reward_model"]["ground_truth"]

                matches = boxed_pattern.findall(i_response_str)
                last_boxed = matches[-1] if matches else ""

                has_text_field = "text" in last_boxed
                answer_has_digit = bool(re.search(r"\d", i_answer))
                answer_has_yn = bool(yn_pattern.search(i_answer))
                only_has_abcde = i_answer.lower() in {"a", "b", "c", "d", "e"}

                delete_flag = (has_text_field) and (not answer_has_digit) and (not answer_has_yn) and (not only_has_abcde)
                uid_dict_mllm[i_uid]['text_all'].append(delete_flag)

        for ikey in uid_dict_mllm.keys():
            acc_score_all = uid_dict_mllm[ikey]['acc_score_all']
            assert len(acc_score_all) == self.n_samples_per_prompt
            acc_score_flag = sum(acc_score_all) <= 0
            uid_dict_mllm[ikey]['acc_score_flag'] = acc_score_flag

            text_all = uid_dict_mllm[ikey]['text_all']
            uid_dict_mllm[ikey]['text_flag'] = any(text_all)

            uid_dict_mllm[ikey]['delete_flag'] = uid_dict_mllm[ikey]['acc_score_flag'] and uid_dict_mllm[ikey]['text_flag']

        delete_set_mllm = set()

        for i in range(total_tasks):
            prompt_ids = data[i].batch["prompts"]
            prompt_length = prompt_ids.shape[-1]
            valid_response_length=data[i].batch["attention_mask"][prompt_length:].sum()
            reward_method=data[i].non_tensor_batch["reward_method"]
            enable_thinking_flag = data[i].non_tensor_batch["extra_info"]["enable_thinking_flag"]

            if scores[i] and not (scores[i].get("error",None) or scores[i].get("error") == "") and scores[i].get("reward_score",None) is not None:
                if reward_method in reward_counters_results:
                    scores[i] = process_general_reward_results(reward_method, scores[i])

                # 过程监督的数据不使用长度惩罚
                if reward_method not in group_process_reward_key:
                    scores[i]["reward_score"] = scores[i]["reward_score"] + scores[i]["overlong_reward"]

                scores[i]["overlong_filter_flag"] = overlong_filter_flags[i]

                # 多模态/过程监督数据 
                if "mllm" not in reward_method and "sft" not in reward_method and (reward_method not in group_process_reward_key) and scores[i]["lang_consistency_ratio"] < 0.0:
                    scores[i]["reward_score"] = scores[i]["reward_score"] + scores[i]["lang_consistency_ratio"]
                
                ## 格式惩罚
                if "general" not in reward_method and "sft" not in reward_method and (reward_method not in group_process_reward_key) and scores[i]["format"] < 0.0:
                    scores[i]["reward_score"] = scores[i]["reward_score"] + scores[i]["format"]


            else:
                if not scores[i] or not (scores[i].get("error",None) or scores[i].get("error") == ""):
                    print("score is None....")
                else:
                    print("score error.")
                    print(scores[i].get("error"))
                    print(scores[i].get("traceback"))

                scores[i]["reward_score"] = -3.0
                scores[i]["acc_score"]=-1.0
                scores[i]["overlong_reward"]=-1.0
                scores[i]["overlong"]=True
                scores[i]["overlong_filter_flag"]=None
                scores[i]["lang_consistency_ratio"]=-1.0

            i_uid = data[i].non_tensor_batch["uid"]
            if i_uid in uid_dict_mllm:
                if uid_dict_mllm[i_uid]['delete_flag']:
                    scores[i]["reward_score"] = -111
                    delete_set_mllm.add(i_uid)

            reward_tensor[i, valid_response_length - 1] = scores[i]["reward_score"]

            for key in all_keys:
                if key not in scores[i]:
                    scores[i][key] = None
            for key, value in scores[i].items():
                if "process" in key:
                    continue
                if key == "acc_score":
                    key = "acc"
                reward_extra_info[key].append(value)



        if return_dict:
            return {
                "reward_tensor": reward_tensor,
                "reward_extra_info": reward_extra_info,
                "delete_set_mllm": delete_set_mllm,
            }
        else:
            return reward_tensor


