#### Requirements

git  https://github.com/ShishirPatil/gorilla/tree/cd9429ccf3d4d04156affe883c495b3b047e6b64

```
cd gorilla/berkeley-function-call-leaderboard

pip install -e .
```

Modify the source code and replace the base_url with the specified API service.

#### Inference

```
bfcl generate --model model_path --test-category all --num-threads 16 --include-input-log 
```

 Save the results to `gorilla/berkeley-function-call-leaderboard/result`

#### Score

```
python3 -m bfcl_eval.eval_checker.eval_runner --model model_path --test-category all --result-dir result
```

 Save the results to `gorilla/berkeley-function-call-leaderboard/score`.

The scores are aggregated in `gorilla/berkeley-function-call-leaderboard/score/data_overall.csv`.