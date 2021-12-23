# ParameterSearch

Research Project about parameter search

# 仕様

main のファイルは `parameter_test.py` です. その内部で 問題 (`jm.problem`) と 問題のインスタンスを引っ張ってきて繰り返しときます.

みなさんおのおのが書き込む場所は, `user_script.py`の`transpile_problem(problem: Problem)`, `make_initial_multipliers(problem: Problem)` と `parameter_update()` です.

`Problem` のディレクトリ内でそれぞれの問題（qubo）を作成する関数を書いています.

`Instances` のディレクトリ内で, Problem で作成したそれぞれの問題に対応するインスタンスを保存してあります.

現状 AGC の問題しかないです.

# 実行方法

`ParameterSearch`ディレクトリに入って,

```
python -m parameter_test
```

で実行してください.

# パラメータのアップデート (`user_script.py`)


`transpile_problem(problem: Problem)` : 問題の形式を変形したい場合はここで変更してください. 必ず class Problem を返してください.

`make_initial_multipliers(problem: Problem)`: 問題を受け取って, パラメータの初期値を設定してください. 今書かれていいるスクリプトは初期値が全て 1 になるようにしています. 適宜修正してください.

`parameter_update(problem: Problem, decode: DecodedSamples, multipliers: Dict[str, float]))`: 問題・解（DecodedSamples）・現在の multiliers を入力として, 次の multipliers を出力してください. 例えば今書かれているスクリプトでは, 制約を守らない項のパラメータを 5 倍にするようにしています.
