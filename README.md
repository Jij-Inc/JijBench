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



# Benchmark Instances
## Nurse Scheduling Problem
使用インスタンス: http://www.schedulingbenchmarks.org/nrp/

問題サイズ
| Instance   | Weeks | Employees | Shift types | Best known lower bound | Best known solution | 
| ---------- | ----: | --------: | ----------: | ---------------------: | ------------------: | 
| Instance1  | 2     | 8         | 1           | 607                    | 607                 | 
| Instance2  | 2     | 14        | 2           | 828                    | 828                 | 
| Instance3  | 2     | 20        | 3           | 1001                   | 1001                | 
| Instance4  | 4     | 10        | 2           | 1716                   | 1716                | 
| Instance5  | 4     | 16        | 2           | 1143                   | 1143                | 
| Instance6  | 4     | 18        | 3           | 1950                   | 1950                | 
| Instance7  | 4     | 20        | 3           | 1056                   | 1056                | 
| Instance8  | 4     | 30        | 4           | 1300                   | 1300                | 
| Instance9  | 4     | 36        | 4           | 439                    | 439                 | 
| Instance10 | 4     | 40        | 5           | 4631                   | 4631                | 
| Instance11 | 4     | 50        | 6           | 3443                   | 3443                | 
| Instance12 | 4     | 60        | 10          | 4040                   | 4040                | 
| Instance13 | 4     | 120       | 18          | 1348                   | 1348                | 
| Instance14 | 6     | 32        | 4           | 1278                   | 1278                | 
| Instance15 | 6     | 45        | 6           | 3829                   | 3831                | 
| Instance16 | 8     | 20        | 3           | 3225                   | 3225                | 
| Instance17 | 8     | 32        | 4           | 5746                   | 5746                | 
| Instance18 | 12    | 22        | 3           | 4459                   | 4459                | 
| Instance19 | 12    | 40        | 5           | 3149                   | 3149                | 
| Instance20 | 26    | 50        | 6           | 4769                   | 4769                | 
| Instance21 | 26    | 100       | 8           | 21133                  | 21133               | 
| Instance22 | 52    | 50        | 10          | 30240                  | 30244               | 
| Instance23 | 52    | 100       | 16          | 16990                  | 17428               | 
| Instance24 | 52    | 150       | 32          | 26571                  | 42463               | 
