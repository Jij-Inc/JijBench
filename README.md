# ParameterSearch

# 仕様
## **class Experiment**
### **Parameters**
|名前|説明||
|:---|:---|:---|
|**run_id**: int|solverの実行を区別するID|
|**experiment_id**: int or str |Experimentインスタンスを区別するID|
|**benchmark_id**: int or str|Benchmarkインスタンスを区別するID|
|**autosave**: bool|実験結果を自動で保存したい場合はTrue,そうでない場合はFalse|
|**autosave_dir**: str|`autosave=True`の時の実験結果の保存先|

### **Attributes**
|名前|説明||
|:---|:---|:---|
|**table**: pandas.DataFrame|実験設定や実験結果を格納する。ユーザは好きな情報を格納できる。||

### **Methods**
|名前|説明||
|:---|:---|:---|
|**insert_into_table(record: dict)**|**record**に記述されたデータを**table**に挿入する。**record**はdict型で書かなければならず、キーはtableの列名に使われ、値は対応するセルに代入される。この時、行方向の指定には**run_id**が使われる。このメソッドを呼ぶと最後に__next__が呼ばれ**run_id**を一つ進める。||
|**save(save_file: str)**|**table**をcsvで実験結果を保存する。`autosave=True`の時は**autosave_dir**で指定されたディレクトリ以下に**benchmark_{bechmark_id}/tables**というディレクトリが自動作成され、そのディレクトリ以下に**experiment_id_{experiment_id}.csv**というファイル名で**table**を保存する。`autosave=False`の場合、**save_file**で指定されるファイル名で**table**を保存する。
|**load(load_file: str)**|saveメソッドで保存した結果を読み込み、**table**に代入する。`autosave=True`場合、 **autosave_dir**以下の**experiment_id**、**benchmark_id**で指定される結果を自動で読み込み、`autosave=False`の場合、**load_file**で指定されるファイルを読み込む。||

### **Examples**
最も単純な使い方
```python
# Example 1
# ユーザ定義のsolverの帰り値（何でも良い）
sample_response = {"hoge": {"fuga": 1}}

with Experiment() as experiment:
    for param in [10, 100, 1000]:
        for step in range(3):
            # solverは上のsample_responseを返す想定
            # sample_response = solver()
            # experiment.tableに登録するrecordを辞書型で作成
            record = {
                "step": step,
                "param": param,
                "results": sample_response,
            }
            experiment.insert_into_table(record)
    experiment.save()

```
実験結果を保存したい場所を指定する。experiment_idとbenchmark_idを明示的に指定すると結果の保存と読み込みの対応関係がつけやすくなる。
```python
# Example 2
save_dir = "/home/azureuser/data/jijbench"
experiment_id = "test"
benchmark_id = 0

with Experiment(
    experiment_id=experiment_id, benchmark_id=benchmark_id, autosave_dir=save_dir
) as experiment:
    for param in [10, 100, 1000]:
        for step in range(3):
            # sample_response = solver()
            record = {
                "step": step,
                "param": param,
                "results": sample_response,
            }
            experiment.insert_into_table(record)
    experiment.save()

# 以前実験した結果を読み込む。experiment_idとbenchmark_idを覚えていれば対応する実験を読み込める。
# もちろんファイル名を直接指定しても良い。その場合はautosave=Falseにしてloadの引数でファイル名を指定する。
with Experiment(
    experiment_id=experiment_id, benchmark_id=benchmark_id, autosave_dir=save_dir
) as experiment:
    experiment.load()
```

# 実行方法
# パラメータのアップデート

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