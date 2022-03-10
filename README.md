# JijBenchmark

[![Test](https://github.com/Jij-Inc/JijBenchmark/actions/workflows/python-test.yml/badge.svg)](https://github.com/Jij-Inc/JijBenchmark/actions/workflows/python-test.yml)
## Install from JFrog
```shell
pip install jijbench --extra-index-url https://jij.jfrog.io/artifactory/api/pypi/Jij-Private/simple
```

## For Developers

### poetryで開発用のパッケージを管理しています

テストコードで必要なライブラリがあれば

```shell
poetry add -D "package name"
```

でパッケージを追加するようにしてください。

### pytest を使ってテストを書いています

以下のいずれかでテストを実行することができます。

```shell
python -m pytest tests # 全てのテストの実行
python -m pytest tests/"file name" # テストファイルを指定して実行
python -m pytest tests/"file name"::"function name"  # 関数を指定して実行
```



# 仕様
## **class Experiment**
### **Parameters**
|名前|説明||
|:---|:---|:---|
|**run_id**: int|solverの実行を区別するID|
|**experiment_id**: int or str |Experimentインスタンスを区別するID|
|**benchmark_id**: int or str|Benchmarkインスタンスを区別するID|
|**autosave**: bool|実験結果を自動で保存したい場合はTrue,そうでない場合はFalse|
|**save_dir**: str|`autosave=True`の時の実験結果の保存先|

### **Attributes**
|名前|説明||
|:---|:---|:---|
|**table**: pandas.DataFrame|実験設定や実験結果を格納する。ユーザは`pandas.DataFrame`で扱える値であれば好きな情報を格納できる。||
|**artifact**: dict|実験設定や実験結果を格納する。ユーザは好きな情報を格納できる。||

### **Methods**
|名前|説明||
|:---|:---|:---|
|**store_as_table(record: dict)**|**record**に記述されたデータを**table**に挿入する。**record**はdict型で書かなければならず、キーはtableの列名に使われ、値は対応するセルに代入される。この時、行方向の指定には**run_id**が使われる。もし**record**のvalueが**dimod.SampleSet**もしくは**DecodedSamples**型の場合、エネルギー値などの取得可能な量を自動で取得し、対応するキーを生成してtableへ格納する。**autosave**がTrueの場合、このメソッドを呼び出すたびに逐次csvファイルへ追記される。||
|**store_as_artifact(artifact: dict)**|**record**に記述されたデータを**artifact**に挿入する。**artifact**はdict型で書かなければならない。挿入する際には、artfifactの辞書に`{[現在のrun_id]: [artifactデータ]}`のkeyとvalueが追加される。**autosave**がTrueの場合、このメソッドを呼び出すたびに逐次pickleファイルへ追記される。||
|**store(results: dict, table_keys: List[str], artifact_keys: List[str])**|**store_as_table**と**store_as_artifact**を同時に実行できるメソッドであり、**table_keys**に指定されたキーに対応する量は**store_as_table**で処理され、**artifact_keys**に指定されたキーに対応する量は**store_as_artifact**で処理される。||
|**save(save_file: str)**|現在の**table**をcsvで保存し、**artifact**をpickleファイルで保存する。`autosave=True`の時はこのメソッドを明示的に呼び出す必要はない。
|**load(experiment_id: str, benchmark_id: str, autosave: bool)**|指定した**experiment_id**, **benchmark_id**に対応する保存した結果をファイルから読み込み、**table, artifact**に代入する。||

### **Examples**
最も単純な使い方
```python
# Example 1
# ユーザ定義のsolverの返り値（何でも良い）
sample_response = {"hoge": {"fuga": 1}}

import jijbench as jb

experiment = jb.Experiment()

for param in [10, 100, 1000]:
    for step in range(3):
        with experiment.start():
            # solverは上のsample_responseを返す想定
            # sample_response = solver()
            # experiment.tableに登録するrecordを辞書型で作成
            record = {
                "step": step,
                "param": param,
                "results": sample_response,
            }
            experiment.store(record) # recordがtable, artifactどちらにも保存される

            #experiment.store(record, table_keys=["step", "param"], artifact_keys=["results"]) # step, paramはtableに、resultsはartifactに保存される。
            # 下のように分割して書いても良い
            #experiment.store_as_table({"step": step, "param": param}) 
            #experiment.store_as_artifact({"results": sample_response})  

```
実験結果を保存したい場所を指定する。experiment_idとbenchmark_idを明示的に指定すると結果の保存と読み込みの対応関係がつけやすくなる。
```python
# Example 2
save_dir = "/home/azureuser/data/jijbench"
experiment_id = "test"
benchmark_id = 0

experiment = jb.Experiment(experiment_id=experiment_id, benchmark_id=benchmark_id, save_dir=save_dir)

for param in [10, 100, 1000]:
    for step in range(3):
        with experiment.start():
            # sample_response = solver()
            record = {
                "step": step,
                "param": param,
                "results": sample_response,
            }
            experiment.store(record)

# 以前実験した結果を読み込む。experiment_idとbenchmark_idを覚えていれば対応する実験を読み込める。
# もちろんファイル名を直接指定しても良い。その場合はautosave=Falseにしてloadの引数でファイル名を指定する。
experiment = Experiment.load(experiment_id=experiment_id, benchmark_id=benchmark_id, autosave_dir=save_dir)
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
