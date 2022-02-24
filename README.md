# ParameterSearch

# 仕様
### class `Experiment`
|**Parameters**|説明||
|:---|:---|:---|
|**run_id**: int|solverの実行を区別するID||
|**experiment_id**: int or str |Experimentインスタンスを区別するID||
|**benchmark_id**: int or str|Benchmarkインスタンスを区別するID||
|**autosave**: bool|実験結果を自動で保存したい場合はTrue,そうでない場合はFalse||
|**autosave_dir**: str|`autosave=True`の時の実験結果の保存先||

|**Attribute**|説明||
|:---|:---|:---|
|**table**: pandas.DataFrame||実験設定や実験結果を格納する。ユーザは好きな情報を格納できる。||

|**Method**|説明||
|:---|:---|:---|
|**insert_into_table(record: dict)**|`table`に`record`に記述されたデータを挿入する。`record`はdict型で書かなければならず、キーはtableの列名に使われ、値は対応するセルに代入される。この時、行方向の指定には`run_id`が使われる。このメソッドを呼ぶと最後に__next__メソッドが呼ばれ`run_id`を一つ進める。||
|**save(save_file: str)**|`table`をcsvで実験結果を保存する。`autosave=True`の時は`autosave_dir`で指定されたディレクトリ以下に`benchmark_{bechmark_id}/tables`というディレクトリが自動で作成され、そのディレクトリ以下に`experiment_id_{experiment_id}.csv`というファイル名で`table`が保存される。`autosave=False`の場合、`save_file`で指定されるファイル名で`table`が保存される。
|**load(load_file: str)**|saveメソッドで保存した結果を読み込み、`table`に代入する。`autosave=True`場合、 `autosave_dir`以下の`experiment_id`、`benchmark_id`で指定される結果を自動で読み込み、`autosave=False`の場合、`load_file`で指定されるファイルを読み込む。||
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