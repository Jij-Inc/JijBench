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

`ParameterSearch`ディレクトリに入って,

```
python -m parameter_test
```

で実行してください.

# パラメータのアップデート

