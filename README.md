# ParameterSearch

Research Project about parameter search

# 仕様

main のファイルは `parameter_test.py` です. その内部で 問題 (`jm.problem`) と 問題のインスタンスを引っ張ってきて繰り返しときます.

みなさんおのおのが書き込む場所は, `update.py`の`make_initial_multipliers(problem: Problem)` と `parameter_update()` です.

# パラメータのアップデート (`update.py`)

`make_initial_multipliers(problem: Problem)`: 問題を受け取って, パラメータの初期値を設定してください. 今書かれていいるスクリプトは初期値が全て 1 になるようにしています. 適宜修正してください.

`parameter_update(problem: Problem, decoded: DecodedSamples, multipliers: Dict[str, float]))`: 問題・解・現在の multiliers を入力として, 次の multipliers を出力してください. 例えば今書かれているスクリプトでは, 制約を守らない項のパラメータを 5 倍にするようにしています.

# 未対応

- 拡張ラグランジュ関数へのコンパイルをまだ書いていないです.
- 結果の保存・可視化をまだ書けていないです
