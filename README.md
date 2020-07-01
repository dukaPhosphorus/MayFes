# MayFes
五月祭の自動作曲展示

Bach, 嵐，back number風の曲を生成，メロディを入力するとそれぞれの雰囲気のコードを付ける．

#### データセット
- Bach music21に収録されているコラールのデータ352曲を全調に移調しデータを水増しして2503曲に
- 嵐 musescoreに68曲を手打ち，短調と長調で曲調に差があったため長調のみを学習データとした
- backnumber musescoreに28曲を手打ち
- bachは一曲当たり30小節程度，嵐とbacknumberは100小節程度

#### 実行方法
auto.py を実行し，ARASHI, backnumber, Bachのディレクトリ（各自この名前で空のフォルダを生成）にmidiファイルを追加するとその雰囲気に合わせたコードを付けてmusescoreで表示される．

#### サンプル
sample_midiに生成例がある（kerokero, hana, kirakira, Mt.Fujiはかえるのうた，滝廉太郎の花，きらきら星，富士山をメロディとして入力する場合のためのもの）．
〇〇生成例とついているのはメロディも生成した場合．
demo/backnumber_kerokero はbacknumber風のコードを付けたかえるのうた，demo/backnumber_extended はbacknumber風にメロディから生成した曲．

#### ファイル形式
〇〇.msczはmuse scoreの形式で楽譜が出力される，〇〇.midはmidi形式

#### 使用環境
- Keras: 2.1.6
- numpy: 1.16.3
- Musescore2: 2.3.2
- music21: 5.5.0
