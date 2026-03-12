# fifty-nlp

- A byte‐level file fragment type classifier based on FiFTy, enhanced with NLP models
- FiFTyをベースに、NLPモデルで強化したバイト単位ファイル断片分類器

## Usage

`graphviz` をインストールする。

モデルの種類は `config/config.yml` の `type` で `cnn` / `lstm` / `gru` / `transformer`
を指定して切り替えられます。

Arch Linux の場合:

```
sudo pacman -S graphviz
```

Ubuntu の場合:

```
sudo apt update
sudo apt install graphviz
```

実行する。

```
python3 -m venv venv
source venv/bin/activate.fish
pip install -r requirements.txt
echo '' > nohup.out && nohup sh -c 'python3 main.py; cn "main.py"; git pull && git add
result/ && git commit -m "add: ログ追加" && git push' &;
deactivate
```

### 結果集計と学習曲線の出力

最新実行結果を集計して `summary_results.csv`、学習曲線の PNG、
および混同行列サマリ `summary_confusion_matrix.png` を出力する。

```
python src/utils/summarize_results.py
```

白黒印刷向けに `summary_learning_curve_accuracy_mono.png` も出力したい場合は
`--mono-accuracy` を付ける。

```
python src/utils/summarize_results.py --mono-accuracy
```

`summary_learning_curve_accuracy_mono.png` では各線のマーカーを「全体で約12個になる間隔」
で間引いています（`markevery = max(1, int(len(x_values) / 12))`）。
モデルはマーカー形状に加えて 4 段階のグレースケール濃度でも識別できます。

`summary_confusion_matrix.png` は、各モデルの「最新 run」に保存された
`confusion_matrix.npy`（生カウント）を 1 枚に並べた図です。

- 縦軸: 真のクラス
- 横軸: 予測クラス
- カラーバー `Count`: 件数（サンプル数）
- 対角成分: 正分類件数
- 非対角成分: 誤分類件数（どのクラス同士を取り違えたか）

「全て間違わない場合」の理論上の見え方は、非対角成分がすべて 0 で、
対角成分 `(i, i)` が各クラスのサンプル数 `N_i` になります。
このとき 1 セルの理論上の最大値は `max_i N_i`
（評価データ中で最も件数の多いクラスのサンプル数）です。

論文・レポート向けキャプション例:
「各モデル（CNN, LSTM, GRU, Transformer）の最新実行結果に対する混同行列。
縦軸は真のクラス，横軸は予測クラスを示し，色の濃さ（Count）はサンプル件数を表す。
対角成分が大きいほど正分類が多く，非対角成分はクラス間の誤分類傾向を示す。」

## AMP (mixed precision)

本実装では AMP を使用せず、学習/評価ともに fp32 固定で実行します。

- CUDA AMP を有効にすると、環境によって不安定な挙動（例: CUDA error）が発生するため
- FiFTy の論文では AMP を使用していないため、再現性の観点で合わせるため

## Config

設定ファイルは `config/config.yml`。ここでは検証の評価母集団を「割合」か
「全量」かで切り替えられるようになっています。

### 検証フェーズと評価範囲の関係

- `training.common.phase`
  - `search`: ハイパーパラ探索を想定、 `val_subset_ratio` 未指定なら 0.4 を適用
  - `final`: 最終学習を想定、 `val_subset_ratio` / `val_max_batches` を null にして全量評価
  - 内部的には `search` の場合に `val_subset_ratio` が未指定なら 0.4 に自動設定されるだけの違い
- `training.common.val_subset_ratio`
  - `0.4` のように指定すると、検証データを割合でサブセット評価
  - `null` なら全量評価
- `training.common.val_max_batches`
  - 指定すると最大バッチ数で評価を打ち切り
  - 再現性重視のときは `null` 推奨
- `training.common.val_subset_seed`
  - サブセット抽出時の乱数シード、未指定の場合は `experiment.seed` を使います

ログには `val_samples_used/total` と `val_subset_ratio` が出力されます。

### 基本的な config 例（final 想定）

```yaml
training:
  common:
    phase: "final"
    val_subset_ratio: null
    val_max_batches: null
    val_subset_seed: 42
```

### 探索時の config 例（search 想定）

```yaml
training:
  common:
    phase: "search"
    val_subset_ratio: 0.4
    val_max_batches: null
    val_subset_seed: 42
```

### Early Stopping の挙動

`training.common.early_stopping` で監視指標と判定方向を設定します。

- `metric`: `val_loss` / `val_acc` / `val_macro_f1`
- `mode`:
  - `min` のときは「小さいほど良い」（例: `val_loss`）
  - `max` のときは「大きいほど良い」（例: `val_acc`）
- `min_delta`: 改善とみなす最小差分
- `patience`: 改善が無いエポックが連続でこの回数になったら停止

改善判定は次の条件で行われます。

- `mode: "min"` → `metric_value < best - min_delta`
- `mode: "max"` → `metric_value > best + min_delta`

## 出力ファイル

学習を開始すると `config.yml` の `experiment.result_dir` 以下に
`YYYY-MM-DD_HH-MM-SS_<tag>/` という形式のランディレクトリが作成されます。
そこには次の主要ファイルが保存されます。

- `log.txt`: ロガー出力をまとめた実行ログ
- `learning_curve.csv`: 各エポックの学習損失・検証損失・精度・学習率の履歴
- `checkpoint_latest.pt`: 直近エポックのモデル・オプティマイザ状態。`resume_from` に指定して再開する
- `checkpoint_epoch_XXX.pt`: エポックごとに保存されるスナップショット。`XXX` は 001 からの通番
- `torchview.svg`: `torchview` で描画したモデル構造の SVG 図
- `torchinfo.txt`: `torchinfo.summary` の実行結果、各レイヤーの入出力形状とパラメータ数を記録
- `label_map.json`: クラス ID とラベル名の対応表と、元データの参照情報
- `confusion_matrix.npy`: テスト評価時の混同行列（`int64` の `N x N`）
- `per_class_metrics.csv`: 混同行列から算出したクラス別指標（support/TP/FP/FN/precision/recall/F1）
- `per_class_f1_bottom10.csv`: F1 が低い順の下位 10 クラスの一覧（`bottom_k` 指定で変動）
- `top_confusion_pairs.csv`: 誤分類が多い上位ペア（真ラベル×予測ラベル、`top_k` 指定で変動）

### 3つのCSVの違い

- `per_class_metrics.csv`: 混同行列に基づいて全クラスを対象に算出したクラス別評価指標の完全表であり、各クラスの支持度（support）と誤りの内訳（TP/FP/FN）に加えて、precision・recall・F1 を一貫した定義で報告する。主な列は `label_id,label_name,support,tp,fp,fn,precision,recall,f1`
- `per_class_f1_bottom10.csv`: `per_class_metrics.csv` を基底とし、F1 スコアの昇順に下位 `bottom_k` クラスのみを抽出した診断用の要約表である。性能劣化が顕著なクラス群を迅速に同定することを目的としており、順位付けを明示する `rank` 列を付与する。行数は基本的に「ヘッダ + `bottom_k` 件」
- `top_confusion_pairs.csv`: 誤分類の構造を「真ラベル → 予測ラベル」の有向ペアとして集計し、頻度の高い取り違え上位 `top_k` 件を提示する分析表である。単一クラスの良否では捉えにくい系統的混同（systematic confusion）を可視化することを目的とする。主な列は `rank,true_id,true_name,pred_id,pred_name,count,rate_in_true`

各 CSV のヘッダーの意味は以下の通りです。

### per_class_metrics.csv

- `label_id`: クラス ID
- `label_name`: クラス名
- `support`: 真のラベルがそのクラスだったサンプル数
- `tp`: True Positive（そのクラスを正しく予測）
- `fp`: False Positive（他クラスをそのクラスと誤予測）
- `fn`: False Negative（そのクラスを他クラスと誤予測）
- `precision`: `tp / (tp + fp)`（分母 0 の場合は 0）
- `recall`: `tp / (tp + fn)`（分母 0 の場合は 0）
- `f1`: `2 * precision * recall / (precision + recall)`（分母 0 の場合は 0）

### per_class_f1_bottom10.csv

- `rank`: F1 の低い順の順位（1 が最下位）
- `label_id`: クラス ID
- `label_name`: クラス名
- `support`: 真のラベルがそのクラスだったサンプル数
- `tp`: True Positive（そのクラスを正しく予測）
- `fp`: False Positive（他クラスをそのクラスと誤予測）
- `fn`: False Negative（そのクラスを他クラスと誤予測）
- `precision`: `tp / (tp + fp)`（分母 0 の場合は 0）
- `recall`: `tp / (tp + fn)`（分母 0 の場合は 0）
- `f1`: `2 * precision * recall / (precision + recall)`（分母 0 の場合は 0）

### top_confusion_pairs.csv

- `rank`: 誤分類数の多い順の順位
- `true_id`: 正解ラベルのクラス ID
- `true_name`: 正解ラベルのクラス名
- `pred_id`: 予測ラベルのクラス ID
- `pred_name`: 予測ラベルのクラス名
- `count`: `true_id -> pred_id` の誤分類件数
- `rate_in_true`: `count / support(true_id)`（その正解クラス内での誤分類率）

## 評価指標（論文向けメモ）

ログや `summary_results.csv` に出力される指標の定義を明文化しておく。

- 混同行列（論文向け定義）
  - クラス集合を \(\mathcal{Y}=\{0,1,\dots,74\}\)（全 75 クラス）、系列長を \(T=512\)、
    バッチサイズ上限を \(B\) とする。テスト集合をミニバッチに分割し、
    \[
    \mathcal{D}_{\mathrm{test}}
    =
    \bigsqcup_{m=1}^{M}\{(x*{m,b},y*{m,b})\}_{b=1}^{B_m},
    \quad
    x_{m,b}\in\{0,1,\dots,255\}^{T},\ y*{m,b}\in\mathcal{Y},\ B_m\le B
    \]
    と表す（最終バッチでは \(B_m<B\) となり得る）。ここで、\(m\in\{1,\dots,M\}\) は
    バッチ番号、\(M\) は評価で実際に処理したバッチ総数、\(b\in\{1,\dots,B_m\}\) は
    バッチ内インデックス、\(B_m\) はバッチ \(m\) の実サンプル数を表す。また
    \(x*{m,b}\) はバッチ \(m\) の \(b\) 番目サンプルの入力系列、\(y\_{m,b}\) はその
    真のクラスラベルである。
  - モデル（パラメータ \(\theta\)）は各サンプルに対してロジット
    \(z*{m,b}=f*\theta(x*{m,b})\in\mathbb{R}^{75}\) を出力し、
    予測ラベル \(\hat y*{m,b}\) を
    \[
    \hat{y}_{m,b}=\operatorname\*{arg\,max}_{k\in\mathcal{Y}} z\_{m,b,k}
    \]
    により定義する（実装は `argmax(logits, dim=1)`）。
  - 混同行列を \(C\in\mathbb{N}^{75\times 75}\) とし、その \((i,j)\) 要素 \(C*{ij}\) を
    「真のラベルが \(i\)、予測ラベルが \(j\) となった件数」として
    \[
    C*{ij}
    =
    \sum*{m=1}^{M}\sum*{b=1}^{B*m}
    \mathbf{1}[y*{m,b}=i]\,
    \mathbf{1}[\hat{y}_{m,b}=j],
    \quad (i,j\in\mathcal{Y})
    \]
    で定義する。ここで \(\mathbf{1}[\cdot]\) は指示関数（条件が真なら 1、偽なら 0）
    である。行 \(i\) が真のクラス、列 \(j\) が予測クラスを表す。なお、本実装で保存
    される混同行列は正規化前の生カウントであり、さらに
    \[
    \sum*{i\in\mathcal{Y}}\sum*{j\in\mathcal{Y}} C*{ij}
    =
    \sum*{m=1}^{M} B_m
    \]
    を満たすことを検証している（`src/training/training_loop.py:950`,
    `src/training/training_loop.py:967`, `src/training/training_loop.py:1022`）。
  - クラス \(k\in\mathcal{Y}\) に対する TP/FP/FN は、混同行列 \(C\) を用いて
    \[
    \mathrm{TP}_k=C_{kk},\quad
    \mathrm{FP}_k=\sum_{i\in\mathcal{Y},\,i\ne k}C*{ik},\quad
    \mathrm{FN}\_k=\sum*{j\in\mathcal{Y},\,j\ne k}C\_{kj}
    \]
    と定義する。
  - これらから、クラス \(k\) の Precision / Recall / F1 を
    \[
    \mathrm{Precision}\_k=\frac{\mathrm{TP}\_k}{\mathrm{TP}\_k+\mathrm{FP}\_k},\quad
    \mathrm{Recall}\_k=\frac{\mathrm{TP}\_k}{\mathrm{TP}\_k+\mathrm{FN}\_k}
    \]
    \[
    \mathrm{F1}\_k=
    \frac{2\,\mathrm{Precision}\_k\,\mathrm{Recall}\_k}
    {\mathrm{Precision}\_k+\mathrm{Recall}\_k}
    \]
    で定義する（分母が 0 の場合は 0 とする実装である：
    `src/utils/confusion_reports.py:73`）。

- Accuracy (Top-1): `accuracy = correct_top1 / total`
  - `correct_top1`: Top-1 予測が正解だったサンプル数
  - `total`: 評価対象サンプル数
- Top-3 Accuracy: `top3_accuracy = correct_top3 / total`
  - `correct_top3`: 正解ラベルが Top-3 予測に含まれたサンプル数
  - `total`: 評価対象サンプル数
- Loss: `loss = mean(cross_entropy(logits, targets))`
  - `logits`: 各クラスの未正規化スコア
  - `targets`: 正解ラベル
- Macro F1: `macro_f1 = mean_c(F1_c)`, `F1_c = (2*TP_c) / (2*TP_c + FP_c + FN_c)`
  - `TP_c`: クラス c の True Positive 数
  - `FP_c`: クラス c の False Positive 数
  - `FN_c`: クラス c の False Negative 数
  - 分母が 0 の場合はそのクラスの `F1_c` を 0 とし、全クラス平均を取る
- JPEG Acc: `jpeg_acc = jpeg_correct / jpeg_support`
  - `jpeg_correct`: JPEG クラスのサンプルで Top-1 予測が正解だった数
  - `jpeg_support`: JPEG クラスのサンプル数
  - JPEG クラスは `classes_Human-readable_labels.json` のうち、学習・評価に使う
    クラス数 `n_classes` と一致するシナリオから、名前を正規化した結果が
    `jpg`/`jpeg` のクラス index を採用する
  - `jpeg_support = 0` の場合は評価をエラーとする
  - Scenario #2 は JPEG を単独クラスとして分離せず
    「Bitmaps / RAW / Video ...」のような用途グルーピングで定義されるため、
    JPEG クラス限定の正解率を定義できず `jpeg_acc` は `not_available` とする

さらに `experiment.interpretability.enable` を `true` にすると
`interpretability/epoch_XXX/` 以下に解析用の追加ファイルが出力されます。

- `summary.json`: 解析対象サンプルのラベル・予測・確信度を一覧化したメタ情報
- `sample_XXXXX/metadata.json`: 各サンプルの詳細メタデータ（ラベルや予測確率など）
- `sample_XXXXX/*.npy`, `*.npz`: CNN/LSTM/GRU の中間表現やゲート動作を保存したテンソル

### パフォーマンスの警告 `W1015`

無視して良いが、もし性能低下が気になる場合は次を行うと改善される場合がある。

```
pip install --upgrade torch torchvision triton
```

## Useage: Google Colab + Google Drive

1. Drive をマウント
   - Colab セルで次を実行
     ```
     from google.colab import drive;
     drive.mount("/content/drive")
     ```
   - OAuth で自身の Drive を接続
2. リポジトリを配置し、依存関係をインストール

```
! git clone https://github.com/rayfiyo/fifty-nlp.git /content/fifty-nlp && \
 cd /content/fifty-nlp && \
 pip install -r requirements.txt
```

3. 設定ファイルを調整
   - `config.yml` の `experiment.result_dir` や `data.base_dir` を
     Drive 上の保存先に書き換える
     - 例: `/content/drive/MyDrive/fifty-nlp/...`
     - マウントした Google Drive のルートディレクトリは `/content/drive/MyDrive/` である
   - `experiment.device` を `gpu` にすると GPU を利用（CPU に戻すときは `cpu`）
   - 学習再開時は `experiment.resume_from` に `<timestamp>_<tag>` または
     `<timestamp>_<tag>/checkpoint_latest.pt` を指定
     - 相対パスで指定すると、`result_root` からの相対パスになる
     - エポック終了毎に保存される
4. GPU の有効化
   - ランタイム > ランタイムのタイプを変更 > ハードウェア アクセラレータ > GPU
5. 実行する

```
!cd /content/fifty-nlp && python main.py
```

### Tips

- 環境変数を使った一時的な上書き: Colab セルで
  `os.environ["FIFTY_DATA_BASE_DIR"] = "/content/drive/..."` のように指定
- 実行: `!python main.py` などで学習を開始すると、結果・データが Drive 側に保存されます。
- 途中再開: 各エポック終了時に `checkpoint_latest.pt` が更新され、
  同ディレクトリ配下に `checkpoint_epoch_XXX.pt` も保存されます。
  次回は `config.yml` の `experiment.resume_from` に該当パスを設定して再開できます。

## LR の探索

- src/utils/lr_finder.py
- 指数スイープでLRを上げながら num_iters バッチだけ学習し、損失と指数移動平均を記録。損失が悪化し過ぎたら早期終了

```
usage: lr_search.py [-h] [-l LEARNING_RATES [LEARNING_RATES ...] | --logspace START END NUM] [--run-types RUN_TYPES [RUN_TYPES ...]] [--epochs EPOCHS] [--batch-size BATCH_SIZE]
                    [--n-subset N_SUBSET] [--warmup-epochs WARMUP_EPOCHS] [--val-interval VAL_INTERVAL] [--device DEVICE] [--seed SEED] [--tag TAG] [--compile-model]

FiFTy の学習率を探索するスクリプト

options:
  -h, --help            show this help message and exit
  -l LEARNING_RATES [LEARNING_RATES ...], --learning-rates LEARNING_RATES [LEARNING_RATES ...]
                        明示的に試す学習率を指定（空白区切り）
  --logspace START END NUM
                        幾何間隔で START→END まで NUM 個生成（例: 1e-4 1e-2 5）
  --run-types RUN_TYPES [RUN_TYPES ...]
                        対象の run_type を指定（未指定なら config.yml の type を使用）
  --epochs EPOCHS       各 LR で回すエポック数（未指定なら config.yml の値）
  --batch-size BATCH_SIZE
                        バッチサイズを上書き（未指定なら config.yml の値）
  --n-subset N_SUBSET   学習データをこの件数にサブセット（未指定なら config.yml の値）
  --warmup-epochs WARMUP_EPOCHS
                        ウォームアップエポック数を上書き
  --val-interval VAL_INTERVAL
                        検証を実施するエポック間隔を上書き
  --device DEVICE       利用デバイスを指定（cpu / cuda / auto / gpu）
  --seed SEED           乱数シード（未指定なら config.yml の experiment.seed）
  --tag TAG             結果ディレクトリ名に付与するタグ
  --compile-model       torch.compile を有効化（短時間検証ではデフォルト無効）
```

### 例

早期終了を無効化した状態で、
`1e-6` から `1e-1` までを Log scale で 12 等分した LR の値を試行したい場合。

```bash
venv/bin/python3 src/utils/lr_search.py --no-early-stopping --epochs 12 --logspace 1e-6 1e-1 12
```

```bash
venv/bin/python3 src/utils/lr_search.py --no-early-stopping --epochs 12 -l 1e-06 2.848035868435799e-06 8.111308307896873e-06 2.310129700083158e-05 6.579332246575683e-05 0.0001873817422860383 0.0005336699231206307 0.0015199110829529332 0.004328761281083057 0.012328467394420659 0.03511191734215127 0.1
```

早期終了を無効化した状態で、
`1e-6 3e-6 1e-5 3e-5 1e-4 3e-4 1e-3 3e-3 1e-2 3e-2 1e-1 3e-1`
のそれぞれで LR を試行したい場合。

```bash
venv/bin/python3 src/utils/lr_search.py --no-early-stopping --epochs 12 -l 1e-6 3e-6 1e-5 3e-5 1e-4 3e-4 1e-3 3e-3 1e-2 3e-2 1e-1 3e-1
```

### LR探索サマリのグラフ化（LR vs Accuracy）

LR探索の結果ディレクトリ、または `lr_search_summary.csv` を引数に与えると、
同じディレクトリに `lr_search_acc_vs_lr.png` を生成します。
また、`src/utils/lr_search.py` の探索完了時にも自動で生成されます。
さらに探索完了後に `cn` などの post-run コマンドも実行されます。

- 主系列（実線）: `val_acc_at_end`
- 補助系列（破線）: `best_val_acc`（列がある場合）
- 横軸: LR（logスケール）

```bash
python src/utils/plot_lr_search.py result/CNN/lr_search/2026-01-27_17-24-55_lr_search/
```

主系列を変更したい場合:

```bash
python src/utils/plot_lr_search.py result/... --acc-col best_val_acc
```

# Special thanks

## FiFTy

- https://arxiv.org/abs/1908.06148v2
- https://arxiv.org/pdf/1908.06148v2
- https://www.alphaxiv.org/overview/1908.06148v2
- https://github.com/mittalgovind/fifty
