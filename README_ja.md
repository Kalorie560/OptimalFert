# 🎯 Playground Series S5E6 - OptimalFert ML パイプライン

Kaggle Playground Series S5E6コンペティション向けの包括的な機械学習ソリューション。高度なモデル訓練、ClearML実験追跡、インタラクティブなWebアプリケーションを提供します。

## 📊 コンペティション概要

**コンペティション**: [Playground Series S5E6](https://www.kaggle.com/competitions/playground-series-s5e6/)
- **タイプ**: 二値分類問題
- **評価指標**: ROC AUC (Receiver Operating Characteristic曲線下面積)
- **目標**: 複数の特徴量から二値ターゲット変数を予測

## 🏗️ プロジェクト構成

```
OptimalFert/
├── 📁 data/                    # データセットファイル
│   ├── train.csv              # 訓練データ
│   ├── test.csv               # テストデータ
│   └── sample_submission.csv  # 提出形式サンプル
├── 📁 src/                    # ソースコード
│   ├── 📁 data/               # データ処理モジュール
│   │   └── preprocessing.py   # データ前処理パイプライン
│   ├── 📁 models/             # 機械学習モデル
│   │   ├── train_model.py     # ClearML連携モデル訓練
│   │   └── predict.py         # 予測・提出ファイル生成
│   ├── 📁 visualization/      # EDA・可視化
│   │   └── eda.py            # 探索的データ分析
│   └── 📁 web_app/           # Webアプリケーション
│       └── streamlit_app.py  # Streamlit予測アプリ
├── 📁 models/                 # 保存済みモデル・前処理器
├── 📁 outputs/                # EDA可視化結果
├── 📁 notebooks/              # Jupyter notebook（オプション）
├── train_pipeline.py          # 完全訓練パイプライン
├── generate_submission.py     # 提出ファイル生成スクリプト
├── requirements.txt           # Python依存関係
├── config.yaml.template       # ClearML設定テンプレート
└── .gitignore                # Git除外ルール
```

## 🚀 クイックスタート

### 1. 環境セットアップ

```bash
# リポジトリをクローン
git clone <repository-url>
cd OptimalFert

# 依存関係をインストール
pip install -r requirements.txt
```

### 2. ClearML設定

```bash
# 設定テンプレートをコピー
cp config.yaml.template config.yaml

# config.yamlにClearMLの認証情報を設定
# APIキーは以下から取得: https://app.clear.ml/settings/workspace-configuration
```

`config.yaml`の例:
```yaml
clearml:
  api_host: "https://api.clear.ml"
  web_host: "https://app.clear.ml"
  files_host: "https://files.clear.ml"
  api_key: "YOUR_API_KEY_HERE"
  api_secret_key: "YOUR_SECRET_KEY_HERE"
```

### 3. データ準備

コンペティションデータファイルを`data/`ディレクトリに配置:
- `train.csv` - 訓練用データセット
- `test.csv` - テストデータセット  
- `sample_submission.csv` - 提出形式

または、テスト用のサンプルデータ生成機能を使用:
```python
from src.data.preprocessing import create_sample_data
create_sample_data(n_samples=2000, n_features=15, save_path="data/")
```

## 🔄 訓練パイプライン

### 完全自動パイプライン

ワンコマンドで全ML パイプラインを実行:

```bash
python train_pipeline.py
```

実行内容:
1. **データ読み込み・検証**
2. **探索的データ分析** (`outputs/`に可視化を保存)
3. **データ前処理** (特徴量エンジニアリング、エンコーディング、スケーリング)
4. **モデル訓練** (5種類のアルゴリズムと交差検証)
5. **ハイパーパラメータ最適化** (Optuna使用)
6. **アンサンブル作成** (最良モデルの組み合わせ)
7. **提出ファイル生成** (`submission.csv`)

### 個別コンポーネント実行

各コンポーネントを個別に実行することも可能:

```bash
# 提出ファイルのみ生成（訓練後）
python generate_submission.py --model models/best_model.pkl --test_data data/test.csv

# カスタムパラメータ指定
python generate_submission.py \
  --model models/best_model.pkl \
  --preprocessor models/preprocessor.pkl \
  --test_data data/test.csv \
  --output my_submission.csv \
  --validate
```

## 🧠 機械学習パイプライン

### データ前処理
- **自動特徴量タイプ検出**: 数値・カテゴリ特徴量の自動識別
- **欠損値処理**: 数値は中央値、カテゴリは最頻値で補完
- **特徴量エンコーディング**: カテゴリ特徴量のOne-hotエンコーディング
- **スケーリング**: 数値特徴量の標準化
- **パイプライン永続化**: テストデータでの再利用可能な前処理

### モデルアーキテクチャ
- **LightGBM**: ハイパーパラメータ最適化付き勾配ブースティング
- **XGBoost**: 極勾配ブースティング
- **CatBoost**: カテゴリブースティングアルゴリズム
- **Random Forest**: 決定木のアンサンブル
- **Logistic Regression**: 線形ベースラインモデル
- **アンサンブル**: 最高性能モデルの重み付き平均

### 最適化戦略
- **交差検証**: 堅牢な評価のための5-fold層化交差検証
- **ハイパーパラメータチューニング**: Optunaベースの最適化（50+試行）
- **メトリック重視**: 全体を通じたROC AUC最大化
- **モデル選択**: 最良個別モデル vs アンサンブルの比較

### 実験追跡（ClearML）
- **ハイパーパラメータ**: 全モデル設定の記録
- **メトリクス**: 交差検証スコア、ROC AUC追跡
- **アーティファクト**: モデル、前処理器、可視化結果
- **再現性**: ランダムシード管理とバージョニング

## 🌐 Webアプリケーション

インタラクティブ予測インターフェースの起動:

```bash
streamlit run src/web_app/streamlit_app.py
```

### 機能:
- **単一予測**: 手動で特徴量を入力してリアルタイム予測
- **バッチ予測**: CSVファイルアップロードによる複数予測
- **インタラクティブUI**: バリデーション付きユーザーフレンドリーな特徴量入力
- **可視化**: 予測ゲージと確率分布表示
- **エクスポート機能**: バッチ予測結果のダウンロード

### 使用方法:
1. Webインターフェースにアクセス（通常 `http://localhost:8501`）
2. サイドバーで特徴量の値を入力
3. リアルタイムで予測確率を表示
4. バッチモードで複数予測を実行

## 📈 探索的データ分析

EDAモジュールが自動生成する内容:

### 統計分析
- **データセット概要**: 形状、欠損値、メモリ使用量
- **ターゲット分布**: クラスバランス分析
- **特徴量統計**: 全特徴量の記述統計
- **相関分析**: 特徴量間・ターゲットとの相関関係

### 可視化（`outputs/`に保存）
- `target_distribution.png` - ターゲットクラス分布
- `numeric_features_distribution.png` - 特徴量分布
- `correlation_heatmap.png` - 特徴量相関行列
- `feature_importance.png` - 相互情報量スコア

### 特徴量洞察
- **相互情報量**: 特徴量重要度ランキング
- **カテゴリ分析**: カテゴリ別ターゲット率
- **外れ値検出**: 統計的外れ値識別
- **分布分析**: 歪度・尖度メトリクス

## 🎯 モデル性能

### 評価指標
- **主要指標**: ROC AUC（コンペティション指標）
- **交差検証**: 5-fold層化検証
- **アンサンブル**: 上位モデルの重み付き平均
- **ベースライン**: 複数アルゴリズム比較

### 期待性能
- **個別モデル**: ROC AUC 0.75-0.85（データ依存）
- **最適化モデル**: ハイパーパラメータ調整によりROC AUC 0.80-0.90
- **アンサンブル**: 最良個別モデルより通常1-3%改善

## 📝 提出ファイル生成

### 自動プロセス
```bash
python generate_submission.py
```

### 手動プロセス
```python
from src.models.predict import generate_submission

submission = generate_submission(
    model_path="models/best_model.pkl",
    preprocessor_path="models/preprocessor.pkl", 
    test_data_path="data/test.csv",
    output_path="submission.csv"
)
```

### バリデーション機能
- **形式準拠**: `sample_submission.csv`への厳密準拠
- **ID照合**: 正しいテストサンプル順序の保証
- **値範囲**: 確率範囲[0, 1]の検証
- **欠損値**: null予測の確認・防止

## 🔧 設定

### モデルパラメータ
主要ハイパーパラメータ（Optunaで最適化）:
- **LightGBM**: `num_leaves`, `learning_rate`, `feature_fraction`
- **XGBoost**: `max_depth`, `learning_rate`, `subsample`
- **CatBoost**: `depth`, `learning_rate`, `l2_leaf_reg`

### 訓練設定
- **交差検証**: 5-fold（設定可能）
- **ランダムシード**: 42（再現性確保）
- **最適化試行**: モデルあたり50+回
- **アンサンブル**: 上位3モデルの重み付き平均

## 🚨 トラブルシューティング

### よくある問題

**ClearML接続エラー**:
```bash
# config.yamlのAPI認証情報を確認
# clear.mlへのインターネット接続を確認
```

**メモリ問題**:
```bash
# テスト用にデータセットサイズを削減
# 大きなデータセットには特徴量選択を使用
```

**モデル訓練失敗**:
```bash
# データ形式と前処理を確認
# 交差検証に十分なサンプル数があることを確認
```

### デバッグモード
```python
# 詳細ログ有効化
import logging
logging.basicConfig(level=logging.DEBUG)
```

## 🤝 貢献

1. リポジトリをフォーク
2. 機能ブランチを作成（`git checkout -b feature/amazing-feature`）
3. 変更をコミット（`git commit -m 'Add amazing feature'`）
4. ブランチにプッシュ（`git push origin feature/amazing-feature`）
5. プルリクエストを作成

## 📄 ライセンス

このプロジェクトはMITライセンスの下でライセンスされています。詳細は[LICENSE](LICENSE)ファイルをご覧ください。

## 🙏 謝辞

- **Kaggle**: Playground Seriesコンペティションプラットフォームの提供
- **ClearML**: 実験追跡・モデル管理機能
- **Streamlit**: Webアプリケーションフレームワーク
- **Optuna**: ハイパーパラメータ最適化
- **オープンソースコミュニティ**: 素晴らしいMLライブラリ群（scikit-learn、LightGBM、XGBoostなど）

## 📞 サポート

質問・サポートについて:
- 📧 このリポジトリでissueを作成
- 💬 上記トラブルシューティングセクションを確認
- 📖 詳細なコードドキュメントを参照

---

**機械学習を楽しみましょう! 🚀**