"""
Playground Series S5E6 予測のためのStreamlitウェブアプリケーション
"""

import streamlit as st
import pandas as pd
import numpy as np
import sys
import os
import plotly.express as px
import plotly.graph_objects as go

# Add project root to path
project_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.append(project_root)

from src.models.predict import CompetitionPredictor
import logging

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


# ページ設定
st.set_page_config(
    page_title="Playground Series S5E6 予測システム",
    page_icon="🎯",
    layout="wide",
    initial_sidebar_state="expanded"
)


@st.cache_resource
def load_predictor():
    """訓練済みモデルと前処理器を読み込み"""
    try:
        model_path = os.path.join(project_root, "models", "best_model.pkl")
        preprocessor_path = os.path.join(project_root, "models", "preprocessor.pkl")
        
        if not os.path.exists(model_path):
            st.error(f"モデルファイルが見つかりません: {model_path}")
            return None
        
        if not os.path.exists(preprocessor_path):
            st.error(f"前処理器ファイルが見つかりません: {preprocessor_path}")
            return None
        
        predictor = CompetitionPredictor(model_path, preprocessor_path)
        return predictor
    
    except Exception as e:
        st.error(f"モデル読み込みエラー: {e}")
        return None


@st.cache_data
def load_sample_data():
    """特徴量の範囲を理解するためのサンプルデータを読み込み"""
    try:
        train_path = os.path.join(project_root, "data", "train.csv")
        if os.path.exists(train_path):
            return pd.read_csv(train_path)
        else:
            return None
    except Exception as e:
        logger.warning(f"サンプルデータを読み込めませんでした: {e}")
        return None


def create_feature_inputs(predictor, sample_data):
    """全ての特徴量の入力ウィジェットを作成"""
    feature_values = {}
    
    if predictor is None:
        st.error("モデルが読み込まれていません。モデルファイルを確認してください。")
        return {}
    
    # 前処理器から特徴量情報を取得
    numeric_features = predictor.preprocessor.numeric_features
    categorical_features = predictor.preprocessor.categorical_features
    
    st.sidebar.header("📊 特徴量入力")
    
    # 数値特徴量
    if numeric_features:
        st.sidebar.subheader("数値特徴量")
        
        for feature in numeric_features:
            # サンプルデータから特徴量統計を取得
            if sample_data is not None and feature in sample_data.columns:
                feature_stats = sample_data[feature].describe()
                min_val = float(feature_stats['min'])
                max_val = float(feature_stats['max'])
                mean_val = float(feature_stats['mean'])
                std_val = float(feature_stats['std'])
                
                # 適切な範囲で入力を作成
                feature_values[feature] = st.sidebar.number_input(
                    f"{feature}",
                    min_value=min_val - 2*std_val,
                    max_value=max_val + 2*std_val,
                    value=mean_val,
                    step=(max_val - min_val) / 100,
                    help=f"データ範囲: [{min_val:.2f}, {max_val:.2f}], 平均: {mean_val:.2f}"
                )
            else:
                # サンプルデータなしの場合のデフォルト入力
                feature_values[feature] = st.sidebar.number_input(
                    f"{feature}",
                    value=0.0,
                    help="数値を入力してください"
                )
    
    # カテゴリカル特徴量
    if categorical_features:
        st.sidebar.subheader("カテゴリカル特徴量")
        
        for feature in categorical_features:
            if sample_data is not None and feature in sample_data.columns:
                # サンプルデータから一意の値を取得
                unique_values = sorted(sample_data[feature].dropna().unique())
                
                if len(unique_values) <= 20:  # カテゴリ数が少ない場合はセレクトボックスを使用
                    feature_values[feature] = st.sidebar.selectbox(
                        f"{feature}",
                        options=unique_values,
                        help=f"利用可能な選択肢: {unique_values}"
                    )
                else:  # カテゴリ数が多い場合はテキスト入力を使用
                    feature_values[feature] = st.sidebar.text_input(
                        f"{feature}",
                        value=str(unique_values[0]) if unique_values else "",
                        help=f"サンプル値: {unique_values[:5]}..."
                    )
            else:
                # サンプルデータなしの場合のデフォルト入力
                feature_values[feature] = st.sidebar.text_input(
                    f"{feature}",
                    value="",
                    help="カテゴリ値を入力してください"
                )
    
    return feature_values


def main():
    """メインアプリケーション"""
    
    # タイトルと説明
    st.title("🎯 Playground Series S5E6 予測システム")
    st.markdown("""
    このウェブアプリケーションは、Kaggle Playground Series S5E6 コンペティション用の予測を提供します。
    サイドバーで特徴量の値を入力すると、リアルタイムで予測結果を取得できます。
    """)
    
    # モデルとデータを読み込み
    predictor = load_predictor()
    sample_data = load_sample_data()
    
    if predictor is None:
        st.error("⚠️ モデルが利用できません。モデルファイルが存在することを確認してください。")
        st.info("""
        このアプリケーションを使用するには:
        1. 訓練パイプラインを使用してモデルを訓練
        2. モデルファイルが`models/`ディレクトリに保存されていることを確認
        3. アプリケーションを再起動
        """)
        return
    
    # メインレイアウトを作成
    col1, col2 = st.columns([2, 1])
    
    with col1:
        st.header("📈 予測結果")
        
        # 特徴量入力を取得
        feature_values = create_feature_inputs(predictor, sample_data)
        
        if feature_values:
            try:
                # 予測を実行
                prediction = predictor.predict_single_sample(feature_values)
                
                # 予測を表示
                st.metric(
                    label="予測確率",
                    value=f"{prediction:.4f}",
                    help="正例クラス（target=1）の確率"
                )
                
                # 予測の解釈
                if prediction > 0.7:
                    st.success("🟢 正例クラスの高い確率")
                elif prediction > 0.3:
                    st.warning("🟡 中程度の確率")
                else:
                    st.info("🔵 正例クラスの低い確率")
                
                # 予測ゲージ
                fig = go.Figure(go.Indicator(
                    mode = "gauge+number+delta",
                    value = prediction,
                    domain = {'x': [0, 1], 'y': [0, 1]},
                    title = {'text': "予測スコア"},
                    delta = {'reference': 0.5},
                    gauge = {'axis': {'range': [None, 1]},
                             'bar': {'color': "darkblue"},
                             'steps' : [
                                 {'range': [0, 0.3], 'color': "lightgray"},
                                 {'range': [0.3, 0.7], 'color': "gray"},
                                 {'range': [0.7, 1], 'color': "lightgreen"}],
                             'threshold' : {'line': {'color': "red", 'width': 4},
                                          'thickness': 0.75, 'value': 0.5}}))
                
                st.plotly_chart(fig, use_container_width=True)
                
            except Exception as e:
                st.error(f"予測エラー: {e}")
    
    with col2:
        st.header("ℹ️ 情報")
        
        # モデル情報
        st.subheader("モデル詳細")
        st.info(f"""
        **特徴量数**: {len(predictor.preprocessor.numeric_features) + len(predictor.preprocessor.categorical_features)}
        - 数値: {len(predictor.preprocessor.numeric_features)}
        - カテゴリカル: {len(predictor.preprocessor.categorical_features)}
        
        **ターゲット**: 二値分類 (0/1)
        **評価指標**: ROC AUC
        """)
        
        # 特徴量重要度（利用可能な場合）
        if sample_data is not None:
            st.subheader("データセット概要")
            st.info(f"""
            **訓練サンプル数**: {len(sample_data):,}
            **ターゲット分布**:
            - クラス 0: {(sample_data['target'] == 0).sum():,} ({(sample_data['target'] == 0).mean()*100:.1f}%)
            - クラス 1: {(sample_data['target'] == 1).sum():,} ({(sample_data['target'] == 1).mean()*100:.1f}%)
            """)
        
        # 使用方法
        st.subheader("使用方法")
        st.markdown("""
        1. 📝 サイドバーで特徴量の値を入力
        2. 🔄 予測は自動的に更新されます
        3. 📊 予測確率とゲージを確認
        4. 🎯 1.0に近い値ほど正例クラスの可能性が高いことを示します
        """)


def batch_prediction_page():
    """CSVファイルからのバッチ予測用ページ"""
    st.title("📊 バッチ予測")
    st.markdown("CSVファイルをアップロードして、複数のサンプルの予測を取得できます。")
    
    predictor = load_predictor()
    
    if predictor is None:
        st.error("モデルが利用できません。")
        return
    
    # ファイルアップロード
    uploaded_file = st.file_uploader(
        "CSVファイルを選択",
        type="csv",
        help="訓練データと同じ特徴量を含むCSVファイルをアップロードしてください"
    )
    
    if uploaded_file is not None:
        try:
            # ファイル読み込み
            df = pd.read_csv(uploaded_file)
            st.write("**アップロードされたデータのプレビュー:**")
            st.dataframe(df.head())
            
            # 予測を実行
            if st.button("予測を生成"):
                with st.spinner("予測を生成中..."):
                    predictions = []
                    
                    for idx, row in df.iterrows():
                        pred = predictor.predict_single_sample(row.to_dict())
                        predictions.append(pred)
                    
                    # データフレームに予測を追加
                    result_df = df.copy()
                    result_df['prediction'] = predictions
                    result_df['predicted_class'] = (np.array(predictions) > 0.5).astype(int)
                    
                    # 結果を表示
                    st.success("予測が正常に生成されました！")
                    st.dataframe(result_df)
                    
                    # ダウンロードボタン
                    csv = result_df.to_csv(index=False)
                    st.download_button(
                        label="予測結果をCSVでダウンロード",
                        data=csv,
                        file_name="predictions.csv",
                        mime="text/csv"
                    )
                    
                    # 予測統計
                    st.subheader("予測サマリー")
                    col1, col2, col3 = st.columns(3)
                    
                    with col1:
                        st.metric("総サンプル数", len(predictions))
                    with col2:
                        st.metric("平均確率", f"{np.mean(predictions):.4f}")
                    with col3:
                        st.metric("予測正例数", f"{sum(np.array(predictions) > 0.5)}")
                    
                    # 予測のヒストグラム
                    fig = px.histogram(
                        x=predictions,
                        bins=20,
                        title="予測確率の分布"
                    )
                    st.plotly_chart(fig, use_container_width=True)
        
        except Exception as e:
            st.error(f"ファイル処理エラー: {e}")


if __name__ == "__main__":
    # ページナビゲーションを作成
    page = st.sidebar.selectbox(
        "ナビゲーション",
        ["単一予測", "バッチ予測"]
    )
    
    if page == "単一予測":
        main()
    elif page == "バッチ予測":
        batch_prediction_page()