"""
肥料名予測のためのStreamlitウェブアプリケーション
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

from src.models.predict import FertilizerPredictor
import logging

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


# ページ設定
st.set_page_config(
    page_title="OptimalFert 肥料推奨システム",
    page_icon="🌱",
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
        
        predictor = FertilizerPredictor(model_path, preprocessor_path)
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
    st.title("🌱 OptimalFert 肥料推奨システム")
    st.markdown("""
    このウェブアプリケーションは、土壌条件、作物情報、環境データに基づいて最適な肥料を推奨します。
    サイドバーで農業条件を入力すると、リアルタイムで肥料推奨結果を取得できます。
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
        st.header("🌱 肥料推奨結果")
        
        # 特徴量入力を取得
        feature_values = create_feature_inputs(predictor, sample_data)
        
        if feature_values:
            try:
                # 予測を実行
                fertilizer_name, probabilities = predictor.predict_single_sample(feature_values)
                
                # 推奨肥料を表示
                st.success(f"🎯 **推奨肥料**: {fertilizer_name}")
                
                # 確信度を表示
                max_prob = max(probabilities.values())
                confidence_text = "高い" if max_prob > 0.6 else "中程度" if max_prob > 0.4 else "低い"
                st.metric(
                    label="確信度",
                    value=f"{max_prob:.1%}",
                    help=f"推奨肥料の予測確信度: {confidence_text}"
                )
                
                # 全肥料タイプの確率分布を表示
                st.subheader("📊 全肥料タイプの確率分布")
                
                # 確率を降順にソート
                sorted_probs = sorted(probabilities.items(), key=lambda x: x[1], reverse=True)
                
                # 横棒グラフで表示
                fertilizers = [item[0] for item in sorted_probs]
                probs = [item[1] for item in sorted_probs]
                
                fig = go.Figure(go.Bar(
                    x=probs,
                    y=fertilizers,
                    orientation='h',
                    marker=dict(
                        color=probs,
                        colorscale='Viridis',
                        showscale=True,
                        colorbar=dict(title="確率")
                    )
                ))
                
                fig.update_layout(
                    title="肥料タイプ別推奨確率",
                    xaxis_title="確率",
                    yaxis_title="肥料タイプ",
                    height=400
                )
                
                st.plotly_chart(fig, use_container_width=True)
                
                # 上位3つの肥料を詳細表示
                st.subheader("🏆 上位推奨肥料")
                for i, (fert, prob) in enumerate(sorted_probs[:3]):
                    emoji = "🥇" if i == 0 else "🥈" if i == 1 else "🥉"
                    st.write(f"{emoji} **{fert}**: {prob:.1%}")
                
            except Exception as e:
                st.error(f"予測エラー: {e}")
    
    with col2:
        st.header("ℹ️ 情報")
        
        # モデル情報
        st.subheader("システム詳細")
        st.info(f"""
        **特徴量数**: {len(predictor.preprocessor.numeric_features) + len(predictor.preprocessor.categorical_features)}
        - 数値: {len(predictor.preprocessor.numeric_features)}
        - カテゴリカル: {len(predictor.preprocessor.categorical_features)}
        
        **肥料タイプ数**: {len(predictor.preprocessor.target_classes)}
        **タスク**: 多クラス分類
        **評価指標**: Accuracy / F1-Macro
        """)
        
        # 利用可能な肥料タイプを表示
        st.subheader("📋 利用可能な肥料タイプ")
        for fert_type in sorted(predictor.preprocessor.target_classes):
            st.write(f"• {fert_type}")
        
        # データセット情報（利用可能な場合）
        if sample_data is not None:
            st.subheader("データセット概要")
            # 肥料名の分布を取得
            target_col = predictor.preprocessor.target_column
            if target_col in sample_data.columns:
                fert_counts = sample_data[target_col].value_counts()
                most_common = fert_counts.index[0]
                st.info(f"""
                **訓練サンプル数**: {len(sample_data):,}
                **最も一般的な肥料**: {most_common} ({fert_counts.iloc[0]} サンプル)
                **肥料タイプ分布**: {len(fert_counts)} 種類
                """)
            else:
                st.info(f"**訓練サンプル数**: {len(sample_data):,}")
        
        # 使用方法
        st.subheader("使用方法")
        st.markdown("""
        1. 🌾 サイドバーで農業条件を入力
           - 土壌特性（pH、NPK濃度など）
           - 環境条件（気温、降水量など）
           - 作物情報（種類、成長段階など）
        2. 🔄 肥料推奨は自動的に更新されます
        3. 📊 推奨確率と候補肥料を確認
        4. 🎯 最も確率の高い肥料が推奨されます
        """)


def batch_prediction_page():
    """CSVファイルからのバッチ肥料推奨用ページ"""
    st.title("📊 バッチ肥料推奨")
    st.markdown("CSVファイルをアップロードして、複数の農業条件に対する肥料推奨を取得できます。")
    
    predictor = load_predictor()
    
    if predictor is None:
        st.error("モデルが利用できません。")
        return
    
    # ファイルアップロード
    uploaded_file = st.file_uploader(
        "CSVファイルを選択",
        type="csv",
        help="農業条件データ（土壌、環境、作物情報）を含むCSVファイルをアップロードしてください"
    )
    
    if uploaded_file is not None:
        try:
            # ファイル読み込み
            df = pd.read_csv(uploaded_file)
            st.write("**アップロードされたデータのプレビュー:**")
            st.dataframe(df.head())
            
            # 予測を実行
            if st.button("肥料推奨を生成"):
                with st.spinner("肥料推奨を生成中..."):
                    fertilizer_names = []
                    max_probabilities = []
                    
                    for idx, row in df.iterrows():
                        fert_name, prob_dict = predictor.predict_single_sample(row.to_dict())
                        fertilizer_names.append(fert_name)
                        max_probabilities.append(max(prob_dict.values()))
                    
                    # データフレームに推奨結果を追加
                    result_df = df.copy()
                    result_df['Recommended_Fertilizer'] = fertilizer_names
                    result_df['Confidence'] = max_probabilities
                    
                    # 結果を表示
                    st.success("肥料推奨が正常に生成されました！")
                    st.dataframe(result_df)
                    
                    # ダウンロードボタン
                    csv = result_df.to_csv(index=False)
                    st.download_button(
                        label="推奨結果をCSVでダウンロード",
                        data=csv,
                        file_name="fertilizer_recommendations.csv",
                        mime="text/csv"
                    )
                    
                    # 推奨統計
                    st.subheader("推奨サマリー")
                    col1, col2, col3 = st.columns(3)
                    
                    with col1:
                        st.metric("総サンプル数", len(fertilizer_names))
                    with col2:
                        st.metric("平均確信度", f"{np.mean(max_probabilities):.1%}")
                    with col3:
                        unique_fertilizers = len(set(fertilizer_names))
                        st.metric("推奨肥料タイプ数", unique_fertilizers)
                    
                    # 肥料推奨の分布
                    fertilizer_counts = pd.Series(fertilizer_names).value_counts()
                    fig = px.bar(
                        x=fertilizer_counts.values,
                        y=fertilizer_counts.index,
                        orientation='h',
                        title="推奨肥料の分布",
                        labels={'x': '推奨回数', 'y': '肥料タイプ'}
                    )
                    st.plotly_chart(fig, use_container_width=True)
        
        except Exception as e:
            st.error(f"ファイル処理エラー: {e}")


if __name__ == "__main__":
    # ページナビゲーションを作成
    page = st.sidebar.selectbox(
        "ナビゲーション",
        ["単一推奨", "バッチ推奨"]
    )
    
    if page == "単一推奨":
        main()
    elif page == "バッチ推奨":
        batch_prediction_page()