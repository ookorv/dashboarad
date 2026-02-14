import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import glob
import os
from sklearn.feature_extraction.text import TfidfVectorizer

# 1. 페이지 설정
st.set_page_config(
    page_title="Naver API Insight Dashboard",
    page_icon="💖",
    layout="wide",
)

# 2. 스타일 설정 (분홍색 테마)
st.markdown("""
<style>
    :root {
        --primary-color: #FF69B4;
    }
    .main {
        background-color: #FFF0F5;
    }
    .stTabs [data-baseweb="tab-list"] {
        gap: 24px;
    }
    .stTabs [data-baseweb="tab"] {
        height: 50px;
        white-space: pre-wrap;
        background-color: #FFB6C1;
        border-radius: 4px 4px 0px 0px;
        color: white;
        padding-left: 20px;
        padding-right: 20px;
    }
    .stTabs [aria-selected="true"] {
        background-color: #FF69B4 !important;
        font-weight: bold;
    }
    h1, h2, h3 {
        color: #C71585;
    }
    .stMetric {
        background-color: white;
        padding: 15px;
        border-radius: 10px;
        box-shadow: 2px 2px 5px rgba(0,0,0,0.1);
        border-left: 5px solid #FF69B4;
    }
</style>
""", unsafe_allow_html=True)

# 3. 데이터 로드 및 전처리 캐싱
@st.cache_data
def load_all_data():
    DATA_DIR = 'data'
    files = glob.glob(os.path.join(DATA_DIR, '*.csv'))
    
    shopping_search_list = []
    blog_search_list = []
    shopping_trend_list = []
    
    for f in files:
        try:
            df = pd.read_csv(f, encoding='utf-8-sig')
        except:
            df = pd.read_csv(f, encoding='cp949')
        
        name = os.path.basename(f)
        if 'shopping_search' in name:
            shopping_search_list.append(df)
        elif 'blog_search' in name:
            blog_search_list.append(df)
        elif 'shopping_trend' in name:
            shopping_trend_list.append(df)
            
    return (
        pd.concat(shopping_search_list, ignore_index=True) if shopping_search_list else pd.DataFrame(),
        pd.concat(blog_search_list, ignore_index=True) if blog_search_list else pd.DataFrame(),
        pd.concat(shopping_trend_list, ignore_index=True) if shopping_trend_list else pd.DataFrame()
    )

shopping_df, blog_df, trend_df = load_all_data()

# 4. 사이드바 구성
st.sidebar.title("💖 Insight Filters")
keywords = sorted(trend_df['category'].unique()) if not trend_df.empty else []
selected_keywords = st.sidebar.multiselect("분석할 키워드를 선택하세요", keywords, default=keywords)

# 데이터 필터링
if not trend_df.empty:
    f_trend = trend_df[trend_df['category'].isin(selected_keywords)]
if not shopping_df.empty:
    f_shopping = shopping_df[shopping_df['keyword'].isin(selected_keywords)]
if not blog_df.empty:
    # 블로그 데이터에 키워드 정보가 없을 경우 전체 출력 (또는 파일명에서 추출하는 로직 필요)
    f_blog = blog_df

# 메인 타이틀
st.title("🛍️ Naver API 쇼핑/트렌드 인사이트")
st.markdown("분홍색 테마의 인터랙티브 대시보드입니다. 키워드별 시장 현황을 한눈에 파악해 보세요!")

# 탭 구성
tabs = st.tabs(["📊 트렌드 비교", "🛒 쇼핑 EDA", "📝 블로그 키워드", "🔍 데이터 조회"])

# 탭 1: 트렌드 비교
with tabs[0]:
    st.subheader("일별 클릭량 트렌드 (Plotly)")
    if not f_trend.empty:
        fig_trend = px.line(f_trend, x='period', y='ratio', color='category', 
                            title="일별 클릭 비중(Ratio) 변화",
                            color_discrete_sequence=px.colors.qualitative.Pastel)
        fig_trend.update_layout(plot_bgcolor='white', paper_bgcolor='rgba(0,0,0,0)')
        st.plotly_chart(fig_trend, use_container_width=True)
        
        col1, col2 = st.columns(2)
        with col1:
            st.markdown("### 📋 트렌드 기술 통계표")
            st.table(f_trend.groupby('category')['ratio'].describe())
        with col2:
            st.markdown("### 📋 피벗 테이블 (평균 클릭량)")
            p_table = f_trend.pivot_table(index='period', columns='category', values='ratio', aggfunc='mean').tail(10)
            st.dataframe(p_table, use_container_width=True)
    else:
        st.warning("분석할 트렌드 데이터가 없습니다.")

# 탭 2: 쇼핑 EDA
with tabs[1]:
    if not f_shopping.empty:
        c1, c2 = st.columns(2)
        with c1:
            st.subheader("최저가 분포 (Histogram)")
            fig_price = px.histogram(f_shopping, x='lprice', color='keyword', 
                                     marginal='box', title="상품 최저가 분포",
                                     color_discrete_sequence=['#FF69B4', '#FFB6C1'])
            st.plotly_chart(fig_price, use_container_width=True)
            
            st.markdown("### 📋 상위 10개 브랜드 테이블")
            brand_rank = f_shopping['brand'].value_counts().head(10).to_frame().reset_index()
            brand_rank.columns = ['브랜드', '상품수']
            st.table(brand_rank)
            
        with c2:
            st.subheader("상위 쇼핑몰 점유율 (Pie)")
            mall_counts = f_shopping['mallName'].value_counts().head(15)
            fig_mall = px.pie(values=mall_counts.values, names=mall_counts.index, 
                              title="상위 15개 쇼핑몰 비중",
                              color_discrete_sequence=px.colors.sequential.RdPu)
            st.plotly_chart(fig_mall, use_container_width=True)
            
            st.markdown("### 📋 쇼핑몰별 평균가 통계")
            mall_price = f_shopping[f_shopping['mallName'].isin(mall_counts.index)].groupby('mallName')['lprice'].mean().sort_values(ascending=False).to_frame()
            st.dataframe(mall_price, use_container_width=True)
            
        st.divider()
        st.subheader("브랜드별 가격 편차 (Boxplot)")
        top_10_brands = f_shopping['brand'].value_counts().head(10).index
        fig_brand_price = px.box(f_shopping[f_shopping['brand'].isin(top_10_brands)], 
                                 x='brand', y='lprice', color='keyword',
                                 title="상위 10개 브랜드 내 제품 가격대 비교",
                                 color_discrete_sequence=['#DB7093', '#FFC0CB'])
        st.plotly_chart(fig_brand_price, use_container_width=True)
    else:
        st.warning("쇼핑 검색 데이터가 없습니다.")

# 탭 3: 블로그 키워드
with tabs[2]:
    if not f_blog.empty:
        st.subheader("블로그 제목 주요 핵심어 (TF-IDF)")
        
        # TF-IDF 분석 시뮬레이션 (상위 20개)
        tfidf = TfidfVectorizer(max_features=20, lowercase=True)
        # HTML 태그 제거
        titles = f_blog['title'].str.replace('<b>', '').str.replace('</b>', '')
        tfidf_matrix = tfidf.fit_transform(titles)
        words = tfidf.get_feature_names_out()
        sums = tfidf_matrix.sum(axis=0).A1
        df_words = pd.DataFrame({'keyword': words, 'score': sums}).sort_values('score', ascending=False)
        
        fig_blog_kw = px.bar(df_words, x='score', y='keyword', orientation='h',
                             title="주요 키워드 중요도 (TF-IDF Score)",
                             color='score', color_continuous_scale='RdPu')
        st.plotly_chart(fig_blog_kw, use_container_width=True)
        
        col_b1, col_b2 = st.columns(2)
        with col_b1:
            st.markdown("### 📋 키워드 중요도 데이터")
            st.dataframe(df_words, use_container_width=True)
        with col_b2:
            st.warning("💡 분석 팁: 블로그 제목에 포함된 '후기', '추천', '비교' 등의 단어 비중을 통해 소비자 선호도를 파악할 수 있습니다.")
    else:
        st.warning("블로그 데이터가 없습니다.")

# 탭 4: 데이터 조회
with tabs[3]:
    st.subheader("데이터 원본 조회")
    data_choice = st.radio("원본 데이터를 선택하세요", ["쇼핑 검색", "쇼핑 트렌드", "블로그 검색"], horizontal=True)
    
    if data_choice == "쇼핑 검색":
        st.dataframe(f_shopping, use_container_width=True)
    elif data_choice == "쇼핑 트렌드":
        st.dataframe(f_trend, use_container_width=True)
    else:
        st.dataframe(f_blog, use_container_width=True)
    
    st.download_button("데이터 다운로드 (CSV)", f_shopping.to_csv(index=False), "data.csv", "text/csv")
