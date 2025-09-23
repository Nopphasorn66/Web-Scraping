# # app.py
# # -*- coding: utf-8 -*-
# import ast, re, warnings, numpy as np, pandas as pd
# import streamlit as st
# import altair as alt

# from functools import partial
# from sklearn.model_selection import train_test_split
# from sklearn.pipeline import Pipeline, FeatureUnion
# from sklearn.preprocessing import FunctionTransformer
# from sklearn.feature_extraction.text import TfidfVectorizer
# from sklearn.linear_model import Ridge
# from sklearn.metrics import mean_absolute_error, mean_squared_error

# warnings.filterwarnings("ignore", category=UserWarning)

# # =========================
# # Streamlit UI
# # =========================
# st.set_page_config(page_title="นักแสดงและผู้กำกับชั้นนำ", page_icon="🎬", layout="wide")
# st.title("🎬 นักแสดงและผู้กำกับชั้นนำ")
# st.caption("อิงข้อมูล Rotten Tomatoes + วิเคราะห์ด้วย ML (Ridge + TF-IDF) เพื่อค้นหาว่าใคร ‘หนุนคะแนน’ ทั้งผู้ชม (Audience) และนักวิจารณ์ (Tomatometer)")

# ALL_GENRES = ['action','adventure','comedy','crime',
#               'documentary','history','horror','music',
#               'romance','sports','war']

# # =========================
# # Utilities
# # =========================
# def safe_rmse(y_true, y_pred):
#     try:
#         return mean_squared_error(y_true, y_pred, squared=False)
#     except TypeError:
#         return np.sqrt(mean_squared_error(y_true, y_pred))

# def pct_to_float(x):
#     if pd.isna(x): return np.nan
#     m = re.search(r"(\d+(\.\d+)?)", str(x))
#     return float(m.group(1)) if m else np.nan

# def to_list(x):
#     if isinstance(x, list): return x
#     if pd.isna(x): return []
#     s = str(x).strip()
#     try:
#         v = ast.literal_eval(s)
#         if isinstance(v, list): return [str(t).strip() for t in v]
#         if isinstance(v, str): return [v.strip()] if v.strip() else []
#     except Exception:
#         pass
#     if "," in s: return [t.strip() for t in s.split(",") if t.strip()]
#     return [s] if s else []

# def normalize_tokens(L):
#     out=[]
#     for w in L:
#         w = re.sub(r"\s+", "_", str(w).strip().lower())
#         w = re.sub(r"[^a-z0-9_+\'\-]", "", w)
#         if w: out.append(w)
#     return out

# def list_to_token_str(L):  # actors/directors
#     return " ".join(normalize_tokens(L))

# def genre_to_token_str(L):
#     return " ".join([re.sub(r"\s+","_",str(g).strip().lower()) for g in L if str(g).strip()])

# def _get_vocab(vec):
#     try: return vec.get_feature_names_out()
#     except AttributeError: return vec.get_feature_names()

# def tok2name(tok):
#     return tok.replace("_"," ").title()

# # ---------- FunctionTransformer helpers (no lambda) ----------
# def select_field(X, field):
#     # X เป็น DataFrame; คืน Series/array ของคอลัมน์ที่ต้องการ
#     return X[field]

# def make_field_selector(field):
#     # ใช้ฟังก์ชัน top-level + kw_args เพื่อให้ picklable
#     return FunctionTransformer(select_field, validate=False, kw_args={"field": field})

# # =========================
# # Load data
# # =========================
# @st.cache_data(show_spinner=True)
# def load_movies(csv_path="movies.csv"):
#     df = pd.read_csv(csv_path)

#     # scores
#     df["tom_score_num"] = df.get("tom_score", np.nan).apply(pct_to_float)
#     df["pop_score_num"] = df.get("pop_score", np.nan).apply(pct_to_float)

#     # drop เฉพาะกรณี "ทั้งคู่" เป็น 0%
#     both_zero = (df.get("tom_score","").astype(str).str.strip()=="0%") & \
#                 (df.get("pop_score","").astype(str).str.strip()=="0%")
#     df = df[~both_zero].copy()

#     # lists
#     for col in ["ld_actors","ld_directors","genres"]:
#         if col in df.columns:
#             df[col] = df[col].apply(to_list)
#         else:
#             df[col] = [[] for _ in range(len(df))]

#     # strings for vectorizers
#     df["actors_str"]    = df["ld_actors"].apply(list_to_token_str)
#     df["directors_str"] = df["ld_directors"].apply(list_to_token_str)
#     df["genres_str"]    = df["genres"].apply(genre_to_token_str)

#     # display fields
#     df["release_date_disp"] = df.get("release_date","").astype(str)
#     df["score_mean"] = df[["tom_score_num","pop_score_num"]].mean(axis=1)

#     return df

# df = load_movies("movies.csv")
# if df.empty:
#     st.error("ไม่พบข้อมูลใน movies.csv หรือไฟล์ว่าง")
#     st.stop()

# # =========================
# # Build features / Train
# # =========================
# def build_features(max_actors=2000, max_directors=800, max_genres=120):
#     actors_pipe = Pipeline([
#         ("sel", make_field_selector("actors_str")),
#         ("tfidf", TfidfVectorizer(max_features=max_actors,
#                                   token_pattern=r"(?u)\b\w[\w_\-\+']+\b"))
#     ])
#     directors_pipe = Pipeline([
#         ("sel", make_field_selector("directors_str")),
#         ("tfidf", TfidfVectorizer(max_features=max_directors,
#                                   token_pattern=r"(?u)\b\w[\w_\-\+']+\b"))
#     ])
#     genres_pipe = Pipeline([
#         ("sel", make_field_selector("genres_str")),
#         ("tfidf", TfidfVectorizer(max_features=max_genres,
#                                   token_pattern=r"(?u)\b\w[\w_\-\+']+\b"))
#     ])
#     return FeatureUnion([("actors", actors_pipe),
#                          ("directors", directors_pipe),
#                          ("genres", genres_pipe)])

# @st.cache_resource(show_spinner=True)
# def train_and_eval(df, target_col="pop_score_num", test_size=0.2, alpha=5.0):
#     df_ = df.dropna(subset=[target_col]).copy()
#     if df_.empty:
#         return None, None, {"target": target_col, "error": "no rows"}

#     X = df_[["actors_str","directors_str","genres_str"]]
#     y = df_[target_col].values

#     Xtr, Xte, ytr, yte = train_test_split(X, y, test_size=test_size, random_state=42)

#     feats = build_features()
#     XtrF = feats.fit_transform(Xtr)
#     XteF = feats.transform(Xte)

#     model = Ridge(alpha=alpha)
#     model.fit(XtrF, ytr)
#     pred = model.predict(XteF)

#     mae, rmse = mean_absolute_error(yte, pred), safe_rmse(yte, pred)
#     baseline = float(np.mean(ytr))
#     mae_bl = mean_absolute_error(yte, np.full_like(yte, baseline))
#     rmse_bl = safe_rmse(yte, np.full_like(yte, baseline))

#     try:
#         from sklearn.metrics import r2_score
#         r2 = r2_score(yte, pred)
#     except Exception:
#         r2 = np.nan

#     report = dict(target=target_col, n_train=len(ytr), n_test=len(yte),
#                   MAE=float(mae), RMSE=float(rmse), R2=float(r2),
#                   baseline_mean=baseline, MAE_baseline=float(mae_bl), RMSE_baseline=float(rmse_bl))
#     return feats, model, report

# def get_token_weights(features_union, model):
#     if features_union is None or model is None:
#         return pd.DataFrame(columns=["block","token","weight"])
#     names=[]; weights=[]; start=0
#     for name, pipe in features_union.transformer_list:
#         vec = pipe.named_steps["tfidf"]
#         vocab = _get_vocab(vec)
#         coef_slice = model.coef_[start:start+len(vocab)]
#         start += len(vocab)
#         for tok, w in zip(vocab, coef_slice):
#             names.append((name, tok)); weights.append(w)
#     fw = pd.DataFrame({"block":[b for b,_ in names],
#                        "token":[t for _,t in names],
#                        "weight":weights}).sort_values("weight", ascending=False)
#     return fw

# # =========================
# # Controls
# # =========================
# colA, colB, colC, colD = st.columns([1.8,1.2,1,1])
# with colA:
#     sel_genres = st.multiselect("เลือกประเภทหนัง (เลือกได้หลายประเภท)", ALL_GENRES, default=["comedy"])
# with colB:
#     top_n = st.number_input("Top N หนัง (โชว์/อ้างอิง)", 5, 100, 20, step=5)
# with colC:
#     min_app = st.number_input("ขั้นต่ำจำนวนผลงานต่อชื่อ (appearances)", 1, 10, 2, step=1)
# with colD:
#     focus = st.selectbox("โฟกัสคะแนน", ["balanced","audience","critics"])

# if not sel_genres:
#     st.info("โปรดเลือกอย่างน้อย 1 ประเภท")
#     st.stop()

# # =========================
# # Train models + metrics
# # =========================
# with st.spinner("กำลังฝึกโมเดลและประเมินผล..."):
#     feats_pop, model_pop, rep_pop = train_and_eval(df, target_col="pop_score_num")
#     feats_tom, model_tom, rep_tom = train_and_eval(df, target_col="tom_score_num")
#     fw_pop = get_token_weights(feats_pop, model_pop)
#     fw_tom = get_token_weights(feats_tom, model_tom)

# st.subheader("📈 ผลการประเมินโมเดล")
# met_col1, met_col2 = st.columns(2)
# with met_col1:
#     st.markdown("**Audience model (pop_score_num)**")
#     st.write(rep_pop)
# with met_col2:
#     st.markdown("**Critics model (tom_score_num)**")
#     st.write(rep_tom)
# st.caption("หมายเหตุ: Ridge + TF-IDF ให้ ‘ความสัมพันธ์’ ไม่ใช่เหตุเป็นผลโดยตรง — ดูค่า appearances ประกอบเสมอ")

# # =========================
# # Slide show (ในแนวที่เลือก)
# # =========================
# mask_sel = df["genres"].apply(lambda L: any(g.lower() in [str(x).lower() for x in L] for g in sel_genres))
# df_sel = df.loc[mask_sel].copy().sort_values("score_mean", ascending=False).head(int(top_n))

# st.markdown("---")
# st.subheader("🖼️ สไลด์โชว์คุณภาพหนัง (Tom vs Audience) — ในแนวที่เลือก")
# if df_sel.empty:
#     st.warning("ไม่มีหนังในแนวที่เลือก")
# else:
#     cols = st.columns([1,1,6])
#     if "slide_idx" not in st.session_state:
#         st.session_state.slide_idx = 0
#     with cols[0]:
#         if st.button("⏮️ Prev"):
#             st.session_state.slide_idx = (st.session_state.slide_idx - 1) % len(df_sel)
#     with cols[1]:
#         if st.button("Next ⏭️"):
#             st.session_state.slide_idx = (st.session_state.slide_idx + 1) % len(df_sel)

#     cur = df_sel.iloc[st.session_state.slide_idx]
#     st.subheader(f"{st.session_state.slide_idx+1}/{len(df_sel)} — {cur['title']}")
#     st.caption(f"Release: {cur['release_date_disp']}  |  Link: {cur.get('url','-')}")
#     slide_scores = pd.DataFrame({
#         "Metric": ["Tomatometer", "Audience"],
#         "Score": [cur.get("tom_score_num", np.nan), cur.get("pop_score_num", np.nan)]
#     })
#     chart = alt.Chart(slide_scores).mark_bar().encode(
#         x=alt.X("Metric:N", title=None),
#         y=alt.Y("Score:Q", scale=alt.Scale(domain=[0, 100])),
#         tooltip=["Metric","Score"]
#     ).properties(height=220)
#     st.altair_chart(chart, use_container_width=True)

# # =========================
# # Recommendations
# # =========================
# def appearances_in_genre_combo(df, genres_selected):
#     m = df["genres"].apply(lambda L: any(g.lower() in [str(x).lower() for x in L] for g in genres_selected))
#     dfg = df.loc[m]
#     if dfg.empty:
#         return pd.Series(dtype=int), pd.Series(dtype=int), dfg
#     act_counts = pd.Series([a for L in dfg["ld_actors"] for a in L]).value_counts()
#     dir_counts = pd.Series([d for L in dfg["ld_directors"] for d in L]).value_counts()
#     return act_counts, dir_counts, dfg

# def combine_fw(fw_aud, fw_crit, block, mode="balanced"):
#     A = fw_aud[fw_aud["block"]==block][["token","weight"]].rename(columns={"weight":"w_aud"})
#     C = fw_crit[fw_crit["block"]==block][["token","weight"]].rename(columns={"weight":"w_crit"})
#     M = pd.merge(A, C, on="token", how="outer").fillna(0.0)
#     if mode=="audience":
#         M["w"] = M["w_aud"]
#     elif mode=="critics":
#         M["w"] = M["w_crit"]
#     else:
#         # balanced: standardize แล้วเฉลี่ย
#         if len(M)>0:
#             for c in ["w_aud","w_crit"]:
#                 std = M[c].std(ddof=0)
#                 if std > 1e-9:
#                     M[c] = (M[c]-M[c].mean())/std
#         M["w"] = 0.5*M["w_aud"] + 0.5*M["w_crit"]
#     return M[["token","w"]].sort_values("w", ascending=False)

# def recommend_people(df, fw_pop, fw_tom, genres_selected, focus="balanced", top_k=20, min_appearances=2):
#     acts_combo = combine_fw(fw_pop, fw_tom, "actors", mode=focus)
#     dirs_combo = combine_fw(fw_pop, fw_tom, "directors", mode=focus)

#     act_counts, dir_counts, dfg = appearances_in_genre_combo(df, genres_selected)

#     def top_block(combo_df, counts, top_k):
#         out = combo_df.copy()
#         out["name"] = out["token"].apply(tok2name)
#         if not counts.empty:
#             out["appearances"] = out["name"].map(counts).fillna(0).astype(int)
#             out = out[out["appearances"] >= int(min_appearances)]
#         else:
#             out["appearances"] = 0
#         return out.nlargest(top_k, "w")[["name","w","appearances"]]

#     return {
#         "actors": top_block(acts_combo, act_counts, top_k),
#         "directors": top_block(dirs_combo, dir_counts, top_k),
#         "df_subset": dfg
#     }

# st.markdown("---")
# st.subheader("🏆 รายชื่อนักแสดงและผู้กำกับที่โมเดลแนะนำ (ภายในแนวที่เลือก)")

# recs = recommend_people(df, fw_pop, fw_tom, sel_genres,
#                         focus=focus, top_k=30, min_appearances=int(min_app))

# colR1, colR2 = st.columns(2)
# with colR1:
#     st.markdown("**Actors (เรียงตามน้ำหนักที่หนุนคะแนน)**")
#     st.dataframe(recs["actors"].rename(columns={"w":"weight"}), use_container_width=True)
# with colR2:
#     st.markdown("**Directors (เรียงตามน้ำหนักที่หนุนคะแนน)**")
#     st.dataframe(recs["directors"].rename(columns={"w":"weight"}), use_container_width=True)

# # =========================
# # Reference titles
# # =========================
# st.markdown("---")
# st.subheader("🎞️ หนังอ้างอิงในแนวที่เลือก (ใช้คัดทีมงาน)")
# st.dataframe(
#     df_sel[["title","release_date_disp","tom_score","pop_score","url"]]
#       .rename(columns={"release_date_disp":"release_date"}),
#     use_container_width=True
# )

# # =========================
# # Practical summary
# # =========================
# st.markdown("---")
# st.header("🧾 สรุปเชิงปฏิบัติ")
# actors_short = ", ".join(recs["actors"]["name"].head(5).tolist()) if not recs["actors"].empty else "-"
# dirs_short   = ", ".join(recs["directors"]["name"].head(3).tolist()) if not recs["directors"].empty else "-"
# st.success(
#     f"หากต้องการสร้างหนังแนว **{', '.join([g.capitalize() for g in sel_genres])}**\n"
#     f"- **นักแสดงแนะนำ (ตัวอย่าง 5 อันดับแรก):** {actors_short}\n"
#     f"- **ผู้กำกับแนะนำ (ตัวอย่าง 3 อันดับแรก):** {dirs_short}\n"
#     f"- ใช้เกณฑ์ `appearances ≥ {int(min_app)}` เพื่อลดอคติจากตัวอย่างน้อย และอ้างอิงจาก Top N = {int(top_n)} เรื่องคะแนนสูงในแนวที่เลือก\n"
#     f"- โฟกัสการจัดอันดับ: **{focus}** (audience/critics/balanced)"
# )


# app.py
# -*- coding: utf-8 -*-
import ast, re, warnings, numpy as np, pandas as pd
import streamlit as st
import altair as alt

from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline, FeatureUnion
from sklearn.preprocessing import FunctionTransformer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_absolute_error, mean_squared_error

warnings.filterwarnings("ignore", category=UserWarning)

# =========================
# Page config & Router
# =========================
st.set_page_config(page_title="นักแสดงและผู้กำกับชั้นนำ", page_icon="🎬", layout="wide")
page = st.sidebar.radio("เลือกหน้า", ["หน้า 1: แนะนำทีมสร้างจากคะแนน", "หน้า 2: อธิบายโมเดล (ML)"])

ALL_GENRES = ['action','adventure','comedy','crime',
              'documentary','history','horror','music',
              'romance','sports','war']

# =========================
# Utilities (no lambda for pickling safety)
# =========================
def safe_rmse(y_true, y_pred):
    try:
        return mean_squared_error(y_true, y_pred, squared=False)
    except TypeError:
        return np.sqrt(mean_squared_error(y_true, y_pred))

def pct_to_float(x):
    if pd.isna(x): return np.nan
    m = re.search(r"(\d+(\.\d+)?)", str(x))
    return float(m.group(1)) if m else np.nan

def to_list(x):
    if isinstance(x, list): return x
    if pd.isna(x): return []
    s = str(x).strip()
    try:
        v = ast.literal_eval(s)
        if isinstance(v, list): return [str(t).strip() for t in v]
        if isinstance(v, str): return [v.strip()] if v.strip() else []
    except Exception:
        pass
    if "," in s: return [t.strip() for t in s.split(",") if t.strip()]
    return [s] if s else []

def normalize_tokens(L):
    out=[]
    for w in L:
        w = re.sub(r"\s+", "_", str(w).strip().lower())
        w = re.sub(r"[^a-z0-9_+\'\-]", "", w)
        if w: out.append(w)
    return out

def list_to_token_str(L):  # for actors/directors
    return " ".join(normalize_tokens(L))

def genre_to_token_str(L):
    return " ".join([re.sub(r"\s+","_",str(g).strip().lower()) for g in L if str(g).strip()])

def _get_vocab(vec):
    try: return vec.get_feature_names_out()
    except AttributeError: return vec.get_feature_names()

def tok2name(tok):
    return tok.replace("_"," ").title()

def select_field(X, field):
    return X[field]

def make_field_selector(field):
    return FunctionTransformer(select_field, validate=False, kw_args={"field": field})

# =========================
# Load data
# =========================
@st.cache_data(show_spinner=True)
def load_movies(csv_path="movies.csv"):
    df = pd.read_csv(csv_path)

    # scores
    df["tom_score_num"] = df.get("tom_score", np.nan).apply(pct_to_float)
    df["pop_score_num"] = df.get("pop_score", np.nan).apply(pct_to_float)

    # drop เฉพาะกรณี "ทั้งคู่" เป็น 0%
    both_zero = (df.get("tom_score","").astype(str).str.strip()=="0%") & \
                (df.get("pop_score","").astype(str).str.strip()=="0%")
    df = df[~both_zero].copy()

    # lists
    for col in ["ld_actors","ld_directors","genres"]:
        if col in df.columns:
            df[col] = df[col].apply(to_list)
        else:
            df[col] = [[] for _ in range(len(df))]

    # strings for vectorizers
    df["actors_str"]    = df["ld_actors"].apply(list_to_token_str)
    df["directors_str"] = df["ld_directors"].apply(list_to_token_str)
    df["genres_str"]    = df["genres"].apply(genre_to_token_str)

    # display fields
    df["release_date_disp"] = df.get("release_date","").astype(str)
    df["score_mean"] = df[["tom_score_num","pop_score_num"]].mean(axis=1)

    return df

df = load_movies("movies.csv")
if df.empty:
    st.error("ไม่พบข้อมูลใน movies.csv หรือไฟล์ว่าง")
    st.stop()

# =========================
# Build features / Train
# =========================
def build_features(max_actors=2000, max_directors=800, max_genres=120):
    actors_pipe = Pipeline([
        ("sel", make_field_selector("actors_str")),
        ("tfidf", TfidfVectorizer(max_features=max_actors,
                                  token_pattern=r"(?u)\b\w[\w_\-\+']+\b"))
    ])
    directors_pipe = Pipeline([
        ("sel", make_field_selector("directors_str")),
        ("tfidf", TfidfVectorizer(max_features=max_directors,
                                  token_pattern=r"(?u)\b\w[\w_\-\+']+\b"))
    ])
    genres_pipe = Pipeline([
        ("sel", make_field_selector("genres_str")),
        ("tfidf", TfidfVectorizer(max_features=max_genres,
                                  token_pattern=r"(?u)\b\w[\w_\-\+']+\b"))
    ])
    return FeatureUnion([("actors", actors_pipe),
                         ("directors", directors_pipe),
                         ("genres", genres_pipe)])

@st.cache_resource(show_spinner=True)
def train_and_eval(df, target_col="pop_score_num", test_size=0.2, alpha=5.0):
    df_ = df.dropna(subset=[target_col]).copy()
    if df_.empty:
        return None, None, {"target": target_col, "error": "no rows"}

    X = df_[["actors_str","directors_str","genres_str"]]
    y = df_[target_col].values

    Xtr, Xte, ytr, yte = train_test_split(X, y, test_size=test_size, random_state=42)

    feats = build_features()
    XtrF = feats.fit_transform(Xtr)
    XteF = feats.transform(Xte)

    model = Ridge(alpha=alpha)
    model.fit(XtrF, ytr)
    pred = model.predict(XteF)

    mae, rmse = mean_absolute_error(yte, pred), safe_rmse(yte, pred)
    baseline = float(np.mean(ytr))
    mae_bl = mean_absolute_error(yte, np.full_like(yte, baseline))
    rmse_bl = safe_rmse(yte, np.full_like(yte, baseline))

    try:
        from sklearn.metrics import r2_score
        r2 = r2_score(yte, pred)
    except Exception:
        r2 = np.nan

    report = dict(target=target_col, n_train=len(ytr), n_test=len(yte),
                  MAE=float(mae), RMSE=float(rmse), R2=float(r2),
                  baseline_mean=baseline, MAE_baseline=float(mae_bl), RMSE_baseline=float(rmse_bl))
    return feats, model, report

def get_token_weights(features_union, model):
    if features_union is None or model is None:
        return pd.DataFrame(columns=["block","token","weight"])
    names=[]; weights=[]; start=0
    for name, pipe in features_union.transformer_list:
        vec = pipe.named_steps["tfidf"]
        vocab = _get_vocab(vec)
        coef_slice = model.coef_[start:start+len(vocab)]
        start += len(vocab)
        for tok, w in zip(vocab, coef_slice):
            names.append((name, tok)); weights.append(w)
    fw = pd.DataFrame({"block":[b for b,_ in names],
                       "token":[t for _,t in names],
                       "weight":weights}).sort_values("weight", ascending=False)
    return fw

# =========================
# Common helpers
# =========================
def appearances_in_genre_combo(df, genres_selected):
    m = df["genres"].apply(lambda L: any(g.lower() in [str(x).lower() for x in L] for g in genres_selected))
    dfg = df.loc[m]
    if dfg.empty:
        return pd.Series(dtype=int), pd.Series(dtype=int), dfg
    act_counts = pd.Series([a for L in dfg["ld_actors"] for a in L]).value_counts()
    dir_counts = pd.Series([d for L in dfg["ld_directors"] for d in L]).value_counts()
    return act_counts, dir_counts, dfg

def combine_fw(fw_aud, fw_crit, block, mode="balanced"):
    A = fw_aud[fw_aud["block"]==block][["token","weight"]].rename(columns={"weight":"w_aud"})
    C = fw_crit[fw_crit["block"]==block][["token","weight"]].rename(columns={"weight":"w_crit"})
    M = pd.merge(A, C, on="token", how="outer").fillna(0.0)
    if mode=="audience":
        M["w"] = M["w_aud"]
    elif mode=="critics":
        M["w"] = M["w_crit"]
    else:
        # balanced: z-score แล้วเฉลี่ย
        if len(M)>0:
            for c in ["w_aud","w_crit"]:
                std = M[c].std(ddof=0)
                if std > 1e-9:
                    M[c] = (M[c]-M[c].mean())/std
        M["w"] = 0.5*M["w_aud"] + 0.5*M["w_crit"]
    return M[["token","w"]].sort_values("w", ascending=False)

def recommend_people(df, fw_pop, fw_tom, genres_selected, focus="balanced", top_k=20, min_appearances=2):
    acts_combo = combine_fw(fw_pop, fw_tom, "actors", mode=focus)
    dirs_combo = combine_fw(fw_pop, fw_tom, "directors", mode=focus)
    act_counts, dir_counts, dfg = appearances_in_genre_combo(df, genres_selected)

    def top_block(combo_df, counts, top_k):
        out = combo_df.copy()
        out["name"] = out["token"].apply(tok2name)
        if not counts.empty:
            out["appearances"] = out["name"].map(counts).fillna(0).astype(int)
            out = out[out["appearances"] >= int(min_appearances)]
        else:
            out["appearances"] = 0
        return out.nlargest(top_k, "w")[["name","w","appearances"]]

    return {
        "actors": top_block(acts_combo, act_counts, top_k),
        "directors": top_block(dirs_combo, dir_counts, top_k),
        "df_subset": dfg
    }

def best_title_for_person(df_subset, person_name, role="actor"):
    """หา 1 เรื่องที่คะแนนรวมสูงสุดของบุคคลนั้นในแนวที่เลือก"""
    if df_subset.empty: return None
    col = "ld_actors" if role=="actor" else "ld_directors"
    m = df_subset[col].apply(lambda L: person_name in L)
    dfx = df_subset.loc[m].copy()
    if dfx.empty: return None
    dfx = dfx.sort_values("score_mean", ascending=False)
    row = dfx.iloc[0]
    return {
        "title": row.get("title","-"),
        "audience_score": row.get("pop_score","-"),
        "critics_score": row.get("tom_score","-"),
        "url": row.get("url","-")
    }

# =========================
# Train models once per session
# =========================
@st.cache_resource(show_spinner=True)
def train_models_once(df):
    feats_pop, model_pop, rep_pop = train_and_eval(df, target_col="pop_score_num")
    feats_tom, model_tom, rep_tom = train_and_eval(df, target_col="tom_score_num")
    fw_pop = get_token_weights(feats_pop, model_pop)
    fw_tom = get_token_weights(feats_tom, model_tom)
    return (feats_pop, model_pop, rep_pop,
            feats_tom, model_tom, rep_tom,
            fw_pop, fw_tom)

(feats_pop, model_pop, rep_pop,
 feats_tom, model_tom, rep_tom,
 fw_pop, fw_tom) = train_models_once(df)

# =========================
# PAGE 1: แนะนำทีมสร้างจากคะแนน
# =========================
if page.startswith("หน้า 1"):
    st.title("🎬 นักแสดงและผู้กำกับชั้นนำ — แนะนำจากโมเดล")
    st.caption("เลือกแนว/เกณฑ์ → ดูสไลด์คะแนน → รายชื่อที่หนุนคะแนน → หนังอ้างอิง → สรุปเชิงปฏิบัติ")

    # ----- ส่วนที่ 1: ตัวกรอง + สไลด์โชว์ -----
    st.subheader("ส่วนที่ 1: ตั้งค่า & สไลด์โชว์คะแนน")
    colA, colB, colC, colD = st.columns([1.8,1.2,1,1])
    with colA:
        sel_genres = st.multiselect("เลือกประเภทหนัง (หลายประเภทได้)", ALL_GENRES, default=["comedy"])
    with colB:
        top_n = st.number_input("Top N หนัง (โชว์/อ้างอิง)", 5, 100, 20, step=5)
    with colC:
        min_app = st.number_input("ขั้นต่ำจำนวนผลงาน/ชื่อ (appearances)", 1, 10, 2, step=1)
    with colD:
        focus = st.selectbox("โฟกัสคะแนน", ["balanced","audience","critics"])

    if not sel_genres:
        st.info("โปรดเลือกอย่างน้อย 1 ประเภท")
        st.stop()

    # สไลด์โชว์
    mask_sel = df["genres"].apply(lambda L: any(g.lower() in [str(x).lower() for x in L] for g in sel_genres))
    df_sel = df.loc[mask_sel].copy().sort_values("score_mean", ascending=False).head(int(top_n))

    if df_sel.empty:
        st.warning("ไม่มีหนังในแนวที่เลือก")
    else:
        cols = st.columns([1,1,6])
        if "slide_idx" not in st.session_state:
            st.session_state.slide_idx = 0
        with cols[0]:
            if st.button("⏮️ Prev"):
                st.session_state.slide_idx = (st.session_state.slide_idx - 1) % len(df_sel)
        with cols[1]:
            if st.button("Next ⏭️"):
                st.session_state.slide_idx = (st.session_state.slide_idx + 1) % len(df_sel)

        cur = df_sel.iloc[st.session_state.slide_idx]
        st.subheader(f"{st.session_state.slide_idx+1}/{len(df_sel)} — {cur['title']}")
        st.caption(f"Release: {cur['release_date_disp']}  |  Link: {cur.get('url','-')}")
        slide_scores = pd.DataFrame({
            "Metric": ["Tomatometer", "Audience"],
            "Score": [cur.get("tom_score_num", np.nan), cur.get("pop_score_num", np.nan)]
        })
        chart = alt.Chart(slide_scores).mark_bar().encode(
            x=alt.X("Metric:N", title=None),
            y=alt.Y("Score:Q", scale=alt.Scale(domain=[0, 100])),
            tooltip=["Metric","Score"]
        ).properties(height=220)
        st.altair_chart(chart, use_container_width=True)

    # ----- ส่วนที่ 2: รายชื่อแนะนำ -----
    st.markdown("---")
    st.subheader("ส่วนที่ 2: รายชื่อนักแสดงและผู้กำกับที่โมเดลแนะนำ")
    recs = recommend_people(df, fw_pop, fw_tom, sel_genres,
                            focus=focus, top_k=30, min_appearances=int(min_app))
    colR1, colR2 = st.columns(2)
    with colR1:
        st.markdown("**Actors (เรียงตามน้ำหนักที่หนุนคะแนน)**")
        st.dataframe(recs["actors"].rename(columns={"w":"weight"}),
                     use_container_width=True)
    with colR2:
        st.markdown("**Directors (เรียงตามน้ำหนักที่หนุนคะแนน)**")
        st.dataframe(recs["directors"].rename(columns={"w":"weight"}),
                     use_container_width=True)

        # ----- ส่วนที่ 3: หนังอ้างอิง -----
    st.markdown("---")
    st.subheader("ส่วนที่ 3: หนังอ้างอิงในแนวที่เลือก (ใช้คัดทีมงาน)")

    if not df_sel.empty:
        ref_tbl = df_sel.copy()

        # ลบ release_date เดิมออกกันชนกัน
        if "release_date" in ref_tbl.columns and "release_date_disp" in ref_tbl.columns:
            ref_tbl = ref_tbl.drop(columns=["release_date"])

        ref_tbl = ref_tbl.rename(columns={
            "title": "name",
            "release_date_disp": "release_date_show",
            "pop_score": "audience_score",
            "tom_score": "critics_score"
        })

        st.dataframe(ref_tbl[["name","release_date_show","audience_score","critics_score","url"]],
                    use_container_width=True)
    else:
        st.info("ยังไม่มีข้อมูลอ้างอิงสำหรับแนวที่เลือก")


    # ----- ส่วนที่ 4: สรุปเชิงปฏิบัติ -----
    st.markdown("---")
    st.subheader("ส่วนที่ 4: สรุปเชิงปฏิบัติ")

    actors_short = ", ".join(recs["actors"]["name"].head(5).tolist()) if not recs["actors"].empty else "-"
    dirs_short   = ", ".join(recs["directors"]["name"].head(3).tolist()) if not recs["directors"].empty else "-"

    # หาผลงานเด่นคนละ 1 เรื่อง
    best_cast_work = "-"
    best_dir_work  = "-"
    if not recs["df_subset"].empty:
        if not recs["actors"].empty:
            top_actor = recs["actors"]["name"].iloc[0]
            ba = best_title_for_person(recs["df_subset"], top_actor, role="actor")
            if ba:
                best_cast_work = f"{top_actor} → {ba['title']} (Aud: {ba['audience_score']}, Crit: {ba['critics_score']})"
        if not recs["directors"].empty:
            top_dir = recs["directors"]["name"].iloc[0]
            bd = best_title_for_person(recs["df_subset"], top_dir, role="director")
            if bd:
                best_dir_work = f"{top_dir} → {bd['title']} (Aud: {bd['audience_score']}, Crit: {bd['critics_score']})"

    st.success(
        f"นักแสดงแนะนำ (ตัวอย่าง 5 อันดับแรก): {actors_short}\n\n"
        f"ผู้กำกับแนะนำ (ตัวอย่าง 3 อันดับแรก): {dirs_short}\n\n"
        f"ผลงานของนักแสดงและผู้กำกับ (คนละ 1 เรื่องที่ได้รับคะแนนสูง):\n"
        f"- {best_cast_work}\n"
        f"- {best_dir_work}\n\n"
        f"ใช้เกณฑ์ appearances ≥ {int(min_app)} เพื่อลดอคติจากตัวอย่างน้อย และอ้างอิงจาก Top N = {int(top_n)} เรื่องคะแนนสูงในแนวที่เลือก\n"
        f"โฟกัสการจัดอันดับ: {focus} (audience/critics/balanced)"
    )

# =========================
# PAGE 2: อธิบายโมเดล
# =========================
else:
    st.title("🧠 อธิบายโมเดล: ทำไม TF-IDF + Ridge?")
    st.markdown("""
**เป้าหมาย**: พยากรณ์คะแนนผู้ชม (Audience) และนักวิจารณ์ (Tomatometer) จากรายชื่อนักแสดง/ผู้กำกับ/แนวหนัง  
**แนวคิดหลัก**: แปลงรายชื่อให้เป็นเวกเตอร์ด้วย **TF-IDF** แล้วใช้ **Ridge Regression** เรียนรู้ความสัมพันธ์เชิงเส้น พร้อม L2 เพื่อลด overfit

### ทำไม TF-IDF?
- ลดอิทธิพลคำ/ชื่อที่โผล่บ่อย (เช่น super-star) ให้ชื่อเฉพาะทางโดดเด่นขึ้น
- เหมาะกับข้อมูลสปาร์ส (รายชื่อกระจาย) และ **ตีความง่าย**

### ทำไม Ridge?
- ฟีเจอร์จาก TF-IDF มีจำนวนมากและสปาร์ส → L2 เหมาะในการ **คุมความซับซ้อน**
- ได้ **น้ำหนัก** ของแต่ละโทเค็นตรง ๆ → นำไปจัดอันดับรายชื่อที่ “หนุนคะแนน” ได้

### เทียบวิธีอื่น
- **RandomForest/XGBoost**: อาจแม่นขึ้นเมื่อข้อมูลใหญ่ แต่ตีความ “ชื่อใครสำคัญ” ยากกว่า
- **ElasticNet**: คัดฟีเจอร์ได้คม แต่ในข้อมูลเล็กอาจทิ้งชื่อหายากเร็วไป
- **Deep Learning**: ต้องใช้ดาต้ามาก, ฝึกยากกว่า, ความโปร่งใสน้อยกว่า

### การประเมินผล
- แบ่ง Train/Test (80/20)  
- รายงาน **MAE / RMSE / R²** และเทียบกับ **Baseline = เดาค่าเฉลี่ย**  
- แนะนำให้เพิ่มฟีเจอร์ในอนาคต (ปีฉาย, ประเทศ/ภาษา, rating, runtime, สตูดิโอ, จำนวนรีวิว ฯลฯ) และลอง cross-validation/time-split

### ข้อควรระวัง
- น้ำหนักคือ **ความสัมพันธ์** ไม่ใช่เหตุเป็นผล  
- ดูค่า `appearances` (จำนวนผลงานในแนวที่เลือก) ร่วมด้วยเสมอ  
""")

    st.subheader("📈 ผลการประเมินล่าสุด (จากเซสชันนี้)")
    col1, col2 = st.columns(2)
    with col1:
        st.markdown("**Audience model (pop_score_num)**")
        st.write(rep_pop)
    with col2:
        st.markdown("**Critics model (tom_score_num)**")
        st.write(rep_tom)

    st.info("Tip: ถ้าข้อมูลเพิ่มขึ้น ลอง ElasticNet/LightGBM และเพิ่มฟีเจอร์เวลา/เรตติ้ง/ประเทศ เพื่อยกระดับ R² และความแม่น")
