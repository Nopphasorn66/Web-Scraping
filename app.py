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

# ============== UI & PAGES ==============
st.set_page_config(page_title="นักแสดงและผู้กำกับชั้นนำ", page_icon="🎬", layout="wide")
st.sidebar.title("เมนู")
page = st.sidebar.radio("ไปที่", ["🎬ค้นหาทีมที่คุณต้องการ", "🏅เทรนความนิยม"])

ALL_GENRES = ['action','adventure','comedy','crime',
              'documentary','history','horror','music',
              'romance','sports','war']

# ============== Utils ==============
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

def list_to_token_str(L):
    return " ".join(normalize_tokens(L))

def genre_to_token_str(L):
    return " ".join([re.sub(r"\s+","_",str(g).strip().lower()) for g in L if str(g).strip()])

def _get_vocab(vec):
    try: return vec.get_feature_names_out()
    except AttributeError: return vec.get_feature_names()

def tok2name(tok): return tok.replace("_"," ").title()

# ----- FunctionTransformer selector (no lambda -> picklable) -----
def select_field(X, field):
    return X[field]
def make_field_selector(field):
    return FunctionTransformer(select_field, validate=False, kw_args={"field": field})

# ============== Data Loader ==============
@st.cache_data(show_spinner=True)
def load_movies(csv_path="movies2.csv"):
    df = pd.read_csv(csv_path)

    # แปลงคะแนน
    df["tom_score_num"] = df.get("tom_score", np.nan).apply(pct_to_float)
    df["pop_score_num"] = df.get("pop_score", np.nan).apply(pct_to_float)

    # ดรอปเฉพาะกรณี "ทั้งคู่" เป็น 0%
    both_zero = (df.get("tom_score","").astype(str).str.strip()=="0%") & \
                (df.get("pop_score","").astype(str).str.strip()=="0%")
    df = df[~both_zero].copy()

    # list fields
    for col in ["ld_actors","ld_directors","genres"]:
        if col in df.columns:
            df[col] = df[col].apply(to_list)
        else:
            df[col] = [[] for _ in range(len(df))]

    # strings for vectorizers
    df["actors_str"]    = df["ld_actors"].apply(list_to_token_str)
    df["directors_str"] = df["ld_directors"].apply(list_to_token_str)
    df["genres_str"]    = df["genres"].apply(genre_to_token_str)

    # display + year
    df["release_date_disp"] = df.get("release_date","").astype(str)
    # ดึงปีจาก release_date (ถ้าไม่มีจะเป็น NaN)
    df["year"] = df["release_date_disp"].str.extract(r"(\d{4})").astype(float)
    df["score_mean"] = df[["tom_score_num","pop_score_num"]].mean(axis=1)

    return df

df = load_movies("movies2.csv")
if df.empty:
    st.error("ไม่พบข้อมูลใน movies2.csv หรือไฟล์ว่าง")
    st.stop()

# ============== Features & Model ==============
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

    from sklearn.metrics import r2_score
    try: r2 = r2_score(yte, pred)
    except Exception: r2 = np.nan

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

# ============== Helpers for Recommendations ==============
def has_any_genre(gen_list, chosen):
    try:
        gl = [str(x).lower() for x in (gen_list or [])]
    except Exception:
        gl = []
    for g in chosen:
        if g.lower() in gl:
            return True
    return False

def appearances_in_genre_combo(df, genres_selected):
    m = df["genres"].apply(lambda L: has_any_genre(L, genres_selected))
    dfg = df.loc[m]
    if dfg.empty:
        return pd.Series(dtype=int), pd.Series(dtype=int), dfg
    act_counts = pd.Series([a for L in dfg["ld_actors"] for a in L]).value_counts()
    dir_counts = pd.Series([d for L in dfg["ld_directors"] for d in L]).value_counts()
    return act_counts, dir_counts, dfg

def combine_fw_dual(fw_aud, fw_crit, block):
    A = fw_aud[fw_aud["block"]==block][["token","weight"]].rename(columns={"weight":"aud_weight"})
    C = fw_crit[fw_crit["block"]==block][["token","weight"]].rename(columns={"weight":"crit_weight"})
    M = pd.merge(A, C, on="token", how="outer").fillna(0.0)
    # ชื่ออ่านง่าย
    M["name"] = M["token"].apply(tok2name)
    return M[["token","name","aud_weight","crit_weight"]]

def sort_by_focus(df_in, focus="balanced"):
    df = df_in.copy()
    if focus == "audience":
        df["weight"] = df["aud_weight"]
    elif focus == "critics":
        df["weight"] = df["crit_weight"]
    else:
        # balanced: Z-score แล้วเฉลี่ย
        for c in ["aud_weight","crit_weight"]:
            std = df[c].std(ddof=0)
            if std > 1e-9:
                df[c] = (df[c]-df[c].mean())/std
        df["weight"] = 0.5*df["aud_weight"] + 0.5*df["crit_weight"]
    return df.sort_values("weight", ascending=False)

def recommend_people(df, fw_pop, fw_tom, genres_selected, focus="balanced", top_k=20, min_appearances=2):
    # สร้างตารางน้ำหนักแยก audience/critics
    acts_dual = combine_fw_dual(fw_pop, fw_tom, "actors")
    dirs_dual = combine_fw_dual(fw_pop, fw_tom, "directors")

    # นับจำนวนผลงานในแนวที่เลือก
    act_counts, dir_counts, dfg = appearances_in_genre_combo(df, genres_selected)

    def top_block(dual_df, counts, top_k):
        out = dual_df.copy()
        if not counts.empty:
            out["appearances"] = out["name"].map(counts).fillna(0).astype(int)
            out = out[out["appearances"] >= int(min_appearances)]
        else:
            out["appearances"] = 0
        out = sort_by_focus(out, focus=focus)
        return out.head(top_k)[["name","aud_weight","crit_weight","appearances","weight"]]

    return {
        "actors": top_block(acts_dual, act_counts, top_k),
        "directors": top_block(dirs_dual, dir_counts, top_k),
        "df_subset": dfg
    }

def best_title_for_person(df_subset, person_name, role="actor"):
    """หา 1 เรื่องที่คะแนนเฉลี่ยสูงสุดที่มีคนนี้อยู่ (role: 'actor' หรือ 'director')"""
    if df_subset.empty: return None
    if role=="actor":
        m = df_subset["ld_actors"].apply(lambda L: person_name in L)
    else:
        m = df_subset["ld_directors"].apply(lambda L: person_name in L)
    sub = df_subset.loc[m]
    if sub.empty: return None
    row = sub.sort_values("score_mean", ascending=False).iloc[0]
    return dict(
        title=row.get("title","-"),
        release=row.get("release_date_disp","-"),
        audience=row.get("pop_score","-"),
        critics=row.get("tom_score","-"),
        url=row.get("url","-")
    )

# ============== Train once (both targets) ==============
with st.spinner("กำลังฝึกโมเดลและประเมินผล..."):
    feats_pop, model_pop, rep_pop = train_and_eval(df, target_col="pop_score_num")
    feats_tom, model_tom, rep_tom = train_and_eval(df, target_col="tom_score_num")
    fw_pop = get_token_weights(feats_pop, model_pop)
    fw_tom = get_token_weights(feats_tom, model_tom)

# ============== PAGE 1 ==============
if page == "🎬ค้นหาทีมที่คุณต้องการ":
    st.title("🏆 นักแสดงและผู้กำกับชั้นนำ")

    # -------- ตัวกรองหลัก (PAGE 1 Header) --------
    colA, colB, colC = st.columns([1.6, 1, 1])
    with colA:
        sel_genres = st.multiselect("เลือกประเภทหนัง (เลือกได้หลายประเภท)", ALL_GENRES)
    with colB:
        min_app = st.number_input("ขั้นต่ำจำนวนผลงานต่อชื่อ (appearances)", 1, 10, 2, step=1)
    with colC:
        focus = st.selectbox("โฟกัสการจัดอันดับ (Scores)", ["balanced","audience","critics"], index=0)

    # ---------- ส่วนที่ 1: Actors/Directors แนะนำ ----------
    st.markdown("---")
    st.subheader("🏆 รายชื่อนักแสดงและผู้กำกับที่โมเดลแนะนำ (ภายในแนวที่เลือก)")

    if sel_genres:
        # ตัวเลื่อนกำหนดจำนวนชื่อ
        c1, c2, _ = st.columns([1, 1, 2])
        with c1:
            k_actors = st.slider("จำนวนชื่อนักแสดง (0–10)", min_value=0, max_value=10, value=10, step=1)
        with c2:
            k_dirs   = st.slider("จำนวนชื่อผู้กำกับ (0–5)", min_value=0, max_value=5,  value=5,  step=1)

        ask_topk = max(k_actors, k_dirs, 1)
        recs = recommend_people(
            df, fw_pop, fw_tom, sel_genres,
            focus=focus, top_k=ask_topk, min_appearances=int(min_app)
        )

        def rank_and_slice(df_in: pd.DataFrame, n: int, who="actor"):
            if df_in is None or df_in.empty or n <= 0:
                return pd.DataFrame(columns=["ลำดับ", "ชื่อ", "คะแนนผู้ชม", "คะแนนนักวิจารณ์", "จำนวนครั้ง"])
            out = df_in.head(n).reset_index(drop=True)
            out.insert(0, "ลำดับ", range(1, len(out) + 1))
            out = out.rename(columns={
                "name": "ชื่อ",
                "aud_weight": "คะแนนผู้ชม",
                "crit_weight": "คะแนนนักวิจารณ์",
                "appearances": "จำนวนครั้ง"
            })
            return out[["ลำดับ","ชื่อ","คะแนนผู้ชม","คะแนนนักวิจารณ์","จำนวนครั้ง"]]

        colR1, colR2 = st.columns(2)
        with colR1:
            st.markdown("**Actors (เรียงตามน้ำหนักที่หนุนคะแนน)**")
            actors_tbl = rank_and_slice(recs["actors"], k_actors, "actor")
            st.dataframe(actors_tbl, use_container_width=True, hide_index=True)

        with colR2:
            st.markdown("**Directors (เรียงตามน้ำหนักที่หนุนคะแนน)**")
            dirs_tbl = rank_and_slice(recs["directors"], k_dirs, "director")
            st.dataframe(dirs_tbl, use_container_width=True, hide_index=True)
    else:
        st.info("โปรดเลือกอย่างน้อย 1 ประเภท")

    # ---------- ส่วนที่ 2: Top N Chart ----------
    st.markdown("---")
    st.subheader("🖼️ คุณภาพหนังของ Top N ที่เลือก (Tom vs Audience)")
    colT1, colT2 = st.columns([1,1])
    with colT1:
        top_n = st.number_input("Top N หนัง (ใช้ทำกราฟ/อ้างอิง)", 5, 100, 20, step=5)
    with colT2:
        st.write("")  # spacer

    if not sel_genres:
        st.info("โปรดเลือกอย่างน้อย 1 ประเภท")
    else:
        mask_sel = df["genres"].apply(lambda L: has_any_genre(L, sel_genres))
        df_sel = df.loc[mask_sel].copy().sort_values("score_mean", ascending=False).head(int(top_n))

        if df_sel.empty:
            st.warning("ไม่มีหนังในแนวที่เลือก")
        else:
            plot_df = pd.DataFrame({
                "title": np.repeat(df_sel["title"].values, 2),
                "Metric": ["Tomatometer", "Audience"] * len(df_sel),
                "Score": np.r_[df_sel["tom_score_num"].values, df_sel["pop_score_num"].values]
            })
            title_order = df_sel["title"].tolist()

            chart = (
                alt.Chart(plot_df)
                .mark_bar()
                .encode(
                    x=alt.X("title:N", title="Title", sort=title_order),
                    y=alt.Y("Score:Q", title="Score (%)", scale=alt.Scale(domain=[0, 100])),
                    color=alt.Color(
                        "Metric:N",
                        title="Metric",
                        scale=alt.Scale(
                            domain=["Audience", "Tomatometer"],
                            range=["#fed627", "#f96d12"] 
                        )
                    ),
                    xOffset="Metric:N",
                    tooltip=["title:N", "Metric:N", alt.Tooltip("Score:Q", format=".1f")]
                )
                .properties(height=320)
            )
            st.altair_chart(chart, use_container_width=True)

    # ---------- ส่วนที่ 3: หนังอ้างอิง ----------
    st.markdown("---")
    st.subheader("🎞️ หนังอ้างอิงในแนวที่เลือก (ใช้คัดทีมงาน)")
    if sel_genres and not df.empty:
        mask_sel = df["genres"].apply(lambda L: has_any_genre(L, sel_genres))
        df_sel_tbl = df.loc[mask_sel].copy().sort_values("score_mean", ascending=False).head(int(top_n))
        tbl = df_sel_tbl[["title","release_date_disp","pop_score","tom_score","url"]].rename(columns={
            "title":"name",
            "release_date_disp":"release_date",
            "pop_score":"audience_score",
            "tom_score":"critics_score"
        })
        st.dataframe(tbl, use_container_width=True, hide_index=True)
    else:
        st.info("โปรดเลือกอย่างน้อย 1 ประเภท")

    # ---------- ส่วนที่ 4: สรุปเชิงปฏิบัติ ----------
    st.markdown("---")
    st.header("🧾 สรุปเชิงปฏิบัติ")
    if sel_genres:
        # ใช้ผลจาก recs เดิม (ถ้าไม่มีเพราะยังไม่เลือก genres ให้ข้าม)
        if 'recs' not in locals():
            recs = recommend_people(
                df, fw_pop, fw_tom, sel_genres,
                focus=focus, top_k=20, min_appearances=int(min_app)
            )

        actors_short = ", ".join(recs["actors"]["name"].head(5).tolist()) if not recs["actors"].empty else "-"
        dirs_short   = ", ".join(recs["directors"]["name"].head(3).tolist()) if not recs["directors"].empty else "-"

        actor_example = "-"
        director_example = "-"
        if not recs["actors"].empty or not recs["directors"].empty:
            _, _, dfg = appearances_in_genre_combo(df, sel_genres)
            if not recs["actors"].empty:
                a_name = recs["actors"]["name"].iloc[0]
                a_best = best_title_for_person(dfg, a_name, role="actor")
                if a_best:
                    actor_example = f"{a_name} → {a_best['title']} ({a_best['audience']} / {a_best['critics']})"
            if not recs["directors"].empty:
                d_name = recs["directors"]["name"].iloc[0]
                d_best = best_title_for_person(dfg, d_name, role="director")
                if d_best:
                    director_example = f"{d_name} → {d_best['title']} ({d_best['audience']} / {d_best['critics']})"

        st.success(
            f"- **นักแสดงแนะนำ (ตัวอย่าง 5 อันดับแรก):** {actors_short}\n"
            f"- **ผู้กำกับแนะนำ (ตัวอย่าง 3 อันดับแรก):** {dirs_short}\n"
            f"- **ผลงานของนักแสดงที่เด่น (1 เรื่อง):** {actor_example}\n"
            f"- **ผลงานของผู้กำกับที่เด่น (1 เรื่อง):** {director_example}\n\n"
            f"ใช้เกณฑ์ `appearances ≥ {int(max(min_app,2))}` เพื่อลดอคติจากตัวอย่างน้อย และอ้างอิงจาก Top N = {int(max(top_n,20))} เรื่องคะแนนสูงในแนวที่เลือก\n"
            f"โฟกัสการจัดอันดับ: **{focus}** (audience/critics/balanced) เพื่อหาทีมที่ทั้งผู้ชมและนักวิจารณ์น่าจะชอบ"
        )
    else:
        st.info("โปรดเลือกอย่างน้อย 1 ประเภท")

    # # ---------- ส่วนที่ 5: ผลประเมินโมเดล ----------
    # st.markdown("#### 📈 ผลการประเมินโมเดล (สรุป)")
    # met_col1, met_col2 = st.columns(2)
    # with met_col1:
    #     st.markdown("**Audience model**")
    #     st.write(rep_pop)
    # with met_col2:
    #     st.markdown("**Critics model**")
    #     st.write(rep_tom)
    # st.caption("หมายเหตุ: Ridge + TF-IDF ให้ ‘ความสัมพันธ์’ ไม่ใช่เหตุเป็นผลโดยตรง — ดูค่า appearances ประกอบเสมอ")

# ============== PAGE 2 ==============
if page == "🏅เทรนความนิยม":
    st.title("📊 หน้า 2: ความนิยมตามแนว (ผู้ชม vs นักวิจารณ์)")
    st.caption("เลือกแนว/ช่วงปี เพื่อดูค่าเฉลี่ยคะแนน และเทียบความนิยมภาพรวม")

    # ตัวกรอง
    c1, c2 = st.columns([2, 1])
    with c1:
        sel_genres2 = st.multiselect("เลือกประเภทหนัง (หลายประเภทได้)", ALL_GENRES, default=ALL_GENRES[:3])
    with c2:
        year_min, year_max = st.slider("ช่วงปี (2000–2025)", 2000, 2025, (2005, 2025), step=1)

    # เตรียมข้อมูลและกรอง
    df_g = df.copy()
    df_g["genres_norm"] = df_g["genres"].apply(lambda L: [str(x).strip().lower() for x in to_list(L)])
    df_g = df_g.explode("genres_norm").rename(columns={"genres_norm":"genre"})
    df_g = df_g.dropna(subset=["genre"])
    df_g["genre"] = df_g["genre"].astype(str)

    # กรองแนวที่เลือก + ช่วงปี
    df_g = df_g[df_g["genre"].isin([g.lower() for g in sel_genres2])]
    df_g = df_g[(df_g["year"].fillna(0) >= year_min) & (df_g["year"].fillna(0) <= year_max)]

    if df_g.empty:
        st.warning("ไม่มีข้อมูลตามตัวกรองที่เลือก")
        st.stop()

    # ตารางสรุปแบบ Year x Genre (Audience / Critics แยกกัน)
    aud_pivot = df_g.pivot_table(
        index="year", columns="genre", values="pop_score_num", aggfunc="mean"
    ).sort_index()
    cri_pivot = df_g.pivot_table(
        index="year", columns="genre", values="tom_score_num", aggfunc="mean"
    ).sort_index()

    st.subheader("📋 ตารางค่าเฉลี่ยรายปี × แนว (Audience)")
    st.dataframe(aud_pivot.round(2), use_container_width=True)

    st.subheader("📋 ตารางค่าเฉลี่ยรายปี × แนว (Critics)")
    st.dataframe(cri_pivot.round(2), use_container_width=True)

    # กราฟแท่งความนิยมตามแนว (ค่าเฉลี่ยรวมช่วงปีที่กรอง)
    mean_aud = df_g.groupby("genre")["pop_score_num"].mean().reset_index(name="audience_mean").sort_values("audience_mean", ascending=False)
    mean_cri = df_g.groupby("genre")["tom_score_num"].mean().reset_index(name="critics_mean").sort_values("critics_mean", ascending=False)


    col1, col2 = st.columns(2)
    with col1:
        st.subheader("ความนิยมตามผู้ชม (Audience)")
        chart_a = alt.Chart(mean_aud).mark_bar(color="#4cc0b4").encode(
            x=alt.X("audience_mean:Q", title="Average Audience Score"),
            y=alt.Y("genre:N", sort="-x", title="Genre"),
            tooltip=["genre","audience_mean"]
        ).properties(height=400)
        st.altair_chart(chart_a, use_container_width=True)

    with col2:
        st.subheader("ความนิยมตามนักวิจารณ์ (Critics)")
        chart_c = alt.Chart(mean_cri).mark_bar(color="#00796c").encode(
            x=alt.X("critics_mean:Q", title="Average Critics Score"),
            y=alt.Y("genre:N", sort="-x", title="Genre"),
            tooltip=["genre","critics_mean"]
        ).properties(height=400)
        st.altair_chart(chart_c, use_container_width=True)

