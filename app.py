import streamlit as st
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches

# ─── Page Config ────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Student Success Dashboard",
    page_icon="🎓",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ─── Custom CSS ─────────────────────────────────────────────────────────────
st.markdown("""
<style>
  @import url('https://fonts.googleapis.com/css2?family=DM+Sans:wght@300;400;500;600;700&family=DM+Serif+Display:ital@0;1&display=swap');

  html, body, [class*="css"] {
    font-family: 'DM Sans', sans-serif;
  }

  /* Background */
  .stApp {
    background: linear-gradient(135deg, #0f172a 0%, #1e293b 60%, #0f2744 100%);
    min-height: 100vh;
  }

  /* Sidebar */
  [data-testid="stSidebar"] {
    background: linear-gradient(180deg, #1e293b 0%, #162032 100%) !important;
    border-right: 1px solid rgba(99,179,237,0.15);
  }
  [data-testid="stSidebar"] * {
    color: #cbd5e1 !important;
  }
  [data-testid="stSidebar"] .stSlider label {
    font-size: 0.82rem;
    font-weight: 500;
    text-transform: uppercase;
    letter-spacing: 0.06em;
    color: #94a3b8 !important;
  }

  /* Cards */
  .card {
    background: rgba(255,255,255,0.04);
    border: 1px solid rgba(255,255,255,0.08);
    border-radius: 16px;
    padding: 24px 28px;
    backdrop-filter: blur(12px);
    margin-bottom: 16px;
  }
  .card-accent {
    border-left: 4px solid #38bdf8;
  }

  /* Metric tiles */
  .metric-row {
    display: flex;
    gap: 16px;
    margin-bottom: 16px;
  }
  .metric-tile {
    flex: 1;
    background: rgba(255,255,255,0.05);
    border: 1px solid rgba(255,255,255,0.1);
    border-radius: 14px;
    padding: 20px 22px;
    text-align: center;
  }
  .metric-tile .label {
    font-size: 0.72rem;
    font-weight: 600;
    letter-spacing: 0.1em;
    text-transform: uppercase;
    color: #e2e8f0;
    margin-bottom: 6px;
  }
  .metric-tile .value {
    font-family: 'DM Serif Display', serif;
    font-size: 2.2rem;
    font-weight: 400;
    line-height: 1;
  }
  .metric-tile .sub {
    font-size: 0.75rem;
    color: #cbd5e1;
    margin-top: 4px;
  }

  /* Risk badge */
  .badge {
    display: inline-block;
    padding: 6px 18px;
    border-radius: 999px;
    font-size: 0.82rem;
    font-weight: 600;
    letter-spacing: 0.08em;
    text-transform: uppercase;
  }
  .badge-low    { background: rgba(52,211,153,0.15); color: #34d399; border: 1px solid rgba(52,211,153,0.3); }
  .badge-mod    { background: rgba(251,191,36,0.15);  color: #fbbf24; border: 1px solid rgba(251,191,36,0.3); }
  .badge-high   { background: rgba(248,113,113,0.15); color: #f87171; border: 1px solid rgba(248,113,113,0.3); }

  /* Section headers */
  .section-title {
    font-family: 'DM Serif Display', serif;
    font-size: 1.35rem;
    color: #e2e8f0;
    margin-bottom: 4px;
  }
  .section-sub {
    font-size: 0.78rem;
    color: #cbd5e1;
    margin-bottom: 18px;
    letter-spacing: 0.03em;
  }

  /* Page title */
  .page-header {
    padding: 10px 0 24px 0;
    border-bottom: 1px solid rgba(255,255,255,0.06);
    margin-bottom: 28px;
  }
  .page-title {
    font-family: 'DM Serif Display', serif;
    font-size: 2.4rem;
    color: #f1f5f9;
    line-height: 1.1;
  }
  .page-subtitle {
    font-size: 0.9rem;
    color: #e2e8f0;
    margin-top: 4px;
  }

  /* Progress bar custom */
  .prob-bar-wrap {
    background: rgba(255,255,255,0.07);
    border-radius: 999px;
    height: 10px;
    overflow: hidden;
    margin: 10px 0 4px 0;
  }

  /* Divider */
  hr { border-color: rgba(255,255,255,0.06) !important; }

  /* Hide default streamlit header/footer */
  #MainMenu, footer, header { visibility: hidden; }
</style>
""", unsafe_allow_html=True)

# ─── Load Data & Train Model ─────────────────────────────────────────────────
@st.cache_data
def load_data():
    df = pd.read_csv('data/student_data.csv')
    return df

@st.cache_resource
def train_model(df):
    features = ['studytime', 'failures', 'absences', 'G1', 'G2']
    X = df[features]
    y = df['G3'].apply(lambda x: 1 if x >= 10 else 0)
    model = RandomForestClassifier(n_estimators=100, random_state=42)
    model.fit(X, y)
    return model, features

df = load_data()
features = ['studytime', 'failures', 'absences', 'G1', 'G2']
model, features = train_model(df)

# ─── Sidebar ─────────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("### 🎓 Student Profile")
    st.markdown("<hr/>", unsafe_allow_html=True)
    st.markdown("**Academic Input**")
    G1 = st.slider("Grade 1 (G1)", 0, 20, 10)
    G2 = st.slider("Grade 2 (G2)", 0, 20, 10)
    st.markdown("**Behaviour & Attendance**")
    studytime = st.slider("Study Time (hrs/week)", 1, 4, 2,
                          help="1=<2hrs, 2=2–5hrs, 3=5–10hrs, 4=>10hrs")
    absences  = st.slider("Absences (days)", 0, 30, 5)
    failures  = st.slider("Past Failures", 0, 3, 0)
    st.markdown("<hr/>", unsafe_allow_html=True)
    st.caption("Adjust the sliders to update predictions in real time.")

# ─── Bootstrap Uncertainty ───────────────────────────────────────────────────
input_data = np.array([[studytime, failures, absences, G1, G2]])

@st.cache_data
def bootstrap_preds(df_hash, studytime, failures, absences, G1, G2, n=60):
    inp = np.array([[studytime, failures, absences, G1, G2]])
    preds = []
    for _ in range(n):
        sample = df.sample(frac=1, replace=True)
        X_s = sample[features]
        y_s = sample['G3'].apply(lambda x: 1 if x >= 10 else 0)
        m = RandomForestClassifier(n_estimators=50, random_state=np.random.randint(9999))
        m.fit(X_s, y_s)
        preds.append(m.predict_proba(inp)[0][1])
    return preds

preds = bootstrap_preds(len(df), studytime, failures, absences, G1, G2)
mean_prob = np.mean(preds)
std_prob  = np.std(preds)
main_pred = model.predict_proba(input_data)[0][1]

if mean_prob > 0.75:
    risk, badge_cls, risk_emoji = "Low Risk",      "badge-low",  "✅"
elif mean_prob > 0.55:
    risk, badge_cls, risk_emoji = "Moderate Risk", "badge-mod",  "⚠️"
else:
    risk, badge_cls, risk_emoji = "High Risk",     "badge-high", "🚨"

# ─── Page Header ─────────────────────────────────────────────────────────────
st.markdown("""
<div class="page-header">
  <div class="page-title">Student Success Prediction</div>
  <div class="page-subtitle">Machine-learning powered academic risk analysis · Random Forest · Bootstrap uncertainty</div>
</div>
""", unsafe_allow_html=True)

# ─── Top Metric Tiles ─────────────────────────────────────────────────────────
col1, col2, col3, col4 = st.columns(4)

prob_color = "#34d399" if mean_prob > 0.75 else ("#fbbf24" if mean_prob > 0.55 else "#f87171")
with col1:
    st.markdown(f"""
    <div class="metric-tile">
      <div class="label">Pass Probability</div>
      <div class="value" style="color:{prob_color}">{mean_prob*100:.1f}%</div>
      <div class="sub">Bootstrap mean (n=60)</div>
    </div>""", unsafe_allow_html=True)

with col2:
    st.markdown(f"""
    <div class="metric-tile">
      <div class="label">Uncertainty (±)</div>
      <div class="value" style="color:#94a3b8">{std_prob*100:.1f}%</div>
      <div class="sub">Bootstrap std dev</div>
    </div>""", unsafe_allow_html=True)

with col3:
    st.markdown(f"""
    <div class="metric-tile">
      <div class="label">Risk Level</div>
      <div class="value" style="font-size:1.5rem; padding-top:6px">{risk_emoji}</div>
      <div style="margin-top:6px"><span class="badge {badge_cls}">{risk}</span></div>
    </div>""", unsafe_allow_html=True)

with col4:
    avg_g = (G1 + G2) / 2
    st.markdown(f"""
    <div class="metric-tile">
      <div class="label">Avg Grade (G1+G2)</div>
      <div class="value" style="color:#818cf8">{avg_g:.1f}<span style="font-size:1rem;color:#475569">/20</span></div>
      <div class="sub">Prior period average</div>
    </div>""", unsafe_allow_html=True)

st.markdown("<div style='height:10px'></div>", unsafe_allow_html=True)

# ─── Main Charts ─────────────────────────────────────────────────────────────
left, right = st.columns([1.2, 1], gap="large")

# Chart styling
DARK_BG   = "#0f172a"
CARD_BG   = "#1e293b"
TEXT_COL  = "#e2e8f0"
GRID_COL  = "#1e2d3d"
ACCENT    = "#38bdf8"
ACCENT2   = "#818cf8"

def style_ax(ax, fig):
    fig.patch.set_facecolor(DARK_BG)
    ax.set_facecolor(CARD_BG)
    ax.tick_params(colors=TEXT_COL, labelsize=9)
    ax.spines[:].set_color(GRID_COL)
    ax.yaxis.grid(True, color=GRID_COL, linewidth=0.7, linestyle='--')
    ax.set_axisbelow(True)
    for label in ax.get_xticklabels() + ax.get_yticklabels():
        label.set_color(TEXT_COL)
        label.set_fontsize(9)

with left:
    st.markdown("""
    <div class="section-title">Prediction Distribution</div>
    <div class="section-sub">Bootstrap sample spread — tighter is more confident</div>
    """, unsafe_allow_html=True)

    fig, ax = plt.subplots(figsize=(6.5, 3.4))
    style_ax(ax, fig)

    n_bins = 14
    counts, bin_edges = np.histogram(preds, bins=n_bins)
    colors = []
    for edge in bin_edges[:-1]:
        mid = edge + (bin_edges[1]-bin_edges[0])/2
        if mid < 0.55:   colors.append("#f87171")
        elif mid < 0.75: colors.append("#fbbf24")
        else:            colors.append("#34d399")

    ax.bar(bin_edges[:-1], counts, width=(bin_edges[1]-bin_edges[0])*0.85,
           color=colors, align='edge', zorder=3, linewidth=0)
    ax.axvline(mean_prob, color=ACCENT, linewidth=1.8, linestyle='--', label=f'Mean: {mean_prob:.2f}')
    ax.axvspan(mean_prob - std_prob, mean_prob + std_prob, alpha=0.08, color=ACCENT)
    ax.set_xlabel("Pass Probability", color=TEXT_COL, fontsize=9, labelpad=8)
    ax.set_ylabel("Frequency", color=TEXT_COL, fontsize=9, labelpad=8)
    ax.legend(frameon=False, labelcolor=TEXT_COL, fontsize=8)

    low_p   = mpatches.Patch(color='#f87171', label='High Risk (<55%)')
    mod_p   = mpatches.Patch(color='#fbbf24', label='Moderate (55–75%)')
    low_r   = mpatches.Patch(color='#34d399', label='Low Risk (>75%)')
    ax.legend(handles=[low_p, mod_p, low_r], frameon=False, labelcolor=TEXT_COL, fontsize=7.5,
              loc='upper left')

    plt.tight_layout(pad=1.5)
    st.pyplot(fig, use_container_width=True)
    plt.close()

with right:
    st.markdown("""
    <div class="section-title">Feature Importance</div>
    <div class="section-sub">Which factors drive the model's decision most</div>
    """, unsafe_allow_html=True)

    importances = model.feature_importances_
    sorted_idx  = np.argsort(importances)
    feat_labels = [features[i] for i in sorted_idx]
    feat_vals   = importances[sorted_idx]

    feat_display = {
        'studytime': 'Study Time',
        'failures':  'Past Failures',
        'absences':  'Absences',
        'G1':        'Grade 1',
        'G2':        'Grade 2',
    }
    feat_labels_nice = [feat_display.get(f, f) for f in feat_labels]

    fig2, ax2 = plt.subplots(figsize=(5.5, 3.4))
    style_ax(ax2, fig2)

    bar_colors = [ACCENT2 if v == max(feat_vals) else ACCENT for v in feat_vals]
    bars = ax2.barh(feat_labels_nice, feat_vals, color=bar_colors,
                    height=0.55, zorder=3)
    for bar, val in zip(bars, feat_vals):
        ax2.text(val + 0.005, bar.get_y() + bar.get_height()/2,
                 f'{val:.3f}', va='center', ha='left',
                 color=TEXT_COL, fontsize=8.5)
    ax2.set_xlabel("Importance Score", color=TEXT_COL, fontsize=9, labelpad=8)
    ax2.xaxis.grid(True, color=GRID_COL, linewidth=0.7, linestyle='--')
    ax2.yaxis.grid(False)

    plt.tight_layout(pad=1.5)
    st.pyplot(fig2, use_container_width=True)
    plt.close()

# ─── Input Summary Card ───────────────────────────────────────────────────────
st.markdown("<div style='height:6px'></div>", unsafe_allow_html=True)
st.markdown("""
<div class="section-title">Student Input Summary</div>
<div class="section-sub">Current values used for prediction</div>
""", unsafe_allow_html=True)

c1, c2, c3, c4, c5 = st.columns(5)
inputs = [
    ("Study Time", f"{studytime}/4 hrs", "#38bdf8"),
    ("Failures",   f"{failures} prev",   "#f87171"),
    ("Absences",  f"{absences} days",    "#fbbf24"),
    ("Grade 1",   f"{G1}/20",            "#818cf8"),
    ("Grade 2",   f"{G2}/20",            "#a78bfa"),
]
for col, (label, val, color) in zip([c1, c2, c3, c4, c5], inputs):
    col.markdown(f"""
    <div class="metric-tile">
      <div class="label">{label}</div>
      <div class="value" style="color:{color}; font-size:1.6rem">{val}</div>
    </div>""", unsafe_allow_html=True)

# ─── Footer ──────────────────────────────────────────────────────────────────
st.markdown("<div style='height:32px'></div>", unsafe_allow_html=True)
st.markdown("""
<div style='text-align:center; color:#cbd5e1; font-size:0.75rem; border-top:1px solid rgba(255,255,255,0.05); padding-top:16px'>
  Student Success Dashboard · Random Forest Classifier · Bootstrap Uncertainty Estimation (n=60)
</div>
""", unsafe_allow_html=True)