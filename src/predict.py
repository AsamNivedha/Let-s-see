
import html
from pathlib import Path

import pandas as pd
import streamlit as st

from src.predictor import explain_prediction, load_model, predict_price


BASE_DIR = Path(__file__).resolve().parent.parent
model = load_model()
REFERENCE_DATA = pd.read_csv(BASE_DIR / "data.csv")

st.set_page_config(
    page_title="House Price Predictor",
    page_icon="◆",
    layout="wide",
    initial_sidebar_state="collapsed",
)


# ----------------------------------------------------------------------------
# Design system
#   Display: Instrument Serif (hero + price)   UI: DM Sans
#   One accent (indigo) + a cyan hairline. Glow is reserved for the price.
# ----------------------------------------------------------------------------
CSS = """
<style>
@import url('https://fonts.googleapis.com/css2?family=Instrument+Serif&family=DM+Sans:wght@400;500;600&display=swap');

:root {
    --bg: #07080d;
    --glass: rgba(255, 255, 255, 0.04);
    --glass-2: rgba(255, 255, 255, 0.07);
    --line: rgba(255, 255, 255, 0.09);
    --line-2: rgba(255, 255, 255, 0.16);
    --text: #f1f2f8;
    --muted: #8a90a8;
    --accent: #7c83ff;
    --accent-soft: rgba(124, 131, 255, 0.16);
    --cyan: #5ad7ee;
    --display: 'Instrument Serif', Georgia, 'Times New Roman', serif;
}

/* ---------- Remove default Streamlit chrome ---------- */
#MainMenu, footer, header[data-testid="stHeader"], [data-testid="stToolbar"],
[data-testid="stDecoration"], [data-testid="stStatusWidget"] { display: none !important; }

html, body, .stApp, [class*="css"] {
    font-family: 'DM Sans', system-ui, -apple-system, 'Segoe UI', sans-serif;
    color: var(--text);
}
.stApp {
    background:
        radial-gradient(1000px 560px at 85% -10%, rgba(124, 131, 255, 0.17), transparent 62%),
        radial-gradient(800px 520px at -5% 105%, rgba(90, 215, 238, 0.07), transparent 62%),
        var(--bg);
}
.block-container { max-width: 1160px; padding: 2.2rem 1.5rem 3rem !important; }
* { caret-color: var(--accent); }

/* ---------- Hero ---------- */
.hero { position: relative; padding: 1.2rem 0 2.6rem; }
.hero::before {                       /* blueprint grid: the property-domain motif */
    content: ""; position: absolute; inset: -2.2rem -1.5rem 0 -1.5rem; z-index: 0; pointer-events: none;
    background-image:
        linear-gradient(var(--line) 1px, transparent 1px),
        linear-gradient(90deg, var(--line) 1px, transparent 1px);
    background-size: 56px 56px;
    opacity: 0.45;
    -webkit-mask-image: radial-gradient(ellipse 70% 100% at 20% 0%, #000 0%, transparent 75%);
    mask-image: radial-gradient(ellipse 70% 100% at 20% 0%, #000 0%, transparent 75%);
}
.hero > * { position: relative; z-index: 1; }
.badge {
    display: inline-flex; align-items: center; gap: 0.55rem; padding: 0.34rem 0.85rem;
    border: 1px solid var(--line-2); border-radius: 999px; background: rgba(7, 8, 13, 0.6);
    font-size: 0.7rem; font-weight: 600; letter-spacing: 0.16em; color: #c3c6e6;
}
.badge i { width: 6px; height: 6px; border-radius: 50%; background: var(--cyan); box-shadow: 0 0 8px var(--cyan); }
.hero h1 {
    font-family: var(--display); font-weight: 400; color: var(--text);
    font-size: clamp(3.1rem, 8vw, 6rem); line-height: 0.96; letter-spacing: -0.02em;
    margin: 1.3rem 0 1.1rem; padding: 0;
}
.hero p { color: var(--muted); font-size: 1.1rem; line-height: 1.55; margin: 0; max-width: 30rem; }

/* ---------- Form card (quiet glass) ---------- */
div[data-testid="stVerticalBlockBorderWrapper"], div[class*="st-key-props_card"] {
    background: linear-gradient(165deg, var(--glass-2), var(--glass));
    border: 1px solid var(--line) !important; border-radius: 24px !important;
    padding: 1.1rem 1.1rem 0.7rem;
    backdrop-filter: blur(18px); -webkit-backdrop-filter: blur(18px);
    box-shadow: inset 0 1px 0 rgba(255, 255, 255, 0.05), 0 30px 60px -40px #000;
}
div[data-testid="stVerticalBlockBorderWrapper"] div[data-testid="stVerticalBlockBorderWrapper"] {
    background: none; border: none !important; box-shadow: none; padding: 0; backdrop-filter: none;
}
.sec { display: flex; align-items: baseline; justify-content: space-between; margin: 0.3rem 0 1rem; }
.sec h3 { font-family: var(--display); font-weight: 400; font-size: 1.55rem; margin: 0; padding: 0; letter-spacing: -0.01em; }
.sec span { font-size: 0.8rem; color: var(--muted); }
.rule { height: 1px; margin: 1.3rem 0 1.4rem; background: var(--line); }

/* ---------- Inputs ---------- */
[data-testid="stWidgetLabel"] p { font-size: 0.8rem !important; font-weight: 500; color: var(--muted) !important; }
div[data-baseweb="input"], div[data-baseweb="base-input"], div[data-baseweb="select"] > div {
    background: rgba(5, 6, 12, 0.7) !important; border: 1px solid var(--line) !important;
    border-radius: 12px !important; transition: border-color 0.2s ease, box-shadow 0.2s ease;
}
div[data-baseweb="input"]:focus-within, div[data-baseweb="select"] > div:focus-within {
    border-color: var(--accent) !important; box-shadow: 0 0 0 3px var(--accent-soft) !important;
}
input, div[data-baseweb="select"] * { color: var(--text) !important; font-family: 'DM Sans', sans-serif !important; }
[data-testid="stNumberInputStepUp"], [data-testid="stNumberInputStepDown"] {
    background: transparent !important; color: var(--muted) !important; border: none !important;
}
[data-testid="stNumberInputStepUp"]:hover, [data-testid="stNumberInputStepDown"]:hover { color: var(--cyan) !important; }
div[data-baseweb="popover"] ul, div[data-baseweb="menu"] {
    background: #0e1019 !important; border: 1px solid var(--line); border-radius: 12px;
}
li[role="option"]:hover, li[aria-selected="true"] { background: var(--accent-soft) !important; }

/* Yes/No inclusions: each feature is a tile, answers are a segmented pill (no default red dot) */
div[data-testid="stRadio"] {
    background: rgba(5, 6, 12, 0.5); border: 1px solid var(--line); border-radius: 14px;
    padding: 0.75rem 0.9rem 0.8rem; margin-bottom: 0.7rem; transition: border-color 0.2s ease;
}
div[data-testid="stRadio"]:has(input[value="0"]:checked) { border-color: rgba(124, 131, 255, 0.4); }
div[role="radiogroup"] { gap: 0.4rem; flex-wrap: nowrap; }
div[role="radiogroup"] > label { margin: 0; flex: 1; }
div[role="radiogroup"] > label > div:first-child { display: none !important; }   /* hide radio circle */
div[role="radiogroup"] > label > div:last-child { width: 100%; margin: 0 !important; }
div[role="radiogroup"] > label div[data-testid="stMarkdownContainer"] p {
    text-align: center; width: 100%; padding: 0.38rem 0; border-radius: 9px; font-size: 0.88rem;
    color: var(--muted) !important; border: 1px solid var(--line); background: transparent;
    transition: all 0.2s ease; cursor: pointer;
}
div[role="radiogroup"] > label:has(input:checked) div[data-testid="stMarkdownContainer"] p {
    color: #fff !important; border-color: rgba(124, 131, 255, 0.55);
    background: linear-gradient(120deg, rgba(79, 124, 255, 0.35), rgba(124, 131, 255, 0.35));
}
div[role="radiogroup"] > label:hover div[data-testid="stMarkdownContainer"] p { border-color: var(--line-2); }

/* ---------- Primary button ---------- */
.stButton > button, button[data-testid="stBaseButton-primary"] {
    width: 100%; padding: 1rem 1.4rem !important; border-radius: 14px !important;
    border: 1px solid rgba(255, 255, 255, 0.18) !important;
    background: linear-gradient(115deg, #4a7cff 0%, #7c83ff 60%, #8f7bff 100%) !important;
    box-shadow: 0 12px 28px -14px rgba(124, 131, 255, 0.9), inset 0 1px 0 rgba(255, 255, 255, 0.28);
    transition: transform 0.2s ease, filter 0.2s ease;
}
.stButton > button p, button[data-testid="stBaseButton-primary"] p {
    font-size: 1.02rem !important; font-weight: 600; letter-spacing: 0.01em; color: #fff !important;
}
.stButton > button:hover, button[data-testid="stBaseButton-primary"]:hover { transform: translateY(-1px); filter: brightness(1.1); }
.stButton > button:active { transform: none; }
.stButton > button:focus-visible { outline: 2px solid var(--cyan) !important; outline-offset: 3px; }

/* ---------- Estimate: the focal point ---------- */
@media (min-width: 861px) {
    div[data-testid="stColumn"]:has(.est), div[data-testid="column"]:has(.est) {
        position: sticky; top: 1.4rem; align-self: flex-start;
    }
}
.est {
    position: relative; overflow: hidden; border-radius: 28px; padding: 2.6rem 2.3rem 2.2rem;
    min-height: 480px; display: flex; flex-direction: column;
    border: 1px solid var(--line-2);
    background:
        radial-gradient(520px 320px at 100% 0%, rgba(124, 131, 255, 0.26), transparent 70%),
        radial-gradient(420px 260px at 0% 100%, rgba(90, 215, 238, 0.10), transparent 70%),
        linear-gradient(165deg, rgba(255, 255, 255, 0.08), rgba(255, 255, 255, 0.025));
    box-shadow: inset 0 1px 0 rgba(255, 255, 255, 0.08), 0 40px 80px -40px rgba(124, 131, 255, 0.35);
}
.est::after {                          /* cyan hairline along the top edge */
    content: ""; position: absolute; top: 0; left: 12%; right: 12%; height: 1px;
    background: linear-gradient(90deg, transparent, var(--cyan), transparent); opacity: 0.7;
}
.est .label { font-size: 0.72rem; font-weight: 600; letter-spacing: 0.18em; color: #b2b6e8; }
.price {
    font-family: var(--display); font-weight: 400; color: #fff; line-height: 1;
    font-size: clamp(3.2rem, 6.4vw, 5.6rem); letter-spacing: -0.02em;
    margin: 1.6rem 0 0.4rem; word-break: break-word;
    text-shadow: 0 0 60px rgba(124, 131, 255, 0.55);
    animation: reveal 0.9s cubic-bezier(.2, .8, .2, 1) 1;
}
.price .cur { font-size: 0.5em; color: var(--cyan); margin-right: 0.12em; vertical-align: 0.42em; text-shadow: none; }
@keyframes reveal {
    from { opacity: 0; transform: translateY(16px); filter: blur(12px); }
    to { opacity: 1; transform: none; filter: blur(0); }
}
.price.ghost { color: rgba(255, 255, 255, 0.13); text-shadow: none; animation: breathe 3.4s ease-in-out infinite; }
.price.ghost .cur { color: rgba(90, 215, 238, 0.35); }
@keyframes breathe { 0%, 100% { opacity: 0.55; } 50% { opacity: 1; } }
.est .lead { font-family: var(--display); font-size: 1.7rem; margin: 2.2rem 0 0.5rem; letter-spacing: -0.01em; }
.est p.sub { color: var(--muted); line-height: 1.6; margin: 0; max-width: 24rem; }
.err .label { color: #ff9db3; }

.sheet { margin-top: auto; padding-top: 1.8rem; }
.specs { display: grid; grid-template-columns: repeat(3, 1fr); border-top: 1px solid var(--line); }
.specs div { padding: 0.9rem 0.8rem 0.9rem 0; border-bottom: 1px solid var(--line); }
.specs small { display: block; font-size: 0.72rem; color: var(--muted); margin-bottom: 0.2rem; }
.specs strong { font-weight: 500; font-size: 0.98rem; }
.incl { display: flex; flex-wrap: wrap; gap: 0.5rem 1.2rem; margin-top: 1rem; font-size: 0.84rem; color: var(--muted); }
.incl span { display: inline-flex; align-items: center; gap: 0.45rem; }
.incl span::before { content: ""; width: 7px; height: 7px; border-radius: 50%; border: 1px solid var(--muted); }
.incl span.on { color: var(--text); }
.incl span.on::before { background: var(--cyan); border-color: var(--cyan); box-shadow: 0 0 8px rgba(90, 215, 238, 0.7); }
.note { margin-top: 1.4rem; font-size: 0.78rem; line-height: 1.55; color: var(--muted); }

/* ---------- Why this estimate? ---------- */
.explain {
margin-top: 1rem;
padding: 1.6rem 1.7rem 1.4rem;
border-radius: 24px;
border: 1px solid var(--line);
background: linear-gradient(165deg, var(--glass-2), var(--glass));
backdrop-filter: blur(18px);
-webkit-backdrop-filter: blur(18px);
box-shadow: inset 0 1px 0 rgba(255, 255, 255, 0.05), 0 30px 60px -40px #000;
}
.explain-head {
display: flex;
align-items: baseline;
justify-content: space-between;
gap: 1rem;
}
.explain-head h3 {
font-family: var(--display);
font-weight: 400;
font-size: 1.5rem;
letter-spacing: -0.01em;
margin: 0;
padding: 0;
}
                                    .explain-head span {
                                      font-size: 0.68rem;
                                        font-weight: 600;
                                          letter-spacing: 0.16em;
                                            color: #b2b6e8;
                                              white-space: nowrap;
                                              }
                                              .explain-sub {
                                                color: var(--muted);
                                                  font-size: 0.86rem;
                                                    line-height: 1.55;
                                                      margin: 0.55rem 0 0;
                                                        max-width: 30rem;
                                                        }
                                                        .fgroup { margin-top: 1.3rem; padding-top: 1.1rem; border-top: 1px solid var(--line); }
                                                        .fgroup-label {
                                                          display: flex;
                                                            align-items: center;
                                                              gap: 0.5rem;
                                                                margin-bottom: 0.85rem;
                                                                  font-size: 0.68rem;
                                                                    font-weight: 600;
                                                                      letter-spacing: 0.16em;
                                                                        color: var(--muted);
                                                                        }
                                                                        .fgroup-label::before {
                                                                          content: "";
                                                                            width: 6px;
                                                                              height: 6px;
                                                                                border-radius: 50%;
                                                                                }
                                                                                .fgroup-label.up::before { background: var(--cyan); box-shadow: 0 0 8px rgba(90, 215, 238, 0.6); }
                                                                                .fgroup-label.down::before { background: var(--accent); box-shadow: 0 0 8px rgba(124, 131, 255, 0.6); }
                                                                                .factor { margin-bottom: 0.8rem; }
                                                                                .factor:last-child { margin-bottom: 0; }
                                                                                .factor-top {
                                                                                  display: flex;
                                                                                    align-items: baseline;
                                                                                      justify-content: space-between;
                                                                                        gap: 1rem;
                                                                                          margin-bottom: 0.4rem;
                                                                                          }
                                                                                          .factor-name {
                                                                                            min-width: 0;
                                                                                              overflow: hidden;
                                                                                                text-overflow: ellipsis;
                                                                                                  white-space: nowrap;
                                                                                                    font-size: 0.92rem;
                                                                                                      color: var(--text);
                                                                                                      }
                                                                                                      .factor-value {
                                                                                                        flex-shrink: 0;
                                                                                                          font-size: 0.88rem;
                                                                                                            font-weight: 500;
                                                                                                              font-variant-numeric: tabular-nums;
                                                                                                              }
                                                                                                              .factor-value.up { color: var(--cyan); }
                                                                                                              .factor-value.down { color: #a9aeff; }
                                                                                                              .factor-track {
                                                                                                                height: 5px;
                                                                                                                  border-radius: 999px;
                                                                                                                    background: rgba(255, 255, 255, 0.06);
                                                                                                                      overflow: hidden;
                                                                                                                      }
                                                                                                                      .factor-bar { height: 100%; border-radius: 999px; }
                                                                                                                      .factor-bar.up { background: linear-gradient(90deg, rgba(90, 215, 238, 0.25), var(--cyan)); }
                                                                                                                      .factor-bar.down { background: linear-gradient(90deg, rgba(124, 131, 255, 0.25), var(--accent)); }
                                                                                                                      .reference {
                                                                                                                        display: grid;
                                                                                                                          grid-template-columns: 1fr 1fr auto;
                                                                                                                            align-items: end;
                                                                                                                              gap: 1rem 1.4rem;
                                                                                                                                margin-top: 1.4rem;
                                                                                                                                  padding-top: 1.2rem;
                                                                                                                                    border-top: 1px solid var(--line-2);
                                                                                                                                    }
                                                                                                                                    .reference small {
                                                                                                                                      display: block;
                                                                                                                                        font-size: 0.66rem;
                                                                                                                                          font-weight: 600;
                                                                                                                                            letter-spacing: 0.16em;
                                                                                                                                              color: var(--muted);
                                                                                                                                                margin-bottom: 0.3rem;
                                                                                                                                                }
                                                                                                                                                .reference strong {
                                                                                                                                                  font-family: var(--display);
                                                                                                                                                    font-weight: 400;
                                                                                                                                                      font-size: 1.55rem;
                                                                                                                                                        letter-spacing: -0.01em;
                                                                                                                                                          color: var(--muted);
                                                                                                                                                          }
                                                                                                                                                          .reference .ref-you strong { color: var(--text); }
                                                                                                                                                          .reference .ref-delta {
                                                                                                                                                            padding: 0.3rem 0.75rem;
                                                                                                                                                              border-radius: 999px;
                                                                                                                                                                border: 1px solid rgba(124, 131, 255, 0.4);
                                                                                                                                                                  background: var(--accent-soft);
                                                                                                                                                                    font-size: 0.8rem;
                                                                                                                                                                      font-weight: 600;
                                                                                                                                                                        color: #fff;
                                                                                                                                                                          white-space: nowrap;
                                                                                                                                                                          }
                                                                                                                                                                          .explain-empty { margin: 1.2rem 0 0; font-size: 0.86rem; color: var(--muted); }
                                                                                                                                                                          @media (max-width: 860px) {
                                                                                                                                                                            .explain { padding: 1.3rem 1.1rem 1.1rem; }
                                                                                                                                                                              .explain-head { flex-direction: column; gap: 0.3rem; }
                                                                                                                                                                                .reference { grid-template-columns: 1fr 1fr; }
                                                                                                                                                                                  .reference .ref-delta { grid-column: 1 / -1; justify-self: start; }
                                                                                                                                                                                  }
/* ---------- How it works + project info ---------- */
.band { margin-top: 4rem; }
.band h3 { font-family: var(--display); font-weight: 400; font-size: 1.7rem; margin: 0 0 1.4rem; padding: 0; }
.flow { display: grid; grid-template-columns: repeat(3, 1fr); gap: 2rem; }
.node { position: relative; padding-top: 1.5rem; border-top: 1px solid var(--line-2); }
.node::before {
    content: ""; position: absolute; top: -4px; left: 0; width: 7px; height: 7px; border-radius: 50%;
    background: var(--accent); box-shadow: 0 0 10px var(--accent);
}
.node b { display: block; font-weight: 600; font-size: 0.98rem; margin-bottom: 0.4rem; }
.node span { color: var(--muted); font-size: 0.9rem; line-height: 1.55; }
.meta {
    display: flex; flex-wrap: wrap; gap: 0.6rem 2.6rem; margin-top: 3.6rem; padding-top: 1.3rem;
    border-top: 1px solid var(--line); font-size: 0.84rem; color: var(--muted);
}
.meta b { color: var(--text); font-weight: 500; margin-left: 0.4rem; }

@media (max-width: 860px) {
    .block-container { padding: 1.2rem 1rem 2.4rem !important; }
    .hero { padding-bottom: 1.8rem; }
    .est { min-height: 0; padding: 2rem 1.4rem 1.7rem; }
    .specs { grid-template-columns: repeat(2, 1fr); }
    .flow { grid-template-columns: 1fr; gap: 1.6rem; }
    .meta { flex-direction: column; gap: 0.5rem; }
}
@media (prefers-reduced-motion: reduce) { *, *::before, *::after { animation: none !important; transition: none !important; } }
</style>
"""
st.markdown(CSS, unsafe_allow_html=True)


# ----------------------------------------------------------------------------
# Display helpers (never touch the model or its inputs)
# ----------------------------------------------------------------------------
def format_inr(value) -> str:
    """Indian digit grouping, e.g. 12,34,567."""
    number = int(round(float(value)))
    sign = "-" if number < 0 else ""
    digits = str(abs(number))
    if len(digits) > 3:
        head, tail = digits[:-3], digits[-3:]
        groups = []
        while len(head) > 2:
            groups.insert(0, head[-2:])
            head = head[:-2]
        if head:
            groups.insert(0, head)
        digits = ",".join(groups + [tail])
    return f"{sign}{digits}"


def spec(label, value) -> str:
    return f"<div><small>{html.escape(label)}</small><strong>{html.escape(str(value))}</strong></div>"


# ----------------------------------------------------------------------------
# Hero
# ----------------------------------------------------------------------------
st.markdown(
    """
<div class="hero">
<span class="badge"><i></i>MACHINE LEARNING MODEL</span>
<h1>House Price<br>Predictor</h1>
<p>Estimate a property's value using machine learning.</p>
</div>
""",
    unsafe_allow_html=True,
)


# ----------------------------------------------------------------------------
# Main experience
# ----------------------------------------------------------------------------
left, right = st.columns([1.05, 1], gap="large")

with left:
    try:
        card = st.container(border=True, key="props_card")
    except TypeError:  # older Streamlit without `key`
        card = st.container(border=True)

    with card:
        st.markdown(
            '<div class="sec"><h3>Property details</h3><span>Size and layout</span></div>',
            unsafe_allow_html=True,
        )

        col1, col2 = st.columns(2)

        with col1:
            area = st.number_input(
                "Area (sq ft)",
                min_value=300,
                max_value=20000,
                value=5000,
                step=100,
            )

            bedrooms = st.number_input(
                "Bedrooms",
                min_value=1,
                max_value=10,
                value=3,
                step=1,
            )

            bathrooms = st.number_input(
                "Bathrooms",
                min_value=1,
                max_value=10,
                value=2,
                step=1,
            )

        with col2:
            stories = st.number_input(
                "Stories",
                min_value=1,
                max_value=10,
                value=2,
                step=1,
            )

            parking = st.number_input(
                "Parking Spaces",
                min_value=0,
                max_value=5,
                value=2,
                step=1,
            )

            furnishingstatus = st.selectbox(
                "Furnishing Status",
                ["furnished", "semi-furnished", "unfurnished"],
                format_func=lambda v: v.capitalize(),
            )

        st.markdown('<div class="rule"></div>', unsafe_allow_html=True)
        st.markdown(
            '<div class="sec"><h3>Inclusions</h3><span>What the property has</span></div>',
            unsafe_allow_html=True,
        )

        col1, col2 = st.columns(2)

        with col1:
            mainroad = st.radio("Main Road", ["yes", "no"], horizontal=True, format_func=str.capitalize)
            guestroom = st.radio("Guest Room", ["yes", "no"], horizontal=True, format_func=str.capitalize)
            basement = st.radio("Basement", ["yes", "no"], horizontal=True, format_func=str.capitalize)

        with col2:
            hotwaterheating = st.radio(
                "Hot Water Heating",
                ["yes", "no"],
                horizontal=True,
                format_func=str.capitalize,
            )
            airconditioning = st.radio(
                "Air Conditioning",
                ["yes", "no"],
                horizontal=True,
                format_func=str.capitalize,
            )
            prefarea = st.radio("Preferred Area", ["yes", "no"], horizontal=True, format_func=str.capitalize)

        input_data = pd.DataFrame({
            "area": [area],
            "bedrooms": [bedrooms],
            "bathrooms": [bathrooms],
            "stories": [stories],
            "mainroad": [mainroad],
            "guestroom": [guestroom],
            "basement": [basement],
            "hotwaterheating": [hotwaterheating],
            "airconditioning": [airconditioning],
            "parking": [parking],
            "prefarea": [prefarea],
            "furnishingstatus": [furnishingstatus],
        })

        st.markdown('<div style="height:0.5rem"></div>', unsafe_allow_html=True)
        predict_clicked = st.button(
            "Predict House Price →",
            type="primary",
            use_container_width=True,
        )

# Prediction logic and error handling are unchanged. The inputs used are kept
# alongside the result so the summary always matches the estimate shown.
if predict_clicked:
    try:
        prediction = predict_price(model, input_data)[0]
        explanation = explain_prediction(model, input_data, REFERENCE_DATA.drop(columns=["price"]))
        st.session_state["result"] = {
            "price": prediction,
            "area": area,
            "bedrooms": bedrooms,
            "bathrooms": bathrooms,
            "stories": stories,
            "parking": parking,
            "furnishingstatus": furnishingstatus,
            "mainroad": mainroad,
            "guestroom": guestroom,
            "basement": basement,
            "hotwaterheating": hotwaterheating,
            "airconditioning": airconditioning,
            "prefarea": prefarea,
            "explanation": explanation,
        }
        st.session_state["failed"] = False
    except Exception:
        st.session_state["result"] = None
        st.session_state["failed"] = True

with right:
    result = st.session_state.get("result")
    failed = st.session_state.get("failed", False)

    if failed:
        st.markdown(
            """
<div class="est err">
<div class="label">PREDICTION UNAVAILABLE</div>
<div class="lead">Unable to generate a prediction.</div>
<p class="sub">Please check the entered values and try again.</p>
</div>
""",
            unsafe_allow_html=True,
        )
    elif result:
        specs = "".join([
            spec("Area", f"{result['area']:,} sq ft"),
            spec("Bedrooms", result["bedrooms"]),
            spec("Bathrooms", result["bathrooms"]),
            spec("Stories", result["stories"]),
            spec("Parking", result["parking"]),
            spec("Furnishing", result["furnishingstatus"].capitalize()),
        ])
        features = [
            ("mainroad", "Main road"),
            ("guestroom", "Guest room"),
            ("basement", "Basement"),
            ("hotwaterheating", "Hot water heating"),
            ("airconditioning", "Air conditioning"),
            ("prefarea", "Preferred area"),
        ]
        incl = "".join(
            f'<span class="{"on" if result[k] == "yes" else ""}">{label}</span>'
            for k, label in features
        )
        st.markdown(
            f"""
<div class="est">
<div class="label">ESTIMATED HOUSE PRICE</div>
<div class="price"><span class="cur">₹</span>{format_inr(result['price'])}</div>
<div class="sheet">
<div class="specs">{specs}</div>
<div class="incl">{incl}</div>
<div class="note">This estimate is generated by the trained machine learning model and should not be treated as a guaranteed market price.</div>
</div>
</div>
""",
            unsafe_allow_html=True,
        )

        explanation = result.get("explanation")
        if explanation:
            non_zero = [ item for item in explanation["contributions"]
                if abs(item["contribution"]) >= 1000
            ]                                                                                                                                                                 
            factors = non_zero[:6]
            max_abs = max((abs(item["contribution"]) for item in factors), default=1)
            def lakh(value):
                return f"₹{abs(float(value)) / 100000:,.2f}L"
            def factor_row(item, direction):
                value = item["contribution"]
                sign = "+" if value >= 0 else "−"
                width=max(6,int(abs(value)/max_abs*100))
                return(
                    '<div class="factor">'
                    '<div class="factor-top">'
                    f'<span class="factor-name">{html.escape(str(item["feature"]))}</span>'
                    f'<span class="factor-value {direction}">{sign}{lakh(value)}</span>'
                    '</div>'
                    f'<div class="factor-track"><div class="factor-bar {direction}" style="width:{width}%"></div></div>'
                    '</div>'
                )

            def factor_group(title, items, direction):
                if not items:
                    return ""
                rows = "".join(factor_row(item, direction) for item in items)
                return (
                            '<div class="fgroup">'
                            f'<div class="fgroup-label {direction}">{title}</div>'
                            f'{rows}'
                            '</div>'
                )

            ups = [item for item in factors if item["contribution"] >= 0]
            downs = [item for item in factors if item["contribution"] < 0]

            if factors:
                groups_html = (
                    factor_group("ABOVE TYPICAL", ups, "up")
                    + factor_group("BELOW TYPICAL", downs, "down")
                )
            else:
                groups_html = ('<p class="explain-empty">''No significant feature contributions to display.''</p>')

            reference_price = float(explanation["reference_prediction"])
            delta = float(result["price"]) - reference_price
            delta_sign = "+" if delta >= 0 else ""
            delta_word = "above" if delta >= 0 else "below"

            st.markdown(
                '<div class="explain">'
                '<div class="explain-head"><h3>Why this estimate?</h3><span>MODEL EXPLANATION</span></div>'
                "<p class=\"explain-sub\">These factors had the biggest influence on this property's predicted value compared with a typical property.</p>"
                f'{groups_html}'
                '<div class="reference">'
                f'<div class="ref-typical"><small>TYPICAL PROPERTY</small><strong>₹{format_inr(reference_price)}</strong></div>'
                f'<div class="ref-you"><small>YOUR ESTIMATE</small><strong>₹{format_inr(result["price"])}</strong></div>'
                f'<div class="ref-delta">{delta_sign}{lakh(delta)} {delta_word} typical</div>'
                '</div>'
                '</div>',                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                            unsafe_allow_html=True,
                )        
    else:
        st.markdown(
            """<div class="est">
                <div class="label">READY TO ESTIMATE</div>
                <div class="price ghost"><span class="cur">₹</span>––,––,–––</div>
                <div class="lead">Your estimate appears here.</div>
                <p class="sub">Enter the property details and let the model calculate an estimated value.</p>
                </div>
            """,
            unsafe_allow_html=True,
        )


# ----------------------------------------------------------------------------

# How it works + project info
# ----------------------------------------------------------------------------
st.markdown(
    """
<div class="band">
<h3>How it works</h3>
<div class="flow">
<div class="node"><b>Property details</b><span>You describe the size, layout and inclusions of the property.</span></div>
<div class="node"><b>Machine learning model</b><span>The trained model reads those details and calculates a value.</span></div>
<div class="node"><b>Estimated price</b><span>The predicted price is shown with a summary of your inputs.</span></div>
</div>
</div>
<div class="meta">
<span>Model<b>Existing trained ML pipeline</b></span>
<span>Interface<b>Streamlit</b></span>
<span>Prediction<b>House price estimation</b></span>
</div>
""",
    unsafe_allow_html=True,
)
