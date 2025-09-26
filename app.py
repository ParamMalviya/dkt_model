# app.py - simple Streamlit front-end for the DKT model
import streamlit as st
import pandas as pd
from inference import predict_probs, load_q2idx, load_model
import numpy as np

st.set_page_config(page_title="DKT — JEE Recommender (Demo)", layout="centered")

st.title("DKT JEE Inference — demo")
st.markdown("Enter a recent student session (chronological). The model returns stepwise probabilities and a next-step probability.")

# Load q2idx for validation / autocompletion
try:
    q2idx = load_q2idx()
    all_qids = sorted(list(q2idx.keys()))
except Exception:
    q2idx = {}
    all_qids = []

with st.sidebar:
    st.header("Model / Info")
    st.write("Model path: `models/dkt_model_jee_adapted_finetuned.keras`")
    st.write("Sequence length:", "199 (training seq_len)")

st.header("Input session")
st.markdown("Provide question IDs (one per line) and correctness flags (0/1) one per line.")
col1, col2 = st.columns(2)

with col1:
    qtext = st.text_area("Question IDs (one per line)", height=220,
                         value="\n".join(all_qids[:10]) if len(all_qids)>0 else "q123\nq345\nq789")
with col2:
    ctext = st.text_area("Correct flags (0 or 1) (one per line)", height=220,
                         value="1\n0\n1")

st.markdown("Or upload a CSV with two columns: `question_id,correct`.")
uploaded = st.file_uploader("Upload CSV (optional)", type=["csv"])
if uploaded:
    df_up = pd.read_csv(uploaded)
    if set(df_up.columns) >= {"question_id","correct"}:
        qlist = df_up['question_id'].astype(str).tolist()
        clist = df_up['correct'].astype(int).tolist()
    else:
        st.error("CSV must contain columns `question_id` and `correct`.")
        qlist, clist = [], []
else:
    qlist = [l.strip() for l in qtext.splitlines() if l.strip()!=""]
    clist = [l.strip() for l in ctext.splitlines() if l.strip()!=""]

# basic input length check
if len(qlist) != len(clist):
    st.warning("Number of question IDs and correctness flags differ. Fix input.")
    st.stop()

if len(qlist) == 0:
    st.info("Enter some question IDs and flags to run inference.")
    st.stop()

# run prediction
st.write("### Run model")
if st.button("Predict"):
    with st.spinner("Loading model & running inference..."):
        try:
            # ensure model is loaded
            load_model()
            out = predict_probs(qlist, [int(x) for x in clist], return_full_sequence=True)
        except Exception as e:
            st.error(f"Error during prediction: {e}")
            raise

    st.success("Done — results below")

    # show results
    step_probs = out["step_probs"]
    next_prob = out["next_prob"]
    st.metric(label="Predicted next-step correctness probability", value=f"{next_prob:.3f}")

    # table of per-step
    rows = []
    for i, (q, c, p) in enumerate(zip(qlist, clist, step_probs)):
        rows.append({"step": i+1, "question_id": q, "correct": int(c), "pred_prob": float(p)})
    df_res = pd.DataFrame(rows)
    st.write("Per-step predictions (most recent last):")
    st.dataframe(df_res)

    # simple chart
    st.line_chart(df_res[["pred_prob"]].rename(columns={"pred_prob":"Pred prob"}))

    # optional: download CSV of results
    csv = df_res.to_csv(index=False).encode()
    st.download_button("Download predictions CSV", data=csv, file_name="dkt_predictions.csv", mime="text/csv")
