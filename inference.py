# inference.py - helper functions for DKT model inference
import numpy as np
import pandas as pd
import os
from tensorflow import keras

# === Paths ===
DATA_DIR = os.path.join("processed")   # adjust if different
MODEL_PATH = os.path.join("models", "dkt_model_jee_adapted_finetuned.keras")
QMAP_PATH = os.path.join(DATA_DIR, "jee_question_index_map.csv")

# globals
_model = None
_q2idx = None
_seq_len = 199   # fixed during training (truncate/pad to this)


# === Load mapping ===
def load_q2idx():
    global _q2idx
    if _q2idx is None:
        df = pd.read_csv(QMAP_PATH)
        _q2idx = dict(zip(df['question_id'].astype(str), df['index'].astype(int)))
    return _q2idx


# === Load model ===
def load_model():
    global _model
    if _model is None:
        _model = keras.models.load_model(MODEL_PATH, compile=False)
    return _model


# === Preprocess sequence ===
def _prepare_sequence(question_ids, correct_flags, q2idx, seq_len=199):
    """
    question_ids: list of question IDs as strings
    correct_flags: list of 0/1 ints, same length
    Returns padded numpy arrays for model input
    """
    # map to indices (unknowns → 0 padding)
    q_idx = [q2idx.get(q, 0) for q in question_ids]
    c_idx = [int(c) for c in correct_flags]

    # truncate if longer than seq_len
    if len(q_idx) > seq_len:
        q_idx = q_idx[-seq_len:]
        c_idx = c_idx[-seq_len:]

    # pad at left (so recent at end)
    pad_len = seq_len - len(q_idx)
    q_arr = [0]*pad_len + q_idx
    c_arr = [0]*pad_len + c_idx

    return np.array([q_arr], dtype="int32"), np.array([c_arr], dtype="float32")


# === Predict function ===
def predict_probs(question_ids, correct_flags, return_full_sequence=False):
    """
    Run inference on a student session.
    - question_ids: list of strings
    - correct_flags: list of ints (0/1)
    - return_full_sequence: if True, return per-step predictions

    Returns dict with:
      step_probs: list of probabilities for each timestep
      next_prob : probability of correctness at next step
    """
    model = load_model()
    q2idx = load_q2idx()

    Q_in, C_in = _prepare_sequence(question_ids, correct_flags, q2idx, seq_len=_seq_len)

    preds = model.predict([Q_in, C_in], verbose=0)[0].reshape(-1)

    # mask out padding
    non_pad = np.where(Q_in[0] > 0)[0]
    if len(non_pad) > 0:
        valid_start = non_pad[0]
    else:
        valid_start = len(Q_in[0])

    step_probs = preds[valid_start:].tolist()
    next_prob = float(step_probs[-1]) if len(step_probs) > 0 else float("nan")

    return {"step_probs": step_probs, "next_prob": next_prob}
