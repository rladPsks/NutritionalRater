import pandas as pd
import numpy as np

# Try to import external additives dictionary
try:
    from AdditivesDictionary import risk_dict as ADDITIVE_RISK_DICT
except ImportError:
    ADDITIVE_RISK_DICT = None  # fallback: treat additives as "limited risk"


# ---------- Basic helper ----------

def clip01(x: float) -> float:
    """Clip a value to the [0, 100] interval."""
    return max(0.0, min(100.0, float(x)))


def _quality_label(score: float) -> str:
    """Return a simple qualitative label for a 0–100 score."""
    if score >= 80:
        return "very good"
    if score >= 60:
        return "good"
    if score >= 40:
        return "average"
    return "poor"


# ---------- Nutri-Score helpers ----------

def _pts_by_threshold(x: float, cuts) -> int:
    """Return 0..10 points based on Nutri-Score cutoffs."""
    for i, t in enumerate(cuts):
        if x < t:
            return i
    return 10


def _neg_points(row, is_beverage=False, log=None):
    """Negative Nutri-Score points A (0..40)."""
    kcal  = float(row.get('energy_kcal_100g', 0) or 0)
    sugar = float(row.get('sugars_100g', 0) or 0)
    sat   = float(row.get('saturated_fat_100g', 0) or 0)
    salt  = float(row.get('salt_100g', 0) or 0)
    sodium_mg = salt * 400.0

    kcal_food = [50, 80, 110, 140, 170, 200, 230, 270, 310, 450]
    kcal_bev  = [10, 20, 30, 40, 50, 60, 70, 80, 90, 100]
    kcal_pts = _pts_by_threshold(kcal, kcal_bev if is_beverage else kcal_food)

    sugar_pts = _pts_by_threshold(sugar, [4.5, 9, 13.5, 18, 22.5, 27, 31, 36, 40, 45])
    sat_pts   = _pts_by_threshold(sat,   [1, 2, 3, 4, 5, 6, 7, 8, 9, 10])
    sod_pts   = _pts_by_threshold(sodium_mg, [90, 180, 270, 360, 450, 540, 630, 720, 810, 900])

    if log is not None:
        kind = "beverage" if is_beverage else "food"
        log.append("=== Nutrition: how the nutrients impact your score ===")
        log.append(f"We detect this product as a {kind}.")
        log.append("For each nutrient, the penalty goes from 0/10 (no problem) "
                   "to 10/10 (very high and unhealthy).")
        log.append("")
        log.append("Nutrients that count against the score:")

        log.append(f"  • Energy: {kcal:.1f} kcal per 100 g "
                   f"→ penalty {kcal_pts}/10 (higher is worse).")
        log.append(f"  • Sugars: {sugar:.1f} g per 100 g "
                   f"→ penalty {sugar_pts}/10.")
        log.append(f"  • Saturated fat: {sat:.1f} g per 100 g "
                   f"→ penalty {sat_pts}/10.")
        log.append(f"  • Salt (as sodium): {sodium_mg:.0f} mg per 100 g "
                   f"→ penalty {sod_pts}/10.")
        log.append("")

    return kcal_pts + sugar_pts + sat_pts + sod_pts


def _pos_points(row, is_beverage=False, log=None):
    """Positive Nutri-Score points C (0..10)."""
    fiber = float(row.get('fiber_100g', 0) or 0)
    protein = float(row.get('proteins_100g', 0) or 0)

    fiber_pts = _pts_by_threshold(fiber, [1, 2, 2.8, 3.4, 3.9, 99])

    if is_beverage:
        if log is not None:
            log.append("Good nutrients that help the score:")
            log.append(f"  • Fiber: {fiber:.1f} g per 100 g "
                       f"→ bonus {fiber_pts}/5 (higher is better).")
            log.append("  • Protein is not considered for drinks.")
        return fiber_pts

    protein_pts = _pts_by_threshold(protein, [1.6, 3.2, 4.8, 6.4, 8, 99])

    if log is not None:
        log.append("Good nutrients that help the score:")
        log.append(f"  • Fiber: {fiber:.1f} g per 100 g "
                   f"→ bonus {fiber_pts}/5.")
        log.append(f"  • Protein: {protein:.1f} g per 100 g "
                   f"→ bonus {protein_pts}/5.")
        log.append("")

    return fiber_pts + protein_pts


def score_nutrition(row, verbose=False):
    """Compute Q_nutri ∈ [0,100] with optional user-friendly explanation."""
    log = [] if verbose else None

    # Detect beverage or bread classification from category tags
    tags_raw = (str(row.get('categories_tags') or '') + ' ' +
                str(row.get('source_category') or '')).lower()

    
    for sep in [",", ";", "|", ">", "/"]:
        tags_raw = tags_raw.replace(sep, " ")
    tokens = [t.strip() for t in tags_raw.split() if t.strip()]

    bev_tokens = {"beverages", "beverage", "soft-drinks", "juices", "sodas", "drinks"}
    bread_tokens = {"bread", "breads", "bakery", "baked-goods"}

    is_bev = any(t in bev_tokens for t in tokens)
    # Override beverage detection for bread/bakery items
    if any(t in bread_tokens for t in tokens):
        is_bev = False

    A = _neg_points(row, is_beverage=is_bev, log=log)
    C = _pos_points(row, is_beverage=is_bev, log=log)
    R = A - C
    q = clip01(100 * (40 - R) / 50)

    if log is not None:
        log.append(f"In total, the \"unhealthy\" part (energy, sugar, saturated fat, salt)")
        log.append(f"adds up to {A} penalty points, and the good nutrients (fiber, protein)")
        log.append(f"recover {C} points.")
        log.append("")
        log.append(f"As a result, the nutrition score for this product is "
                   f"{q:.1f} out of 100 (higher is better).")
        log.append(f"Nutritional quality: {_quality_label(q)}.")
        print("\n".join(log))

    return q

# ---------- Additives score (dictionary-based) ----------

def score_additives(row, verbose=False):
    """
    Computes Q_add with exponential penalty using the dictionary
    from AdditivesDictionary.py.

    Risk levels:
        'no risk'      -> severity 0
        'limited risk' -> severity 1
        'high risk'    -> severity 3

    If no dictionary is found: treat all additives as severity 1.
    """
    log = [] if verbose else None
    add_n = int(pd.to_numeric(row.get("additives_n", 0), errors="coerce") or 0)

    # No additives at all → perfect score and positive feedback
    if add_n == 0:
        if log is not None:
            log.append("=== Additives: how extra ingredients impact your score ===")
            log.append("This product does not list any additives.")
            log.append("That is very positive: the additives score is 100/100.")
            print("\n".join(log))
        return 100.0

    dict_to_use = ADDITIVE_RISK_DICT
    tags = (str(row.get("additives_tags") or "") + " " +
            str(row.get("additives") or "")).lower()

    tokens = tags.replace(",", " ").split()
    codes = [tok for tok in tokens if tok.startswith(("e", "ins", "di-", "diphosphate", "potassium"))]

    severity_map = {
        "no risk": 0,
        "limited risk": 1,
        "high risk": 3,
    }

    severities = []
    num_no_risk = 0
    num_limited = 0
    num_high = 0

    if log is not None:
        log.append("=== Additives: how extra ingredients impact your score ===")
        log.append(f"This product declares {add_n} additive(s).")

    if dict_to_use:
        for c in codes:
            risk = dict_to_use.get(c.lower())
            if risk is None:
                sev = 1
                label = "not in our list → treated as limited risk"
                num_limited += 1
            else:
                sev = severity_map.get(risk, 1)
                label = risk
                if risk == "no risk":
                    num_no_risk += 1
                elif risk == "limited risk":
                    num_limited += 1
                elif risk == "high risk":
                    num_high += 1

            severities.append(sev)
            if log is not None:
                log.append(f"  • {c.upper()}: {label} (severity {sev}).")
    else:
        # No dictionary → all additives = severity 1
        severities = [1] * add_n
        num_limited = add_n
        if log is not None:
            log.append("We do not have a detailed dictionary of additives,")
            log.append("so each additive is treated as limited risk (severity 1).")

    S = sum(severities)
    lam = 0.25
    q = clip01(100 * np.exp(-lam * S))

    if log is not None:
        log.append("")
        if num_high == 0 and S == 0:
            log.append("All additives present are classified as 'no risk',")
            log.append("so they do not reduce the score at all.")
        elif num_high == 0 and S <= 2:
            log.append("The additives are mostly low risk and in small amount,")
            log.append("so their impact on the score is very small.")
        elif num_high == 0:
            log.append("There are some additives with limited risk; they slightly reduce the score.")
        else:
            log.append(f"There {'is' if num_high == 1 else 'are'} {num_high} high-risk additive(s),")
            log.append("which strongly reduces the additives score.")

        log.append("")
        log.append(f"Taking all this into account, the additives score is {q:.1f} out of 100.")
        log.append(f"Additives quality: {_quality_label(q)}.")
        print("\n".join(log))

    return q


# ---------- Final health score (MAIN FUNCTION) ----------

def health_score(row, verbose=False):
    """
    Global health score ∈ [0,100].
    This is the main function to be called from the application.
    """
    if verbose:
        print("======================================")
        print("Health check for this product")
        print("======================================")
        print(f"Product: {row.get('product_name', 'Unknown product')}\n")
        print("1) We analyse the nutritional quality (nutrients).")

    q_nutri = score_nutrition(row, verbose=verbose)

    if verbose:
        print("\n2) We analyse the additives.")
    q_add = score_additives(row, verbose=verbose)

    rating = round(0.60 * q_nutri + 0.40 * q_add)
    label = _quality_label(rating)

    if verbose:
        print("\n3) Final result:")
        print(f"   - Nutrition score (60%): {q_nutri:.1f}")
        print(f"   - Additives score (40%): {q_add:.1f}")
        print(f"→ Overall health score: {rating} / 100 "
              f"({label} overall quality).")
        print("======================================\n")

    return rating


# ---------- Optional DataFrame helper ----------

def score_dataframe(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    out["Q_nutri"] = out.apply(lambda r: score_nutrition(r, verbose=False), axis=1)
    out["Q_add"]   = out.apply(lambda r: score_additives(r, verbose=False), axis=1)
    out["health_score"] = out.apply(lambda r: health_score(r, verbose=False), axis=1)
    out["rating"] = out["health_score"]
    return out


# ---------- Local testing ----------

if __name__ == "__main__":
    from recommend_alternatives import recommend_alternatives

    df = pd.read_csv("data/merged_products.csv")
    scored = score_dataframe(df)

    example = scored.iloc[0]
    health_score(example, verbose=True)

    print("\n=== Healthier alternatives in the same category ===")
    alts = recommend_alternatives(scored, example["product_name"], top_k=5)
    print(alts)
