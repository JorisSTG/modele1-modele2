import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import norm

# ---- STYLE sombre pour se fondre avec le thème Streamlit ----
plt.style.use("dark_background")
plt.rcParams.update({
    "figure.facecolor": "none",
    "axes.facecolor": "none",
    "savefig.facecolor": "none",
    "axes.edgecolor": "#FFFFFF",
    "axes.labelcolor": "#FFFFFF",
    "xtick.color": "#DDDDDD",
    "ytick.color": "#DDDDDD",
    "text.color": "#FFFFFF",
})

st.title("Comparaison multisource")
st.markdown(
    """
    L’objectif de cette application est d’évaluer la précision et la cohérence entre deux jeux de données météorologie (température uniquement) à des fins de simulations STD.
    """,
    unsafe_allow_html=True
)

# -------- Paramètres --------

heures_par_mois = [744, 672, 744, 720, 744, 720, 744, 744, 720, 744, 720, 744]
percentiles_list = [10, 25, 50, 75, 90]
couleur_modele =  "goldenrod"
couleur_TRACC = "lightgray"
vmaxT = 5
vminT = -5
vmaxP = 100
vminP = 50
vmaxH = 100
vminH = -100
vmaxDJU = 150
vminDJU = -150

# -------- Noms des mois --------
mois_noms = {
    1: "01 - Janvier", 2: "02 - Février", 3: "03 - Mars",
    4: "04 - Avril", 5: "05 - Mai", 6: "06 - Juin",
    7: "07 - Juillet", 8: "08 - Août", 9: "09 - Septembre",
    10: "10 - Octobre", 11: "11 - Novembre", 12: "12 - Décembre"
}

# -------- Upload des fichiers CSV --------
uploaded_model1 = st.file_uploader("Déposer le fichier CSV de la SOURCE 1 (colonne unique T°C) :", type=["csv"])
uploaded_model2 = st.file_uploader("Déposer le fichier CSV de la SOURCE 2 (colonne unique T°C) :", type=["csv"])

if uploaded_model1 and uploaded_model2:
    st.markdown("")

    # Use uploaded filenames as labels in plots and tables
    label_mod = uploaded_model1.name.replace(".csv", "")
    label_obs = uploaded_model2.name.replace(".csv", "")

    # -------- Lecture des fichiers CSV --------
    model_values = pd.read_csv(uploaded_model1, header=0).iloc[:, 0].values
    obs_series = pd.read_csv(uploaded_model2, header=0).iloc[:, 0].values

    # -------- Création de df_obs (pour compatibilité avec la suite du code) --------
    df_obs = pd.DataFrame({"T2m": obs_series})
    df_obs["year"] = 2023  # Année fictive pour compatibilité
    df_obs["month_num"] = pd.concat([pd.Series([m] * h) for m, h in enumerate(heures_par_mois, start=1)], ignore_index=True)
    df_obs["month"] = df_obs["month_num"].map(mois_noms)
    df_obs["day"] = pd.concat([pd.Series(range(1, h // 24 + 2)) for h in heures_par_mois], ignore_index=True)[:len(obs_series)]

    # -------- RMSE --------
    def rmse(a, b):
        min_len = min(len(a), len(b))
        a_sorted = np.sort(a[:min_len])
        b_sorted = np.sort(b[:min_len])
        return np.sqrt(np.nanmean((a_sorted - b_sorted) ** 2))

    # -------- Nouvelle fonction : indice de recouvrement --------
    def precision_overlap(a, b, bin_width=1.0):
        """
        Calcule l'indice de recouvrement (%) entre deux séries de données.
        bin_width : largeur des tranches pour l'histogramme (en °C)
        """
        if len(a) == 0 or len(b) == 0:
            return np.nan
    
        # Définir les bornes de l'histogramme
        min_val = min(np.min(a), np.min(b))
        max_val = max(np.max(a), np.max(b))
        bins = np.arange(min_val, max_val + bin_width, bin_width)
    
        # Calcul des histogrammes normalisés
        hist_a, _ = np.histogram(a, bins=bins, density=True)
        hist_b, _ = np.histogram(b, bins=bins, density=True)
    
        # Indice de recouvrement
        overlap = np.sum(np.minimum(hist_a, hist_b) * bin_width)
        indice_percent = overlap * 100
        return round(indice_percent, 2)

    # -------- Boucle sur les mois --------
    results_rmse = []
    obs_mois_all = []
    start_idx_model = 0

    for mois_num, nb_heures in enumerate(heures_par_mois, start=1):
        mois = mois_noms[mois_num]
        mod_mois = model_values[start_idx_model:start_idx_model + nb_heures]
        obs_mois_vals = df_obs[df_obs["month_num"] == mois_num]["T2m"].values
        obs_mois_all.append(obs_mois_vals)
        val_rmse = rmse(mod_mois, obs_mois_vals)
        pct_precision = precision_overlap(mod_mois, obs_mois_vals)
        results_rmse.append({
            "Mois": mois,
            "RMSE (°C)": round(val_rmse, 2),
            "Précision percentile (%)": pct_precision
        })
        start_idx_model += nb_heures

    # -------- DataFrame final --------
    df_rmse = pd.DataFrame(results_rmse)
    df_rmse_styled = (
        df_rmse.style
        .background_gradient(subset=["Précision percentile (%)"], cmap="RdYlGn", vmin=vminP, vmax=vmaxP, axis=None)
        .format({"Précision percentile (%)": "{:.2f}", "RMSE (°C)": "{:.2f}"})
    )

    st.subheader(f"Précision de {label_mod} par rapport à {label_obs} : RMSE et précision via écarts des percentiles")
    st.dataframe(df_rmse_styled, hide_index=True)

    # -------- Précision globale annuelle --------
    model_annee = model_values[:sum(heures_par_mois)]
    obs_annee = np.concatenate(obs_mois_all)
    
    precision_annuelle = precision_overlap(model_annee, obs_annee)
    st.subheader(f"Précision globale annuelle : {precision_annuelle} %")

    # -------- Suite de votre code --------
    t_sup_thresholds = st.text_input("Seuils Tmax supérieur (°C, séparés par des / )", "25/30")
    t_inf_thresholds = st.text_input("Seuils Tmin inférieur (°C, séparés par des / )", "0/5")
    t_sup_thresholds_list = [int(float(x.strip())) for x in t_sup_thresholds.split("/")]
    t_inf_thresholds_list = [int(float(x.strip())) for x in t_inf_thresholds.split("/")]

    stats_sup = []
    stats_inf = []

    for mois_num, nb_heures in enumerate(heures_par_mois, start=1):
        mois = mois_noms[mois_num]
        idx0 = sum(heures_par_mois[:mois_num-1])
        idx1 = sum(heures_par_mois[:mois_num])
        mod_mois = model_values[idx0:idx1]
        obs_mois = obs_mois_all[mois_num-1]

        # Seuils supérieurs
        for seuil in t_sup_thresholds_list:
            heures_obs = np.sum(obs_mois > seuil)
            nb_heures_mod = np.sum(mod_mois > seuil)
            ecart = nb_heures_mod - heures_obs
            stats_sup.append({
                "Mois": mois,
                "Seuil (°C)": f"{seuil}",
                f"Heures {label_mod}": nb_heures_mod,
                f"Heures {label_obs}": heures_obs,
                f"Ecart ({label_mod} - {label_obs})": ecart
            })

        # Seuils inférieurs
        for seuil in t_inf_thresholds_list:
            heures_obs = np.sum(obs_mois < seuil)
            nb_heures_mod = np.sum(mod_mois < seuil)
            ecart = nb_heures_mod - heures_obs
            stats_inf.append({
                "Mois": mois,
                "Seuil (°C)": f"{seuil}",
                f"Heures {label_mod}": nb_heures_mod,
                f"Heures {label_obs}": heures_obs,
                f"Ecart ({label_mod} - {label_obs})": ecart
            })

    # Création des DataFrames
    df_sup = pd.DataFrame(stats_sup)
    df_inf = pd.DataFrame(stats_inf)

    # Style : seuils supérieurs → rouge = plus chaud
    df_sup_styled = (
        df_sup.style
        .background_gradient(subset=[f"Ecart ({label_mod} - {label_obs})"], cmap="bwr", vmin=vminH, vmax=vmaxH, axis=None)
    )
    st.subheader("Nombre d'heures supérieur au(x) seuil(s)")
    st.dataframe(df_sup_styled, hide_index=True)

    # Style : seuils inférieurs → rouge = plus froid
    df_inf_styled = (
        df_inf.style
        .background_gradient(subset=[f"Ecart ({label_mod} - {label_obs})"], cmap="bwr_r", vmin=vminH, vmax=vmaxH, axis=None)
    )
    st.subheader("Nombre d'heures inférieur au(x) seuil(s)")
    st.dataframe(df_inf_styled, hide_index=True)

    # =====================================
    # ======= SOMMES ANNUELLES =============
    # =====================================
    
    obs_all = np.concatenate(obs_mois_all)
    mod_all = np.array(model_values)
    
    annual_sup = []
    annual_inf = []
    
    # ----- Supérieurs -----
    for seuil in t_sup_thresholds_list:
        heures_obs = np.sum(obs_all > seuil)
        heures_mod = np.sum(mod_all > seuil)
        ecart = heures_mod - heures_obs
    
        annual_sup.append({
            "Période": "Année",
            "Seuil (°C)": f"{seuil}",
            f"Heures {label_mod}": int(heures_mod),
            f"Heures {label_obs}": int(heures_obs),
            f"Ecart ({label_mod} - {label_obs})": int(ecart)
        })
    
    # ----- Inférieurs -----
    for seuil in t_inf_thresholds_list:
        heures_obs = np.sum(obs_all < seuil)
        heures_mod = np.sum(mod_all < seuil)
        ecart = heures_mod - heures_obs
    
        annual_inf.append({
            "Période": "Année",
            "Seuil (°C)": f"{seuil}",
            f"Heures {label_mod}": int(heures_mod),
            f"Heures {label_obs}": int(heures_obs),
            f"Ecart ({label_mod} - {label_obs})": int(ecart)
        })
    
    df_sup_year = pd.DataFrame(annual_sup)
    df_inf_year = pd.DataFrame(annual_inf)
    
    # =====================================
    # ======= AFFICHAGE ANNUEL =============
    # =====================================
    
    st.subheader("Somme annuelle — Nombre d'heures supérieur au(x) seuil(s)")
    df_sup_year_styled = (
        df_sup_year.style
        .background_gradient(subset=[f"Ecart ({label_mod} - {label_obs})"], cmap="bwr", vmin=vminH*12, vmax=vmaxH*12, axis=None)
    )
    st.dataframe(df_sup_year_styled, hide_index=True)
    
    st.subheader("Somme annuelle — Nombre d'heures inférieur au(x) seuil(s)")
    df_inf_year_styled = (
        df_inf_year.style
        .background_gradient(subset=[f"Ecart ({label_mod} - {label_obs})"], cmap="bwr_r", vmin=vminH*12, vmax=vmaxH*12, axis=None)
    )
    st.dataframe(df_inf_year_styled, hide_index=True)

    # -------- Histogrammes par plage de température --------
    st.subheader(f"Histogrammes horaire : {label_mod} et {label_obs}")
    st.markdown(
        """
        La valeur de chaque barre est égale au total d'heures compris entre [ X°C , X+1°C [
        """,
        unsafe_allow_html=True
    )

    # Bins correspondant à [X, X+1[ pour chaque température entière
    bin_edges = bins = np.arange(-5, 46, 1)
    bin_labels = bin_edges[:-1].astype(int)

    def count_hours_in_bins(temp_hourly, bins):
        counts, _ = np.histogram(temp_hourly, bins=bins)
        return counts

    for mois_num in range(1, 13):
        mois = mois_noms[mois_num]

        # Observations
        obs_hourly = obs_mois_all[mois_num-1]
        obs_counts = count_hours_in_bins(obs_hourly, bin_edges)

        # modèle
        idx0 = sum(heures_par_mois[:mois_num-1])
        idx1 = sum(heures_par_mois[:mois_num])
        mod_hourly = model_values[idx0:idx1]
        mod_counts = count_hours_in_bins(mod_hourly, bin_edges)

        # Préparer le DataFrame pour le plot
        df_plot = pd.DataFrame({
            "Temp_Num": bin_labels,
            "Température": bin_labels.astype(str),
            label_obs: obs_counts,
            label_mod: mod_counts
        }).sort_values("Temp_Num")

        # Création du plot
        fig, ax = plt.subplots(figsize=(14, 4))
        ax.bar(df_plot["Temp_Num"] - 0.2, df_plot[label_obs], width=0.45, label=label_obs, color=couleur_TRACC)
        ax.bar(df_plot["Temp_Num"] + 0.2, df_plot[label_mod], width=0.45, label=label_mod, color=couleur_modele)
        ax.set_title(f"{mois} - Durée en heure par seuil de température")
        ax.set_xlabel("Température (°C)")
        ax.set_ylabel("Durée en heure")
        ax.legend()
        st.pyplot(fig)
        plt.close(fig)

    # -------- Histogramme annuel par plage de température --------
    st.subheader(f"Histogramme annuel : {label_mod} et {label_obs}")
    st.markdown(
        """
        La valeur de chaque barre est égale au total d'heures compris entre [ X°C , X+1°C [
        sur l'année entière.
        """,
        unsafe_allow_html=True
    )

    # Bins correspondant à [X, X+1[
    bin_edges = np.arange(-5, 46, 1)
    bin_labels = bin_edges[:-1].astype(int)

    # -------- Regroupement ANNUEL --------
    # Observations : concaténer tous les mois
    obs_hourly_annual = np.concatenate(obs_mois_all)

    # modèle : toutes les valeurs de l'année
    mod_hourly_annual = model_values

    # Comptages annuels
    obs_counts_annual = count_hours_in_bins(obs_hourly_annual, bin_edges)
    mod_counts_annual = count_hours_in_bins(mod_hourly_annual, bin_edges)
    diff_counts_annual_obs = np.maximum(0, obs_counts_annual - mod_counts_annual)
    diff_counts_annual_mod = np.maximum(0, mod_counts_annual - obs_counts_annual)

    # Préparer DataFrame pour le plot
    df_plot_year = pd.DataFrame({
        "Temp_Num": bin_labels,
        "Température": bin_labels.astype(str),
        label_obs: obs_counts_annual,
        label_mod: mod_counts_annual
    }).sort_values("Temp_Num")

    # Plot
    fig, ax = plt.subplots(figsize=(16, 5))
    ax.bar(df_plot_year["Temp_Num"] - 0.2, df_plot_year[label_obs], width=0.45,
           label=label_obs, color=couleur_TRACC)
    ax.bar(df_plot_year["Temp_Num"] + 0.2, df_plot_year[label_mod], width=0.45,
           label=label_mod, color=couleur_modele)
    fig_hist_year = fig
    ax.set_title("Année entière - Durée en heures par seuil de température")
    ax.set_xlabel("Température (°C)")
    ax.set_ylabel("Durée en heure")
    ax.legend()
    st.pyplot(fig)
    plt.close(fig)

    # Préparer DataFrame pour le plot des différences
    df_plot_year_diff = pd.DataFrame({
        "Temp_Num": bin_labels,
        "Température": bin_labels.astype(str),
        f"Différence absolue de {label_mod}": diff_counts_annual_mod,
        f"Différence absolue de {label_obs}": diff_counts_annual_obs
    }).sort_values("Temp_Num")

    # Plot différences
    fig, ax = plt.subplots(figsize=(16, 5))
    ax.bar(df_plot_year_diff["Temp_Num"], df_plot_year_diff[f"Différence absolue de {label_mod}"], width=0.8,
           label=f"Différence : {label_mod} > {label_obs}", color=couleur_modele)

    ax.bar(df_plot_year_diff["Temp_Num"], df_plot_year_diff[f"Différence absolue de {label_obs}"], width=0.8,
           label=f"Différence : {label_mod} < {label_obs}", color=couleur_TRACC)

    ax.set_title("Année entière - Différence en heures par seuil de température")
    ax.set_xlabel("Température (°C)")
    ax.set_ylabel("Durée en heure")
    ax.legend()
    fig_hist_diff = fig
    st.pyplot(fig)
    plt.close(fig)

    st.markdown(
        f"""
        La couleur de la différence est définie ainsi :

        Barres jaunes : {label_mod} compte davantage d’heures que {label_obs} dans cette plage de température.

        Barres blanches : {label_obs} compte davantage d’heures que {label_mod} dans cette plage de température.

        La conclusion dépend donc de l’endroit où se situe cette différence. Une analyse doit être réalisée manuellement : par exemple, si {label_obs} présente plus d’heures dans les plages[...]
        Comme les deux séries possèdent le même nombre total d’heures, un excès d’heures froides dans {label_obs} implique mécaniquement un excès d’heures chaudes dans {label_mod} (et inve[...]
        """,
        unsafe_allow_html=True
    )

    # =============================
    # Comparaison annuelle histogrammes horaires
    # =============================
    
    # Comparaison pour les températures élevées
    tx_seuil_chaud = 25
    heures_obs_chaud = np.sum(obs_hourly_annual > tx_seuil_chaud)
    heures_mod_chaud = np.sum(mod_hourly_annual > tx_seuil_chaud)
    
    if heures_obs_chaud > heures_mod_chaud:
        phrase_tx_chaud = f"{label_obs} a plus d'heures avec une T>{tx_seuil_chaud}°C ({heures_obs_chaud}) que {label_mod} ({heures_mod_chaud})."
    elif heures_obs_chaud < heures_mod_chaud:
        phrase_tx_chaud = f"{label_mod} a plus d'heures avec une T>{tx_seuil_chaud}°C ({heures_mod_chaud}) que {label_obs} ({heures_obs_chaud})."
    else:
        phrase_tx_chaud = f"{label_mod} et {label_obs} ont le même nombre d'heures supérieures à T={tx_seuil_chaud}°C."

    tn_seuil_froid = 5
    heures_obs_froid = np.sum(obs_hourly_annual < tn_seuil_froid)
    heures_mod_froid = np.sum(mod_hourly_annual < tn_seuil_froid)
    
    if heures_obs_froid > heures_mod_froid:
        phrase_tn_froid = f"{label_obs} a plus d'heures avec une T<{tn_seuil_froid}°C ({heures_obs_froid}) que {label_mod} ({heures_mod_froid})."
    elif heures_obs_froid < heures_mod_froid:
        phrase_tn_froid = f"{label_mod} a plus d'heures avec une T<{tn_seuil_froid}°C ({heures_mod_froid}) que {label_obs} ({heures_obs_froid})."
    else:
        phrase_tn_froid = f"{label_mod} et {label_obs} ont le même nombre d'heures inférieures à T={tn_seuil_froid}°C."

    # Stocker dans st.session_state pour la page Résumé
    st.session_state["resume_hist"] = [phrase_tx_chaud, phrase_tn_froid]
    
    # Optionnel : affichage sur la page actuelle
    st.subheader("Résumé comparatif histogrammes horaires/annuels")
    for p in st.session_state["resume_hist"]:
        st.write("- " + p)


    # -------- Précision par créneau horaire --------
    results_temp = []
    def rmse_hours(obs_counts, mod_counts):
        min_len = min(len(obs_counts), len(mod_counts))
        return np.sqrt(np.nanmean((np.array(obs_counts[:min_len]) - np.array(mod_counts[:min_len]))**2))

    for mois_num in range(1, 13):
        mois = mois_noms[mois_num]
        obs_hourly = obs_mois_all[mois_num-1]
        idx0 = sum(heures_par_mois[:mois_num-1])
        idx1 = sum(heures_par_mois[:mois_num])
        mod_hourly = model_values[idx0:idx1]
        obs_counts = count_hours_in_bins(obs_hourly, bins)
        mod_counts = count_hours_in_bins(mod_hourly, bins)
        total_hours = 2*heures_par_mois[mois_num-1]
        hours_error = sum(abs(np.array(obs_counts) - np.array(mod_counts)))
        pct_precision = round(100 * (1 - hours_error / total_hours), 2)
        val_rmse = rmse_hours(obs_counts, mod_counts)
        results_temp.append({
            "Mois": mois,
            "RMSE (heure)": round(val_rmse,2),
            "Précision (%)": pct_precision
        })

    df_temp_precision = pd.DataFrame(results_temp)
    df_temp_precision_styled = df_temp_precision.style \
        .background_gradient(subset=["Précision (%)"], cmap="RdYlGn", vmin=vminP, vmax=vmaxP, axis=None) \
        .format({"Précision (%)": "{:.2f}", "RMSE (heure)": "{:.2f}"})

    st.subheader(f"Précision des sources sur la répartition des durées des plages de température ({label_mod} vs {label_obs})")
    st.markdown(
        """
        Le RMSE correspond à l’écart absolu moyen d'heure pour chaque intervalle de température.
        La précision est calculée à partir de la différence totale d’heures de chaque intervalle 
        """,
        unsafe_allow_html=True
    )
    st.dataframe(df_temp_precision_styled, hide_index=True)

    # ============================
    #   COURBES Tn / Tmoy / Tx
    # ============================
    st.subheader(f"Évolution mensuelle : Tn_mois / Tmoy_mois / Tx_mois ({label_mod} vs {label_obs})")
    st.markdown(
        """  
        - Les valeurs tracées représentent les températures minimales et maximales **absolues** du mois (c’est-à-dire P0 et P100)
        - De ce fait, les températures du mois ne dépassent jamais les bornes définies par Tn_mois et Tx_mois.
        - La température moyenne (Tmoy_mois) correspond à la moyenne mensuelle calculée sur l’ensemble des pas de temps
        """,
        unsafe_allow_html=True
    )
    # Calcul des Tn/Tmoy/Tx pour 12 mois
    results_tstats = []
    for mois_num in range(1, 12+1):
        mois = mois_noms[mois_num]
    
        # Observations
        obs_vals = obs_mois_all[mois_num-1]
        obs_tn = np.min(obs_vals)
        obs_tm = np.mean(obs_vals)
        obs_tx = np.max(obs_vals)
    
        # modèle
        idx0 = sum(heures_par_mois[:mois_num-1])
        idx1 = sum(heures_par_mois[:mois_num])
        mod_vals = model_values[idx0:idx1]
        mod_tn = np.min(mod_vals)
        mod_tm = np.mean(mod_vals)
        mod_tx = np.max(mod_vals)
    
        results_tstats.append({
            "Mois": mois,
            f"{label_obs}_Tn": obs_tn, f"{label_mod}_Tn": mod_tn,
            f"{label_obs}_Tm": obs_tm, f"{label_mod}_Tm": mod_tm,
            f"{label_obs}_Tx": obs_tx, f"{label_mod}_Tx": mod_tx
        })
    
    df_tstats = pd.DataFrame(results_tstats)
    
    # ---- Plot ----
    fig, ax = plt.subplots(figsize=(14,4))

    ax.plot(df_tstats["Mois"], df_tstats[f"{label_mod}_Tx"], color="red", label=f"{label_mod} Tx", linestyle="-")
    ax.plot(df_tstats["Mois"], df_tstats[f"{label_mod}_Tm"], color="white", label=f"{label_mod} Tmoy", linestyle="-")
    ax.plot(df_tstats["Mois"], df_tstats[f"{label_mod}_Tn"], color="cyan", label=f"{label_mod} Tn", linestyle="-")

    ax.plot(df_tstats["Mois"], df_tstats[f"{label_obs}_Tx"], color="red", label=f"{label_obs} Tx", linestyle="--")
    ax.plot(df_tstats["Mois"], df_tstats[f"{label_obs}_Tm"], color="white", label=f"{label_obs} Tmoy", linestyle="--")
    ax.plot(df_tstats["Mois"], df_tstats[f"{label_obs}_Tn"], color="cyan", label=f"{label_obs} Tn", linestyle="--")

    ax.set_title(f"Tn_mois / Tmoy_mois / Tx_mois – {label_mod} vs {label_obs} ")
    ax.set_ylabel("Température (°C)")
    ax.tick_params(axis='x', rotation=45)
    ax.legend(facecolor="black")

    fig_tn_tx_mois = fig
    
    st.pyplot(fig)
    plt.close(fig)
    
    # ---- Tableau correspondant ----
    st.write("Tableau Tn_mois / Tmoy_mois / Tx_mois")
    st.dataframe(df_tstats.round(2), hide_index=True)

    # ---- Tableau des différences (modèle - obs) ----
    df_diff = pd.DataFrame({
        "Mois": df_tstats["Mois"],
        "Diff_Tn_mois": df_tstats[f"{label_mod}_Tn"] - df_tstats[f"{label_obs}_Tn"],
        "Diff_Tmoy_mois": df_tstats[f"{label_mod}_Tm"] - df_tstats[f"{label_obs}_Tm"],
        "Diff_Tx_mois": df_tstats[f"{label_mod}_Tx"] - df_tstats[f"{label_obs}_Tx"],
    })
    
    df_diff_round = df_diff.copy()
    df_diff_round[["Diff_Tn_mois","Diff_Tmoy_mois","Diff_Tx_mois"]] = df_diff_round[["Diff_Tn_mois","Diff_Tmoy_mois","Diff_Tx_mois"]].round(2)
    
    st.write(f"Différences {label_mod} - {label_obs} (Tn_mois / Tmoy_mois / Tx_mois)")
        
    # ---- Coloration avec background_gradient ----
    st.dataframe(
        df_diff_round.style
            .background_gradient(cmap="bwr", vmin=vminT, vmax=vmaxT)
            .format("{:.2f}", subset=["Diff_Tn_mois","Diff_Tmoy_mois","Diff_Tx_mois"]),
        hide_index=True,
        use_container_width=True
    )

    # =============================
    # Comparaison moyenne annuelle
    # =============================
    
    # Moyenne annuelle sur 12 mois pour obs et modèle
    mean_obs_Tx = df_tstats[f"{label_obs}_Tx"].mean()
    mean_mod_Tx = df_tstats[f"{label_mod}_Tx"].mean()
    
    mean_obs_Tm = df_tstats[f"{label_obs}_Tm"].mean()
    mean_mod_Tm = df_tstats[f"{label_mod}_Tm"].mean()
    
    mean_obs_Tn = df_tstats[f"{label_obs}_Tn"].mean()
    mean_mod_Tn = df_tstats[f"{label_mod}_Tn"].mean()
    
    # Générer les phrases
    if mean_obs_Tx > mean_mod_Tx:
        phrase_Tx = f"En moyenne, {label_obs} est plus chaude que {label_mod} pour les températures maximales (Tx)."
    else:
        phrase_Tx = f"En moyenne, {label_mod} est plus chaud que {label_obs} pour les températures maximales (Tx)."
    
    if mean_obs_Tm > mean_mod_Tm:
        phrase_Tm = f"En moyenne, {label_obs} est plus chaude que {label_mod} pour les températures moyennes (Tmoy)."
    else:
        phrase_Tm = f"En moyenne, {label_mod} est plus chaud que {label_obs} pour les températures moyennes (Tmoy)."
    
    if mean_obs_Tn > mean_mod_Tn:
        phrase_Tn = f"En moyenne, {label_obs} est plus chaude que {label_mod} pour les températures minimales (Tn)."
    else:
        phrase_Tn = f"En moyenne, {label_mod} est plus chaud que {label_obs} pour les températures minimales (Tn)."
    
    # Stocker dans st.session_state pour pouvoir les réutiliser dans la page Résumé
    st.session_state["resume_temp"] = [phrase_Tx, phrase_Tm, phrase_Tn]
    
    # Optionnel : afficher directement les phrases sur cette page
    st.subheader("Résumé comparatif annuel des températures")
    for p in st.session_state["resume_temp"]:
        st.write("- " + p)


    # ============================
    #  SECTION: Tn / Tmoy / Tx journaliers
    # ============================
    st.subheader(f"Tn_jour / Tmoy_jour / Tx_jour — CDF par mois et tableaux de percentiles ({label_mod} vs {label_obs})")
    
    def daily_stats_from_hourly(hourly):
        """
        Retourne trois tableaux journaliers (min, mean, max).
        Tronque si nécessaire pour avoir des jours complets (24h).
        """
        if len(hourly) < 24:
            return np.array([]), np.array([]), np.array([])
        n_full_days = len(hourly) // 24
        arr = np.array(hourly[: n_full_days * 24]).reshape((n_full_days, 24))
        daily_min = arr.min(axis=1)
        daily_mean = arr.mean(axis=1)
        daily_max = arr.max(axis=1)
        return daily_min, daily_mean, daily_max
    
    # percentiles pour les petits tableaux
    pct_table = percentiles_list  # utilise la liste déjà définie en haut (ex: [10,25,50,75,90])
    pct_for_cdf = np.linspace(0, 100, 100)  # pour tracer les CDF
    
    Tx_jour_all = []
    Tn_jour_all = []
    Tm_jour_all = []

    Tx_jour_mod_all = []
    Tn_jour_mod_all = []
    Tm_jour_mod_all = []
    
    # boucle mois par mois
    for mois_num in range(1, 13):
        mois = mois_noms[mois_num]
    
        # ---- extraire hourly pour le mois: obs et modèle ----
        obs_hourly = obs_mois_all[mois_num - 1] if len(obs_mois_all) >= mois_num else np.array([])
        idx0 = sum(heures_par_mois[:mois_num - 1])
        idx1 = sum(heures_par_mois[:mois_num])
        model_hourly = model_values[idx0:idx1]
    
        # ---- calculer stats journalières ----
        obs_tn, obs_tm, obs_tx = daily_stats_from_hourly(obs_hourly)
        mod_tn, mod_tm, mod_tx = daily_stats_from_hourly(model_hourly)
        
        # Stocker les séries journalières OBS uniquement
        Tn_jour_all.append(obs_tn)
        Tm_jour_all.append(obs_tm)
        Tx_jour_all.append(obs_tx)

        # Stocker les séries journalières modèle
        Tn_jour_mod_all.append(mod_tn)
        Tm_jour_mod_all.append(mod_tm)
        Tx_jour_mod_all.append(mod_tx)
    
        # Si pas de données, passer
        if obs_tn.size == 0 or mod_tn.size == 0:
            st.write(f"{mois} — données insuffisantes pour calculer les statistiques journalières.")
            continue
    
        # ---- préparer CDFs (percentiles des séries journalières) ----
        obs_tn_cdf = np.percentile(obs_tn, pct_for_cdf)
        mod_tn_cdf = np.percentile(mod_tn, pct_for_cdf)
        obs_tm_cdf = np.percentile(obs_tm, pct_for_cdf)
        mod_tm_cdf = np.percentile(mod_tm, pct_for_cdf)
        obs_tx_cdf = np.percentile(obs_tx, pct_for_cdf)
        mod_tx_cdf = np.percentile(mod_tx, pct_for_cdf)
    
        # ---- tracé : un seul graphique regroupant Tn / Tmoy / Tx ----
        fig, ax = plt.subplots(figsize=(12, 4))
    
        # Couleurs cohérentes pour chaque variable
        colors = {
            "Tn": "cyan",
            "Tm": "white",
            "Tx": "red"
        }
    
        # Tracer modèle
        ax.plot(pct_for_cdf, mod_tx_cdf, linestyle="-", linewidth=2, label=f"{label_mod} Tx", color=colors["Tx"])
        ax.plot(pct_for_cdf, mod_tm_cdf, linestyle="-", linewidth=2, label=f"{label_mod} Tmoy", color=colors["Tm"])
        ax.plot(pct_for_cdf, mod_tn_cdf, linestyle="-", linewidth=2, label=f"{label_mod} Tn", color=colors["Tn"])
    
        # Tracer obs
        ax.plot(pct_for_cdf, obs_tx_cdf, linestyle="--", linewidth=1.7, label=f"{label_obs} Tx", color=colors["Tx"])
        ax.plot(pct_for_cdf, obs_tm_cdf, linestyle="--", linewidth=1.7, label=f"{label_obs} Tmoy", color=colors["Tm"])
        ax.plot(pct_for_cdf, obs_tn_cdf, linestyle="--", linewidth=1.7, label=f"{label_obs} Tn", color=colors["Tn"])
    
        # Mise en forme
        ax.set_title(f"{mois} — CDF Tn_jour / Tmoy_jour / Tx_jour ({label_mod} vs {label_obs})", color="white")
        ax.set_xlabel("Percentile", color="white")
        ax.set_ylabel("Température (°C)", color="white")
        ax.tick_params(colors="white")
        ax.legend(facecolor="black")
        ax.set_facecolor("none")
    
        st.pyplot(fig)
        plt.close(fig)
    
        def pct_table_values(arr, pct_list):
            return [np.percentile(arr, p) for p in pct_list]
    
        # ---- Tableau des percentiles ----
        tab = pd.DataFrame({
            "Percentile": [f"P{p}" for p in pct_table],
            f"{label_obs}_Tn": np.round(pct_table_values(obs_tn, pct_table), 2),
            f"{label_mod}_Tn": np.round(pct_table_values(mod_tn, pct_table), 2),
            f"{label_obs}_Tm": np.round(pct_table_values(obs_tm, pct_table), 2),
            f"{label_mod}_Tm": np.round(pct_table_values(mod_tm, pct_table), 2),
            f"{label_obs}_Tx": np.round(pct_table_values(obs_tx, pct_table), 2),
            f"{label_mod}_Tx": np.round(pct_table_values(mod_tx, pct_table), 2),
        })
    
        st.write(f"{mois} — Table des percentiles journaliers (Tn_jour / Tmoy_jour / Tx_jour)")
    
        num_cols = tab.select_dtypes(include=[np.number]).columns
        tab[num_cols] = tab[num_cols].apply(pd.to_numeric, errors="coerce")
        styler = tab.style.format({col: "{:.2f}" for col in num_cols})
        st.dataframe(styler, hide_index=True)
    
        # ---- Tableau des différences (modèle - obs) ----
        df_diff = pd.DataFrame({
            "Percentile": tab["Percentile"],
            "Diff_Tn_jour": tab[f"{label_mod}_Tn"] - tab[f"{label_obs}_Tn"],
            "Diff_Tm_jour": tab[f"{label_mod}_Tm"] - tab[f"{label_obs}_Tm"],
            "Diff_Tx_jour": tab[f"{label_mod}_Tx"] - tab[f"{label_obs}_Tx"],
        })
        
        # Convertir en float + arrondir
        num_cols_diff = ["Diff_Tn_jour", "Diff_Tm_jour", "Diff_Tx_jour"]
        df_diff[num_cols_diff] = df_diff[num_cols_diff].apply(pd.to_numeric, errors="coerce").round(2)

    
        st.write(f"{mois} — Différences {label_mod} - {label_obs} (Tn_jour / Tmoy_jour / Tx_jour)")
    
        df_diff_styled = (
            df_diff.style
            .background_gradient(cmap="bwr", vmin=vminT, vmax=vmaxT, subset=num_cols_diff)
            .format({col: "{:.2f}" for col in num_cols_diff})
        )
        st.dataframe(df_diff_styled, hide_index=True)

    # =========================
    # ===== CDF ANNUELLE ======
    # =========================
    st.subheader(f"CDF annuel Tn / Tx : {label_mod} vs {label_obs}")
    # Concaténation annuelle
    obs_tn_year = np.concatenate(Tn_jour_all) if len(Tn_jour_all) > 0 else np.array([])
    obs_tm_year = np.concatenate(Tm_jour_all) if len(Tm_jour_all) > 0 else np.array([])
    obs_tx_year = np.concatenate(Tx_jour_all) if len(Tx_jour_all) > 0 else np.array([])
    
    mod_tn_year = np.concatenate(Tn_jour_mod_all) if len(Tn_jour_mod_all) > 0 else np.array([])
    mod_tm_year = np.concatenate(Tm_jour_mod_all) if len(Tm_jour_mod_all) > 0 else np.array([])
    mod_tx_year = np.concatenate(Tx_jour_mod_all) if len(Tx_jour_mod_all) > 0 else np.array([])
    
    # Percentiles annuels
    pct_for_cdf = np.linspace(0, 100, 100)
    obs_tn_cdf_year = np.percentile(obs_tn_year, pct_for_cdf) if obs_tn_year.size else np.array([])
    mod_tn_cdf_year = np.percentile(mod_tn_year, pct_for_cdf) if mod_tn_year.size else np.array([])
    
    obs_tm_cdf_year = np.percentile(obs_tm_year, pct_for_cdf) if obs_tm_year.size else np.array([])
    mod_tm_cdf_year = np.percentile(mod_tm_year, pct_for_cdf) if mod_tm_year.size else np.array([])
    
    obs_tx_cdf_year = np.percentile(obs_tx_year, pct_for_cdf) if obs_tx_year.size else np.array([])
    mod_tx_cdf_year = np.percentile(mod_tx_year, pct_for_cdf) if mod_tx_year.size else np.array([])
    
    # Tracé
    fig, ax = plt.subplots(figsize=(12, 5))
    
    colors = {
        "Tn": "cyan",
        "Tm": "white",
        "Tx": "red"
    }
    
    # Modèle
    if mod_tx_cdf_year.size:
        ax.plot(pct_for_cdf, mod_tx_cdf_year, "-",  lw=2, label=f"{label_mod} Tx",   color=colors["Tx"])
    if mod_tm_cdf_year.size:
        ax.plot(pct_for_cdf, mod_tm_cdf_year, "-",  lw=2, label=f"{label_mod} Tmoy", color=colors["Tm"])
    if mod_tn_cdf_year.size:
        ax.plot(pct_for_cdf, mod_tn_cdf_year, "-",  lw=2, label=f"{label_mod} Tn",   color=colors["Tn"])
    
    # Observations
    if obs_tx_cdf_year.size:
        ax.plot(pct_for_cdf, obs_tx_cdf_year, "--", lw=1.7, label=f"{label_obs} Tx",   color=colors["Tx"])
    if obs_tm_cdf_year.size:
        ax.plot(pct_for_cdf, obs_tm_cdf_year, "--", lw=1.7, label=f"{label_obs} Tmoy", color=colors["Tm"])
    if obs_tn_cdf_year.size:
        ax.plot(pct_for_cdf, obs_tn_cdf_year, "--", lw=1.7, label=f"{label_obs} Tn",   color=colors["Tn"])
    
    ax.set_title(f"Année complète — CDF Tn_jour / Tmoy_jour / Tx_jour ({label_mod} vs {label_obs})", color="white")
    ax.set_xlabel("Percentile", color="white")
    ax.set_ylabel("Température (°C)", color="white")
    ax.tick_params(colors="white")
    ax.legend(facecolor="black")
    ax.set_facecolor("none")
    
    st.pyplot(fig)
    plt.close(fig)

    # ---------------- Histogramme annuel Tn / Tx (modèle vs obs) ----------------
    st.subheader(f"Histogramme annuel Tn / Tx : {label_mod} et {label_obs}")
    st.markdown(
        """
        La valeur de chaque barre correspond **au nombre de jours** dans lesquels la température
        minimale ou maximale journalière est comprise dans l’intervalle **[X°C , X+1°C[**.
        """,
        unsafe_allow_html=True
    )
    
    # --- Définition des classes de température ---
    bin_edges = np.arange(-10, 45, 1)  # Ajuste selon tes données
    bin_labels = bin_edges[:-1].astype(int)
    
    # --- Concaténation annuelle ---
    Tn_obs_annual = np.concatenate(Tn_jour_all) if len(Tn_jour_all) > 0 else np.array([])
    Tx_obs_annual = np.concatenate(Tx_jour_all) if len(Tx_jour_all) > 0 else np.array([])
    
    Tn_mod_annual = np.concatenate(Tn_jour_mod_all) if len(Tn_jour_mod_all) > 0 else np.array([])
    Tx_mod_annual = np.concatenate(Tx_jour_mod_all) if len(Tx_jour_mod_all) > 0 else np.array([])

    def count_days_in_bins(daily_values, bin_edges):
        """
        Retourne un tableau : nombre de jours dont la valeur tombe dans [X, X+1[ pour chaque X.
        """
        return np.histogram(daily_values, bins=bin_edges)[0] if daily_values.size else np.zeros(len(bin_edges)-1, dtype=int)
    
    # --- Comptage dans les classes ---
    obs_counts_Tn = count_days_in_bins(Tn_obs_annual, bin_edges)
    mod_counts_Tn = count_days_in_bins(Tn_mod_annual, bin_edges)
    
    obs_counts_Tx = count_days_in_bins(Tx_obs_annual, bin_edges)
    mod_counts_Tx = count_days_in_bins(Tx_mod_annual, bin_edges)
    
    # --- Préparer DataFrame ---
    df_hist = pd.DataFrame({
        "Temp_Num": bin_labels,
        "Température": bin_labels.astype(str) + "°C",
        f"{label_obs}_Tn": obs_counts_Tn,
        f"{label_mod}_Tn": mod_counts_Tn,
        f"{label_obs}_Tx": obs_counts_Tx,
        f"{label_mod}_Tx": mod_counts_Tx
    }).sort_values("Temp_Num")
    
    # ---------------- FIGURE Tn ----------------
    fig, ax = plt.subplots(figsize=(15, 5))
    ax.bar(df_hist["Temp_Num"] - 0.25, df_hist[f"{label_obs}_Tn"], width=0.4,
           label=f"{label_obs} Tn", color=couleur_TRACC)
    ax.bar(df_hist["Temp_Num"] + 0.25, df_hist[f"{label_mod}_Tn"], width=0.4,
           label=f"{label_mod} Tn", color=couleur_modele)
    
    ax.set_title("Histogramme annuel – Nombre de jours par classe de Tn")
    ax.set_xlabel("Température (°C)")
    ax.set_ylabel("Nombre de jours")
    ax.legend(fontsize='large')
    st.pyplot(fig)
    plt.close(fig)

    pct_precision_Tn = precision_overlap(mod_counts_Tn, obs_counts_Tn)
    st.write(f"Précision de {label_mod} sur les Tn_jour : **{pct_precision_Tn} %**")
    
    # ---------------- FIGURE Tx ----------------
    fig, ax = plt.subplots(figsize=(15, 5))
    ax.bar(df_hist["Temp_Num"] - 0.25, df_hist[f"{label_obs}_Tx"], width=0.4,
           label=f"{label_obs} Tx", color=couleur_TRACC)
    ax.bar(df_hist["Temp_Num"] + 0.25, df_hist[f"{label_mod}_Tx"], width=0.4,
           label=f"{label_mod} Tx", color=couleur_modele)
    
    ax.set_title("Histogramme annuel – Nombre de jours par classe de Tx")
    ax.set_xlabel("Température (°C)")
    ax.set_ylabel("Nombre de jours")
    ax.legend(fontsize='large')
    st.pyplot(fig)
    plt.close(fig)

    pct_precision_Tx = precision_overlap(mod_counts_Tx, obs_counts_Tx)
    st.write(f"Précision de {label_mod} sur les Tx_jour : **{pct_precision_Tx} %**")
     
    # --- Fonction nombre de jours de vague ---
    def nombre_jours_vague(T):
        T = np.array(T)
        n = len(T)
        jours_vague = np.zeros(n, dtype=bool)
        jours_vague[T >= 25.3] = True
        i = 0
        while i < n:
            if i + 2 < n and np.all(T[i:i+3] >= 23.4):
                debut = i
                fin = i + 2
                j = fin + 1
                while j < n and T[j] >= 23.4:
                    fin = j
                    j += 1
                prolong = fin + 1
                compteur = 0
                while prolong < n and compteur < 2:
                    if T[prolong] < 22.4:
                        break
                    fin = prolong
                    compteur += 1
                    prolong += 1
                jours_vague[debut:fin+1] = True
                i = fin + 1
            else:
                i += 1
        return int(jours_vague.sum()), jours_vague
    
    # ---------------- Calcul Tm et nombre de jours de vague sur l'année complète ----------------

    # 1) Longueur de chaque mois
    jours_par_mois = [len(Tx_jour_all[m]) for m in range(12)]
    
    # 2) Construire Tm sur toute l'année (continu)
    Tm_obs_all = np.concatenate([
        (np.array(Tx_jour_all[m]) + np.array(Tn_jour_all[m])) / 2 for m in range(12)
    ]) if all(len(Tx_jour_all[m])>0 for m in range(12)) else np.array([])
    
    Tm_mod_all = np.concatenate([
        (np.array(Tx_jour_mod_all[m]) + np.array(Tn_jour_mod_all[m])) / 2 for m in range(12)
    ]) if all(len(Tx_jour_mod_all[m])>0 for m in range(12)) else np.array([])
    
    # 3) Calculer les jours de vague en continu sur l'année
    _, jours_vague_obs_all = nombre_jours_vague(Tm_obs_all) if Tm_obs_all.size else (0, np.array([]))
    _, jours_vague_mod_all = nombre_jours_vague(Tm_mod_all) if Tm_mod_all.size else (0, np.array([]))
    
    # 4) Re-découper par mois
    jours_vague_obs = []
    jours_vague_mod = []
    
    idx = 0
    for L in jours_par_mois:
        if len(jours_vague_obs_all) >= idx+L:
            jours_vague_obs.append(int(jours_vague_obs_all[idx:idx+L].sum()))
        else:
            jours_vague_obs.append(0)
        if len(jours_vague_mod_all) >= idx+L:
            jours_vague_mod.append(int(jours_vague_mod_all[idx:idx+L].sum()))
        else:
            jours_vague_mod.append(0)
        idx += L
    
    # ---------------- Tableau ----------------
    df_vagues = pd.DataFrame({
        "Mois": df_tstats["Mois"],
        label_obs: jours_vague_obs,
        label_mod: jours_vague_mod
    })
    st.subheader("Nombre de jours de vague de chaleur par mois")
    st.dataframe(df_vagues, hide_index=True, use_container_width=True)
    
    # ---------------- Graphique bâtons ----------------
    fig, ax = plt.subplots(figsize=(12, 5))
    x = np.arange(1, 13)
    ax.bar(x - 0.2, jours_vague_obs, width=0.4, label=label_obs, color=couleur_TRACC)
    ax.bar(x + 0.2, jours_vague_mod, width=0.4, label=label_mod, color=couleur_modele)
    ax.set_xlabel("Mois")
    ax.set_ylabel("Nombre de jours de vague de chaleur")
    ax.set_title("Nombre de jours de vague de chaleur par mois")
    ax.set_xticks(x)
    ax.legend()
    st.pyplot(fig)
    plt.close(fig)

    # ============================
    # GRAPHIQUES : Jours chauds et nuits tropicales par mois
    # ============================

    st.subheader("Graphiques : jours chauds et nuits tropicales par mois")
    
    # Choix seuil pour Tx
    tx_seuil = st.number_input("Seuil Tx_jour (°C) pour jours chauds :", min_value=-50.0, max_value=60.0, value=30.0, step=1.0)
    tn_seuil = st.number_input("Seuil Tn_jour (°C) pour nuits tropicales :", min_value=-50.0, max_value=60.0, value=20.0, step=1.0) 
    
    # Préparer listes pour stocker les valeurs par mois
    jours_chauds_obs = []
    jours_chauds_mod = []
    nuits_tropicales_obs = []
    nuits_tropicales_mod = []
    
    jours_chauds_total_obs = 0
    jours_chauds_total_mod = 0
    nuits_tropicales_total_obs = 0
    nuits_tropicales_total_mod = 0
    
    for mois_num in range(1, 13):
        # obs
        obs_tx_jour = Tx_jour_all[mois_num - 1]
        obs_tn_jour = Tn_jour_all[mois_num - 1]
        jours_tx = np.sum(obs_tx_jour > tx_seuil) if obs_tx_jour.size else 0
        nuits_trop = np.sum(obs_tn_jour > tn_seuil) if obs_tn_jour.size else 0
        jours_chauds_obs.append(jours_tx)
        nuits_tropicales_obs.append(nuits_trop)
        jours_chauds_total_obs += jours_tx
        nuits_tropicales_total_obs += nuits_trop
    
        # modèle
        mod_tx_jour = Tx_jour_mod_all[mois_num - 1]
        mod_tn_jour = Tn_jour_mod_all[mois_num - 1]
        jours_tx_mod = np.sum(mod_tx_jour > tx_seuil) if mod_tx_jour.size else 0
        nuits_trop_mod = np.sum(mod_tn_jour > tn_seuil) if mod_tn_jour.size else 0
        jours_chauds_mod.append(jours_tx_mod)
        nuits_tropicales_mod.append(nuits_trop_mod)
        jours_chauds_total_mod += jours_tx_mod
        nuits_tropicales_total_mod += nuits_trop_mod
    
    # Labels pour les mois
    mois_labels = [mois_noms[m] for m in range(1, 13)]
    x = np.arange(len(mois_labels))
    
    # ---- Diagramme jours chauds ----
    fig, ax = plt.subplots(figsize=(14, 4))
    ax.bar(x - 0.25, jours_chauds_obs, width=0.5, color=couleur_TRACC, label=label_obs)
    ax.bar(x + 0.25, jours_chauds_mod, width=0.5, color=couleur_modele, label=label_mod)
    ax.set_xticks(x)
    ax.set_xticklabels(mois_labels, rotation=45)
    ax.set_ylabel(f"Nombre de jours Tx_jour > {tx_seuil}°C")
    ax.set_title("Jours chauds par mois")
    ax.legend()
    fig_jourschaud=fig
    st.pyplot(fig)
    plt.close(fig)
    
    # ---- Diagramme nuits tropicales ----
    fig, ax = plt.subplots(figsize=(14, 4))
    ax.bar(x - 0.25, nuits_tropicales_obs, width=0.5, color=couleur_TRACC, label=label_obs)
    ax.bar(x + 0.25, nuits_tropicales_mod, width=0.5, color=couleur_modele, label=label_mod)
    ax.set_xticks(x)
    ax.set_xticklabels(mois_labels, rotation=45)
    ax.set_ylabel(f"Nombre de nuits Tn_jour > {tn_seuil}°C")
    ax.set_title("Nuits tropicales par mois")
    ax.legend()
    fig_nuittrop=fig
    st.pyplot(fig)
    plt.close(fig)
    
    # ---- Affichage des totaux ----
    st.markdown(f"**Total jours chauds {label_mod} :** {jours_chauds_total_mod} / **{label_obs} :** {jours_chauds_total_obs}")
    st.markdown(f"**Total nuits tropicales {label_mod} :** {nuits_tropicales_total_mod} / **{label_obs} :** {nuits_tropicales_total_obs}")

    # =============================
    # Comparaison annuelle jours chauds / nuits tropicales
    # =============================
    
    # Jours chauds
    if jours_chauds_total_obs > jours_chauds_total_mod:
        phrase_jours = f"{label_obs} enregistre plus de jours chauds (Tx>{tx_seuil}°C) sur l'année ({jours_chauds_total_obs}) que {label_mod} ({jours_chauds_total_mod})."
    else:
        phrase_jours = f"{label_mod} enregistre plus de jours chauds (Tx>{tx_seuil}°C) sur l'année ({jours_chauds_total_mod}) que {label_obs} ({jours_chauds_total_obs})."
    
    # Nuits tropicales
    if nuits_tropicales_total_obs > nuits_tropicales_total_mod:
        phrase_nuits = f"{label_obs} enregistre plus de nuits tropicales (Tn>{tn_seuil}°C) sur l'année ({nuits_tropicales_total_obs}) que {label_mod} ({nuits_tropicales_total_mod})."
    else:
        phrase_nuits = f"{label_mod} enregistre plus de nuits tropicales (Tn>{tn_seuil}°C) sur l'année ({nuits_tropicales_total_mod}) que {label_obs} ({nuits_tropicales_total_obs})."
    
    # Stocker dans st.session_state pour la page Résumé
    st.session_state["resume_chaud_nuit"] = [phrase_jours, phrase_nuits]
    
    # Optionnel : affichage sur la page actuelle
    st.subheader("Résumé comparatif jours chauds / nuits tropicales")
    for p in st.session_state["resume_chaud_nuit"]:
        st.write("- " + p)

    # ============================
    # Calcul DJC (chauffage) et DJF (froid)
    # ============================
    
    st.subheader("DJC (chauffage) et DJF (froid) journaliers — modèle vs observations")
    
    T_base_chauffage = float(st.text_input("Base DJC (°C) — chauffage", "19"))
    T_base_froid = float(st.text_input("Base DJF (°C) — refroidissement", "23"))
    
    results_djc = []
    results_djf = []
    mois_noms_sans_num = {
    1: "Janvier",   2: "Février",  3: "Mars",
    4: "Avril",     5: "Mai",      6: "Juin",
    7: "Juillet",   8: "Août",     9: "Septembre",
    10: "Octobre", 11: "Novembre", 12: "Décembre"
    }

    for mois_num in range(1, 13):
        mois = mois_noms_sans_num[mois_num]
    
        # Séries journalières déjà calculées
        Tx_obs = Tx_jour_all[mois_num-1]
        Tn_obs = Tn_jour_all[mois_num-1]
    
        idx0 = sum(heures_par_mois[:mois_num-1])
        idx1 = sum(heures_par_mois[:mois_num])
        model_hourly = model_values[idx0:idx1]
        Tx_mod, Tm_mod, Tn_mod = daily_stats_from_hourly(model_hourly)
    
        DJC_obs_jours, DJF_obs_jours = [], []
        DJC_mod_jours, DJF_mod_jours = [], []
    
        n_jours = len(Tx_obs)
        for j in range(n_jours):
            Tm_obs = (Tx_obs[j] + Tn_obs[j]) / 2
            DJC_obs_jours.append(max(0, T_base_chauffage - Tm_obs))
            DJF_obs_jours.append(max(0, Tm_obs - T_base_froid))
    
            if j < len(Tx_mod):
                Tm_mod_j = (Tx_mod[j] + Tn_mod[j]) / 2
                DJC_mod_jours.append(max(0, T_base_chauffage - Tm_mod_j))
                DJF_mod_jours.append(max(0, Tm_mod_j - T_base_froid))
            else:
                DJC_mod_jours.append(0)
                DJF_mod_jours.append(0)
    
        DJC_obs_sum = float(np.nansum(DJC_obs_jours))
        DJC_mod_sum = float(np.nansum(DJC_mod_jours))
        DJF_obs_sum = float(np.nansum(DJF_obs_jours))
        DJF_mod_sum = float(np.nansum(DJF_mod_jours))
    
        results_djc.append({
            "Mois": mois,
            label_mod: DJC_mod_sum,
            label_obs: DJC_obs_sum,
            "Différence": DJC_mod_sum - DJC_obs_sum
        })
        results_djf.append({
            "Mois": mois,
            label_mod: DJF_mod_sum,
            label_obs: DJF_obs_sum,
            "Différence": DJF_mod_sum - DJF_obs_sum
        })
    
    df_DJC = pd.DataFrame(results_djc).fillna(0)
    df_DJF = pd.DataFrame(results_djf).fillna(0)
    
    # Convertir explicitement les colonnes numériques en float
    for df in [df_DJC, df_DJF]:
        for col in [label_mod, label_obs, "Différence"]:
            df[col] = df[col].astype(float)
    
    # --------------------------
    # Affichage tables Streamlit
    # --------------------------
    st.subheader("DJC – Chauffage (somme journalière par mois)")
    st.dataframe(df_DJC.round(2))  # Arrondi à 2 décimales
    
    st.subheader("DJF – Refroidissement (somme journalière par mois)")
    st.dataframe(df_DJF.round(2))  # Arrondi à 2 décimales
    
    # --------------------------
    # Diagrammes bâtons mensuels
    # --------------------------
    st.subheader("Diagrammes bâtons mensuels — DJC et DJF")

    # Convertir en DataFrames
    df_DJC = pd.DataFrame(results_djc)
    df_DJF = pd.DataFrame(results_djf)
    
    # -----------------------------
    # Diagrammes en bâtons par mois
    # -----------------------------
    figures = {}   # dictionnaire où on stocke les figures

    for df, titre in zip([df_DJC, df_DJF], ["DJC", "DJF"]):
        fig, ax = plt.subplots(figsize=(14, 4))
        ax.bar(df.index - 0.25, df[label_obs], width=0.5,
               color=couleur_TRACC, label=label_obs)
        ax.bar(df.index + 0.25, df[label_mod], width=0.5,
               color=couleur_modele, label=label_mod)
    
        ax.set_xticks(df.index)
        ax.set_xticklabels(df["Mois"])
        ax.set_title(f"{titre} mensuel — {label_mod} vs {label_obs}")
        ax.set_ylabel(f"{titre} (°C·jour)")
        ax.set_xlabel("Mois")
        ax.legend()
    
        # enregistrer la figure dans le dictionnaire
        figures[titre] = fig
    
        st.pyplot(fig)
        plt.close(fig)

    # --------------------------
    # Somme annuelle DJC et DJF
    # --------------------------
    total_DJC_obs = df_DJC[label_obs].sum()
    total_DJC_mod = df_DJC[label_mod].sum()
    
    total_DJF_obs = df_DJF[label_obs].sum()
    total_DJF_mod = df_DJF[label_mod].sum()
    
    st.subheader("Sommes annuelles")
    st.write(f"DJC annuel : {label_mod} = {total_DJC_mod:.0f}   /    {label_obs} = {total_DJC_obs:.0f}")
    st.write(f"DJF annuel : {label_mod} = {total_DJF_mod:.0f}   /   {label_obs} = {total_DJF_obs:.0f}")

    # =============================
    # Résumé automatique DJC / DJF
    # =============================
    
    # DJC (chauffage)
    if total_DJC_obs > total_DJC_mod:
        phrase_djc = f"{label_obs} a une demande de chauffage annuelle plus élevée ({total_DJC_obs:.0f} °C·jour) que {label_mod} ({total_DJC_mod:.0f} °C·jour)."
    elif total_DJC_mod > total_DJC_obs:
        phrase_djc = f"{label_mod} a une demande de chauffage annuelle plus élevée ({total_DJC_mod:.0f} °C·jour) que {label_obs} ({total_DJC_obs:.0f} °C·jour)."
    else:
        phrase_djc = f"{label_mod} et {label_obs} ont la même demande de chauffage annuelle."
    
    # DJF (refroidissement)
    if total_DJF_obs > total_DJF_mod:
        phrase_djf = f"{label_obs} a une demande de refroidissement annuelle plus élevée ({total_DJF_obs:.0f} °C·jour) que {label_mod} ({total_DJF_mod:.0f} °C·jour)."
    elif total_DJF_mod > total_DJF_obs:
        phrase_djf = f"{label_mod} a une demande de refroidissement annuelle plus élevée ({total_DJF_mod:.0f} °C·jour) que {label_obs} ({total_DJF_obs:.0f} °C·jour)."
    else:
        phrase_djf = f"{label_mod} et {label_obs} ont la même demande de refroidissement annuelle."
    
    # Stocker dans st.session_state pour la page Résumé
    st.session_state["resume_djc_djf"] = [phrase_djc, phrase_djf]
    
    # Optionnel : affichage sur la page actuelle
    st.subheader("Résumé comparatif DJC / DJF")
    for p in st.session_state["resume_djc_djf"]:
        st.write("- " + p)

    # ======================================
    #  COURBES DES PERCENTILES PAR MOIS
    # ======================================
    st.subheader("Évolution mensuelle des percentiles (modèle vs obs)")

    df_percentiles_all = []
    percentiles_list2 = [10,50,90]
    
    for mois_num in range(1, 13):
        mois = mois_noms[mois_num]
    
        # Observations
        obs_vals = obs_mois_all[mois_num-1]
    
        # modèle
        idx0 = sum(heures_par_mois[:mois_num-1])
        idx1 = sum(heures_par_mois[:mois_num])
        mod_vals = model_values[idx0:idx1]

        
        # Ajout des percentiles
        for p in percentiles_list2:
            df_percentiles_all.append({
                "Mois": mois,
                "Percentile": f"P{p}",
                label_obs: np.percentile(obs_vals, p),
                label_mod: np.percentile(mod_vals, p)
            })

    # Table ordonnée pour faciliter les tracés
    df_percentiles_ordered = (
        pd.DataFrame(df_percentiles_all)
        .assign(Pnum=lambda d: d["Percentile"].str.extract("(\d+)").astype(int))
        .sort_values(["Pnum", "Mois"])
    )
    
    # Construction du graphique par percentile
    fig, ax = plt.subplots(figsize=(14,5))
    colors_perc = ["darkcyan", "khaki", "firebrick"]
    i=0
    for p in percentiles_list2:
        dfp = df_percentiles_ordered[df_percentiles_ordered["Pnum"] == p]
        # obs : ligne pointillée
        ax.plot(
            dfp["Mois"], dfp[label_obs],
            linestyle="--", label=f"{label_obs} P{p}", color=colors_perc[i]
        )
        # modèle : ligne pleine
        ax.plot(
            dfp["Mois"], dfp[label_mod],
            linestyle="-", label=f"{label_mod} P{p}", color=colors_perc[i]
        )
        i+=1
    
    ax.set_title(f"Percentiles {percentiles_list} – {label_mod} vs {label_obs} ")
    ax.set_ylabel("Température (°C)")
    ax.tick_params(axis="x", rotation=45)
    ax.legend(ncol=2, facecolor="black")
    st.pyplot(fig)
    plt.close(fig)

    # -------- Calcul des percentiles P1 à P100 --------
    percentiles = np.arange(1, 101)
    P_obs = np.percentile(obs_annee, percentiles)
    P_mod = np.percentile(model_annee, percentiles)
    
    # -------- Graphique PXX modèle vs obs avec croix et couleurs conditionnelles --------
    fig, ax = plt.subplots(figsize=(6,6))
    
    # Définir les couleurs selon qui est plus chaud
    colors = [couleur_TRACC if obs > mod else couleur_modele for obs, mod in zip(P_obs, P_mod)]
    
    # Tracer les croix
    ax.scatter(P_obs, P_mod, color=colors, marker='x', s=50, label='Percentiles')
    
    # Diagonale y=x
    ax.plot([-10, 45], [-10, 45], color='white', linestyle='--', label='y=x')
    
    # Carré : même échelle sur x et y
    ax.set_xlim(-10, 45)
    ax.set_ylim(-10, 45)
    ax.set_aspect('equal', 'box')
    
    ax.set_xlabel(f"PXX {label_obs} (°C)")
    ax.set_ylabel(f"PXX {label_mod} (°C)")
    ax.set_title("Comparaison des percentiles annuels")
    ax.grid(True, linestyle=':', color='gray', alpha=0.5)
    ax.legend()
    fig_quantilequantile = fig
    st.pyplot(fig)


    # -------- Graphiques CDF et percentiles mensuels --------
    st.subheader("Fonctions de répartition mensuelles (CDF)")
    df_percentiles_all = []
    
    for mois_num in range(1, 13):
        mois = mois_noms[mois_num]
        obs_mois = obs_mois_all[mois_num-1]
        mod_mois = model_values[sum(heures_par_mois[:mois_num-1]):sum(heures_par_mois[:mois_num])]
        obs_percentiles_100 = np.percentile(obs_mois, np.linspace(0, 100, 100)) if obs_mois.size else np.array([])
        mod_percentiles_100 = np.percentile(mod_mois, np.linspace(0, 100, 100)) if mod_mois.size else np.array([])

        fig, ax = plt.subplots(figsize=(12, 4))
        if mod_percentiles_100.size:
            ax.plot(np.linspace(0, 100, 100), mod_percentiles_100, label=label_mod, color=couleur_modele)
        if obs_percentiles_100.size:
            ax.plot(np.linspace(0, 100, 100), obs_percentiles_100, label=label_obs, color=couleur_TRACC)
        ax.set_title(f"{mois} - Fonction de répartition ({label_mod} vs {label_obs})", color="white")
        ax.set_xlabel("Percentile", color="white")
        ax.set_ylabel("Température (°C)", color="white")
        ax.tick_params(colors="white")
        ax.legend(facecolor="black")
        ax.set_facecolor("none")
        st.pyplot(fig)
        plt.close(fig)

        obs_p = np.percentile(obs_mois, percentiles_list) if obs_mois.size else np.array([np.nan]*len(percentiles_list))
        mod_p = np.percentile(mod_mois, percentiles_list) if mod_mois.size else np.array([np.nan]*len(percentiles_list))
        df_p = pd.DataFrame({
            "Percentile": [f"P{p}" for p in percentiles_list],
            f"{label_obs}": obs_p,
            f"{label_mod}": mod_p
        }).round(2)
        st.write(f"{mois} - Percentiles")
        st.dataframe(df_p, hide_index=True)

        for i, p in enumerate(percentiles_list):
            df_percentiles_all.append({
                "Mois": mois,
                "Percentile": f"P{p}",
                "Obs": obs_p[i],
                "Mod": mod_p[i]
            })

    # -------- Fonction de répartition ANNUELLE --------
    st.subheader("Fonction de répartition annuelle (CDF)")
    
    # Regroupement annuel
    obs_annual = np.concatenate(obs_mois_all)         # Observations - toutes les heures de l'année
    mod_annual = model_values                         # modèle : déjà toutes les heures
    
    # Percentiles pour CDF (0–100)
    percentiles_cdf = np.linspace(0, 100, 100)
    obs_percentiles_annual = np.percentile(obs_annual, percentiles_cdf)
    mod_percentiles_annual = np.percentile(mod_annual, percentiles_cdf)
    
    # ----- Plot de la CDF annuelle -----
    fig, ax = plt.subplots(figsize=(12, 4))
    ax.plot(percentiles_cdf, mod_percentiles_annual, label=label_mod, color=couleur_modele)
    ax.plot(percentiles_cdf, obs_percentiles_annual, label=label_obs, color=couleur_TRACC)
    
    ax.set_title("Année entière - Fonction de répartition", color="white")
    ax.set_xlabel("Percentile", color="white")
    ax.set_ylabel("Température (°C)", color="white")
    ax.tick_params(colors="white")
    ax.legend(facecolor="black")
    ax.set_facecolor("none")
    
    fig_cdf = fig
    
    st.pyplot(fig)
    plt.close(fig)
    
    # ------ Tableau des percentiles annuels ------
    obs_p_annual = np.percentile(obs_annual, percentiles_list)
    mod_p_annual = np.percentile(mod_annual, percentiles_list)
    
    df_p_annual = pd.DataFrame({
        "Percentile": [f"P{p}" for p in percentiles_list],
        f"{label_obs}": obs_p_annual,
        f"{label_mod}": mod_p_annual
    }).round(2)
    
    st.write("Année entière - Percentiles")
    st.dataframe(df_p_annual, hide_index=True)


    st.subheader(f"Bilan de {label_mod} vs {label_obs}  ({label_mod} - {label_obs})") 
    # Création du DataFrame bilan à partir des percentiles mensuels collectés plus haut
    df_bilan = pd.DataFrame(df_percentiles_all).round(2)
    df_bilan["Ecart"] = df_bilan["Mod"] - df_bilan["Obs"]
    # Extraire le numéro du percentile pour imposer l'ordre
    df_bilan["Percentile_num"] = df_bilan["Percentile"].str.extract("(\d+)").astype(int)
    # Imposer l'ordre des percentiles
    df_bilan["Percentile"] = pd.Categorical(df_bilan["Percentile"], 
                                            categories=[f"P{p}" for p in percentiles_list], 
                                            ordered=True)
    # Pivot pour affichage
    df_bilan_pivot = df_bilan.pivot(index="Percentile", columns="Mois", values="Ecart").round(2)
    # Affichage stylé avec couleurs selon l'écart
    st.dataframe(
        df_bilan_pivot.style
        .background_gradient(cmap="bwr", vmin=vminT, vmax=vmaxT)
        .format("{:.2f}")
    )
    
    # ---- Stockage des figures et DataFrames dans session_state (facultatif) ----
    st.session_state["fig_quantilequantile"] = fig_quantilequantile
    st.session_state["fig_hist_year"] = fig_hist_year
    st.session_state["fig_hist_diff"] = fig_hist_diff
    st.session_state["df_rmse"] = df_rmse
    st.session_state["df_rmse_styled"] = df_rmse_styled
    st.session_state["fig_tn_tx_mois"] = fig_tn_tx_mois
    st.session_state["fig_jourschaud"] = fig_jourschaud
    st.session_state["fig_nuittrop"] = fig_nuittrop
    st.session_state["fig_cdf"] = fig_cdf
    st.session_state["fig_DJC"] = figures.get("DJC")
    st.session_state["fig_DJF"] = figures.get("DJF")

