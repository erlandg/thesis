import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from wordcloud import WordCloud


def _generate_table(groups, index):
    groups = [(group.iloc[:,index].sum(), len(group.iloc[:,index])) for group in groups]
    table = np.zeros((3, len(groups)+1))
    for i, (positive, total) in enumerate(groups):
        table[:,i] = np.array([positive, total-positive, total])
    table[:,-1] = table[:].sum(1)
    return table


def _import_diagnosis_data(fname="../data/processed/diagnoses.csv", **kwargs):
    return pd.read_csv(fname, **kwargs)


def plot_p_histogram(ps, save_fname, title='p-value histogram', label="p-values", set_baseline=True):
    plt.title(title)
    if set_baseline:
        plt.hist(
            np.linspace(0,1,1000),
            bins=40,
            range=(0,1),
            density=False,
            color='blue',
            alpha=.4,
            label="baseline"
        )
    plt.hist(
        ps,
        bins=40,
        range=(0,1),
        density=False,
        color='red',
        alpha=.4,
        label=label
    )
    if set_baseline: plt.legend()
    plt.savefig(save_fname, bbox_inches='tight')
    plt.clf()


def _find_split(*diagnosis_matrix_groups, return_p=False, two_sided=True, tol = 1e-8):
    p = pd.concat(diagnosis_matrix_groups, axis=0).mean()
    k = len(diagnosis_matrix_groups)
    if k == 2:
        # Two-sample T-test for proportion
        p1, p2 = [group.mean() for group in diagnosis_matrix_groups]
        n1, n2 = [len(group) for group in diagnosis_matrix_groups]
        results = np.abs((p1 - p2)/np.sqrt(p * (1-p) * (1/n1 + 1/n2))) # z-value
    else:
        results = np.zeros(len(p))
        # Chi-Square Test of Homogeneity for each feature separately
        df = len(diagnosis_matrix_groups) - 1
        for i in range(len(results)):
            table_i = _generate_table(diagnosis_matrix_groups, i)

            Es = (table_i[:-1,-1].reshape(-1,1) @ table_i[-1,:-1].reshape(1,-1)) / table_i[-1,-1]

            chis = ((table_i[:-1,:-1] - Es)**2)/Es
            results[i] = chis.sum()

    if return_p:
        import scipy.stats as stats
        if k == 2:
            if two_sided:
                ps = 2 * stats.norm.sf(results)
            else:
                ps = stats.norm.sf(results)
        else:
            ps = 1 - stats.chi2.cdf(results, df=df)
        return results, ps
    return results


def analyse_grouping(diagnoses, views, pred, min_diagnosis_count=100, print_N_best=20, save_folder=None):
    diagnoses = diagnoses.loc[:,diagnoses.sum() >= min_diagnosis_count]
    views = np.concatenate(views, axis=1)
    idxs = [[] for _ in np.unique(pred)]
    for k in np.unique(pred):
        idxs[k].extend(list(np.where(pred == k)[0]))

    results, ps = _find_split(
        *[diagnoses.iloc[idx] for idx in idxs],
        return_p=True
    )
    bonferroni_ps = np.clip(ps * len(ps), 0, 1)
    print(f"{100*sum(bonferroni_ps < .05)/len(bonferroni_ps)}% of the diagnoses were significant post-Bonferroni correction")
    
    splits_ = pd.concat([diagnoses.iloc[idx].sum() for idx in idxs], axis=1)
    splits_ = splits_.apply(lambda row: tuple(row), axis=1)
    
    sig_table = pd.DataFrame(
        {
            "Split" : splits_.values,
            "p-values" : ps,
            "Corrected p-values" : bonferroni_ps,
            "Significant" : bonferroni_ps < .05,
        },
        index = diagnoses.columns
    )
    sig_table = sig_table.sort_values(by="Corrected p-values")
    if not save_folder is None:
        plot_p_histogram(ps, save_folder+"p-value_hist.png", set_baseline=True)
        plot_p_histogram(bonferroni_ps, save_folder+"bf_p-value_hist.png", title='Bonferroni p-value histogram', label="Bonferroni p-values", set_baseline=False)
        sig_table.to_csv(save_folder+"p-vals.csv")

    best_idxs = results.argsort()[::-1]
    for i, best_idx in enumerate(best_idxs[:print_N_best]):
        splits = [diagnoses.iloc[idx,best_idx].sum() for idx in idxs]
        print(f"#{i+1} {diagnoses.columns[best_idx]} :\n- Split: {splits}\n- p-value: {ps[best_idx]}")
    return list(diagnoses.columns[bonferroni_ps < .05])


def create_word_cloud(diagnoses, columns, pred, labels=None, save_folder=None):
    k = np.unique(pred)
    diagnoses = diagnoses[columns]
    diag_counts = pd.concat([diagnoses[pred == k_].sum() for k_ in k], axis=1)
    diag_counts = diag_counts.apply(lambda row: row - row.min(), axis=1)
    word_counts = diag_counts.apply(lambda x: x.index.repeat(x), axis=0)

    dims = (np.floor(len(k)).astype(int), np.ceil(len(k)).astype(int))
    exceptions = [
        "unspecified", "other", "without mention", "and",
        "of", "the", "with", "in", "to", "for", "at", "or"
    ]
    
    with plt.style.context("dark_background"):
        plt.figure(dpi=300)
        for i in word_counts.index:
            words = " ".join([_.lower() for _ in word_counts[i]])
            if words == '':
                words = "empty"
            wc = WordCloud(
                width=960,
                height=720,
                collocations=False,
                stopwords=exceptions,
            ).generate(words)
            plt.subplot(dims[0],dims[1],i+1)
            if labels is not None:
                plt.title(f"Predicted {labels[k[i]]}")
            else:
                plt.title(f"Predicted {k[i]}")
            plt.imshow(wc, interpolation='bilinear')
            plt.axis("off")
    if save_folder is not None:
        plt.savefig(save_folder+f'word_cloud.png', bbox_inches='tight')
    plt.close()
