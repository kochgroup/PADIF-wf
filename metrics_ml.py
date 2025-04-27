
import glob
import numpy as np
import pandas as pd
import seaborn as sns
from tqdm import tqdm
from itertools import groupby
import matplotlib.pyplot as plt
from pycaret.classification import *
from sklearn.metrics import balanced_accuracy_score
from rdkit.ML.Scoring.Scoring import CalcEnrichment

def add_line(ax, xpos, ypos):
    line = plt.Line2D([ypos, ypos+ .85], [xpos, xpos], color='black', transform=ax.transAxes, alpha=.3, linestyle="--")
    line.set_clip_on(False)
    ax.add_line(line)

def label_len(my_index,level):
    labels = my_index.get_level_values(level)
    return [(k, sum(1 for i in g)) for k,g in groupby(labels)]

def label_group_bar_table(ax, df):
    xpos = -1.5
    scale = 1./df.index.size
    for level in range(df.index.nlevels):
        pos = df.index.size
        for label, rpos in label_len(df.index,level):
            add_line(ax, pos*scale, xpos)
            pos -= rpos
            lypos = (pos + .5 * rpos)*scale
            ax.text(xpos+.4, lypos, label, ha='center', transform=ax.transAxes, size='small') 
        add_line(ax, pos*scale , xpos)
        xpos = -0.9

def pre_test(ref, test):
    test.columns = [*test.columns[:-1], "activity"]
    list_add = [col for col in ref.columns.tolist() if col not in test.columns.tolist()]
    tm = test.reindex(columns=[*test.columns.tolist(), *list_add], fill_value=0.0)
    tm["activity"] = test["activity"]

    return tm

### Metrics for models with true inactive

types1 = ['true',"dcm", "znc", "div"]
df = pd.read_parquet("files/targets_information.parquet")
targets = df.name.tolist()

all_data = []
mts = []
for target in tqdm(targets):

    lista = []
    data_x = []
    test_list = []

    for type in types1:
        print(f'{target} from {type}')

        er = []
        bal_acc = []
        models = []   
        ref = pd.read_parquet(f'files/ml-models/{target}/{type}/data_to_model/Random/Test.parquet')
        test = pd.read_parquet(f'files/ml-models/{target}/true/data_to_model/Random/Test.parquet')
        test_f = pre_test(ref, test) 
        test_list.append(test_list)
        data_unk = []       

        for model in glob.glob(f'files/ml-models/{target}/{type}/models/Random/*.pkl'):
            md = model[:-4]
            md2 = load_model(md)
            pred = predict_model(md2, test_f)
            act = pred[pred["prediction_label"] == 1].sort_values(["score"],ascending=False)
            ina = pred[pred["prediction_label"] == 0].sort_values(["score"],ascending=False)
            pred = pd.concat([act, ina])
            scores = [[x] for x in pred.activity]
            bal_acc.append(round(balanced_accuracy_score(pred.activity, pred.prediction_label),4))
            er.append(CalcEnrichment(scores, 0, [0.01, 0.25]))
            models.append(model.split("/")[-1].split(".")[0])
            pred['model'] = model.split("/")[-1].split(".")[0]
            data_unk.append(pred)

        metrics = pd.DataFrame(
            list(zip(bal_acc, models)), 
            columns=["BA", "model"]
        )

        enrichment = pd.DataFrame(er, columns=["EF1%", "EF25%"]) 
        metrics = pd.concat([metrics, enrichment], axis=1)
        metrics["Decoys"] = type

        data_u = pd.concat(data_unk)
        data_u['Decoys'] = type   
        data_x.append(data_u)
        
        lista.append(metrics)
    
    final = pd.concat(lista).reset_index(drop=True)
    final["name"] =  target

    d = pd.concat(data_x)
    d['name'] = target
    all_data.append(d)

    mts.append(final)

### concat the results
true_ina = pd.concat(mts).reset_index(drop=True)
order = ['true',"dcm", "znc", "div"]
true_ina["Decoys"] = pd.Categorical(true_ina.Decoys, ordered=True, categories=order)
true_ina = true_ina.sort_values(by=["Decoys"])
true_ina = true_ina.merge(df, on='name')

### Create Figures 
sns.set_style("ticks")
fig, ax = plt.subplots(figsize=(15,12))
sns.set(rc={'figure.figsize': (15, 15)})
sns.set(font_scale=2)
sns.boxplot(y="set",x="BA", hue='Decoys',orient="h",data=true_ina, ax =ax, palette=["#007fff",'#d9f0a3', '#77c679', '#228343'])
ax.set(ylabel="Target",xlabel="$balanced - accuracy$", xlim=(0.5, 1))
_ = ax.legend(loc='upper right', bbox_to_anchor=(1.15, 1))
sns.despine(right=True, top=True)
[ax.axhline(x, color = 'black', linestyle=':') for x in [0.5,1.5,2.5,3.5,4.5,5.5,6.5,7.5,8.5]]
plt.savefig("pictures/BA_true_test-set.png", dpi=300, bbox_inches="tight")

### Classify by CHEMPLP score

types1 = ["true", "dcm", "znc", "div"]

chemplp_score = []
for target in tqdm(targets):

    lista = []

    for type in types1:

        er = []
        bal_acc = []         
        test = pd.read_parquet(f'files/ml-models/{target}/{type}/data_to_model/Random/Test.parquet')     
 
        test = test.astype({"score":"float64"})
        test = test.sort_values(by=["score"], ascending=False)
        test["prediction_label"] = [1 for val in range(int(len(test)*0.25))] + [0 for val in range(len(test) - int(len(test)*0.25))]
        act = test[test["prediction_label"] == 1].sort_values(["score"],ascending=False)
        ina = test[test["prediction_label"] == 0].sort_values(["score"],ascending=False)
        pred = pd.concat([act, ina])
        scores = [[x] for x in pred.activity]
        er.append(CalcEnrichment(scores, 0, [0.01, 0.25]))
        
        metrics = pd.DataFrame(er, columns=['EF1%', 'EF25%'])
        metrics['BA'] = round(balanced_accuracy_score(pred.activity, pred.prediction_label),4)
        metrics["Decoys"] = type   
        
        lista.append(metrics)
    
    final = pd.concat(lista).reset_index(drop=True)
    final["name"] =  target

    chemplp_score.append(final)

cs = pd.concat(chemplp_score)
cs = cs.merge(df, on='name')
cs['model'] = 'chemplp_score'

data_true_metrics = pd.concat([true_ina, cs]).reset_index(drop=True)

### Organize Data to plot
order = ["rf", "svm", "xgboost", "mlp", "chemplp_score"]
true = data_true_metrics[data_true_metrics["Decoys"] == "true"]
true["model"] = pd.Categorical(true.model, ordered=True, categories=order)
true = true.sort_values(by=["model"])
true.index = true.model 
true = true.groupby("set", group_keys=True).apply(lambda x:x)
true1 = true[["BA", "EF1%", "EF25%", "model"]].reset_index(drop=True)

dcm = data_true_metrics[data_true_metrics["Decoys"] == "dcm"]
dcm["model"] = pd.Categorical(dcm.model, ordered=True, categories=order)
dcm = dcm.sort_values(by=["model"])
dcm = dcm.groupby("set", group_keys=True).apply(lambda x:x)
dcm = dcm[["BA", "EF1%", "EF25%", "model"]].reset_index(drop=True)

znc = data_true_metrics[data_true_metrics["Decoys"] == "znc"]
znc["model"] = pd.Categorical(znc.model, ordered=True, categories=order)
znc = znc.sort_values(by=["model"])
znc = znc.groupby("set", group_keys=True).apply(lambda x:x)
znc = znc[["BA", "EF1%", "EF25%", "model"]].reset_index(drop=True)

div = data_true_metrics[data_true_metrics["Decoys"] == "div"]
div["model"] = pd.Categorical(div.model, ordered=True, categories=order)
div = div.sort_values(by=["model"])
div = div.groupby("set", group_keys=True).apply(lambda x:x)
div = div[["BA", "EF1%", "EF25%", "model"]].reset_index(drop=True)

### Create the heatmap
fig, ax = plt.subplots(1,8, figsize=(12,15))
fig.subplots_adjust(wspace=0.03)
sns.set(font_scale=1)

g = sns.heatmap(ax = ax[0], data = true1[["BA"]], annot=True,cmap="Blues", cbar=False, vmax=1, vmin=0.3, yticklabels=False)
sns.heatmap(ax = ax[1], data = true1[["EF1%", "EF25%"]], annot=True,cmap="Reds",  yticklabels=False, cbar=False, vmax=5.5, vmin=0)
sns.heatmap(ax = ax[2], data = dcm[["BA"]], annot=True,cmap="Blues", cbar=False, yticklabels=False, vmax=1, vmin=0.3)
sns.heatmap(ax = ax[3], data = dcm[["EF1%", "EF25%"]], annot=True,cmap="Reds",  yticklabels=False, cbar=False, vmax=5.5, vmin=0)
sns.heatmap(ax = ax[4], data = znc[["BA"]], annot=True,cmap="Blues", cbar=False, yticklabels=False,vmax=1, vmin=0.3)
sns.heatmap(ax = ax[5], data = znc[["EF1%", "EF25%"]], annot=True,cmap="Reds",  yticklabels=False, cbar=False, vmax=5.5, vmin=0)
sns.heatmap(ax = ax[6], data = div[["BA"]], annot=True,cmap="Blues", cbar=False, yticklabels=False, vmax=1, vmin=0.3)  
sns.heatmap(ax = ax[7], data = div[["EF1%", "EF25%"]], annot=True,cmap="Reds",  yticklabels=False, cbar=False, vmax=5.5, vmin=0)

for val in range(8):
    ax[val].set(ylabel="")
    ax[val].xaxis.tick_top()
    ax[val].xaxis.label.set_size(0.5)
    [ax[val].axhline(x, color = 'black', linestyle='--') for x in range(0,50,5)]

plt.figtext(0.22,0.92,"True decoys", va="center", ha="center", size=16)
plt.figtext(0.42,0.92,"DCM decoys", va="center", ha="center", size=16)
plt.figtext(0.61,0.92,"ZNC decoys", va="center", ha="center", size=16)
plt.figtext(0.80,0.92,"DIV decoys", va="center", ha="center", size=16)

label_group_bar_table(g, true)

plt.savefig("pictures/Heatmap_BA-ER_true_test-set.png", dpi=300, bbox_inches="tight")

### Metrics for fingerprint split

all_data = []
mts_fp = []
for target in tqdm(targets):

    lista = []
    data_x = []
    test_list = []

    for type in types1:
        print(f'{target} from {type}')

        er = []
        bal_acc = []
        models = []         
        ref = pd.read_parquet(f'files/ml-models/{target}/{type}/data_to_model/Fingerprint/Test.parquet')
        ref = ref.drop(["id", "smiles"], axis=1)
        test = pd.read_parquet(f'files/ml-models/{target}/true/data_to_model/Fingerprint/Test.parquet')
        test = test.drop(["id", "smiles"], axis=1)
        test_f = pre_test(ref, test) 
        test_list.append(test_list)
        data_unk = []       

        for model in glob.glob(f'files/ml-models/{target}/{type}/models/Fingerprint/*.pkl'):
            md = model[:-4]
            md2 = load_model(md)
            pred = predict_model(md2, test_f)
            act = pred[pred["prediction_label"] == 1].sort_values(["score"],ascending=False)
            ina = pred[pred["prediction_label"] == 0].sort_values(["score"],ascending=False)
            pred = pd.concat([act, ina])
            scores = [[x] for x in pred.activity]
            bal_acc.append(round(balanced_accuracy_score(pred.activity, pred.prediction_label),4))
            er.append(CalcEnrichment(scores, 0, [0.01, 0.25]))
            models.append(model.split("/")[-1].split(".")[0])
            pred['model'] = model.split("/")[-1].split(".")[0]
            data_unk.append(pred)

        metrics = pd.DataFrame(
            list(zip(bal_acc, models)), 
            columns=["BA", "model"]
        )

        enrichment = pd.DataFrame(er, columns=["EF1%", "EF25%"]) 
        metrics = pd.concat([metrics, enrichment], axis=1)
        metrics["Decoys"] = type

        data_u = pd.concat(data_unk)
        data_u['Decoys'] = type   
        data_x.append(data_u)
        
        lista.append(metrics)
    
    final = pd.concat(lista).reset_index(drop=True)
    final["name"] =  target

    d = pd.concat(data_x)
    d['name'] = target
    all_data.append(d)
    mts_fp.append(final)

### concat the metrics
true_ina_fp = pd.concat(mts_fp).reset_index(drop=True)
order = ['true',"dcm", "znc", "div"]
true_ina_fp["Decoys"] = pd.Categorical(true_ina_fp.Decoys, ordered=True, categories=order)
true_ina_fp = true_ina_fp.sort_values(by=["Decoys"])
true_ina_fp = true_ina_fp.merge(df, on='name')

chemplp_score = []
for target in tqdm(targets):

    lista = []

    for type in types1:

        er = []
        bal_acc = []         
        test = pd.read_parquet(f'files/ml-models/{target}/{type}/data_to_model/Fingerprint/Test.parquet')     
 
        test = test.astype({"score":"float64"})
        test = test.sort_values(by=["score"], ascending=False)
        test["prediction_label"] = [1 for val in range(int(len(test)*0.25))] + [0 for val in range(len(test) - int(len(test)*0.25))]
        act = test[test["prediction_label"] == 1].sort_values(["score"],ascending=False)
        ina = test[test["prediction_label"] == 0].sort_values(["score"],ascending=False)
        pred = pd.concat([act, ina])
        scores = [[x] for x in pred.activity]
        er.append(CalcEnrichment(scores, 0, [0.01, 0.25]))
        
        metrics = pd.DataFrame(er, columns=['EF1%', 'EF25%'])
        metrics['BA'] = round(balanced_accuracy_score(pred.activity, pred.prediction_label),4)
        metrics["Decoys"] = type   
        
        lista.append(metrics)
    
    final = pd.concat(lista).reset_index(drop=True)
    final["name"] =  target

    chemplp_score.append(final)

cs = pd.concat(chemplp_score)
cs = cs.merge(df, on='name')
cs['model'] = 'chemplp_score'

data_fp = pd.concat([true_ina_fp, cs]).reset_index(drop=True)

order = ["rf", "svm", "xgboost", "mlp", "chemplp_score"]
true_fp = data_fp[data_fp["Decoys"] == "true"]
true_fp["model"] = pd.Categorical(true_fp.model, ordered=True, categories=order)
true_fp = true_fp.sort_values(by=["model"])
true_fp.index = true_fp.model 
true_fp = true_fp.groupby("set", group_keys=True).apply(lambda x:x)
true_fp1 = true_fp[["BA", "EF1%", "EF25%", "model"]].reset_index(drop=True)

dcm_fp = data_fp[data_fp["Decoys"] == "dcm"]
dcm_fp["model"] = pd.Categorical(dcm_fp.model, ordered=True, categories=order)
dcm_fp = dcm_fp.sort_values(by=["model"])
dcm_fp = dcm_fp.groupby("set", group_keys=True).apply(lambda x:x)
dcm_fp = dcm_fp[["BA", "EF1%", "EF25%", "model"]].reset_index(drop=True)

znc_fp = data_fp[data_fp["Decoys"] == "znc"]
znc_fp["model"] = pd.Categorical(znc_fp.model, ordered=True, categories=order)
znc_fp = znc_fp.sort_values(by=["model"])
znc_fp = znc_fp.groupby("set", group_keys=True).apply(lambda x:x)
znc_fp = znc_fp[["BA", "EF1%", "EF25%", "model"]].reset_index(drop=True)

div_fp = data_fp[data_fp["Decoys"] == "div"]
div_fp["model"] = pd.Categorical(div_fp.model, ordered=True, categories=order)
div_fp = div_fp.sort_values(by=["model"])
div_fp = div_fp.groupby("set", group_keys=True).apply(lambda x:x)
div_fp = div_fp[["BA", "EF1%", "EF25%", "model"]].reset_index(drop=True)

fig, ax = plt.subplots(1,8, figsize=(12,15))
fig.subplots_adjust(wspace=0.03)
sns.set(font_scale=1)

g = sns.heatmap(ax = ax[0], data = true_fp1[["BA"]], annot=True,cmap="Blues", cbar=False, vmax=1, vmin=0.3, yticklabels=False)
sns.heatmap(ax = ax[1], data = true_fp1[["EF1%", "EF25%"]], annot=True,cmap="Reds",  yticklabels=False, cbar=False, vmax=5.5, vmin=0)
sns.heatmap(ax = ax[2], data = dcm_fp[["BA"]], annot=True,cmap="Blues", cbar=False, yticklabels=False, vmax=1, vmin=0.3)
sns.heatmap(ax = ax[3], data = dcm_fp[["EF1%", "EF25%"]], annot=True,cmap="Reds",  yticklabels=False, cbar=False, vmax=5.5, vmin=0)
sns.heatmap(ax = ax[4], data = znc_fp[["BA"]], annot=True,cmap="Blues", cbar=False, yticklabels=False,vmax=1, vmin=0.3)
sns.heatmap(ax = ax[5], data = znc_fp[["EF1%", "EF25%"]], annot=True,cmap="Reds",  yticklabels=False, cbar=False, vmax=5.5, vmin=0)
sns.heatmap(ax = ax[6], data = div_fp[["BA"]], annot=True,cmap="Blues", cbar=False, yticklabels=False, vmax=1, vmin=0.3)  
sns.heatmap(ax = ax[7], data = div_fp[["EF1%", "EF25%"]], annot=True,cmap="Reds",  yticklabels=False, cbar=False, vmax=5.5, vmin=0)

for val in range(8):
    ax[val].set(ylabel="")
    ax[val].xaxis.tick_top()
    ax[val].xaxis.label.set_size(0.5)
    [ax[val].axhline(x, color = 'black', linestyle='--') for x in range(0,50,5)]

plt.figtext(0.22,0.92,"True decoys", va="center", ha="center", size=16)
plt.figtext(0.42,0.92,"DCM decoys", va="center", ha="center", size=16)
plt.figtext(0.61,0.92,"ZNC decoys", va="center", ha="center", size=16)
plt.figtext(0.80,0.92,"DIV decoys", va="center", ha="center", size=16)

label_group_bar_table(g, true)

plt.savefig("pictures/Heatmap_BA-ER_Fingerprint-set.png", dpi=300, bbox_inches="tight")

### 
### Metrics for scaffold split

all_data = []
mts_scf = []
for target in tqdm(targets):

    lista = []
    data_x = []
    test_list = []

    for type in types1:
        print(f'{target} from {type}')

        er = []
        bal_acc = []
        models = []         
        ref = pd.read_parquet(f'files/ml-models/{target}/{type}/data_to_model/Scaffold/Test.parquet')
        ref = ref.drop(["id", "smiles"], axis=1)
        test = pd.read_parquet(f'files/ml-models/{target}/true/data_to_model/Scaffold/Test.parquet')
        test = test.drop(["id", "smiles"], axis=1)
        test_f = pre_test(ref, test) 
        test_list.append(test_list)
        data_unk = []       

        for model in glob.glob(f'files/ml-models/{target}/{type}/models/Scaffold/*.pkl'):
            md = model[:-4]
            md2 = load_model(md)
            pred = predict_model(md2, test_f)
            act = pred[pred["prediction_label"] == 1].sort_values(["score"],ascending=False)
            ina = pred[pred["prediction_label"] == 0].sort_values(["score"],ascending=False)
            pred = pd.concat([act, ina])
            scores = [[x] for x in pred.activity]
            bal_acc.append(round(balanced_accuracy_score(pred.activity, pred.prediction_label),4))
            er.append(CalcEnrichment(scores, 0, [0.01, 0.25]))
            models.append(model.split("/")[-1].split(".")[0])
            pred['model'] = model.split("/")[-1].split(".")[0]
            data_unk.append(pred)

        metrics = pd.DataFrame(
            list(zip(bal_acc, models)), 
            columns=["BA", "model"]
        )

        enrichment = pd.DataFrame(er, columns=["EF1%", "EF25%"]) 
        metrics = pd.concat([metrics, enrichment], axis=1)
        metrics["Decoys"] = type

        data_u = pd.concat(data_unk)
        data_u['Decoys'] = type   
        data_x.append(data_u)
        
        lista.append(metrics)
    
    final = pd.concat(lista).reset_index(drop=True)
    final["name"] =  target

    d = pd.concat(data_x)
    d['name'] = target
    all_data.append(d)
    mts_scf.append(final)

### concat the metrics
true_ina_scf = pd.concat(mts_scf).reset_index(drop=True)
order = ['true',"dcm", "znc", "div"]
true_ina_scf["Decoys"] = pd.Categorical(true_ina_scf.Decoys, ordered=True, categories=order)
true_ina_scf = true_ina_scf.sort_values(by=["Decoys"])
true_ina_scf = true_ina_scf.merge(df, on='name')

chemplp_score = []
for target in tqdm(targets):

    lista = []

    for type in types1:

        er = []
        bal_acc = []         
        test = pd.read_parquet(f'files/ml-models/{target}/{type}/data_to_model/Scaffold/Test.parquet')     
 
        test = test.astype({"score":"float64"})
        test = test.sort_values(by=["score"], ascending=False)
        test["prediction_label"] = [1 for val in range(int(len(test)*0.25))] + [0 for val in range(len(test) - int(len(test)*0.25))]
        act = test[test["prediction_label"] == 1].sort_values(["score"],ascending=False)
        ina = test[test["prediction_label"] == 0].sort_values(["score"],ascending=False)
        pred = pd.concat([act, ina])
        scores = [[x] for x in pred.activity]
        er.append(CalcEnrichment(scores, 0, [0.01, 0.25]))
        
        metrics = pd.DataFrame(er, columns=['EF1%', 'EF25%'])
        metrics['BA'] = round(balanced_accuracy_score(pred.activity, pred.prediction_label),4)
        metrics["Decoys"] = type   
        
        lista.append(metrics)
    
    final = pd.concat(lista).reset_index(drop=True)
    final["name"] =  target

    chemplp_score.append(final)

cs = pd.concat(chemplp_score)
cs = cs.merge(df, on='name')
cs['model'] = 'chemplp_score'

data_scf = pd.concat([true_ina_scf, cs]).reset_index(drop=True)

order = ["rf", "svm", "xgboost", "mlp", "chemplp_score"]
true_scf = data_scf[data_scf["Decoys"] == "true"]
true_scf["model"] = pd.Categorical(true_scf.model, ordered=True, categories=order)
true_scf = true_scf.sort_values(by=["model"])
true_scf.index = true_scf.model 
true_scf = true_scf.groupby("set", group_keys=True).apply(lambda x:x)
true_scf1 = true_scf[["BA", "EF1%", "EF25%", "model"]].reset_index(drop=True)

dcm_scf = data_scf[data_scf["Decoys"] == "dcm"]
dcm_scf["model"] = pd.Categorical(dcm_scf.model, ordered=True, categories=order)
dcm_scf = dcm_scf.sort_values(by=["model"])
dcm_scf = dcm_scf.groupby("set", group_keys=True).apply(lambda x:x)
dcm_scf = dcm_scf[["BA", "EF1%", "EF25%", "model"]].reset_index(drop=True)

znc_scf = data_scf[data_scf["Decoys"] == "znc"]
znc_scf["model"] = pd.Categorical(znc_scf.model, ordered=True, categories=order)
znc_scf = znc_scf.sort_values(by=["model"])
znc_scf = znc_scf.groupby("set", group_keys=True).apply(lambda x:x)
znc_scf = znc_scf[["BA", "EF1%", "EF25%", "model"]].reset_index(drop=True)

div_scf = data_scf[data_scf["Decoys"] == "div"]
div_scf["model"] = pd.Categorical(div_scf.model, ordered=True, categories=order)
div_scf = div_scf.sort_values(by=["model"])
div_scf = div_scf.groupby("set", group_keys=True).apply(lambda x:x)
div_scf = div_scf[["BA", "EF1%", "EF25%", "model"]].reset_index(drop=True)

fig, ax = plt.subplots(1,8, figsize=(12,15))
fig.subplots_adjust(wspace=0.03)
sns.set(font_scale=1)

g = sns.heatmap(ax = ax[0], data = true_scf1[["BA"]], annot=True,cmap="Blues", cbar=False, vmax=1, vmin=0.3, yticklabels=False)
sns.heatmap(ax = ax[1], data = true_scf1[["EF1%", "EF25%"]], annot=True,cmap="Reds",  yticklabels=False, cbar=False, vmax=5.5, vmin=0)
sns.heatmap(ax = ax[2], data = dcm_scf[["BA"]], annot=True,cmap="Blues", cbar=False, yticklabels=False, vmax=1, vmin=0.3)
sns.heatmap(ax = ax[3], data = dcm_scf[["EF1%", "EF25%"]], annot=True,cmap="Reds",  yticklabels=False, cbar=False, vmax=5.5, vmin=0)
sns.heatmap(ax = ax[4], data = znc_scf[["BA"]], annot=True,cmap="Blues", cbar=False, yticklabels=False,vmax=1, vmin=0.3)
sns.heatmap(ax = ax[5], data = znc_scf[["EF1%", "EF25%"]], annot=True,cmap="Reds",  yticklabels=False, cbar=False, vmax=5.5, vmin=0)
sns.heatmap(ax = ax[6], data = div_scf[["BA"]], annot=True,cmap="Blues", cbar=False, yticklabels=False, vmax=1, vmin=0.3)  
sns.heatmap(ax = ax[7], data = div_scf[["EF1%", "EF25%"]], annot=True,cmap="Reds",  yticklabels=False, cbar=False, vmax=5.5, vmin=0)

for val in range(8):
    ax[val].set(ylabel="")
    ax[val].xaxis.tick_top()
    ax[val].xaxis.label.set_size(0.5)
    [ax[val].axhline(x, color = 'black', linestyle='--') for x in range(0,50,5)]

plt.figtext(0.22,0.92,"True decoys", va="center", ha="center", size=16)
plt.figtext(0.42,0.92,"DCM decoys", va="center", ha="center", size=16)
plt.figtext(0.61,0.92,"ZNC decoys", va="center", ha="center", size=16)
plt.figtext(0.80,0.92,"DIV decoys", va="center", ha="center", size=16)

label_group_bar_table(g, true)

plt.savefig("pictures/Heatmap_BA-ER_Scaffold-set.png", dpi=300, bbox_inches="tight")

### Mean values for each splitter


fig, axes = plt.subplots(2, 10, figsize=(14, 6))
fig.subplots_adjust(wspace=0.03, hspace=0.04) 

dataframes = [tr_fp, dcm_fp, znc_fp, div_fp, cs1_fp]
dataframes_2 = [tr_scf, dcm_scf, znc_scf, div_scf, cs_scf]

for i, df in enumerate(dataframes*2):
    if i == 0:
        sns.heatmap(ax = axes[0,i], data = df[["BA"]], annot=True,cmap="Blues", cbar=False, vmax=1, vmin=0.3)
    elif i % 2 == 0:
        sns.heatmap(ax = axes[0,i], data = df[["BA"]], annot=True,cmap="Blues", cbar=False, vmax=1, vmin=0.3,  yticklabels=False)
    else:
        sns.heatmap(ax = axes[0,i], data = df[["EF1%", "EF25%"]], annot=True,cmap="Reds",  yticklabels=False, cbar=False, vmax=5.5, vmin=0)

for i, df in enumerate(dataframes_2*2):
    if i == 0:
        sns.heatmap(ax = axes[1,i], data = df[["BA"]], annot=True,cmap="Blues", cbar=False, vmax=1, vmin=0.3)
    elif i % 2 == 0:
        sns.heatmap(ax = axes[1,i], data = df[["BA"]], annot=True,cmap="Blues", cbar=False, vmax=1, vmin=0.3,  yticklabels=False)
    else:
        sns.heatmap(ax = axes[1,i], data = df[["EF1%", "EF25%"]], annot=True,cmap="Reds",  yticklabels=False, cbar=False, vmax=5.5, vmin=0)
        
for val in range(10):
    axes[0,val].xaxis.tick_top()
    axes[1, val].set_xticklabels([])
    for val2 in range(2):
        axes[val2, val].set(ylabel="")

axes[0,0].set(ylabel='Fingerprint')
axes[1,0].set(ylabel='Scaffold')
    

plt.figtext(0.21,0.98,"True decoys", va="center", ha="center", size=12)
plt.figtext(0.36,0.98,"DCM decoys", va="center", ha="center", size=12)
plt.figtext(0.51,0.98,"ZNC decoys", va="center", ha="center", size=12)
plt.figtext(0.66,0.98,"DIV decoys", va="center", ha="center", size=12)
plt.figtext(0.83,0.98,"ChemPLP score", va="center", ha="center", size=12)

plt.savefig('pictures/Heatmap_BA_ER_splitters.png', dpi=300, bbox_inches="tight")