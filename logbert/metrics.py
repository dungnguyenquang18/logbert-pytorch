import os
from datetime import datetime
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pytz
from sklearn.metrics import roc_curve, auc, confusion_matrix, ConfusionMatrixDisplay

vntz = pytz.timezone("Asia/Ho_Chi_Minh")


def visualize_roc_auc(y_true, y_scores, save_fig, output_dir="./"):
    fpr, tpr, thresholds = roc_curve(y_true, y_scores)
    roc_auc = auc(fpr, tpr)
    print(f"AUC: {roc_auc}")
    fig, ax = plt.subplots(nrows=1, ncols=1, figsize=(12,12))
    ax.plot(fpr, tpr, color='darkorange', lw=2, label='ROC curve (area = %0.3f)' % roc_auc)
    ax.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
    ax.set_xlim([0.0, 1.0])
    ax.set_ylim([0.0, 1.05])
    ax.set_xlabel('False Positive Rate')
    ax.set_ylabel('True Positive Rate')
    ax.set_title('ROC Curve')
    ax.legend(loc="lower right")

    if save_fig:
        os.makedirs(output_dir, exist_ok=True)
        save_time = datetime.now(vntz).strftime("%d-%m-%y_%H-%M")
        fig.savefig(os.path.join(output_dir, f"roc_auc_viz_{save_time}"))
        print(f"ROC AUC viz saved at {os.path.join(output_dir, f'roc_auc_viz_{save_time}')}")
        plt.close()


def plot_cm(y, pred, out):
    os.makedirs(os.path.dirname(out), exist_ok=True)
    disp = ConfusionMatrixDisplay(confusion_matrix(y, pred))
    disp.plot(colorbar=False)
    plt.savefig(out)
    plt.close()


def return_percentile_gain_chart(pred_df,true_col="y_true",y_pred="y_pred",y_proba="y_proba",number_of_thresholds=10,save_fig=False,output_dir=None,plot_name=None):
    df=pred_df.copy().sort_values(by=y_proba,ascending=False).reset_index(drop=True)

    df['decile']=pd.qcut(df.index,q=number_of_thresholds,labels=False,duplicates="drop")+1
    
    total_events=df[true_col].sum()
    gain_table=df.groupby('decile').agg(
        No_of_Observations=(true_col,'count'),
        Number_of_Events=(true_col,'sum'),
        Non_Events=(true_col,lambda x:(x==0).sum()),
    ).reset_index()

    gain_table['Cumulative_Events']=gain_table.Number_of_Events.cumsum()
    gain_table['Cumulative_Gain (%)']=(gain_table['Cumulative_Events']/total_events*100).round(2)
    
    gain_table['Cumulative_Observations']=gain_table.No_of_Observations.cumsum()
    gain_table['Precision_Pct'] = (gain_table['Cumulative_Events']/gain_table['Cumulative_Observations']*100).round(2)
    gain_table['decile_low']=number_of_thresholds+1-gain_table['decile']

    bar_table=gain_table.sort_values(by='decile_low')

    fig,ax1=plt.subplots(figsize=(18,6))
    x=bar_table['decile_low']
    ax1.bar(x,bar_table['Number_of_Events'],label='Down State',color='skyblue',alpha=0.6)
    ax1.bar(x,bar_table['Non_Events'],bottom=bar_table['Number_of_Events'],label='Normal State',color='orange',alpha=0.6)
    ax1.set_xlabel('Decile (Low -> High Probability of Down)')
    ax1.set_ylabel('Count')
    ax1.set_xticks(x)
    ax1.set_xticklabels(x.astype(str),fontsize=8)
    ax1.legend(bbox_to_anchor=(1.22,0.85),loc='upper right')

    ax2=ax1.twinx()
    x_line=number_of_thresholds+1-gain_table['decile']
    ax2.plot(x_line,gain_table['Cumulative_Gain (%)'],marker='o',color='red',label='Cumulative Coverage (%)')
    random_line=x_line*(gain_table['Cumulative_Gain (%)'].max()/x_line.max())
    ax2.plot(x_line,random_line,linestyle='--',color='grey',label='Random Model')
    ax2.plot(x,bar_table['Precision_Pct'],marker='s',linestyle='-',color='green',label='Cumulative Precision (%)')
    ax2.set_ylabel('Percentage')

    # Annotate
    for xi,yi in zip(x_line,gain_table['Cumulative_Gain (%)']):
        ax2.text(xi,yi+1.1,f"{yi:.2f}%",ha='center',va='bottom',fontsize=10,color='red')
    for xi,yi in zip(x,bar_table['Precision_Pct']):
        ax2.text(xi,yi+1.1,f"{yi:.2f}%",ha='center',va='bottom',fontsize=10,color='green')

    ax2.legend(bbox_to_anchor=(1.22,1),loc='upper right')
    if plot_name:
        plt.title(plot_name)
    plt.tight_layout()

    if save_fig and output_dir:
        os.makedirs(output_dir,exist_ok=True)
        fig.savefig(os.path.join(output_dir,plot_name.replace(' ','_').lower()+f'_{datetime.now(vntz).strftime("%d-%m-%y_%H-%M")}.png'),bbox_inches='tight',dpi=300)
