import json
import pandas as pd
import numpy as np
import argparse
import seaborn as sns
import matplotlib.pyplot as plt
import os
from sklearn.metrics import cohen_kappa_score

parser = argparse.ArgumentParser()
parser.add_argument('--expt', type=str, default='bin') # Should match with `prompts/<expt>/`
parser.add_argument('--model', type=str, default='openaichat') 

args = parser.parse_args()

RESULTS_FNAME = 'data/annotations/sample_dataset.csv' 
LABEL_PROMPTS_DIR = os.path.join('prompts', args.expt, 'label_prompts')
PLOT_DIR = 'results/'
RESULTS_DIR = 'results'
model_name  = args.model.replace('/', '-')


VALID_NAMES = ['H1', 'H2', 'openaichat']

VALID_LABELS = [
    'general', 
    'confusion', 
    'pedagogy', 
    'setup', 
    'gratitude', 
    'personal_experience', 
    'clarification', 
    'non_english', 
    'na'
    ]

EXPT2LABELS = {
    'bin_zeroshot': "0-shot", 
    'bin_kshot': "3-shot",
    'bin_kshot_reasoning': "3-shot-R",
}


COLORS = [ # From Paul Tol
    '#EE8866', # orange
    '#77AADD', # blue
]

labels2ver = dict()

def calculate_interannotator(datasplit_df, labels2ver):
    interannotator_df = [] # {'label': ##, 'H1-H2': ##, 'H1-M': ##, 'H2-M': ##}

    label2human_agreement = {}
    label2human_model_agreement = {}

    # Run analysis for each label (comment category)
    for label in VALID_LABELS:
        # print('Label:', label)

        # Get annotation per name
        name2key = dict()
        name2annotation = dict()
        for name in VALID_NAMES:
            if name == "openaichat":
                label_with_ver = labels2ver[label]
                annotation_key = f'annotator_{name}_{label_with_ver}'
                
            else:
                annotation_key = f'annotator_{name}_{label}'
            
            name2key[name] = annotation_key

            # Get the annotation for this name where it is not null
            name2annotation[name] = datasplit_df[datasplit_df[annotation_key].notnull()]

        for i in range(len(VALID_NAMES)):
            for j in range(i + 1, len(VALID_NAMES)):
                name1, name2 = VALID_NAMES[i], VALID_NAMES[j]

                # Ensure that the annotations are ordered by comment_id
                a_key = name2key[name1]
                b_key = name2key[name2]
                a_annotations = name2annotation[name1].sort_values('comment_id')[a_key].values
                b_annotations = name2annotation[name2].sort_values('comment_id')[b_key].values
                a_comment_ids = set(name2annotation[name1]['comment_id'].values)
                b_comment_ids = set(name2annotation[name2]['comment_id'].values)
                common_comment_ids = a_comment_ids.intersection(b_comment_ids)
                if len(a_annotations) != len(b_annotations):
                    # Keep intersection of comment_ids
                    a_annotations = name2annotation[name1][name2annotation[name1]['comment_id'].isin(common_comment_ids)][a_key].values
                    b_annotations = name2annotation[name2][name2annotation[name2]['comment_id'].isin(common_comment_ids)][b_key].values
                    
                corr = cohen_kappa_score(a_annotations, b_annotations) 

                if name1 == 'H1' and name2 == 'H2': 
                    # Add to human agreement
                    label2human_agreement[label] = corr
                else: # human-model agreement
                    if label not in label2human_model_agreement:
                        label2human_model_agreement[label] = [corr]
                    else:
                        label2human_model_agreement[label].append(corr)

    # Create interannotator df
    for label in label2human_model_agreement.keys():
        for human_model_corr in label2human_model_agreement[label]:
            interannotator_df.append({
                'label': label,
                'H1-H2': label2human_agreement[label],
                'H-M': human_model_corr
            })
    return pd.DataFrame(interannotator_df)
    
def plot_regression(df):
    # Plot H1-H2 vs H1-M, and H1-H2 vs H2-M
    
    # Pretty plot
    sns.set_style("whitegrid")
    sns.set_context("paper", font_scale=1.5, rc={"lines.linewidth": 2.5})
    sns.set_palette("colorblind")
    # Enable latex
    plt.rc('text', usetex=True)

    # Plot H1-H2 vs H1-M, in COLORS[0], color friendly pale colors
    sns.scatterplot(data=df, x='H1-H2', y='H-M', hue='label', palette= sns.color_palette("colorblind", len(VALID_LABELS)))
    # Fit line 
    sns.regplot(data=df, x='H1-H2', y='H-M', color=COLORS[0], scatter=False)

    # \textbf{\texttt{human}}
    plt.xlabel(r'$\textbf{\texttt{human}}$')

    tag = EXPT2LABELS[args.expt]
    plt.ylabel(r'$\textbf{\texttt{' + tag + '}}$')

    # Set legend more up top with 3 columns
    plt.legend(bbox_to_anchor=(0.5, 1.30), loc='upper center', ncol=3, fontsize=14)

    # plt.show()
    fname = os.path.join(PLOT_DIR, f'regression_{args.expt}.pdf')
    plt.savefig(fname, bbox_inches='tight')
    print(f'Saved plot to {fname}')


def run_average_annotation(df):
    # Go through unique label -> print average H-M
    labels = df['label'].unique()
    for label in labels:
        label_df = df[df['label'] == label]
        print(f'Label: {label}')
        print(f"H1-H2: {label_df['H1-H2'].mean()}")
        print(f'Average H-M: {label_df["H-M"].mean()}')
        print()

def plot_category_distribution(datasplit_df, labels2ver):
    df = [] # {'label': ##, 'Annotator': H1/H2/M, 'value': 0/1, 'comment_id': ##}

    order_labels = []

    # Run analysis for each label (comment category)
    for comment_id in datasplit_df['comment_id'].unique():
        comment_df = datasplit_df[datasplit_df['comment_id'] == comment_id]

        # Check num rows
        if len(comment_df) > 1:
            import pdb; pdb.set_trace()

        for label in VALID_LABELS:
            for name in VALID_NAMES:
                if name == "openaichat":
                    label_with_ver = labels2ver[label]
                    annotation_key = f'annotator_{name}_{label_with_ver}'
                else:
                    annotation_key = f'annotator_{name}_{label}'

                if label == 'personal_experience':
                    df_label = 'personal'
                elif label == 'non_english':
                    df_label = 'nonenglish'
                else:
                    df_label = label

                if df_label not in order_labels:
                    order_labels.append(df_label)

                df.append({
                    'label': df_label,
                    'Annotator': name,
                    'value': comment_df[annotation_key].values[0],
                    'comment_id': comment_df['comment_id'].values[0]
                })
    
    # Count number of each label per annotator
    df = pd.DataFrame(df)

    # Map nan to 0 in value
    df['value'] = df['value'].fillna(0)

    count_df = df.groupby(['label', 'Annotator']).sum().reset_index()
    # Plot count, x is category, hue is annotator
    sns.set(style="whitegrid")
    plt.figure(figsize=(15, 5))

    # Order the x axis to be in the order of VALID_LABELS
    ax = sns.barplot(x="label", y="value", hue="Annotator", data=count_df, order=order_labels)
    ax = plt.gca()
    # ax.set_xticklabels(VALID_LABELS)
    plt.xlabel('Category')
    plt.ylabel('Count')
    # Save 
    plt.savefig(os.path.join(PLOT_DIR, f'category_distribution_{args.expt}.pdf'), bbox_inches='tight')
    # plt.show()

    # Print counts
    sum_h1=0
    for label in order_labels:
        label_df = df[df['label'] == label]
        print(f'Label: {label}')
        print(f"H1: {label_df[label_df['Annotator'] == 'H1']['value'].sum()}")
        sum_h1 += label_df[label_df['Annotator'] == 'H1']['value'].sum()
        print(f'H2: {label_df[label_df["Annotator"] == "H2"]["value"].sum()}')
        print(f'M: {label_df[label_df["Annotator"] == "openaichat"]["value"].sum()}')

        # Percentage of positive annotations from humans
        h1 = label_df[label_df['Annotator'] == 'H1']['value'].to_numpy()
        h2 = label_df[label_df['Annotator'] == 'H2']['value'].to_numpy()
        h = h1 + h2
        # Count number of at least 1 
        count = np.count_nonzero(h)
        total = len(h)
        perc = (count / total) * 100
        # Round to 2 decimal places
        perc = round(perc, 2)
        print(f"Count of at least 1 positive annotation from humans: {count}")
        print(f'Percentage of positive annotations from humans: {perc}%')
        print()



if __name__ == '__main__':
    # Populate labels2ver
    for prompt_file in os.listdir(LABEL_PROMPTS_DIR):
        label_with_ver = prompt_file[:-4] # removes ".txt" to get to label name 
        label_no_ver_start = label_with_ver.find("_") + 1
        label = label_with_ver[label_no_ver_start:]
        labels2ver[label] = label_with_ver

    # Load results csv
    df = pd.read_csv(RESULTS_FNAME)

    # Calculate inter-annotator agreement
    interannotator_df = calculate_interannotator(df, labels2ver)

    # Report average per label
    run_average_annotation(interannotator_df)

    # Plot regression
    plot_regression(interannotator_df)

    # Plot distribution of labels 
    plot_category_distribution(df, labels2ver)
