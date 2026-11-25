"""Matplotlib paper figures from the paired household MPC panel."""
from pathlib import Path
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


def write_paper_figures(panel, output_dir):
    """Unweighted boxes and means; transfer-weighted means as diamonds."""
    from src.mpc_analysis import assign_quantile_bin
    panel = panel.copy()
    specs = []
    for field, title in [('income', 'Household income'), ('labor_income', 'Labour income'),
                         ('non_property_income', 'Non-property income'), ('property_income', 'Property income'),
                         ('liquid_financial_assets', 'Liquid financial assets'),
                         ('net_liquid_financial_assets_all_debt', 'Net liquid assets'),
                         ('illiquid_financial_assets', 'Illiquid financial assets'),
                         ('housing_wealth', 'Housing wealth')]:
        group = field + '_paper_group'
        panel[group] = pd.Series(pd.NA, index=panel.index, dtype='string')
        if field in ('illiquid_financial_assets', 'housing_wealth', 'labor_income', 'property_income'):
            positive = panel[field] > 0
            panel.loc[panel[field] == 0, group] = 'Zero'
            panel.loc[panel[field] < 0, group] = 'Negative'
            panel.loc[positive, group] = assign_quantile_bin(panel.loc[positive, field]).astype('string')
        else:
            panel[group] = assign_quantile_bin(panel[field]).astype('string')
        specs.append((group, title, ['Negative', 'Zero', 'Q1', 'Q2', 'Q3', 'Q4', 'Q5']))
    panel['tenure'] = panel.housing_tenure.map({'renter': 'Renters', 'owner_mortgage': 'Mortgagors',
        'owner_outright': 'Outright\nowners', 'owner_other': 'Other tenure', 'unknown': 'Unknown'})
    panel['labour'] = panel.activity_bracket_t0.map({1:'Employed',2:'Unemployed',3:'Inactive',4:'Retired'})
    consumer = panel.consumption_loan_debt > 1e-8
    mortgage = panel.mortgage_debt > 1e-8
    panel['borrowing'] = np.select([consumer & mortgage, consumer, mortgage], ['Both','Consumer\ncredit','Mortgage'], default='None')
    panel['constraint'] = panel.mpc_regime.map({'liquidity_constrained':'Constrained','unconstrained':'Unconstrained'})
    specs += [('tenure','Housing tenure',['Renters','Mortgagors','Outright\nowners','Other tenure','Unknown']),
              ('labour','Reference person',['Employed','Unemployed','Inactive','Retired']),
              ('borrowing','Borrowing',['None','Consumer\ncredit','Mortgage','Both']),
              ('constraint','Liquidity constraint',['Constrained','Unconstrained'])]
    output_dir = Path(output_dir); output_dir.mkdir(parents=True, exist_ok=True)
    fig, axes = plt.subplots(3,4,figsize=(18,11),sharey=True)
    rows=[]
    for ax,(field,title,order) in zip(axes.flat,specs):
        arrays=[];labels=[];weighted=[]
        for label in order:
            subset=panel.loc[panel[field].eq(label).fillna(False)]
            valid=np.isfinite(subset.real_mpc_impact) & np.isfinite(subset.real_shock_amount) & (subset.real_shock_amount>0)
            subset=subset.loc[valid]
            if subset.empty: continue
            arrays.append(subset.real_mpc_impact);labels.append(label)
            weighted.append(np.average(subset.real_mpc_impact,weights=subset.real_shock_amount))
            rows.append(dict(variable=field,group=label,n=len(subset),mean=subset.real_mpc_impact.mean(),transfer_weighted_mpc=weighted[-1]))
        if arrays:
            ax.boxplot(arrays,patch_artist=True,showfliers=False,boxprops=dict(facecolor='#799AB7'),medianprops=dict(color='#14213D',linewidth=1.6))
            positions=np.arange(1,len(arrays)+1)
            ax.scatter(positions,[a.mean() for a in arrays],s=18,color='black',label='Household mean',zorder=4)
            ax.scatter(positions + .10,weighted,s=22,marker='D',color='#C46A27',label='Transfer-weighted mean',zorder=5)
            ax.set_xticks(positions,labels,rotation=0,fontsize=9)
        ax.set_title(title,fontsize=12);ax.axhline(0,color='grey',lw=.7)
        ax.spines[['top','right']].set_visible(False)
    fig.supylabel('MPC',fontsize=14)
    handles,labels=axes.flat[0].get_legend_handles_labels()
    fig.legend(handles,labels,loc='lower center',ncol=2,frameon=False)
    fig.tight_layout(rect=(.025,.045,1,1))
    for ext in ('pdf','png'): fig.savefig(output_dir/f'paper_impact_mpc.{ext}',dpi=300)
    plt.close(fig)
    pd.DataFrame(rows).to_csv(output_dir/'paper_figure_group_means.csv',index=False)
