"""
Recency, Frequency, Monetary
============================

RFM model combines three different customer attributes to rank customers:

    Recency (R): Who's bought lately? The longer it's been since their last purchase, the less valuable they are.
    Frequency (F): Who's bought a lot? Customers who've made more purchases get a higher score.
    Monetary Value(M): Who's spent a lot? Customers who've spent more money overall get more points.
    
How to Create Segments
======================

1. Produce the RFM dataset via identifying 4 key informations: 1) customer id; order id; 2) order amount; 3) order date
2. Create bins or segments for each of the three RFM components, here we can use `Pandas.qcut` function
3. Segment customers based on a set of rules

Credit
======

this code is greatly inspired and borrowed from [rfm: Python Package for RFM Analysis and Customer Segmentation](https://github.com/sonwanesuresh95/rfm/tree/main)
"""

import pandas as pd
import numpy as np
import warnings
import janitor
import matplotlib.pyplot as plt
from typing import Dict
from collections import namedtuple
warnings.filterwarnings('ignore')
import seaborn as sns

Rule = namedtuple('Rule', ['R', 'F', 'M'], defaults = [None, None, None])


def cut_label(df, colname, new_colname, q, label):
    df[new_colname] = pd.qcut(df[colname].rank(method='first'), q, label).astype(int)

def _freq_plot(data, label, savepath = ''):
    assert f'rfm_{label}' in data, "label doesn't exist in the dataset"
    fig, ax = plt.subplots(figsize=(10,4))
    palette = sns.color_palette("Blues_r", n_colors=13)
    sns.countplot(x=f'rfm_{label}', data=data, palette=palette)
    ax.set_title('Number of customers - %s'%label)
    if label == 'label':
        ax.tick_params(axis='x', rotation=-45)
    if savepath: fig.savefig(savepath)

class RFM:
    def __init__(self, df: pd.DataFrame, customer_id:str = None, order_id:str = None, order_amount:str = None, order_date:str = None, automated=True): # type: ignore
        """
        #### df: dataset to make RFM analysis on 
            - make sure df[{order_amount}] is numeric type,  \n
              you can use janitor.currency_column_to_numeric function
            - make sure df[{customer_id}] exists, \n
              you can use janitor.count_cumulative_unique function to obtain them

        ```py
        df = (
            pd.read_csv(ojoin(PATH, 'superstore.csv'))
            .pipe(clean_names)
            .count_cumulative_unique('customer_name', 'customer_id') 
            .currency_column_to_numeric("sales")
            .rename(columns = {'sales': 'order_amount'})
        )  
        >>> rfm = RFM(df)
        >>> rtn = rfm.rfm_df
        >>> path = lambda f: osimplify(ojoin(__file__, '../..', 'figs', f))
        >>> rfm.get_label_dist(
            rtn, label='frequency', savepath=path('dist_of_freq.png')
            segments=['loyal customers', 'hibernating', 'potential loyalist'], 
        )
        >>> rfm.get_freq_hist(
            rtn, label_path=path('count_of_label.png'), 
            score_path=path('count_of_score.png')
        )
        ```
        """
        self.df = df
        self.customer_id = customer_id or 'customer_id'
        self.order_id = order_id or 'order_id'
        self.order_amount = order_amount or 'order_amount'
        self.order_date = order_date or 'order_date'
        self.labels = ['recency', 'frequency', 'monetary']
        self.scores = ['r_score', 'f_score', 'm_score']
        self.rfm_df = pd.DataFrame()
        if automated:
            rfm_df = self.produce_rfm_dateset(df.copy())
            rfm_df = self.calculate_rfm_score(rfm_df)
            self.rfm_df = self.get_rfm_segment(
            df = rfm_df,
            rules = { 
                'champions': Rule((5,5),(4,5)),
                'loyal customers': Rule((3,4),(4,5)),
                'potential loyalist': Rule((4,5),(2,3)),
                'new customers': Rule((5,5),(1,1)),
                'promising': Rule((4,4),(1,1)),
                'needing attention': Rule((3,3),(3,3)),
                'about to sleep': Rule((3,3),(1,2)),
                'at risk': Rule((1,2),(3,4)),
                'cant loose them': Rule((1,2),(5,5)),
                'hibernating': Rule((1,2),(1,2))
            })

    @staticmethod
    def read_csv(path, **kwargs):
        df = pd.read_csv(path, **kwargs)
        return df.pipe(janitor.clean_names)
    def produce_rfm_dateset(self, df: pd.DataFrame = None) -> pd.DataFrame: # type: ignore
        """
        Produce the RFM dataset (df), grouping by customer id, finding RFM values
        """
        #| 1. convert date to python datetime type
        df[self.order_date] = pd.to_datetime(df[self.order_date])
        df = df.sort_values(by=self.order_date, na_position='first')
        #| 2. drop missing value and duplicates
        rfm_cols = [self.customer_id,self.order_id, self.order_amount, self.order_date]
        df = df.dropna(subset=rfm_cols)
        df = df.drop_duplicates()
        df = df.reset_index().drop(columns=['index'], axis=1)
        #| 3. group by customer id and find R,F,M values
        TODAY = df[self.order_date].max()
        agg_fn = {  self.order_id: 'count', self.order_date: 'max', self.order_amount: 'sum' }
        df_grp = df.groupby(self.customer_id).agg(agg_fn).reset_index()
        df_grp.columns = ['customer_id', 'frequency', 'latest_order', 'monetary']
        df_grp['recency'] = (TODAY - df_grp['latest_order']).dt.days 
        df_grp = df_grp.drop(['latest_order'], axis=1)
        return df_grp
    def calculate_rfm_score(self, df: pd.DataFrame, q = 5):
        """
        Calculates RFM scores based on RFM-dataframe
        """
        cut_label(df, 'recency', 'r_score', q, range(q,0,-1))
        cut_label(df, 'frequency', 'f_score', q, range(1,q+1))
        cut_label(df, 'monetary', 'm_score', q, range(1,q+1))
        df['rfm_score'] = df['r_score'] + df['f_score'] + df['m_score']
        df = df.sort_values(by='rfm_score',ascending=False).reset_index(drop=True)
        return df
    def get_rfm_segment(self, df: pd.DataFrame, rules: Dict[str, Rule]):
        """
        Finds customer segments based on the rfm scores
        >>> rules = {'At Risk': Rule((,2), (1,5), (1,2))}
        """
        def fix(df: pd.DataFrame, x: Rule):
            r_rule = df['r_score'].between(*x.R) if x.R else True
            f_rule = df['f_score'].between(*x.F) if x.F else True
            m_rule = df['m_score'].between(*x.M) if x.M else True
            return r_rule & f_rule & m_rule
        df['rfm_label'] = ""
        for label, rule in rules.items():
            df.loc[fix(df, rule), 'rfm_label'] = label
        return df
    def get_label_dist(self, df: pd.DataFrame, label: str, segments: list = [], savepath: str = ''):
        assert label in self.labels, f"label must be one of {self.labels}"
        fig, ax = plt.subplots(figsize=(7,4))
        if len(segments) == 0:
            sns.distplot(df[label])
        else:
            for segment in segments:
                sub = df.query('rfm_label=="%s"'%segment)
                sns.distplot(sub[label], label=segment)
        ax.set_title('Distribution of %s' % label)
        ax.legend()
        if savepath: fig.savefig(savepath)
    def get_freq_hist(self, df: pd.DataFrame, label_path=None, score_path=None):
        if label_path: _freq_plot(df, 'label', label_path)
        if score_path: _freq_plot(df, 'score', score_path)

    