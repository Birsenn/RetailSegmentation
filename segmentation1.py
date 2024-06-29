import numpy as np
import pandas as pd
import seaborn as sns
import datetime as dt
import matplotlib.pyplot as plt
from scipy import stats
from sklearn.preprocessing import MinMaxScaler
from sklearn.cluster import KMeans
from scipy.cluster.hierarchy import dendrogram
from scipy.cluster.hierarchy import linkage
from yellowbrick.cluster import KElbowVisualizer
from sklearn.cluster import AgglomerativeClustering


pd.set_option('display.max_columns', None)
pd.set_option('display.max_rows', None)
pd.set_option('display.float_format', lambda x: '%.3f' % x)
pd.set_option('display.width', 500)

df = pd.read_excel("january.xlsx")
df.head()
df.columns
df.shape #(893707, 12)
df.describe()
df.info()

df.columns = ['ID', 'SEGMENT', 'FREQUENCY', 'EVER_FREQUENCY', 'LAST90DAYS_FREQUENCY', 'QUANTITY', 'RECENCY', 'MONETARY', 'DISCOUNT',
              'POCKET', 'TOTAL_VOUCHER_AMOUNT', 'ORGANIC_RATIO']

df1 = df[['FREQUENCY', 'EVER_FREQUENCY', 'RECENCY', 'MONETARY']] #first step just used these columns
df1.head()

def check_skew(df_skew, column):
    skew = stats.skew(df_skew[column])
    skewtest = stats.skewtest(df_skew[column])
    plt.title('Distribution of ' + column)
    sns.distplot(df_skew[column],color = "g")
    print("{}'s: Skew: {}, : {}".format(column, skew, skewtest))
    return

plt.figure(figsize=(9, 9))
plt.subplot(6, 1, 1)
check_skew(df1,'FREQUENCY')
plt.subplot(6, 1, 2)
check_skew(df1,'EVER_FREQUENCY')
plt.subplot(6, 1, 3)
check_skew(df1,'RECENCY')
plt.subplot(6, 1, 4)
check_skew(df1,'MONETARY')
plt.tight_layout()
plt.savefig('before_transform.png', format='png', dpi=1000)
plt.show(block=True)


#Log transformation
df3 = df1.copy()
df3['FREQUENCY']=np.log1p(df3['FREQUENCY'])
df3['EVER_FREQUENCY']=np.log1p(df3['EVER_FREQUENCY'])
df3['RECENCY']=np.log1p(df3['RECENCY'])
df3['MONETARY']=np.log1p(df3['MONETARY'])

# Scaling
sc = MinMaxScaler((0, 1))
model_scaling = sc.fit_transform(df3)
df3=pd.DataFrame(model_scaling,columns=df3.columns)
df3.head()

#check the skewness
plt.figure(figsize=(9, 9))
plt.subplot(6, 1, 1)
check_skew(df3,'FREQUENCY')
plt.subplot(6, 1, 2)
check_skew(df3,'EVER_FREQUENCY')
plt.subplot(6, 1, 3)
check_skew(df3,'RECENCY')
plt.subplot(6, 1, 4)
check_skew(df3,'MONETARY')
plt.tight_layout()
plt.savefig('before_transform.png', format='png', dpi=1000)
plt.show(block=True)


#determine of the optimum clustering number
kmeans = KMeans()
elbow = KElbowVisualizer(kmeans, k=(2, 20))
elbow.fit(df3)
elbow.show()

#modelling and create segments
k_means = KMeans(n_clusters = 6, random_state= 42).fit(df3)
segments=k_means.labels_
segments

final_df = df[['FREQUENCY', 'EVER_FREQUENCY', 'RECENCY', 'MONETARY']]
final_df["SEGMENT"] = segments
final_df.head()

final_df['SEGMENT'].value_counts(normalize=True)
final_df.groupby("SEGMENT").agg({"FREQUENCY":["mean","min","max"],
                                  "EVER_FREQUENCY":["mean","min","max"],
                                  "RECENCY":["mean","min","max"],
                                  "MONETARY":["mean","min","max"]})

# Mapping
mapping = {
    0: 'champions',
    1: 'need_attention',
    2: 'at_risk',
    3: 'loyal',
    4: 'hibernating',
    5: 'potential_loyal'
}

final_df['NEW_SEGMENT'] = final_df['SEGMENT'].map(mapping)
final_df.head(10)

#after that, analyze all of the values like frequency, recency, monetary for all segments.
#Then new features will be added and create new segments.