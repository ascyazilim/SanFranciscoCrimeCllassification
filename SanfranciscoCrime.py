from sklearn.cluster import KMeans
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
from matplotlib import pyplot as plt
import plotly.express as px
import plotly

df = pd.read_csv("train.csv")
ilkbes = df.head()


df = df.drop(['PdDistrict', 'Address', 'Resolution', 'Descript', 'DayOfWeek'], axis = 1)

#filtreleme

f = lambda x: (x["Dates"].split())[0]   #Sadece Tarihi al zamandan kurtul
df["Dates"] = df.apply(f, axis=1)
df.head()


f = lambda x: (x["Dates"].split('-'))[0]
df["Dates"] = df.apply(f, axis=1) #sütun axis=1 satır axis=0
df.head()

#2014 yılı verilerini alma
df_2014 = df[(df.Dates == '2014')]
df_2014.head()

#Normalizasyon
scaler = MinMaxScaler()
#Y enlem X boylam
scaler.fit(df_2014[['X']])
df_2014['X_scaled'] = scaler.transform(df_2014[['X']])

scaler.fit(df_2014[['Y']])
df_2014['Y_scaled'] = scaler.transform(df_2014[['Y']])



k_range = range(1,15)

list_dist = []

for k in k_range:
    model = KMeans(n_clusters=k)
    model.fit(df_2014[['X_scaled', 'Y_scaled']])
    list_dist.append(model.inertia_)


plt.xlabel('K')
plt.ylabel('inertia')
plt.plot(k_range, list_dist)
plt.show()


#K-Means modeli oluşturma
model = KMeans(n_clusters=5)
y_predicted = model.fit_predict(df_2014[['X_scaled', 'Y_scaled']])

df_2014['cluster'] = y_predicted


#Haritalama Y Enlem(latitude) X Boylam(longitude)
figure = px.scatter_mapbox(df_2014, lat='Y', lon='X',
                           center = dict(lat = 37.8, lon = -122.4), #Sanfrancisco koordinat
                           zoom = 9,
                           opacity= .9,
                           mapbox_style= 'open-street-map',
                           color = 'cluster',
                           title = 'San Francisco Crime',
                           width = 1100,
                           height = 700,
                           hover_data = ['cluster', 'Category', 'Y', 'X']
                           )
figure.show()


plotly.offline.plot(figure, filename='maptest.html', auto_open = True)
















































