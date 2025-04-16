[← Back](/index.md)

## NYC Taxi (2017)
---
The objective of this project was performing several studies over the NYC taxi 2017 dataset using PySpark and SQL syntax.

__*Importing libraries*__


```python
import pandas as pd
import csv
import matplotlib as plt
from pyspark.sql import SparkSession
import networkx as nx
```

# 1. Data Preparation

## 1.1. Create Spark session


```python
# Create a Spark session
spark = SparkSession \
    .builder \
    .appName("Python Spark SQL basic example") \
    .config("spark.some.config.option", "some-value") \
    .getOrCreate()
```

    Setting default log level to "WARN".
    To adjust logging level use sc.setLogLevel(newLevel). For SparkR, use setLogLevel(newLevel).
    25/04/16 19:09:15 WARN NativeCodeLoader: Unable to load native-hadoop library for your platform... using builtin-java classes where applicable


## 1.2. Data Cleaning
On the first place, the dataset was imported in the form of a dataframe using the *read.csv* function from PySpark library. 

Since some entries were not containing valid logs, the whole data set was cleared up. The criteria followed was:
- The drop-off time must be after the pick-up time.
- The trip distance must be greater than 0.
- The passenger count must be greater than 0.
- The trip time must be less than 22 hours and greater than 0.
- The calculated speed (trip distance divided by trip time) must be greater than 0 and less than 100.
- The pick-up location ID must not be the same as the drop-off location ID.


```python
# Import the data
df = spark.read.csv("tripdata_2017-01.csv",header=True)
df = df.withColumn("tpep_pickup_datetime", df["tpep_pickup_datetime"].cast("timestamp"))
df = df.withColumn("tpep_dropoff_datetime", df["tpep_dropoff_datetime"].cast("timestamp"))
df = df.withColumn("trip_time", (df["tpep_dropoff_datetime"].cast("long") - df["tpep_pickup_datetime"].cast("long"))/3600)

# Clean up the data
df.registerTempTable("df")
clean = spark.sql("""select *,
                  round(trip_distance/trip_time) as speed from df 
                  where tpep_dropoff_datetime>tpep_pickup_datetime 
                  and trip_distance>0
                  and passenger_count>0
                  and trip_time<22
                  and trip_time>0
                  and round(trip_distance/trip_time)>0
                  and round(trip_distance/trip_time)<100
                  and PULocationID != DOLocationID""")
clean.registerTempTable("clean")

```

    /opt/anaconda3/envs/address/lib/python3.12/site-packages/pyspark/sql/dataframe.py:329: FutureWarning: Deprecated in 2.0, use createOrReplaceTempView instead.
      warnings.warn("Deprecated in 2.0, use createOrReplaceTempView instead.", FutureWarning)
    /opt/anaconda3/envs/address/lib/python3.12/site-packages/pyspark/sql/dataframe.py:329: FutureWarning: Deprecated in 2.0, use createOrReplaceTempView instead.
      warnings.warn("Deprecated in 2.0, use createOrReplaceTempView instead.", FutureWarning)
    /opt/anaconda3/envs/address/lib/python3.12/site-packages/pyspark/sql/dataframe.py:329: FutureWarning: Deprecated in 2.0, use createOrReplaceTempView instead.
      warnings.warn("Deprecated in 2.0, use createOrReplaceTempView instead.", FutureWarning)


## 1.3. Merge Data
The cleaned up table was joined with both the location and Rate Code ID lookup tables in order to make the data easier to understand in the analysis.


```python
# Locations
with open('taxi+_zone_lookup.csv', 'r') as f:
    locations_pre = list(csv.reader(f, delimiter=';'))
    locations = [locations_pre[n][0].replace('"', '').split(',') for n in range(len(locations_pre))]

locations = pd.DataFrame(locations)
locations.columns = ['LocationID', 'Borough', 'Zone', 'service_zone']
locations = locations.loc[1:]
locations['LocationID'] = locations['LocationID'].astype(int)
locations = spark.createDataFrame(locations)
locations.registerTempTable("locations")

# Rate code ID
ratecodes_dict = {1: "Standard rate",
                    2: "JFK",
                    3: "Newark",
                    4: "Nassau or Westchester",
                    5: "Negotiated fare",
                    6: "Group ride"}

ratecodes = pd.DataFrame(ratecodes_dict.items(), columns=['RateCodeID', 'RateCode'])
ratecodes['RateCodeID'] = ratecodes['RateCodeID'].astype(int)
ratecodes = spark.createDataFrame(ratecodes)
ratecodes.registerTempTable("ratecodes")

df_with_locations = spark.sql("""select C.*,
                              D.Borough as BoroughDO,
                              D.Zone as ZoneDO,
                              D.Service_zone as Service_zoneDO
                            from (select A.*, 
                                    B.Borough as BoroughPU,
                                    B.Zone as ZonePU,
                                    B.Service_zone as Service_zonePU
                                    from clean A left join locations B
                                    on A.PULocationID = B.LocationID) C
                            left join locations D
                            on C.DOLocationID = D.LocationID""")
df_with_locations.registerTempTable("df_with_locations")

df_with_ratecodes = spark.sql("""select A.*,
                                B.RateCode
                                 from df_with_locations A
                                left join ratecodes B
                                on A.RatecodeID = B.RateCodeID""")
df_with_ratecodes.registerTempTable("df_with_ratecodes")
```

    25/04/16 19:47:20 WARN SparkStringUtils: Truncated the string representation of a plan since it was too large. This behavior can be adjusted by setting 'spark.sql.debug.maxToStringFields'.


# 2. EDA
In this section you may find the most interesting data exploratory analysis found during the development of the project. Other more simple analysis such as bare counts and means were discarded as they might not be as entertaining for the readers.

## 2.1. Average tip per passenger by rate


```python
avg_tip = spark.sql("""select RateCode, 
                    avg(tip_amount/passenger_count) as average 
                    from df_with_ratecodes 
                    group by RateCode
                    order by average desc""").toPandas()

avg_tip.plot.barh(x='RateCode', y='average', 
                  color='#A9CDD7', 
                  xlabel='Average tip amount per passenger', 
                  ylabel = 'Rate', 
                  title='Average tip amount per passenger by rate code', 
                  legend = False)
plt.show()
```

                                                                                    


    
![png](images/NYC%20Taxi_10_1.png)
    


## 2.2. Taxi Trip Flow


```python
trip = spark.sql("""select ZonePU, ZoneDO, 
                 count(*) as numTrips 
                 from df_with_ratecodes 
                 group by ZonePU,ZoneDO 
                 order by numTrips desc
                 limit 200""").toPandas()

def create_graph(df):
    G = nx.DiGraph()
    for _, row in df.iterrows():
        G.add_edge(row.ZonePU, row.ZoneDO, weight=row.numTrips_shortened)
    return G


def plot_graph_with_weights(G):
    plt.figure(figsize=(10, 10))
    pos = nx.kamada_kawai_layout(G)
    weights = [G[u][v]['weight'] for u, v in G.edges()]
    nx.draw(G, pos, node_size=2000, font_size=11, node_color='#F3D17C', with_labels=True, width=weights, edge_color='gray')
    plt.title('Taxi Trip Flow in NYC', fontsize=15)
    plt.axis('off')
    plt.show()

trip['numTrips_shortened'] = trip['numTrips'].apply(lambda x: round(x/2000, 3))
graph = create_graph(trip)
plot_graph_with_weights(graph)


```

                                                                                    


    
![png](images/NYC%20Taxi_12_1.png)
    


For a better understanding of the plot above, please take into consideration the width of the edges (arrows). They indicate the number of trips performed between two areas. Additionally, edges can be either unidirectional or bidirectional, as this corresponds only to a sample of the top 200 most common trips between NY areas.

## 2.3. Trips per hour


```python
# nr of trips per hour
month = spark.sql("""select hour(tpep_pickup_datetime) as hour,
                     count(*) as numTrips 
                     from df_with_ratecodes 
                     group by hour 
                     order by hour asc""").toPandas()

month.plot(kind='bar',
           x='hour',y='numTrips', 
            xlabel='Hour of the day',
            ylabel='Number of trips',
            title='Number of trips per hour',
            color='#83B8C6',
           legend=False)
plt.show()
```

                                                                                    


    
![png](images/NYC%20Taxi_14_1.png)
    


## 2.4. Distribution of trip average speed


```python
avg_speed = spark.sql("""select round(speed/10)*10 as speed,
                        count(*) as numTrips
                        from df_with_ratecodes
                        group by round(speed/10)*10
                        order by round(speed/10)*10 asc""").toPandas()

avg_speed.plot(kind='bar', 
               x='speed',y='numTrips', 
               legend=False, 
               color="#A9E4DE",
               xlabel='Speed (mi/hour)', ylabel='Number of trips', 
               title='Number of trips per average speed')
plt.show()

```

                                                                                    


    
![png](images/NYC%20Taxi_16_1.png)
    

