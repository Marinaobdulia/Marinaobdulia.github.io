[← Back](/index.md)

## NYC Taxi
---

The objective of this project was performing three studies over the NYC taxi 2017 dataset using PySpark and SQL syntax.

### Data Preparation

Firstly the dataset was imported in the form of a dataframe using the “read.csv” function. Since some entries were not containing valid logs, the whole data set was cleared up. The criteria used to do so was that the trips should have finished after the pickup time, their distance should be higher than 0 and the PULocationID and DOlocationID should be different.


```python
import pyspark
import pandas
import csv
import matplotlib as plt
from pyspark.sql import SparkSession

spark = SparkSession \
    .builder \
    .appName("Python Spark SQL basic example") \
    .config("spark.some.config.option", "some-value") \
    .getOrCreate()

# Import the data
df = spark.read.csv("tripdata_2017-01.csv",header=True)

#Clean up the data
df.registerTempTable("df")
clean = spark.sql("select * from df where tpep_dropoff_datetime>tpep_pickup_datetime and trip_distance>0 and PULocationID != DOLocationID")
clean.registerTempTable("clean")

# Locations
with open('taxi+_zone_lookup.csv', 'r') as f:
    locations_pre = list(csv.reader(f, delimiter=';'))
    locations = [locations_pre[n][0].replace('"', '').split(',') for n in range(len(locations_pre))]

# Rate code ID
ratecodes = ["Standard rate","JFK", "Newark", "Nassau or Westchester","Negotiated fare","Group ride"]
```


```python
# Preview of the results
df.limit(10).toPandas()
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>VendorID</th>
      <th>tpep_pickup_datetime</th>
      <th>tpep_dropoff_datetime</th>
      <th>passenger_count</th>
      <th>trip_distance</th>
      <th>RatecodeID</th>
      <th>store_and_fwd_flag</th>
      <th>PULocationID</th>
      <th>DOLocationID</th>
      <th>payment_type</th>
      <th>fare_amount</th>
      <th>extra</th>
      <th>mta_tax</th>
      <th>tip_amount</th>
      <th>tolls_amount</th>
      <th>improvement_surcharge</th>
      <th>total_amount</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <td>0</td>
      <td>1</td>
      <td>2017-01-09 11:13:28</td>
      <td>2017-01-09 11:25:45</td>
      <td>1</td>
      <td>3.30</td>
      <td>1</td>
      <td>N</td>
      <td>263</td>
      <td>161</td>
      <td>1</td>
      <td>12.5</td>
      <td>0</td>
      <td>0.5</td>
      <td>2</td>
      <td>0</td>
      <td>0.3</td>
      <td>15.3</td>
    </tr>
    <tr>
      <td>1</td>
      <td>1</td>
      <td>2017-01-09 11:32:27</td>
      <td>2017-01-09 11:36:01</td>
      <td>1</td>
      <td>.90</td>
      <td>1</td>
      <td>N</td>
      <td>186</td>
      <td>234</td>
      <td>1</td>
      <td>5</td>
      <td>0</td>
      <td>0.5</td>
      <td>1.45</td>
      <td>0</td>
      <td>0.3</td>
      <td>7.25</td>
    </tr>
    <tr>
      <td>2</td>
      <td>1</td>
      <td>2017-01-09 11:38:20</td>
      <td>2017-01-09 11:42:05</td>
      <td>1</td>
      <td>1.10</td>
      <td>1</td>
      <td>N</td>
      <td>164</td>
      <td>161</td>
      <td>1</td>
      <td>5.5</td>
      <td>0</td>
      <td>0.5</td>
      <td>1</td>
      <td>0</td>
      <td>0.3</td>
      <td>7.3</td>
    </tr>
    <tr>
      <td>3</td>
      <td>1</td>
      <td>2017-01-09 11:52:13</td>
      <td>2017-01-09 11:57:36</td>
      <td>1</td>
      <td>1.10</td>
      <td>1</td>
      <td>N</td>
      <td>236</td>
      <td>75</td>
      <td>1</td>
      <td>6</td>
      <td>0</td>
      <td>0.5</td>
      <td>1.7</td>
      <td>0</td>
      <td>0.3</td>
      <td>8.5</td>
    </tr>
    <tr>
      <td>4</td>
      <td>2</td>
      <td>2017-01-01 00:00:00</td>
      <td>2017-01-01 00:00:00</td>
      <td>1</td>
      <td>.02</td>
      <td>2</td>
      <td>N</td>
      <td>249</td>
      <td>234</td>
      <td>2</td>
      <td>52</td>
      <td>0</td>
      <td>0.5</td>
      <td>0</td>
      <td>0</td>
      <td>0.3</td>
      <td>52.8</td>
    </tr>
    <tr>
      <td>5</td>
      <td>1</td>
      <td>2017-01-01 00:00:02</td>
      <td>2017-01-01 00:03:50</td>
      <td>1</td>
      <td>.50</td>
      <td>1</td>
      <td>N</td>
      <td>48</td>
      <td>48</td>
      <td>2</td>
      <td>4</td>
      <td>0.5</td>
      <td>0.5</td>
      <td>0</td>
      <td>0</td>
      <td>0.3</td>
      <td>5.3</td>
    </tr>
    <tr>
      <td>6</td>
      <td>2</td>
      <td>2017-01-01 00:00:02</td>
      <td>2017-01-01 00:39:22</td>
      <td>4</td>
      <td>7.75</td>
      <td>1</td>
      <td>N</td>
      <td>186</td>
      <td>36</td>
      <td>1</td>
      <td>22</td>
      <td>0.5</td>
      <td>0.5</td>
      <td>4.66</td>
      <td>0</td>
      <td>0.3</td>
      <td>27.96</td>
    </tr>
    <tr>
      <td>7</td>
      <td>1</td>
      <td>2017-01-01 00:00:03</td>
      <td>2017-01-01 00:06:58</td>
      <td>1</td>
      <td>.80</td>
      <td>1</td>
      <td>N</td>
      <td>162</td>
      <td>161</td>
      <td>1</td>
      <td>6</td>
      <td>0.5</td>
      <td>0.5</td>
      <td>1.45</td>
      <td>0</td>
      <td>0.3</td>
      <td>8.75</td>
    </tr>
    <tr>
      <td>8</td>
      <td>1</td>
      <td>2017-01-01 00:00:05</td>
      <td>2017-01-01 00:08:33</td>
      <td>2</td>
      <td>.90</td>
      <td>1</td>
      <td>N</td>
      <td>48</td>
      <td>50</td>
      <td>1</td>
      <td>7</td>
      <td>0.5</td>
      <td>0.5</td>
      <td>0</td>
      <td>0</td>
      <td>0.3</td>
      <td>8.3</td>
    </tr>
    <tr>
      <td>9</td>
      <td>2</td>
      <td>2017-01-01 00:00:05</td>
      <td>2017-01-01 00:05:04</td>
      <td>5</td>
      <td>1.76</td>
      <td>1</td>
      <td>N</td>
      <td>140</td>
      <td>74</td>
      <td>2</td>
      <td>7</td>
      <td>0.5</td>
      <td>0.5</td>
      <td>0</td>
      <td>0</td>
      <td>0.3</td>
      <td>8.3</td>
    </tr>
  </tbody>
</table>
</div>



### Zone with the most Pick-ups


```python
most_PUlocation=spark.sql("select PULocationID, count(*) as PUcount from clean group by PUlocationID order by PUcount desc")

# Plot with the top ten LocationIDs
most_PUlocation.limit(10).toPandas().plot(kind='bar',x='PULocationID',y='PUcount')

```


    
![png](images/taxi_output_5_0.png)
    


### Rate that reported the highest gains


```python
richer=spark.sql("select RatecodeID, SUM(total_amount) as Yearly_total from clean group by RatecodeID order by Yearly_total desc")

richer.toPandas().plot(kind='pie',x='RatecodeID',y='Yearly_total', labels=ratecodes)

```


    
![png](images/taxi_output_7_0.png)
    


### Average tip per passenger given the rate


```python
avg_tip = spark.sql("select RatecodeID, avg(tip_amount/passenger_count) as average from clean group by RatecodeID order by RatecodeID")
avg_tip.toPandas().plot(kind='bar',x='RatecodeID',y='average')
```


    
![png](images/taxi_output_9_0.png)
    


### Most common trips


```python
trip = spark.sql("select PULocationID, DOLocationID, count(*) as numTrips from clean group by PULocationID,DOLocationID order by numTrips desc")
trip.limit(10).toPandas()

```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>PULocationID</th>
      <th>DOLocationID</th>
      <th>numTrips</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <td>0</td>
      <td>230</td>
      <td>246</td>
      <td>2515</td>
    </tr>
    <tr>
      <td>1</td>
      <td>237</td>
      <td>236</td>
      <td>1841</td>
    </tr>
    <tr>
      <td>2</td>
      <td>142</td>
      <td>238</td>
      <td>1774</td>
    </tr>
    <tr>
      <td>3</td>
      <td>79</td>
      <td>170</td>
      <td>1772</td>
    </tr>
    <tr>
      <td>4</td>
      <td>236</td>
      <td>237</td>
      <td>1592</td>
    </tr>
    <tr>
      <td>5</td>
      <td>238</td>
      <td>142</td>
      <td>1471</td>
    </tr>
    <tr>
      <td>6</td>
      <td>249</td>
      <td>79</td>
      <td>1439</td>
    </tr>
    <tr>
      <td>7</td>
      <td>186</td>
      <td>230</td>
      <td>1429</td>
    </tr>
    <tr>
      <td>8</td>
      <td>79</td>
      <td>186</td>
      <td>1409</td>
    </tr>
    <tr>
      <td>9</td>
      <td>239</td>
      <td>236</td>
      <td>1366</td>
    </tr>
  </tbody>
</table>
</div>


