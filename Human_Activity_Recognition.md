[← Back](/index.md)

### Human Activity Recognition
---
The aim of this project was labelling different human activities upon the data register by 5 dimensional signals a 2-axes accelerometer and a 3-axes gyroscope.

The set of activities recorded for labelling were:

- running
- walking
- standing
- sitting
- lying

The data provided for this assignment corresponded to 10 different people, 8 of them labelled and 2 of them unlabelled. Each of the recordings lasted less than 20 minutes.


```python
import numpy as np
from scipy.io import loadmat
import pandas as pd
from scipy import stats
from sklearn.preprocessing import OneHotEncoder
import keras
from sklearn.metrics import confusion_matrix
import altair as alt
import matplotlib.pyplot as plt
%matplotlib
```

# Load dataset

The labelled data was transferred into a pandas dataframe, which resulted into a matrix of 141,426 rows and 8 columns. Each row corresponded to a measurement performed every 0.0625s, as the recording frequency was 16 Hz. On the other hand, the first five columns corresponded to the sensors that recorded the movements and the remaining three accounted for the user number, the labelled activity and the timestamp.


```python
# Open the file and read in data
data = loadmat('HAR_database.mat')
```

# EDA
After a preliminar data formatting, a simple exploratory data analysis was perfomed to observe the data distribution accross the labels and the users wearing the sensors.


```python
# Transform the data into a nice formatted dataset
def format_data(data):
    n_users = data['database_training'].shape[0]

    df = pd.DataFrame()
    for i in range(n_users):
        df2 = pd.DataFrame(data['database_training'][i][0].transpose(), columns=['acc_z', 'acc_XY', 'gy_x','gy_y', 'gy_z'])
        df2['User']=len(data['database_training'][i][1][0])*[i]
        df2['Activity']=data['database_training'][i][1][0]
        df2['Timestamp']=range(len(data['database_training'][i][1][0]))
        df = pd.concat([df, df2], ignore_index=True)
    
    return df

dict_labels = {
    1: 'Running',
    2: 'Walking',
    3: 'Standing',
    4: 'Sitting',
    5: 'Lying'
}

def add_labels(df, dict_labels=dict_labels):
    df['Activity_label'] = df['Activity'].map(dict_labels)
    return df

df = format_data(data)
df = add_labels(df)
```


```python
df.iloc[0]
```
    acc_z             0.024173
    acc_XY            0.594417
    gy_x             -0.022736
    gy_y              0.111962
    gy_z              0.060499
    User                     0
    Activity                 3
    Timestamp                0
    Activity_label    Standing
    Name: 0, dtype: object




```python
# Plot the distribution of the labelled activities using altair
def plot_distribution(df, col, title):
    """
    Plot the distribution of a given column in the dataframe.
    """
    df_grouped = df.groupby(col).size().reset_index(name='Count')
    chart = alt.Chart(df_grouped).mark_bar().encode(
        alt.X(col+':O', title=col),
        y='Count',
    ).properties(
        title=title,
        width=400,
        height=300
    )
    return chart

# Plot the distribution of activities
chart = plot_distribution(df, 'Activity_label', 'Distribution of Activities')
chart.display()

# Plot the distribution of users
chart = plot_distribution(df, 'User', 'Distribution of Users')
chart.display()
```



<style>
  #altair-viz-9933cc2ae39d4677af6f89d9d0c5a8d7.vega-embed {
    width: 100%;
    display: flex;
  }

  #altair-viz-9933cc2ae39d4677af6f89d9d0c5a8d7.vega-embed details,
  #altair-viz-9933cc2ae39d4677af6f89d9d0c5a8d7.vega-embed details summary {
    position: relative;
  }
</style>
<div id="altair-viz-9933cc2ae39d4677af6f89d9d0c5a8d7"></div>
<script type="text/javascript">
  var VEGA_DEBUG = (typeof VEGA_DEBUG == "undefined") ? {} : VEGA_DEBUG;
  (function(spec, embedOpt){
    let outputDiv = document.currentScript.previousElementSibling;
    if (outputDiv.id !== "altair-viz-9933cc2ae39d4677af6f89d9d0c5a8d7") {
      outputDiv = document.getElementById("altair-viz-9933cc2ae39d4677af6f89d9d0c5a8d7");
    }

    const paths = {
      "vega": "https://cdn.jsdelivr.net/npm/vega@5?noext",
      "vega-lib": "https://cdn.jsdelivr.net/npm/vega-lib?noext",
      "vega-lite": "https://cdn.jsdelivr.net/npm/vega-lite@5.20.1?noext",
      "vega-embed": "https://cdn.jsdelivr.net/npm/vega-embed@6?noext",
    };

    function maybeLoadScript(lib, version) {
      var key = `${lib.replace("-", "")}_version`;
      return (VEGA_DEBUG[key] == version) ?
        Promise.resolve(paths[lib]) :
        new Promise(function(resolve, reject) {
          var s = document.createElement('script');
          document.getElementsByTagName("head")[0].appendChild(s);
          s.async = true;
          s.onload = () => {
            VEGA_DEBUG[key] = version;
            return resolve(paths[lib]);
          };
          s.onerror = () => reject(`Error loading script: ${paths[lib]}`);
          s.src = paths[lib];
        });
    }

    function showError(err) {
      outputDiv.innerHTML = `<div class="error" style="color:red;">${err}</div>`;
      throw err;
    }

    function displayChart(vegaEmbed) {
      vegaEmbed(outputDiv, spec, embedOpt)
        .catch(err => showError(`Javascript Error: ${err.message}<br>This usually means there's a typo in your chart specification. See the javascript console for the full traceback.`));
    }

    if(typeof define === "function" && define.amd) {
      requirejs.config({paths});
      let deps = ["vega-embed"];
      require(deps, displayChart, err => showError(`Error loading script: ${err.message}`));
    } else {
      maybeLoadScript("vega", "5")
        .then(() => maybeLoadScript("vega-lite", "5.20.1"))
        .then(() => maybeLoadScript("vega-embed", "6"))
        .catch(showError)
        .then(() => displayChart(vegaEmbed));
    }
  })({"config": {"view": {"continuousWidth": 300, "continuousHeight": 300}}, "data": {"name": "data-25a062eb66f2a17cf2342b29baeb5c46"}, "mark": {"type": "bar"}, "encoding": {"x": {"field": "Activity_label", "title": "Activity_label", "type": "ordinal"}, "y": {"field": "Count", "type": "quantitative"}}, "height": 300, "title": "Distribution of Activities", "width": 400, "$schema": "https://vega.github.io/schema/vega-lite/v5.20.1.json", "datasets": {"data-25a062eb66f2a17cf2342b29baeb5c46": [{"Activity_label": "Lying", "Count": 24831}, {"Activity_label": "Running", "Count": 4807}, {"Activity_label": "Sitting", "Count": 40570}, {"Activity_label": "Standing", "Count": 41846}, {"Activity_label": "Walking", "Count": 29372}]}}, {"mode": "vega-lite"});
</script>




<style>
  #altair-viz-e4584d25a0f64cc7bf608f3880f84d33.vega-embed {
    width: 100%;
    display: flex;
  }

  #altair-viz-e4584d25a0f64cc7bf608f3880f84d33.vega-embed details,
  #altair-viz-e4584d25a0f64cc7bf608f3880f84d33.vega-embed details summary {
    position: relative;
  }
</style>
<div id="altair-viz-e4584d25a0f64cc7bf608f3880f84d33"></div>
<script type="text/javascript">
  var VEGA_DEBUG = (typeof VEGA_DEBUG == "undefined") ? {} : VEGA_DEBUG;
  (function(spec, embedOpt){
    let outputDiv = document.currentScript.previousElementSibling;
    if (outputDiv.id !== "altair-viz-e4584d25a0f64cc7bf608f3880f84d33") {
      outputDiv = document.getElementById("altair-viz-e4584d25a0f64cc7bf608f3880f84d33");
    }

    const paths = {
      "vega": "https://cdn.jsdelivr.net/npm/vega@5?noext",
      "vega-lib": "https://cdn.jsdelivr.net/npm/vega-lib?noext",
      "vega-lite": "https://cdn.jsdelivr.net/npm/vega-lite@5.20.1?noext",
      "vega-embed": "https://cdn.jsdelivr.net/npm/vega-embed@6?noext",
    };

    function maybeLoadScript(lib, version) {
      var key = `${lib.replace("-", "")}_version`;
      return (VEGA_DEBUG[key] == version) ?
        Promise.resolve(paths[lib]) :
        new Promise(function(resolve, reject) {
          var s = document.createElement('script');
          document.getElementsByTagName("head")[0].appendChild(s);
          s.async = true;
          s.onload = () => {
            VEGA_DEBUG[key] = version;
            return resolve(paths[lib]);
          };
          s.onerror = () => reject(`Error loading script: ${paths[lib]}`);
          s.src = paths[lib];
        });
    }

    function showError(err) {
      outputDiv.innerHTML = `<div class="error" style="color:red;">${err}</div>`;
      throw err;
    }

    function displayChart(vegaEmbed) {
      vegaEmbed(outputDiv, spec, embedOpt)
        .catch(err => showError(`Javascript Error: ${err.message}<br>This usually means there's a typo in your chart specification. See the javascript console for the full traceback.`));
    }

    if(typeof define === "function" && define.amd) {
      requirejs.config({paths});
      let deps = ["vega-embed"];
      require(deps, displayChart, err => showError(`Error loading script: ${err.message}`));
    } else {
      maybeLoadScript("vega", "5")
        .then(() => maybeLoadScript("vega-lite", "5.20.1"))
        .then(() => maybeLoadScript("vega-embed", "6"))
        .catch(showError)
        .then(() => displayChart(vegaEmbed));
    }
  })({"config": {"view": {"continuousWidth": 300, "continuousHeight": 300}}, "data": {"name": "data-9bb7d8afd9e6409cb254f08bbecc0142"}, "mark": {"type": "bar"}, "encoding": {"x": {"field": "User", "title": "User", "type": "ordinal"}, "y": {"field": "Count", "type": "quantitative"}}, "height": 300, "title": "Distribution of Users", "width": 400, "$schema": "https://vega.github.io/schema/vega-lite/v5.20.1.json", "datasets": {"data-9bb7d8afd9e6409cb254f08bbecc0142": [{"User": 0, "Count": 17736}, {"User": 1, "Count": 18411}, {"User": 2, "Count": 17440}, {"User": 3, "Count": 17826}, {"User": 4, "Count": 16920}, {"User": 5, "Count": 17427}, {"User": 6, "Count": 18056}, {"User": 7, "Count": 17610}]}}, {"mode": "vega-lite"});
</script>



```python
# Function to plot the 5 sensor recordings given certain activity
def plot_activity(activity, df):
    data = df[df['Activity'] == activity][['Timestamp', 'acc_z', 'acc_XY', 'gy_x', 'gy_y', 'gy_z']][:50]
    data = data.melt('Timestamp', var_name='Sensor', value_name='Value')
    
    chart = alt.Chart(data).mark_line().encode(
        x='Timestamp',
        y='Value',
        color='Sensor:N'
    ).properties(
        title=f"Sensor recordings for activity: {dict_labels[activity]}"
    )
    
    return chart

# Plot the sensor recordings for activity 1 (Running)
chart = plot_activity(1, df)
chart.display()

# Plot the sensor recordings for activity 2 (Walking)
chart = plot_activity(2, df)
chart.display()

# Plot the sensor recordings for activity 5 (Lying)
chart = plot_activity(5, df)
chart.display()
```



<style>
  #altair-viz-ae5f93a43ee143c5821c2df77fe24b40.vega-embed {
    width: 100%;
    display: flex;
  }

  #altair-viz-ae5f93a43ee143c5821c2df77fe24b40.vega-embed details,
  #altair-viz-ae5f93a43ee143c5821c2df77fe24b40.vega-embed details summary {
    position: relative;
  }
</style>
<div id="altair-viz-ae5f93a43ee143c5821c2df77fe24b40"></div>
<script type="text/javascript">
  var VEGA_DEBUG = (typeof VEGA_DEBUG == "undefined") ? {} : VEGA_DEBUG;
  (function(spec, embedOpt){
    let outputDiv = document.currentScript.previousElementSibling;
    if (outputDiv.id !== "altair-viz-ae5f93a43ee143c5821c2df77fe24b40") {
      outputDiv = document.getElementById("altair-viz-ae5f93a43ee143c5821c2df77fe24b40");
    }

    const paths = {
      "vega": "https://cdn.jsdelivr.net/npm/vega@5?noext",
      "vega-lib": "https://cdn.jsdelivr.net/npm/vega-lib?noext",
      "vega-lite": "https://cdn.jsdelivr.net/npm/vega-lite@5.20.1?noext",
      "vega-embed": "https://cdn.jsdelivr.net/npm/vega-embed@6?noext",
    };

    function maybeLoadScript(lib, version) {
      var key = `${lib.replace("-", "")}_version`;
      return (VEGA_DEBUG[key] == version) ?
        Promise.resolve(paths[lib]) :
        new Promise(function(resolve, reject) {
          var s = document.createElement('script');
          document.getElementsByTagName("head")[0].appendChild(s);
          s.async = true;
          s.onload = () => {
            VEGA_DEBUG[key] = version;
            return resolve(paths[lib]);
          };
          s.onerror = () => reject(`Error loading script: ${paths[lib]}`);
          s.src = paths[lib];
        });
    }

    function showError(err) {
      outputDiv.innerHTML = `<div class="error" style="color:red;">${err}</div>`;
      throw err;
    }

    function displayChart(vegaEmbed) {
      vegaEmbed(outputDiv, spec, embedOpt)
        .catch(err => showError(`Javascript Error: ${err.message}<br>This usually means there's a typo in your chart specification. See the javascript console for the full traceback.`));
    }

    if(typeof define === "function" && define.amd) {
      requirejs.config({paths});
      let deps = ["vega-embed"];
      require(deps, displayChart, err => showError(`Error loading script: ${err.message}`));
    } else {
      maybeLoadScript("vega", "5")
        .then(() => maybeLoadScript("vega-lite", "5.20.1"))
        .then(() => maybeLoadScript("vega-embed", "6"))
        .catch(showError)
        .then(() => displayChart(vegaEmbed));
    }
  })({"config": {"view": {"continuousWidth": 300, "continuousHeight": 300}}, "data": {"name": "data-154c9621850b732c80d91d8473b85783"}, "mark": {"type": "line"}, "encoding": {"color": {"field": "Sensor", "type": "nominal"}, "x": {"field": "Timestamp", "type": "quantitative"}, "y": {"field": "Value", "type": "quantitative"}}, "title": "Sensor recordings for activity: Running", "$schema": "https://vega.github.io/schema/vega-lite/v5.20.1.json", "datasets": {"data-154c9621850b732c80d91d8473b85783": [{"Timestamp": 7687, "Sensor": "acc_z", "Value": -8.271773924152772}, {"Timestamp": 7688, "Sensor": "acc_z", "Value": -10.476722267796585}, {"Timestamp": 7689, "Sensor": "acc_z", "Value": 5.125201983933618}, {"Timestamp": 7690, "Sensor": "acc_z", "Value": 10.067431482871338}, {"Timestamp": 7691, "Sensor": "acc_z", "Value": 18.221547967744392}, {"Timestamp": 7692, "Sensor": "acc_z", "Value": -2.2603128900414764}, {"Timestamp": 7693, "Sensor": "acc_z", "Value": -9.416962478838423}, {"Timestamp": 7694, "Sensor": "acc_z", "Value": -9.571800150828366}, {"Timestamp": 7695, "Sensor": "acc_z", "Value": 4.356760564453864}, {"Timestamp": 7696, "Sensor": "acc_z", "Value": 14.70960177938383}, {"Timestamp": 7697, "Sensor": "acc_z", "Value": -0.07550803278987551}, {"Timestamp": 7698, "Sensor": "acc_z", "Value": -8.999835179560222}, {"Timestamp": 7699, "Sensor": "acc_z", "Value": -14.002630108997883}, {"Timestamp": 7700, "Sensor": "acc_z", "Value": -7.155048115255584}, {"Timestamp": 7701, "Sensor": "acc_z", "Value": 23.74262857235614}, {"Timestamp": 7702, "Sensor": "acc_z", "Value": 0.7591353189179367}, {"Timestamp": 7703, "Sensor": "acc_z", "Value": -9.289126729171429}, {"Timestamp": 7704, "Sensor": "acc_z", "Value": -10.187598400232252}, {"Timestamp": 7705, "Sensor": "acc_z", "Value": -4.610369964365248}, {"Timestamp": 7706, "Sensor": "acc_z", "Value": 37.26108648591844}, {"Timestamp": 7707, "Sensor": "acc_z", "Value": -12.26438028852333}, {"Timestamp": 7708, "Sensor": "acc_z", "Value": -6.593703279718183}, {"Timestamp": 7709, "Sensor": "acc_z", "Value": -10.299655269395451}, {"Timestamp": 7710, "Sensor": "acc_z", "Value": -1.7048704122605862}, {"Timestamp": 7711, "Sensor": "acc_z", "Value": -0.2998208347052634}, {"Timestamp": 7712, "Sensor": "acc_z", "Value": 20.02309215293528}, {"Timestamp": 7713, "Sensor": "acc_z", "Value": 0.4782580400231378}, {"Timestamp": 7714, "Sensor": "acc_z", "Value": -9.877538758303258}, {"Timestamp": 7715, "Sensor": "acc_z", "Value": -11.46268346003564}, {"Timestamp": 7716, "Sensor": "acc_z", "Value": 0.6663213458103394}, {"Timestamp": 7717, "Sensor": "acc_z", "Value": 12.303540404645533}, {"Timestamp": 7718, "Sensor": "acc_z", "Value": 5.079938825532428}, {"Timestamp": 7719, "Sensor": "acc_z", "Value": -16.68976694459121}, {"Timestamp": 7720, "Sensor": "acc_z", "Value": -8.140024294593477}, {"Timestamp": 7721, "Sensor": "acc_z", "Value": 6.222022619997364}, {"Timestamp": 7722, "Sensor": "acc_z", "Value": 5.536892857192192}, {"Timestamp": 7723, "Sensor": "acc_z", "Value": 23.384155845486777}, {"Timestamp": 7724, "Sensor": "acc_z", "Value": -4.922097604812824}, {"Timestamp": 7725, "Sensor": "acc_z", "Value": -10.127121206747237}, {"Timestamp": 7726, "Sensor": "acc_z", "Value": -9.914917652996534}, {"Timestamp": 7727, "Sensor": "acc_z", "Value": 2.2547096676455514}, {"Timestamp": 7728, "Sensor": "acc_z", "Value": 29.306469587893204}, {"Timestamp": 7729, "Sensor": "acc_z", "Value": 8.174178092682059}, {"Timestamp": 7730, "Sensor": "acc_z", "Value": -14.381300058454162}, {"Timestamp": 7731, "Sensor": "acc_z", "Value": -9.988446067246347}, {"Timestamp": 7732, "Sensor": "acc_z", "Value": 6.4435531882927695}, {"Timestamp": 7733, "Sensor": "acc_z", "Value": 17.35860758569978}, {"Timestamp": 7734, "Sensor": "acc_z", "Value": 11.739891183283461}, {"Timestamp": 7735, "Sensor": "acc_z", "Value": -3.697201503966874}, {"Timestamp": 7736, "Sensor": "acc_z", "Value": -9.45397926651848}, {"Timestamp": 7687, "Sensor": "acc_XY", "Value": 0.795743248582753}, {"Timestamp": 7688, "Sensor": "acc_XY", "Value": 6.87907097992685}, {"Timestamp": 7689, "Sensor": "acc_XY", "Value": 10.55621049215301}, {"Timestamp": 7690, "Sensor": "acc_XY", "Value": 5.033890521468188}, {"Timestamp": 7691, "Sensor": "acc_XY", "Value": 8.491439728475825}, {"Timestamp": 7692, "Sensor": "acc_XY", "Value": 1.6719602114693946}, {"Timestamp": 7693, "Sensor": "acc_XY", "Value": 3.70245825580178}, {"Timestamp": 7694, "Sensor": "acc_XY", "Value": 4.42758196438227}, {"Timestamp": 7695, "Sensor": "acc_XY", "Value": 11.06559728232789}, {"Timestamp": 7696, "Sensor": "acc_XY", "Value": 5.924194463069367}, {"Timestamp": 7697, "Sensor": "acc_XY", "Value": 0.5085753043763569}, {"Timestamp": 7698, "Sensor": "acc_XY", "Value": 3.117331185538671}, {"Timestamp": 7699, "Sensor": "acc_XY", "Value": 8.253330154438371}, {"Timestamp": 7700, "Sensor": "acc_XY", "Value": 10.445799819839065}, {"Timestamp": 7701, "Sensor": "acc_XY", "Value": 8.46199757504636}, {"Timestamp": 7702, "Sensor": "acc_XY", "Value": 5.240715130724379}, {"Timestamp": 7703, "Sensor": "acc_XY", "Value": 1.6908702442787427}, {"Timestamp": 7704, "Sensor": "acc_XY", "Value": 5.578650277435023}, {"Timestamp": 7705, "Sensor": "acc_XY", "Value": 2.051010568681312}, {"Timestamp": 7706, "Sensor": "acc_XY", "Value": 7.998471061417039}, {"Timestamp": 7707, "Sensor": "acc_XY", "Value": 4.500026484149584}, {"Timestamp": 7708, "Sensor": "acc_XY", "Value": 2.2643998444309243}, {"Timestamp": 7709, "Sensor": "acc_XY", "Value": 9.181882624627121}, {"Timestamp": 7710, "Sensor": "acc_XY", "Value": 8.350042387321977}, {"Timestamp": 7711, "Sensor": "acc_XY", "Value": 8.374198930271284}, {"Timestamp": 7712, "Sensor": "acc_XY", "Value": 3.3087931839905624}, {"Timestamp": 7713, "Sensor": "acc_XY", "Value": 4.497955803868779}, {"Timestamp": 7714, "Sensor": "acc_XY", "Value": 5.1170300866541885}, {"Timestamp": 7715, "Sensor": "acc_XY", "Value": 6.3638713256201}, {"Timestamp": 7716, "Sensor": "acc_XY", "Value": 25.637866569885148}, {"Timestamp": 7717, "Sensor": "acc_XY", "Value": 8.764944918989565}, {"Timestamp": 7718, "Sensor": "acc_XY", "Value": 5.320548465846909}, {"Timestamp": 7719, "Sensor": "acc_XY", "Value": 3.876412147449976}, {"Timestamp": 7720, "Sensor": "acc_XY", "Value": 8.214306902125445}, {"Timestamp": 7721, "Sensor": "acc_XY", "Value": 22.73030312329699}, {"Timestamp": 7722, "Sensor": "acc_XY", "Value": 23.16027353311338}, {"Timestamp": 7723, "Sensor": "acc_XY", "Value": 9.77955915198199}, {"Timestamp": 7724, "Sensor": "acc_XY", "Value": 2.5202768985879582}, {"Timestamp": 7725, "Sensor": "acc_XY", "Value": 2.799478790975845}, {"Timestamp": 7726, "Sensor": "acc_XY", "Value": 7.485469880327661}, {"Timestamp": 7727, "Sensor": "acc_XY", "Value": 25.348964511117433}, {"Timestamp": 7728, "Sensor": "acc_XY", "Value": 8.12337445242679}, {"Timestamp": 7729, "Sensor": "acc_XY", "Value": 4.660044743202661}, {"Timestamp": 7730, "Sensor": "acc_XY", "Value": 3.3817313198779053}, {"Timestamp": 7731, "Sensor": "acc_XY", "Value": 8.877666055553371}, {"Timestamp": 7732, "Sensor": "acc_XY", "Value": 11.028942837194542}, {"Timestamp": 7733, "Sensor": "acc_XY", "Value": 20.70467499157396}, {"Timestamp": 7734, "Sensor": "acc_XY", "Value": 11.015117141751801}, {"Timestamp": 7735, "Sensor": "acc_XY", "Value": 5.548487019766645}, {"Timestamp": 7736, "Sensor": "acc_XY", "Value": 4.459263348013633}, {"Timestamp": 7687, "Sensor": "gy_x", "Value": -1.2859339506649767}, {"Timestamp": 7688, "Sensor": "gy_x", "Value": 2.6357928663982855}, {"Timestamp": 7689, "Sensor": "gy_x", "Value": 5.15093025056362}, {"Timestamp": 7690, "Sensor": "gy_x", "Value": -1.4462500498414137}, {"Timestamp": 7691, "Sensor": "gy_x", "Value": -4.117073870953858}, {"Timestamp": 7692, "Sensor": "gy_x", "Value": -1.32673770136787}, {"Timestamp": 7693, "Sensor": "gy_x", "Value": 1.1802957229273823}, {"Timestamp": 7694, "Sensor": "gy_x", "Value": 2.839122745259511}, {"Timestamp": 7695, "Sensor": "gy_x", "Value": 2.650381726476655}, {"Timestamp": 7696, "Sensor": "gy_x", "Value": 2.2252483800119607}, {"Timestamp": 7697, "Sensor": "gy_x", "Value": -2.237356940736877}, {"Timestamp": 7698, "Sensor": "gy_x", "Value": -0.3124792017545788}, {"Timestamp": 7699, "Sensor": "gy_x", "Value": 1.2973668944411616}, {"Timestamp": 7700, "Sensor": "gy_x", "Value": 1.0915442190886195}, {"Timestamp": 7701, "Sensor": "gy_x", "Value": -2.9084394758781222}, {"Timestamp": 7702, "Sensor": "gy_x", "Value": -2.055578696857849}, {"Timestamp": 7703, "Sensor": "gy_x", "Value": 0.17754731573558283}, {"Timestamp": 7704, "Sensor": "gy_x", "Value": 2.1043336995992594}, {"Timestamp": 7705, "Sensor": "gy_x", "Value": 1.717976756578765}, {"Timestamp": 7706, "Sensor": "gy_x", "Value": -2.637368979194626}, {"Timestamp": 7707, "Sensor": "gy_x", "Value": 0.04844487510146949}, {"Timestamp": 7708, "Sensor": "gy_x", "Value": -0.4351728036717091}, {"Timestamp": 7709, "Sensor": "gy_x", "Value": -0.13733956360379918}, {"Timestamp": 7710, "Sensor": "gy_x", "Value": 1.8430999749810877}, {"Timestamp": 7711, "Sensor": "gy_x", "Value": -1.6932511186245784}, {"Timestamp": 7712, "Sensor": "gy_x", "Value": -4.386374339419582}, {"Timestamp": 7713, "Sensor": "gy_x", "Value": -1.4152287192547501}, {"Timestamp": 7714, "Sensor": "gy_x", "Value": 0.31243038549387103}, {"Timestamp": 7715, "Sensor": "gy_x", "Value": 3.2734413870946986}, {"Timestamp": 7716, "Sensor": "gy_x", "Value": 2.393906749144503}, {"Timestamp": 7717, "Sensor": "gy_x", "Value": 4.731300619845542}, {"Timestamp": 7718, "Sensor": "gy_x", "Value": -2.081965107910485}, {"Timestamp": 7719, "Sensor": "gy_x", "Value": -1.1330177405892625}, {"Timestamp": 7720, "Sensor": "gy_x", "Value": 1.7028028843266991}, {"Timestamp": 7721, "Sensor": "gy_x", "Value": 9.96023927259631}, {"Timestamp": 7722, "Sensor": "gy_x", "Value": -2.24869924549832}, {"Timestamp": 7723, "Sensor": "gy_x", "Value": -4.398613198461243}, {"Timestamp": 7724, "Sensor": "gy_x", "Value": -1.7844224340860335}, {"Timestamp": 7725, "Sensor": "gy_x", "Value": 0.6482283212638371}, {"Timestamp": 7726, "Sensor": "gy_x", "Value": 2.369606906910932}, {"Timestamp": 7727, "Sensor": "gy_x", "Value": 4.988026508898128}, {"Timestamp": 7728, "Sensor": "gy_x", "Value": 4.973040431571568}, {"Timestamp": 7729, "Sensor": "gy_x", "Value": -2.498394589003502}, {"Timestamp": 7730, "Sensor": "gy_x", "Value": -0.6780036804974944}, {"Timestamp": 7731, "Sensor": "gy_x", "Value": 1.1996991279173097}, {"Timestamp": 7732, "Sensor": "gy_x", "Value": 4.0604327869485}, {"Timestamp": 7733, "Sensor": "gy_x", "Value": -0.2824254235646248}, {"Timestamp": 7734, "Sensor": "gy_x", "Value": -7.0276443666297475}, {"Timestamp": 7735, "Sensor": "gy_x", "Value": -1.706016515747959}, {"Timestamp": 7736, "Sensor": "gy_x", "Value": 0.38679879847451926}, {"Timestamp": 7687, "Sensor": "gy_y", "Value": 0.5326590948557546}, {"Timestamp": 7688, "Sensor": "gy_y", "Value": 2.2814008491091844}, {"Timestamp": 7689, "Sensor": "gy_y", "Value": -0.5700311934010632}, {"Timestamp": 7690, "Sensor": "gy_y", "Value": -0.4201899430370948}, {"Timestamp": 7691, "Sensor": "gy_y", "Value": 0.7807366826009305}, {"Timestamp": 7692, "Sensor": "gy_y", "Value": 0.1687151809158033}, {"Timestamp": 7693, "Sensor": "gy_y", "Value": 0.17488273317762126}, {"Timestamp": 7694, "Sensor": "gy_y", "Value": 0.6138169782886285}, {"Timestamp": 7695, "Sensor": "gy_y", "Value": 0.7826099622309569}, {"Timestamp": 7696, "Sensor": "gy_y", "Value": -1.2701498948857313}, {"Timestamp": 7697, "Sensor": "gy_y", "Value": -0.4264549107578362}, {"Timestamp": 7698, "Sensor": "gy_y", "Value": 0.6756391772876523}, {"Timestamp": 7699, "Sensor": "gy_y", "Value": 1.8177947447689433}, {"Timestamp": 7700, "Sensor": "gy_y", "Value": 0.025893310928320562}, {"Timestamp": 7701, "Sensor": "gy_y", "Value": 1.0697243670735521}, {"Timestamp": 7702, "Sensor": "gy_y", "Value": 0.28152828463421564}, {"Timestamp": 7703, "Sensor": "gy_y", "Value": 0.1813388028157992}, {"Timestamp": 7704, "Sensor": "gy_y", "Value": -0.07263411611436812}, {"Timestamp": 7705, "Sensor": "gy_y", "Value": 0.4211196718055788}, {"Timestamp": 7706, "Sensor": "gy_y", "Value": -0.355101030322445}, {"Timestamp": 7707, "Sensor": "gy_y", "Value": -0.8443144970737168}, {"Timestamp": 7708, "Sensor": "gy_y", "Value": -0.11243647657683142}, {"Timestamp": 7709, "Sensor": "gy_y", "Value": 0.8178326987886001}, {"Timestamp": 7710, "Sensor": "gy_y", "Value": 0.5367502212452282}, {"Timestamp": 7711, "Sensor": "gy_y", "Value": -0.8194605080103393}, {"Timestamp": 7712, "Sensor": "gy_y", "Value": 1.8814544901784394}, {"Timestamp": 7713, "Sensor": "gy_y", "Value": 0.09417257916860328}, {"Timestamp": 7714, "Sensor": "gy_y", "Value": 0.10696581701361936}, {"Timestamp": 7715, "Sensor": "gy_y", "Value": -0.2452110931797048}, {"Timestamp": 7716, "Sensor": "gy_y", "Value": 1.7249183247545652}, {"Timestamp": 7717, "Sensor": "gy_y", "Value": -0.45517648033495217}, {"Timestamp": 7718, "Sensor": "gy_y", "Value": -1.7378831535745185}, {"Timestamp": 7719, "Sensor": "gy_y", "Value": 0.34413629422743613}, {"Timestamp": 7720, "Sensor": "gy_y", "Value": 1.5428856277977059}, {"Timestamp": 7721, "Sensor": "gy_y", "Value": 1.596883858963666}, {"Timestamp": 7722, "Sensor": "gy_y", "Value": 0.16339876939557574}, {"Timestamp": 7723, "Sensor": "gy_y", "Value": -0.23972688584980845}, {"Timestamp": 7724, "Sensor": "gy_y", "Value": 0.9359435964141815}, {"Timestamp": 7725, "Sensor": "gy_y", "Value": -0.7425646375879172}, {"Timestamp": 7726, "Sensor": "gy_y", "Value": 0.1054343959750566}, {"Timestamp": 7727, "Sensor": "gy_y", "Value": 0.5459352334180347}, {"Timestamp": 7728, "Sensor": "gy_y", "Value": -1.336441543145995}, {"Timestamp": 7729, "Sensor": "gy_y", "Value": -1.4510999709729657}, {"Timestamp": 7730, "Sensor": "gy_y", "Value": 0.311851292758807}, {"Timestamp": 7731, "Sensor": "gy_y", "Value": 1.7526711846328271}, {"Timestamp": 7732, "Sensor": "gy_y", "Value": -0.4977486239472834}, {"Timestamp": 7733, "Sensor": "gy_y", "Value": -1.6086613841064183}, {"Timestamp": 7734, "Sensor": "gy_y", "Value": 1.4698184679917925}, {"Timestamp": 7735, "Sensor": "gy_y", "Value": 0.49363257254452286}, {"Timestamp": 7736, "Sensor": "gy_y", "Value": -0.3521762177816459}, {"Timestamp": 7687, "Sensor": "gy_z", "Value": -0.7386071615050951}, {"Timestamp": 7688, "Sensor": "gy_z", "Value": -0.059818485886908745}, {"Timestamp": 7689, "Sensor": "gy_z", "Value": 0.6258652528015617}, {"Timestamp": 7690, "Sensor": "gy_z", "Value": 0.6790169795949083}, {"Timestamp": 7691, "Sensor": "gy_z", "Value": -3.001028414960162}, {"Timestamp": 7692, "Sensor": "gy_z", "Value": -0.17061999610727963}, {"Timestamp": 7693, "Sensor": "gy_z", "Value": -0.4345759191863411}, {"Timestamp": 7694, "Sensor": "gy_z", "Value": 1.7985383265943249}, {"Timestamp": 7695, "Sensor": "gy_z", "Value": 8.026602913090972}, {"Timestamp": 7696, "Sensor": "gy_z", "Value": 0.7636094295161038}, {"Timestamp": 7697, "Sensor": "gy_z", "Value": -0.9326695360383804}, {"Timestamp": 7698, "Sensor": "gy_z", "Value": -1.0074493236589548}, {"Timestamp": 7699, "Sensor": "gy_z", "Value": -0.24769853418926008}, {"Timestamp": 7700, "Sensor": "gy_z", "Value": -0.20906387708397878}, {"Timestamp": 7701, "Sensor": "gy_z", "Value": -0.6155501900023109}, {"Timestamp": 7702, "Sensor": "gy_z", "Value": -0.8327400851406779}, {"Timestamp": 7703, "Sensor": "gy_z", "Value": -0.8754871524262489}, {"Timestamp": 7704, "Sensor": "gy_z", "Value": 0.6378660490966563}, {"Timestamp": 7705, "Sensor": "gy_z", "Value": 1.0328270616378468}, {"Timestamp": 7706, "Sensor": "gy_z", "Value": -0.41367338398165465}, {"Timestamp": 7707, "Sensor": "gy_z", "Value": 0.2820059711930934}, {"Timestamp": 7708, "Sensor": "gy_z", "Value": -0.238935780799971}, {"Timestamp": 7709, "Sensor": "gy_z", "Value": -0.6979586821791625}, {"Timestamp": 7710, "Sensor": "gy_z", "Value": 0.06060366859284849}, {"Timestamp": 7711, "Sensor": "gy_z", "Value": 3.962671072874211}, {"Timestamp": 7712, "Sensor": "gy_z", "Value": -2.769893718348199}, {"Timestamp": 7713, "Sensor": "gy_z", "Value": 0.08481831561220823}, {"Timestamp": 7714, "Sensor": "gy_z", "Value": -0.32549916755617053}, {"Timestamp": 7715, "Sensor": "gy_z", "Value": 1.100829957422194}, {"Timestamp": 7716, "Sensor": "gy_z", "Value": 4.235559476461084}, {"Timestamp": 7717, "Sensor": "gy_z", "Value": -0.20429253430334143}, {"Timestamp": 7718, "Sensor": "gy_z", "Value": -0.3758384869277872}, {"Timestamp": 7719, "Sensor": "gy_z", "Value": -0.6915774400798579}, {"Timestamp": 7720, "Sensor": "gy_z", "Value": -0.38531856758784977}, {"Timestamp": 7721, "Sensor": "gy_z", "Value": -1.8134676740602331}, {"Timestamp": 7722, "Sensor": "gy_z", "Value": 1.2577702213998774}, {"Timestamp": 7723, "Sensor": "gy_z", "Value": -1.6310582684089776}, {"Timestamp": 7724, "Sensor": "gy_z", "Value": -0.34808218803432467}, {"Timestamp": 7725, "Sensor": "gy_z", "Value": -1.0270445619121855}, {"Timestamp": 7726, "Sensor": "gy_z", "Value": 1.89678290156422}, {"Timestamp": 7727, "Sensor": "gy_z", "Value": -1.2118908546982878}, {"Timestamp": 7728, "Sensor": "gy_z", "Value": 1.3557230410821457}, {"Timestamp": 7729, "Sensor": "gy_z", "Value": -0.10970906246193733}, {"Timestamp": 7730, "Sensor": "gy_z", "Value": -0.17855904553539798}, {"Timestamp": 7731, "Sensor": "gy_z", "Value": -1.0359918985736372}, {"Timestamp": 7732, "Sensor": "gy_z", "Value": -1.6260098466313861}, {"Timestamp": 7733, "Sensor": "gy_z", "Value": 4.437200320501562}, {"Timestamp": 7734, "Sensor": "gy_z", "Value": -3.684105490108748}, {"Timestamp": 7735, "Sensor": "gy_z", "Value": -0.23172201483516872}, {"Timestamp": 7736, "Sensor": "gy_z", "Value": -1.0857422298071873}]}}, {"mode": "vega-lite"});
</script>




<style>
  #altair-viz-6cb4e5bbb7ae49b6b17567fd695898d6.vega-embed {
    width: 100%;
    display: flex;
  }

  #altair-viz-6cb4e5bbb7ae49b6b17567fd695898d6.vega-embed details,
  #altair-viz-6cb4e5bbb7ae49b6b17567fd695898d6.vega-embed details summary {
    position: relative;
  }
</style>
<div id="altair-viz-6cb4e5bbb7ae49b6b17567fd695898d6"></div>
<script type="text/javascript">
  var VEGA_DEBUG = (typeof VEGA_DEBUG == "undefined") ? {} : VEGA_DEBUG;
  (function(spec, embedOpt){
    let outputDiv = document.currentScript.previousElementSibling;
    if (outputDiv.id !== "altair-viz-6cb4e5bbb7ae49b6b17567fd695898d6") {
      outputDiv = document.getElementById("altair-viz-6cb4e5bbb7ae49b6b17567fd695898d6");
    }

    const paths = {
      "vega": "https://cdn.jsdelivr.net/npm/vega@5?noext",
      "vega-lib": "https://cdn.jsdelivr.net/npm/vega-lib?noext",
      "vega-lite": "https://cdn.jsdelivr.net/npm/vega-lite@5.20.1?noext",
      "vega-embed": "https://cdn.jsdelivr.net/npm/vega-embed@6?noext",
    };

    function maybeLoadScript(lib, version) {
      var key = `${lib.replace("-", "")}_version`;
      return (VEGA_DEBUG[key] == version) ?
        Promise.resolve(paths[lib]) :
        new Promise(function(resolve, reject) {
          var s = document.createElement('script');
          document.getElementsByTagName("head")[0].appendChild(s);
          s.async = true;
          s.onload = () => {
            VEGA_DEBUG[key] = version;
            return resolve(paths[lib]);
          };
          s.onerror = () => reject(`Error loading script: ${paths[lib]}`);
          s.src = paths[lib];
        });
    }

    function showError(err) {
      outputDiv.innerHTML = `<div class="error" style="color:red;">${err}</div>`;
      throw err;
    }

    function displayChart(vegaEmbed) {
      vegaEmbed(outputDiv, spec, embedOpt)
        .catch(err => showError(`Javascript Error: ${err.message}<br>This usually means there's a typo in your chart specification. See the javascript console for the full traceback.`));
    }

    if(typeof define === "function" && define.amd) {
      requirejs.config({paths});
      let deps = ["vega-embed"];
      require(deps, displayChart, err => showError(`Error loading script: ${err.message}`));
    } else {
      maybeLoadScript("vega", "5")
        .then(() => maybeLoadScript("vega-lite", "5.20.1"))
        .then(() => maybeLoadScript("vega-embed", "6"))
        .catch(showError)
        .then(() => displayChart(vegaEmbed));
    }
  })({"config": {"view": {"continuousWidth": 300, "continuousHeight": 300}}, "data": {"name": "data-e22990425a0d6512fde4ca94b062c824"}, "mark": {"type": "line"}, "encoding": {"color": {"field": "Sensor", "type": "nominal"}, "x": {"field": "Timestamp", "type": "quantitative"}, "y": {"field": "Value", "type": "quantitative"}}, "title": "Sensor recordings for activity: Walking", "$schema": "https://vega.github.io/schema/vega-lite/v5.20.1.json", "datasets": {"data-e22990425a0d6512fde4ca94b062c824": [{"Timestamp": 1226, "Sensor": "acc_z", "Value": 0.03577991833171609}, {"Timestamp": 1227, "Sensor": "acc_z", "Value": 0.819298551889627}, {"Timestamp": 1228, "Sensor": "acc_z", "Value": -0.13984111226476625}, {"Timestamp": 1229, "Sensor": "acc_z", "Value": 0.1912952248074763}, {"Timestamp": 1230, "Sensor": "acc_z", "Value": 0.04632501079637909}, {"Timestamp": 1231, "Sensor": "acc_z", "Value": -0.6005997898564384}, {"Timestamp": 1232, "Sensor": "acc_z", "Value": -0.1536913968872566}, {"Timestamp": 1233, "Sensor": "acc_z", "Value": -0.0038678612948524482}, {"Timestamp": 1234, "Sensor": "acc_z", "Value": -0.871597643529175}, {"Timestamp": 1235, "Sensor": "acc_z", "Value": 0.09172254666361818}, {"Timestamp": 1236, "Sensor": "acc_z", "Value": 1.4923669046794636}, {"Timestamp": 1237, "Sensor": "acc_z", "Value": -0.6331906016833562}, {"Timestamp": 1238, "Sensor": "acc_z", "Value": -0.17703498930682132}, {"Timestamp": 1239, "Sensor": "acc_z", "Value": 0.09747498082753905}, {"Timestamp": 1240, "Sensor": "acc_z", "Value": 0.31496287086771946}, {"Timestamp": 1241, "Sensor": "acc_z", "Value": -0.03670902388195982}, {"Timestamp": 1242, "Sensor": "acc_z", "Value": -0.07265968562865421}, {"Timestamp": 1243, "Sensor": "acc_z", "Value": -0.24793074135743778}, {"Timestamp": 1244, "Sensor": "acc_z", "Value": -0.2401612549263468}, {"Timestamp": 1245, "Sensor": "acc_z", "Value": 0.06638074279832595}, {"Timestamp": 1246, "Sensor": "acc_z", "Value": -0.31322584262103714}, {"Timestamp": 1247, "Sensor": "acc_z", "Value": -0.1357130989613775}, {"Timestamp": 1248, "Sensor": "acc_z", "Value": 0.31450618349450166}, {"Timestamp": 1249, "Sensor": "acc_z", "Value": 2.684149276855714}, {"Timestamp": 1250, "Sensor": "acc_z", "Value": 0.6156345495684576}, {"Timestamp": 1251, "Sensor": "acc_z", "Value": -1.9988115920711236}, {"Timestamp": 1252, "Sensor": "acc_z", "Value": 2.1382021383093317}, {"Timestamp": 1253, "Sensor": "acc_z", "Value": -1.3387723169819918}, {"Timestamp": 1254, "Sensor": "acc_z", "Value": -1.0780967077175072}, {"Timestamp": 1255, "Sensor": "acc_z", "Value": -0.1989493311066801}, {"Timestamp": 1256, "Sensor": "acc_z", "Value": -1.083633830586602}, {"Timestamp": 1257, "Sensor": "acc_z", "Value": -1.0550642231878724}, {"Timestamp": 1258, "Sensor": "acc_z", "Value": -0.5902902733018163}, {"Timestamp": 1259, "Sensor": "acc_z", "Value": 2.2766638725649284}, {"Timestamp": 1260, "Sensor": "acc_z", "Value": -0.16708433595501918}, {"Timestamp": 1261, "Sensor": "acc_z", "Value": 3.206140532804512}, {"Timestamp": 1262, "Sensor": "acc_z", "Value": -0.8939258594930082}, {"Timestamp": 1263, "Sensor": "acc_z", "Value": -0.7957864490460599}, {"Timestamp": 1264, "Sensor": "acc_z", "Value": -0.017911909693582584}, {"Timestamp": 1265, "Sensor": "acc_z", "Value": -1.1555601536371436}, {"Timestamp": 1266, "Sensor": "acc_z", "Value": -0.4982280987649309}, {"Timestamp": 1267, "Sensor": "acc_z", "Value": -0.4861445513086118}, {"Timestamp": 1268, "Sensor": "acc_z", "Value": -0.36684634816338657}, {"Timestamp": 1269, "Sensor": "acc_z", "Value": -0.41326149859537126}, {"Timestamp": 1270, "Sensor": "acc_z", "Value": 0.60917893144585}, {"Timestamp": 1271, "Sensor": "acc_z", "Value": 1.4385151670400447}, {"Timestamp": 1272, "Sensor": "acc_z", "Value": 1.9570887515322317}, {"Timestamp": 1273, "Sensor": "acc_z", "Value": -0.4603816107609776}, {"Timestamp": 1274, "Sensor": "acc_z", "Value": 1.065015307889452}, {"Timestamp": 1275, "Sensor": "acc_z", "Value": 0.347036894440054}, {"Timestamp": 1226, "Sensor": "acc_XY", "Value": 0.631558184192848}, {"Timestamp": 1227, "Sensor": "acc_XY", "Value": 0.5122062665629761}, {"Timestamp": 1228, "Sensor": "acc_XY", "Value": 0.8787617319720799}, {"Timestamp": 1229, "Sensor": "acc_XY", "Value": 1.2029981413296478}, {"Timestamp": 1230, "Sensor": "acc_XY", "Value": 1.362386756046621}, {"Timestamp": 1231, "Sensor": "acc_XY", "Value": 1.4890231552745499}, {"Timestamp": 1232, "Sensor": "acc_XY", "Value": 1.5802493339951438}, {"Timestamp": 1233, "Sensor": "acc_XY", "Value": 1.5951177280445732}, {"Timestamp": 1234, "Sensor": "acc_XY", "Value": 2.122564429288695}, {"Timestamp": 1235, "Sensor": "acc_XY", "Value": 1.3182015931817763}, {"Timestamp": 1236, "Sensor": "acc_XY", "Value": 1.1719432137874346}, {"Timestamp": 1237, "Sensor": "acc_XY", "Value": 1.0081397246040562}, {"Timestamp": 1238, "Sensor": "acc_XY", "Value": 0.7484993009181339}, {"Timestamp": 1239, "Sensor": "acc_XY", "Value": 0.8322024206787373}, {"Timestamp": 1240, "Sensor": "acc_XY", "Value": 0.779090535439991}, {"Timestamp": 1241, "Sensor": "acc_XY", "Value": 0.9255088540834814}, {"Timestamp": 1242, "Sensor": "acc_XY", "Value": 1.1367923908081106}, {"Timestamp": 1243, "Sensor": "acc_XY", "Value": 1.2923974657139503}, {"Timestamp": 1244, "Sensor": "acc_XY", "Value": 1.2538387114831266}, {"Timestamp": 1245, "Sensor": "acc_XY", "Value": 1.484005826788265}, {"Timestamp": 1246, "Sensor": "acc_XY", "Value": 1.1848556343972376}, {"Timestamp": 1247, "Sensor": "acc_XY", "Value": 2.198425457199322}, {"Timestamp": 1248, "Sensor": "acc_XY", "Value": 3.6644282957436167}, {"Timestamp": 1249, "Sensor": "acc_XY", "Value": 4.802373972975034}, {"Timestamp": 1250, "Sensor": "acc_XY", "Value": 2.539889358673128}, {"Timestamp": 1251, "Sensor": "acc_XY", "Value": 1.7622043068688498}, {"Timestamp": 1252, "Sensor": "acc_XY", "Value": 1.8687447921280518}, {"Timestamp": 1253, "Sensor": "acc_XY", "Value": 0.6935818920707052}, {"Timestamp": 1254, "Sensor": "acc_XY", "Value": 1.1830470793916164}, {"Timestamp": 1255, "Sensor": "acc_XY", "Value": 1.0817504772606645}, {"Timestamp": 1256, "Sensor": "acc_XY", "Value": 1.4625697066683476}, {"Timestamp": 1257, "Sensor": "acc_XY", "Value": 1.6343982349258426}, {"Timestamp": 1258, "Sensor": "acc_XY", "Value": 2.0295255087585784}, {"Timestamp": 1259, "Sensor": "acc_XY", "Value": 3.4024882315491807}, {"Timestamp": 1260, "Sensor": "acc_XY", "Value": 1.5413558041843447}, {"Timestamp": 1261, "Sensor": "acc_XY", "Value": 2.8967583232549483}, {"Timestamp": 1262, "Sensor": "acc_XY", "Value": 1.0416241891541251}, {"Timestamp": 1263, "Sensor": "acc_XY", "Value": 1.3662094673877723}, {"Timestamp": 1264, "Sensor": "acc_XY", "Value": 0.8499366796049955}, {"Timestamp": 1265, "Sensor": "acc_XY", "Value": 0.7434534601025726}, {"Timestamp": 1266, "Sensor": "acc_XY", "Value": 1.0481773890241517}, {"Timestamp": 1267, "Sensor": "acc_XY", "Value": 1.3464890724944218}, {"Timestamp": 1268, "Sensor": "acc_XY", "Value": 1.0985159256605084}, {"Timestamp": 1269, "Sensor": "acc_XY", "Value": 1.5986351290170533}, {"Timestamp": 1270, "Sensor": "acc_XY", "Value": 1.9745797425136298}, {"Timestamp": 1271, "Sensor": "acc_XY", "Value": 2.2436688077452582}, {"Timestamp": 1272, "Sensor": "acc_XY", "Value": 4.2800467866709155}, {"Timestamp": 1273, "Sensor": "acc_XY", "Value": 3.726959115255556}, {"Timestamp": 1274, "Sensor": "acc_XY", "Value": 2.0426752241307273}, {"Timestamp": 1275, "Sensor": "acc_XY", "Value": 0.5826069689366241}, {"Timestamp": 1226, "Sensor": "gy_x", "Value": 0.09482874726115896}, {"Timestamp": 1227, "Sensor": "gy_x", "Value": 0.22079189384941916}, {"Timestamp": 1228, "Sensor": "gy_x", "Value": 0.22034242385680022}, {"Timestamp": 1229, "Sensor": "gy_x", "Value": 0.15929290707451818}, {"Timestamp": 1230, "Sensor": "gy_x", "Value": 0.19379381520373962}, {"Timestamp": 1231, "Sensor": "gy_x", "Value": 0.25033968992019184}, {"Timestamp": 1232, "Sensor": "gy_x", "Value": 0.11012885323219684}, {"Timestamp": 1233, "Sensor": "gy_x", "Value": 0.0924652294838345}, {"Timestamp": 1234, "Sensor": "gy_x", "Value": 0.17494966327880548}, {"Timestamp": 1235, "Sensor": "gy_x", "Value": -0.4236233323949019}, {"Timestamp": 1236, "Sensor": "gy_x", "Value": -0.09202321514741424}, {"Timestamp": 1237, "Sensor": "gy_x", "Value": -0.3128644623261029}, {"Timestamp": 1238, "Sensor": "gy_x", "Value": -0.16173917683772374}, {"Timestamp": 1239, "Sensor": "gy_x", "Value": -0.3938804913594465}, {"Timestamp": 1240, "Sensor": "gy_x", "Value": -0.36838470140184987}, {"Timestamp": 1241, "Sensor": "gy_x", "Value": -0.23715781266721023}, {"Timestamp": 1242, "Sensor": "gy_x", "Value": -0.21599397150387498}, {"Timestamp": 1243, "Sensor": "gy_x", "Value": -0.20022168827423442}, {"Timestamp": 1244, "Sensor": "gy_x", "Value": -0.21558866180850683}, {"Timestamp": 1245, "Sensor": "gy_x", "Value": -0.19240694653389212}, {"Timestamp": 1246, "Sensor": "gy_x", "Value": -0.17234086405793925}, {"Timestamp": 1247, "Sensor": "gy_x", "Value": -0.3745882061530953}, {"Timestamp": 1248, "Sensor": "gy_x", "Value": -0.14380480914945143}, {"Timestamp": 1249, "Sensor": "gy_x", "Value": -0.27777690219819073}, {"Timestamp": 1250, "Sensor": "gy_x", "Value": 0.2069493389296636}, {"Timestamp": 1251, "Sensor": "gy_x", "Value": 0.7447936198509398}, {"Timestamp": 1252, "Sensor": "gy_x", "Value": 1.0388159949233602}, {"Timestamp": 1253, "Sensor": "gy_x", "Value": 0.8992494688530076}, {"Timestamp": 1254, "Sensor": "gy_x", "Value": 0.6747447899936972}, {"Timestamp": 1255, "Sensor": "gy_x", "Value": 0.23586698865432218}, {"Timestamp": 1256, "Sensor": "gy_x", "Value": -0.14527830585818297}, {"Timestamp": 1257, "Sensor": "gy_x", "Value": -0.22524782659658402}, {"Timestamp": 1258, "Sensor": "gy_x", "Value": -0.32369202154198085}, {"Timestamp": 1259, "Sensor": "gy_x", "Value": 0.3228999346854008}, {"Timestamp": 1260, "Sensor": "gy_x", "Value": 0.641858467906975}, {"Timestamp": 1261, "Sensor": "gy_x", "Value": -0.39286225105206873}, {"Timestamp": 1262, "Sensor": "gy_x", "Value": 0.31190235224440355}, {"Timestamp": 1263, "Sensor": "gy_x", "Value": -0.9744405727520066}, {"Timestamp": 1264, "Sensor": "gy_x", "Value": 0.19655946232174015}, {"Timestamp": 1265, "Sensor": "gy_x", "Value": -0.8906281756951794}, {"Timestamp": 1266, "Sensor": "gy_x", "Value": -0.47647486421839913}, {"Timestamp": 1267, "Sensor": "gy_x", "Value": -0.27053454515910186}, {"Timestamp": 1268, "Sensor": "gy_x", "Value": -0.41008502120333273}, {"Timestamp": 1269, "Sensor": "gy_x", "Value": -0.44028609142406555}, {"Timestamp": 1270, "Sensor": "gy_x", "Value": -0.04842003170289777}, {"Timestamp": 1271, "Sensor": "gy_x", "Value": -0.3388917536817609}, {"Timestamp": 1272, "Sensor": "gy_x", "Value": 0.41360745990834696}, {"Timestamp": 1273, "Sensor": "gy_x", "Value": 0.17756670331749821}, {"Timestamp": 1274, "Sensor": "gy_x", "Value": 1.1520939037051456}, {"Timestamp": 1275, "Sensor": "gy_x", "Value": 0.8520474528263747}, {"Timestamp": 1226, "Sensor": "gy_y", "Value": 0.05103009706828464}, {"Timestamp": 1227, "Sensor": "gy_y", "Value": -0.10908581871153013}, {"Timestamp": 1228, "Sensor": "gy_y", "Value": -0.1458523467458041}, {"Timestamp": 1229, "Sensor": "gy_y", "Value": -0.15582789630614574}, {"Timestamp": 1230, "Sensor": "gy_y", "Value": -0.10875988658299307}, {"Timestamp": 1231, "Sensor": "gy_y", "Value": -0.00434000066138604}, {"Timestamp": 1232, "Sensor": "gy_y", "Value": 0.08874064937211616}, {"Timestamp": 1233, "Sensor": "gy_y", "Value": 0.17483287858942098}, {"Timestamp": 1234, "Sensor": "gy_y", "Value": 0.29061485194721076}, {"Timestamp": 1235, "Sensor": "gy_y", "Value": 0.580150556349993}, {"Timestamp": 1236, "Sensor": "gy_y", "Value": 0.1063123059147232}, {"Timestamp": 1237, "Sensor": "gy_y", "Value": 0.15589038481326298}, {"Timestamp": 1238, "Sensor": "gy_y", "Value": 0.19834444323494885}, {"Timestamp": 1239, "Sensor": "gy_y", "Value": 0.2662460318932391}, {"Timestamp": 1240, "Sensor": "gy_y", "Value": 0.36886257624558383}, {"Timestamp": 1241, "Sensor": "gy_y", "Value": 0.30744138259723025}, {"Timestamp": 1242, "Sensor": "gy_y", "Value": 0.19904887118546585}, {"Timestamp": 1243, "Sensor": "gy_y", "Value": 0.11632087190175722}, {"Timestamp": 1244, "Sensor": "gy_y", "Value": 0.03295592080886246}, {"Timestamp": 1245, "Sensor": "gy_y", "Value": 0.06545178671075659}, {"Timestamp": 1246, "Sensor": "gy_y", "Value": 0.13341096708226052}, {"Timestamp": 1247, "Sensor": "gy_y", "Value": 0.26704443444896886}, {"Timestamp": 1248, "Sensor": "gy_y", "Value": 0.20477421746490063}, {"Timestamp": 1249, "Sensor": "gy_y", "Value": 0.40552698084738026}, {"Timestamp": 1250, "Sensor": "gy_y", "Value": -0.20564685708087477}, {"Timestamp": 1251, "Sensor": "gy_y", "Value": -0.3513504827876579}, {"Timestamp": 1252, "Sensor": "gy_y", "Value": -0.11577164498622161}, {"Timestamp": 1253, "Sensor": "gy_y", "Value": -0.34832835655184713}, {"Timestamp": 1254, "Sensor": "gy_y", "Value": -0.10627885918466261}, {"Timestamp": 1255, "Sensor": "gy_y", "Value": -0.08468635627166485}, {"Timestamp": 1256, "Sensor": "gy_y", "Value": 0.06219803091932674}, {"Timestamp": 1257, "Sensor": "gy_y", "Value": 0.11782774374297762}, {"Timestamp": 1258, "Sensor": "gy_y", "Value": 0.2855231647373701}, {"Timestamp": 1259, "Sensor": "gy_y", "Value": -0.22211727763742256}, {"Timestamp": 1260, "Sensor": "gy_y", "Value": -0.3298836571272431}, {"Timestamp": 1261, "Sensor": "gy_y", "Value": -0.46767614443282834}, {"Timestamp": 1262, "Sensor": "gy_y", "Value": 0.009053076628361578}, {"Timestamp": 1263, "Sensor": "gy_y", "Value": 0.4823834623778427}, {"Timestamp": 1264, "Sensor": "gy_y", "Value": 0.3559372360596335}, {"Timestamp": 1265, "Sensor": "gy_y", "Value": 0.7438052771816576}, {"Timestamp": 1266, "Sensor": "gy_y", "Value": 0.2610984067513375}, {"Timestamp": 1267, "Sensor": "gy_y", "Value": 0.24202037800204745}, {"Timestamp": 1268, "Sensor": "gy_y", "Value": 0.22358639263677324}, {"Timestamp": 1269, "Sensor": "gy_y", "Value": 0.01993001498150865}, {"Timestamp": 1270, "Sensor": "gy_y", "Value": -0.028605916549748974}, {"Timestamp": 1271, "Sensor": "gy_y", "Value": 0.48851953769407336}, {"Timestamp": 1272, "Sensor": "gy_y", "Value": -0.07618200629598243}, {"Timestamp": 1273, "Sensor": "gy_y", "Value": -0.13032262533653308}, {"Timestamp": 1274, "Sensor": "gy_y", "Value": -0.30492849322591914}, {"Timestamp": 1275, "Sensor": "gy_y", "Value": -0.3225450207368302}, {"Timestamp": 1226, "Sensor": "gy_z", "Value": 0.27811674787952034}, {"Timestamp": 1227, "Sensor": "gy_z", "Value": 0.2937491780353129}, {"Timestamp": 1228, "Sensor": "gy_z", "Value": 0.3277487983130555}, {"Timestamp": 1229, "Sensor": "gy_z", "Value": 0.44137446921808926}, {"Timestamp": 1230, "Sensor": "gy_z", "Value": 0.6193612807322018}, {"Timestamp": 1231, "Sensor": "gy_z", "Value": 0.9143712061968713}, {"Timestamp": 1232, "Sensor": "gy_z", "Value": 1.1735866620219302}, {"Timestamp": 1233, "Sensor": "gy_z", "Value": 1.1713943302789367}, {"Timestamp": 1234, "Sensor": "gy_z", "Value": 1.1822121081884305}, {"Timestamp": 1235, "Sensor": "gy_z", "Value": 1.2455381362826399}, {"Timestamp": 1236, "Sensor": "gy_z", "Value": 0.9697592509453195}, {"Timestamp": 1237, "Sensor": "gy_z", "Value": 1.1423973510776237}, {"Timestamp": 1238, "Sensor": "gy_z", "Value": 1.2394228385723225}, {"Timestamp": 1239, "Sensor": "gy_z", "Value": 1.3190517435993727}, {"Timestamp": 1240, "Sensor": "gy_z", "Value": 1.2680665309527497}, {"Timestamp": 1241, "Sensor": "gy_z", "Value": 1.0470495593197144}, {"Timestamp": 1242, "Sensor": "gy_z", "Value": 0.876931408418455}, {"Timestamp": 1243, "Sensor": "gy_z", "Value": 0.7645393466647942}, {"Timestamp": 1244, "Sensor": "gy_z", "Value": 0.7417990644617026}, {"Timestamp": 1245, "Sensor": "gy_z", "Value": 0.6118045590577711}, {"Timestamp": 1246, "Sensor": "gy_z", "Value": 0.44276398972809305}, {"Timestamp": 1247, "Sensor": "gy_z", "Value": 0.14584449313853637}, {"Timestamp": 1248, "Sensor": "gy_z", "Value": 0.3594946806395436}, {"Timestamp": 1249, "Sensor": "gy_z", "Value": 1.016349014445326}, {"Timestamp": 1250, "Sensor": "gy_z", "Value": 1.0246383279939968}, {"Timestamp": 1251, "Sensor": "gy_z", "Value": 1.3074731167703653}, {"Timestamp": 1252, "Sensor": "gy_z", "Value": 1.5623042675092367}, {"Timestamp": 1253, "Sensor": "gy_z", "Value": 1.0566388880334059}, {"Timestamp": 1254, "Sensor": "gy_z", "Value": 0.9051324685899671}, {"Timestamp": 1255, "Sensor": "gy_z", "Value": 0.7971204862663928}, {"Timestamp": 1256, "Sensor": "gy_z", "Value": 0.745362227442292}, {"Timestamp": 1257, "Sensor": "gy_z", "Value": 1.0683548446087352}, {"Timestamp": 1258, "Sensor": "gy_z", "Value": 1.0583296203685422}, {"Timestamp": 1259, "Sensor": "gy_z", "Value": 0.3753499111124233}, {"Timestamp": 1260, "Sensor": "gy_z", "Value": 0.8123149748850297}, {"Timestamp": 1261, "Sensor": "gy_z", "Value": 0.3614310699438354}, {"Timestamp": 1262, "Sensor": "gy_z", "Value": -0.08307512658412898}, {"Timestamp": 1263, "Sensor": "gy_z", "Value": -0.0012792986775167076}, {"Timestamp": 1264, "Sensor": "gy_z", "Value": -0.005985266077391117}, {"Timestamp": 1265, "Sensor": "gy_z", "Value": 0.2629870156178711}, {"Timestamp": 1266, "Sensor": "gy_z", "Value": 0.5869264802315289}, {"Timestamp": 1267, "Sensor": "gy_z", "Value": 0.7807874042358163}, {"Timestamp": 1268, "Sensor": "gy_z", "Value": 0.7763059913580064}, {"Timestamp": 1269, "Sensor": "gy_z", "Value": 0.6307334645239273}, {"Timestamp": 1270, "Sensor": "gy_z", "Value": 0.7103918416038005}, {"Timestamp": 1271, "Sensor": "gy_z", "Value": 0.2997803315340578}, {"Timestamp": 1272, "Sensor": "gy_z", "Value": 0.6766645089805465}, {"Timestamp": 1273, "Sensor": "gy_z", "Value": 1.4458817429521507}, {"Timestamp": 1274, "Sensor": "gy_z", "Value": 1.3931731391648379}, {"Timestamp": 1275, "Sensor": "gy_z", "Value": 1.356033650860466}]}}, {"mode": "vega-lite"});
</script>




<style>
  #altair-viz-0021ffae7dbc48b2a2afa194dc0d9ac2.vega-embed {
    width: 100%;
    display: flex;
  }

  #altair-viz-0021ffae7dbc48b2a2afa194dc0d9ac2.vega-embed details,
  #altair-viz-0021ffae7dbc48b2a2afa194dc0d9ac2.vega-embed details summary {
    position: relative;
  }
</style>
<div id="altair-viz-0021ffae7dbc48b2a2afa194dc0d9ac2"></div>
<script type="text/javascript">
  var VEGA_DEBUG = (typeof VEGA_DEBUG == "undefined") ? {} : VEGA_DEBUG;
  (function(spec, embedOpt){
    let outputDiv = document.currentScript.previousElementSibling;
    if (outputDiv.id !== "altair-viz-0021ffae7dbc48b2a2afa194dc0d9ac2") {
      outputDiv = document.getElementById("altair-viz-0021ffae7dbc48b2a2afa194dc0d9ac2");
    }

    const paths = {
      "vega": "https://cdn.jsdelivr.net/npm/vega@5?noext",
      "vega-lib": "https://cdn.jsdelivr.net/npm/vega-lib?noext",
      "vega-lite": "https://cdn.jsdelivr.net/npm/vega-lite@5.20.1?noext",
      "vega-embed": "https://cdn.jsdelivr.net/npm/vega-embed@6?noext",
    };

    function maybeLoadScript(lib, version) {
      var key = `${lib.replace("-", "")}_version`;
      return (VEGA_DEBUG[key] == version) ?
        Promise.resolve(paths[lib]) :
        new Promise(function(resolve, reject) {
          var s = document.createElement('script');
          document.getElementsByTagName("head")[0].appendChild(s);
          s.async = true;
          s.onload = () => {
            VEGA_DEBUG[key] = version;
            return resolve(paths[lib]);
          };
          s.onerror = () => reject(`Error loading script: ${paths[lib]}`);
          s.src = paths[lib];
        });
    }

    function showError(err) {
      outputDiv.innerHTML = `<div class="error" style="color:red;">${err}</div>`;
      throw err;
    }

    function displayChart(vegaEmbed) {
      vegaEmbed(outputDiv, spec, embedOpt)
        .catch(err => showError(`Javascript Error: ${err.message}<br>This usually means there's a typo in your chart specification. See the javascript console for the full traceback.`));
    }

    if(typeof define === "function" && define.amd) {
      requirejs.config({paths});
      let deps = ["vega-embed"];
      require(deps, displayChart, err => showError(`Error loading script: ${err.message}`));
    } else {
      maybeLoadScript("vega", "5")
        .then(() => maybeLoadScript("vega-lite", "5.20.1"))
        .then(() => maybeLoadScript("vega-embed", "6"))
        .catch(showError)
        .then(() => displayChart(vegaEmbed));
    }
  })({"config": {"view": {"continuousWidth": 300, "continuousHeight": 300}}, "data": {"name": "data-ae383adf670f6802295645a33b50130c"}, "mark": {"type": "line"}, "encoding": {"color": {"field": "Sensor", "type": "nominal"}, "x": {"field": "Timestamp", "type": "quantitative"}, "y": {"field": "Value", "type": "quantitative"}}, "title": "Sensor recordings for activity: Lying", "$schema": "https://vega.github.io/schema/vega-lite/v5.20.1.json", "datasets": {"data-ae383adf670f6802295645a33b50130c": [{"Timestamp": 5046, "Sensor": "acc_z", "Value": -8.837217089956603}, {"Timestamp": 5047, "Sensor": "acc_z", "Value": -8.766358651375114}, {"Timestamp": 5048, "Sensor": "acc_z", "Value": -8.823548584851572}, {"Timestamp": 5049, "Sensor": "acc_z", "Value": -8.92115411716717}, {"Timestamp": 5050, "Sensor": "acc_z", "Value": -8.896996062511057}, {"Timestamp": 5051, "Sensor": "acc_z", "Value": -8.749357148732914}, {"Timestamp": 5052, "Sensor": "acc_z", "Value": -8.793249723842385}, {"Timestamp": 5053, "Sensor": "acc_z", "Value": -8.728958000556961}, {"Timestamp": 5054, "Sensor": "acc_z", "Value": -8.845936964696273}, {"Timestamp": 5055, "Sensor": "acc_z", "Value": -8.838578318524668}, {"Timestamp": 5056, "Sensor": "acc_z", "Value": -8.791160631806623}, {"Timestamp": 5057, "Sensor": "acc_z", "Value": -8.685802378201767}, {"Timestamp": 5058, "Sensor": "acc_z", "Value": -8.81468942967407}, {"Timestamp": 5059, "Sensor": "acc_z", "Value": -8.858046624893328}, {"Timestamp": 5060, "Sensor": "acc_z", "Value": -8.820560432854654}, {"Timestamp": 5061, "Sensor": "acc_z", "Value": -8.749590709270638}, {"Timestamp": 5062, "Sensor": "acc_z", "Value": -8.82943715312783}, {"Timestamp": 5063, "Sensor": "acc_z", "Value": -8.825237620493501}, {"Timestamp": 5064, "Sensor": "acc_z", "Value": -8.838810250020906}, {"Timestamp": 5065, "Sensor": "acc_z", "Value": -8.702073062844303}, {"Timestamp": 5066, "Sensor": "acc_z", "Value": -8.749582202174416}, {"Timestamp": 5067, "Sensor": "acc_z", "Value": -8.79702517289358}, {"Timestamp": 5068, "Sensor": "acc_z", "Value": -8.815959300001062}, {"Timestamp": 5069, "Sensor": "acc_z", "Value": -8.862317691720136}, {"Timestamp": 5070, "Sensor": "acc_z", "Value": -8.750232531480872}, {"Timestamp": 5071, "Sensor": "acc_z", "Value": -8.829391244624237}, {"Timestamp": 5072, "Sensor": "acc_z", "Value": -8.69619569152515}, {"Timestamp": 5073, "Sensor": "acc_z", "Value": -8.811147478424493}, {"Timestamp": 5074, "Sensor": "acc_z", "Value": -8.823129889235222}, {"Timestamp": 5075, "Sensor": "acc_z", "Value": -8.842486276300516}, {"Timestamp": 5076, "Sensor": "acc_z", "Value": -8.551973212986235}, {"Timestamp": 5077, "Sensor": "acc_z", "Value": -8.726011384790198}, {"Timestamp": 5078, "Sensor": "acc_z", "Value": -8.80884407579035}, {"Timestamp": 5079, "Sensor": "acc_z", "Value": -8.857354597892103}, {"Timestamp": 5080, "Sensor": "acc_z", "Value": -8.71960036054924}, {"Timestamp": 5081, "Sensor": "acc_z", "Value": -8.731503029655626}, {"Timestamp": 5082, "Sensor": "acc_z", "Value": -8.802791165811918}, {"Timestamp": 5083, "Sensor": "acc_z", "Value": -8.790746839681278}, {"Timestamp": 5084, "Sensor": "acc_z", "Value": -8.77940308851283}, {"Timestamp": 5085, "Sensor": "acc_z", "Value": -8.782003806701088}, {"Timestamp": 5086, "Sensor": "acc_z", "Value": -8.768247811625121}, {"Timestamp": 5087, "Sensor": "acc_z", "Value": -8.789739615687807}, {"Timestamp": 5088, "Sensor": "acc_z", "Value": -8.733743215568916}, {"Timestamp": 5089, "Sensor": "acc_z", "Value": -8.819367777543736}, {"Timestamp": 5090, "Sensor": "acc_z", "Value": -8.781218651532367}, {"Timestamp": 5091, "Sensor": "acc_z", "Value": -8.82118806431467}, {"Timestamp": 5092, "Sensor": "acc_z", "Value": -8.738364345837901}, {"Timestamp": 5093, "Sensor": "acc_z", "Value": -8.803133618815561}, {"Timestamp": 5094, "Sensor": "acc_z", "Value": -8.819746687054673}, {"Timestamp": 5095, "Sensor": "acc_z", "Value": -8.825041864120598}, {"Timestamp": 5046, "Sensor": "acc_XY", "Value": 9.632477918954647}, {"Timestamp": 5047, "Sensor": "acc_XY", "Value": 9.643650172258383}, {"Timestamp": 5048, "Sensor": "acc_XY", "Value": 9.779263881314373}, {"Timestamp": 5049, "Sensor": "acc_XY", "Value": 9.634093803389549}, {"Timestamp": 5050, "Sensor": "acc_XY", "Value": 9.69473325935737}, {"Timestamp": 5051, "Sensor": "acc_XY", "Value": 9.702275114013446}, {"Timestamp": 5052, "Sensor": "acc_XY", "Value": 9.7759692955085}, {"Timestamp": 5053, "Sensor": "acc_XY", "Value": 9.666243332180771}, {"Timestamp": 5054, "Sensor": "acc_XY", "Value": 9.675872591352096}, {"Timestamp": 5055, "Sensor": "acc_XY", "Value": 9.61237464588887}, {"Timestamp": 5056, "Sensor": "acc_XY", "Value": 9.743953408388744}, {"Timestamp": 5057, "Sensor": "acc_XY", "Value": 9.618494147494046}, {"Timestamp": 5058, "Sensor": "acc_XY", "Value": 9.674488352271473}, {"Timestamp": 5059, "Sensor": "acc_XY", "Value": 9.70677585259265}, {"Timestamp": 5060, "Sensor": "acc_XY", "Value": 9.633259862579783}, {"Timestamp": 5061, "Sensor": "acc_XY", "Value": 9.797459621903267}, {"Timestamp": 5062, "Sensor": "acc_XY", "Value": 9.710977681447748}, {"Timestamp": 5063, "Sensor": "acc_XY", "Value": 9.597323646187823}, {"Timestamp": 5064, "Sensor": "acc_XY", "Value": 9.65063207493151}, {"Timestamp": 5065, "Sensor": "acc_XY", "Value": 9.637271762066048}, {"Timestamp": 5066, "Sensor": "acc_XY", "Value": 9.692950632100098}, {"Timestamp": 5067, "Sensor": "acc_XY", "Value": 9.683809496383837}, {"Timestamp": 5068, "Sensor": "acc_XY", "Value": 9.677195380396064}, {"Timestamp": 5069, "Sensor": "acc_XY", "Value": 9.958990604653676}, {"Timestamp": 5070, "Sensor": "acc_XY", "Value": 9.57130875932418}, {"Timestamp": 5071, "Sensor": "acc_XY", "Value": 9.517310058106005}, {"Timestamp": 5072, "Sensor": "acc_XY", "Value": 9.789739306459458}, {"Timestamp": 5073, "Sensor": "acc_XY", "Value": 9.641235033280834}, {"Timestamp": 5074, "Sensor": "acc_XY", "Value": 9.702211216588744}, {"Timestamp": 5075, "Sensor": "acc_XY", "Value": 9.678320671912058}, {"Timestamp": 5076, "Sensor": "acc_XY", "Value": 9.71577692084211}, {"Timestamp": 5077, "Sensor": "acc_XY", "Value": 9.669585221309223}, {"Timestamp": 5078, "Sensor": "acc_XY", "Value": 9.629007728968269}, {"Timestamp": 5079, "Sensor": "acc_XY", "Value": 9.700771827812579}, {"Timestamp": 5080, "Sensor": "acc_XY", "Value": 9.676687932316767}, {"Timestamp": 5081, "Sensor": "acc_XY", "Value": 9.676048108998316}, {"Timestamp": 5082, "Sensor": "acc_XY", "Value": 9.731259672075197}, {"Timestamp": 5083, "Sensor": "acc_XY", "Value": 9.655124306601612}, {"Timestamp": 5084, "Sensor": "acc_XY", "Value": 9.681830641014033}, {"Timestamp": 5085, "Sensor": "acc_XY", "Value": 9.717629503444538}, {"Timestamp": 5086, "Sensor": "acc_XY", "Value": 9.63643929606956}, {"Timestamp": 5087, "Sensor": "acc_XY", "Value": 9.72102863773334}, {"Timestamp": 5088, "Sensor": "acc_XY", "Value": 9.590616238375862}, {"Timestamp": 5089, "Sensor": "acc_XY", "Value": 9.704529428046714}, {"Timestamp": 5090, "Sensor": "acc_XY", "Value": 9.65380084143581}, {"Timestamp": 5091, "Sensor": "acc_XY", "Value": 9.670607895832331}, {"Timestamp": 5092, "Sensor": "acc_XY", "Value": 9.715407514719697}, {"Timestamp": 5093, "Sensor": "acc_XY", "Value": 9.687020927132851}, {"Timestamp": 5094, "Sensor": "acc_XY", "Value": 9.650106147909742}, {"Timestamp": 5095, "Sensor": "acc_XY", "Value": 9.704361563303355}, {"Timestamp": 5046, "Sensor": "gy_x", "Value": -0.03977485882669787}, {"Timestamp": 5047, "Sensor": "gy_x", "Value": -0.004309954909500167}, {"Timestamp": 5048, "Sensor": "gy_x", "Value": 0.0009114605441039349}, {"Timestamp": 5049, "Sensor": "gy_x", "Value": -0.021404395412249904}, {"Timestamp": 5050, "Sensor": "gy_x", "Value": -0.030957742026353936}, {"Timestamp": 5051, "Sensor": "gy_x", "Value": -0.058784760030819956}, {"Timestamp": 5052, "Sensor": "gy_x", "Value": -0.03727323251632133}, {"Timestamp": 5053, "Sensor": "gy_x", "Value": -0.0011539575698426771}, {"Timestamp": 5054, "Sensor": "gy_x", "Value": -0.01629053168610025}, {"Timestamp": 5055, "Sensor": "gy_x", "Value": -0.010937262573562465}, {"Timestamp": 5056, "Sensor": "gy_x", "Value": -0.028351958780866243}, {"Timestamp": 5057, "Sensor": "gy_x", "Value": -0.02879530018122422}, {"Timestamp": 5058, "Sensor": "gy_x", "Value": -0.01609378473725174}, {"Timestamp": 5059, "Sensor": "gy_x", "Value": -0.0025216810588861126}, {"Timestamp": 5060, "Sensor": "gy_x", "Value": -0.021512575133746617}, {"Timestamp": 5061, "Sensor": "gy_x", "Value": -0.053192910499246335}, {"Timestamp": 5062, "Sensor": "gy_x", "Value": -0.004323822949777633}, {"Timestamp": 5063, "Sensor": "gy_x", "Value": -0.028275174754820363}, {"Timestamp": 5064, "Sensor": "gy_x", "Value": -0.032227110054458594}, {"Timestamp": 5065, "Sensor": "gy_x", "Value": -0.007338516277837371}, {"Timestamp": 5066, "Sensor": "gy_x", "Value": -0.024262688856036933}, {"Timestamp": 5067, "Sensor": "gy_x", "Value": -0.01902062074290127}, {"Timestamp": 5068, "Sensor": "gy_x", "Value": -0.03409466487324536}, {"Timestamp": 5069, "Sensor": "gy_x", "Value": -0.032541215751600724}, {"Timestamp": 5070, "Sensor": "gy_x", "Value": -0.011640159572756409}, {"Timestamp": 5071, "Sensor": "gy_x", "Value": -0.021737597284015746}, {"Timestamp": 5072, "Sensor": "gy_x", "Value": -0.03294324430920531}, {"Timestamp": 5073, "Sensor": "gy_x", "Value": -0.021111465644020666}, {"Timestamp": 5074, "Sensor": "gy_x", "Value": -0.02513832271866293}, {"Timestamp": 5075, "Sensor": "gy_x", "Value": -0.03139637754934423}, {"Timestamp": 5076, "Sensor": "gy_x", "Value": -0.033451062970319814}, {"Timestamp": 5077, "Sensor": "gy_x", "Value": -0.00904776178065244}, {"Timestamp": 5078, "Sensor": "gy_x", "Value": -0.0019273837721287478}, {"Timestamp": 5079, "Sensor": "gy_x", "Value": -0.018890960312697536}, {"Timestamp": 5080, "Sensor": "gy_x", "Value": -0.030298364326976773}, {"Timestamp": 5081, "Sensor": "gy_x", "Value": -0.024340443441123228}, {"Timestamp": 5082, "Sensor": "gy_x", "Value": -0.015168806448647938}, {"Timestamp": 5083, "Sensor": "gy_x", "Value": -0.012256709771725303}, {"Timestamp": 5084, "Sensor": "gy_x", "Value": -0.03421394968314553}, {"Timestamp": 5085, "Sensor": "gy_x", "Value": -0.023310202222937813}, {"Timestamp": 5086, "Sensor": "gy_x", "Value": -0.014426354078983513}, {"Timestamp": 5087, "Sensor": "gy_x", "Value": -0.025501189171524694}, {"Timestamp": 5088, "Sensor": "gy_x", "Value": -0.016738807205220148}, {"Timestamp": 5089, "Sensor": "gy_x", "Value": -0.013958584395200382}, {"Timestamp": 5090, "Sensor": "gy_x", "Value": -0.018405334649935975}, {"Timestamp": 5091, "Sensor": "gy_x", "Value": -0.02601359522695379}, {"Timestamp": 5092, "Sensor": "gy_x", "Value": -0.013260738484662977}, {"Timestamp": 5093, "Sensor": "gy_x", "Value": 0.005925825073403347}, {"Timestamp": 5094, "Sensor": "gy_x", "Value": -0.02137782029820302}, {"Timestamp": 5095, "Sensor": "gy_x", "Value": -0.027739274158029504}, {"Timestamp": 5046, "Sensor": "gy_y", "Value": 0.12790137864161905}, {"Timestamp": 5047, "Sensor": "gy_y", "Value": 0.11973071517588531}, {"Timestamp": 5048, "Sensor": "gy_y", "Value": 0.09662574316872657}, {"Timestamp": 5049, "Sensor": "gy_y", "Value": 0.10936830322324954}, {"Timestamp": 5050, "Sensor": "gy_y", "Value": 0.11022194986944578}, {"Timestamp": 5051, "Sensor": "gy_y", "Value": 0.11205191630626855}, {"Timestamp": 5052, "Sensor": "gy_y", "Value": 0.10150417632063703}, {"Timestamp": 5053, "Sensor": "gy_y", "Value": 0.09886860155502583}, {"Timestamp": 5054, "Sensor": "gy_y", "Value": 0.10960153714733978}, {"Timestamp": 5055, "Sensor": "gy_y", "Value": 0.09607344078034705}, {"Timestamp": 5056, "Sensor": "gy_y", "Value": 0.10975783578915635}, {"Timestamp": 5057, "Sensor": "gy_y", "Value": 0.12010209533029874}, {"Timestamp": 5058, "Sensor": "gy_y", "Value": 0.11707785794413295}, {"Timestamp": 5059, "Sensor": "gy_y", "Value": 0.10920742568556807}, {"Timestamp": 5060, "Sensor": "gy_y", "Value": 0.10426275108913262}, {"Timestamp": 5061, "Sensor": "gy_y", "Value": 0.10296196062596148}, {"Timestamp": 5062, "Sensor": "gy_y", "Value": 0.0990862361387371}, {"Timestamp": 5063, "Sensor": "gy_y", "Value": 0.1054469518713269}, {"Timestamp": 5064, "Sensor": "gy_y", "Value": 0.10220426925793939}, {"Timestamp": 5065, "Sensor": "gy_y", "Value": 0.11792815539548117}, {"Timestamp": 5066, "Sensor": "gy_y", "Value": 0.10871384660329403}, {"Timestamp": 5067, "Sensor": "gy_y", "Value": 0.1102974978046647}, {"Timestamp": 5068, "Sensor": "gy_y", "Value": 0.1155927125373361}, {"Timestamp": 5069, "Sensor": "gy_y", "Value": 0.09726093592127558}, {"Timestamp": 5070, "Sensor": "gy_y", "Value": 0.06346034056129478}, {"Timestamp": 5071, "Sensor": "gy_y", "Value": 0.08633008998211075}, {"Timestamp": 5072, "Sensor": "gy_y", "Value": 0.11144181508144432}, {"Timestamp": 5073, "Sensor": "gy_y", "Value": 0.12476451064405655}, {"Timestamp": 5074, "Sensor": "gy_y", "Value": 0.09939197562261806}, {"Timestamp": 5075, "Sensor": "gy_y", "Value": 0.08700697597223156}, {"Timestamp": 5076, "Sensor": "gy_y", "Value": 0.1140766275871834}, {"Timestamp": 5077, "Sensor": "gy_y", "Value": 0.13413164813016848}, {"Timestamp": 5078, "Sensor": "gy_y", "Value": 0.11086730054475646}, {"Timestamp": 5079, "Sensor": "gy_y", "Value": 0.11467230353916438}, {"Timestamp": 5080, "Sensor": "gy_y", "Value": 0.10978225065659088}, {"Timestamp": 5081, "Sensor": "gy_y", "Value": 0.11720665439173826}, {"Timestamp": 5082, "Sensor": "gy_y", "Value": 0.11466776466564649}, {"Timestamp": 5083, "Sensor": "gy_y", "Value": 0.1069486163952329}, {"Timestamp": 5084, "Sensor": "gy_y", "Value": 0.1133271775684278}, {"Timestamp": 5085, "Sensor": "gy_y", "Value": 0.1127686998471629}, {"Timestamp": 5086, "Sensor": "gy_y", "Value": 0.10576266618947798}, {"Timestamp": 5087, "Sensor": "gy_y", "Value": 0.1022016554906891}, {"Timestamp": 5088, "Sensor": "gy_y", "Value": 0.11722060924555222}, {"Timestamp": 5089, "Sensor": "gy_y", "Value": 0.10987206802273723}, {"Timestamp": 5090, "Sensor": "gy_y", "Value": 0.10150351780838042}, {"Timestamp": 5091, "Sensor": "gy_y", "Value": 0.11136482302308415}, {"Timestamp": 5092, "Sensor": "gy_y", "Value": 0.10209135738761206}, {"Timestamp": 5093, "Sensor": "gy_y", "Value": 0.10072851731122637}, {"Timestamp": 5094, "Sensor": "gy_y", "Value": 0.10227359334745487}, {"Timestamp": 5095, "Sensor": "gy_y", "Value": 0.1149980099553734}, {"Timestamp": 5046, "Sensor": "gy_z", "Value": 0.08702166758853483}, {"Timestamp": 5047, "Sensor": "gy_z", "Value": 0.10733964826264855}, {"Timestamp": 5048, "Sensor": "gy_z", "Value": 0.12822381966611285}, {"Timestamp": 5049, "Sensor": "gy_z", "Value": 0.09198592729925269}, {"Timestamp": 5050, "Sensor": "gy_z", "Value": 0.08084460974367494}, {"Timestamp": 5051, "Sensor": "gy_z", "Value": 0.061438595062158645}, {"Timestamp": 5052, "Sensor": "gy_z", "Value": 0.056055875100236165}, {"Timestamp": 5053, "Sensor": "gy_z", "Value": 0.0313076536086713}, {"Timestamp": 5054, "Sensor": "gy_z", "Value": 0.10041665168847479}, {"Timestamp": 5055, "Sensor": "gy_z", "Value": 0.061093622678820556}, {"Timestamp": 5056, "Sensor": "gy_z", "Value": 0.045460428563407214}, {"Timestamp": 5057, "Sensor": "gy_z", "Value": 0.04168292780917654}, {"Timestamp": 5058, "Sensor": "gy_z", "Value": 0.08125119943757403}, {"Timestamp": 5059, "Sensor": "gy_z", "Value": 0.11374957033223636}, {"Timestamp": 5060, "Sensor": "gy_z", "Value": 0.11552152105750896}, {"Timestamp": 5061, "Sensor": "gy_z", "Value": 0.0887298597185701}, {"Timestamp": 5062, "Sensor": "gy_z", "Value": 0.04039455614259685}, {"Timestamp": 5063, "Sensor": "gy_z", "Value": 0.04146675718729933}, {"Timestamp": 5064, "Sensor": "gy_z", "Value": 0.025407890762289508}, {"Timestamp": 5065, "Sensor": "gy_z", "Value": 0.06068077986363692}, {"Timestamp": 5066, "Sensor": "gy_z", "Value": 0.08669517856946023}, {"Timestamp": 5067, "Sensor": "gy_z", "Value": 0.09809458310520874}, {"Timestamp": 5068, "Sensor": "gy_z", "Value": 0.08548994163901312}, {"Timestamp": 5069, "Sensor": "gy_z", "Value": 0.028493767320684364}, {"Timestamp": 5070, "Sensor": "gy_z", "Value": -0.05397516434403189}, {"Timestamp": 5071, "Sensor": "gy_z", "Value": 0.043341867403223565}, {"Timestamp": 5072, "Sensor": "gy_z", "Value": 0.06991450331467125}, {"Timestamp": 5073, "Sensor": "gy_z", "Value": 0.07648992129104618}, {"Timestamp": 5074, "Sensor": "gy_z", "Value": 0.08300551018893416}, {"Timestamp": 5075, "Sensor": "gy_z", "Value": 0.06422347687525612}, {"Timestamp": 5076, "Sensor": "gy_z", "Value": 0.047506381365988795}, {"Timestamp": 5077, "Sensor": "gy_z", "Value": 0.0961282117312355}, {"Timestamp": 5078, "Sensor": "gy_z", "Value": 0.07966874107021889}, {"Timestamp": 5079, "Sensor": "gy_z", "Value": 0.10063921214658746}, {"Timestamp": 5080, "Sensor": "gy_z", "Value": 0.08780558407574793}, {"Timestamp": 5081, "Sensor": "gy_z", "Value": 0.10394242670048216}, {"Timestamp": 5082, "Sensor": "gy_z", "Value": 0.09080645932700697}, {"Timestamp": 5083, "Sensor": "gy_z", "Value": 0.0717163574696667}, {"Timestamp": 5084, "Sensor": "gy_z", "Value": 0.06534435920914079}, {"Timestamp": 5085, "Sensor": "gy_z", "Value": 0.06347540229180298}, {"Timestamp": 5086, "Sensor": "gy_z", "Value": 0.0640336957115615}, {"Timestamp": 5087, "Sensor": "gy_z", "Value": 0.057735699881447805}, {"Timestamp": 5088, "Sensor": "gy_z", "Value": 0.04792546199514436}, {"Timestamp": 5089, "Sensor": "gy_z", "Value": 0.08081260154882453}, {"Timestamp": 5090, "Sensor": "gy_z", "Value": 0.07642382786113024}, {"Timestamp": 5091, "Sensor": "gy_z", "Value": 0.07303214871140899}, {"Timestamp": 5092, "Sensor": "gy_z", "Value": 0.07634793368910678}, {"Timestamp": 5093, "Sensor": "gy_z", "Value": 0.09224840320288148}, {"Timestamp": 5094, "Sensor": "gy_z", "Value": 0.07414989322377463}, {"Timestamp": 5095, "Sensor": "gy_z", "Value": 0.06871311966058241}]}}, {"mode": "vega-lite"});
</script>


# Format data for modelling

This section consists on four procedurals used to clear out the data and to set it ready for fitting it to the model. Those are:

- Splitting the data into train and test sets.
- Creating sequences.
- One-hot encoding the categorical variables.

Sequences were created so the model was able to handle the data more easily. Thus, the raw data was summarized into sequences of 1 timestep, were each step was worth 10 measurements. The labelling for every step was created upon the mode of the previous mentioned measurements. The remaining sequences had the shape (12382, 1, 5), which can be understood as a total of 12,382 sequences, each 1 timestep long, containing the 5 sensor measurements with its corresponding label.


```python
def split_train_test(df, n_users_test=2):
    df_train = df[df['User'] >= n_users_test]
    df_test = df[df['User'] < n_users_test]
    
    return df_train, df_test

def create_sequence(X, y, time_steps=1, step=1):
    Xs, ys = [], []
    for i in range(0, len(X) - time_steps, step):
        v = X.iloc[i:(i + time_steps)].values
        labels = y.iloc[i: i + time_steps]
        Xs.append(v)
        ys.append(stats.mode(labels)[0])
    return np.array(Xs), np.array(ys).reshape(-1, 1)

def one_hot_encode(y_train, y_test):
    ohe = OneHotEncoder()
    y_train_encoded = ohe.fit_transform(y_train)
    y_test_encoded = ohe.transform(y_test)
    return y_train_encoded.toarray(), y_test_encoded.toarray()

def format_to_model(df):
    df_train, df_test = split_train_test(df)

    X_train = df_train[['acc_z', 'acc_XY', 'gy_x', 'gy_y', 'gy_z']]
    X_test = df_test[['acc_z', 'acc_XY', 'gy_x', 'gy_y', 'gy_z']]
    X_train, y_train = create_sequence(X_train, df_train['Activity'], time_steps=1, step=10)
    X_test, y_test = create_sequence(X_test, df_test['Activity'], time_steps=1, step=10)

    y_train_encoded, y_test_encoded = one_hot_encode(y_train, y_test)
    return X_train, y_train_encoded, X_test, y_test_encoded

X_train, y_train_encoded, X_test, y_test_encoded = format_to_model(df)
```

# Create and fit the model
Long Short Term Memory (LSTM) is a deep learning algorithm designed for classifying time series data by learning long-term dependencies in sequences. Unlike traditional neural networks, LSTMs incorporate feedback connections and memory cells with gates to control input, output, and forgetting, enabling them to maintain state memory across sequences. This simplifies feature extraction, dimensionality reduction, and classification into a single process. In this project, a bidirectional LSTM model was used, complemented by a dropout layer to prevent overfitting and activation layers for non-linearity. The model was trained over 20 epochs on 11,143 samples and validated on 1,239 samples, effectively learning to classify the input sequences.


```python
# Create the LSTM model with keras
model = keras.Sequential()
model.add(
    keras.layers.Bidirectional(
      keras.layers.LSTM(
          units=32,
          input_shape=[X_train.shape[1], X_train.shape[2]]
      )
    )
)
model.add(keras.layers.Dropout(rate=0.5))
model.add(keras.layers.Dense(units=16, activation='relu'))
model.add(keras.layers.Dense(y_train_encoded.shape[1], activation='softmax'))

model.compile(
  loss='categorical_crossentropy',
  optimizer='adam',
  metrics=['acc']
)

```


```python
# Fit the data to the model
history = model.fit(
    X_train, y_train_encoded,
    epochs=20,
    batch_size=64,
    validation_split=0.2,
    shuffle=False)
```

    Epoch 1/20
    132/132 [==============================] - 9s 14ms/step - loss: 1.3695 - acc: 0.5317 - val_loss: 1.0216 - val_acc: 0.8272
    Epoch 2/20
    132/132 [==============================] - 1s 6ms/step - loss: 0.9780 - acc: 0.6852 - val_loss: 0.7076 - val_acc: 0.7749
    Epoch 3/20
    132/132 [==============================] - 1s 6ms/step - loss: 0.7707 - acc: 0.7376 - val_loss: 0.6033 - val_acc: 0.8010
    Epoch 4/20
    132/132 [==============================] - 1s 6ms/step - loss: 0.6877 - acc: 0.7606 - val_loss: 0.5601 - val_acc: 0.8186
    Epoch 5/20
    132/132 [==============================] - 1s 6ms/step - loss: 0.6378 - acc: 0.7818 - val_loss: 0.5309 - val_acc: 0.8305
    Epoch 6/20
    132/132 [==============================] - 1s 6ms/step - loss: 0.5980 - acc: 0.8017 - val_loss: 0.4924 - val_acc: 0.8371
    Epoch 7/20
    132/132 [==============================] - 1s 6ms/step - loss: 0.5755 - acc: 0.8149 - val_loss: 0.4847 - val_acc: 0.8443
    Epoch 8/20
    132/132 [==============================] - 1s 6ms/step - loss: 0.5448 - acc: 0.8316 - val_loss: 0.4675 - val_acc: 0.8523
    Epoch 9/20
    132/132 [==============================] - 1s 8ms/step - loss: 0.5247 - acc: 0.8416 - val_loss: 0.4531 - val_acc: 0.8618
    Epoch 10/20
    132/132 [==============================] - 1s 9ms/step - loss: 0.5046 - acc: 0.8492 - val_loss: 0.4402 - val_acc: 0.8656
    Epoch 11/20
    132/132 [==============================] - 1s 6ms/step - loss: 0.4795 - acc: 0.8598 - val_loss: 0.4273 - val_acc: 0.8670
    Epoch 12/20
    132/132 [==============================] - 1s 6ms/step - loss: 0.4661 - acc: 0.8650 - val_loss: 0.4190 - val_acc: 0.8704
    Epoch 13/20
    132/132 [==============================] - 1s 7ms/step - loss: 0.4446 - acc: 0.8722 - val_loss: 0.4102 - val_acc: 0.8746
    Epoch 14/20
    132/132 [==============================] - 1s 11ms/step - loss: 0.4305 - acc: 0.8772 - val_loss: 0.3946 - val_acc: 0.8780
    Epoch 15/20
    132/132 [==============================] - 1s 7ms/step - loss: 0.4161 - acc: 0.8828 - val_loss: 0.3838 - val_acc: 0.8780
    Epoch 16/20
    132/132 [==============================] - 1s 6ms/step - loss: 0.3951 - acc: 0.8892 - val_loss: 0.3940 - val_acc: 0.8808
    Epoch 17/20
    132/132 [==============================] - 1s 7ms/step - loss: 0.3909 - acc: 0.8924 - val_loss: 0.3900 - val_acc: 0.8818
    Epoch 18/20
    132/132 [==============================] - 1s 6ms/step - loss: 0.3772 - acc: 0.8950 - val_loss: 0.3864 - val_acc: 0.8827
    Epoch 19/20
    132/132 [==============================] - 1s 6ms/step - loss: 0.3686 - acc: 0.8992 - val_loss: 0.3660 - val_acc: 0.8832
    Epoch 20/20
    132/132 [==============================] - 1s 6ms/step - loss: 0.3609 - acc: 0.9028 - val_loss: 0.3642 - val_acc: 0.8865



```python
def plot_history(history):
    """
    Plot the training history of the model.
    """
    fig, ax = plt.subplots(1, 2, figsize=(10, 4))
    
    # Plot accuracy
    ax[0].plot(history.history['acc'], label='Train Accuracy')
    ax[0].plot(history.history['val_acc'], label='Validation Accuracy')
    ax[0].set_title('Model Accuracy')
    ax[0].set_ylabel('Accuracy')
    ax[0].set_xlabel('Epoch')
    ax[0].legend()
    
    # Plot loss
    ax[1].plot(history.history['loss'], label='Train Loss')
    ax[1].plot(history.history['val_loss'], label='Validation Loss')
    ax[1].set_title('Model Loss')
    ax[1].set_ylabel('Loss')
    ax[1].set_xlabel('Epoch')
    ax[1].legend()
    
    plt.tight_layout()
    plt.show()

plot_history(history)
```


    
![png](images/Human%20Activity%20Recognition_14_0.png)
    


# Model evaluation
The aim of this section was to evaluate the model in terms of overfitting and accuracy. The model showed no relevant overfitting and an accuracy of 85.6%.

Moreover, there was created a confusion matrix in order to asses the labelling of each of the classes. It showed the most misclassified class was lying, which was confused with seating. 


```python
# Evaluate the accuracy on the test set
model.evaluate(x=X_test,y=y_test_encoded)
```

    113/113 [==============================] - 0s 3ms/step - loss: 0.3241 - acc: 0.8653




```python
# Create confusion matrix
y_pred = model.predict(X_test)
y_test = y_test_encoded.argmax(1)
y_pred = y_pred.argmax(1)
conf = confusion_matrix(y_test, y_pred)

def plot_confusion_matrix(conf):
    plt.figure(figsize=(10,7))
    plt.imshow(conf, interpolation='nearest', cmap=plt.cm.Blues)
    # add numbers
    for i in range(len(conf)):
        for j in range(len(conf)):
            plt.text(j, i, conf[i][j], ha='center', va='center', color='white' if conf[i][j] > conf.max()/2 else 'black')

    # translate the axis into the activity labels
    plt.xticks(np.arange(len(dict_labels)), list(dict_labels.values()), rotation=45)
    plt.yticks(np.arange(len(dict_labels)), list(dict_labels.values()))
    plt.title('Confusion Matrix')
    plt.xlabel('Predicted Label')
    plt.ylabel('True Label')
    plt.show()

plot_confusion_matrix(conf)
```

    113/113 [==============================] - 0s 3ms/step



    
![png](images/Human%20Activity%20Recognition_17_1.png)
    

