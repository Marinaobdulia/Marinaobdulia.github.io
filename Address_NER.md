[← Back](/index.md)

# Training a spaCy Model for Named Entity Recognition (NER)
---
This project aims to train a natural language processing (NLP) model using the spaCy library to perform named entity recognition (NER) tasks.

Named entity recognition is a key technique in NLP that allows identifying and classifying specific entities in a text, such as names of people, organizations, locations, dates, quantities, among others.

## Key goals

1. **Prepare the training data**: Generate a labeled dataset for training the model.
2. **Configure the model**: Define the model parameters and configure spaCy to work with custom data.
3. **Train the model**: Use the data to fine-tune the spaCy model and improve its ability to detect the desired entities.
4. **Evaluate the model**: Validate the model's performance using metrics such as precision, recall, and F1-score.
5. **Save and reuse the model**: Export the trained model to integrate it into other applications.

---

*__Importing libraries__*


```python
import pandas as pd
import random

import pandas as pd
from sklearn.model_selection import train_test_split
import spacy

import random
from spacy.util import minibatch, compounding
from spacy.training.example import Example
from pathlib import Path
from spacy.tokens import Doc, Span, Token
from spacy.scorer import Scorer
```

# 1. Prepare Training Data
### 1.1 Obtain Street Examples

To create our training set, we will use the streets of the city of Madrid. This way, we will create examples with real names that will later help the model generalize.

We download the data from the Madrid street directory from [here](https://datos.madrid.es/portal/site/egob/menuitem.c05c1f754a33a9fbe4b2e4b284f1a5a0/?vgnextoid=b3c41f3cf6a6c410VgnVCM2000000c205a0aRCRD&vgnextchannel=374512b9ace9f310VgnVCM100000171f5a0aRCRD&vgnextfmt=default)



```python
df_callejero = pd.read_csv('../data/VialesVigentes_20241226.csv', sep = ';', encoding = 'cp1252')
```

Once the data is loaded, we visualize it to understand the information we are working with. We find 15 columns, of which only 3 are of interest to us: VIA_CLASE, VIA_PAR, and VIA_NOMBRE. That is type of street, particle and name.


```python
df_callejero.head(5)
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
      <th>COD_VIA</th>
      <th>VIA_CLASE</th>
      <th>VIA_PAR</th>
      <th>VIA_NOMBRE</th>
      <th>VIA_NOMBRE_ACENTOS</th>
      <th>COD_VIA_COMIENZA</th>
      <th>CLASE_COMIENZA</th>
      <th>PARTICULA_COMIENZA</th>
      <th>NOMBRE_COMIENZA</th>
      <th>NOMBRE_ACENTOS_COMIENZA</th>
      <th>COD_VIA_TERMINA</th>
      <th>CLASE_TERMINA</th>
      <th>PARTICULA_TERMINA</th>
      <th>NOMBRE_TERMINA</th>
      <th>NOMBRE_ACENTOS_TERMINA</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>31001337</td>
      <td>AUTOVÍA</td>
      <td>NaN</td>
      <td>A-1</td>
      <td>A-1</td>
      <td>31001349</td>
      <td>AUTOVÍA</td>
      <td>NaN</td>
      <td>M-30</td>
      <td>M-30</td>
      <td>99000003</td>
      <td>LUGAR</td>
      <td>NaN</td>
      <td>LIMITE TERMINO MUNICIPAL</td>
      <td>LÍMITE TÉRMINO MUNICIPAL</td>
    </tr>
    <tr>
      <th>1</th>
      <td>31001336</td>
      <td>AUTOVÍA</td>
      <td>NaN</td>
      <td>A-2</td>
      <td>A-2</td>
      <td>310200</td>
      <td>CALLE</td>
      <td>DE</td>
      <td>FRANCISCO SILVELA</td>
      <td>FRANCISCO SILVELA</td>
      <td>99000003</td>
      <td>LUGAR</td>
      <td>NaN</td>
      <td>LIMITE TERMINO MUNICIPAL</td>
      <td>LÍMITE TÉRMINO MUNICIPAL</td>
    </tr>
    <tr>
      <th>2</th>
      <td>31001342</td>
      <td>AUTOVÍA</td>
      <td>NaN</td>
      <td>A-3</td>
      <td>A-3</td>
      <td>480800</td>
      <td>PLAZA</td>
      <td>DE</td>
      <td>MARIANO DE CAVIA</td>
      <td>MARIANO DE CAVIA</td>
      <td>99000003</td>
      <td>LUGAR</td>
      <td>NaN</td>
      <td>LIMITE TERMINO MUNICIPAL</td>
      <td>LÍMITE TÉRMINO MUNICIPAL</td>
    </tr>
    <tr>
      <th>3</th>
      <td>31001334</td>
      <td>AUTOVÍA</td>
      <td>NaN</td>
      <td>A-4</td>
      <td>A-4</td>
      <td>31001349</td>
      <td>AUTOVÍA</td>
      <td>NaN</td>
      <td>M-30</td>
      <td>M-30</td>
      <td>99000003</td>
      <td>LUGAR</td>
      <td>NaN</td>
      <td>LIMITE TERMINO MUNICIPAL</td>
      <td>LÍMITE TÉRMINO MUNICIPAL</td>
    </tr>
    <tr>
      <th>4</th>
      <td>31001341</td>
      <td>AUTOVÍA</td>
      <td>NaN</td>
      <td>A-42</td>
      <td>A-42</td>
      <td>468400</td>
      <td>AVENIDA</td>
      <td>DEL</td>
      <td>MANZANARES</td>
      <td>MANZANARES</td>
      <td>99000003</td>
      <td>LUGAR</td>
      <td>NaN</td>
      <td>LIMITE TERMINO MUNICIPAL</td>
      <td>LÍMITE TÉRMINO MUNICIPAL</td>
    </tr>
  </tbody>
</table>
</div>



Once the columns of interest have been selected, we will create an additional column that concatenates the particle and the street name, as we are not interested in having our model detect the particles separately from the street name.

Likewise, we will remove the records that contain empty values.


```python
df_calles = df_callejero[df_callejero.VIA_CLASE != 'AUTOVÍA'][['VIA_CLASE', 'VIA_PAR', 'VIA_NOMBRE']]

df_calles['NOMBRE_VIA'] = df_calles['VIA_PAR'] + ' ' + df_calles['VIA_NOMBRE']
df_calles.drop(columns = ['VIA_PAR', 'VIA_NOMBRE'], inplace  = True)

df_calles.rename(columns={'VIA_CLASE': 'TIPO_VIA'}, inplace = True)
df_calles.dropna(inplace =  True)
```

The result is a table with two columns, the type of street and the name of the street.


```python
df_calles.head(5)
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
      <th>TIPO_VIA</th>
      <th>NOMBRE_VIA</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>7</th>
      <td>CALLE</td>
      <td>DEL ABAD JUAN CATALAN</td>
    </tr>
    <tr>
      <th>8</th>
      <td>CALLE</td>
      <td>DE LA ABADA</td>
    </tr>
    <tr>
      <th>9</th>
      <td>CALLE</td>
      <td>DE LOS ABADES</td>
    </tr>
    <tr>
      <th>10</th>
      <td>CALLE</td>
      <td>DE LA ABADESA</td>
    </tr>
    <tr>
      <th>11</th>
      <td>CALLE</td>
      <td>DE ABALOS</td>
    </tr>
  </tbody>
</table>
</div>



### 1.2. Generate Fictional Addresses

To complete the creation of fictional addresses, we will add more details to the streets of Madrid. For this, we will randomly include information about the number and the floor-letter combination. This step can be improved as we discover real cases that are not covered by the ones we generate.


```python
def generar_datos_entrenamiento(df, n):
    """
    Genera N líneas aleatorias a partir de un DataFrame base.
    
    Args:
        df (pd.DataFrame): DataFrame base para seleccionar líneas.
        n (int): Número de líneas aleatorias a generar.
    
    Returns:
        pd.DataFrame: DataFrame con N líneas aleatorias generadas.
    """
    lineas = []
    for _ in range(n):
        # Seleccionar una línea aleatoria del DataFrame base
        row = df.sample(1).copy()

        # Agregar un número aleatorio
        pre_numero = random.choice(['n*', 'nr', 'nº', '', 'n', 'num', 'núm'])
        numero = random.randint(1, 100)
        post_numero = random.choice(['', '', '', ','])
        row['NUMERO'] = f'{pre_numero}{numero}{post_numero}'

        # Agregar combinación aleatoria de piso y letra
        piso = random.randint(1, 10)
        espacio = random.choice([' ', '*', '*', 'º', 'º', 'º', '', '-', '-', '-'])
        letra = random.choice(['A', 'B', 'C', 'D', 'IZDA', 'DCHA', 'CTRO'])
        combinacion = f"{piso}{espacio}{letra}"
        row['RESTO'] = combinacion

        # Agregar la línea generada a la lista
        lineas.append(row)

    # Combinar todas las líneas en un nuevo DataFrame
    return pd.concat(lineas, ignore_index=True)

```


```python
train_set = generar_datos_entrenamiento(df_calles, 7000)
```

### 1.3. Structuring the Data for the Model
Once we finish generating the fictional addresses, we will shape our data so that we can train the SpaCy model. This structure is specific to entity recognition using the SpaCy library.

It consists of a tuple in which the full text appears in the first position, and the entity definitions are in the second. Here's an example:

```
('CALLE DE LOS MARTIRES DE PARACUELLOS n44 3 D',
{'entities': [(0, 5, 'TIPO_VIA'),
    (6, 36, 'NOMBRE_VIA'),
    (37, 40, 'NUMERO'),
    (41, 44, 'RESTO')]})
```
In our case, the model will have four types of entities: the type of street, the street name, the number, and the rest of the address. This means that the entire text is tagged with some entity.

However, we could also create models that detect, for example, only the street name. In that case, not all of the text would be categorized, but only a portion.




```python
def crear_entidades(row):
    texto = row['texto']
    entidades = []
    start = 0
    for col, label in zip(['TIPO_VIA', 'NOMBRE_VIA', 'NUMERO', 'RESTO'], ['TIPO_VIA', 'NOMBRE_VIA', 'NUMERO', 'RESTO']):
        value = str(row[col])
        start = texto.find(value, start)  # Buscar el índice inicial de la entidad
        if start != -1:
            end = start + len(value)  # Índice final de la entidad
            entidades.append((start, end, label))
            start = end  # Actualizar el inicio para evitar errores en textos repetidos
    return (texto, {'entities': entidades})
```


```python
train_set['texto'] = train_set['TIPO_VIA'] + ' ' + train_set['NOMBRE_VIA'] + ' ' + train_set['NUMERO'].astype(str) + ' ' + train_set['RESTO']
train_set['entidades'] = train_set.apply(crear_entidades, axis=1)

lista = train_set['entidades'].to_list()

train, test = train_test_split(lista, test_size=0.3, random_state=8)
```

## 2. Configure the Model

To train the model, we first need to select which components of the pre-configured model pipeline we want to modify with our training data. In our case, these will be: ner, trf_wordpiecer, and trf_tok2vec.

Besides the ner, the last two are components responsible for tokenization and vector creation. Modifying them will help the ner component improve its generalization ability.




```python
nlp = spacy.load('en_core_web_sm')

ner = nlp.get_pipe('ner')

for _,annotations in train:
    for ent in annotations.get("entities"):
        ner.add_label(ent[2])

pipe_exceptions = ['ner', 'trf_wordpiecer', 'trf_tok2vec']
unaffected_pipe = [pipe for pipe in nlp.pipe_names if pipe not in pipe_exceptions]
```

## 3. Train the Model

To train the model, we will disable the pipeline components that we previously indicated we do not want to modify. Then, we create a training loop with its respective batches, in which we will update the model and calculate the losses at each batch step. We will repeat this process as many times as the number of iterations specified.

It is worth highlighting the "drop" parameter, which we will use to discard training examples and thus avoid overfitting.


```python
nr_iter = 50

with nlp.disable_pipes(*unaffected_pipe):
    for iteration in range(nr_iter):
        random.shuffle(train)
        losses = {}

        batches = minibatch(train, size=8)
        for batch in batches:
            example = []
            for text, annotations in batch:
                doc = nlp.make_doc(text)
                example.append(Example.from_dict(doc, annotations))

            nlp.update(example, drop = 0.3, losses=losses)

        if iteration % 10 == 0:
            print(f"Iteration {iteration} - Losses: {losses}")

```

    Iteration 0 - Losses: {'ner': np.float32(2642.9849)}
    Iteration 10 - Losses: {'ner': np.float32(0.014348858)}
    Iteration 20 - Losses: {'ner': np.float32(32.123077)}
    Iteration 30 - Losses: {'ner': np.float32(3.1996186e-07)}
    Iteration 40 - Losses: {'ner': np.float32(1.6292404)}


## 4. Evaluate the Model

In this section, we will evaluate the performance of the trained model using the test dataset. The evaluation will be carried out using standard metrics such as precision, recall, and F1-score. These metrics will allow us to understand how well the model can identify and classify named entities in texts not seen during training.


```python
examples = []
scorer = Scorer()
for text, annotations in test:
    doc = nlp.make_doc(text)
    example = Example.from_dict(doc, annotations)
    example.predicted = nlp(str(example.predicted))
    examples.append(example)

results_scorer = scorer.score(examples)
results_scorer['ents_per_type']
```




    {'TIPO_VIA': {'p': 0.9980988593155894, 'r': 1.0, 'f': 0.9990485252140818},
     'NOMBRE_VIA': {'p': 0.9895635673624289,
      'r': 0.9933333333333333,
      'f': 0.9914448669201521},
     'NUMERO': {'p': 0.9985734664764622, 'r': 1.0, 'f': 0.9992862241256245},
     'RESTO': {'p': 0.9980961446930033,
      'r': 0.9985714285714286,
      'f': 0.9983337300642704}}


