<div itemscope itemtype="http://schema.org/Dataset">
  <div itemscope itemprop="includedInDataCatalog" itemtype="http://schema.org/DataCatalog">
    <meta itemprop="name" content="TensorFlow Datasets" />
  </div>
  <meta itemprop="name" content="protein_net" />
  <meta itemprop="description" content="ProteinNet is a standardized data set for machine learning of protein structure.&#10;It provides protein sequences, structures (secondary and tertiary), multiple&#10;sequence alignments (MSAs), position-specific scoring matrices (PSSMs), and&#10;standardized training / validation / test splits. ProteinNet builds on the&#10;biennial CASP assessments, which carry out blind predictions of recently solved&#10;but publicly unavailable protein structures, to provide test sets that push the&#10;frontiers of computational methodology. It is organized as a series of data&#10;sets, spanning CASP 7 through 12 (covering a ten-year period), to provide a&#10;range of data set sizes that enable assessment of new methods in relatively data&#10;poor and data rich regimes.&#10;&#10;To use this dataset:&#10;&#10;```python&#10;import tensorflow_datasets as tfds&#10;&#10;ds = tfds.load(&#x27;protein_net&#x27;, split=&#x27;train&#x27;)&#10;for ex in ds.take(4):&#10;  print(ex)&#10;```&#10;&#10;See [the guide](https://www.tensorflow.org/datasets/overview) for more&#10;informations on [tensorflow_datasets](https://www.tensorflow.org/datasets).&#10;&#10;" />
  <meta itemprop="url" content="https://www.tensorflow.org/datasets/catalog/protein_net" />
  <meta itemprop="sameAs" content="https://github.com/aqlaboratory/proteinnet" />
  <meta itemprop="citation" content="@article{ProteinNet19,&#10;title = {{ProteinNet}: a standardized data set for machine learning of protein structure},&#10;author = {AlQuraishi, Mohammed},&#10;journal = {BMC bioinformatics},&#10;volume = {20},&#10;number = {1},&#10;pages = {1--10},&#10;year = {2019},&#10;publisher = {BioMed Central}&#10;}" />
</div>

# `protein_net`


*   **Description**:

ProteinNet is a standardized data set for machine learning of protein structure.
It provides protein sequences, structures (secondary and tertiary), multiple
sequence alignments (MSAs), position-specific scoring matrices (PSSMs), and
standardized training / validation / test splits. ProteinNet builds on the
biennial CASP assessments, which carry out blind predictions of recently solved
but publicly unavailable protein structures, to provide test sets that push the
frontiers of computational methodology. It is organized as a series of data
sets, spanning CASP 7 through 12 (covering a ten-year period), to provide a
range of data set sizes that enable assessment of new methods in relatively data
poor and data rich regimes.

*   **Homepage**:
    [https://github.com/aqlaboratory/proteinnet](https://github.com/aqlaboratory/proteinnet)

*   **Source code**:
    [`tfds.structured.proteinnet.ProteinNet`](https://github.com/tensorflow/datasets/tree/master/tensorflow_datasets/structured/proteinnet/proteinnet.py)

*   **Versions**:

    *   **`1.0.0`** (default): Initial release.

*   **Download size**: `Unknown size`

*   **Dataset size**: `Unknown size`

*   **Auto-cached**
    ([documentation](https://www.tensorflow.org/datasets/performances#auto-caching)):
    Unknown

*   **Splits**:

Split | Examples
:---- | -------:

*   **Feature structure**:

```python
FeaturesDict({
    'evolutionary': Tensor(shape=(None, 21), dtype=tf.float32),
    'id': Text(shape=(), dtype=tf.string),
    'length': tf.int32,
    'mask': Tensor(shape=(None,), dtype=tf.bool),
    'primary': Sequence(ClassLabel(shape=(), dtype=tf.int64, num_classes=20)),
    'tertiary': Tensor(shape=(None, 3), dtype=tf.float32),
})
```

*   **Feature documentation**:

Feature      | Class                | Shape      | Dtype      | Description
:----------- | :------------------- | :--------- | :--------- | :----------
             | FeaturesDict         |            |            |
evolutionary | Tensor               | (None, 21) | tf.float32 |
id           | Text                 |            | tf.string  |
length       | Tensor               |            | tf.int32   |
mask         | Tensor               | (None,)    | tf.bool    |
primary      | Sequence(ClassLabel) | (None,)    | tf.int64   |
tertiary     | Tensor               | (None, 3)  | tf.float32 |

*   **Supervised keys** (See
    [`as_supervised` doc](https://www.tensorflow.org/datasets/api_docs/python/tfds/load#args)):
    `('primary', 'tertiary')`

*   **Figure**
    ([tfds.show_examples](https://www.tensorflow.org/datasets/api_docs/python/tfds/visualization/show_examples)):
    Not supported.

*   **Citation**:

```
@article{ProteinNet19,
title = {{ProteinNet}: a standardized data set for machine learning of protein structure},
author = {AlQuraishi, Mohammed},
journal = {BMC bioinformatics},
volume = {20},
number = {1},
pages = {1--10},
year = {2019},
publisher = {BioMed Central}
}
```


## protein_net/casp7 (default config)

*   **Examples**
    ([tfds.as_dataframe](https://www.tensorflow.org/datasets/api_docs/python/tfds/as_dataframe)):

<!-- mdformat off(HTML should not be auto-formatted) -->

{% framebox %}

<button id="displaydataframe">Display examples...</button>
<div id="dataframecontent" style="overflow-x:auto"></div>
<script>
const url = "https://storage.googleapis.com/tfds-data/visualization/dataframe/protein_net-casp7-1.0.0.html";
const dataButton = document.getElementById('displaydataframe');
dataButton.addEventListener('click', async () => {
  // Disable the button after clicking (dataframe loaded only once).
  dataButton.disabled = true;

  const contentPane = document.getElementById('dataframecontent');
  try {
    const response = await fetch(url);
    // Error response codes don't throw an error, so force an error to show
    // the error message.
    if (!response.ok) throw Error(response.statusText);

    const data = await response.text();
    contentPane.innerHTML = data;
  } catch (e) {
    contentPane.innerHTML =
        'Error loading examples. If the error persist, please open '
        + 'a new issue.';
  }
});
</script>

{% endframebox %}

<!-- mdformat on -->

## protein_net/casp8

*   **Examples**
    ([tfds.as_dataframe](https://www.tensorflow.org/datasets/api_docs/python/tfds/as_dataframe)):

<!-- mdformat off(HTML should not be auto-formatted) -->

{% framebox %}

<button id="displaydataframe">Display examples...</button>
<div id="dataframecontent" style="overflow-x:auto"></div>
<script>
const url = "https://storage.googleapis.com/tfds-data/visualization/dataframe/protein_net-casp8-1.0.0.html";
const dataButton = document.getElementById('displaydataframe');
dataButton.addEventListener('click', async () => {
  // Disable the button after clicking (dataframe loaded only once).
  dataButton.disabled = true;

  const contentPane = document.getElementById('dataframecontent');
  try {
    const response = await fetch(url);
    // Error response codes don't throw an error, so force an error to show
    // the error message.
    if (!response.ok) throw Error(response.statusText);

    const data = await response.text();
    contentPane.innerHTML = data;
  } catch (e) {
    contentPane.innerHTML =
        'Error loading examples. If the error persist, please open '
        + 'a new issue.';
  }
});
</script>

{% endframebox %}

<!-- mdformat on -->

## protein_net/casp9

*   **Examples**
    ([tfds.as_dataframe](https://www.tensorflow.org/datasets/api_docs/python/tfds/as_dataframe)):

<!-- mdformat off(HTML should not be auto-formatted) -->

{% framebox %}

<button id="displaydataframe">Display examples...</button>
<div id="dataframecontent" style="overflow-x:auto"></div>
<script>
const url = "https://storage.googleapis.com/tfds-data/visualization/dataframe/protein_net-casp9-1.0.0.html";
const dataButton = document.getElementById('displaydataframe');
dataButton.addEventListener('click', async () => {
  // Disable the button after clicking (dataframe loaded only once).
  dataButton.disabled = true;

  const contentPane = document.getElementById('dataframecontent');
  try {
    const response = await fetch(url);
    // Error response codes don't throw an error, so force an error to show
    // the error message.
    if (!response.ok) throw Error(response.statusText);

    const data = await response.text();
    contentPane.innerHTML = data;
  } catch (e) {
    contentPane.innerHTML =
        'Error loading examples. If the error persist, please open '
        + 'a new issue.';
  }
});
</script>

{% endframebox %}

<!-- mdformat on -->

## protein_net/casp10

*   **Examples**
    ([tfds.as_dataframe](https://www.tensorflow.org/datasets/api_docs/python/tfds/as_dataframe)):

<!-- mdformat off(HTML should not be auto-formatted) -->

{% framebox %}

<button id="displaydataframe">Display examples...</button>
<div id="dataframecontent" style="overflow-x:auto"></div>
<script>
const url = "https://storage.googleapis.com/tfds-data/visualization/dataframe/protein_net-casp10-1.0.0.html";
const dataButton = document.getElementById('displaydataframe');
dataButton.addEventListener('click', async () => {
  // Disable the button after clicking (dataframe loaded only once).
  dataButton.disabled = true;

  const contentPane = document.getElementById('dataframecontent');
  try {
    const response = await fetch(url);
    // Error response codes don't throw an error, so force an error to show
    // the error message.
    if (!response.ok) throw Error(response.statusText);

    const data = await response.text();
    contentPane.innerHTML = data;
  } catch (e) {
    contentPane.innerHTML =
        'Error loading examples. If the error persist, please open '
        + 'a new issue.';
  }
});
</script>

{% endframebox %}

<!-- mdformat on -->

## protein_net/casp11

*   **Examples**
    ([tfds.as_dataframe](https://www.tensorflow.org/datasets/api_docs/python/tfds/as_dataframe)):

<!-- mdformat off(HTML should not be auto-formatted) -->

{% framebox %}

<button id="displaydataframe">Display examples...</button>
<div id="dataframecontent" style="overflow-x:auto"></div>
<script>
const url = "https://storage.googleapis.com/tfds-data/visualization/dataframe/protein_net-casp11-1.0.0.html";
const dataButton = document.getElementById('displaydataframe');
dataButton.addEventListener('click', async () => {
  // Disable the button after clicking (dataframe loaded only once).
  dataButton.disabled = true;

  const contentPane = document.getElementById('dataframecontent');
  try {
    const response = await fetch(url);
    // Error response codes don't throw an error, so force an error to show
    // the error message.
    if (!response.ok) throw Error(response.statusText);

    const data = await response.text();
    contentPane.innerHTML = data;
  } catch (e) {
    contentPane.innerHTML =
        'Error loading examples. If the error persist, please open '
        + 'a new issue.';
  }
});
</script>

{% endframebox %}

<!-- mdformat on -->

## protein_net/casp12

*   **Examples**
    ([tfds.as_dataframe](https://www.tensorflow.org/datasets/api_docs/python/tfds/as_dataframe)):

<!-- mdformat off(HTML should not be auto-formatted) -->

{% framebox %}

<button id="displaydataframe">Display examples...</button>
<div id="dataframecontent" style="overflow-x:auto"></div>
<script>
const url = "https://storage.googleapis.com/tfds-data/visualization/dataframe/protein_net-casp12-1.0.0.html";
const dataButton = document.getElementById('displaydataframe');
dataButton.addEventListener('click', async () => {
  // Disable the button after clicking (dataframe loaded only once).
  dataButton.disabled = true;

  const contentPane = document.getElementById('dataframecontent');
  try {
    const response = await fetch(url);
    // Error response codes don't throw an error, so force an error to show
    // the error message.
    if (!response.ok) throw Error(response.statusText);

    const data = await response.text();
    contentPane.innerHTML = data;
  } catch (e) {
    contentPane.innerHTML =
        'Error loading examples. If the error persist, please open '
        + 'a new issue.';
  }
});
</script>

{% endframebox %}

<!-- mdformat on -->