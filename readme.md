# A probabilistic stacking–based heterogeneous ensemble classifier for drug–drug interaction classification using textual drug descriptors

This repository contains the experimental infrastructure used in an academic study on ensemble methods for drug–drug interaction (DDI) text-only classification.

The project is organized around configuration files to support reproducible experiments. Embeddings, datasets, models, and experiment settings are defined separately and combined dynamically when the experiments are run.

## Project Structure

```text
.
├── colab/
├── embedding_configuration/
├── experiment_configurations/
└── research_datasets/
```

## `colab`

The `colab` directory contains the files required to run the experiments in Google Colab.

The embedding generation notebook is used to create embeddings from DrugBank text data. The other notebooks are used to run the experiments.

## `embedding_configuration`

The `embedding_configuration` directory contains the configuration required to generate embeddings from DrugBank text data.

The `embedding_config.json` file defines:

- The embedding models.
- The text splitters and their parameters.
- The DrugBank fields used as metadata.
- The text fields used to generate embeddings.
- The embedding batch size.
- Additional model parameters.

The configured embedding models include:

- `BAAI/bge-large-en-v1.5`
- `sentence-transformers/all-mpnet-base-v2`

The text fields used for embedding generation include:

- `merged_text`
- `description`
- `indication`
- `mechanism_of_action`
- `pharmacodynamics`

## `experiment_configurations`

The `experiment_configurations` directory contains the JSON files that define the experiments.

These configuration files specify the components and parameters used in each experiment, including:

- The embedding dataset.
- The research dataset.
- The classification model.
- Model parameters.
- Other experiment-specific settings.

The Colab notebooks read these JSON files and run the experiments dynamically. This allows different embeddings, datasets, and models to be evaluated without changing the notebook implementation.

## `research_datasets`

The `research_datasets` directory contains the configuration and preprocessing settings required to use the research datasets with this project.

The project uses the following datasets:

- DDIMDL
- MDF-SA-DDI

Both datasets are available in the literature. Their data and structure were adapted for use with the experimental infrastructure in this repository.

## Reproducing the Experiments

A typical workflow is:

1. Prepare the required DrugBank text data.
2. Configure the embedding models and text fields in `embedding_configuration/embedding_config.json`.
3. Run the embedding generation notebook in the `colab` directory.
4. Select or create an experiment configuration in `experiment_configurations`.
5. Open the corresponding experiment notebook in the `colab` directory.
6. Run the notebook in Google Colab.

The notebooks use the configuration files to determine which embeddings, datasets, models, and parameters are used for each experiment.

## Notes

- The required input data and generated embeddings may not be included in this repository.
- Paths and environment-specific settings may need to be adjusted before running the notebooks.
- The experiments are designed to run in Google Colab.