# CSE 472 — Machine Learning Sessional

Coursework, experiments, and a final project for Machine Learning Sessional. This repository covers data preprocessing, classification, ensemble learning, neural networks, dimensionality reduction, expectation-maximization, and Bangla medical question answering with a large language model.

## Contents

| Directory | Topics | Main notebook |
| --- | --- | --- |
| [Assignment 1](Assignment%201/) | Exploratory data analysis, feature encoding, scaling, and logistic regression on employee attrition data | [Notebook](Assignment%201/1905074/1905074.ipynb) |
| [Assignment 2](Assignment%202/) | Logistic regression implementation, bagging, majority voting, stacking, and classification metrics | [Notebook](Assignment%202/1905074/1905074.ipynb) |
| [Assignment 3](Assignment%203/) | Fashion-MNIST classification with a custom neural network, batch normalization, dropout, and Adam | [Notebook](Assignment%203/1905074/1905074.ipynb) |
| [Assignment 4](Assingment%204/) | PCA, UMAP, t-SNE, and expectation-maximization for a two-component Poisson mixture | [Notebook](Assingment%204/1905074.ipynb) |
| [Project](Project/) | Llama 3.1 LoRA fine-tuning and retrieval-augmented generation for Bangla medical Q&A | [Notebook](Project/doctor_qa_dataset_fine_tuning_and_RAG_using_Llama_3.ipynb) |

Assignment directories also contain supporting materials such as problem statements, datasets, reports, and code. The fourth assignment's directory is named `Assingment 4` in the repository.

## Final project: Bangla medical Q&A

The project explores answering medical questions in Bengali using a Llama 3.1 8B checkpoint and retrieved examples from the Bangla Doctor Q&A dataset. It prepares conversational training data, configures 4-bit LoRA fine-tuning with Unsloth, and uses Bangla BERT embeddings with FAISS to retrieve relevant question–answer pairs before generation.

The notebook currently loads a previously fine-tuned checkpoint; its training cells are commented out. **No measured answer-accuracy percentage or held-out evaluation is reported.** Saved examples demonstrate retrieval and generation, with answer quality still requiring evaluation.

See the [project README](Project/README.md) for setup, separate inference and training instructions, model settings, accuracy discussion, export options, and known limitations. The medical Q&A system is an educational prototype and is not clinically validated.

## Getting started

For the coursework notebooks, create a Python virtual environment and install the common libraries:

```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install jupyterlab numpy pandas matplotlib seaborn scikit-learn scipy
jupyter lab
```

On Windows, activate the environment with `.venv\Scripts\activate` instead. Select the environment's Python kernel when opening a notebook.

Some assignments need additional packages:

```bash
# Assignment 3: dataset loading and tensor/data-loader utilities
python -m pip install datasets torch

# Assignment 4: UMAP visualization
python -m pip install umap-learn
```

These commands provide a starting environment; the repository has no pinned dependency specification. The Llama project has separate GPU and dependency requirements documented in [Project/README.md](Project/README.md#setup).

## Running the notebooks

1. Open the notebook for the assignment or project you want to explore.
2. Check its imports and dataset-loading cells before execution. Run coursework notebooks with their containing directory as the working directory when they use relative data paths.
3. Update machine-specific dataset paths. For example, the Assignment 4 notebook contains absolute Windows paths that need to match your local files.
4. Run cells in order, checking intermediate outputs before training or evaluation.

The final project downloads model weights and data and requires a CUDA-capable GPU for its current implementation. Follow its documented execution steps before running cells: training statistics reference commented-out training code, and some model-upload cells are active. Upload cells read the `HF_TOKEN` environment variable; configure it securely in the runtime and revoke previously exposed Hugging Face tokens.

## Results and reproducibility

Coursework notebooks contain evaluation code and saved outputs for their individual tasks, including classification accuracy, precision, recall, F1, and other metrics where applicable. Interpret each result with its dataset, split, preprocessing, and model configuration; there is no single accuracy score for the repository.

Saved notebook outputs may reflect previous executions. Dependency versions, random initialization, data availability, and local paths can affect reruns. The final project's [accuracy and evaluation section](Project/README.md#accuracy-and-evaluation) distinguishes recorded evidence from evaluation still to be performed.
