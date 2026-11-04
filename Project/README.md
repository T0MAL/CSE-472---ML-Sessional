# Bangla Medical Question Answering with Llama 3.1 and RAG

A notebook-based machine learning project exploring **Bangla medical question answering** through LoRA fine-tuning and retrieval-augmented generation (RAG). It combines a Llama 3.1 8B model with similar question–answer pairs retrieved from a Bangla medical dataset.

[Open the notebook](doctor_qa_dataset_fine_tuning_and_RAG_using_Llama_3.ipynb)

## Overview

The project brings together two approaches:

- **Fine-tuning:** prepare Bangla medical conversations and configure supervised fine-tuning with Unsloth and LoRA on a 4-bit model.
- **Retrieval:** embed dataset questions, search them with FAISS, and add the matching questions and answers to the model's conversation before generation.

The notebook currently loads an existing fine-tuned checkpoint. Its training cells are commented out, while retrieval and inference cells are active. It also contains model export examples for merged weights, LoRA adapters, and GGUF.

## Workflow

```text
Bangla medical Q&A dataset
          |
          v
Parse questions and answers
          |
          +--> Llama-formatted conversations --> LoRA fine-tuning (optional)
          |
          +--> Question embeddings --> FAISS index
                                           ^
                                           |
                                    User question
                                           |
                                           v
                               Retrieve top 2 Q&A pairs
                                           |
                                           v
                           System prompt + retrieved pairs
                                 + current question
                                           |
                                           v
                              Llama 3.1 --> Bangla answer
```

Retrieval uses **questions** as the indexed documents. Their corresponding answers are inserted as assistant messages, giving the model retrieved conversation examples rather than passages from external medical references.

## Dataset and models

| Component | Used in the notebook |
| --- | --- |
| Dataset | `shetumohanto/doctor_qa_bangla` |
| Dataset size | 5,135 training examples in the saved execution output |
| Base model referenced | `unsloth/Meta-Llama-3.1-8B-bnb-4bit` |
| Checkpoint loaded by default | `tomal001/Meta-Llama-3.1-8B-bnb-4bit_Fine-tuned_on_doctor_qa_bangla` |
| Embedding model | `sagorsarker/bangla-bert-base` via Sentence Transformers |
| Vector index | FAISS `IndexFlatL2` |
| Retrieval count | 2 Q&A pairs for generation; 3 results by default in the search helper |

The dataset's original `text` field is split at `[/INST]` to extract the question and answer. The transformation produces `user_text`, `assistant_text`, a JSON-serialized `conversations` field, and a Llama-formatted `text` field for training.

Sentence Transformers applies mean pooling to the Bangla BERT model, as recorded in the notebook output. The implementation uses L2 distance on the resulting embeddings without explicit normalization.

## Project files

```text
Project/
├── README.md
├── doctor_qa_dataset_fine_tuning_and_RAG_using_Llama_3.ipynb
├── Dataset Loading.ipynb
└── B1_7.pdf
```

The main notebook contains the implementation. Model weights and datasets are downloaded at runtime; the FAISS index is built in memory. Index persistence is present only as a commented example.

## Setup

Use a GPU-enabled Google Colab runtime or a compatible local Jupyter environment with an NVIDIA GPU and CUDA. The saved run used a **Tesla T4 with approximately 15 GB VRAM**. Downloads, merged exports, and long sequences also require substantial RAM and disk space.

The notebook installs Unsloth and then replaces it with the GitHub development version. It separately installs `faiss-gpu`. Other imports require PyTorch, Transformers, Datasets, Sentence Transformers, PEFT, and TRL.

Before the embedding section, ensure Sentence Transformers is installed:

```python
%pip install sentence-transformers
```

If the notebook's `faiss-gpu` installation is unavailable for your Python environment, replace that installation cell with:

```python
%pip install faiss-cpu
```

The code constructs a CPU FAISS index even when the GPU package is installed, so GPU FAISS is unnecessary for this implementation. Llama inference still uses CUDA.

**Reproducibility:** dependencies are not pinned. Saved outputs report Unsloth `2024.12.4`, Transformers `4.46.3`, and PyTorch `2.5.1+cu121`; these describe the recorded run, not a tested installation specification. Package API differences may require adjustments, particularly in the training section.

## Run retrieval and inference

1. Open the main notebook and select a GPU runtime.
2. Run the installation, model-loading, and dataset-preparation cells. The default model identifier loads the existing fine-tuned checkpoint.
3. For inference with that checkpoint, skip the **LoRA adapter creation** cell; new adapters are needed when training. Skip the training section and the final training statistics cell, which references the undefined `trainer_stats` while training is commented out.
4. Run the system prompt, FAISS installation, document-list construction, embedding, and index-building cells.
5. Initialize `messages`, define `get_context_faiss_mbert`, retrieve examples, and append the actual user question.
6. Run the chat-template and generation cells, or the streaming example.
7. Stop before the saving and Hub-upload cells unless you intend to export or publish a model. Some upload calls are active by default.

For a new question, replace the notebook's example retrieval and question-append cells with:

```python
query = "আমার বাচ্চা সারাক্ষণ কান্নাকাটি করছে, কী করব?"

# Start a fresh conversation for each independent question.
messages = [{"role": "system", "content": system_message}]
get_context_faiss_mbert(query, index, messages)
messages.append({"role": "user", "content": query})
```

Then run the generation cell. `get_context_faiss_mbert` modifies `messages` in place and returns `None`; its result does not need to be assigned to `context`.

The normal generation cell decodes the entire conversation. To display only newly generated tokens, replace its final decoding expression with:

```python
answer_tokens = outputs[:, inputs.shape[-1]:]
print(tokenizer.batch_decode(answer_tokens, skip_special_tokens=True)[0])
```

## Fine-tune from the base model

To train a fresh adapter, change `model_name` in the loading cell to `unsloth/Meta-Llama-3.1-8B-bnb-4bit`, then run the LoRA configuration and dataset preparation. Uncomment the trainer setup, response-only masking, and `trainer.train()` cells. Run the final training statistics cell after training completes.

The settings below come from the code, including the commented training configuration:

| Setting | Value |
| --- | --- |
| Maximum sequence length | 10,000 tokens |
| Model loading | 4-bit |
| LoRA rank / alpha | 16 / 16 |
| LoRA dropout | 0 |
| Target modules | `q_proj`, `k_proj`, `v_proj`, `o_proj`, `gate_proj`, `up_proj`, `down_proj` |
| Gradient checkpointing | `unsloth` |
| Training epochs | 1 |
| Per-device batch size | 2 |
| Gradient accumulation | 4 steps |
| Learning rate | `2e-4` |
| Optimizer | `adamw_8bit` |
| Warmup | 5 steps |
| Weight decay | 0.01 |
| Scheduler | Linear |
| Random seed | 3407 |
| Training output directory | `outputs/` |

The optional response-only masking excludes the system and user portions from the training loss. Although the notebook's prose mentions a 60-step demonstration, the actual trainer configuration specifies one epoch and leaves `max_steps` commented out. Reduce sequence length or batch size if GPU memory is insufficient.

## Export

The notebook includes examples for:

- Saving merged 16-bit weights and the tokenizer to `model/`.
- Uploading merged weights to Hugging Face.
- Saving merged 4-bit weights or LoRA adapters through optional branches.
- Exporting GGUF with `q8_0`, `f16`, or `q4_k_m` quantization through optional branches.

The reload example points to `lora_model`, but the active save path exports merged weights to `model`. Save an adapter explicitly before using the adapter reload example, or adjust the loading path and method to match your export.

**Credentials:** upload cells read the `HF_TOKEN` environment variable instead of storing a token in the notebook. Set it securely in the runtime before uploading; for an interactive session, use `import os; from getpass import getpass; os.environ["HF_TOKEN"] = getpass("Hugging Face token: ")`. Do not put a token literal in a cell or print it. Previously embedded tokens must be revoked and replaced in Hugging Face. Set upload destinations to a repository you control.

## Accuracy and evaluation

**Accuracy has not been quantitatively measured in the supplied notebook.** There is no held-out test set, scored answer evaluation, or reported accuracy percentage. The saved examples demonstrate the pipeline, but do not establish how reliably it answers unseen medical questions.

| Evaluation area | Available evidence | What can be concluded |
| --- | --- | --- |
| Retrieval | A saved search retrieves an identical dataset question at rank 1 with an L2 distance of approximately `2.25e-11` | The index can recover that indexed example; this does not measure performance on unseen questions |
| Answer quality | Saved Bengali generation examples, including malformed continuation text | The model generates Bengali text, but correctness and generation reliability are unverified |
| Fine-tuning benefit | Training configuration and a referenced fine-tuned checkpoint | No measured improvement over the base model is reported |
| RAG benefit | Retrieved Q&A pairs are supplied before generation | No scored comparison with generation without retrieval is reported |
| Clinical accuracy | No clinician-scored evaluation is included | No clinical accuracy claim can be made |

FAISS distance measures embedding similarity; it is **not an accuracy percentage or confidence score**. Likewise, a prompt asking for accurate answers does not verify the medical content of a response. The accompanying [presentation](B1_7.pdf) describes Precision@k, response time, and feedback incorporation rate as performance metrics, but supplies no numerical results for them.

### How to measure accuracy

A future evaluation should use a fixed, held-out set of Bangla medical questions whose answers are reviewed by qualified medical professionals. Split the data before training and indexing, and group duplicate or paraphrased questions into the same split to reduce leakage. Keep test questions and their answers out of both training and the retrieval corpus.

- **Retrieval relevance:** label which indexed Q&A pairs are relevant to each test question. Report `Precision@2 = relevant retrieved pairs / 2`, averaged over the test questions. Recall@k can also be reported when the relevant-document set is labeled.
- **Answer correctness:** have reviewers score factual correctness, relevance, completeness, and potentially harmful statements. Define the scoring rubric first. An answer accuracy percentage can then be reported as `100 × answers judged correct / evaluated answers`, alongside the sample size and review method.
- **Fine-tuning and RAG contribution:** evaluate the base model, fine-tuned model, and fine-tuned model with RAG on the same questions under the same generation settings.
- **Language and usability:** assess Bengali fluency, adherence to the requested answer length, malformed continuations, and end-to-end response latency.

Text-overlap metrics such as BLEU or ROUGE may supplement this evaluation, but matching a reference answer's wording does not establish medical correctness. These evaluation steps are proposed work; they are not implemented or scored in the current notebook.

## Results and limitations

Saved outputs demonstrate dataset transformation, nearest-neighbor retrieval, model loading, generation, and merged-model saving. The notebook does **not** report held-out evaluation, retrieval metrics, clinical validation, or a measured comparison between fine-tuning alone and RAG.

- The same dataset supplies training examples and retrieval context; no train/test split is implemented.
- Saved generation includes malformed continuation text, so answer quality and stopping behavior need further evaluation.
- The prompt requests answers within 100 words, but this limit is not enforced in code.
- Generation parameters alone do not establish repeatable answers; explicit sampling and reproducibility controls would be needed.
- Retrieved examples are dataset answers, with no implemented source verification or citation mechanism.
- Saved outputs reflect earlier executions and may differ from the current cell contents.

This is an educational prototype for medical-language modeling. Its generated answers are not clinically validated and should not be used for diagnosis or treatment decisions.

## Resources and attribution

- [Bangla Doctor Q&A dataset](https://huggingface.co/datasets/shetumohanto/doctor_qa_bangla)
- [Fine-tuned checkpoint referenced in the notebook](https://huggingface.co/tomal001/Meta-Llama-3.1-8B-bnb-4bit_Fine-tuned_on_doctor_qa_bangla)
- [Unsloth base checkpoint](https://huggingface.co/unsloth/Meta-Llama-3.1-8B-bnb-4bit)
- [Bangla BERT embedding backbone](https://huggingface.co/sagorsarker/bangla-bert-base)
- [Unsloth](https://github.com/unslothai/unsloth)
- [FAISS](https://github.com/facebookresearch/faiss)

The notebook incorporates Unsloth example material. Consult the respective dataset, model, and library licenses before redistributing or reusing their artifacts.
