# Third-party components

This repository contains application source, configuration, and generated test fixtures. It does not redistribute model weights, source recordings, or evaluation datasets. Installed libraries and downloaded models remain subject to their upstream terms.

## Model references

| Use | Source | Pinned revision |
| --- | --- | --- |
| English ASR | [Systran/faster-whisper-base.en](https://huggingface.co/Systran/faster-whisper-base.en) | `3d3d5dee26484f91867d81cb899cfcf72b96be6c` |
| Passage embeddings | [sentence-transformers/multi-qa-MiniLM-L6-cos-v1](https://huggingface.co/sentence-transformers/multi-qa-MiniLM-L6-cos-v1) | `b207367332321f8e44f96e224ef15bc607f4dbf0` |

The ASR checkpoint is SYSTRAN's CTranslate2 conversion of OpenAI's Whisper base.en model. Its upstream model card identifies the MIT license. The embedding checkpoint is maintained by Sentence Transformers; consult its model repository for its applicable terms. AudioSeek does not claim authorship of either model. File-size and hash metadata in `audioseek/config/` identify the downloaded upstream artifacts.

## Libraries

- [Gradio](https://github.com/gradio-app/gradio)
- [faster-whisper](https://github.com/SYSTRAN/faster-whisper)
- [CTranslate2](https://github.com/OpenNMT/CTranslate2)
- [Sentence Transformers](https://github.com/UKPLab/sentence-transformers)
- [Transformers](https://github.com/huggingface/transformers)
- [PyTorch](https://github.com/pytorch/pytorch)
- [rank-bm25](https://github.com/dorianbrown/rank_bm25)
- [PyAV](https://github.com/PyAV-Org/PyAV)
- [ONNX Runtime](https://github.com/microsoft/onnxruntime)

This list identifies the principal runtime components; it is not a replacement for the licenses distributed with installed packages and their dependencies.
