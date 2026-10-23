# Validation scope

## Completed locally

- Created a separate Windows/Python 3.12 virtual environment without inheriting the development environment's site-packages.
- Installed CPU PyTorch and all declared application dependencies from package indexes.
- Built the Python distributions, installed the wheel, and checked installed dependency consistency.
- Ran the model-independent regression tests against the installed wheel in Python's isolated mode.
- Started the installed command from a working directory outside the source checkout and model/session storage directory.
- Uploaded generated English speech in an isolated Edge browser, then exercised BM25, SBERT, Hybrid retrieval and audio playback.
- Checked download-only model preparation and reuse of verified local model files.
- Tested interrupted transfers, incorrect sizes and checksums, invalid caches, progress messages, and bounded retries using generated HTTP fixtures.

The screenshots and silent walkthrough show this application running with generated English speech. They are workflow examples, not retrieval-accuracy or latency benchmarks. Model weights and source audio are not distributed with the repository.

## CI configuration

The workflow in `.github/workflows/tests.yml` defines a four-job matrix: Windows and Ubuntu, each with Python 3.12 and 3.13. Every job builds an sdist and wheel, installs the wheel, checks dependencies, runs tests with `python -I`, and checks launcher help. The normal CI jobs do not download model weights or measure ASR accuracy.

GitHub-hosted jobs have not run for this local release directory. Push the repository and inspect its Actions tab before claiming those platforms passed. Linux, Python 3.13 and macOS have not been validated locally.

## Reproduce

After installing the dependencies and application:

```bash
python -m pip check
python -m unittest discover -s tests -v
python -m audioseek --help
python -m audioseek --download-models
```

To reproduce the installed-package check, build distributions with `python -m build`, install the generated wheel into a fresh environment, then run `python -I -m unittest discover -s tests -v` from a source checkout containing the tests. Install CPU PyTorch first as described in the README.
