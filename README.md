# Setup

First, create conda env and activate:

```
conda env create -f conda_env.yml
conda activate genhop
```

If running on cuda 11, install pytorch 2 for cuda 11:

```
pip3 install --upgrade --force-reinstall torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
```

To check, start a terminal with python 3 and check that
```
import torch
torch.cuda.is_available()
```
returns True.

To evaluate answer via NER, it is necessary to install the spacy model

```
python3 -m spacy download en_core_web_sm
```

Additionally, to run models from OpenAI, add the OpenAI API Key by

```
export OPENAI_API_KEY=*api_key*
```

## macOS
To run on macOS, it might be necessary to install no-mkl versions of numpy and pandas.

```
conda install nomkl
```

then

```
conda install numpy pandas
```

followed by

```
conda remove mkl mkl-service
```

# Run

To evaluate all models from the paper, run

```
run_evaluation.sh
```
