# GVQA
Code for the Grounded Visual Question Answering (GVQA) model from the below paper:

[Don't Just Assume; Look and Answer: Overcoming Priors for Visual Question Answering](https://www.cc.gatech.edu/~aagrawal307/vqa-cp/)  
Aishwarya Agrawal, Dhruv Batra, Devi Parikh, Aniruddha Kembhavi  
IEEE Conference on Computer Vision and Pattern Recognition (CVPR), 2018  
https://arxiv.org/abs/1712.00377

## Inference Code ##
The GVQA model consists of the following modules:
- Question Classifier
- Visual Concept Classifier (VCC)
- Answer Cluster Predictor (ACP)
- Concept Extractor (CE)
- Answer Predictor (AP)
- Visual Verifier (VV)

In order to run inference on GVQA, we need to run inference on each of the above modules in a sequential manner so that the predictions from one module could be used as input features to the following modules. 

So, first we run inference on the Question Classifier as:

```
th eval_question_classifier.lua
```

And then we run inference on the VCC module as:

```
th eval_vcc.lua
```

And then we run inference on the ACP module as:

```
th eval_acp.lua
```
And then we run inference on the AP module as:

```
th eval_ap.lua
```

And then we run inference on the VV module as:

```
th eval_vv.lua
```

We then need to combine the predictions of the ap and the vv module as:

```
python combine_ap_and_vv_results.py
```

In order to run the above scripts, please place the processed inputs provided [here](https://computing.ece.vt.edu/~aish/vqacp/code/processed_inputs/) in a directory called `processed_inputs` at `GVQA/` and please place the trained models provided [here](https://computing.ece.vt.edu/~aish/vqacp/code/models/) in a directory called `models/` at `GVQA/`.

The processed inputs contain the output of the CE module (as the CE module is just a simple rule followed by GloVe embedding extraction). The processing scripts to obtain these processed inputs from the raw questions and images will be released soon.

## Training Code ##
Coming Soon.

