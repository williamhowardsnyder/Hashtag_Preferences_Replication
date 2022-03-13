# Hashtag_Preferences_Replication
This repository documents our attempt to replicate the results in the paper "[#HowYouTagTweets: Learning User Hashtagging Preferences via Personalized Topic Attention](https://aclanthology.org/2021.emnlp-main.616/)" paper published in EMNLP 2021. In this paper, the authors propose a model (PTA) that uses a novel Personalized Topic Attention mechanism to predict future user-hashtag engagement. This repository and set of directions is meant to supplement the author's repository. To follow along, you can navigate [here](https://github.com/polyusmart/Personalized-Hashtag-Preferences), download their source code, and copy our files into their main directory.


## Training the Model
When we followed the instructions in the repository linked above, we were met with several missing files and dependencies. To train the model you will first need to set up a conda environemnt the using the `requirements.txt` file we provide. This contains more than the mere `python 3.8` and `pytorch 1.7.1` that the authors say you need. For our experiments, we skipped the data preprocess steps described in the author's repository and went straight to the **Training and Testing** section.

The authors list the following command for this section
> For training and testing, turn to the main directory, run:
>```bash
>python predict.py
>```

However, there is no file named `predict.py`. Instead you should use 
```bash
python prediction.py
```

This trains the model and creates a output file `pre.txt` and `test2.dat` in the `Records` folder that contains the predictions for each of the user-hashtag pairs in the test set.

## Evaluation
The evaluation of the predictions involves combining the predictions in `pre.txt` and the ground truth labels in `test2.dat`. To do this you must

```bash
python sortByPredict.py
```

Again this is different from the command that the authors provide in their repository. The output will be in the `sortTest.dat` file located in the `Records` directory. This file contains the sorted predictions that allows for easy evaluation with [RankLib](https://sourceforge.net/p/lemur/wiki/RankLib/) software package (you must download RankLib youreslf from the link). After installing RankLib, use the following command

```bash
java -jar RankLib.jar -test sortTest.dat -metric2T <metric> -idv <file>
```

where `<metric>` is the metric used to evaluate on the test data (e.g., <mark>MAP</mark>, <mark>P@5</mark>, <mark>nDCG@5</mark>) and `<file>` is the output file to print model performance on individual ranked lists to.


## Accuracy Results

When we train and predict using the author's model, we get an 0.2553 MAP, 0.0975 P@5, and 0.2644 nDCG@5. These results are significantly lower than those that the authors found and published in their paper. Below is a table summarizing the comparisons between the author's model, and ours.
<center>
<table class="tg">
<tbody>
   <tr>
    <td></td>
    <td>MAP</td>
    <td>P@5</td>
    <td>nDCG@5</td>
  </tr>
   <tr>
    <td>LSTM-Att (SOTA)</td>
    <td>0.2765</td>
    <td>0.1043</td>
    <td>0.2873</td>
  </tr>
  <tr>
    <td>Our Model</td>
    <td>0.2553</td>
    <td>0.0975</td>
    <td>0.2644</td>
  </tr>
  <tr>
    <td>Author's Model</td>
    <td>0.3114</td>
    <td>0.1148</td>
    <td>0.3257</td>
  </tr>
</tbody>
</table>
</center>
As you can see, the results that we found were lower than the SOTA model LSTM-Att. So, we failed to replicate the main claim of the paper, which was that the Personalized Topic Attention model performs better than state of the art.

## Varying User and Hashtag Contexts
In the paper, the authors also claim that (i) PTA outperforms LSTM-Att when user history is sparse, and (ii) PTA significantly outperforms LSTM-Att when hashtag context is large. The authors did not provide code for these experiments so we provide our own scripts, `pred_rich_info.py` and `context_experiments.py`, to test these claims.

To run these scripts, follow the **Traing and Testing Section** above to get the best model. Then, run

```bash
python pred_rich_info.py
```

This uses the best model from training to predict on the train set. It outputs the predictions to `model_outputs.txt` in the `Records` directory. Where this script differs from `prediction.py` is that it outputs the user, hashtag and ground truth label (i.e., 0/1) in a tab separated line with each prediction. After generating the augmented precition file run

```bash
python context_experiments.py
```
This script uses the `model_outputs.txt` file, computes the P@5 for the various user and hashtag contexts, and creates two bar plots corresponding to those precision values.

'<img src="https://raw.githubusercontent.com/Yb-Z/images/main/20211017144829.png" width="380"/>'

## Conclusion

Overall, we failed to reproduce the main claims of this paper. We would recommend that the authors update their published results, or provide more precise instructions on how to replicate the results that they achieved.


## Citation

```
@inproceedings{zhang-etal-2021-howyoutagtweets,
    title = "{\#}{H}ow{Y}ou{T}ag{T}weets: Learning User Hashtagging Preferences via Personalized Topic Attention",
    author = "Zhang, Yuji  and
      Zhang, Yubo  and
      Xu, Chunpu  and
      Li, Jing  and
      Jiang, Ziyan  and
      Peng, Baolin",
    booktitle = "Proceedings of the 2021 Conference on Empirical Methods in Natural Language Processing",
    month = nov,
    year = "2021",
    address = "Online and Punta Cana, Dominican Republic",
    publisher = "Association for Computational Linguistics",
    url = "https://aclanthology.org/2021.emnlp-main.616",
    pages = "7811--7820",
    abstract = "Millions of hashtags are created on social media every day to cross-refer messages concerning similar topics. To help people find the topics they want to discuss, this paper characterizes a user{'}s hashtagging preferences via predicting how likely they will post with a hashtag. It is hypothesized that one{'}s interests in a hashtag are related with what they said before (user history) and the existing posts present the hashtag (hashtag contexts). These factors are married in the deep semantic space built with a pre-trained BERT and a neural topic model via multitask learning. In this way, user interests learned from the past can be customized to match future hashtags, which is beyond the capability of existing methods assuming unchanged hashtag semantics. Furthermore, we propose a novel personalized topic attention to capture salient contents to personalize hashtag contexts. Experiments on a large-scale Twitter dataset show that our model significantly outperforms the state-of-the-art recommendation approach without exploiting latent topics.",
}
