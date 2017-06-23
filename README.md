# Neural Tokenizer

## Motivation
Tokenization, or segmentation, is often the first step in text processing. It is not a trivial problem in such languages as Chinese, Japanese, or Vietnamese. For English, generally speaking, tokenization is not as important as those languages. However, it may not be so in the mobile environment as people often neglect it. Plus, in my opinion, English is a good testbed before we attack other challenging languages. Tradionally Conditional Random Fields have been successfully employed for tokenization, but neural networks can be an alternative. This is a simple, and/but fun task. Probably you can see the results in less than 10 minutes on a single GPU!

# Model Description
Modified CBHG model, which was introduced in [Tacotron: Towards End-to-End Speech Synthesis](https://arxiv.org/abs/1703.10135), is employed. It is a very powerful architecture with a reasonable number of hyperparameters.

## Data
We use the brown corpus which can be obtained from `nltk`. It is not big enough, but publicly available. Besides, we don't have to clean it.

## Requirements
 * NumPy >= 1.11.1
 * TensorFlow = 1.2
 * nltk >= 3.2.1 (You need to download `brown` corpus)
 * tqdm >= 4.14.0

## File description

 * `hyperparams.py` includes all hyper parameters that are needed.
 * `data_load.py` loads data and put them in queues.
 * `modules.py` contains building blocks for the network.
 * `train.py` is for training.
 * `eval.py` is for evaluation.

## Training
  * STEP 0. Make sure you meet the requirements.
  * STEP 1. Adjust hyper parameters in hyperparams.py if necessary.
  * STEP 2. Run `train.py` or download my [pretrained files](https://u42868014.dl.dropboxusercontent.com/u/42868014/neural_tokenizer/logdir.zip).

## Evaluation
  * Run `eval.py`.

## Results
I got an test accuracy of 0.9877 against the model of 15 epochs, or 4,914 global steps. The baseline result is acquired when we assume we didn't touch anything on the untokenized data. Some of the results are shown below. Details are available in the `results` folder.

Final Accuracy = 209086/211699=0.9877
Baseline Accuracy = 166107/211699=0.7846

▌Expected: Likewise the ivory Chinese female figure known as a doctor lady '' provenance Honan<br>
▌Got: Likewise theivory Chinese female figure known as a doctor lady '' provenance Honan<br>

▌Expected: a friend of mine removing her from the curio cabinet for inspection was felled as if by a hammer but he had previously drunk a quantity of applejack
▌Got: a friend of mine removing her from the curiocabinet for inspection was felled as if by a hammer but he had previously drunk a quantity of apple jack

▌Expected: The three Indian brass deities though Ganessa Siva and Krishna are an altogether different cup of tea
▌Got: The three Indian brass deities though Ganess a Siva and Krishna are an altogether different cup of tea

▌Expected: They hail from Travancore a state in the subcontinent where Kali the goddess of death is worshiped
▌Got: They hail from Travan core a state in the subcontinent where Kalit he goddess of deat his worshiped

▌Expected: Have you ever heard of Thuggee
▌Got: Have you ever heard of Thuggee

▌Expected: Oddly enough this is an amulet against housebreakers presented to the mem and me by a local rajah in
▌Got: Oddly enough this is an a mulet against house breakers presented to the memand me by a local rajahin

▌Expected: Inscribed around its base is a charm in Balinese a dialect I take it you don't comprehend
▌Got: Inscribed around its base is a charm in Baline seadialect I take it you don't comprehend

▌Expected: Neither do I but the Tjokorda Agoeng was good enough to translate and I'll do as much for you
▌Got: Neither do I but the Tjokord a Agoeng was good enough to translate and I'll do as much for you

▌Expected: Whosoever violates our rooftree the legend states can expect maximal sorrow
▌Got: Who so ever violate sour roof treethe legend states can expect maximal s orrow

▌Expected: The teeth will rain from his mouth like pebbles his wife will make him cocu with fishmongers and a trolley car will grow in his stomach
▌Got: The teeth will rain from his mouth like pebbles his wife will make him cocu with fish mongers and a trolley car will grow in his stomach

▌Expected: Furthermore and this to me strikes an especially warming note it shall avail the vandals naught to throw away or dispose of their loot
▌Got: Furthermore and this tome strikes an especially warming note it shall avail the vand alsnaught to throw away or dispose of their loot

▌Expected: The cycle of disaster starts the moment they touch any belonging of ours and dogs them unto the fortyfifth generation
▌Got: The cycle of disaster starts the moment they touch any belonging of ours and dogs them un to the fortyfifth generation

▌Expected: Sort of remorseless isn't it
▌Got: Sort of remorseless isn't it

▌Expected: Still there it is
▌Got: Still there it is

▌Expected: Now you no doubt regard the preceding as pap
▌Got: Now you no doubt regard the preceding aspap

▌Expected: In that case listen to what befell another wisenheimer who tangled with our joss
▌Got: In that case listen to what be fell anotherwisen heimer who tangled with our joss

▌Expected: A couple of years back I occupied a Village apartment whose outer staircase contained the type of niche called a coffin turn ''
▌Got: A couple of yearsback I occupied a Village apartment whose outerstair case contained the type of niche called a coffinturn ''

▌Expected: After a while we became aware that the money was disappearing as fast as we replenished it
▌Got: After a while we became aware that the money was disappearing as fast as were plenished it

▌Expected: The more I probed into this young man's activities and character the less savory I found him
▌Got: The more I probed into this young man's activities and character the less savory I found him

▌Expected: His energy was prodigious
▌Got: His energy was prodigious

▌Expected: In short and to borrow an arboreal phrase slash timber
▌Got: In short and toborrow an arboreal phrases lash timber





