# neural_tokenizer

## Requirements
  * numpy >= 1.11.1
  * sugartensor >= 0.0.1.8 (Check [this repository](https://github.com/buriburisuri/sugartensor) for installing sugartensor)
  * nltk >= 3.2.1 (You need to download `brown` corpus)
  * gensim >= 0.13.1 (For creating word2vec models)

## Research Topic
Can we segment untokenized English sentences correctly using neural networks?

## Background
  * In many cases, tokenization is the first step in text processing.
  * In some languages such as Chinese, Japanese, or Vietnamese, tokenization is non-trivial.
  * For English, generally speaking, tokenization is not as important as those languages, but
  * Tokenization can be needed in mobile environments as people often neglect it.
  * And English is a good language to test before we attack other challenging languages.
  * CRFs are known to be good at tokenization.
  * Neural networks, concretely RNN, can be an alternative to CRFs.

## Main Idea
We use bidirectional GRU layers with layer normalization <br/> as
tokenization of a certain time step is dependent on its future as well as its past. <br />
Layer normalization is applied to boost the performance.

## Data
We use the brown corpus which can be obtained from `nltk`. <br />
It is not big enough, but publicly available.<br />
And we don't have to clean it.

## Results
After having seen 11,537 samples for 5 epochs, we got .93 of the tokenization accuracy.<br/> 
Here are some snippets of the test results.

▌Expected: The aimless milling about of what had been a well-trained , well-organized crew struck Alexander with horror .
▌Got: The aim less milling about of what had been awell-trained , well -or ganized crews truck Alex ander with horror .

▌Expected: Adrien Deslonde hastened to Alexander's side .
▌Got: Adrien De slond ehas tened to Alexander 's side .

▌Expected: `` Small violently jerked the weather-royal brace with full intention to carry away the mast .
▌Got: `` Small violently jerked the weat her -roy albracewith full intentionto carry away the mast.

▌Expected: I saw him myself and it was done after consultation with Cromwell .
▌Got: I saw him my self and it was done after consultation with Cromwell .

▌Expected: I swear it , sir '' .
▌Got: I swearit, sir '' .

▌Expected: Then , with disappointment evident upon their faces , they moved to the work .
▌Got: The n , with disappoin tment evidentupon the ir faces , they moved to the work .

▌Expected: Wilson , shackled and snarling , was thrown with the other prisoners and was soon joined by Green , McKee and McKinley .
▌Got: Wilson , shackled and snarling , was thrown with the otherprisoners and was soon joined by Green , McKeeand McKinley .

▌Expected: Not a man on the brig , loyal or villainous , could be unaffected by the sight of seven men involved in the crime of mutiny .
▌Got: Nota man on the brig, loy alor villain ous , could be unaffected by the sight of seven meninvolved in the crimeof mutiny .

▌Expected: Adrien Deslonde hastened to Alexander's side .
▌Got: Adrien De slond ehas tened to Alexander 's side .

▌Expected: `` Small violently jerked the weather-royal brace with full intention to carry away the mast .
▌Got: `` Small violently jerked the weat her -roy albracewith full intentionto carry away the mast.

Final Accuracy = 164078/176689=0.93

## Conclusion
Bidirectional RNNs turns out to be effective in English tokenization.

## Further Study
We can apply this idea to other languages such as Chinese, Japanese, Vietnamese, or Thai.




