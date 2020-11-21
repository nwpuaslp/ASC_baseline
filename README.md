# IEEE SLT 2021 Alpha-mini Speech Challenge(ASC) Baseline Systems
## Introduction
This is the page for official ASC challenge baseline systems. There are three projects:

+ **SSL** is the baseline system for the sound source localization (SSL) track, described in the challenge introduction paper.
+ **DeepKWS** is the DeepKWS recipe for the keyword spotting (KWS) track, described in the challenge introduction paper.
+ **Keyword-filler** is the keyword-filler Kaldi recipe for the keyword spotting (KWS) track, described in the challenge introduction paper.

## Notice
+ Models in DeepKWS and Keyword-filler are zipped. Please unzip and run the script to get the results.
+ Keyword-filler is based on kaldi, so please create a soft link to wsj/s5/utils and wsj/s5/steps in Keyword-filler/ and change the path.sh to your kaldi path
+ You should put KWS-Dev.zip and SSL-Dev.zip in to the Keyword-filler/data/source/original and then run the script
+ Please view the requirements in requirement.txt of each project. 

## Paper
Please kindly cite the following challenge description paper if you use the codes provided here.
Yihui, Zhuoyuan Yao, Weipeng He, Jian Wu, Xiong Wang, Zhanheng Yang, Shimin Zhang, Lei Xie, Dongyan Huang, Hui Bu, Petr Motlicek and Jean-Marc Odobez, IEEE SLT 2021 Alpha-mini Speech Challenge: Open Datasets, Tracks, Rules and Baselines, IEEE SLT 2021.
The paper can be downloaded from: https://arxiv.org/abs/2011.02198.

### Bibtex:
```
@inproceedings{ASC2021,
    title={IEEE SLT 2021 Alpha-mini Speech Challenge: Open Datasets, Tracks, Rules and Baselines},
    author={Fu, Yihui and Yao, Zhuoyuan and He, Weipeng and Wu, Jian and Wang, Xiong and Yang, Zhanheng and Zhang, Shimin and Xie, Lei and Huang, Dongyan and Bu, Hui and Motlicek, Petr and Odobez, Jean-Marc},
    booktitle = {{IEEE SLT 2021}},
    address = {Shenzhen, China},
    year = {2021},
    month = January,
  }
```
Please visit the challenge official website for more details. 
<http://asc.ubtrobot.com/ >
