# Spot-the-diff
Harsh Jhamtani, Taylor Berg-Kirkpatrick. Learning to Describe Differences Between Pairs of Similar Images. EMNLP 2018 </br>
Link: https://arxiv.org/pdf/1808.10584.pdf

## Dataset
- v0.1 of dataset is present in data/.  </br>
#### Annotations:
- data/annotations/ contains threee json files representing train,val,test splits
- format of each json file is as follows: each file represents a list. each item in the list is a dictionary consisting of 'img_id' and 'sentences' keys. e.g. </br>
{"img_id": "400", "sentences": ["two of the three people in the front of the picture have moved", "there is a vehicle in the far back that is only in image two"] </br>
#### Images
- data/resized_images/ contains the relevant images. 
- naming convention: <img_id>.png, <img_id>_2.png
- we have also provided the corresponding diff images: <img_id>_diff.jpg
- All images have been resized to 224,224
- Original size images: bit.ly/spot_diff_data

#### Cluster data
- We provide clusters of differing pixels computed under suggested paramter settings and clustering algorithm.
- For more details, check Code/usage.ipynb

## Others
- Clustering code has been added

TODO
- Model Predictions (multi)


## Reference
If you use the data or code, please consider citing

```
@inproceedings{jhamtani2018learning,
  title={Learning to Describe Differences Between Pairs of Similar Images},
  author={Jhamtani, Harsh and Berg-Kirkpatrick, Taylor},
  booktitle={Proceedings of the 2018 Conference on Empirical Methods in Natural Language Processing (EMNLP)},
  year={2018}
}
```
