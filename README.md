# ML Projects

Hey there!

This repository contains simple ML/DL experiments I make in my time of exploring Pytorch and other ML Libraries.

A project I hope to make some day in this regard if it's possible (probably) is a tool that takes in code and spits out an animation of it, like https://pythontutor.com/visualize.html#mode=edit , and suggestions on possible improvements. 

For instance, take the famous Two-Sum problem from leetcode: https://leetcode.com/problems/two-sum/ . Input an image of a quadratic time solution, and it would output the animation, as well as suggestions/hints on making it O(n), or more efficient. I believe this tool could be used in classrooms, students, and teachers. Needless to say, it wouldn't directly provide a solution — that would defeat the purpose of learning. But, by nudging the person towards the correct solution, it could truly help solidify foundations, challenge perspectives, and open the boundaries for learning.

Here is a recent project I completed:
# Molecule Toxicity Detection:
This project aims to fine-tune and optimize a method to generate accurate predictions for the toxicity of molecules in the Tox21 dataset. Although we spent the majority of our time using data manipulation for data imbalances, fine-tuning, and trying different methods and models, given a week, we were unable to find a way to overcome the hurdle of a 80% validation score. Nevertheless, our test results were very high, most likely due to the preprocessing step that accounts for the data imbalances.

## Contributions
Long Nguyen, Ming-Shih Wang: data preprocessing, model fine-tuning, different-model searching, report, training and testing

## Preprocess The Data

### Balancing Each Property Individually:
The data was extremely imbalanced — the 0:1 proportion was close to 0.95% for some, and so we focused much of our time on accounting for this, as this was a likely contributor to the overfitting of our model. The key is to resample in a way that improves the representation of underrepresented labels without significantly distorting the distribution of other labels. Here is our approach. 

<img width="1440" alt="Screen Shot 2023-12-25 at 3 34 55 PM" src="https://github.com/MingCWang/molecular-toxicity-prediction/assets/73949957/5273ee2d-3ecb-4833-9186-b68316120ae3">

Given this is a multi-label classification problem, the distribution of the 12 labels in each data entry is dependent on each other; Which means that simply replicating the minority class would not produce a balanced distribution. Therefore, through trial and error, we found the best combination of data to replicate, which produced fairly balanced data as shown in the graphs below.

*Before oversampling* 	<---->   *After oversampling*

<img width="260" alt="Screen Shot 2023-12-25 at 3 35 07 PM" src="https://github.com/MingCWang/molecular-toxicity-prediction/assets/73949957/7d939dba-a1d7-4b31-94ba-268e2f874dd4">

<img width="250" alt="Screen Shot 2023-12-25 at 3 35 02 PM" src="https://github.com/MingCWang/molecular-toxicity-prediction/assets/73949957/f9f32cab-8a56-4b6d-ad4a-188d161879f7">


### Integrating SMILES and Molecular Data
Another issue we thought was the cause of overfitting, in addition to the “12-task” imbalance was the lack of data. To account for this, we tried adding more features using the rdkit to utilize SMILES and Molecular data. 

<img width="1440" alt="Screen Shot 2023-12-25 at 3 35 15 PM" src="https://github.com/MingCWang/molecular-toxicity-prediction/assets/73949957/8a66d3ed-875a-4eeb-a911-975fe7e6cda1">
<img width="1440" alt="Screen Shot 2023-12-25 at 3 35 23 PM" src="https://github.com/MingCWang/molecular-toxicity-prediction/assets/73949957/4ed0fb7c-b710-40df-8be7-536904cc8105">

By using RDKit and MolVS libraries, the code standardizes molecules from their SMILES representation and extracts their parent structure. Then, it generates Morgan fingerprints, a form of circular fingerprint that encapsulates the molecular structure. These fingerprints are integrated into our dataset by expanding them to match the graph representation of each molecule, thereby augmenting the original node features. This enrichment with detailed chemical information aims to provide a more comprehensive dataset, potentially enhancing the predictive accuracy of our models in tasks such as molecular property prediction. However, this caused more overfitting, quickly going from 68% to 84%, to 93% within 3 epochs for the training ROC-AUC score — and leaving the validation score still to be around 75%.

### Weighted Loss Function:
To account for data imbalance, we also calculated the weights in each label class respectively. 
The original loss calculation simply converts the two-dimensional toxicity label dataset into a huge single-dimension array while excluding all the nan values. This would not work if class weights were applied, we were experiencing dimension mismatches because of the missing values. To account for the dimension difference, because the labels are in a binary format that represents the absence of the property associated with the label, I replaced nan values with 0. 
After class weights calculation, I applied them to a custom loss calculation function using the BCEWithLogitsLoss()function that works better with binary values.

<img width="1440" alt="Screen Shot 2023-12-25 at 3 35 29 PM" src="https://github.com/MingCWang/molecular-toxicity-prediction/assets/73949957/7d7aa4c6-f382-4708-bf2b-77b4bd60de0b">
<img width="1440" alt="Screen Shot 2023-12-25 at 3 35 36 PM" src="https://github.com/MingCWang/molecular-toxicity-prediction/assets/73949957/c1ebadac-d85b-4804-b4ed-5311c5f90224">

## Model Optimization and Fine-Tuning:
Needless to say, our optimizations did not stop in preprocessing, as we implemented a random search for tuning combinations of layers and hyperparameters. 

- Models: GraphConv, GATConv, Regular Neural Network, GNN, GCNConv, BatchNorm, ReLu
- Hyperparameters: dropout_rate, fingerprint_dim, num_classes, hidden_classes, num_rdkit_features

However, we continued to hit a plateau of around 78.98% with these models and combinations of layers and hyperparameter tuning. 

<img width="1440" alt="Screen Shot 2023-12-25 at 3 35 46 PM" src="https://github.com/MingCWang/molecular-toxicity-prediction/assets/73949957/984ac07a-a137-42b4-abba-a21934feaae7">

In the end, we used the model shown above, with a combination of Graph Convolutional Layers, Batch 
Normalizations, Linears, and layers that utilize the fingerprint data. The most optimal hyperparameters were found to be: 

*hidden_channels = 32, dropout_rate = 0.6, fingerprint_dim = 8192, num_node_features = 9*

## Conclusion and Takeaways
All in all, this project was much harder than expected. Admittedly, if there was more time, we would spend more of it understanding the fundamentals of the complex data, as all the efforts of preprocessing with the imbalances, data concatenation, smiles, and molecular data, and all the model and hyperparameter optimization only resulted in a negligible increase in AUC-ROC score. Ultimately, we learned that the APIs of Pytorch, Keras, and scikit-learn make it accessible and quite easy to implement machine learning models and techniques, truly understanding the data, the intricacies of how models work is crucial in achieving a high success rate.
