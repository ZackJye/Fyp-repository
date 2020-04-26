# Fyp-repository

FYP repository 

Dataset of this research can be obtain from ChaLearn First Impressions Challenge @ECCV2016
The link: http://chalearnlap.cvc.uab.es/dataset/20/description/

The dataset contains the video data that categorised as training data, validation data and testing data, followed by the groundtruth for each of data.
Big 5 personality trait will be adopted which human personality can be divided into 5 dimension include
- Extroversion
- Agreeableness
- Neuroticism
- Conscientiouness
- Openness to experience

Abstract
Here, an Audio-LSTM model for personality trait recognition using video data is developed. A selective sampling (First Impression rule) applied to the dataset to generalize the model better. Beside, this research also investigate the relationship between the information exposure and the number of modalities. Audio and visual features will be extracted from the video data. NO feature engineering or visual analysis performed on the visual data.

## Implementation
The model architecture is shown in model.py.
1. Download the repository.
2. Download the dataset from the official web.
3. Execute the Demo.ipynb to with different video to get personality score. (required trained model)
can request the trained model from zijye98@1utar.my

Excel file
- audio_training_15.csv contains 26 audio features for each of the video in the training dataset(also the landmarks)
- audio_validation.csv contains 26 audio features for each of the video in the validation dataset(also the landmarks)



train folder contain useful utilities function on training and evaluating the model. (further modification required according to the situation)
- Preprocess_video.ipynb ->for extract audio features into excel and video into images
- Audio-LSTM.ipynb -> to rebuild to whole model


