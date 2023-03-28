# Crop Recommendation System

This Application recommends the best crop to plant based on Soil and Weather conditions!

This application was created based on a Kaggle dataset that can be accessed by the link below:

[Link](https://www.kaggle.com/datasets/atharvaingle/crop-recommendation-dataset)

To access my Crop Recommendation System you can click [HERE](https://prevelato-crop-recommendation-system.streamlit.app/)

## Crop recommendation system models

All the models that were created and tested to be used on this application are in the [models.ipynb](https://github.com/MPrevelato/Crop_Recommendation_System/blob/main/models.ipynb)

For the requirements you can access the [requirements.txt](https://github.com/MPrevelato/Crop_Recommendation_System/blob/main/requirements.txt)

## Streamlit

The Web App was made on Streamlit, you can see the code on the [app.py](https://github.com/MPrevelato/Crop_Recommendation_System/blob/main/app.py),you can use use it and edit offline on your machine by installing the requirements.txt and by running this code in the Terminal:

```
streamlit run app.py
```

## Docker

It is possible to run the Streamlit App using Docker. So, for this I've uploaded the [Dockerfile](https://github.com/MPrevelato/Crop_Recommendation_System/blob/main/Dockerfile) into this repository. So on the root of this folder you can use this command in the Terminal to build the container image:
```
docker image build -f Dockerfile -t crop_app:latest .
```
Using -f to references the Dockerfile to build the image and -t to tag that image. The " ." in the end indicates that the Dockerfile are in the current directory.

After this you just need to run the container with the Streamlit App:

```
docker container run -p 8501:8501 crop_app:latest
```

The -p flag is used to map the container’s port to the host port, by default Streamlit uses port 8501.

Now you can access your application here: [http://localhost:8501/](http://localhost:8501/)

