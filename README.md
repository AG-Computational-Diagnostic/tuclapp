
# TUCL APP

The TUmor CLassifier (TUCL) app is an [R Shiny](https://www.rstudio.com/products/shiny/) app that can be used to apply a [TensorFlow](https://www.tensorflow.org/) image classifier on digital whole slides and inspect the results. The app was developed within the [DIAMANT – Digitale Bildanalyse und bildgebende Massenspektro-metrie zur Differenzierung von nichtkleinzelligem Lungenkrebs](https://www.gesundheitsforschung-bmbf.de/de/diamant-digitale-bildanalyse-und-bildgebende-massenspektro-metrie-zur-differenzierung-von-10889.php) project.

This README only describes how to build the application using docker and get it started. This README does not cover how to use the app, which should be explained within the app itself. For installing the app without docker (e.g., when developing), see the dockerfile `Dockerfile` for install steps.

## Build

Simply run:

```shell
docker build -t tuclapp .
```

Note that this takes quite some time (especially installing the R packages).

## Start

First check for the build docker image via:

```shell
docker images
```

Then use the following command to run the image locally.

``` shell
docker run -it --rm -p 80:3838 tuclapp
```

If your system has a GPU and your docker instance is set up to use it (see e.g., [here]()) instead ruin the container using following command.

``` shell
docker run -it --gpu all --rm -p 80:3838 tuclapp
```

Now open your webbrowser and open the page `http://localhost/`. This should open the application.

## Developing

### New Models

To check out how to include new models, see the included dummy models (e.g., `app/models/DEBUG/L`) and the python functions `load_full_model` or `load_model_from_weights` in `app/python/tucl-app.py`.

### Torch Models

The code for the app uses TensorFlow for prediction. Hence, the usage of pytorch models is not straightforward. The simplest option is to translate the torch model to a TensorFlow model. However, in the long run an adaption of the python code to use torch as well would be very advisable.
 
### Ussage of R Shiny 
 
While the app is written in R, most of its important functions (e.g., tile and classify slides) are written in python (see `app/python/tucl-app.py`) and the important zoomable slide viewer uses the JavaScript package [OpenSeadragon](https://openseadragon.github.io/]. Hence, it might be a very good idea to also translate the remaing R parts to python (e.g., using [Flask](https://flask.palletsprojects.com/en/2.3.x/)). This might render the app much more lightweight.

### Misssing progress bar

I have not found any way to include the python progress bar into the app (i.e., that shows how many tiles have been classified). However, this would be really useful.

### Batch Size

At the moment the used batch size is always 1, which is clearly not optimal. This should be adopted to used models / available resources.

### Testing

Well, there is no tests at the moment but they would be an extremely good idea.

## License

Licensed under the Apache License, Version 2.0.  
Copyright &copy; 2023, Universitätsklinikum Heidelberg, Katharina Kriegsmann, Mark Kriegsmann, Georg Steinbuß


