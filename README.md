# Corsica earthquake school tutorials

This repository contains the tutorial files for the [3rd Corsica earthquake school](https://earthquakes3.sciencesconf.org/).

## Installation instructions

To ensure a smooth tutorial experience regardless of OS and software environment, we have bundled all of the tutorial files into a [Docker](https://www.docker.com/) image. 
After installing Docker, clone this repository with `git clone https://github.com/martijnende/corsica2021` and navigate to `docker` within the repository directory. 
There, run `docker-compose up` to launch the Docker image.
This will automatically launch a Jupyter notebook server, which you can access by navigating your browser to `https://localhost:9000/?token=corsica`.
