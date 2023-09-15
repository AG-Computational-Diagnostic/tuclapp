
# # Some useful docker commands (can be ignored)
# # sudo usermod -aG docker <user_name>
# # systemctl start docker
# docker build -t tuclapp .
# docker images
# docker run --rm -it -p 80:3838 tuclapp
# docker ps
# docker exec -it <container id> /bin/bash

FROM tensorflow/tensorflow:2.13.0-gpu
ENV R_VERSION=4.2.1-3.2004.0 
# Check for new SHINY_VERSION: https://posit.co/download/shiny-server/
ENV SHINY_VERSION=1.5.20.1002
ENV OPS_VERSION=4.1.0 

COPY ./app /app/
COPY renv.lock /
COPY requirements-docker.txt /
COPY ./shiny-server.conf /etc/shiny-server/

RUN apt-get update -qq
RUN apt-get install -y wget

# Install R
RUN curl https://cloud.r-project.org/bin/linux/ubuntu/marutter_pubkey.asc | tee -a /etc/apt/trusted.gpg.d/cran_ubuntu_key.asc
RUN add-apt-repository "deb https://cloud.r-project.org/bin/linux/ubuntu $(lsb_release -cs)-cran40/"
RUN apt-get install -y r-base-core=${R_VERSION}
RUN apt-get install -y r-base-dev=${R_VERSION}

# Install Shiny Server
RUN apt-get install -y gdebi-core
RUN curl https://download3.rstudio.org/ubuntu-18.04/x86_64/shiny-server-${SHINY_VERSION}-amd64.deb -o shiny-server.deb
RUN gdebi --non-interactive shiny-server.deb
RUN chown shiny:shiny /var/lib/shiny-server # see https://github.com/rocker-org/shiny/issues/49

# Install additional javascripts
RUN curl https://d3js.org/d3.v7.min.js -o /app/www/d3.min.js
RUN curl https://raw.githubusercontent.com/openseadragon/svg-overlay/master/openseadragon-svg-overlay.js -o /app/www/openseadragon-svg-overlay.js
RUN wget https://github.com/openseadragon/openseadragon/releases/download/v${OPS_VERSION}/openseadragon-bin-${OPS_VERSION}.zip -O /app/www/openseadragon.zip
RUN unzip /app/www/openseadragon.zip -d /app/www
RUN mv /app/www/openseadragon-bin-${OPS_VERSION} /app/www/openseadragon-bin
RUN rm /app/www/openseadragon.zip
RUN wget https://github.com/peterthomet/openseadragon-flat-toolbar-icons/archive/refs/heads/master.zip -O /app/www/osd-images.zip
RUN unzip /app/www/osd-images.zip -d /app/www
RUN rm /app/www/osd-images.zip

# Install openslide
RUN apt-get install -y libopenslide0 

# Install libraries required for R packages
RUN	apt-get install -y libtiff-dev libx11-dev # For imager
RUN apt-get install -y libgl1 # For opencv

# Set up R
RUN R -e "install.packages(c('renv'), repo='https://cloud.r-project.org/', clean=TRUE)"
RUN R -e "renv::restore(library='/usr/local/lib/R/site-library', lockfile='/renv.lock')"

# Set up python
RUN apt-get install -y libpython3.10 python3-pip
RUN pip install -r /requirements-docker.txt

USER shiny:shiny
CMD shiny-server
