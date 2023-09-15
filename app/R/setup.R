
# Python code ------------------------------------------------------------------

reticulate::use_python('/usr/bin/python3', required = TRUE)
openslide <- reticulate::import("openslide") # Needed to verify if slide can be loaded
dz <- reticulate::import("openslide.deepzoom") # Needed for zoomable window
tf <- reticulate::import("tensorflow") # Needed to get softmax from logits
np <- reticulate::import("numpy") # Needed because of cimg and array reshape
reticulate::source_python("/app/python/tucl-app.py", convert = FALSE)

# Allowed uploads --------------------------------------------------------------

# Data types from https://openslide.org/
openslide_ext <- c(
  "svs", "tif", # Aperio (.tif also Trestle and Generic tiled TIFF)
  "vms", "vmu", "ndpi", # Hamamatsu 
  "scn", # Leica
  "mrxs",  # MIRAX
  "tiff", # Philips
  "svslide", # Sakura 
  "bif" # Ventana (also has .tif)
) 
pillow_ext <- c("png", "bmp", "jpeg", "jpg") # -> Use PILLOW
allowed_exts <- c(pillow_ext, openslide_ext) 

# Defaults ---------------------------------------------------------------------

class_mask_max_size = 2048L
batch_size <- 1L # TODO Adjust batch size according to resources
options(shiny.maxRequestSize = Inf) # Infinite upload sizes (slides can get big)
av_time <- 28800L # 8h in seconds
