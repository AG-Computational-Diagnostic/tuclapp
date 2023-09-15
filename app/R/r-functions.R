
startBusy <- function(task_str) {
  shinybusy::show_modal_spinner(spin='self-building-square', text = task_str)
}
endBusy <- function() {
  shinybusy::remove_modal_spinner()
}

showInfoModal <- function(modal_name) {
  file_name <- paste0(
    "markdowns/info/", 
    modal_name,
    ".md"
  )
  if(file.exists(file_name)) {
    infomodal <- shiny::modalDialog(
      div(
        class="inof-head",
        "Info"
      ),
      includeMarkdown(file_name),
      easyClose = TRUE
    )
    showModal(infomodal)
  } else {
    warning(paste(
      "Markdown",  file_name, "not found!"
    ))
  }
}

assignListToReactive <- function(react, named_list) {
  for(this_name in names(named_list)) {
    react[[this_name]] <- named_list[[this_name]]
  }
}

assignMultiple <- function(named_list, env) {
  for (i in 1:length(named_list)) {
    assign(names(named_list)[i], named_list[i], env)
  }
}

getModelDir <- function(classifier_name) {
  return(paste0("models/", classifier_name, "/"))
}

sumClassifier <- function(classifier_name) {
  
  model_dir <- getModelDir(classifier_name)
  meta <- rjson::fromJSON(file = paste0(model_dir, "meta.json"))
  
  cl_split <- strsplit(classifier_name, split = "/")[[1]]
  
  return(list(
    full_name = classifier_name,
    organ = cl_split[1],
    name = cl_split[2],
    model_dir = model_dir,
    patch_size_px = meta$patch_size_px,
    patch_size_um = meta$patch_size_um,
    magnification = meta$magnification,
    scale_imgs = meta$scale_imgs,
    app = meta$app,
    input_shape = as.integer(meta$input_shape),
    n_classes = meta$n_classes,
    inv_class_dict = paste0(model_dir, "class_dict.json") |>
      get_inv_class_dict() |>
      reticulate::py_to_r(),
    class_meta = get_dict(paste0(model_dir, "class_meta.json"))
  ))
  
}

getClassExamples <- function(classifier_name, class_id) {
  model_dir <- getModelDir(classifier_name)
  example_path <- paste0(model_dir, 'examples/', class_id)
  return(list.files(example_path,  full.names = TRUE))
}

tryLoading <- function(file_path) {
  os_slide <- NULL
  fext <- tolower(tools::file_ext(file_path))
  if(fext %in% openslide_ext) {
    try(os_slide <- openslide$OpenSlide(file_path))
  }
  if(fext %in% pillow_ext) {
    try(os_slide <- openslide$ImageSlide(file_path))
  }
  return(os_slide)
}


getSlideProp <- function(slide, key, conv_fun=I()) {
  val <- tryCatch({
    conv_fun(slide$properties[key])
  },
  error = function(e) {
    return(NULL)
  }
  )
  return(val)
}


#' Function that creates HTTP response for deep zoom tile
#'
#' @param data A list with 'dir' (the directory) tiles 
#' are saved in and 'dz_gen' the tile generator.
#' @param req The HTTP request received.
#' 
#' @description 
respondWithTile <- function(data, req) {
  
  # Parse the query
  query <- shiny::parseQueryString(req$QUERY_STRING)
  
  # From the query, construct the tile file name
  tile_file <- paste0(
    data$dir, query$lvl, '/', query$x, '_', query$y, '.png'
  )
  
  # If tile does not exist yet, create it
  if (!file.exists(tile_file)) {
    tile <- data$dz_gen$get_tile(
      level = as.integer(query$lvl), 
      address = as.integer(c(query$x, query$y))
    )
    dir.create(dirname(tile_file), showWarnings = FALSE)
    tile$save(tile_file)
  }
  
  # Send the tile back
  shiny:::httpResponse(
    200,
    'image/png',
    readBin(tile_file, 'raw', file.info(tile_file)[, 'size'])
  )
  
}

getDonwloadFilename <- function(what, fext) {
  return(sprintf(
    '%s%s_RESEARCH-ONLY.%s',
    what, 
    format(Sys.time(), "_%Y-%m-%d_%H%M%S"),
    fext
  ))
}

classify <- function(
    model_dir, 
    slide_path,
    tile_shape,
    overlap,
    scale_imgs,
    app,
    input_shape,
    max_size_class_mask
  ) { 
  
  # Get class dictionaries
  # I could set these when selecting the classifier already. However,
  # they might slow down the app slightly then.
  inv_class_dict <- get_inv_class_dict(paste0(model_dir, "class_dict.json"))
  meta_classes <- get_dict(paste0(model_dir, "class_meta.json"))
  
  # Get initial colors 
  col_dict <- get_colors(length(inv_class_dict))
  show_dict <- get_shows(col_dict)

  # Get model
  if (grepl(pattern = "/DEBUG/", x = model_dir)) {
    model <- load_full_model(paste0(model_dir, "full_model.h5"))
  } else {
    model <- load_model_from_weights(
      weights_file = paste0(model_dir, "weights.hdf5"), 
      app = app, 
      input_shape = input_shape, 
      n_classes = length(inv_class_dict)
    )
  }
  
  # Classify
  classification_result = classify_ws(
    slide_path = slide_path,
    tile_shape = as.integer(tile_shape),
    model = model,
    overlap = as.integer(overlap),
    empty_mask_max_size = 1024L,
    class_mask_max_size = max_size_class_mask,
    scale = scale_imgs,
    tiles_dir = NULL # <- No extraction
  )
  
  result_list <- list(
    model = model,
    empty_tiles = reticulate::py_to_r(classification_result[0]),
    nonempty_tiles = reticulate::py_to_r(classification_result[1]),
    logits_mat = classification_result[2],
    col_dict = reticulate::py_to_r(col_dict),
    show_dict = reticulate::py_to_r(show_dict)
  )
  
  return(result_list)
  
}


asCimgNoWarn <- function(arr)  {
  return(suppressWarnings(imager::as.cimg(arr)))
}

doLogout <- function() {
  conf_list <- rjson::fromJSON(file = "logout_donate-config.json")
  js_cmd <- sprintf(
    "window.location.pathname = '%s'", 
    conf_list["logout_path"]
  )
  shinyjs::runjs(js_cmd)
}

