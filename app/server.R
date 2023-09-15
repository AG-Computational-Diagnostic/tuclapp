
server <- function(input, output, session) {
  
  observeEvent(input$test, {
    thumb = get_thumbnail(
      slide = upload$os,
      max_size = 2048L
    )
    
    thumb_cols = get_thumb_colors(
      thumbnail_shape = thumb$size,
      class_mask = classification$class_mask,
      col_dict = to_dict_w_int_keys(classification$col_dict), 
      show_dict = to_dict_w_int_keys(classification$show_dict)
    )
    
  })
  
  # Reactive Values ------------------------------------------------------------
  invalidate_time <- Inf
  if (file.exists('logout_donate-config.json')) {
    conf_list <- rjson::fromJSON(file = "logout_donate-config.json")
    if(conf_list["logout_path"] != "") {
      invalidate_time <- 1000 # 60000 for minute
    }
  } 
  autoInvalidate <- reactiveTimer(invalidate_time) 
  start_time <- Sys.time()
  remaining_time <- reactiveVal(av_time)  
  
  ## classifier_meta -----------------------------------------------------------
  classifier_meta <- reactiveValues(
    full_name = NULL, # Character
    organ = NULL, # Character
    name = NULL, # Character
    model_dir = NULL, # Character
    patch_size_um = NULL, # Integer
    patch_size_px = NULL, # Integer; Fallback if um to px conversion not available
    magnification = NULL, # Integer
    scale_imgs = NULL, # Bool
    app = NULL, # Character
    input_shape = NULL, # Integer vector
    n_classes = NULL, # Integer
    inv_class_dict = NULL, # List
    class_meta = NULL # Python dictionary
  )
  
  ## upload --------------------------------------------------------------------
  upload <- reactiveValues(
    num = 0L, # Count uploads such that each has some uniqueness (-> reactives)
    is_loadable = NULL, # Bool
    n_patches = NULL, # Integer vector (length 2)
    os = NULL, # openslide,
    magnification = NULL, # Integer
    um_to_px = NULL # Numeric vector (length 2)
  )
  
  # These depend on the selected classifier AND the uploaded slide
  classification_pars <- reactiveValues(
    patch_size_px = NULL, # Integer
    slide_large_enough = FALSE
  )
  
  ## classification ------------------------------------------------------------
  classification <- reactiveValues(
    model = NULL, # TF model
    empty_tiles = NULL, # Data frame
    nonempty_tiles = NULL, # Data frame
    logits_mat = NULL, # Numeric matrix/array
    col_dict = NULL, # R list
    show_dict = NULL, # R list
    class_mask = NULL, # Numeric matrix/array
    sum_table = NULL, # Character (HTML)
    gj_anno = NULL # Python dict
  )
  
  ## selected_tile -------------------------------------------------------------
  selected_tile <- reactiveValues(
    pos = NULL, # Two floats
    tab = NULL, # Data frame; if class_int is in columns: is tile wise
    id = NULL, # Character of an int (index in classification$nonempty_tiles)
    img_arr = NULL, # Numpy array
    mask = NULL, # ??
  )
  
  ## example -------------------------------------------------------------------
  example <- reactiveValues(
    available = character(), # Character vector
    shown = NULL # Integer
  )
  
  # Reactive Functions ---------------------------------------------------------
  
  ## REACT_updateClassifierPars ------------------------------------------------
  REACT_updateClassifierPars <- reactive({
    if(shiny::isTruthy(upload$is_loadable)) {
      if(is.null(upload$um_to_px)) {
        patch_size_px <- rep(classifier_meta$patch_size_px, 2)
      } else {
        patch_size_px <- get_pixel_shape(
          slide = upload$os,
          tile_shape_um = c(
            classifier_meta$patch_size_um,
            classifier_meta$patch_size_um
          )
        )
        patch_size_px <- reticulate::py_to_r(patch_size_px)
      }
      slide_dims <- unlist(upload$os$dimensions)
      slide_large_enough <- all(
        slide_dims > patch_size_px
      )
    } else {
      patch_size_px <- NULL
      slide_large_enough <- FALSE
    }
    classification_pars$patch_size_px <- patch_size_px
    classification_pars$slide_large_enough <- slide_large_enough
  })
  
  ## REACT_enableClassify ------------------------------------------------------
  REACT_enableClassify <- reactive({
    # ATM this is called only if
    #   - classification_pars$slide_large_enough is changed 

    if(classification_pars$slide_large_enough) {
      shinyjs::enable("classify")
    } else {
      shinyjs::disable("classify")
    }
    
  })
  
  ## REACT_compClassMask -------------------------------------------------------
  REACT_compClassMask <- reactive({
    # This is called if
    #   - initial classification is done or
    #   - softmax threshold is changed.
    
    if (!is.null(classification$nonempty_tiles)) {
      classification$class_mask <- get_class_mask(
        tiles_tab = classification$nonempty_tiles,
        logits_mat = classification$logits_mat,
        slide_shape = as.integer(unlist(upload$os$dimensions)),
        tile_shape = as.integer(classification_pars$patch_size_px),
        max_size = class_mask_max_size,
        soft_thresh = input$pred_thres
      ) 
    } else {
      classification$class_mask <- NULL
    }
    
  })
  
  ## REACT_updateSlideSummary --------------------------------------------------
  REACT_updateSlideSummary <- reactive({
    # This is called if either
    #   - class_mask is changed (by initial classification or different threshold) or
    #   - col_dict is changed.
    
    if (!is.null(classification$class_mask)) {
      
      # Compute summary table
      prop_dict <- summarize_class_mask(
        class_mask = classification$class_mask,
        inv_class_dict = to_dict_w_int_keys(classifier_meta$inv_class_dict),
        meta_classes = classifier_meta$class_meta
      )
      classification$sum_table <- get_class_mask_html_tab(
        prop_dict = prop_dict,
        col_dict = to_dict_w_int_keys(classification$col_dict),
        show_dict = to_dict_w_int_keys(classification$show_dict),
        meta_classes = classifier_meta$class_meta,
        inv_class_dict = to_dict_w_int_keys(classifier_meta$inv_class_dict)
      )
      
    } else {
      
      classification$sum_table <- NULL
      
    }
    
  })
  
  ## REACT_updateAnnotation ----------------------------------------------------
  REACT_updateAnnotation <- reactive({
    # This is called if either
    #   - class_mask is changed (by initial classification or different threshold),
    #   - col_dict is changed,
    #   - show_dict is changed or,
    #   - the overlay amount is changed.
    
    session$sendCustomMessage(type = 'clear-anno', message=list())
    
    if (!is.null(classification$class_mask)) { 
      
      class_mask_shapes <- get_scaled(
        original_shape = as.integer(unlist(upload$os$dimensions)),
        max_size = class_mask_max_size,
        tile_shape = as.integer(classification_pars$patch_size_px)
      )
      
      classification$gj_anno <- get_geojson(
        class_mask = classification$class_mask,
        inv_class_dict = to_dict_w_int_keys(classifier_meta$inv_class_dict),
        col_dict = to_dict_w_int_keys(classification$col_dict),
        show_dict = to_dict_w_int_keys(classification$show_dict),
        sfcts = class_mask_shapes$facts
      )
      
      session$sendCustomMessage(
        type = 'add-annotation', 
        message = list(
          gj_anno = as.character(classification$gj_anno),
          alpha = input$overlay_map
        )
      )
      
    } # END if (!is.null(classification$class_mask))
    
  })
  
  ## REACT_clearClassification -------------------------------------------------
  REACT_clearClassification <- reactive({
    # ATM only called by new classifier/slide
    
    clean_classification_results <- list(
      empty_tiles = NULL, # Data frame
      nonempty_tiles = NULL, # Data frame
      logits_mat = NULL, # Numeric matrix/array
      inv_class_dict = NULL, # Python dict
      meta_classes = NULL, # Python dict
      col_dict = NULL, # Python dict
      class_mask = NULL, # Numeric matrix/array
      sum_table = NULL # Character (HTML)
    )
    assignListToReactive(classification, clean_classification_results)
    
    clean_selected_tile <- list(
      pos = NULL, # Two floats
      tab = NULL, # Data frame; if class_int is in columns: is tile wise
      id = NULL, # Character of an int (index in classification$nonempty_tiles)
      img_arr = NULL, # Numpy array
      mask = NULL # ??
    )
    assignListToReactive(selected_tile, clean_selected_tile)
    
    REACT_updateAnnotation()
    
  })
  
  # Timer ----------------------------------------------------------------------
  observe({
    autoInvalidate()
    time_passed <- difftime(Sys.time(), start_time, units = "secs")
    time_passed <- as.integer(ceiling(time_passed))
    time_left <- av_time - time_passed
    if(time_left <= 0) {
      doLogout()
    }
    remaining_time(time_left)  
  })
  
  
  # UI inputs ------------------------------------------------------------------ 
  
  ## input$logout_button -------------------------------------------------------
  observeEvent(input$logout_button, {
    
    confirm_modal <- shiny::modalDialog(
      div(
        class="sure-head",
        "Are you sure?"
      ),
      "This will delete all current classification results.",
      easyClose = TRUE, 
      footer = splitLayout(
        cellWidths = c("50%", "50%"),
        actionButton(
          inputId = "logout_confirm", 
          label = "Yes", 
          width = "100%",
          class = "btn-warning"
        ),
        modalButton(label = "Cancel"),
      )
    )
    showModal(confirm_modal)
    
  })
  
  ## input$logout_confirm ------------------------------------------------------
  observeEvent(input$logout_confirm, {
    doLogout()
  })
  
  ## input$donate_button -------------------------------------------------------
  observeEvent(input$donate_button, {
    conf_list <- rjson::fromJSON(file = "logout_donate-config.json")
    js_cmd <- sprintf(
      "window.open('%s', '_blank').focus();", 
      conf_list["donate_href"]
    )
    shinyjs::runjs(js_cmd)
  })
  
  ## input$classifier ----------------------------------------------------------
  observeEvent(input$classifier, {
    
    cl_meta <- sumClassifier(input$classifier)
    assignListToReactive(classifier_meta, cl_meta)
    
    REACT_updateClassifierPars()
    
    cl_choices <- cl_meta$class_meta |>
      reticulate::py_to_r()
    for(meta_i in 1:length(cl_choices)) {
      meta <- cl_choices[[meta_i]] |>
        as.character()
      meta_cl_choices <- cl_meta$inv_class_dict[meta] |>
        unlist() 
      cl_choices[[meta_i]] <- as.integer(names(meta_cl_choices))
      names(cl_choices[[meta_i]]) <- meta_cl_choices
    }
    updateSelectInput(
      inputId = "example_class", 
      choices = c(list("None"), cl_choices)
    )
    
  })
  
  ## input$slide ---------------------------------------------------------------
  observeEvent(input$slide, {
    session$sendCustomMessage("setSlideInput", list(
      name="<DELETED FOR PRIVACY PROTECTION>",
      size=input$slide$size,
      datapath=input$slide$datapath
    ))
    os_slide <- tryLoading(input$slide$datapath)
    is_loadable <- !is.null(os_slide)
    if (is_loadable) {
      
      upload$os <- os_slide
      upload$magnification <- getSlideProp(
        slide = os_slide, 
        key = "openslide.objective-power",
        conv_fun = as.numeric
      )
      upload$um_to_px <- c(
        getSlideProp(
          slide = os_slide, 
          key = "openslide.mpp-x",
          conv_fun = as.numeric
        ),
        getSlideProp(
          slide = os_slide, 
          key = "openslide.mpp-y",
          conv_fun = as.numeric
        )
      )
    } else {
      upload$os <- "UNLOADABLE"
      upload$magnification <- NULL
      upload$um_to_px <- NULL
    }
    upload$is_loadable <- is_loadable
    upload$num <- upload$num + 1L
    REACT_updateClassifierPars()
  })
  
  ## input$classify ------------------------------------------------------------
  observeEvent(input$classify, {
    # Just to make sure:
    validate(need(classification_pars$slide_large_enough, "Need valid slide"))
    
    # Hide/show/enable/disable things
    shinyjs::disable("classifier")
    shinyjs::hide("slide")
    shinyjs::disable("tile_overlap")
    shinyjs::hide("classify")
    shinyjs::show("new_classifier_slide")
    
    startBusy("Classify slide...")
    
    classification_result <- classify(
      model_dir = classifier_meta$model_dir,
      slide_path = input$slide$datapath,
      tile_shape = classification_pars$patch_size_px,
      overlap = as.integer(input$tile_overlap),
      scale_imgs = classifier_meta$scale_imgs,
      app = classifier_meta$app,
      input_shape = classifier_meta$input_shape,
      max_size_class_mask = class_mask_max_size
    )
    
    classification_result$nonempty_tiles <- classification_result$nonempty_tiles |>
      dplyr::mutate(ID = as.character(1:dplyr::n()))
    
    classification_result$empty_tiles <- classification_result$empty_tiles |>
      dplyr::mutate(ID = 1:dplyr::n()) |>
      dplyr::mutate(ID = paste0("E_", ID))
    
    startBusy("Render Heatmap...")
    assignListToReactive(classification, classification_result)
    REACT_compClassMask()
    
    updateTabsetPanel(
      session, inputId = "sidebar_tabset", selected = "Slide"
    )
    endBusy()
    
  })
  
  ## input$new_classifier_slide ------------------------------------------------
  observeEvent(input$new_classifier_slide, {
    
    confirm_modal <- shiny::modalDialog(
      div(
        class="sure-head",
        "Are you sure?"
      ),
      "This will delete all current classification results.",
      easyClose = TRUE, 
      footer = splitLayout(
        cellWidths = c("50%", "50%"),
        actionButton(
          inputId = "confirm_new_cl_sl", 
          label = "Yes", 
          width = "100%",
          class = "btn-warning"
        ),
        actionButton(
          inputId = "deny_new_cl_sl", 
          label = "No", 
          width = "100%",
          class = "btn-primary"
        )
      )
    )
    showModal(confirm_modal)
    
  })
  observeEvent(input$deny_new_cl_sl, {
    removeModal()
  })
  observeEvent(input$confirm_new_cl_sl, {
    # Hide/show/enable/disable things
    shinyjs::enable("classifier")
    shinyjs::show("slide")
    shinyjs::enable("tile_overlap")
    shinyjs::show("classify")
    shinyjs::hide("new_classifier_slide")
    # Delete classification results
    REACT_clearClassification()
    removeModal()
  })
  
  
  ## input$overlay_map ---------------------------------------------------------
  observeEvent(input$overlay_map, {
    REACT_updateAnnotation()
  })
  
  
  ## input$pred_thres ----------------------------------------------------------
  observeEvent(input$pred_thres, {
    REACT_compClassMask()
  })
  
  ## Change colors -------------------------------------------------------------
  
  ### input$clicked_huebox -----------------------------------------------------
  observeEvent(input$clicked_huebox, {
    
    cl_int <- as.integer(substring(input$clicked_huebox, first = 7))
    old_hue <- classification$col_dict[[as.character(cl_int)]]
    
    # Get text color of old hue
    rgb_col <- col2rgb(old_hue)
    if(mean(rgb_col) < 50) {
      txt_col = "white"
    } else {
      txt_col = "black"
    }
    
    col_picker <- colourpicker::colourInput(
      inputId = "new_hue", 
      label = NULL, 
      value = old_hue
    )
    
    # Check if shown
    is_shown <- classification$show_dict[[as.character(cl_int)]] == 1
    if (!is_shown) {
      col_picker <- shinyjs::disabled(col_picker)
    }
    
    showModal(modalDialog(
      uiOutput("select_colors_modal"),
      div("Current Hue:", class="huemodal-h1"),
      div(class="oldhue-box", 
          old_hue,
          style=sprintf(
            "background-color: %s; color: %s", 
            old_hue, txt_col
          )
      ),
      hr(),
      div("New Hue:", class="huemodal-h1"),
      div(
        style = "display: flex; justify-content: center;",
        div(
          style="width: 50%;",
          div(
            shinyWidgets::materialSwitch(
              inputId = "show_class",
              label = "Show this class?", 
              status = "primary",
              right = FALSE,
              value = is_shown
            ),
            style="line-height:34px;"
          )
        ),
        div(
          style="width: 50%; overflow: visible;",
          col_picker  
        )
      ),
      footer = tagList(
        modalButton("Cancel"),
        actionButton("change_hue", "Change Hue", class = "btn-primary")
      )
    ))
  })
  
  ### input$show_class ---------------------------------------------------------
  observeEvent(input$show_class, {
    if (input$show_class) {
      shinyjs::enable("new_hue")
    } else {
      shinyjs::disable("new_hue")
    }
  })
  
  ### input$change_hue ---------------------------------------------------------
  observeEvent(input$change_hue, {
    cl_int <- as.integer(substring(input$clicked_huebox, first = 7))
    if (input$show_class) {
      classification$col_dict[[as.character(cl_int)]] <- input$new_hue
      classification$show_dict[[as.character(cl_int)]] <- as.integer(input$show_class)
    } else {
      # Do not update color here
      classification$show_dict[[as.character(cl_int)]] <- as.integer(input$show_class)
    }
    removeModal()
  })
  
  ## input$viewer_detail -------------------------------------------------------
  # This is called from JavaScript (when right clicking the viewer)
  observeEvent(input$viewer_detail, {
    
    selected_tile$pos <- input$viewer_detail
    
    selected_empty <- classification$empty_tiles |>
      dplyr::filter(
        x_from < input$viewer_detail$x,
        x_from > input$viewer_detail$x - classification_pars$patch_size_px[1],
        y_from < input$viewer_detail$y,
        y_from > input$viewer_detail$y - classification_pars$patch_size_px[2],
      ) |>
      dplyr::arrange(prop_empty) |>
      dplyr::mutate(Empty = "Yes")
    
    selected_nonempty <- classification$nonempty_tiles |>
      dplyr::filter(
        x_from < input$viewer_detail$x,
        x_from > input$viewer_detail$x - classification_pars$patch_size_px[1],
        y_from < input$viewer_detail$y,
        y_from > input$viewer_detail$y - classification_pars$patch_size_px[2],
      ) |>
      dplyr::mutate(Empty = "No")
    
    selected_tile$tab <- dplyr::bind_rows(
        selected_empty,
        selected_nonempty
      ) |>
      dplyr::arrange(Empty)
    
  })
  
  ## input$to_detail_tile ------------------------------------------------------
  observeEvent(input$to_detail_tile, {
    if (input$to_detail_tile != " " & input$to_detail_tile != "") {
      selected_tile$id <- as.integer(input$to_detail_tile)
    } else {
      selected_tile$id <- NULL
    }
  })
  
  ## input$example_class -------------------------------------------------------
  observeEvent(input$example_class, {
    example$shown = 1
  })
  
  ## input$inspect_tile_example ------------------------------------------------
  observeEvent(input$inspect_tile_example, {
    
    example$available <- getClassExamples(
      classifier_name = input$classifier,
      class_id = input$example_class
    )
    
    ui_example <- list(
      div(
        class = "example-modal-text",
        textOutput(outputId = "example_info", inline = TRUE),
        plotOutput(outputId = "example_imgs")
      )
    )
    
    # Show Examples if available
    if (length(example$available) > 0) {
      
      btn_left <- actionButton(inputId = "example_left", label = "<", width = "100%", style="height: 400px;")
      if(example$shown == 1) {
        btn_left <- shinyjs::disabled(btn_left)
      } 
      
      ui_example <- c(
        ui_example,
        list(
          splitLayout(
            cellWidths = c("10%", "80%", "10%"),
            btn_left,
            plotOutput(outputId = "example_imgs"),
            actionButton(inputId = "example_right", label = ">", width = "100%", style="height: 400px;")
          )
        )
      )

    }
    
    ui_sel_tel <- list(
      div(
        class = "example-modal-text",
        textOutput(outputId = "selected_tile_info", inline = TRUE)
      )
    )
    
    # Show selected tile if available
    if (is.integer(selected_tile$id)) {
      
      ui_sel_tel <- c(
        ui_sel_tel,
        list(plotOutput(outputId = "plot_sel_tile"))
      )
      
      # Get the tile as array
      startBusy('Visualizing selected tile...')
      
      sel_tile_row <- selected_tile$tab |>
        dplyr::filter(ID == as.character(selected_tile$id))
      
      selected_tile$img_arr <- get_tile(
        slide = upload$os, 
        froms = as.integer(c(sel_tile_row$x_from, sel_tile_row$y_from)), 
        tile_shape = as.integer(classification_pars$patch_size_px)
      )
      
      # Get prediction mask
      if (length(classification$logits_mat$shape) > 2) {
        selected_tile$mask <- get_tile_mask_predicted(
          model = classification$model, 
          input_dims = as.integer(classifier_meta$input_shape[1:2]), 
          scale = classifier_meta$scale_imgs, 
          tile = selected_tile$img_arr,
          col_dict = to_dict_w_int_keys(classification$col_dict)
        )  
        ui_sel_tel <- c(
          ui_sel_tel,
          list(sliderInput(
            "overlay_map_tile",
            label = "",
            min = 0, 
            max = 1, 
            value = 0.5, 
            width = "100%",
            ticks = FALSE
          ))
        )
      } else {
        selected_tile$mask <- NULL
      }
      
      endBusy()

    } 
    
    if(input$example_class == "None") {
      ui_modal_dialog <- shiny::modalDialog(
        ui_sel_tel,
        size = c("xl"),
        easyClose = TRUE
      )
    } else {
      ui_modal_dialog <- shiny::modalDialog(
        ui_example,
        hr(),
        ui_sel_tel,
        size = c("xl"),
        easyClose = TRUE
      )
    }
    
    showModal(ui_modal_dialog)    

  })

  ## input$example_left --------------------------------------------------------
  observeEvent(input$example_left, {
    example$shown <- example$shown - 1
  })
  
  ## input$example_right -------------------------------------------------------
  observeEvent(input$example_right, {
    example$shown <- example$shown + 1
  })
  
  
  # UI outputs -----------------------------------------------------------------
  
  ## output$currentTime --------------------------------------------------------
  output$currentTime <- renderText({
    hours <- floor(remaining_time() / 60 / 60)
    minutes <- floor((remaining_time() - hours*60*60) / 60)
    sprintf("%iH %iM", hours, minutes) 
  })
  
  ## output$classifier_info ----------------------------------------------------
  output$classifier_info <- renderText({
    
    validate(need(
      classifier_meta$name, 
      "Select a classifier to view additonal information on that classifier."
    ))
    
    # Read template for classifier box 
    infobox_template <- readLines(
      paste0("models/", input$classifier, "/infobox.md")
    )[2]
    
    # Fill template with values from classifier_meta
    cl_info <- stringr::str_glue_data(
      .x = shiny::reactiveValuesToList(classifier_meta),
      infobox_template
    )
    
    return(cl_info)
    
  })
  
  ## output$upload_details -----------------------------------------------------
  output$upload_details <- renderText({
    
    validate(
      need(
        upload$num > 0, 
        "Upload a file to view details."
      )
    )
    
    show_err <- FALSE
    
    ### Upload size ------------------------------------------------------------
    size_in_mb <- paste(round(input$slide$size/2^20, 2), "MB")
    html_out <- paste("The upload is", size_in_mb, "large")
    if(!upload$is_loadable) {
      err_msg <- "Your upload can not be loaded, see details of your upload."
      html_out <- paste0(
        html_out, " but <b style='color:rgb(201,48,44)'>unloadable</b>."
      )
      html_out <- paste0(
        html_out, 
        " Only files of type ", 
        paste(allowed_exts, collapse = ", "),
        " can be loaded. Please check if your upload is a valid file of such type."
      )
      show_err <- TRUE
    } else {
      # For a good green: color:rgb(68,157,68)
      err_msg <- "Reduced classification accuracy, see details of your upload."
      html_out <- paste0(
        html_out, 
        " and its dimensions are ",
        paste(unlist(upload$os$dimensions), collapse = " x "),
        " px."
      )
    }
    
    ### Magnification ----------------------------------------------------------
    if(upload$is_loadable) {
      if (is.null(upload$magnification)) {
        msg_template <- "markdowns/messages/no-magnification.md"
        show_err <- TRUE
      } else if (upload$magnification != classifier_meta$magnification) {
        msg_template <- "markdowns/messages/magnification-mismatch.md"
        show_err <- TRUE
      } else if (upload$magnification == classifier_meta$magnification) {
        msg_template <- "markdowns/messages/magnification-match.md"
      }
      md_template <- readLines(msg_template)[1]
      mag_info <- stringr::str_glue_data(
        .x = list(
          classifier_meta = shiny::reactiveValuesToList(classifier_meta),
          upload = shiny::reactiveValuesToList(upload)
        ),
        md_template
      )
      html_out <- paste0(
        html_out, " ", mag_info
      )
    }
    
    ### Conversion -------------------------------------------------------------
    if (is.null(upload$um_to_px)) {
      msg_template <- "markdowns/messages/no-conversion.md"
      show_err <- TRUE
    } else {
      msg_template <- "markdowns/messages/conversion-exists.md"
    }
    md_template <- readLines(msg_template)[1]
    mag_info <- stringr::str_glue_data(
      .x = shiny::reactiveValuesToList(classification_pars),
      md_template
    )
    html_out <- paste0(
      html_out, " ", mag_info
    )
    
    # Show error message if there is one
    if(show_err) {
      showNotification(err_msg, type = "error", duration = 20)
    }
    
    return(html_out)
    
  })
  
  ## output$overlap_html -------------------------------------------------------
  output$overlap_html <- renderUI({
    validate(need(
      classification_pars$patch_size_px,
      "Upload a valid slide to be able to specify tile overlap."
    ))
    sliderInput(
      inputId = "tile_overlap",
      label = NULL,
      value = 0, 
      min = 0, 
      max = min(classification_pars$patch_size_px) - 1, 
      width = "100%", 
      step = 1,
      sep = "",
      post = " px"
    )
  })
  
  ## output$info_classify_btn --------------------------------------------------
  output$info_classify_btn <- renderText({
    validate(need(
      classification_pars$patch_size_px,
      "Upload a valid slide to be able to start the classification."
    ))
    
    txt <- ""
    
    # Slide not large enough
    if (!isTruthy(classification_pars$slide_large_enough)) {
      showNotification(
        "Slide is to small for selected classifier!", 
        type = "error", 
        duration = 20
      )
      txt <- paste0(
        txt, readLines("markdowns/messages/slide-to-small.md")[1]
      )
    }
    
    if(txt != "") {
      return(txt)
    }
    
  })
  
  ## output$pred_hist -----------------------------------------------------------
  output$pred_hist <- renderPlot({
    
    validate(
      need(
        classification$nonempty_tiles,
        "Classify a slide to see the histogram of predictions."
      )
    )

    if ("soft" %in% colnames(classification$nonempty_tiles)) {
      pred <- classification$nonempty_tiles$soft
    } else {
      # TODO
      pred <- tf$convert_to_tensor(classification$logits_mat)
      pred <- tf$keras$activations$softmax(pred)
      pred <- tf$math$reduce_max(pred, axis=-1L)$numpy() |>
        as.numeric()
    }
    
    par(oma=rep(0,4), mar=rep(0,4), yaxs="i", xaxs="i")
    hist(
      pred,
      main = "",
      axes = FALSE,
      ylab = "",
      xlab = "",
      # breaks = 20,
      col = "darkgray",
      xlim = c(0,1)
    )
    abline(v = input$pred_thres, col=rgb(0, 0, 1, 0.4), lwd=2)
    
  })
  
  ## output$slide_sum_tab ------------------------------------------------------
  output$slide_sum_tab <- renderText({
    validate(
      need(
        classification$sum_table,
        "Classify a slide to see the summary."
      )
    )
    reticulate::py_to_r(classification$sum_table)
  })
  
  ## output$download_btns ------------------------------------------------------
  output$download_btns <- renderUI({
    
    validate(
      need(
        classification$sum_table,
        "Classify a slide to enable downloads."
      )
    )
    
    splitLayout(
      # style="margin-bottom:15px;",
      cellWidths = c("42%", "28%", "30%"),
      downloadButton(
        outputId = "download_geojson", 
        label = "GeoJSON", 
        style="width:100%"
      ),
      downloadButton(
        outputId = "download_csv",
        label = "CSV",
        style="width:100%"
      ),
      downloadButton(
        outputId = "download_html",
        label = "HTML",
        style="width:100%"
      )
    )
    
  })
  
  ### output$download_geojson --------------------------------------------------
  output$download_geojson <- downloadHandler(
    filename = getDonwloadFilename(what = "annotation", fext = "geojson"),
    content = function(file) {
      
      startBusy("Creating download ...")
      
      write_gj(classification$gj_anno, file)
      
      endBusy()
      
    }
  )
  
  ### output$download_csv ------------------------------------------------------
  output$download_csv <- downloadHandler(
    filename = getDonwloadFilename(what = "all-tiles", fext = "csv"),
    content = function(file) {
      
      startBusy("Creating download ...")
      
      write.csv(classification$nonempty_tiles, file = file, row.names = FALSE)
      
      endBusy()
      
    }
  )
  
  ### HTML ---------------------------------------------------------------------
  output$download_html <- downloadHandler(
    filename = getDonwloadFilename(what = "full-report", fext = "html"),
    content = function(file) {
      
      startBusy("Creating download ...")
      
      thumb = get_thumbnail(
        slide = upload$os,
        max_size = 2048L
      )
      
      thumb_cols = get_thumb_colors(
        thumbnail_shape = thumb$size,
        class_mask = classification$class_mask,
        col_dict = to_dict_w_int_keys(classification$col_dict), 
        show_dict = to_dict_w_int_keys(classification$show_dict),
        alpha = input$overlay_map
      )
      
      colored_thumb = get_colored_thumb(
        thumb = thumb,
        thumb_cols = thumb_cols
      )
      
      img_file <- tempfile(fileext = ".png")
      colored_thumb$save(img_file)
      colored_thumb_R <- imager::load.image(img_file)
      file.remove(img_file) # Just to make sure
      
      if ("soft" %in% colnames(classification$nonempty_tiles)) {
        pred_prob <- classification$nonempty_tiles$soft
      } else {
        # TODO
        pred_prob <- tf$convert_to_tensor(classification$logits_mat)
        pred_prob <- tf$keras$activations$softmax(pred_prob)
        pred_prob <- tf$math$reduce_max(pred_prob, axis=-1L)$numpy() |>
          as.numeric()
      }
      
      markdown_env <- new.env(parent = globalenv())
      var_list = list(
        classifier = classifier_meta$full_name,
        pred_thresh = input$pred_thres,
        pred_prob = pred_prob,
        mix_prop=input$overlay_map,
        heatmap=colored_thumb_R,
        classes_table_str=reticulate::py_to_r(classification$sum_table)
      )
      assignMultiple(var_list, env = markdown_env)
      
      rmarkdown::render(
        input = "markdowns/download-slide-result.Rmd",
        output_file = file,
        params = par_list,
        intermediates_dir=tempdir(),
        envir = markdown_env 
      )
    
      endBusy()
      
    }
  )
  
  ## output$selected_tiles_tab -------------------------------------------------
  output$selected_tiles_tab <- renderTable(
    width="100%",
    striped = FALSE,
    hover = TRUE,
    bordered = FALSE,
    {

      validate(
        need(
          classification$nonempty_tiles,
          "Classify a slide to be able to select specific tiles."
        )
      )

      validate(
        need(
          selected_tile$tab,
          "Select a tile via right click on the slide."
        )
      )

      # | Number   | Empty         | Prop. Empty |
      # |----------|:-------------:|-------------|
      # | 1        | No            | 0.342       |
      # | 2        | Yes           | 1           |

      selected_tile$tab |>
        dplyr::rename(`Prop. Empty` = prop_empty) |>
        dplyr::select(ID, Empty, `Prop. Empty`)

    }
  )
  
  ## output$tile_detail_tab ----------------------------------------------------
  output$tile_detail_tab <- renderTable(
    width="100%", 
    striped = FALSE,
    hover = TRUE,
    bordered = FALSE,
    {
    
      print("tile_detail_tab")
      
      validate(
        need(
          classification$logits_mat$shape,
          "Classify a slide to be able to see tile details."
        )
      )
      
      validate(
        need(
          is.integer(selected_tile$id),
          "Select a non-empty tile to see details."
        )
      )
      
      py_idx <- as.integer(selected_tile$id) - 1 # Python starts at 0
      
      if (length(classification$logits_mat$shape) > 2) {
        ### Segmentation -------------------------------------------------------
        # | Class           | %   |
        # |:----------------|-----|
        # | nontumor_sadasd | 43  |
        # | tumor_dddddsa   | 22  |

        pred_mask <- tf$convert_to_tensor(classification$logits_mat[py_idx])
        pred_mask <- tf$keras$activations$softmax(pred_mask)
        pred_mask <- tf$math$argmax(pred_mask, axis=-1L)$numpy()
        pred_cnts <- np$unique(pred_mask, return_counts=TRUE)

        tab <- tibble::tibble(
          Class = character(),
          `%` = numeric()
        )
        for (cl_int_str in names(classifier_meta$inv_class_dict)) {
          idx_cnts <- which(pred_cnts[[1]] == as.integer(cl_int_str))
          if (length(idx_cnts) == 1) {
            perc <- 100 * pred_cnts[[2]][idx_cnts] / length(pred_mask)
          } else {
            perc <- 0
          }
          tab <- tab |>
            dplyr::add_row(
              Class = classifier_meta$inv_class_dict[[cl_int_str]],
              `%` = perc
            )
        }
        
        tab  |>
          dplyr::arrange(dplyr::desc(`%`))
        
      } else {
        ### Tile based ---------------------------------------------------------
        # Classified as nontumor_sadasd.
        # 
        # | Class           | Softmax |
        # |:----------------|---------|
        # | nontumor_sadasd | 0.942   |
        # | tumor_dddddsa   | 0.1     |   
        
        softmax_values <- classification$logits_mat[py_idx] |>
          tf$convert_to_tensor() |>
          tf$expand_dims(axis=0L) |>
          tf$keras$activations$softmax() |>
          tf$squeeze()
        softmax_values <- softmax_values$numpy()
        
        
        class_strs <- c()
        for (class_int in 1:length(softmax_values) - 1) {
          class_str <- to_dict_w_int_keys(classifier_meta$inv_class_dict)[class_int] |>
            reticulate::py_to_r()
          class_strs <- c(class_strs, class_str)
        }
        
        tab <- data.frame(
          Class = class_strs,
          Softmax = softmax_values
        ) |>
          dplyr::arrange(dplyr::desc(Softmax))
      }
      
      tab
    
    }
  )
  
  
  ## output$example_info -------------------------------------------------------
  output$example_info <- renderText({
    
    if (length(example$available) > 0) {
      sprintf(
        "Example %i of %i from %s", 
        example$shown,
        length(example$available),
        classifier_meta$inv_class_dict[input$example_class]
      )  
    } else {
      sprintf(
        "Training Example from %s", 
        classifier_meta$inv_class_dict[input$example_class]
      )  
    }
    
  })
  
  ## output$selected_tile_info -------------------------------------------------
  output$selected_tile_info <- renderText({
    
    validate(
      need(
        is.numeric(selected_tile$id),
        "Select a non-empty tile to see it here."
      )
    )
    
    sel_tile_row <- selected_tile$tab |>
      dplyr::filter(ID == as.character(selected_tile$id))
    
    if ('class_int' %in% colnames(sel_tile_row)) {
    # Tile based  
      res_str <- sprintf(
        "Tile %s predicted with %s", 
        input$to_detail_tile,
        to_dict_w_int_keys(classifier_meta$inv_class_dict)[sel_tile_row$class_int] |>
          reticulate::py_to_r()
      )
    } else {
    # Segmentation
      res_str <- sprintf("Tile %s", input$to_detail_tile)
    }
    
    res_str
    
  })
  
  ## output$example_imgs -------------------------------------------------------
  output$example_imgs <- renderPlot({
    
    par(mar=c(0,0,0,0))
    if (length(example$available) > 0) {
      example_cimg <- imager::load.image(example$available[example$shown])
      plot(example_cimg, axes = FALSE, rescale=FALSE)  
    } else {
      plot(0, 0, type="n", axes=FALSE, xlab="", ylab="")
      text(0, 0, "Training Example", cex=2)
      
    }
    
  })
  
  ## output$plot_sel_tile ------------------------------------------------------
  output$plot_sel_tile <- renderPlot({
    validate(
      need(
        selected_tile$img_arr,
        "Select a non-empty tile to see it here."
      )
    )
    
    plot_img <- merge_with_mask(
      im_tile = selected_tile$img_arr, 
      im_mask = selected_tile$mask, 
      alpha = input$overlay_map_tile
    )
    
    par(mar=c(0,0,0,0))
    plot_img <- plot_img |>
      reticulate::py_to_r() |>
      asCimgNoWarn()
    plot(plot_img, axes = FALSE, rescale=FALSE)
    
  })
  
  
  # On Reactives ---------------------------------------------------------------
  
  ## upload$os -----------------------------------------------------------------
  observeEvent(upload$os, {
    if (all(class(upload$os) == "character")) { # Unloadable upload
      session$sendCustomMessage(type = 'close-viewer', message = list())
      shinyjs::hide("osdview")
      shinyjs::show("osdview_ph")
      shinyjs::disable("classify")
    } else { # Upload valid
      
      startBusy('Processing slide...')
      
      # Might be better to adopt to the upload?
      overlap = 0
      tile_size = 395
      
      upload_dir <- dirname(isolate(input$slide$datapath))
      dz_dir <- paste0(upload_dir, '/tiled/')
      dir.create(dz_dir)
      dz_gen <- dz$DeepZoomGenerator(
        osr = isolate(upload$os),
        tile_size = tile_size,
        overlap = 0
      )
      
      # Create the endpoint for the HTTP requests
      base_url <- session$registerDataObj(
        name   = 'deepzoom-tiles', # Add additional hash for better security? 
        data = list(dir=dz_dir, dz_gen=dz_gen),
        filter = respondWithTile
      )
      
      # Initialize the viewer
      slide_shape = unlist(upload$os$dimensions)
      session$sendCustomMessage(
        type = 'new-tilesource', 
        message = list(
          width = slide_shape[1],
          height = slide_shape[2],
          p_size = tile_size,
          overlap = 0,
          tile_base_url = base_url
        )
      )
      
      shinyjs::hide("osdview_ph")
      shinyjs::show("osdview")
      
      endBusy()
    }
  })
  
  ## classification_pars$slide_large_enough ------------------------------------
  observeEvent(classification_pars$slide_large_enough, {
    REACT_enableClassify()
  })
  
  ## classification$class_mask -------------------------------------------------
  observeEvent(classification$class_mask, {
    REACT_updateSlideSummary()
    REACT_updateAnnotation()
  })
  
  ## classification$col_dict ---------------------------------------------------
  observeEvent(classification$col_dict, {
    REACT_updateSlideSummary()
    REACT_updateAnnotation()
  })
  
  ## classification$show_dict --------------------------------------------------
  observeEvent(classification$show_dict, {
    REACT_updateSlideSummary()
    REACT_updateAnnotation()
  })
  
  ## selected_tile$tab ---------------------------------------------------------
  observeEvent(selected_tile$tab, {

    activate_to_detail <- FALSE
    if (!is.null(selected_tile$tab)) {
      nonempty_number <-  selected_tile$tab |>
        dplyr::filter(Empty == "No") |>
        dplyr::pull(ID)
      if (length(nonempty_number) > 0) {
        activate_to_detail <- TRUE
      } 
    }

    if(activate_to_detail) {
      shinyjs::enable("to_detail_tile")
      updateSelectInput(
        inputId = "to_detail_tile",
        choices = nonempty_number
      )
    } else {
      shinyjs::disable("to_detail_tile")
      updateSelectInput(
        inputId = "to_detail_tile",
        choices = list(" "),
        selected = " "
      )
    }

  })
  
  ## example$shown -------------------------------------------------------------
  observeEvent(example$shown, {
    shinyjs::toggleState("example_left", example$shown > 1)
    shinyjs::toggleState("example_right", example$shown < length(example$available))
  })
  
  
  # Info Buttons ---------------------------------------------------------------
  observeEvent(
    input$info_classifier_selection, 
    showInfoModal("classifier_selection")
  )
  observeEvent(
    input$info_slide_upload, 
    showInfoModal("slide_upload")
  )
  observeEvent(
    input$info_tile_overlap, 
    showInfoModal("tile_overlap")
  )
  observeEvent(
    input$info_slide_download, 
    showInfoModal("slide_download")
  )
  observeEvent(
    input$info_amount_overlay, 
    showInfoModal("amount_overlay")
  )
  observeEvent(
    input$info_prediction_threshold, 
    showInfoModal("prediction_threshold")
  )
  observeEvent(
    input$info_slide_summary, 
    showInfoModal("slide_summary")
  )
  observeEvent(
    input$info_selected_tiles, 
    showInfoModal("selected_tiles")
  )
  observeEvent(
    input$info_tile_details, 
    showInfoModal("tile_details")
  )
  observeEvent(
    input$info_inspect_tile, 
    showInfoModal("inspect_tile")
  )
  
}