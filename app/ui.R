

# Info Button ------------------------------------------------------------------

infoButton <- function(id) {
  return(
    actionButton(
      inputId = id,
      label = "i", 
      class = "btn-primary",
      style="height: 15px; line-height: 1px; font-weight: bold;"
    )
  )
}

# Logout/Donate ----------------------------------------------------------------

if (file.exists('logout_donate-config.json')) {
  conf_list <- rjson::fromJSON(file = "logout_donate-config.json")
  if(conf_list["logout_path"] != "") {
    logout_donate <- splitLayout(
      id="logout_donate_ui",
      style="margin-bottom:15px;",
      cellWidths = c("30%", "35%", "35%"),
      div(
        textOutput("currentTime", inline = FALSE),
        style="width: 100%; height: 34px; text-align: right; padding-right: 15px; line-height:34px"
      ),
      actionButton(
        inputId = "logout_button",
        label = "Logout",
        width = "100%",
        class = "btn-danger"
      ),
      actionButton(
        inputId = "donate_button",
        label = "Donate",
        width = "100%",
        class = "btn-info"
      )
    ) 
  } else {
    logout_donate <- div()
  }

} else {
  logout_donate <- div()
}


# Slide Upload -----------------------------------------------------------------

# We alter the fileInput such that the file name is not shown.
slide_upload <- fileInput(
  inputId = "slide",
  label = NULL,
  placeholder = "Please select a slide for upload."
)
slide_upload$attribs$style <- "margin-bottom:5px"
slide_upload$children[[2]]$children[[2]] <- slide_upload$children[[3]]
slide_upload$children[[2]]$children[[2]]$attribs[["style"]] <- "height:33px; margin:0px;"
slide_upload$children[[2]]$children[[2]]$children[[1]]$attribs[["style"]] <- "line-height:33px;"
slide_upload$children[[3]] <- NULL



# Start Tab --------------------------------------------------------------------


## Available classifiers -------------------------------------------------------
av_classifiers <- list()
existing_organs <- list.dirs(
  "models/",
  recursive = FALSE,
  full.names = FALSE
)
for (organ in existing_organs) {
  only_classifier_names <- list.dirs(
    paste0("models/", organ), 
    recursive = FALSE, 
    full.names = FALSE
  )
  organ_classifiers <- paste0(organ, "/", only_classifier_names)
  names(organ_classifiers) <- paste0(organ, "/", only_classifier_names)
  av_classifiers[[organ]] <- organ_classifiers
}

## Actual Tab ------------------------------------------------------------------
start_tab <- tabPanel(
  title="Start",
  ### Classifier selection ----------------------------------------------------- 
  h5(
    style="margin-bottom:-10px",
    infoButton("info_classifier_selection"),
    "Classifier"
  ),  
  selectInput(
    "classifier", 
    label = "", 
    choices = av_classifiers
  ),
  htmlOutput(
    "classifier_info", 
    style="margin-bottom: 15px; margin-top: -15px; text-align: justify"
  ),
  ### Slide upload -------------------------------------------------------------
  h5(
    style="margin-top: 25px;",
    infoButton("info_slide_upload"),
    "Upload"
  ),  
  slide_upload,
  htmlOutput("upload_details", style="margin-bottom:20px; text-align: justify"),
  ### Overlap ------------------------------------------------------------------ 
  h5(
    # style="margin-bottom:-10px",
    infoButton("info_tile_overlap"),
    "Tile Overlap"
  ),  
  uiOutput("overlap_html", style="margin-bottom:20px;"),
  ### Classify Button ----------------------------------------------------------
  hr(style="border: 1px solid gray;"),
  div(
    htmlOutput(outputId = "info_classify_btn"),
    style = "margin-bottom: 5px; text-align: justify;"
  ),
  shinyjs::hidden(
    actionButton(
      "new_classifier_slide", 
      "Change classifer or slide",
      width = "100%", 
      class = "btn-danger"
    )
  ),
  # Only gets enabled when slide is valid
  shinyjs::disabled(
    actionButton(
      "classify", 
      "Classify Slide!",
      width = "100%", 
      class = "btn-primary"
    )   
  )
)


# Slide Tab --------------------------------------------------------------------
slide_tab <- tabPanel(
  title="Slide",
  ## Overlay -------------------------------------------------------------------
  h5(infoButton("info_amount_overlay"), "Prediction Hue Strength"),
  div(style="margin-bottom:-10px",
      "Adjust strength of classification colors."
  ),
  sliderInput(
    "overlay_map",
    label = "",
    min = 0, 
    max = 1, 
    value = 0.5, 
    width = "100%",
    ticks = FALSE
  ),
  ## Threshold -----------------------------------------------------------------
  h5(
    style="padding-top:5px",
    infoButton("info_prediction_threshold"), "Preditcion Threshold"
  ),
  div(style="margin-bottom:5px",
      "Exclude tiles with prediction below set value."
  ),
  div(
    style="margin-bottom:-15px",
    plotOutput(
      "pred_hist", 
      height = "80px", 
      width = "100%"
    )
  ),
  sliderInput(
    "pred_thres",
    label = "",
    min = 0, 
    max = 1, 
    value = 0, 
    width = "100%"
  ),
  ## Summary -------------------------------------------------------------------
  h5(
    infoButton("info_slide_summary"), "Summary" 
    #TODO actionLink(inputId="select_colors", "", icon = icon("palette"))
  ),
  div(style="margin-bottom:5px",
      "Count, percentage and hue of patches by class."
  ),
  htmlOutput('slide_sum_tab', style="margin-bottom: 30px;"),
  ## Download ------------------------------------------------------------------
  h5(infoButton("info_slide_download"), "Download"),
  uiOutput("download_btns")
)


# Tile Tab ---------------------------------------------------------------------
tile_tab <- tabPanel(
  title="Tile",
  ## Selected Tiles ------------------------------------------------------------
  h5(infoButton("info_selected_tiles"), "Selected tile(s)"),
  tableOutput("selected_tiles_tab"),
  ## Tile details --------------------------------------------------------------
  h5(infoButton("info_tile_details"), "Tile details"),
  selectInput("to_detail_tile", label = NULL, choices = list()),
  htmlOutput(outputId = "tile_detail_tab"),
  ## Class examples ------------------------------------------------------------
  h5(infoButton("info_inspect_tile"), "Inspect tile"),
  div("With examples of class:"),
  selectInput("example_class", label = NULL, choices = list()),
  actionButton(
    "inspect_tile_example", 
    "Inspect",
    width = "100%"
  )  
)


# Footer -----------------------------------------------------------------------

page_footer <- list(
  div(
    style="background-color:#f4f6f7; height:28px; position:fixed; bottom:0; width:100%; text-align: center;",
    hr(style="margin-top: 0px; margin-bottom: 2px;"),
    HTML("FOR RESEARCH USE ONLY")
  )
)


# UI ---------------------------------------------------------------------------

ui <- fluidPage(

  shinyjs::useShinyjs(),
  
  ## Custom CSS and JS ---------------------------------------------------------
  tags$head(
    tags$link(rel = "stylesheet", type = "text/css", href = "tucl-app.css"),
  ),
  tags$script(src = "tucl-app.js"),
  tags$script(src = "openseadragon-bin/openseadragon.min.js"),
  tags$script(src = "openseadragon-svg-overlay.js"),
  tags$script(src = "d3.min.js"),
  
  ## Tile selection div --------------------------------------------------------
  div(
    id='selected-patch-highlight',
    style="font-size: large; display: none;",
    "x"
  ),
  
  ## Page ----------------------------------------------------------------------
  sidebarLayout(
    
    ### Side -------------------------------------------------------------------
    sidebarPanel(
      logout_donate,
      tabsetPanel(
        id = "sidebar_tabset",
        type="tabs",
        start_tab,
        slide_tab,
        tile_tab
      )
      
    ), # END sidebarPanel

    ### Main -------------------------------------------------------------------
    mainPanel(
      shinyjs::hidden(div(id = "osdview", class = "osdiv")),
      div(
        id = "osdview_ph", 
        class = "osdiv-ph",
        "Upload a valid slide to see it here."
      )
    ) # END mainPanel
    
  ), # END sidebarLayout
  
  page_footer
  
)