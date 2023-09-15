
var viewer;
var overlay;
var tile_url;

function getPosHeight() {
  return window.innerHeight - 58 + 'px'; 
}

function resizeMainDiv() {
  pos_height = getPosHeight()
  osd_div = document.getElementById("osdview");
  osd_div.style.height = pos_height;
  osd_div_ph = document.getElementById("osdview_ph");
  osd_div_ph.style.height = pos_height;
  osd_div_ph.style.lineHeight = pos_height;
}

// -----------------------------------------------------------------------------

onload = (event) => {
  resizeMainDiv()
};


// -----------------------------------------------------------------------------

onresize = (event) => {
  resizeMainDiv()
};

// -----------------------------------------------------------------------------

onclick = (event) => {
  if(event.target.className == "huebox") {
    Shiny.setInputValue("clicked_huebox", event.target.id, {priority: "event"});
  }
};

// -----------------------------------------------------------------------------

oncontextmenu = (event) => {
  return false;
};

// -----------------------------------------------------------------------------

function getTile(level, x, y) {
  Shiny.setInputValue(
    "req_tile",
    {level: level, x: x, y: y},
    {priority: "event"}
  );
  return tile_url
};

Shiny.addCustomMessageHandler(
  'return-tile',
  function(message) {
    console.log('recived url')
    tile_url = message['url']
  }
);

// -----------------------------------------------------------------------------

function constructTileSource(width, height, p_size, overlap, tile_base_url) {
  
  ts = new OpenSeadragon.TileSource(
    width=width,
    height=height,
    tileSize=p_size,
    tileOverlap=overlap
  );

  ts.getTileUrl = function(level, x, y) {
    return tile_base_url + '&lvl=' + level + '&x=' + x + '&y=' + y
  };
  
  return ts;
  
}

function initViewer(ts) {

  viewer = OpenSeadragon({
      id: 'osdview',
      prefixUrl: 'openseadragon-flat-toolbar-icons-master/images/',
      tileSources: ts
  });

  // This handles right click on viewer
  viewer.addHandler('canvas-nonprimary-press', function(event) {
    if (event['button'] == 2) {
      var viewportPoint = viewer.viewport.pointFromPixel(event['position']);
      viewer.removeOverlay('selected-patch-highlight')
      document.getElementById('selected-patch-highlight').style.display = "block";
      viewer.addOverlay(
        'selected-patch-highlight', // ID of a div element
        viewportPoint,
        OpenSeadragon.Placement.CENTER
      )
      var imagePoint = viewer.viewport.viewportToImageCoordinates(viewportPoint);
      Shiny.setInputValue(
        "viewer_detail",
        imagePoint
      );
    }
  });

  overlay = viewer.svgOverlay();

}

function addTileSource(width, height, p_size, overlap, tile_base_url) {
  
  ts = constructTileSource(
    width=width,
    height=height,
    tileSize=p_size,
    tileOverlap=overlap,
    tile_base_url=tile_base_url
  )
  
  if( viewer ) {
    viewer.open(ts)
  } else {
    initViewer(ts)
  }
  
}

Shiny.addCustomMessageHandler('new-tilesource', function(message) {
  addTileSource(
    width = message['width'],
    height = message['height'],
    p_size = message['p_size'],
    overlap = message['overlap'],
    tile_base_url = message['tile_base_url']
  )
});

Shiny.addCustomMessageHandler('close-viewer', function(message) {
  if( viewer ) {
    viewer.close()
  }
});


// -----------------------------------------------------------------------------

function addAnnotation(geoJson, alpha) {
    
  projection = d3.geoTransform({
    point: function(x, y) {
      vc = overlay._viewer.viewport.imageToViewportCoordinates(x, y)
      this.stream.point(vc.x, vc.y);
    }
  });
  path = d3.geoPath(projection);
  
  for (var i = 0; i < geoJson.features.length ; i++) {
    fill = geoJson.features[i].properties.color;
    fill = "rgb(" + String(fill) + ")";
    
    d3.select(overlay.node())
      .append("path")
      .datum(geoJson.features[i])
      .attr("d", path)
      .attr('fill', fill)
      .attr('fill-opacity', alpha);
  }
}

Shiny.addCustomMessageHandler('add-annotation', function(message) {
  addAnnotation(
    geoJson = JSON.parse(message['gj_anno']), 
    alpha = message['alpha']
  );
});

// -----------------------------------------------------------------------------

function clearOverlay() {
  if (overlay) {
    d3.select(overlay.node()).selectAll("*").remove();
  }
}

Shiny.addCustomMessageHandler('clear-anno', function(message) {
  clearOverlay()
});



