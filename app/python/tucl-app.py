
# Workflow:
# 1. Compute empty mask (via Otsu)
# 2. Classify non-empty tiles with model
# 3. Create color map an overlay a thumbnail

import math
import tempfile
import subprocess
import os
import base64
import collections
import sys
import json

import tensorflow as tf
import openslide as osl
from PIL import Image
from PIL import ImageOps
from PIL import ImageColor
import numpy as np
import pandas as pd
import cv2 as cv
import geojson as gj
import colormap
import tqdm

# ------------------------------------------------------------------------------

def get_thumbnail(slide, max_size):
    thumb = slide.get_thumbnail((1024, 1024))
    return thumb.convert(mode='RGBA')

# ------------------------------------------------------------------------------

def get_empty_mask(slide, max_size):
    """
    Create a thumbnail from slide, turn it into graysacle and apply Otsu's 
    thresholding to get a mask for empty regions of slide.
    """
    thumb = slide.get_thumbnail((max_size, max_size))
    thumb = ImageOps.grayscale(thumb)
    thresh, mask = cv.threshold(
        np.array(thumb),
        0,
        1,
        cv.THRESH_BINARY + cv.THRESH_OTSU
    )
    return mask

# ------------------------------------------------------------------------------

Scaledshape = collections.namedtuple(
    'Scaledshape',
    ['full', 'tile', 'facts']
)

def get_scaled(
        original_shape,
        scaled_shape = None,
        max_size = None,
        tile_shape = None
    ):
    """
    Return a Scaledshape collection that describes size of the scaled
    full slide or tile. Either 'scaled_shape' or 'max_size' must be
    provided.
    """
    if not scaled_shape:
        idx_max_size = np.argmax(original_shape)
        s_fct = original_shape[idx_max_size] / max_size
        scaled_shape = [round(d / s_fct) for d in original_shape]
    scale_fcts = [o_d / s_d for o_d, s_d in zip(original_shape, scaled_shape)]
    if tile_shape:
        tile_shape_scaled = [
            round( d / sfct )  for d, sfct in zip(tile_shape, scale_fcts)
        ]
    else:
        tile_shape_scaled = None
    return Scaledshape(
        full = scaled_shape,
        tile = tile_shape_scaled,
        facts = scale_fcts
    )

# ------------------------------------------------------------------------------

def get_tiles_tab(
        slide,
        empty_mask,
        tile_shape,
        overlap = 0,
        thresh = 0.8,
        tiles_dir = None
    ):
    """
    Will create pandas DataFrames containing information (e.g., location)
    of all tiles in a slide for empty and for nonempty tiles. If 'tiles_dir'
    is set, tiles will be extracted.
    """
    # Get downsampling factors/sizes
    empty_shapes = get_scaled(
        original_shape =  slide.dimensions,
        scaled_shape = empty_mask.shape[1::-1], # Reverse order of shape here
        tile_shape = tile_shape
    )
    # Prepare for DataFrame
    columns = ['i_x', 'i_y', 'x_from', 'y_from', 'prop_empty', 'is_empty']
    if tiles_dir:
        columns += ['file']
        os.makedirs(tiles_dir, exist_ok=True)
    n_x = math.ceil(
        (slide.dimensions[0] + overlap) / (tile_shape[0] - overlap)
    )
    n_y = math.ceil(
        (slide.dimensions[1] + overlap) / (tile_shape[1] - overlap)
    )
    rows = []
    pbar = tqdm.tqdm(total=n_x*n_y)
    for i_x in range(n_x):
        for i_y in range(n_y):
            # Get froms on original slide
            x_from = -overlap + i_x * (tile_shape[0] - overlap)
            y_from = -overlap + i_y * (tile_shape[1] - overlap)
            # Scale down to empty mask
            x_from_mask = max(0, round(x_from / empty_shapes.facts[0]))
            y_from_mask = max(0, round(y_from / empty_shapes.facts[1]))
            x_to_mask = min(
                x_from_mask + empty_shapes.tile[0],
                empty_mask.shape[1]
            )
            y_to_mask = min(
                y_from_mask + empty_shapes.tile[1],
                empty_mask.shape[0]
            )
            # Compute empty proportion
            tile_mask = empty_mask[
                y_from_mask:y_to_mask, x_from_mask:x_to_mask
            ]
            prop_empty = np.sum(tile_mask) / tile_mask.size
            # Check if empty
            is_empty = prop_empty > thresh
            this_row = [i_x, i_y, x_from, y_from, prop_empty, is_empty]
            if tiles_dir and not is_empty:
                file = '{}{}_{}.png'.format(tiles_dir, i_x, i_y)
                tile = slide.read_region([x_from, y_from], 0, tile_shape)
                tile.convert('RGB').save(file)
                this_row += [os.path.basename(file)]
            elif tiles_dir:
                this_row += ['']
            rows += [this_row]
            pbar.update(1)
    tiles_tab = pd.DataFrame(rows, columns=columns).groupby('is_empty')
    empty_tiles = tiles_tab.get_group(True).drop('is_empty', axis=1)
    nonempty_tiles = tiles_tab.get_group(False).drop('is_empty', axis=1)
    pbar.close()
    return empty_tiles, nonempty_tiles

# ------------------------------------------------------------------------------

def get_tfdata(
        slide,
        tiles_tab,
        tile_shape,
        input_size,
        scale,
        batch_size,
        tiles_dir=None
    ):
    """Create tensorflow.data.Dataset containing tiles in a slide.
    
    Can read the tiles from already existing tile files if tiles_dir is set, or
    from the whole slide file if not. To create tile files see 'get_tiles_tab'. 
    """
    def preprocess(tile):
        tile = tf.image.resize(tile, [input_size, input_size])
        if scale:
          tile = (tile / 255.0)
        return tile
    if tiles_dir:
        def read_tile(file):
            tile = tf.io.read_file(tiles_dir + file)
            tile = tf.io.decode_image(tile, channels=3, expand_animations=False)
            return preprocess(tile)
        tfdata = tf.data.Dataset.from_tensor_slices(tiles_tab['file'])
        tfdata = tfdata.map(
            read_tile,
            num_parallel_calls=tf.data.experimental.AUTOTUNE
        )
    else:
        def TileGenerator():
            for loc in tiles_tab[['x_from', 'y_from']].itertuples():
                tile = slide.read_region([loc.x_from, loc.y_from], 0, tile_shape)
                tile = tf.convert_to_tensor(np.array(tile))[:,:,0:3]
                yield preprocess(tile)
        if scale:
            out_type = tf.float32
        else:
            out_type = tf.uint8
        tfdata = tf.data.Dataset.from_generator(
            generator=TileGenerator,
            output_signature=tf.TensorSpec(
                shape=[input_size, input_size, 3],
                dtype=out_type
            )
        )
    return tfdata.batch(batch_size).prefetch(tf.data.AUTOTUNE)

# ------------------------------------------------------------------------------

def tilewise_pred(tiles_tab, model, tfdata):
    """Predict a single class for each tile in tiles_tab.
    
    Returns tiles_tab with the predicted class ('class_int') and softmax score
    ('soft') and logits_mat (shape n_tiles, n_classes) with all logit values.
    """
    model.layers[-1].activation = tf.keras.activations.linear
    logits_mat = model.predict(tfdata, verbose=1, steps=tiles_tab.shape[0])
    tiles_tab['class_int'] = tf.math.argmax(logits_mat, axis=-1).numpy()
    softs_mat = tf.keras.activations.softmax(tf.convert_to_tensor(logits_mat))
    tiles_tab['soft'] = tf.math.reduce_max(softs_mat, axis=-1).numpy()
    return logits_mat, tiles_tab

# ------------------------------------------------------------------------------

def segmentation_pred(class_mask_shapes, n_tiles, model, tfdata):
    """Predict classes for each pixel in each tile (aka. segment the tile).
    
    Returns logits_mat with shape (n_tiles, class_mask_size, class_mask_size, 
    n_classes).
    """
    model.layers[-1].activation = tf.keras.activations.linear
    input = model.input
    x = model(input)
    model = tf.keras.Model(
        inputs=input,
        outputs=tf.keras.layers.Resizing(
            height = class_mask_shapes.tile[0],
            width = class_mask_shapes.tile[1]
        )(x)
    )
    return model.predict(tfdata, verbose=1, steps=n_tiles)

# ------------------------------------------------------------------------------

def get_class_mask(
        tiles_tab,
        logits_mat,
        slide_shape,
        tile_shape,
        max_size,
        soft_thresh
    ):
    """Create an integer mask for the slide with predicted class."""
    # Check if segmentation or tilewise
    is_segmentation = len(logits_mat.shape) > 2
    # Get shape of mask
    mask_shapes = get_scaled(
        original_shape =  slide_shape,
        max_size = max_size,
        tile_shape = tile_shape
    )
    # Create initial masks
    avg_logits = np.zeros(
        shape = mask_shapes.full[1::-1] + [logits_mat.shape[-1]],
        dtype = np.float32
    )
    n_preds = np.zeros(
        shape = mask_shapes.full[1::-1] + [1],
        dtype = np.int8
    )
    # Loop across tiles
    for row_tup in tiles_tab.reset_index().itertuples():
        x_from_mask = max(0, round(row_tup.x_from / mask_shapes.facts[0]))
        y_from_mask = max(0, round(row_tup.y_from / mask_shapes.facts[1]))
        x_to_mask = min(x_from_mask + mask_shapes.tile[0], avg_logits.shape[1])
        y_to_mask = min(y_from_mask + mask_shapes.tile[1], avg_logits.shape[0])
        # Update average:
        avg_logits[
            y_from_mask:y_to_mask,
            x_from_mask:x_to_mask,
            :
        ] *= n_preds[y_from_mask:y_to_mask, x_from_mask:x_to_mask]
        if is_segmentation:
            # With tiles at the border it can happen that
            # logits_mat[row_tup.Index, ] is larger than from-to.
            # Due to the way we tile the slide, either on the right or bottom
            # pixel are missing (or both)
            y_log_to = y_to_mask - y_from_mask
            x_log_to = x_to_mask - x_from_mask
            avg_logits[
                y_from_mask:y_to_mask,
                x_from_mask:x_to_mask,
                :
            ] += logits_mat[row_tup.Index, 0:y_log_to, 0:x_log_to]
        else:
            avg_logits[
                y_from_mask:y_to_mask,
                x_from_mask:x_to_mask,
                :
            ] += logits_mat[row_tup.Index, ]
        n_preds[y_from_mask:y_to_mask, x_from_mask:x_to_mask, :] += 1
        avg_logits[
            y_from_mask:y_to_mask,
            x_from_mask:x_to_mask,
            :
        ] /= n_preds[y_from_mask:y_to_mask, x_from_mask:x_to_mask, :]
    # We use softmax to make sure each value is in [0,1]
    softs = tf.keras.activations.softmax(tf.convert_to_tensor(avg_logits))
    # Multiply with 1.1 to make sure its biggest when true
    empty_dim = tf.cast(n_preds == 0, tf.float32) * 1.2
    class_mask = tf.concat([softs, empty_dim], -1)
    class_mask_soft = tf.math.reduce_max(class_mask, axis=-1).numpy()
    below_thresh_dim = tf.cast(class_mask_soft < soft_thresh, tf.float32) * 1.1
    below_thresh_dim = tf.expand_dims(below_thresh_dim, -1)
    class_mask = tf.concat([class_mask, below_thresh_dim], -1)
    class_mask_int = tf.math.argmax(class_mask, axis=-1)
    # 'int8' needed for 'Image.fromarray'
    return tf.cast(class_mask_int, tf.int8).numpy()

# ------------------------------------------------------------------------------

def get_thumb_colors(thumbnail_shape, class_mask, col_dict, show_dict, alpha):
    """Use an integer calss mask to produce a colored image or classes."""
    class_img = Image.fromarray(class_mask.astype(np.uint8), mode='P')
    class_img = class_img.resize(size=thumbnail_shape)
    col_pal = []
    for class_int in range(len(col_dict)):
        col_pal += list(ImageColor.getrgb(col_dict[class_int])) + [round(alpha * 255 * show_dict[class_int])]
    class_img.putpalette(col_pal, rawmode='RGBA')
    return class_img.convert(mode='RGBA')

# ------------------------------------------------------------------------------

def get_colored_thumb(thumb, thumb_cols):
    """Use an integer calss mask to produce a colored image or classes."""
    return (Image.alpha_composite(thumb, thumb_cols))

# ------------------------------------------------------------------------------

def get_class_polygons(class_mask, c_i, sfcts):
    """Turn class_mask into polygons."""
    contours, hierarchy = cv.findContours(
        image = (class_mask == c_i).astype(np.uint8),
        mode = cv.RETR_LIST,
        method = cv.CHAIN_APPROX_TC89_L1
    )
    polygons = {}
    if len(contours) > 0:
        for i_seg, seg_hierarchy in enumerate(np.squeeze(hierarchy, axis=0)):
            if contours[i_seg].shape[0] >= 3: # Is a valid polygon
                points = np.rint(
                    np.squeeze(contours[i_seg] * sfcts, 1)
                ).astype(np.int32).tolist()
                points += [points[0]] # Each linear ring must end where it started
                if seg_hierarchy[-1] < 0: # Is a parent
                    polygons.update({i_seg : [points]})
                else: # Is a child / hole
                    polygons[seg_hierarchy[-1]] += [points]
            else: # Is not a valid polygon
                print('Segment {} has only {} points and will be ignored'.format(
                    i_seg, contours[i_seg].shape[0]
                ))
    return [polygons[k] for k in polygons]

# ------------------------------------------------------------------------------

def get_geojson(class_mask, inv_class_dict, col_dict, show_dict, sfcts):
    """Turn a class_mask into GeoJSON fromat."""
    features = []
    class_ints = [k for k in show_dict if show_dict[k] == 1]
    for c_i in class_ints:
        polygons = get_class_polygons(
            class_mask = class_mask,
            c_i = c_i,
            sfcts = sfcts
        )
        if len(polygons) > 0:
            if len(polygons) == 1:
                geom = gj.Polygon(polygons[0])
            else:
                geom = gj.MultiPolygon(polygons)
            if c_i in inv_class_dict:
              c_name = inv_class_dict[c_i]
            elif (c_i == len(inv_class_dict)):
              c_name = "Empty"
            elif (c_i == len(inv_class_dict) + 1):
              c_name = "Below Threshold"
            else:
              raise Exception('Class int {} cannot be identified.'.format(c_i))
            feature = gj.Feature(
                geometry=geom,
                properties={
                    "object_type": "detection",
                    "name": c_name,
                    "color": colormap.hex2rgb(col_dict[c_i]),
                    "isLocked": True
                }
            )
            features += [feature]
    return gj.FeatureCollection(features)

# ------------------------------------------------------------------------------

def gen_col():
  """Generate a random  color in RGB."""
  rgb = list(np.random.randint(low=0, high=255, size=3))
  return colormap.rgb2hex(*rgb)

# ------------------------------------------------------------------------------

def get_colors(n_classes, seed=66124):
    """Generate colors for model classes."""
    np.random.seed(seed)
    col_dict = {c_i: gen_col() for c_i in range(n_classes)}
    col_dict.update({
        n_classes: '#f4f6f6', # Empty
        n_classes+1: '#000000', # Below Threshold
    })
    return col_dict
  
def get_shows(col_dict):
  show_dict = {k: 1 for k in col_dict}
  show_dict[len(col_dict) - 2] = 0
  return show_dict

# ------------------------------------------------------------------------------

def classify_ws(
        slide_path,
        tile_shape,
        model,
        overlap = 0,
        empty_mask_max_size = 1024,
        class_mask_max_size = 2048,
        scale = False,
        tiles_dir = None # Set to extract tiles
    ):
    """Classifies a whole-slide."""
    slide = osl.open_slide(slide_path)
    empty_mask = get_empty_mask(
        slide = slide,
        max_size = empty_mask_max_size
    )
    empty_tiles, nonempty_tiles = get_tiles_tab(
        slide = slide,
        empty_mask = empty_mask,
        tile_shape = tile_shape,
        overlap = overlap,
        tiles_dir = tiles_dir
    )
    tfdata = get_tfdata(
        slide = slide,
        tiles_tab = nonempty_tiles,
        tile_shape = tile_shape,
        input_size = model.input.shape[1],
        scale = scale,
        batch_size = 1,
        tiles_dir = tiles_dir
    )
    if len(model.output.shape) > 2:
        class_mask_shapes = get_scaled(
            original_shape =  slide.dimensions,
            max_size = class_mask_max_size,
            tile_shape = tile_shape
        )
        logits_mat = segmentation_pred(
            class_mask_shapes = class_mask_shapes,
            n_tiles = nonempty_tiles.shape[0],
            model = model,
            tfdata = tfdata
        )
    else:
        logits_mat, nonempty_tiles = tilewise_pred(
            tiles_tab = nonempty_tiles,
            model = model,
            tfdata = tfdata
        )
    return empty_tiles, nonempty_tiles, logits_mat

# ------------------------------------------------------------------------------

def summarize_class_mask(class_mask, inv_class_dict, meta_classes):
    n_pix_empty = np.sum(class_mask == len(inv_class_dict))
    n_pix_below_thresh = np.sum(class_mask == len(inv_class_dict) + 1)
    prop_dict = { c_i: int(np.sum(class_mask == c_i)) for c_i in inv_class_dict }
    n_pix_tumor = int(sum([prop_dict[c_i] for c_i in meta_classes['Tumor']]))
    n_pix_nontumor = int(sum([prop_dict[c_i] for c_i in meta_classes['Non-Tumor']]))
    prop_dict.update({
        len(inv_class_dict): n_pix_empty / class_mask.size,
        len(inv_class_dict)+1: n_pix_below_thresh / (class_mask.size - n_pix_empty),
        len(inv_class_dict)+2: n_pix_tumor / (class_mask.size - n_pix_empty),
        len(inv_class_dict)+3: n_pix_nontumor / (class_mask.size - n_pix_empty)
    })
    for m_c in meta_classes:
        meta_n_pix = int(sum([prop_dict[c_i] for c_i in meta_classes[m_c]]))
        for c_i in meta_classes[m_c]:
            if meta_n_pix > 0:
                prop_dict[c_i] /= meta_n_pix
            else:
                prop_dict[c_i] = 0.0
    prop_dict = {
        c_i: prop_dict[c_i] for c_i in sorted(
            prop_dict,
            key = prop_dict.get,
            reverse = True
        )
    }
    return prop_dict

# ------------------------------------------------------------------------------

def get_col_box(cl_int, col, show):
    """Create a <div> for a color."""
    general_style = 'width: 100%; min-width: 28px; height: 15px; line-height: 15px; border: 1px solid; text-align: center;'
    if show > 0:
      return '<div class="huebox" id="huebox{}" style="background-color:{}; opacity:{}; {}">'.format(
        cl_int, col, show, general_style
      )
    else:
      return '<div class="huebox" id="huebox{}" style="{}">Omitted</div>'.format(
        cl_int, general_style
      )

# ------------------------------------------------------------------------------

def get_tab_row_head(title, prop=None):
    row_head =  '<tr>\n'
    row_head += '   <td><b>{}</b></td>\n'.format(title)
    if prop:
        row_head += '   <td><b>{:.2f}</b></td>\n'.format(prop * 100)
        n_cols = 1
    else:
        n_cols = 2
    row_head += '   <td colspan="{}"><hr style="margin:0px; border-top: 1px dashed black"></td>\n'.format(n_cols)
    row_head += '</tr>'
    return row_head

# ------------------------------------------------------------------------------

def get_tab_row(cl_int, cl_str, prop, col, show):
    row =  '<tr>\n'
    row += '   <td><i>{}</i></td>\n'.format(cl_str)
    if prop == 0:
        row += '   <td>{}</td>\n'.format('-')
    else:
        row += '   <td>{:.2f}</td>\n'.format(prop * 100)
    row += '   <td>{}</td>\n'.format(get_col_box(cl_int, col, show))
    row += '</tr>'
    return row

# ------------------------------------------------------------------------------

def get_class_mask_html_tab(
        prop_dict,
        col_dict,
        show_dict,
        meta_classes,
        inv_class_dict
    ):
    # Table Header
    html_tab =  '<table style="width:100%;">\n'
    html_tab += '   <thead>\n'
    html_tab += '       <tr style="text-align: left;">\n'
    html_tab += '           <th>Class</th>\n'
    html_tab += '           <th>%</th>\n'
    html_tab += '           <th>Hue</th>\n'
    html_tab += '       </tr>\n'
    html_tab += '       <tr>\n'
    html_tab += '           <td colspan="3"><hr style="margin:0px; border-top: 1px solid black"></td>\n'
    html_tab += '       </tr>\n'
    html_tab += '   </thead>\n'
    html_tab += '   <tbody>\n'
    # Tumor tiles
    html_tab += get_tab_row_head('Tumor', prop_dict[len(inv_class_dict)+2])
    html_tab += ''.join([
         get_tab_row(
            cl_int = c_i,
            cl_str = inv_class_dict[c_i],
            prop = prop_dict[c_i],
            col = col_dict[c_i],
            show = show_dict[c_i]
        )
        for c_i in prop_dict
        if c_i in meta_classes['Tumor']
    ])
    # Non-Tumor tiles
    html_tab += get_tab_row_head('Non-Tumor', prop_dict[len(inv_class_dict)+3])
    html_tab += ''.join([
         get_tab_row(
            cl_int = c_i,
            cl_str = inv_class_dict[c_i],
            prop = prop_dict[c_i],
            col = col_dict[c_i],
            show = show_dict[c_i]
        )
        for c_i in prop_dict
        if c_i in meta_classes['Non-Tumor']
    ])
    # Others
    html_tab += get_tab_row_head('Others')
    html_tab += ''.join([
         get_tab_row(
            cl_int = c_i,
            cl_str = 'Empty' if c_i == len(inv_class_dict) else 'Below Threshold',
            prop = prop_dict[c_i],
            col = col_dict[c_i],
            show = show_dict[c_i]
        )
        for c_i in prop_dict
        if c_i in [len(inv_class_dict), len(inv_class_dict) + 1]
    ])
    html_tab += '   </tbody>\n'
    html_tab +=  '</table>'
    return html_tab

# ------------------------------------------------------------------------------

def gen_report(report_file, title, colored_thumb, html_tab):
    with open(report_file, 'w') as rf:
        rf.write('<div style="display:flex; justify-content:center;">\n')
        rf.write('<div style="max-width:800px; min-width:500px;">\n')
        rf.write('<h1>Filename: {}</h1>\n'.format(title))
        rf.write(html_tab)
        rf.write('<hr style="margin-bottom:10px">')
        img_file = tempfile.NamedTemporaryFile(suffix='.png')
        colored_thumb.save(img_file)
        with open(img_file.name, 'rb') as temp_file:
            rf.write(
                '<img src="data:image/png;base64, {}" width="100%">\n'.format(
                    base64.b64encode(temp_file.read()).decode("utf-8")
                )
            )
        rf.write('</div>\n')
        rf.write('</div>\n')

# ------------------------------------------------------------------------------

def get_pixel_shape(slide, tile_shape_um):
    mpp = (
        float(slide.properties['openslide.mpp-x']),
        float(slide.properties['openslide.mpp-y'])
    )
    return [round(s / f) for s, f in zip(tile_shape_um, mpp)]

# ------------------------------------------------------------------------------

def get_dict(dict_path):
    with open(dict_path) as json_file:
        read_dict = json.load(json_file)
    return read_dict

# ------------------------------------------------------------------------------

def get_inv_class_dict(class_dict_path):
    class_dict = get_dict(class_dict_path)
    inv_class_dict = {class_dict[k]:k for k in class_dict}
    return inv_class_dict
  
# ------------------------------------------------------------------------------

def load_full_model(model_file):
    tf.keras.backend.clear_session()
    return tf.keras.models.load_model(model_file, compile=False)
  
# ------------------------------------------------------------------------------

def load_model_from_weights(weights_file, app, input_shape, n_classes):
    tf.keras.backend.clear_session()
    model_fun = getattr(tf.keras.applications, app)
    return model_fun(
      weights = weights_file,
      input_shape = input_shape,
      classes = n_classes
    )
  
# ------------------------------------------------------------------------------

def write_gj(gj_anno, anno_file):
  with open(anno_file, 'w') as gj_f:
    gj.dump(gj_anno, gj_f)
    
# ------------------------------------------------------------------------------    

def get_shown_cl(show_dict):
  return [k for k in show_dict if show_dict[k] == 1]

# ------------------------------------------------------------------------------    

def to_dict_w_int_keys(r_dict):
  """
  R will convert intenger keys to character. This functions converts them back
  to integers.
  """
  return {int(k): r_dict[k]  for k in r_dict}
  
# ------------------------------------------------------------------------------

def summarize_logits_mask(logits_mask, inv_class_dict):
    pred_mask = tf.convert_to_tensor(logits_mask)
    pred_mask = tf.keras.activations.softmax(pred_mask)
    pred_mask = tf.math.argmax(pred_mask, axis=-1).numpy()
    prop_rows = [
      [inv_class_dict[c_i], int(np.sum(pred_mask == c_i))/pred_mask.size] 
      for c_i in inv_class_dict 
    ]
    return pd.DataFrame(prop_rows, columns=['class_str', 'prop'])

# ------------------------------------------------------------------------------

def get_tile(slide, froms, tile_shape):
  tile = slide.read_region(froms, 0, tile_shape).convert('RGB')
  return np.array(tile)

# ------------------------------------------------------------------------------

def get_tile_mask_predicted(model, input_dims, scale, tile, col_dict):
  orig_shape = tile.shape[0:2]
  tile = tf.image.resize(tile, input_dims)
  if scale:
    tile = (tile / 255.0)
  tile = tf.expand_dims(tile, axis=0)
  pred = model(tile)
  pred = tf.squeeze(pred)
  pred = tf.keras.activations.softmax(pred)
  pred = tf.math.argmax(pred, axis=-1).numpy()
  pred_img = Image.fromarray(pred.astype(np.uint8), mode='P')
  pred_img = pred_img.resize(orig_shape) 
  col_pal = []
  for class_int in range(len(col_dict)):
      col_pal += list(ImageColor.getrgb(col_dict[class_int]))
  pred_img.putpalette(col_pal, rawmode='RGB')
  return np.array(pred_img.convert('RGB'))

def merge_with_mask(im_tile, im_mask, alpha):
  if not im_mask is None:
    img_blend = Image.blend(
      im1 = Image.fromarray(im_tile), 
      im2 = Image.fromarray(im_mask), 
      alpha = alpha
    )
    img_blend = np.array(img_blend)
  else:
    img_blend = im_tile
  return np.swapaxes(img_blend, 0, 1) / 255



