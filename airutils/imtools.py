"""
Copyright 2018-2021 Accenture

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
"""

'''
Created on Fri Nov 23 11:09:07 2018

Functions for image processing
'''

import numpy as np
import cv2
import os
import io
import matplotlib.pyplot as plt
from scipy import signal
from itertools import compress
from PIL import Image, ImageDraw, ImageFont, ImageColor


def equalize_hist(img):
    # create_display images
    img_yuv = cv2.cvtColor(img, cv2.COLOR_BGR2YUV)
    # equalize the histogram of the Y channel
    img_yuv[:, :, 0] = cv2.equalizeHist(img_yuv[:, :, 0])
    # convert the YUV image back to RGB format
    img_output = cv2.cvtColor(img_yuv, cv2.COLOR_YUV2BGR)
    return img_output


def gray2color(img):
    img = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
    return img


def color2gray(img, mode=None):
    if mode is None:
        img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    elif mode == "hue":
        img = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)[:, :, 0]
    elif mode == "sat":
        img = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)[:, :, 1]
    elif mode == "hue+sat":
        img_hue = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)[:, :, 0].astype(float)
        img_sat = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)[:, :, 1].astype(float)
        img = normalize(img_hue + img_sat)
    else:
        raise ValueError("Invalid mode!")
    return img


def normalize(im, data_type='float'):
    img_norm = im.astype('float')
    if type(img_norm) == np.ndarray:
        img_vals = img_norm.flatten()
    elif type(img_norm) == np.ma.core.MaskedArray:
        img_vals = img_norm.compressed()
    #
    img_norm = img_norm - np.min(img_vals)
    if np.max(img_vals) > 0:
        img_norm = img_norm / np.max(img_vals)
    if data_type == 'uint8':
        img_norm = img_norm * 255
    return img_norm.astype(data_type)


def normalize_list(source, data_type='float'):
    if type(source) == list:
        images = source.copy()
        for i in range(len(images)):
            images[i] = normalize(images[i], data_type)
        return images
    else:
        return normalize(source, data_type)


def image_to_array(image_bytes) -> np.ndarray:
    nparr = np.fromstring(image_bytes, np.uint8)
    return cv2.imdecode(nparr, 0)


def overlay_bounding_boxes(img, bboxes):
    # Overlay bounding boxes on an image
    (height, width) = img.shape
    # Get the color map by name:
    cmap = plt.get_cmap('viridis')
    # Apply the colormap like a function to any array:
    img_det = cmap(img)
    for bbox in bboxes:
        # Draw rectangle
        bbox_scaled = bbox.get_scaled(width, height)
        img_det = cv2.rectangle(img_det,
                                bbox_scaled.left_top_point.to_int_tuple(),
                                bbox_scaled.right_bottom_point.to_int_tuple(),
                                (1, 0, 0, 0), 4)
    img_det = normalize(img_det, 'uint8')
    return img_det


def plot_tiled_imgs(image_list, display=True, normalize_imgs=False, title="", figsize=(6, 4), Ncol=None):
    Nimg = len(image_list)
    tiles = [None] * Nimg
    for i in range(len(image_list)):
        if isinstance(image_list[i], np.ma.MaskedArray):
            image_i = copy_arr(image_list[i])
            image_i = image_i.filled(0)
        else:
            image_i = image_list[i].copy()
        # normalizing and converting to RGB image
        norm_func = normalize if normalize_imgs else lambda x, dtype: x
        if np.ndim(image_i) == 2:
            tiles[i] = np.stack((norm_func(image_i, 'uint8'),) * 3, axis=-1)
        elif np.ndim(image_i) == 3:
            tiles[i] = norm_func(image_i, 'uint8')
        # Adding borders
        tiles[i] = cv2.copyMakeBorder(
            tiles[i], 2, 2, 2, 2, cv2.BORDER_CONSTANT, value=[255, 255, 255])
    # Composing the image
    if not Ncol:
        if Nimg <= 4:
            Ncol = 2
        elif Nimg <= 9:
            Ncol = 3
        else:
            Ncol = 4

    Nrow = int(np.ceil(Nimg / Ncol))
    cols = []
    for i in range(Nrow):
        row = []
        for j in range(Ncol):
            tile_idx = i * Ncol + j
            if tile_idx < Nimg:
                row.append(tiles[tile_idx])
            else:
                row.append(255 * np.ones_like(tiles[0]))
        cols.append(np.concatenate(row, axis=1))
    img_comb = np.concatenate(cols, axis=0)
    # Plotting the image
    if display:
        plt.figure(figsize=(Ncol * figsize[0], Nrow * figsize[1]))
        plt.imshow(img_comb)
        plt.title(title)
        plt.axis('off')
        plt.show()
    return img_comb


def crop(img, p1, p2):
    '''
    Crops rectangle out of an input image ``img`` specified by two
    (x, y) plane points ``p1`` and ``p2``

    Returns the cropped image (``np.ndarray``) of size ``(y2 - y1, x2 - x1)`` 
    if cropping outside of the image borders doesn't occur
    '''
    ymax, xmax = img.shape[:2]
    x = np.clip(p1[0], 0, xmax)
    y = np.clip(p1[1], 0, ymax)
    w = np.clip(p2[0] - x, 0, xmax)
    h = np.clip(p2[1] - y, 0, ymax)
    return img[y:y + h, x:x + w].copy()


def crop_opaque(img, crop_padding=1):
    ''' Crops out transparent borders of the image '''
    opaque_mask = np.where(img[..., 3] > 0, 255, 0).astype(np.uint8)
    rect = cv2.boundingRect(opaque_mask)
    p1 = np.array(rect[:2])
    p2 = p1 + np.array(rect[2:])
    cropped_img = crop(img, p1+crop_padding, p2-crop_padding)
    return cropped_img


def trim(img, trim_color=(0,0,0), trim_margin=0):
    trim_mask = np.any(img != trim_color, axis=-1).astype(np.uint8) * 255
    rect = cv2.boundingRect(trim_mask)
    p1 = np.array(rect[:2])
    p2 = p1 + np.array(rect[2:])
    return crop(img, p1+trim_margin, p2-trim_margin)


def slice_frame(image, slice_height_percent=20):
    ydim, xdim = image.shape[:2]
    ymid = ydim // 2
    h_coeff = slice_height_percent / 200
    y_offset = ymid - int(h_coeff*ydim)
    p1 = (0, ymid - int(h_coeff*ydim))
    p2 = (xdim, ymid + int(h_coeff*ydim))
    image = crop(image, p1, p2)
    return image, y_offset


def show_image(img, title = '', bgr_to_rgb=True):
    img_norm = normalize(img, data_type='uint8')
    if bgr_to_rgb:
        img_norm = cv2.cvtColor(img_norm, cv2.COLOR_BGR2RGB)
    # fig = plt.gcf()
    # fig.set_size_inches(*size)
    plt.clf()
    plt.imshow(img_norm)
    plt.title(title)
    plt.axis('off')
    plt.pause(0.001)


def filter_circular_contours(contours, circularity_thres=0.7, ball_rad=30, rad_tol=0.2):
    '''
    Returns all circular enough contours that are of certain size range based on parameter constraints.
    '''
    accepted_contours = []
    centers = []
    for blob in contours:
        area = cv2.contourArea(blob)
        # NOTE: this canbe improved by fittinga a circle to the blob
        perimeter = cv2.arcLength(blob, True)
        if perimeter == 0:
            continue
        radius_est = perimeter / (2 * np.pi)
        circularity = area / (np.pi * radius_est**2)
        ((x, y), radius_enc) = cv2.minEnclosingCircle(blob)
        # Find center
        M = cv2.moments(blob)
        cX = int(M["m10"] / M["m00"])
        cY = int(M["m01"] / M["m00"])
        # Check circularity
        if circularity > circularity_thres and \
                abs(radius_est - ball_rad) < rad_tol * ball_rad and \
                abs(radius_enc - ball_rad) < rad_tol * ball_rad:
            print("Accepted blob with area: {}, circularity: {}".format(
                area, circularity))
            accepted_contours.append(blob)
            centers.append((cX, cY))
    return accepted_contours, centers


def get_global_displacement_between_two_frames(im_from, im_to, refine_iters=1):
    # convert images to uint8 for cv2 algorithm support
    img_from_gray = color2gray(im_from).astype(np.uint8)
    img_to_gray = color2gray(im_to).astype(np.uint8)
    # Detect feature points in previous frame
    pts_from = cv2.goodFeaturesToTrack(img_from_gray,
                                    maxCorners=200,
                                    qualityLevel=0.05,
                                    minDistance=80,
                                    blockSize=31)
    # Calculate optical flow (i.e. track feature points)
    pts_to, status, _ = cv2.calcOpticalFlowPyrLK(
        img_from_gray, img_to_gray, pts_from, None)
    # Filter only valid points
    idx = np.where(status == 1)[0]
    pts_to = pts_to[idx]
    pts_from = pts_from[idx]
    # Find transformation matrix
    TM, _ = cv2.estimateAffine2D(
        pts_from, pts_to, refineIters=refine_iters)
    global_displacement = TM[:,2]
    return global_displacement


def get_motion_field_between_two_frames(im_from, im_to, filtered_from, filtered_to, prev_flow=None, calculate_global_displacement=True, smoothing_kernel=None):
    if prev_flow is not None:
        flags = cv2.OPTFLOW_FARNEBACK_GAUSSIAN | cv2.OPTFLOW_USE_INITIAL_FLOW
    else:
        flags = cv2.OPTFLOW_FARNEBACK_GAUSSIAN
    motion_field = cv2.calcOpticalFlowFarneback(filtered_from, filtered_to,
                                                flow=prev_flow,
                                                pyr_scale=0.5,
                                                levels=5,
                                                winsize=47,
                                                iterations=3,
                                                poly_n=7,
                                                poly_sigma=1.5,
                                                flags=flags)
                                            
    global_displacement = None
    if calculate_global_displacement:
        global_displacement = get_global_displacement_between_two_frames(im_from, im_to)

    if smoothing_kernel is not None:
        motion_field = cv2.boxFilter(motion_field, 
                                     ddepth=-1,
                                     ksize=smoothing_kernel,
                                     normalize=True)

    return motion_field, global_displacement


def get_motion_mask(motion_field, global_displacement):
    mag, ang = cv2.cartToPolar(motion_field[..., 0], motion_field[..., 1])
    _, g_ang = cv2.cartToPolar(np.array(global_displacement[0]), np.array(global_displacement[1]))
    score_matrix = normalize(mag) ** 2 * np.abs(np.sin((ang - g_ang) / 2))
    return score_matrix


def fig2img(fig, res, dpi=180):
    with io.BytesIO() as buf:
        fig.savefig(buf, format="png", dpi=180)
        buf.seek(0)
        img_arr = np.frombuffer(buf.getvalue(), dtype=np.uint8)
        # buf.close()
        img = cv2.imdecode(img_arr, 1)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        y, x = res
        img = cv2.resize(img, (x, y))
        return img


def get_flow_map(flow, im=None, density=100, scale=5, display=False, color="red"):
    """
    Display a visualisation of optical flow (possibly on top of an image).

    Credits: https://stackoverflow.com/a/38277010
    """
    ydim, xdim = np.shape(flow)[:2]
    aspect_ratio = int(xdim / ydim)

    # Resolve arrow color
    try:
        color = ImageColor.getrgb(color)[::-1]
    except ValueError:
        color = (0, 0, 255)

    # create a blank flowmap, on which lines will be drawn.
    if im is None:
        flowmap = np.zeros((ydim, xdim, 3), dtype="uint8")
    else:
        flowmap = im.copy()
    for X1 in np.linspace(density, xdim-density, aspect_ratio * density, dtype=int):
        for Y1 in np.linspace(density, ydim-density, density, dtype=int):
            X2 = int(X1 + scale*flow[Y1, X1, 0])
            Y2 = int(Y1 + scale*flow[Y1, X1, 1])
            X2 = np.clip(X2, 0, xdim)
            Y2 = np.clip(Y2, 0, ydim)
            #add all the lines to the flowmap
            flowmap = cv2.arrowedLine(flowmap, (X1,Y1), (X2, Y2), color, 5)
    if display:
        show_image(flowmap, "FlowMap")
    return flowmap


def plot_histogram(arr, bins=100, valrange=None):
    if isinstance(arr, np.ma.MaskedArray):
        arr = arr[arr.mask == False]
    if valrange is None:
        valrange = (int(np.floor(arr.min())), int(np.ceil(arr.max())))
    plt.clf()
    plt.hist(arr.ravel(), bins, valrange)
    plt.xlabel("value")
    plt.ylabel("frequency")
    plt.show()


def draw_bounding_boxes(bg_input_img, bin_mask, area_thres=0, color="blue", margin=10, line_width=10):
    bg_img = bg_input_img.copy()
    contours, _ = cv2.findContours(bin_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)

    try:
        color = ImageColor.getrgb(color)[::-1]
    except ValueError:
        color = (0, 0, 255)

    ymax, xmax = bg_img.shape[:2]

    for blob in contours:
        area = cv2.contourArea(blob)
        if area > area_thres:
            x, y, w, h = cv2.boundingRect(blob)
            x, y = max(x - margin, 0), max(y - margin, 0)
            w, h = min(w + 2*margin, xmax), min(h + 2*margin, ymax)
            cv2.rectangle(bg_img, (x,y), (x+w,y+h), color, line_width)
    
    return bg_img


def overlay_text(im, text, pos, color="white", fontsize=3, line_width=10):
    # Resolve text color
    im_copy = im.copy()
    try:
        color = ImageColor.getrgb(color)[::-1]
    except ValueError:
        color = (0, 0, 255)
    font = cv2.FONT_HERSHEY_SIMPLEX
    cv2.putText(im_copy, text, pos, font, fontsize, color, line_width, cv2.LINE_AA)
    return im_copy


def confidence_ellipse(cov_matrix, center, ax, n_std=3.0, facecolor='none', **kwargs):
    """
    Create a plot of the covariance confidence ellipse of *x* and *y*.

    Parameters
    ----------
    cov_matrix : array-like, shape (n,n)
        Covariance matrix of 2D data.

    ax : matplotlib.axes.Axes
        The axes object to draw the ellipse into.

    n_std : float
        The number of standard deviations to determine the ellipse's radiuses.

    Returns
    -------
    matplotlib.patches.Ellipse

    Other parameters
    ----------------
    kwargs : `~matplotlib.patches.Patch` properties
    """
    pearson = cov_matrix[0, 1]/np.sqrt(cov_matrix[0, 0] * cov_matrix[1, 1])
    # Using a special case to obtain the eigenvalues of this
    # two-dimensionl dataset.
    ell_radius_x = np.sqrt(1 + pearson)
    ell_radius_y = np.sqrt(1 - pearson)
    ellipse = Ellipse((0, 0),
        width=ell_radius_x * 2,
        height=ell_radius_y * 2,
        facecolor=facecolor,
        **kwargs)

    # Calculating the stdandard deviation of x from
    # the squareroot of the variance and multiplying
    # with the given number of standard deviations.
    scale_x = np.sqrt(cov_matrix[0, 0]) * n_std
    # calculating the stdandard deviation of y ...
    scale_y = np.sqrt(cov_matrix[1, 1]) * n_std

    transf = transforms.Affine2D() \
        .rotate_deg(45) \
        .scale(scale_x, scale_y) \
        .translate(center[0], center[1])

    ellipse.set_transform(transf + ax.transData)
    return ax.add_patch(ellipse)


def get_contour_mean_speeds(contours, motion_field):
    mean_speeds = []
    for contour in contours:
        contour_speeds = []
        points = contour.squeeze()
        for i in range(points.shape[0]):
            x, y = points[i, :]
            contour_speeds.append(motion_field[y, x, :])
        mean_speeds.append(np.mean(contour_speeds, axis=0))
    return mean_speeds


def plot_detection_timeseries(detection_frequencies, filename=None, mode="lineplot", num_frames=1000, figsize=(22, 1), 
                                    smoothing=False, crop_image=True):

    x = detection_frequencies[:,0]
    y = detection_frequencies[:,1]
    if mode == "barplot":
        if smoothing:
            from scipy.signal import savgol_filter
            y = savgol_filter(y, 5, 3)
        plt.figure(figsize=figsize)
        plt.bar(x, y,
                color="magenta",
                width=0.8/num_frames,
                edgecolor="black",
                align="edge",
                linewidth=1)
    elif mode == "lineplot":
        from scipy.interpolate import interp1d
        fig, ax = plt.subplots(figsize=figsize)
        # p = np.polyfit(x, y, 3)
        # f = interp1d(x, y, kind="cubic")
        f = interp1d(x, y, kind="quadratic")
        x = np.linspace(x[0], x[-1], num=min(num_frames, 10000), endpoint=True)
        y = f(x)
        ax.fill_between(x, 0, y, facecolor='magenta')
        plt.plot(x, y,
                color="black",
                linewidth=2)
    if filename:
        assert filename.endswith(".png"), "Only png file format supported!"
        plt.axis("off")
        print(f"Saving timeseries to {filename}...")
        plt.savefig(filename, transparent=True, bbox_inches="tight", pad_inches=0)
        if crop_image:
            img = cv2.imread(filename, cv2.IMREAD_UNCHANGED)
            cv2.imwrite(filename, crop_opaque(img))
    else:
        plt.xlabel("Time")
        plt.ylabel("Frequency")
        plt.show()


def resolve_color(color):
    if isinstance(color, str):
        try:
            color = np.array(ImageColor.getrgb(color)[::-1], dtype=np.uint8)
        except ValueError:
            color = np.array((0,) * 3, dtype=np.uint8)
    else:
        color = np.array(color, dtype=np.uint8)
    return color


def grid_dim_to_tile_size(im, grid_dim, overlap):
    if grid_dim <= 1:
        return im.shape[:2]
    return (im.shape[0] // grid_dim + overlap, im.shape[1] // grid_dim + overlap)  


def create_overlapping_tiled_mosaic(im, tile_size=None, overlap=0, fill_color="black", no_fill=False):
    '''
    Slices the input image into a subtiles of size specified by ``tile_size``
    which are overlapping each other by ``overlap`` pixels. 
    
    Fill mode used for tiles that go out of the original image is constant color
    given by ``fill_color``.

    #### Args:
    - im (np.ndarray) : input image to be sliced
    - tile_size (tuple) : dimensions of an individual tile (y, x)
    - overlap (int) : number of pixels for the width of the overlapping part between tiles
    - fill_color (str | tuple) : String or RGB tuple specifying the color to use for filling
    - no_fill (bool) : Changes the behaviour of the algorithm quite drastically, <br>
    if ``True`` it tries to center the tiles so that all tiles have equal borders that <br>
    can be clipped while maintaining constant tile size (only works well with tile_size=None)

    #### Returns: 
    - tiles (list) : list of tiles (np.ndarray)
    - offsets (list) : list of offsets (tuple) for each tile from original image origin (e.g. (0,0) for first tile)
    '''

    if tile_size is None:
        # by default, split image into four tiles
        tile_size = grid_dim_to_tile_size(im, 2, overlap)
        # (im.shape[0] // 2 + overlap, im.shape[1] // 2 + overlap)

    if tile_size[0] >= im.shape[0] or tile_size[1] >= im.shape[1]:
        # tile size is bigger than original image --> do nothing
        return [im], [np.zeros(2)]
    
    color = resolve_color(fill_color)
    
    tiles = []
    offsets = []

    tile_dims = np.array(tile_size[::-1])

    y_max, x_max = im.shape[:2]

    if no_fill:
        x_offsets = np.arange(-overlap // 2, x_max - overlap // 2, tile_size[1] - overlap)
        y_offsets = np.arange(-overlap // 2, y_max - overlap // 2, tile_size[0] - overlap)
    else:
        x_offsets = np.arange(0, x_max, tile_size[1] - overlap)
        y_offsets = np.arange(0, y_max, tile_size[0] - overlap)

    # get rid of degenerate left over tiles caused by odd grid dimensions, rounding errors etc.
    if abs(x_offsets[-1] - im.shape[1]) < tile_size[1] - overlap:
        x_offsets = np.delete(x_offsets, -1)
    if abs(y_offsets[-1] - im.shape[0]) < tile_size[0] - overlap:
        y_offsets = np.delete(y_offsets, -1)

    # cartesian product of the above two ranges
    # credit: https://stackoverflow.com/a/11144716
    offsets = np.transpose([np.tile(x_offsets, len(y_offsets)), 
                            np.repeat(y_offsets, len(x_offsets))])

    if no_fill:
        new_offsets = np.zeros_like(offsets)
        for i, offset in enumerate(offsets):
            tile = crop(im, offset, offset + tile_dims)
            tiles.append(tile)
            new_offsets[i, :] = np.clip(offset, 0, None)[::-1]
        return tiles, new_offsets
    else:
        for offset in offsets:
            tile = np.expand_dims(np.ones(tile_size, dtype=np.uint8), 2) * color
            cropped = crop(im, offset, offset + tile_dims)
            crop_y, crop_x = cropped.shape[:2]
            tile[:crop_y, :crop_x] = cropped
            tiles.append(tile)

        # swap offset columns so they're consistent with image coordinates (y, x)
        return tiles, offsets[:, [1, 0]]


def merge_overlapping_tiled_mosaic(mosaic, offsets, overlap=0, fill_color="black", no_fill=False):
    ''' 
    Reverse operation of ``create_overlapping_tiled_mosaic``, should be used with
    same parameters as when creating the mosaic.
    '''

    if len(mosaic) == 1:
        # trivial mosaic of only one tile
        return mosaic[0]

    tiley, tilex = mosaic[0].shape[:2]
    color = resolve_color(fill_color)
    offsets = np.array(offsets)
    if no_fill:
        y_steps, x_steps = 0, 0
    else:
        y_steps = len(np.unique(offsets[:,0])) - 1
        x_steps = len(np.unique(offsets[:,1])) - 1
    ymax, xmax = np.max(offsets, axis=0)
    merged_img = np.zeros((ymax+tiley - y_steps*overlap, xmax+tilex - x_steps*overlap, 3), dtype=np.uint8)
    for tile, offset in zip(mosaic, offsets):
        y_off, x_off = offset # np.clip(offset - (overlap // 2 if no_fill else 0), 0, None)
        tile = trim(tile, color) # remove fill
        trimmed_y_dim, trimmed_x_dim = tile.shape[:2]
        merged_img[y_off:y_off+trimmed_y_dim, x_off:x_off+trimmed_x_dim, :] = tile

    return merged_img


def create_grid_plot(images, captions, shape=None, show=False, bgr_to_rgb=True, figscale=18):
    '''
    Creates a grid plot (subplots) from lists of `ìmages`` and ``captions``.
    If ``shape`` is not provided assumes square grid. If ``show``is ``True``,
    this displays the resulting grid plot.

    The ``images`` and ``captions`` are piled from left to right and 
    from top to down.

    #### Returns:
    - grid_plot (matplotlib.figure.Figure) : The grid plot figure
    '''
    N = len(images)
    if shape is None:
        shape = (int(np.sqrt(N)), int(np.sqrt(N)))

    assert sum(shape) >= 4, f"Invalid shape {shape}: At least 4 images needed to create a grid plot!"

    ydim, xdim = images[0].shape[:2]
    aspect_ratio = xdim / ydim

    grid_plot, axs = plt.subplots(*shape, figsize=(int(figscale * aspect_ratio), figscale))
    plt.subplots_adjust(wspace=0.0, hspace=0.25)

    TITLE_Y_OFFSET = -0.12

    i = 0
    for x in range(shape[0]):
        for y in range(shape[1]):
            axs[x, y].get_xaxis().set_visible(False)
            axs[x, y].get_yaxis().set_visible(False)
            if bgr_to_rgb:
                axs[x, y].imshow(cv2.cvtColor(images[i], cv2.COLOR_BGR2RGB))
            else:
                axs[x, y].imshow(images[i])
            y_off = TITLE_Y_OFFSET if "\n" not in captions[i] else 1.65*TITLE_Y_OFFSET
            axs[x, y].set_title(r"$\bf{(" + chr(97 + i) + r")}$ " + captions[i], y=y_off, fontsize=28)
            i += 1
    
    if show:
        plt.show()
    return grid_plot


def draw_square_grid(im, spacing=100, color="black", alpha=0.5, thickness=2):
    y_max, x_max = im.shape[:2]
    x_offsets = np.arange(spacing, x_max, spacing)
    y_offsets = np.arange(spacing, y_max, spacing)

    bg_img = im.copy()
    grid_img = im.copy()

    # Resolve grid color
    try:
        color = ImageColor.getrgb(color)[::-1]
    except ValueError:
        color = (255, 255, 255)
    
    for y in y_offsets:
        for x in x_offsets:
            grid_img = cv2.line(grid_img, (x, 0), (x, y_max), color, thickness)
            grid_img = cv2.line(grid_img, (0, y), (x_max, y), color, thickness)

    return cv2.addWeighted(grid_img, alpha, bg_img, 1 - alpha, 0)
    


''' AABB (axis aligned bounding box) operations '''

def aabb_dims(aabbs, dim):
    return aabbs[:, dim+2] - aabbs[:, dim]


def aabb_centroids(aabbs):
    x_means = (aabbs[:,2] + aabbs[:,0]) / 2
    y_means = (aabbs[:,3] + aabbs[:,1]) / 2
    return np.concatenate((x_means[:, None], y_means[:, None]), axis=1)


def max_aabb_area(aabbs):
    x_dims = aabb_dims(aabbs, 0)
    y_dims = aabb_dims(aabbs, 1)
    return np.max(x_dims, 0) * np.max(y_dims, 0)


def compute_aabb_area(aabb):
    return abs(aabb[2] - aabb[0]) * abs(aabb[3] - aabb[1])


def get_aabb_max_axis(aabb):
    ''' split heuristic to obtain good subdivision in minimal steps '''
    return np.argmax([abs(aabb[2] - aabb[0]), abs(aabb[3] - aabb[1])])


def enclosing_aabb(aabbs):
    return np.array([
        np.min(aabbs[:,0]),
        np.min(aabbs[:,1]),
        np.max(aabbs[:,2]),
        np.max(aabbs[:,3]),
    ])


if __name__ == "__main__":
    ''' test something here '''

    # bbox1 = np.array([100,200,200,300])
    # bbox2 = np.array([1000,2000,2000,3000])

    # bboxes1 = np.tile(bbox1, (100, 1)).astype(float)
    # bboxes2 = np.tile(bbox2, (100, 1)).astype(float)

    # # p1s = np.random.randint(100, 500, (100, 2)).astype(float)
    # # p2s = np.random.randint(1000, 1500, (100, 2)).astype(float)
    # boxes = np.concatenate((bboxes1, bboxes2), axis=0) #[:, [0,2,1,3]]
    # merged_boxes = merge_close_detections(boxes)

    # print(merged_boxes)

    # test_im = np.ones((3000,4000,3), dtype=np.uint8)
    import time
    GRID_DIM = 1
    test_im = cv2.imread("/Users/pasi.pyrro/Documents/video_analytics/rescue-drone-tool/data/datasets/heridal_keras_retinanet_voc/JPEGImages/test_BLI_0001.jpg")
    tiling_start = time.perf_counter()
    tile_size = grid_dim_to_tile_size(test_im, GRID_DIM, 100)
    test_tiles, test_offsets = create_overlapping_tiled_mosaic(test_im, tile_size=tile_size, overlap=100, no_fill=False)
    after_tiling = time.perf_counter()
    print("Tiling took:", after_tiling - tiling_start)
    print(test_offsets)
    create_grid_plot(test_tiles, captions = [str(i+1) for i in range(GRID_DIM**2)], show=True, figscale=10)
    cv2.imwrite("/Users/pasi.pyrro/Documents/video_analytics/rescue-drone-tool/data/misc/test_BLI_0001_4.jpg", test_tiles[-1])
    # restored_im = merge_overlapping_tiled_mosaic(test_tiles, test_offsets, overlap=100, no_fill=False)
    # print("Merging took:", time.perf_counter() - after_tiling)
    # print(restored_im.shape)
    # cv2.imwrite("/Users/pasi.pyrro/Documents/video_analytics/rescue-drone-tool/data/misc/test_BLI_0001_merged.jpg", restored_im)

    # grid_im = cv2.imread("/Users/pasi.pyrro/Documents/video_analytics/rescue-drone-tool/data/datasets/heridal_keras_retinanet_voc_tiled/JPEGImages/test_BLI_0006_1.jpg")

    # grid_im = draw_square_grid(grid_im, spacing=49)

    # cv2.imwrite("/Users/pasi.pyrro/Documents/video_analytics/rescue-drone-tool/data/misc/sliding_window.jpg", grid_im)