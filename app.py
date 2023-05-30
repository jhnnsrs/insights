from arkitekt import register, log
from rekuest.api.schema import LogLevelInput
from typing import List
from mikro.api.schema import (
    RepresentationFragment,
    TableFragment,
    RepresentationVariety,
    from_df,
    GraphFragment,
    create_graph,
    ROIFragment,
    create_roi,
    InputVector,
    RoiTypeInput,
)
from sklearn.cluster import DBSCAN
import numpy as np
from typing import Optional, Tuple
import pandas as pd
import matplotlib.pyplot as plt
import os
import itertools
import numpy as np
from scipy import ndimage


def bbox2_ND(img: np.ndarray):
    N = img.ndim
    out = []
    for ax in itertools.combinations(reversed(range(N)), N - 1):
        nonzero = np.any(img, axis=ax)
        out.extend(np.where(nonzero)[0][[0, -1]])
    return tuple(out)


@register
def measure_label_sizes(
    images: List[RepresentationFragment],
    bins: int = 10,
    background_label: int = 0,
    label_size_label: str = "Label Size",
    frequency_label: str = "Frequency",
) -> TableFragment:
    """Measure Label Sizes

    Creates a histogram of bin sizes for the labels in the images

    Args:
        images (List[RepresentationFragment]): List of images to measure
        bins (int, optional): Number of bins to use in the histogram. Defaults to 10.

    Returns:
        histogram (TableFragment): Table with the histogram
    """
    segmentation_maps = [image.data.compute() for image in images]

    # Initialize an array to store all sizes from all segmentation maps
    all_sizes = np.array([])

    for segmentation_map in segmentation_maps:
        # Remove background label
        segmentation_map = segmentation_map.data.ravel()
        mask = segmentation_map != background_label
        x, counts = np.unique(segmentation_map[mask], return_counts=True)
        all_sizes = np.concatenate((all_sizes, counts))

    # Calculate frequency for all sizes
    print(all_sizes)
    hist, bins = np.histogram(all_sizes, bins=bins)

    # Create DataFrame
    df = pd.DataFrame({label_size_label: bins[:-1], frequency_label: hist})

    return from_df(df, name="Label Histogram", rep_origins=images)


@register
def plot_histogram(
    table: TableFragment, x_column: Optional[str], y_column: Optional[str]
) -> GraphFragment:
    """Plot Table

    Plots a table as a graph

    Args:
        df (TableFragment): Table to plot

    Returns:
        graph (GraphFragment): Graph with the table
    """

    df = table.data
    x_col, y_col = df.columns[:2]

    x = x_column or x_col
    y = y_column or y_col

    df.plot(kind="bar", x=x, y=y, color="skyblue", edgecolor="black")

    plt.title(f"{x} vs {y}")
    plt.xlabel(x)
    plt.ylabel(y)
    plt.grid(axis="y", alpha=0.75)

    temp_file = f"{table.id}.png"
    # Save the plot to a file
    plt.savefig(temp_file, dpi=300, bbox_inches="tight")

    # Clear the plot so it doesn't show up in the next plotting command
    plt.clf()

    x = create_graph(
        image=open(temp_file, "rb"), name=f"Histogram of {table.name}", tables=[table]
    )
    os.remove(temp_file)

    return x


@register
def get_quantile_values(
    table: TableFragment,
    start_quantile: float = 0.25,
    end_quantile: float = 0.75,
    value_column: Optional[str] = None,
    retrieve_column: Optional[str] = None,
) -> Tuple[float, float]:
    """Get quantile values

    Get the values at the start and end quantiles, interpolating if necessary
    to find the closest value

    Parameters
    ----------
    table : TableFragment
        The table
    end_quantile : float
        The s quantile, by default 0.75
    start_quantile : float, optional
        The start quantile, by default 0.25
    value_column : Optional[str], optional
        The value colum, by default first colum
    retrieve_column : Optional[str], optional
        The retrieve column (colum which value will be returned not used to calculate), by default value_column

    Returns
    -------
    x: float:
        The value at the start quantile

    y: float:
        The value at the end quantile
    """
    df = table.data

    quantiles = [start_quantile, end_quantile]

    x_col = value_column or df.columns[0]
    y_col = retrieve_column or x_col

    quantile_values = df[x_col].quantile(quantiles)

    # Find corresponding 'Label Size' values for each quantile
    x = []

    for q, v in zip(quantiles, quantile_values):
        # Calculate absolute differences between the quantile value and all 'Frequency' values
        differences = abs(df[x_col] - v)
        # Get the index of the minimum difference
        min_diff_index = differences.idxmin()
        # Get the corresponding 'Label Size' value
        corresponding_label_size = df.loc[min_diff_index, y_col]
        x.append(corresponding_label_size)
        print(
            f"{y_col} corresponding to the {x_col} closest to {q*100}% percentile: {corresponding_label_size}"
        )

    return tuple(x)


@register()
def mark_in_range(
    rep: RepresentationFragment, min_value: Optional[int], max_value: Optional[int]
) -> List[ROIFragment]:
    """Mark based on range

    Takes a masked image and marks rois based for
    the label size inbtween the values

    Args:
        rep (RepresentationFragment): The image to label outliers for

    Returns:
        List[LabelFragment]: The labels for the outliers
    """
    assert (
        rep.variety == RepresentationVariety.MASK
    ), "Only mask representations are supported"

    data = rep.data.compute()
    labels, counts = np.unique(data, return_counts=True)

    # As counts represents the number of pixels in each label,
    # we can use this to determine the size of each label

    rois = []

    for label, count in zip(labels, counts):
        if min_value is not None and count < min_value:
            continue
        if max_value is not None and count > max_value:
            continue

        masked_image = data.where(data == label, 0)

        try:
            c1, c2, t1, t2, z1, z2, y1, y2, x1, x2 = bbox2_ND(masked_image)

            roi = create_roi(
                rep,
                vectors=[
                    InputVector(x=x1, y=y1, z=z1, t=t1, c=c1),
                    InputVector(x=x2, y=y1, z=z1, t=t1, c=c1),
                    InputVector(x=x2, y=y2, z=z1, t=t1, c=c1),
                    InputVector(x=x1, y=y2, z=z1, t=t1, c=c1),
                ],
                type=RoiTypeInput.RECTANGLE,
                label="size: {}".format(count),
            )
            rois.append(roi)
        except Exception as e:
            print("Could not create bbox for label {} {}".format(label, e))

    print(rois)
    return rois


@register()
def mark_clusters_of_size(
    rep: RepresentationFragment,
    distance: float,
    min_size: int,
    max_size: Optional[int],
    c: Optional[int] = 0,
    t: Optional[int] = 0,
    z: Optional[int] = 0,
    limit: Optional[int] = None,
) -> List[ROIFragment]:
    """Mark dence clusters

    Takes a masked image and marks rois based for
    the on dense clusters of a certain size (number of labels)

    Args:
        rep (RepresentationFragment): The image to label outliers for
        distance (float): The distance between points in a cluster (eps in DBSCAN)
        min_size (int): The minimum size of a cluster (min_samples in DBSCAN)
        max_size (Optional[int]): The maximum size of a cluster (threshold for number of labels in a cluster)
        c (Optional[int], optional): The channel to use. Defaults to 0.
        t (Optional[int], optional): The timepoint to use. Defaults to 0.
        z (Optional[int], optional): The z-slice to use. Defaults to 0.


    Returns:
        List[ROIFragment]: The rois for the clusters
    """
    assert (
        rep.variety == RepresentationVariety.MASK
    ), "Only mask representations are supported"

    x = rep.data.sel(c=c, t=t, z=z).compute().data

    # %%
    labels = ndimage.label(x)[0]
    centroids = ndimage.center_of_mass(x, labels, range(1, labels.max() + 1))
    centroids = np.array(centroids)

    dbscan = DBSCAN(eps=distance, min_samples=min_size).fit(centroids)
    dblabels = dbscan.labels_
    n_clusters_ = len(set(dblabels)) - (1 if -1 in dblabels else 0)
    n_noise_ = list(dblabels).count(-1)

    print("Estimated number of clusters: %d" % n_clusters_)
    print("Estimated number of noise points: %d" % n_noise_)

    if limit is not None:
        if n_clusters_ > limit:
            print(f"Limiting number of clusters to {limit}")
            n_clusters_ = limit
    rois = []

    for cluster in range(0, n_clusters_):
        centrois = centroids[dblabels == cluster]
        if max_size is not None and len(centrois) > max_size:
            continue

        y, x = centrois.mean(axis=0).tolist()

        roi = create_roi(
            rep,
            vectors=[
                InputVector(x=x, y=y, z=z, t=t, c=c),
            ],
            type=RoiTypeInput.POINT,
            label=f"Cluster {cluster} size: {len(centrois)}",
        )
        rois.append(roi)

    return rois


@register()
def mark_clusters_of_size_rectangle(
    rep: RepresentationFragment,
    distance: float,
    min_size: int,
    max_size: Optional[int],
    c: Optional[int] = 0,
    t: Optional[int] = 0,
    z: Optional[int] = 0,
    limit: Optional[int] = None,
) -> List[ROIFragment]:
    """Mark dence clusters (Rectangle)

    Takes a masked image and marks rois based for
    the on dense clusters of a certain size (number of labels)

    Args:
        rep (RepresentationFragment): The image to label outliers for
        distance (float): The distance between points in a cluster (eps in DBSCAN)
        min_size (int): The minimum size of a cluster (min_samples in DBSCAN)
        max_size (Optional[int]): The maximum size of a cluster (threshold for number of labels in a cluster)
        c (Optional[int], optional): The channel to use. Defaults to 0.
        t (Optional[int], optional): The timepoint to use. Defaults to 0.
        z (Optional[int], optional): The z-slice to use. Defaults to 0.


    Returns:
        List[ROIFragment]: The rois for the clusters
    """
    assert (
        rep.variety == RepresentationVariety.MASK
    ), "Only mask representations are supported"

    x = rep.data.sel(c=c, t=t, z=z).compute().data

    # %%
    labels = ndimage.label(x)[0]
    centroids = ndimage.center_of_mass(x, labels, range(1, labels.max() + 1))
    centroids = np.array(centroids)

    dbscan = DBSCAN(eps=distance, min_samples=min_size).fit(centroids)
    dblabels = dbscan.labels_
    n_clusters_ = len(set(dblabels)) - (1 if -1 in dblabels else 0)
    n_noise_ = list(dblabels).count(-1)

    print("Estimated number of clusters: %d" % n_clusters_)
    print("Estimated number of noise points: %d" % n_noise_)

    if limit is not None:
        if n_clusters_ > limit:
            print(f"Limiting number of clusters to {limit}")
            n_clusters_ = limit
    rois = []

    for cluster in range(0, n_clusters_):
        centrois = centroids[dblabels == cluster]
        if max_size is not None and len(centrois) > max_size:
            continue

        in_labels = []

        for centoid in centrois:
            label_at_centroid = x[int(centoid[0]), int(centoid[1])]
            in_labels.append(label_at_centroid)

        if 0 in in_labels:
            try:
                log("Label at centroid is 0", LogLevelInput.ERROR)
            except:
                pass
            continue

        print(in_labels)

        mask = np.isin(x, in_labels)
        y_coords, x_coords = np.where(mask)

        y1 = np.min(y_coords)
        x1 = np.min(x_coords)
        y2 = np.max(y_coords)
        x2 = np.max(x_coords)

        print(x1, y1, x2, y2)

        roi = create_roi(
            rep,
            vectors=[
                InputVector(x=x1, y=y1, z=z, t=t, c=c),
                InputVector(x=x2, y=y1, z=z, t=t, c=c),
                InputVector(x=x2, y=y2, z=z, t=t, c=c),
                InputVector(x=x1, y=y2, z=z, t=t, c=c),
            ],
            type=RoiTypeInput.RECTANGLE,
            label="size: {}".format(len(centrois)),
        )
        rois.append(roi)

    return rois
