from typing import Tuple
from pathlib import Path
from tqdm import tqdm
import torch
import numpy as np
from configs.logger import l

A_INSIDE_B = torch.tensor([1, 2], dtype=torch.long)
A_COVER_B = torch.tensor([2, 1], dtype=torch.long)
OVERLAP = torch.tensor([3, 3], dtype=torch.long)
BBOXES_PAIR_INDEXES = torch.combinations(torch.arange(36), r=2).long()
INDICES = torch.cat(
    [BBOXES_PAIR_INDEXES, BBOXES_PAIR_INDEXES.flip(1)],
    dim=0,
).T.split(1, dim=0)


def process_relationship(
    dataset_type: str,
    feature_dir: str,
    graph_dir: str,
) -> None:
    """process_relationship

    Args:
        dataset_type (str): The dataset type, train or val
        feature_dir (str): The dir of feature data
        graph_dir (str): The dir of graph(relationship) data
    """
    l.info(f"Processing {dataset_type} relationship data...")
    # assert the output dir (graph_dir) is exist
    graph_dir: Path = Path(graph_dir) / f"{dataset_type}2014"
    graph_dir.mkdir(parents=True, exist_ok=True)

    # load all feature files
    feature_dir: Path = Path(feature_dir) / f"{dataset_type}2014"
    feature_files = feature_dir.glob("*.npz")

    # process each feature file
    for feature_file in tqdm(
        feature_files,
        desc=f"Processing {dataset_type} relationship",
        total=len(list(feature_dir.glob("*.npz"))),
    ):
        # read feature file
        feature = np.load(feature_file)

        # get the bbox coordinates
        bboxes = feature["bbox"]

        # get the bbox pairs coordinates
        batch = torch.from_numpy(bboxes[BBOXES_PAIR_INDEXES])

        # generate the relationship id between bbox pairs
        relationship = __generate_relationship_values(
            batch_bbox_pair=batch,
            image_width=float(feature["image_w"]),
            image_height=float(feature["image_h"]),
            iou_threshold=0.5,
            a_b_distance_ratio_threshold=0.5,
        )

        # initialize the graph
        graph = torch.empty([36, 36]).long()

        # format the relationship indices to (1260 = 36*36-36)
        values = relationship.T.reshape(-1)

        # fill relationship index into graph by indices
        out = graph.index_put(
            indices=INDICES,
            values=values,
        )

        # set diagonal to 0
        out.fill_diagonal_(0)

        # save the graph
        np.savez(f"{graph_dir}/{feature_file.name}", graph=out)


def __generate_relationship_values(
    batch_bbox_pair: torch.FloatTensor,
    image_width: float,
    image_height: float,
    iou_threshold: float = 0.5,
    a_b_distance_ratio_threshold: float = 0.5,
) -> torch.LongTensor:
    """process_relationship

    classes define:
        1: inside
        2: cover
        3: overlap (IoU >= 0.5)
        4-11: (IoU < 0.5 && overlap_ratio < 0.5)
        12-19: (IoU < 0.5 && overlap_ratio >= 0.5)

    Args:
        batch_bbox_pair (torch.Tensor): The shape is (batch, bbox=2, coordinates=4)
        image_width (float): The width of the whole image
        image_height (float): The height of the whole image
        iou_threshold (float, optional): The IoU threshold to determine is overlap or not. Defaults to 0.5.
        a_b_distance_ratio_threshold (float, optional): The phi threshold. Defaults to 0.3.

    Returns:
        torch.Tensor: The shape is (batch, pair, relationship)
    """
    # =================== Preparing variables ===================
    a_bbox, b_bbox, intersaction_bbox = __get_a_b_intersaction_bboxes(batch_bbox_pair)
    a_center, b_center = __get_centers(batch_bbox_pair)
    a_b_center_difference = a_center - b_center
    # =========================================================
    relationship = __generate_relationship_indexes_by_angle(
        a_b_center_difference=a_b_center_difference,
    )
    # =========================================================
    targets_should_be_shifted = __filter_by_bboxes_distance_ratio(
        image_width=image_width,
        image_height=image_height,
        a_b_center_difference=a_b_center_difference,
        a_b_distance_ratio_threshold=a_b_distance_ratio_threshold,
    )
    # shift class 4 to class 12, class 5 to class 13, ...
    update_shifting = 8
    # update result from class 4-11 to class 12-19
    relationship[targets_should_be_shifted] += update_shifting
    # =========================================================
    overlap_position = __filter_overlap_relationship_by_iou(
        a_bbox=a_bbox,
        b_bbox=b_bbox,
        intersaction_bbox=intersaction_bbox,
        iou_threshold=iou_threshold,
        a_b_center_difference=a_b_center_difference,
    )
    relationship[overlap_position] = OVERLAP.repeat(overlap_position.sum(), 1)
    # =========================================================
    equal_to_a = __filter_A_is_inside_B(
        a_bbox=a_bbox,
        intersaction_bbox=intersaction_bbox,
    )
    relationship[equal_to_a] = A_INSIDE_B.repeat(equal_to_a.sum(), 1)
    # =========================================================
    equal_to_b = __filter_A_covers_B(
        b_bbox=b_bbox,
        intersaction_bbox=intersaction_bbox,
    )
    relationship[equal_to_b] = A_COVER_B.repeat(equal_to_b.sum(), 1)
    # =========================================================

    return relationship


def __get_a_b_intersaction_bboxes(
    batch_bbox_pair: torch.Tensor,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """__get_bboxes

    Args:
        batch_bbox_pair (torch.Tensor): The original bbox pair coordinates

    Returns:
        Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        bbox of a, bbox of b, bbox of intersaction of a and b
    """
    # (batch_size, coordinate_values=4)
    # split out to get bbox a and bbox b in shape = (B, 4)
    a_bbox = batch_bbox_pair[:, 0, :]
    b_bbox = batch_bbox_pair[:, 1, :]

    # intersaction of A and B, whit shape = (B, 4)
    intersaction_bbox = torch.cat(
        [
            batch_bbox_pair[..., :2].max(dim=1)[0],
            batch_bbox_pair[..., 2:].min(dim=1)[0],
        ],
        dim=1,
    )

    return a_bbox, b_bbox, intersaction_bbox


def __get_centers(
    batch_bbox_pair: torch.Tensor,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """__get_centers

    Args:
        batch_bbox_pair (torch.Tensor): The original bbox pair coordinates

    Returns:
        Tuple[torch.Tensor, torch.Tensor]: The center of bbox a and the center of bbox b
    """
    # shape = (batch_size, bbox_center_coordinate_values=2)
    # center of a and b
    get_center = lambda x: torch.stack(
        [
            x[..., 0] + (x[..., 2] - x[..., 0]) / 2,
            x[..., 1] + (x[..., 3] - x[..., 1]) / 2,
        ],
        dim=-1,
    )
    centers = get_center(batch_bbox_pair)

    return centers[..., 0], centers[..., 1]


def __generate_relationship_indexes_by_angle(
    a_b_center_difference: torch.Tensor,
) -> torch.Tensor:
    """__generate_relationship_indexes_by_angle

    Args:
        a_b_center_difference (torch.Tensor): The difference of center of a and b

    Returns:
        torch.Tensor: The relationship indexes
    """

    def get_angle(x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        angle = torch.atan(y / x)
        # Since the arctan with the range of [-pi/2, pi/2],
        # we need to project the coordiantes which in 2nd and 3rd quadrant to [pi/2, 3pi/2]
        angle += torch.pi * (x < 0)
        # avoid the negative angle
        angle += torch.pi * 2
        return angle

    def angle_to_index(angle: torch.Tensor) -> torch.Tensor:
        # split the angle into 8 parts
        a_index = (torch.ceil(angle / (torch.pi / 4)) - 1) % 8 + 1
        # the index of b is oppsited to a
        b_index = (a_index + 3) % 8 + 1
        # padding classes 1 to 3
        classes_padding = 3
        return (torch.stack([a_index, b_index]) + classes_padding).T

    # shape = (B, )
    angle = get_angle(
        x=a_b_center_difference[..., 0],
        y=a_b_center_difference[..., 1],
    )

    # shape = (B, 2)
    # type_indexes is the result of spatial_relationship of each pair
    relationship_indexes = angle_to_index(angle).long()

    return relationship_indexes


def __filter_by_bboxes_distance_ratio(
    image_width: int,
    image_height: int,
    a_b_center_difference,
    a_b_distance_ratio_threshold,
) -> torch.Tensor:
    """__filter_by_bboxes_distance_ratio

    Args:
        image_width (int): The width of image
        image_height (int): The height of image
        a_b_center_difference (_type_): The difference of center of a and b
        a_b_distance_ratio_threshold (_type_): The threshold of distance ratio

    Returns:
        torch.Tensor: The filter result
    """
    # compute the diagonal length of image
    image_diagonal = torch.tensor([image_width, image_height], dtype=torch.float).norm()
    # compute the diagonal length of bboxes a and b
    bboxes_diagonal = a_b_center_difference.norm(dim=-1)
    # compute the ratio
    ratio = bboxes_diagonal / image_diagonal
    # filter out the bboxes which distance ratio is larger than threshold
    return ratio >= a_b_distance_ratio_threshold


def __filter_overlap_relationship_by_iou(
    a_bbox: torch.Tensor,
    b_bbox: torch.Tensor,
    intersaction_bbox: torch.Tensor,
    iou_threshold: float,
    a_b_center_difference: torch.Tensor,
) -> torch.Tensor:
    """__filter_overlap_relationship_by_iou

    Args:
        a_bbox (torch.Tensor): The bbox of a
        b_bbox (torch.Tensor): The bbox of b
        intersaction_bbox (torch.Tensor): The bbox of intersaction of a and b
        iou_threshold (float): The threshold of iou
        a_b_center_difference (torch.Tensor): The difference of center of a and b

    Returns:
        torch.Tensor: The filter result
    """
    # compute the bbox area for a, b and intersaction in shape = (B, )
    # shape = (B, )
    get_area = lambda x: (x[..., 3] - x[..., 1]) * (x[..., 2] - x[..., 0])
    area_a = get_area(a_bbox)
    area_b = get_area(b_bbox)
    area_intersaction = get_area(intersaction_bbox)

    # If IoU < threshold
    iou = (area_a + area_b) / area_intersaction
    threshold = 1 + 1 / iou_threshold
    less_than_threshold = iou < threshold

    # FIXED: filter out intersaction is not overlap
    a_b_center_distance = a_b_center_difference.abs()
    a_b_center_distance_x = a_b_center_distance[..., 0]
    a_b_center_distance_y = a_b_center_distance[..., 1]
    union_length_x = (
        a_bbox[..., 2] - a_bbox[..., 0] + b_bbox[..., 2] - b_bbox[..., 0]
    ) / 2
    union_length_y = (
        a_bbox[..., 3] - a_bbox[..., 1] + b_bbox[..., 3] - b_bbox[..., 1]
    ) / 2
    filter_mask = ~(
        (a_b_center_distance_x > union_length_x)
        | (a_b_center_distance_y > union_length_y)
    )

    return less_than_threshold & filter_mask


def __filter_A_is_inside_B(
    a_bbox: torch.Tensor,
    intersaction_bbox: torch.Tensor,
) -> torch.Tensor:
    """__filter_A_is_inside_B

    Args:
        a_bbox (torch.Tensor): The bbox of a
        intersaction_bbox (torch.Tensor): The bbox of intersaction of a and b

    Returns:
        torch.Tensor: The filter result
    """
    # If IoU equal to a, then a is inside b.
    return (a_bbox == intersaction_bbox).all(dim=1)


def __filter_A_covers_B(
    b_bbox: torch.Tensor,
    intersaction_bbox: torch.Tensor,
) -> torch.Tensor:
    """__filter_A_covers_B

    Args:
        b_bbox (torch.Tensor): The bbox of b
        intersaction_bbox (torch.Tensor): The bbox of intersaction of a and b

    Returns:
        torch.Tensor: The filter result
    """
    # If IoU equal to a, then a cover b.
    return (b_bbox == intersaction_bbox).all(dim=1)
