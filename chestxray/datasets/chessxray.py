import os
import warnings
from typing import List

import numpy as np
import pandas as pd
from mmdet.datasets.base_det_dataset import DATASETS, BaseDetDataset
from mmengine.fileio import join_path, list_from_file, load

# warnings.filterwarnings("ignore", category=UndefinedMetricWarning)
# warnings.filterwarnings("ignore", category=UserWarning)


def filename2subfolder(file_name: str) -> str:
    """transform file name into subfolder + file name path"""
    session_id = file_name.split("_")[0]
    subfolder = "/".join([session_id[i : i + 2] for i in range(0, len(session_id), 2)])
    return os.path.join(subfolder, file_name)


@DATASETS.register_module()
class ChessXray(BaseDetDataset):
    def __init__(self, *args, **kwargs):
        super(ChessXray, self).__init__(*args, **kwargs)

    def load_data_list(self) -> List[dict]:
        """Load annotations from an annotation file named as ``self.ann_file``

        If the annotation file does not follow `OpenMMLab 2.0 format dataset
        <https://mmengine.readthedocs.io/en/latest/advanced_tutorials/basedataset.html>`_ .
        The subclass must override this method for load annotations. The meta
        information of annotation file will be overwritten :attr:`METAINFO`
        and ``metainfo`` argument of constructor.

        Returns:
            list[dict]: A list of annotation.
        """  # noqa: E501
        annotations = pd.read_csv(self.ann_file)
        metainfo = ({"dataset_type": "test_dataset", "task_name": "test_task"},)
        for k, v in metainfo.items():
            self._metainfo.setdefault(k, v)

        # load and parse data_infos.
        data_list = []
        for path, group in annotations.groupby("path"):
            data_list.append(
                {
                    "img_path": path,
                    "height": group.iloc[0]["height"],
                    "width": group.iloc[0]["width"],
                    "instances": [
                        {
                            "bbox": [item["Bbox [x"], item["y"], item["w"], item["h]"]],
                            "bbox_label": item["class_id"],
                        }
                        for _, item in group.iterrows()
                    ],
                }
            )

        return data_list
