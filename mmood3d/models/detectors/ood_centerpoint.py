from typing import Optional, Dict, Sequence, List

from torch import Tensor

from mmood3d.registry import MODELS
from mmdet3d.models.detectors.centerpoint import CenterPoint


@MODELS.register_module()
class OODCenterPoint(CenterPoint):

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.inter_features = dict()

    def extract_pts_feat(
            self,
            voxel_dict: Dict[str, Tensor],
            points: Optional[List[Tensor]] = None,
            img_feats: Optional[Sequence[Tensor]] = None,
            batch_input_metas: Optional[List[dict]] = None,
    ) -> Sequence[Tensor]:
        """Extract features of points.

        Args:
            voxel_dict(Dict[str, Tensor]): Dict of voxelization infos.
            points (List[tensor], optional):  Point cloud of multiple inputs.
            img_feats (list[Tensor], tuple[tensor], optional): Features from
                image backbone.
            batch_input_metas (list[dict], optional): The meta information
                of multiple samples. Defaults to True.

        Returns:
            Sequence[tensor]: points features of multiple inputs
            from backbone or neck.
        """
        if not self.with_pts_bbox:
            return None

        inter_feats = {}

        voxel_features = self.pts_voxel_encoder(voxel_dict['voxels'],
                                                voxel_dict['num_points'],
                                                voxel_dict['coors'], img_feats,
                                                batch_input_metas)

        batch_size = voxel_dict['coors'][-1, 0] + 1
        encoder_feats = self.pts_middle_encoder(voxel_features, voxel_dict['coors'], batch_size)
        if isinstance(encoder_feats, (list, tuple)):
            inter_feats['spatial_feats'], inter_feats['multi_scale_3d_feats'] = encoder_feats
        else:
            inter_feats['spatial_feats'] = encoder_feats
            inter_feats['multi_scale_3d_feats'] = None

        backbone_feats = self.pts_backbone(inter_feats['spatial_feats'])
        inter_feats['backbone_feats'] = backbone_feats
        out_feats = backbone_feats
        if self.with_pts_neck:
            out_feats = self.pts_neck(backbone_feats)
            inter_feats['neck_feats'] = out_feats

        # save intermediate feature maps
        self.inter_features = inter_feats
        return out_feats
