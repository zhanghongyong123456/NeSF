import torch
from torch import nn
from model.illumination_model import IlluminationNet
from model.shading_model import ShadingNet
from model.albedo_model import AlbedoNet
from model.rendering_model import RenderingNet

### Eq. 3 in the main paper ###
def render_shading(shading_bases, illum_descriptor):
    expanded_shading_bases = torch.unsqueeze(shading_bases, 2).repeat(1, 1, 3, 1, 1)
    rendered = torch.sum((illum_descriptor * expanded_shading_bases), dim=1)
    return rendered

class NeuRenFrm(nn.Module):
    def __init__(self, K=32):
        super(NeuRenFrm, self).__init__()
        self.K = K

        self.IllumEncNet = IlluminationNet(K=self.K)
        self.ShadingNet = ShadingNet(K=self.K)
        self.AlbedoNet = AlbedoNet()
        self.RenderingNet = RenderingNet()

    def forward(self, depth_map, normal_map, hdr_illum_map, input_unharmonized_img):
        illum_descriptor = self.IllumEncNet(hdr_illum_map)

        depth_nd_normal = torch.cat([depth_map, normal_map], 1)
        shading_bases = self.ShadingNet(depth_nd_normal)
        shading = render_shading(shading_bases, illum_descriptor)

        albedo, albedo_feats = self.AlbedoNet(input_unharmonized_img)

        rendering_input = torch.cat([albedo_feats, shading, input_unharmonized_img], 1)
        harmonized_img = self.RenderingNet(rendering_input)

        return shading, albedo, harmonized_img


if __name__ == '__main__':
    depth = torch.randn(1, 1, 480, 640)
    normal = torch.randn(1, 3, 480, 640)
    hdr_illum_map = torch.randn(1, 3, 128, 256)
    input_unharmonized_img = torch.randn(1, 3, 480, 640)


    net = NeuRenFrm(K=32)
    shad, alb, rb = net(depth, normal, hdr_illum_map, input_unharmonized_img)

    print(shad.shape)
    print(alb.shape)
    print(rb.shape)
