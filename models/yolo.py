# YOLOv5 🚀 by Ultralytics, GPL-3.0 license
"""
YOLO-specific modules

Usage:
    $ python path/to/models/yolo.py --cfg yolov5s.yaml
"""

import argparse
import sys
from copy import deepcopy
from pathlib import Path

import torch

FILE = Path(__file__).resolve()
ROOT = FILE.parents[1]  # YOLOv5 root directory
if str(ROOT) not in sys.path:
    sys.path.append(str(ROOT))  # add ROOT to PATH
# ROOT = ROOT.relative_to(Path.cwd())  # relative

from models.common import *
from models.experimental import *
from utils.autoanchor import check_anchor_order
from utils.general import LOGGER, check_version, check_yaml, make_divisible, print_args
from utils.plots import feature_visualization
from utils.torch_utils import fuse_conv_and_bn, initialize_weights, model_info, scale_img, select_device, time_sync

try:
    import thop  # for FLOPs computation
except ImportError:
    thop = None


class Detect(nn.Module):
    # 类变量，可用类和对象来调用，所有对象共享
    stride = None  # strides computed during build ，在模型初始化 的时候，会对detect层每层下采样倍数进行赋值 tensor([ 8., 16., 32.])
    onnx_dynamic = False  # ONNX export parameter

    def __init__(self, nc=80, anchors=(), ch=(), inplace=True):  # detection layer
        super().__init__()
        self.nc = nc  # number of classes ，类别数量
        self.no = nc + 5  # number of outputs per anchor ，每个anchor 输出数量
        self.nl = len(anchors)  # number of detection ，layers detection layer层的数量
        self.na = len(anchors[0]) // 2  # number of anchors ，每层anchor数量
        self.grid = [torch.zeros(1)] * self.nl  # init grid 初始化网格对象 ,记录左上角位置，用于乘以预测值得到xy
        self.anchor_grid = [torch.zeros(
            1)] * self.nl  # init anchor grid 初始化anchor_grid对象，获得相对grid 大小的anchor，用于 乘以预测值得到wh

        # 向模型中增加一个缓冲区参数（一个与训练参数区分的参数，使用self.anchor调用)
        # pytorch 保存保存模型  torch.save(model.state_dict()) ,其中保存的参数有两种:可训练参数和缓冲区参数
        #         1. nn.Parameter(requires_grad=True)  和 nn.module.register_buffer()
        # TODO 注意，在初始化模型之后 anchor 会在Model类的初始化中 除以下采样倍数stride 变成 相对grid大小

        self.register_buffer('anchors', torch.tensor(anchors).float().view(self.nl, -1, 2))  # shape(nl,na,2)

        # output conv，分别创建len(ch)个卷积层 作为m变量（ m为detect layer多层conv 1*1的 ）加入到model list列表
        self.m = nn.ModuleList(
            nn.Conv2d(in_channels=x, out_channels=self.no * self.na, kernel_size=(1, 1)) for x in ch)

        self.inplace = inplace  # use in-place ops (e.g. slice assignment)

    def forward(self, x):
        # z
        z = []  # inference output 保存输出结果
        for i in range(self.nl):  # 循环遍历nl个detect layer 层处理预测数据，此处self detect 为 一个1*1的卷积层
            x[i] = self.m[i](x[i])  # conv #将x放入不同的卷积层 获得 m( m 为detect layer，包括好nl个 yolo layer)的输出
            bs, _, ny, nx = x[i].shape  # x(bs,255,20,20) to x(bs,3,20,20,85) 转换一下数据的形式
            # view 将数据划分维度，permute 将维度顺序交换[bs,na,no:85,ny,nx]-->[bs,na,ny,nx,no:85]
            x[i] = x[i].view(bs, self.na, self.no, ny, nx).permute(0, 1, 3, 4, 2).contiguous()  #

            # 测试或验证将会对其进一步处理
            if not self.training:  # inference
                if self.onnx_dynamic or self.grid[i].shape[2:4] != x[i].shape[2:4]:
                    self.grid[i], self.anchor_grid[i] = self._make_grid(nx, ny, i)

                y = x[i].sigmoid()  # [x,y,w,h,o,c*80]
                if self.inplace:
                    # 对所有的预测偏移值处理恢复到相对原图大小的真实xywh
                    y[..., 0:2] = (y[..., 0:2] * 2 - 0.5 + self.grid[i]) * self.stride[i]  # xy
                    y[..., 2:4] = (y[..., 2:4] * 2) ** 2 * self.anchor_grid[i]  # wh
                else:  # for YOLOv5 on AWS Inferentia https://github.com/ultralytics/yolov5/pull/2953
                    xy = (y[..., 0:2] * 2 - 0.5 + self.grid[i]) * self.stride[i]  # xy
                    wh = (y[..., 2:4] * 2) ** 2 * self.anchor_grid[i]  # wh
                    y = torch.cat((xy, wh, y[..., 4:]), -1)
                #  [bs,3,ny,nx,no:85]-->[bs,3*ny*nx,no:85]
                z.append(y.view(bs, -1, self.no))
                # z = [[bs,3*ny*nx,no:85],[bs,3*ny*nx,no:85],[bs,3*ny*nx,no:85]]
        # 返回z
        # 如果预测将z在第一维度拼接 z=[bs,3*3*ny*nx,no:85]
        # 如果训练x=[bs,na,ny,nx,no:85]
        return x if self.training else (torch.cat(z, 1), x)

    def _make_grid(self, nx=20, ny=20, i=0):
        """创建网络输出后grid 的带大小，不同size的图片grid 不同"""
        # anchor 为缓冲区的参数[[,,][,,][,,]]

        d = self.anchors[i].device  # 获得anchor的device
        # na是由ymal 中超参数传来的 ，xy/nx是输入img经过网络后下采样得到的 ，如640*640下采样32倍 20*20
        shape = 1, self.na, ny, nx, 2  # grid shape

        # 生成网格坐标 ：meshgrid+stack
        if check_version(torch.__version__, '1.10.0'):  # torch>=1.10.0 meshgrid workaround for torch>=0.7 compatibility
            yv, xv = torch.meshgrid(torch.arange(ny, device=d), torch.arange(nx, device=d), indexing='ij')
        else:
            yv, xv = torch.meshgrid(torch.arange(ny, device=d), torch.arange(nx, device=d))
        # grid 网格，记录左上角坐标
        grid = torch.stack((xv, yv), dim=2).expand(shape).float()  # 生成网格+扩展为shape格式
        # anchor_grid元组(),将相对grid 大小的anchor 恢复到grid大小
        # TODO:看似先下采样到grid大小后恢复原anchor大小，看似多此一举，其实在预测时是这样的，但是在训练时不需要将anchor 恢复到原图大小）
        anchor_grid = (self.anchors[i] * self.stride[i]).view((1, self.na, 1, 1, 2)).expand(shape).float()
        return grid, anchor_grid
        # 1.根据 nx ny大小用meshgrid 获得 yv，xv
        # args:
        # yv,用于创建grid 在y方向的向量
        # xv,用于创建grid 在x方向的向量

        # >>> x = torch.tensor([1, 2, 3])
        # >>> y = torch.tensor([4, 5, 6])
        # >>> grid_x, grid_y = torch.meshgrid(x, y)
        #
        # >>> grid_x
        # tensor([[1, 1, 1],
        #         [2, 2, 2],
        #         [3, 3, 3]])
        # >>> grid_y
        # tensor([[4, 5, 6],
        #         [4, 5, 6],
        #         [4, 5, 6]])
        # 2.用yv，xy获得 grid
        # stack():
        # 是增加新的维度来完成拼接，不改变原维度上的数据大小。
        # cat():
        # 是在现有维度上进行数据的增加（改变了现有维度大小），不增加新的维度。

        # 1.grid=torch.stack((xv, yv), 2)
        # tensor([[[1,4], [1,5], [1,6]],
        #         [[2,4], [2,5], [2,5]],
        #         [[3,4], [3,5], [3,6]]])
        # anchor_grid 就是经过上采样后正常的anchor


class Model(nn.Module):
    def __init__(self, cfg='yolov5s.yaml', ch=3, nc=None, anchors=None):  # model, input channels, number of classes
        """
        args
            cfg:模型参数文件的地址
            ch:网络的输入通道（最开始的卷积核需要，默认三通道rgb）
            nc:类别个数
            anchor:ymal中已经有anchor ，如果需要自己设定可以取github找聚类算法生成自己的anchor
        """
        super().__init__()
        if isinstance(cfg, dict):  # cfg参数是否为字典，反之就是ymal文件
            self.yaml = cfg  # model dict
        else:  # is *.yaml
            import yaml  # for torch hub
            self.yaml_file = Path(cfg).name  # 保存ymal文件的内容
            with open(cfg, encoding='ascii', errors='ignore') as f:  # 打开用ascii编码的方式打开ymal文件
                self.yaml = yaml.safe_load(f)  # model dict ，加载ymal文件，将其保存为字典形式

        # Define model
        ch = self.yaml['ch'] = self.yaml.get('ch', ch)  # input channels dict的get方法=if else如果存在ch超参数则取，否则返回第二项
        if nc and nc != self.yaml['nc']:  # 判断输入类别数量nc 是否等于超参数，否则报错且按输入参数构建网络
            LOGGER.info(f"Overriding model.yaml nc={self.yaml['nc']} with nc={nc}")
            self.yaml['nc'] = nc  # override yaml value
        if anchors:  # 判断是否输入了自定义的anchor
            LOGGER.info(f'Overriding model.yaml anchors with anchors={anchors}')
            self.yaml['anchors'] = round(anchors)  # override yaml value
        # 解析模型，返回模型和 TODO
        self.model, self.save = parse_model(deepcopy(self.yaml), ch=[ch])  # model, savelist
        self.names = [str(i) for i in range(self.yaml['nc'])]  # default names
        self.inplace = self.yaml.get('inplace', True)

        # Build strides, anchors ,初始化用于构建 相对于不同grid 的anchor_grid
        m = self.model[-1]  # Detect()
        if isinstance(m, Detect):  # 如果module 是detct层，添加 inplace/stride/anchors 参数
            s = 256  # 2x min stride
            m.inplace = self.inplace
            # TODO:编造一组256*256的测试图像进入模型，获得输出模型的grid ,再用s=256/grid_w，获得下采样倍数 赋值给m.stride=tensor([ 8., 16., 32.])
            # TODO：在修改 Detect model 中 anchor 为相对于grid的大小
            # 模型默认为默认model.training=True ,所以获得Detect model的预测值不需要经过 anchor的优化，这样拿下采样倍数太巧妙了。注意：(model.train()为true/eval()为Flase),
            m.stride = torch.tensor([s / x.shape[-2] for x in self.forward(torch.zeros(1, ch, s, s))])
            check_anchor_order(m)  # must be in pixel-space (not grid-space)
            m.anchors /= m.stride.view(-1, 1, 1)  # 对anchor 缩放为相对于 每层grid 的大小
            self.stride = m.stride  # 每层的下采样倍数赋值给 整个model
            self._initialize_biases()  # only run once

        # Init weights, biases
        initialize_weights(self)
        self.info()
        LOGGER.info('')

    def forward(self, x, augment=False, profile=False, visualize=False):
        # 如果使用数据增强，则使用_forward_augment，反之_forward_once
        if augment:
            return self._forward_augment(x)  # augmented inference, None
        return self._forward_once(x, profile, visualize)  # single-scale inference, train

    def _forward_augment(self, x):
        img_size = x.shape[-2:]  # height, width
        s = [1, 0.83, 0.67]  # scales
        f = [None, 3, None]  # flips (2-ud, 3-lr)
        y = []  # outputs
        for si, fi in zip(s, f):
            xi = scale_img(x.flip(fi) if fi else x, si, gs=int(self.stride.max()))
            yi = self._forward_once(xi)[0]  # forward
            # cv2.imwrite(f'img_{si}.jpg', 255 * xi[0].cpu().numpy().transpose((1, 2, 0))[:, :, ::-1])  # save
            yi = self._descale_pred(yi, fi, si, img_size)
            y.append(yi)
        y = self._clip_augmented(y)  # clip augmented tails
        return torch.cat(y, 1), None  # augmented inference, train

    def _forward_once(self, x, profile=False, visualize=False):
        y, dt = [], []  # outputs
        for m in self.model:
            if m.f != -1:  # if not from previous layer
                x = y[m.f] if isinstance(m.f, int) else [x if j == -1 else y[j] for j in m.f]  # from earlier layers
            if profile:
                self._profile_one_layer(m, x, dt)
            x = m(x)  # run
            y.append(x if m.i in self.save else None)  # save output
            if visualize:
                feature_visualization(x, m.type, m.i, save_dir=visualize)
        return x

    def _descale_pred(self, p, flips, scale, img_size):
        # de-scale predictions following augmented inference (inverse operation)
        if self.inplace:
            p[..., :4] /= scale  # de-scale
            if flips == 2:
                p[..., 1] = img_size[0] - p[..., 1]  # de-flip ud
            elif flips == 3:
                p[..., 0] = img_size[1] - p[..., 0]  # de-flip lr
        else:
            x, y, wh = p[..., 0:1] / scale, p[..., 1:2] / scale, p[..., 2:4] / scale  # de-scale
            if flips == 2:
                y = img_size[0] - y  # de-flip ud
            elif flips == 3:
                x = img_size[1] - x  # de-flip lr
            p = torch.cat((x, y, wh, p[..., 4:]), -1)
        return p

    def _clip_augmented(self, y):
        # Clip YOLOv5 augmented inference tails
        nl = self.model[-1].nl  # number of detection layers (P3-P5)
        g = sum(4 ** x for x in range(nl))  # grid points
        e = 1  # exclude layer count
        i = (y[0].shape[1] // g) * sum(4 ** x for x in range(e))  # indices
        y[0] = y[0][:, :-i]  # large
        i = (y[-1].shape[1] // g) * sum(4 ** (nl - 1 - x) for x in range(e))  # indices
        y[-1] = y[-1][:, i:]  # small
        return y

    def _profile_one_layer(self, m, x, dt):
        c = isinstance(m, Detect)  # is final layer, copy input as inplace fix
        o = thop.profile(m, inputs=(x.copy() if c else x,), verbose=False)[0] / 1E9 * 2 if thop else 0  # FLOPs
        t = time_sync()
        for _ in range(10):
            m(x.copy() if c else x)
        dt.append((time_sync() - t) * 100)
        if m == self.model[0]:
            LOGGER.info(f"{'time (ms)':>10s} {'GFLOPs':>10s} {'params':>10s}  {'module'}")
        LOGGER.info(f'{dt[-1]:10.2f} {o:10.2f} {m.np:10.0f}  {m.type}')
        if c:
            LOGGER.info(f"{sum(dt):10.2f} {'-':>10s} {'-':>10s}  Total")

    def _initialize_biases(self, cf=None):  # initialize biases into Detect(), cf is class frequency
        # https://arxiv.org/abs/1708.02002 section 3.3
        # cf = torch.bincount(torch.tensor(np.concatenate(dataset.labels, 0)[:, 0]).long(), minlength=nc) + 1.
        m = self.model[-1]  # Detect() module
        for mi, s in zip(m.m, m.stride):  # m 为detect layer 为list 保存3个 1*1的卷积
            b = mi.bias.view(m.na, -1)  # conv.bias(255) to (3,85)
            b.data[:, 4] += math.log(8 / (640 / s) ** 2)  # obj (8 objects per 640 image)
            b.data[:, 5:] += math.log(0.6 / (m.nc - 0.999999)) if cf is None else torch.log(cf / cf.sum())  # cls
            mi.bias = torch.nn.Parameter(b.view(-1), requires_grad=True)

    def _print_biases(self):
        m = self.model[-1]  # Detect() module
        for mi in m.m:  # from
            b = mi.bias.detach().view(m.na, -1).T  # conv.bias(255) to (3,85)
            LOGGER.info(
                ('%6g Conv2d.bias:' + '%10.3g' * 6) % (mi.weight.shape[1], *b[:5].mean(1).tolist(), b[5:].mean()))

    # def _print_weights(self):
    #     for m in self.model.modules():
    #         if type(m) is Bottleneck:
    #             LOGGER.info('%10.3g' % (m.w.detach().sigmoid() * 2))  # shortcut weights

    def fuse(self):  # fuse model Conv2d() + BatchNorm2d() layers
        LOGGER.info('Fusing layers... ')
        for m in self.model.modules():
            if isinstance(m, (Conv, DWConv)) and hasattr(m, 'bn'):
                m.conv = fuse_conv_and_bn(m.conv, m.bn)  # update conv
                delattr(m, 'bn')  # remove batchnorm
                m.forward = m.forward_fuse  # update forward
        self.info()
        return self

    def info(self, verbose=False, img_size=640):  # print model information
        model_info(self, verbose, img_size)

    def _apply(self, fn):
        # Apply to(), cpu(), cuda(), half() to model tensors that are not parameters or registered buffers
        self = super()._apply(fn)
        m = self.model[-1]  # Detect()
        if isinstance(m, Detect):
            m.stride = fn(m.stride)
            m.grid = list(map(fn, m.grid))
            if isinstance(m.anchor_grid, list):
                m.anchor_grid = list(map(fn, m.anchor_grid))
        return self


def parse_model(d, ch):  # model_dict, input_channels(3)
    """
    args
        d：dict，ymal中的dict字典 , 其中ymal中数字会饿被解析成int，float，其他的解析成字符串 如 abc，“abc”
        ch：输入通道

    """
    LOGGER.info(f"\n{'':>3}{'from':>18}{'n':>3}{'params':>10}  {'module':<40}{'arguments':<30}")
    # anchor 列表，类别数，网络深度，网络宽度
    anchors, nc, gd, gw = d['anchors'], d['nc'], d['depth_multiple'], d['width_multiple']
    na = (len(anchors[0]) // 2) if isinstance(anchors,
                                              list) else anchors  # number of anchors,//2是因为ymal文件中保存anchor wh两个值
    no = na * (nc + 5)  # number of outputs = anchors * (classes + 5) ，网络的输出数量

    # layers：将处理后的ymal每层网络保存在layers，最后整体保存在一个序列中
    # savelist：保存concat层中，出除了-1（上一层）的其他层的 index
    # ch：记录该层的输出通道，作为下一层的输入，第一层由参数设置
    layers, save, c2 = [], [], ch[-1]
    # 将backbone 和head两个合并后解析网络
    for i, (f, n, m_type, args) in enumerate(
            d['backbone'] + d['head']):  # from, number, module, args #f输入通道，n该结构数量，m该结构模型，args该结构中参数
        # -------获得yaml中类名,eval将字符串转为类名-------
        m_type = eval(m_type) if isinstance(m_type,
                                            str) else m_type  # eval strings ,导入 models.common，eval 可以将其解析为 导入中的一个类（代表类名，并不初始化）
        # -------根据 m_type 的类型 生成一个 相对于m_type的参数 用于初始化-------
        for j, a in enumerate(args):  # 逐个解析参数，排除其中表达式的情况，将其保存到列表args
            try:

                # 检查 args列表，如果为str 在本地有对应的类名、变量、就将其赋值，如：False,None,nc,anchors
                args[j] = eval(a) if isinstance(a, str) else a
            except NameError:
                pass  # 上采样时 nearest 不是str类型数值表达式会报错 如：'nearest'

        n = n_ = max(round(n * gd), 1) if n > 1 else n  # depth gain,深度因子 模块数量n n>1有用 ,gd*n取整 和1作比较，取最大加大网络深度(保证至少有一个)

        """受深度因子 gw影响的层"""
        if m_type in [Conv, GhostConv, Bottleneck, GhostBottleneck, SPP, SPPF, DWConv, MixConv2d, Focus, CrossConv,
                      BottleneckCSP, C3, C3TR, C3SPP, C3Ghost]:
            # 如果 m输出上述层中，
            c1, c2 = ch[f], args[0]  # c1 输入通道，c2输出通道数
            if c2 != no:
                # if not output ，如果输出通道 ！= no ，不为detect层则用宽度因子将输出通道加宽
                c2 = make_divisible(c2 * gw, 8)

            args = [c1, c2, *args[1:]]  # 重新调整参数args，[输入通道，输出通道，args原先除了0输出通道的其他参数]
            """受深度因子 gd影响的层"""
            if m_type in [BottleneckCSP, C3, C3TR, C3Ghost]:
                args.insert(2, n)  # number of repeats，将n插入到args c1,c2后面
                n = 1
        elif m_type is nn.BatchNorm2d:
            args = [ch[f]]  # bn层输入通道=输出通道，所以只记录 一个输入
        elif m_type is Concat:
            c2 = sum(ch[x] for x in f)  # ymal中concat将[-1, 14]两层相加，其中数字为层数-1代表上一层，相对最后一层
        elif m_type is Detect:  # 如果是detcet层
            args.append([ch[x] for x in f])  # f为输入通道的index[17, 20, 23] 获得 ch对应这些层的输出通道 保存在args中[128,256,512]
            if isinstance(args[1], int):  # number of anchors
                # 如果保存的anchor 是anchor每层的个数（int 3），而不是9个anchor详细信息，则按range生成anchor
                args[1] = [list(range(args[1] * 2))] * len(f)
        elif m_type is Contract:  # 6.0中未使用
            c2 = ch[f] * args[0] ** 2
        elif m_type is Expand:  # 6.0中未使用
            c2 = ch[f] // args[0] ** 2
        else:  # 6.0中未使用
            c2 = ch[f]

        # -------初始化 m_type 并保存在layers，循环执行下一轮ymal的列表 ，直到生成模型-------
        # 1.初始化m_type(*args),args为经过判断后整理好的参数，与common相应类别参数一一对象，*将列表参数解包 ,然后初始化模型
        # 2. 如果n > 1(该模块有多个)我们就把他保存在Sequential，反之保存在 model_sub
        model_sub = nn.Sequential(*(m_type(*args) for _ in range(n))) if n > 1 else m_type(*args)  # module

        t = str(m_type)[8:-2].replace('__main__.', '')  # module type 模块类型信息 如：'models.common.Conv'
        np = sum(x.numel() for x in model_sub.parameters())  # number params 记录该模块的参数量
        # 将信息添加到结构当中，方便获取 i:attach index第几层, f:'from' index 上一层是第几层, type 模块名称类型, number params参数
        model_sub.i, model_sub.f, model_sub.type, model_sub.np = i, f, t, np
        LOGGER.info(f'{i:>3}{str(f):>18}{n_:>3}{np:10.0f}  {t:<40}{str(args):<30}')  # print
        save.extend(x % i for x in ([f] if isinstance(f, int) else f) if x != -1)  # append to savelist
        layers.append(model_sub)  # 在layer 列表中保存该模块，然后继续检查ymal文件将其他结构保存进来
        if i == 0:
            ch = []
        ch.append(c2)
        # 按配置文件排好的层 + 构建好的模型 （使用eval()）

    return nn.Sequential(*layers), sorted(save)


if __name__ == '__main__':
    # 加载可用命令行输入的参数
    parser = argparse.ArgumentParser()
    parser.add_argument('--cfg', type=str, default='yolov5s.yaml', help='model.yaml')  # 要加载模型的yaml文件
    parser.add_argument('--device', default='', help='cuda device, i.e. 0 or 0,1,2,3 or cpu')  # devic设备使用
    parser.add_argument('--profile', action='store_true', help='profile model speed')  # 当命令行输出profile 时为真
    parser.add_argument('--test', action='store_true', help='test all yolo*.yaml')  # 当命令行输入profile 时为真
    opt = parser.parse_args()
    opt.cfg = check_yaml(opt.cfg)  # check YAML
    print_args(FILE.stem, opt)
    device = select_device(opt.device)

    # Create model 创建模型（init时解析ymal文件）
    model = Model(opt.cfg).to(device)
    model.train()

    # Profile
    if opt.profile:
        img = torch.rand(8 if torch.cuda.is_available() else 1, 3, 640, 640).to(device)
        y = model(img, profile=True)

    # Test all models
    if opt.test:
        for cfg in Path(ROOT / 'models').rglob('yolo*.yaml'):
            try:
                _ = Model(cfg)
            except Exception as e:
                print(f'Error in {cfg}: {e}')

    # Tensorboard (not working https://github.com/ultralytics/yolov5/issues/2898)
    # from torch.utils.tensorboard import SummaryWriter
    # tb_writer = SummaryWriter('.')
    # LOGGER.info("Run 'tensorboard --logdir=models' to view tensorboard at http://localhost:6006/")
    # tb_writer.add_graph(torch.jit.trace(model, img, strict=False), [])  # add model graph
