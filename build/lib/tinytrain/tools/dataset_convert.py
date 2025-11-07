import json
from pathlib import Path
from shutil import copy2
from typing import List, Optional
from pycocotools.coco import COCO


def coco2yolo(coco_json_path: str,
              img_root: str,
              out_root: str,
              keep_coco_ids: Optional[List[int]] = None,
              use_segments: bool = False):
    """
    将 COCO instances 标注转换为 YOLO 格式，并可选择是否拷贝图片与保留指定类别。

    生成的目录结构
    --------------
    out_root/
    ├── images/
    │   └── *.jpg
    └── labels/
        └── *.txt

    参数
    ----
    coco_json_path : str
        COCO instances 标注文件路径，如 'instances_val2017.json'。
    img_root : str
        原始图片所在目录，如 'val2017/'。
    out_root : str
        输出根目录，函数会自动创建 `images` 与 `labels` 子目录。
    keep_coco_ids : list[int] | None, optional
        需要保留的 COCO category_id 列表；若为 None 则保留全部类别。
    use_segments : bool, optional
        若为 True，将多边形分割信息写入标签文件（YOLO-segment 格式）；若为 False，
        则仅写入中心点归一化矩形框 (YOLO-det 格式)。
    """
    out_root = Path(out_root)
    img_out_dir = out_root / 'images'
    lbl_out_dir = out_root / 'labels'
    img_out_dir.mkdir(parents=True, exist_ok=True)
    lbl_out_dir.mkdir(parents=True, exist_ok=True)

    coco = COCO(coco_json_path)

    # ---------- 1. 类别过滤 ----------
    cats = coco.loadCats(coco.getCatIds())
    if keep_coco_ids is None:
        keep_coco_ids = [c['id'] for c in cats]

    keep_set = set(keep_coco_ids)
    keep_cats = [c for c in cats if c['id'] in keep_set]
    cat2yolo = {c['id']: idx for idx, c in enumerate(keep_cats)}
    if not cat2yolo:
        raise ValueError('没有有效类别被保留，请检查 keep_coco_ids')

    # ---------- 2. 遍历图片 ----------
    img_ids = coco.getImgIds()
    for img_id in img_ids:
        img_info = coco.loadImgs(img_id)[0]
        h, w = img_info['height'], img_info['width']
        img_name = img_info['file_name']
        src_img_path = Path(img_root) / img_name
        dst_img_path = img_out_dir / img_name

        # 过滤该图的所有标注
        ann_ids = coco.getAnnIds(imgIds=img_id, iscrowd=False)
        lines = []
        for ann in coco.loadAnns(ann_ids):
            coco_cat_id = ann['category_id']
            if coco_cat_id not in cat2yolo:
                continue
            cls_id = cat2yolo[coco_cat_id]

            if use_segments and 'segmentation' in ann and ann['segmentation']:
                seg = ann['segmentation'][0]
                coords = [float(p) for p in seg]
                coords[0::2] = [p / w for p in coords[0::2]]
                coords[1::2] = [p / h for p in coords[1::2]]
                line = [cls_id] + coords
            else:
                x, y, bw, bh = ann['bbox']
                xc = (x + bw / 2) / w
                yc = (y + bh / 2) / h
                bw /= w
                bh /= h
                line = [cls_id, xc, yc, bw, bh]

            lines.append(' '.join(map(str, line)))

        # 仅当该图含有效目标时才拷贝图片并写标签
        if lines:
            # 拷贝图片
            if not dst_img_path.exists():
                copy2(src_img_path, dst_img_path)

            # 写标签
            lbl_path = lbl_out_dir / f"{Path(img_name).stem}.txt"
            lbl_path.write_text('\n'.join(lines) + '\n')

def coco_json2yolo(json_path: str,):
    with open(json_path, 'r') as f:
        data = json.load(f)

    images = data['images']
    annotations = data['annotations']
    print('len(images)', len(annotations))


if __name__ == '__main__':
    # coco2yolo(r'E:\coco\annotations_trainval2017\annotations\instances_val2017.json',
    #           img_root=r'E:\coco\val2017',
    #           out_root='E:\coco\yolo_cal',
    #           keep_coco_ids=[1, 2, 3, 4, 6, 8])

    coco_json2yolo(r"C:\Users\86724\Downloads\train\labels\labels_my-project-name_2025-10-14-08-55-27.json")