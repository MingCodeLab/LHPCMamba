"""LHPCMamba: Lightweight Hybrid Parallel CNN-Mamba Network for Driver State Detection."""

from ultralytics import YOLO
import argparse
from pathlib import Path

ROOT = Path(__file__).resolve().parent


def resolve_path(path):
    path = Path(path)
    return str(path if path.is_absolute() else ROOT / path)


def parse_opt():
    parser = argparse.ArgumentParser()
    parser.add_argument('--data', type=str, default='ultralytics/cfg/datasets/coco.yaml', help='dataset.yaml path')
    parser.add_argument('--config', type=str, default='ultralytics/cfg/models/LHPCMamba/LHPCMamba.yaml', help='model path(s)')
    parser.add_argument('--batch_size', type=int, default=32, help='batch size')
    parser.add_argument('--imgsz', '--img', '--img-size', type=int, default=640, help='inference size (pixels)')
    parser.add_argument('--task', default='train', choices=['train', 'val', 'test'], help='train, val, or test')
    parser.add_argument('--device', default='0', help='cuda device, i.e. 0 or 0,1,2,3 or cpu')
    parser.add_argument('--workers', type=int, default=8, help='max dataloader workers (per RANK in DDP mode)')
    parser.add_argument('--epochs', type=int, default=300)
    parser.add_argument('--optimizer', default='SGD', help='SGD, Adam, AdamW')
    parser.add_argument('--amp', action='store_true', help='open amp')
    parser.add_argument('--project', default='output_dir/coco', help='save to project/name')
    parser.add_argument('--name', default='lhpcmamba', help='save to project/name')
    parser.add_argument('--half', action='store_true', help='use FP16 half-precision inference')
    parser.add_argument('--dnn', action='store_true', help='use OpenCV DNN for ONNX inference')
    opt = parser.parse_args()
    return opt


if __name__ == '__main__':
    opt = parse_opt()
    common_args = {
        "data": resolve_path(opt.data),
        "imgsz": opt.imgsz,
        "workers": opt.workers,
        "batch": opt.batch_size,
        "device": opt.device,
        "project": resolve_path(opt.project),
        "name": opt.name,
    }
    model = YOLO(resolve_path(opt.config))

    if opt.task == 'train':
        train_args = common_args.copy()
        train_args.update({"epochs": opt.epochs, "optimizer": opt.optimizer, "amp": opt.amp})
        model.train(**train_args)
    elif opt.task == 'val':
        val_args = common_args.copy()
        val_args.update({"half": opt.half, "dnn": opt.dnn})
        model.val(**val_args)
    elif opt.task == 'test':
        test_args = common_args.copy()
        test_args.update({"half": opt.half, "dnn": opt.dnn, "split": "test"})
        model.val(**test_args)
