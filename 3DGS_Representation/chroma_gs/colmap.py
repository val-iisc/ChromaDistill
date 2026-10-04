"""COLMAP parsing and the aligned grey / teacher / colour-reference triplets.

The released ChromaDistill LLFF data ships, per scene:

    images/IMG_*.JPG            monochrome inputs, full resolution
    images_bigcolor/IMG_*.jpg   the teacher network's colorizations, same stems
    sparse/0/*.bin              COLMAP reconstruction, image names match images/

and the real colour frames live in a separate release, nerf_llff_data/<scene>/images,
again under the same stems. Everything here is paired by filename stem rather than by
sort order, because the pre-downsampled images_4 folders were renamed to image000.png
and no longer line up with COLMAP.

All three sources are downsampled by the same integer factor and pushed through one
shared undistortion map, so a pixel at (i, j) means the same ray in the grey input, the
teacher image and the colour reference. The Lab chroma loss compares rendered pixels
against teacher pixels directly, so that alignment is a correctness requirement, not a
nicety.
"""

import json
import os

import cv2
import numpy as np
import torch
from pycolmap import SceneManager

from .normalize import (
    align_principle_axes,
    similarity_from_cameras,
    transform_cameras,
    transform_points,
)

# COLMAP camera model ids.
SIMPLE_PINHOLE, PINHOLE, SIMPLE_RADIAL, RADIAL, OPENCV = 0, 1, 2, 3, 4


def _dist_coeffs(cam):
    """OpenCV (k1, k2, p1, p2) for a COLMAP camera; zeros for the pinhole models.

    The vendored pycolmap Camera only defines the attributes its model actually uses,
    so a SIMPLE_RADIAL camera has k1 and no k2 at all.
    """
    g = lambda n: float(getattr(cam, n, 0.0))
    if cam.camera_type in (SIMPLE_PINHOLE, PINHOLE):
        return np.zeros(4, dtype=np.float64)
    return np.array([g("k1"), g("k2"), g("p1"), g("p2")], dtype=np.float64)


class Parser:
    """Cameras, SfM points and image paths for one scene."""

    def __init__(
        self,
        data_dir,
        color_ref_dir=None,
        factor=4,
        normalize=True,
        align_axes=True,
        test_every=8,
    ):
        self.data_dir = data_dir
        self.color_ref_dir = color_ref_dir
        self.factor = factor
        self.test_every = test_every

        manager = SceneManager(os.path.join(data_dir, "sparse", "0"))
        manager.load_cameras()
        manager.load_images()
        manager.load_points3D()

        # COLMAP image ids are arbitrary; order by filename so the hold-every-8 split
        # is reproducible and matches opt/util/llff_dataset.py on the Plenoxels side.
        images = sorted(manager.images.values(), key=lambda im: im.name)
        self.image_names = [im.name for im in images]
        self.camera_ids = [im.camera_id for im in images]

        camtoworlds = []
        for im in images:
            w2c = np.eye(4)
            w2c[:3, :3] = im.R()
            w2c[:3, 3] = im.tvec
            camtoworlds.append(np.linalg.inv(w2c))
        camtoworlds = np.stack(camtoworlds).astype(np.float32)

        points = manager.points3D.astype(np.float32)
        points_rgb = manager.point3D_colors.astype(np.uint8)

        # Which images observe each SfM point, taken from COLMAP's own feature tracks.
        # Using the reconstruction rather than a rendered depth buffer keeps visibility
        # independent of any trained model, so different methods can be compared on an
        # identical set of (point, view) pairs.
        id_to_idx = manager.point3D_id_to_point3D_idx
        self.point_views = [[] for _ in range(points.shape[0])]
        for img_index, im in enumerate(images):
            for pid in np.unique(im.point3D_ids):
                if pid == manager.INVALID_POINT3D:
                    continue
                idx = id_to_idx.get(pid, None) if hasattr(id_to_idx, "get") else None
                if idx is None or idx < 0 or idx >= len(self.point_views):
                    continue
                self.point_views[idx].append(img_index)

        transform = np.eye(4)
        if normalize:
            t1 = similarity_from_cameras(camtoworlds)
            camtoworlds = transform_cameras(t1, camtoworlds)
            points = transform_points(t1, points)
            transform = t1
            if align_axes:
                t2 = align_principle_axes(points)
                camtoworlds = transform_cameras(t2, camtoworlds)
                points = transform_points(t2, points)
                transform = t2 @ t1
        self.transform = transform
        self.camtoworlds = camtoworlds.astype(np.float32)
        self.points = points.astype(np.float32)
        self.points_rgb = points_rgb

        # The frames on disk are not always at the resolution COLMAP recorded. The 3DGS
        # Tanks & Temples release ships 979x546 images against a 1957x1091 camera, and
        # the ratio is not even exactly 2 because of rounding when they were
        # downsampled. Measure the actual files and scale the intrinsics to them rather
        # than trusting the two to agree.
        probe_dir = os.path.join(data_dir, "images")
        probe = sorted(f for f in os.listdir(probe_dir) if not f.startswith("."))
        if not probe:
            raise SystemExit(f"no images in {probe_dir}")
        probe_img = cv2.imread(os.path.join(probe_dir, probe[0]), cv2.IMREAD_GRAYSCALE)
        if probe_img is None:
            raise IOError(f"could not read {probe[0]}")
        self.source_height, self.source_width = probe_img.shape[:2]

        # Undistortion, built once per distinct camera at the downsampled resolution.
        # OpenCV distortion coefficients live in normalized image coordinates, so they
        # are unchanged by the downsample as long as K is scaled with it.
        self._undistort = {}
        for camera_id, cam in manager.cameras.items():
            sx = self.source_width / cam.width
            sy = self.source_height / cam.height
            if abs(sx - 1.0) > 1e-6 or abs(sy - 1.0) > 1e-6:
                print(f"[data] images are {self.source_width}x{self.source_height} but "
                      f"COLMAP camera {camera_id} is {cam.width}x{cam.height}; "
                      f"scaling intrinsics by ({sx:.4f}, {sy:.4f})")
            w = self.source_width // factor
            h = self.source_height // factor
            K = np.array(
                [
                    [cam.fx * sx / factor, 0.0, cam.cx * sx / factor],
                    [0.0, cam.fy * sy / factor, cam.cy * sy / factor],
                    [0.0, 0.0, 1.0],
                ]
            )
            dist = _dist_coeffs(cam)
            if not np.any(dist):
                self._undistort[camera_id] = (K.astype(np.float32), None, None, w, h)
                continue
            # alpha=0 keeps only pixels with valid source data, so the ROI crop removes
            # the black undistortion border entirely instead of leaving it in the loss.
            K_new, roi = cv2.getOptimalNewCameraMatrix(K, dist, (w, h), 0)
            mapx, mapy = cv2.initUndistortRectifyMap(
                K, dist, None, K_new, (w, h), cv2.CV_32FC1
            )
            x, y, rw, rh = roi
            mapx, mapy = mapx[y : y + rh, x : x + rw], mapy[y : y + rh, x : x + rw]
            K_new = K_new.copy()
            K_new[0, 2] -= x
            K_new[1, 2] -= y
            self._undistort[camera_id] = (K_new.astype(np.float32), mapx, mapy, rw, rh)

        self.Ks = np.stack([self._undistort[c][0] for c in self.camera_ids])
        sizes = {(self._undistort[c][3], self._undistort[c][4]) for c in self.camera_ids}
        assert len(sizes) == 1, f"mixed image sizes after undistortion: {sizes}"
        self.width, self.height = sizes.pop()

        # Used to scale the position learning rate and the scale-based prune threshold.
        locations = self.camtoworlds[:, :3, 3]
        dists = np.linalg.norm(locations - locations.mean(axis=0), axis=1)
        self.scene_scale = float(dists.max())

        self.test_indices = [i for i in range(len(images)) if i % test_every == 0]
        self.train_indices = [i for i in range(len(images)) if i % test_every != 0]

    def _source_path(self, root, stem, exts):
        for ext in exts:
            path = os.path.join(root, stem + ext)
            if os.path.isfile(path):
                return path
        return None

    def _rectify(self, path, index, grayscale):
        flag = cv2.IMREAD_GRAYSCALE if grayscale else cv2.IMREAD_COLOR
        img = cv2.imread(path, flag)
        if img is None:
            raise IOError(f"could not read {path}")
        if not grayscale:
            img = img[:, :, ::-1]  # BGR -> RGB
        _, mapx, mapy, w, h = self._undistort[self.camera_ids[index]]
        # Decimate at full resolution first (INTER_AREA is the correct filter for an
        # integer downsample), then resample through the undistortion map.
        img = cv2.resize(
            img,
            (img.shape[1] // self.factor, img.shape[0] // self.factor),
            interpolation=cv2.INTER_AREA,
        )
        if mapx is not None:
            img = cv2.remap(img, mapx, mapy, cv2.INTER_LINEAR)
        assert img.shape[:2] == (h, w), f"{img.shape[:2]} != {(h, w)} for {path}"
        return np.ascontiguousarray(img)

    def load_images(self, cache_dir=None, verbose=True):
        """Return the rectified grey / teacher / colour stacks, caching them on disk.

        Decoding 3 x N full-resolution JPEGs and resampling them takes a minute or two
        per scene, which is wasteful across the dozen runs these experiments need, so
        the rectified factor-N images are cached as lossless PNGs.
        """
        tag = f"{os.path.basename(os.path.normpath(self.data_dir))}_f{self.factor}"
        cache = os.path.join(cache_dir, tag) if cache_dir else None
        meta_path = os.path.join(cache, "meta.json") if cache else None

        sources = {
            "grey": (os.path.join(self.data_dir, "images"), [".JPG", ".jpg", ".png"], True),
            "teacher": (
                os.path.join(self.data_dir, "images_bigcolor"),
                [".jpg", ".JPG", ".png"],
                False,
            ),
        }
        if self.color_ref_dir:
            sources["color"] = (
                os.path.join(self.color_ref_dir, "images"),
                [".JPG", ".jpg", ".png"],
                False,
            )

        if cache and os.path.isfile(meta_path):
            with open(meta_path) as f:
                meta = json.load(f)
            if meta["width"] == self.width and meta["height"] == self.height and set(
                meta["kinds"]
            ) == set(sources):
                out = {}
                for kind in sources:
                    out[kind] = np.stack(
                        [
                            cv2.imread(
                                os.path.join(cache, kind, f"{i:04d}.png"),
                                cv2.IMREAD_GRAYSCALE
                                if kind == "grey"
                                else cv2.IMREAD_COLOR,
                            )
                            for i in range(len(self.image_names))
                        ]
                    )
                    if kind != "grey":
                        out[kind] = out[kind][:, :, :, ::-1]
                    out[kind] = np.ascontiguousarray(out[kind])
                if verbose:
                    print(f"[data] loaded {tag} from cache {cache}")
                return out

        out = {}
        for kind, (root, exts, grayscale) in sources.items():
            stack = []
            for i, name in enumerate(self.image_names):
                stem = os.path.splitext(name)[0]
                path = self._source_path(root, stem, exts)
                if path is None:
                    raise FileNotFoundError(
                        f"no {kind} image for {stem} in {root} (tried {exts})"
                    )
                stack.append(self._rectify(path, i, grayscale))
            out[kind] = np.stack(stack)
            if verbose:
                print(f"[data] rectified {kind}: {out[kind].shape}")

        if cache:
            for kind, stack in out.items():
                os.makedirs(os.path.join(cache, kind), exist_ok=True)
                for i, img in enumerate(stack):
                    write = img if kind == "grey" else img[:, :, ::-1]
                    cv2.imwrite(os.path.join(cache, kind, f"{i:04d}.png"), write)
            with open(meta_path, "w") as f:
                json.dump(
                    {
                        "width": self.width,
                        "height": self.height,
                        "kinds": sorted(out),
                        "factor": self.factor,
                        "image_names": self.image_names,
                    },
                    f,
                    indent=2,
                )
            if verbose:
                print(f"[data] cached {tag} to {cache}")
        return out


class Dataset(torch.utils.data.Dataset):
    """One split of a scene, yielding a camera plus its aligned image triplet."""

    def __init__(self, parser, split="train", images=None, cache_dir=None):
        self.parser = parser
        self.split = split
        self.images = images if images is not None else parser.load_images(cache_dir)
        self.indices = (
            parser.train_indices if split == "train" else parser.test_indices
        )

    def __len__(self):
        return len(self.indices)

    def __getitem__(self, item):
        index = self.indices[item]
        out = {
            "index": index,
            "name": self.parser.image_names[index],
            "camtoworld": torch.from_numpy(self.parser.camtoworlds[index]),
            "K": torch.from_numpy(self.parser.Ks[index]),
            "grey": torch.from_numpy(
                self.images["grey"][index].astype(np.float32) / 255.0
            ).unsqueeze(-1),
            "teacher": torch.from_numpy(
                self.images["teacher"][index].astype(np.float32) / 255.0
            ),
        }
        if "color" in self.images:
            out["color"] = torch.from_numpy(
                self.images["color"][index].astype(np.float32) / 255.0
            )
        return out
