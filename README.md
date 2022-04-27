# Pix2Vox with Silhouette Guidance

## Datasets

We use the [ShapeNet](https://www.shapenet.org/) datasets in our experiments, which are available below:

- ShapeNet rendering images: http://cvgl.stanford.edu/data2/ShapeNetRendering.tgz
- ShapeNet voxelized models: http://cvgl.stanford.edu/data2/ShapeNetVox32.tgz



## Steps to run our code

#### Clone the Code Repository and switch to vox_renderer branch

```
git clone https://github.com/ZjTan4/Pix2Vox_414.git
cd Pix2Vox_414
git checkout vox_renderer
```

#### Install Python Denpendencies

```
pip install -r requirements.txt

In addition, we also used torch, torchvision with cuda and pytorch3d. To install the correct version based on your hardware, please follow the instruction:

Instruction about installing torchvision iwth cuda:
https://pytorch.org/get-started/locally/

Instruction about installing pytorch3d:
- https://github.com/facebookresearch/pytorch3d/blob/main/INSTALL.md
```

#### Update Settings in `config.py`

You need to update the file path of the datasets downloaded from previous section:

```
__C.DATASETS.SHAPENET.TAXONOMY_FILE_PATH    = '/path/ShapeNet.json'
__C.DATASETS.SHAPENET.RENDERING_PATH        = '/path/to/Datasets/ShapeNet/ShapeNetRendering/%s/%s/rendering/%02d.png'
__C.DATASETS.SHAPENET.VOXEL_PATH            = '/path/to/Datasets/ShapeNet/ShapeNetVox32/%s/%s/model.binvox'

```

## How To Run the code

To train the model, run the command:

```
python3 runner.py
```

To test the model:

```
python3 runner.py --test --weights=/path/to/pretrained/model.pth
```


## License

This project is open sourced under MIT license.
